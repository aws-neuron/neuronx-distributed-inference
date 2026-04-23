# coding=utf-8
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GitForCausalLM port for NeuronX Distributed Inference (NXDI).

Architecture overview:
- Vision encoder: CLIP-style ViT (runs on CPU during preprocessing for now;
  see feasibility notes for full-Neuron compilation)
- Visual projection: Linear + LayerNorm (runs on CPU)
- Text decoder: BERT-style causal LM with 6 layers (compiled on Neuron)
  - Absolute position embeddings (no rotary)
  - Post-LN residual blocks (BERT-style, not pre-LN like GPT/LLaMA)
  - Separate Q/K/V projections with bias
  - GELU activation
- LM head: Linear layer mapping to vocab_size=30522

The text decoder receives vision features prepended to text embeddings via the
NXDI vision hooks (encode_vision_to_input).

HF model: microsoft/git-base
HF weight prefix: "git." (stripped via _STATE_DICT_MODEL_PREFIX)
"""

import json
import logging
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


class GitInferenceConfig(InferenceConfig):
    """Configuration for GitForCausalLM inference on NeuronX."""

    def __init__(self, neuron_config=None, **kwargs):
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("use_cache", True)
        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Git uses MHA: num_key_value_heads = num_attention_heads
        if not hasattr(self, "num_key_value_heads") or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "layer_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GitInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        config_dict = {
            "hidden_size": params.get("hidden_size", 768),
            "num_attention_heads": params.get("num_attention_heads", 12),
            "num_hidden_layers": params.get("num_hidden_layers", 6),
            "vocab_size": params.get("vocab_size", 30522),
            "max_position_embeddings": params.get("max_position_embeddings", 1024),
            "intermediate_size": params.get("intermediate_size", 3072),
            "layer_norm_eps": params.get("layer_norm_eps", 1e-12),
            "hidden_act": params.get("hidden_act", "gelu"),
            "pad_token_id": params.get("pad_token_id", 0),
            "bos_token_id": params.get("bos_token_id", 101),
            "eos_token_id": params.get("eos_token_id", 102),
            "tie_word_embeddings": params.get("tie_word_embeddings", False),
            # Vision config (stored for reference)
            "vision_hidden_size": params.get("vision_config", {}).get("hidden_size", 768),
            "vision_num_hidden_layers": params.get("vision_config", {}).get("num_hidden_layers", 12),
            "vision_image_size": params.get("vision_config", {}).get("image_size", 224),
            "vision_patch_size": params.get("vision_config", {}).get("patch_size", 16),
            # Number of vision tokens: (image_size/patch_size)^2 + 1 (CLS token)
            "num_image_tokens": (
                params.get("vision_config", {}).get("image_size", 224)
                // params.get("vision_config", {}).get("patch_size", 16)
            ) ** 2 + 1,
        }

        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronGitAttention(NeuronAttentionBase):
    """
    Git text decoder attention for NeuronX.

    BERT-style MHA with separate Q/K/V projections (with bias).
    No rotary embeddings -- uses absolute position embeddings added at the
    embedding layer level.
    """

    def __init__(self, config: GitInferenceConfig):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # MHA
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,  # No rotary, uses absolute position embeddings
            qkv_bias=True,    # Git/BERT has bias on Q/K/V
            o_bias=True,       # Git/BERT has bias on output projection
            tensor_model_parallel_group=get_tp_group(config),
        )


class NeuronGitMLP(nn.Module):
    """
    Git text decoder MLP for NeuronX.

    BERT-style two-layer MLP: intermediate(x) -> act -> output(x)
    With bias on both layers.
    """

    def __init__(self, config: GitInferenceConfig):
        super().__init__()
        self.config = config

        # intermediate.dense: hidden_size -> intermediate_size
        self.fc_in = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # output.dense: intermediate_size -> hidden_size
        self.fc_out = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        from transformers.activations import ACT2FN
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states, None


class NeuronGitBlock(nn.Module):
    """
    Git text decoder block for NeuronX.

    BERT-style post-LayerNorm residual connections:
    1. attn_output = attention(hidden_states)   [includes o_proj]
    2. hidden_states = LayerNorm(attn_output + hidden_states)  [attention output LN]
    3. mlp_output = mlp(hidden_states)
    4. hidden_states = LayerNorm(mlp_output + hidden_states)   [output LN]

    This differs from GPT/LLaMA pre-LN style.

    Note: NeuronAttentionBase already includes the output projection (o_proj),
    so we do NOT add a separate attn_dense layer.
    """

    def __init__(self, config: GitInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention (includes Q/K/V projections and o_proj)
        self.attn = NeuronGitAttention(config)

        # Post-attention LayerNorm (BERT SelfOutput pattern)
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # MLP
        self.mlp = NeuronGitMLP(config)

        # Post-MLP LayerNorm (BERT Output pattern)
        self.output_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states

        # Self-attention (NeuronAttentionBase handles Q/K/V + o_proj)
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Post-attention LayerNorm with residual
        hidden_states = self.attn_ln(attn_output.hidden_states + residual)

        # MLP with post-LN residual
        residual = hidden_states
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = self.output_ln(mlp_output + residual)

        # Return 5-tuple expected by NXDI base model
        return (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,
        )


class NeuronGitModel(NeuronBaseModel):
    """
    Git text decoder model for NeuronX.

    Overrides get_model_output() only to add absolute position embeddings
    and embedding LayerNorm, then delegates to super() for the layer loop,
    KV cache management, attention masking, etc.
    """

    def setup_attr_for_model(self, config: GitInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # MHA
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: GitInferenceConfig):
        self.vocab_size = config.vocab_size

        # Token embedding
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # LM head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        # Position embeddings (absolute, like BERT)
        self.position_embeddings = ParallelEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # Embedding LayerNorm (Git/BERT-style)
        self.embed_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [NeuronGitBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Git uses post-LN per block, so there is no final layer norm.
        # The base class expects self.norm to exist, so use nn.Identity.
        self.norm = nn.Identity()

    def get_model_output(self, input_ids, position_ids=None, **kwargs):
        """Override to add absolute position embeddings + embedding LayerNorm
        before delegating to the base class layer loop."""
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Generate position_ids if not provided
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Add absolute position embeddings
        position_embeds = self.position_embeddings(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        # Apply embedding LayerNorm
        inputs_embeds = self.embed_ln(inputs_embeds)

        # Pass combined embeddings to base class (handles layer loop, KV cache, etc.)
        kwargs["inputs_embeds"] = inputs_embeds
        return super().get_model_output(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )

    def encode_vision_to_input(
        self,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Inject projected vision features into text embeddings.

        In Git, vision features are prepended to text tokens. The vision_mask
        indicates which positions in the input should be replaced with vision
        features.

        Args:
            inputs_embeds: Text embeddings [batch, seq_len, hidden_size]
            vision_embeddings: Projected vision features [batch, num_vision_tokens, hidden_size]
            vision_mask: Boolean mask [batch, seq_len] indicating vision token positions

        Returns:
            Modified embeddings with vision features placed at masked positions
        """
        batch_size = inputs_embeds.shape[0]
        for b in range(batch_size):
            mask_positions = vision_mask[b].nonzero(as_tuple=True)[0]
            num_vision = min(len(mask_positions), vision_embeddings.shape[1])
            if num_vision > 0:
                inputs_embeds[b, mask_positions[:num_vision]] = vision_embeddings[b, :num_vision]
        return inputs_embeds


class NeuronGitForCausalLM(NeuronBaseForCausalLM):
    """
    GitForCausalLM for NeuronX Distributed Inference.

    Compiles the text decoder on Neuron. Vision encoder currently runs on CPU.
    """

    _model_cls = NeuronGitModel
    _STATE_DICT_MODEL_PREFIX = "git."

    @classmethod
    def from_config(cls, config: GitInferenceConfig):
        return cls(config=config)

    @classmethod
    def get_config_cls(cls):
        return GitInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: GitInferenceConfig) -> dict:
        """
        Convert HuggingFace GitForCausalLM state dict to NeuronX format.

        By the time this is called, the base class has already stripped the "git."
        prefix via _STATE_DICT_MODEL_PREFIX. The LM head keys ("output.weight",
        "output.bias") don't have that prefix so they pass through as-is.

        Weight mapping:
        - embeddings.word_embeddings.weight -> embed_tokens.weight
        - embeddings.position_embeddings.weight -> position_embeddings.weight
        - embeddings.LayerNorm.{weight,bias} -> embed_ln.{weight,bias}
        - encoder.layer.{i}.attention.self.query.{w,b} -> layers.{i}.attn.qkv_proj.q_proj.{w,b}
        - encoder.layer.{i}.attention.self.key.{w,b} -> layers.{i}.attn.qkv_proj.k_proj.{w,b}
        - encoder.layer.{i}.attention.self.value.{w,b} -> layers.{i}.attn.qkv_proj.v_proj.{w,b}
        - encoder.layer.{i}.attention.output.dense.{w,b} -> layers.{i}.attn.o_proj.o_proj.{w,b}
        - encoder.layer.{i}.attention.output.LayerNorm.{w,b} -> layers.{i}.attn_ln.{w,b}
        - encoder.layer.{i}.intermediate.dense.{w,b} -> layers.{i}.mlp.fc_in.{w,b}
        - encoder.layer.{i}.output.dense.{w,b} -> layers.{i}.mlp.fc_out.{w,b}
        - encoder.layer.{i}.output.LayerNorm.{w,b} -> layers.{i}.output_ln.{w,b}
        - output.{weight,bias} -> lm_head.{weight,bias}
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree

        print("Converting HuggingFace GitForCausalLM state dict to NeuronX format...")
        print(f"Original checkpoint has {len(state_dict)} keys")

        for key, value in state_dict.items():
            # Skip vision encoder weights (not compiled in text decoder NEFF)
            if key.startswith("image_encoder.") or key.startswith("visual_projection."):
                continue

            # Token embeddings
            if key == "embeddings.word_embeddings.weight":
                neuron_state_dict["embed_tokens.weight"] = value.clone()
                continue

            # Position embeddings
            if key == "embeddings.position_embeddings.weight":
                neuron_state_dict["position_embeddings.weight"] = value.clone()
                continue

            # Embedding LayerNorm
            if key == "embeddings.LayerNorm.weight":
                neuron_state_dict["embed_ln.weight"] = value.clone()
                continue
            if key == "embeddings.LayerNorm.bias":
                neuron_state_dict["embed_ln.bias"] = value.clone()
                continue

            # LM head (no "git." prefix in HF checkpoint)
            if key == "output.weight":
                neuron_state_dict["lm_head.weight"] = value.clone()
                continue
            if key == "output.bias":
                neuron_state_dict["lm_head.bias"] = value.clone()
                continue

            # Encoder layer weights
            if key.startswith("encoder.layer."):
                parts = key.split(".")
                layer_idx = parts[2]
                rest = ".".join(parts[3:])

                # Attention Q/K/V projections
                if rest.startswith("attention.self.query."):
                    suffix = rest.split("attention.self.query.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.q_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.self.key."):
                    suffix = rest.split("attention.self.key.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.k_proj.{suffix}"] = value.clone()
                elif rest.startswith("attention.self.value."):
                    suffix = rest.split("attention.self.value.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.qkv_proj.v_proj.{suffix}"] = value.clone()

                # Attention output dense -> o_proj.o_proj (double nesting from NeuronAttentionBase)
                elif rest.startswith("attention.output.dense."):
                    suffix = rest.split("attention.output.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn.o_proj.o_proj.{suffix}"] = value.clone()

                # Attention output LayerNorm
                elif rest.startswith("attention.output.LayerNorm."):
                    suffix = rest.split("attention.output.LayerNorm.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.attn_ln.{suffix}"] = value.clone()

                # Intermediate dense (fc_in)
                elif rest.startswith("intermediate.dense."):
                    suffix = rest.split("intermediate.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.mlp.fc_in.{suffix}"] = value.clone()

                # Output dense (fc_out)
                elif rest.startswith("output.dense."):
                    suffix = rest.split("output.dense.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.mlp.fc_out.{suffix}"] = value.clone()

                # Output LayerNorm
                elif rest.startswith("output.LayerNorm."):
                    suffix = rest.split("output.LayerNorm.")[1]
                    neuron_state_dict[f"layers.{layer_idx}.output_ln.{suffix}"] = value.clone()
                else:
                    print(f"  WARNING: Unmapped encoder key: {key}")
                continue

            print(f"  Skipping unmapped key: {key}")

        # Add rank_util tensors required by NeuronAttentionBase
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Base model rank_util
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        print(f"Converted to {len(neuron_state_dict)} NeuronX keys")
        return neuron_state_dict


__all__ = [
    "GitInferenceConfig",
    "NeuronGitModel",
    "NeuronGitForCausalLM",
]
