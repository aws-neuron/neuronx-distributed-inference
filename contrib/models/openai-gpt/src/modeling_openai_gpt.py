# coding=utf-8
# Copyright OpenAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch OpenAI GPT-1 model for NeuronX Distributed Inference.

Key architectural differences from GPT-Neo/GPT-2:
- Post-norm (LayerNorm AFTER residual addition, not before)
- No final LayerNorm after last transformer block
- Conv1D weight format (transposed from nn.Linear)
- Fused QKV (single c_attn weight, not separate q/k/v projections)
- Standard GELU activation (erf-based, not tanh-approximate gelu_new)
- Standard 1/sqrt(d_k) attention scaling
"""

import json
import logging
import os
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class OpenAIGPTInferenceConfig(InferenceConfig):
    """Configuration class for OpenAI GPT-1 inference on NeuronX."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.sliding_window = None

        if not hasattr(self, 'intermediate_size') or self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        # GPT-1 uses MHA
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.output_attentions = False
        self.output_hidden_states = False
        self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "layer_norm_epsilon",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "OpenAIGPTInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        config_dict = {
            "hidden_size": params.get("n_embd", 768),
            "num_attention_heads": params.get("n_head", 12),
            "num_key_value_heads": params.get("n_head", 12),
            "num_hidden_layers": params.get("n_layer", 12),
            "vocab_size": params.get("vocab_size", 40478),
            "max_position_embeddings": params.get("n_positions", 512),
            "intermediate_size": params.get("intermediate_size", None),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-5),
            "afn": params.get("afn", "gelu"),
            "bos_token_id": params.get("bos_token_id", None),
            "eos_token_id": params.get("eos_token_id", None),
        }

        config_dict.update(kwargs)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronOpenAIGPTAttention(NeuronAttentionBase):
    """OpenAI GPT-1 attention for NeuronX.

    GPT-1 uses standard 1/sqrt(d_k) scaling, which matches the base class default.
    No special scaling workaround needed (unlike GPT-Neo which uses unscaled attention).
    """

    def __init__(self, config: OpenAIGPTInferenceConfig, layer_idx: int = None, **kwargs):
        self.layer_idx = layer_idx

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            clip_qkv=None,
            qkv_bias=True,
            o_bias=True,
            sliding_window=None,
            **kwargs,
        )


class NeuronOpenAIGPTMLP(nn.Module):
    """OpenAI GPT-1 MLP: Linear (up) -> GELU -> Linear (down).

    Uses standard GELU (erf-based), not the tanh-approximate gelu_new.
    """

    def __init__(self, config: OpenAIGPTInferenceConfig):
        super().__init__()
        self.config = config

        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states, None


class NeuronOpenAIGPTBlock(nn.Module):
    """OpenAI GPT-1 transformer block with POST-NORM architecture.

    Post-norm block structure (different from pre-norm GPT-2/Neo):
        attn_out = attn(hidden_states)          # attention on raw input
        hidden_states = ln_1(hidden_states + attn_out)  # norm AFTER residual
        mlp_out = mlp(hidden_states)
        hidden_states = ln_2(hidden_states + mlp_out)   # norm AFTER residual

    Returns 5-tuple: (hidden_states, present_key_value, cos_cache, sin_cache, residual)
    as required by NeuronBaseModel.get_model_output.
    """

    def __init__(self, config: OpenAIGPTInferenceConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = NeuronOpenAIGPTAttention(
            config, layer_idx=layer_idx, tensor_model_parallel_group=get_tp_group(config)
        )
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = NeuronOpenAIGPTMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs
    ):
        # Post-norm: attention on raw hidden_states, then norm after residual
        residual = hidden_states
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = self.ln_1(residual + attn_output.hidden_states)

        # Post-norm: MLP, then norm after residual
        residual = hidden_states
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = self.ln_2(residual + mlp_output)

        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronOpenAIGPTModel(NeuronBaseModel):
    """OpenAI GPT-1 base model for NeuronX.

    Key differences from GPT-Neo:
    - Post-norm blocks (no pre-norm)
    - No final LayerNorm (self.norm = Identity)
    - Uses absolute position embeddings (same as GPT-Neo)
    """

    def setup_attr_for_model(self, config: OpenAIGPTInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: OpenAIGPTInferenceConfig):
        self.padding_idx = getattr(config, 'pad_token_id', None)

        # Token embeddings (HF: tokens_embed)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # Absolute position embeddings (HF: positions_embed)
        self.wpe = ParallelEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # LM head (tied to token embeddings)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            bias=False,
            pad=True,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            NeuronOpenAIGPTBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # GPT-1 has NO final LayerNorm (post-norm blocks handle normalization)
        self.norm = nn.Identity()

    def get_model_output(self, input_ids, position_ids=None, **kwargs):
        """Add absolute position embeddings before passing through layers."""
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        position_embeds = self.wpe(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        kwargs['inputs_embeds'] = inputs_embeds
        return super().get_model_output(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )


class NeuronOpenAIGPTForCausalLM(NeuronBaseForCausalLM):
    """OpenAI GPT-1 model with language modeling head for NeuronX."""

    _model_cls = NeuronOpenAIGPTModel
    # GPT-1 HF weights have NO "transformer." prefix (bare keys like h.0.attn.c_attn.weight)
    _STATE_DICT_MODEL_PREFIX = ""

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert HuggingFace OpenAI GPT-1 weights to NeuronX format.

        Handles:
        1. Conv1D weight transposition ([in, out] -> [out, in])
        2. Fused QKV split (c_attn -> separate q_proj, k_proj, v_proj)
        3. Key remapping (HF names -> NXDI module names)
        4. Tied lm_head weights (clone from tokens_embed)
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree
        hidden_size = config.hidden_size

        # Token embeddings: tokens_embed.weight -> embed_tokens.weight
        if "tokens_embed.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["tokens_embed.weight"]

        # Position embeddings: positions_embed.weight -> wpe.weight
        if "positions_embed.weight" in state_dict:
            neuron_state_dict["wpe.weight"] = state_dict["positions_embed.weight"]

        for i in range(config.num_hidden_layers):
            hf = f"h.{i}"
            nxd = f"layers.{i}"

            # Layer norms (post-norm): h.{i}.ln_1/ln_2 -> layers.{i}.ln_1/ln_2
            for ln_name in ["ln_1", "ln_2"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.{ln_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.{ln_name}.{param}"] = state_dict[key]

            # Fused QKV: c_attn.weight [768, 2304] -> transpose -> split into Q, K, V
            attn_w = f"{hf}.attn.c_attn.weight"
            attn_b = f"{hf}.attn.c_attn.bias"
            attn_nxd = f"{nxd}.attn"

            if attn_w in state_dict:
                # Conv1D: [in_features, 3*hidden] -> transpose to [3*hidden, in_features]
                w = state_dict[attn_w].t()
                q_w, k_w, v_w = w.split(hidden_size, dim=0)
                neuron_state_dict[f"{attn_nxd}.qkv_proj.q_proj.weight"] = q_w
                neuron_state_dict[f"{attn_nxd}.qkv_proj.k_proj.weight"] = k_w
                neuron_state_dict[f"{attn_nxd}.qkv_proj.v_proj.weight"] = v_w

            if attn_b in state_dict:
                b = state_dict[attn_b]
                q_b, k_b, v_b = b.split(hidden_size, dim=0)
                neuron_state_dict[f"{attn_nxd}.qkv_proj.q_proj.bias"] = q_b
                neuron_state_dict[f"{attn_nxd}.qkv_proj.k_proj.bias"] = k_b
                neuron_state_dict[f"{attn_nxd}.qkv_proj.v_proj.bias"] = v_b

            # Output projection: c_proj -> o_proj.o_proj (single nesting, preshard_hook adds second)
            out_w = f"{hf}.attn.c_proj.weight"
            out_b = f"{hf}.attn.c_proj.bias"
            if out_w in state_dict:
                # Conv1D: [in, out] -> transpose to [out, in]
                neuron_state_dict[f"{attn_nxd}.o_proj.o_proj.weight"] = state_dict[out_w].t()
            if out_b in state_dict:
                neuron_state_dict[f"{attn_nxd}.o_proj.o_proj.bias"] = state_dict[out_b]

            # rank_util for attention
            neuron_state_dict[f"{attn_nxd}.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

            # MLP: Conv1D weights need transposition
            mlp_fc_w = f"{hf}.mlp.c_fc.weight"
            mlp_fc_b = f"{hf}.mlp.c_fc.bias"
            mlp_proj_w = f"{hf}.mlp.c_proj.weight"
            mlp_proj_b = f"{hf}.mlp.c_proj.bias"

            if mlp_fc_w in state_dict:
                # Conv1D [768, 3072] -> transpose to [3072, 768]
                neuron_state_dict[f"{nxd}.mlp.c_fc.weight"] = state_dict[mlp_fc_w].t()
            if mlp_fc_b in state_dict:
                neuron_state_dict[f"{nxd}.mlp.c_fc.bias"] = state_dict[mlp_fc_b]
            if mlp_proj_w in state_dict:
                # Conv1D [3072, 768] -> transpose to [768, 3072]
                neuron_state_dict[f"{nxd}.mlp.c_proj.weight"] = state_dict[mlp_proj_w].t()
            if mlp_proj_b in state_dict:
                neuron_state_dict[f"{nxd}.mlp.c_proj.bias"] = state_dict[mlp_proj_b]

        # LM head: tied to token embeddings (clone to avoid shared memory error)
        if "tokens_embed.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["tokens_embed.weight"].clone()

        # rank_util for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @classmethod
    def get_config_cls(cls):
        return OpenAIGPTInferenceConfig
