# coding=utf-8
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GPT-Neo model for NeuronX Distributed Inference."""

import json
import logging
import math
import os
from typing import List, Type

import torch
import torch.nn as nn
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


class GPTNeoInferenceConfig(InferenceConfig):
    """Configuration class for GPT-Neo model inference on NeuronX."""

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # GPT-Neo has no sliding window at the model level for NXDI
        # (local attention is handled per-layer in the attention module)
        self.sliding_window = None

        if not hasattr(self, 'intermediate_size') or self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        # GPT-Neo uses MHA, so num_key_value_heads = num_attention_heads
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
            "attention_layers",
            "window_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GPTNeoInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        config_dict = {
            "hidden_size": params.get("hidden_size", 2048),
            "num_attention_heads": params.get("num_heads", 16),
            "num_key_value_heads": params.get("num_heads", 16),
            "num_hidden_layers": params.get("num_layers", 24),
            "vocab_size": params.get("vocab_size", 50257),
            "max_position_embeddings": params.get("max_position_embeddings", 2048),
            "intermediate_size": params.get("intermediate_size", None),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-5),
            "activation_function": params.get("activation_function", "gelu_new"),
            "attention_layers": params.get("attention_layers", []),
            "window_size": params.get("window_size", 256),
            "attention_dropout": params.get("attention_dropout", 0.0),
            "embed_dropout": params.get("embed_dropout", 0.0),
            "resid_dropout": params.get("resid_dropout", 0.0),
            "bos_token_id": params.get("bos_token_id", 50256),
            "eos_token_id": params.get("eos_token_id", 50256),
        }

        config_dict.update(kwargs)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronGPTNeoAttention(NeuronAttentionBase):
    """GPT-Neo attention for NeuronX.

    GPT-Neo does NOT scale attention scores by 1/sqrt(head_dim) - it uses
    unscaled dot-product attention (softmax_scale=1.0). To counteract the
    base class's mandatory scaling, we pre-multiply Q by sqrt(head_dim)
    in prep_qkv_tensors so the net effect is no scaling.

    GPT-Neo alternates local/global attention, but we treat all layers
    as global for NXDI purposes. The local attention pattern (limited
    context window) would need to be handled via attention masking,
    not via sliding_window KV cache resizing. For correctness at the
    cost of some efficiency, all layers use full KV cache.
    """

    def __init__(self, config: GPTNeoInferenceConfig, layer_idx: int = None, **kwargs):
        self.layer_idx = layer_idx

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            clip_qkv=None,
            qkv_bias=False,
            o_bias=True,
            sliding_window=None,
            **kwargs,
        )

    def prep_qkv_tensors(self, position_ids, hidden_states, past_key_value, **kwargs):
        """Override to pre-scale Q by sqrt(head_dim).

        GPT-Neo uses unscaled attention (no 1/sqrt(d_k) division).
        The base class unconditionally divides Q by sqrt(head_dim) in
        perform_prefill and compute_for_token_gen. Pre-multiplying Q
        here cancels that out, giving net scale = 1.0.
        """
        Q, K, V, cos_cache, sin_cache, residual = super().prep_qkv_tensors(
            position_ids, hidden_states, past_key_value, **kwargs
        )
        Q = Q * math.sqrt(self.head_dim)
        return Q, K, V, cos_cache, sin_cache, residual


class NeuronGPTNeoMLP(nn.Module):
    """GPT-Neo MLP: Linear (up) -> GELU -> Linear (down)."""

    def __init__(self, config: GPTNeoInferenceConfig):
        super().__init__()

        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # GPT-Neo uses gelu_new (tanh approximation). nn.GELU enables Neuron compiler fusion.
        self.act = nn.GELU(approximate='tanh')

        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states, None


class NeuronGPTNeoBlock(nn.Module):
    """GPT-Neo transformer block: LN -> Attn -> Residual -> LN -> MLP -> Residual.

    Returns 5-tuple: (hidden_states, present_key_value, cos_cache, sin_cache, residual)
    as required by NeuronBaseModel.get_model_output.
    """

    def __init__(self, config: GPTNeoInferenceConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = NeuronGPTNeoAttention(
            config, layer_idx=layer_idx, tensor_model_parallel_group=get_tp_group(config)
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = NeuronGPTNeoMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs
    ):
        # Attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states + residual

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = mlp_output + residual

        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronGPTNeoModel(NeuronBaseModel):
    """GPT-Neo base model for NeuronX.

    Uses init_model() for module creation (not __init__) as required by NeuronBaseModel.
    Does not override forward() - the base class handles KV cache, masking, sampling, etc.
    Overrides get_model_output() to add absolute position embeddings.
    """

    def setup_attr_for_model(self, config: GPTNeoInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: GPTNeoInferenceConfig):
        self.padding_idx = getattr(config, 'pad_token_id', None)

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # Position embeddings (absolute, not rotary)
        self.wpe = ParallelEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # LM head
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
            NeuronGPTNeoBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def get_model_output(self, input_ids, position_ids=None, **kwargs):
        """Override to add absolute position embeddings before passing through layers."""
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Generate position_ids if not provided (base class does this too late for us)
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Add absolute position embeddings
        position_embeds = self.wpe(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        # Pass combined embeddings to base class
        kwargs['inputs_embeds'] = inputs_embeds
        return super().get_model_output(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )


class NeuronGPTNeoForCausalLM(NeuronBaseForCausalLM):
    """GPT-Neo model with language modeling head for NeuronX."""

    _model_cls = NeuronGPTNeoModel
    # GPT-Neo HF weights use "transformer." prefix (not "model.")
    _STATE_DICT_MODEL_PREFIX = "transformer."

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert HuggingFace GPT-Neo weights to NeuronX format.

        At this point, the framework has already stripped the "transformer." prefix,
        so keys look like: wte.weight, wpe.weight, h.0.attn.attention.q_proj.weight, etc.
        We remap them to match our NeuronGPTNeoModel module structure.
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree

        # Token embeddings: wte.weight -> embed_tokens.weight
        if "wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["wte.weight"]

        # Position embeddings: wpe.weight -> wpe.weight (same name)
        if "wpe.weight" in state_dict:
            neuron_state_dict["wpe.weight"] = state_dict["wpe.weight"]

        # Final layer norm: ln_f -> norm
        if "ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["ln_f.weight"]
        if "ln_f.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["ln_f.bias"]

        for i in range(config.num_hidden_layers):
            hf = f"h.{i}"
            nxd = f"layers.{i}"

            # Layer norms: h.{i}.ln_1 -> layers.{i}.ln_1
            for ln_name in ["ln_1", "ln_2"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.{ln_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.{ln_name}.{param}"] = state_dict[key]

            # Attention Q, K, V: keep separate for GQA preshard hook
            attn_hf = f"{hf}.attn.attention"
            attn_nxd = f"{nxd}.attn"

            for proj in ["q_proj", "k_proj", "v_proj"]:
                key = f"{attn_hf}.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[f"{attn_nxd}.qkv_proj.{proj}.weight"] = state_dict[key]

            # Output projection: out_proj -> o_proj.o_proj
            out_w = f"{attn_hf}.out_proj.weight"
            out_b = f"{attn_hf}.out_proj.bias"
            if out_w in state_dict:
                neuron_state_dict[f"{attn_nxd}.o_proj.o_proj.weight"] = state_dict[out_w]
            if out_b in state_dict:
                neuron_state_dict[f"{attn_nxd}.o_proj.o_proj.bias"] = state_dict[out_b]

            # rank_util for attention
            neuron_state_dict[f"{attn_nxd}.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

            # MLP: h.{i}.mlp.c_fc -> layers.{i}.mlp.c_fc
            for layer_name in ["c_fc", "c_proj"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.mlp.{layer_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.mlp.{layer_name}.{param}"] = state_dict[key]

        # LM head uses tied weights (clone to avoid shared memory error during save)
        if "wte.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["wte.weight"].clone()

        # rank_util for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @classmethod
    def get_config_cls(cls):
        return GPTNeoInferenceConfig
