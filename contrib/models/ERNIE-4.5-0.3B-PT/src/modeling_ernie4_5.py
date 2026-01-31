# coding=utf-8
# Copyright 2025 Baidu Inc. and the HuggingFace Inc. team. All rights reserved.
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
PyTorch ERNIE-4.5 model for NeuronX Distributed Inference

This module implements ERNIE-4.5 model for inference on AWS Trainium/Inferentia hardware.
Based on the 634 port with CRITICAL FIX for GLM-style RoPE.

ERNIE-4.5 Architecture:
- Decoder-only transformer with Grouped Query Attention (GQA)
- 16 query heads, 2 key-value heads
- SwiGLU activation in MLP
- RMSNorm for normalization
- GLM-style RoPE (Rotary Position Embeddings) with INTERLEAVE pattern (NOT Llama-style)
- No bias in linear layers

KEY FIX: ERNIE-4.5 uses GLM-style RoPE with interleaved pattern, different from Llama's mid-split.
  Llama rotate_half: [-x_half2, x_half1] = [-5,-6,-7,-8, 1,2,3,4]
  ERNIE rotate_half: [-x2,x1,-x4,x3,...] = [-2,1,-4,3,-6,5,-8,7] (interleaved)
"""

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Returns appropriate RMSNorm implementation based on execution mode.
    - CustomRMSNorm for NeuronX hardware (optimized)
    - LlamaRMSNorm for CPU (compatible)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


# =============================================================================
# GLM-STYLE ROPE FUNCTIONS (CRITICAL FOR ERNIE-4.5)
# =============================================================================

def rotate_half_ernie(x):
    """
    ERNIE-4.5 GLM-style rotate_half (interleaved pattern).

    Llama-style: [-x2, x1] where x1=x[..., :d//2], x2=x[..., d//2:]
    ERNIE-style: [-x_odd, x_even] interleaved

    For input [1,2,3,4,5,6,7,8]:
      - Llama gives: [-5,-6,-7,-8, 1, 2, 3, 4]
      - ERNIE gives: [-2, 1,-4, 3,-6, 5,-8, 7]
    """
    x1 = x[..., 0::2]  # even indices: [1,3,5,7]
    x2 = x[..., 1::2]  # odd indices:  [2,4,6,8]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_ernie(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply ERNIE-4.5 GLM-style rotary position embeddings.

    The key differences from Llama:
    1. rotate_half uses interleaved pattern
    2. cos/sin need repeat_interleave(2) to match the interleaved Q/K dimensions

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine cache from rotary embedding
        sin: Sine cache from rotary embedding
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        q_embed, k_embed: Rotated Q and K tensors
    """
    original_dtype = q.dtype

    # Unsqueeze for broadcasting to heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # ERNIE uses half the RoPE dimensions but repeats interleaved
    # cos/sin shape is [1, 1, seq, dim] -> need [1, 1, seq, dim*2] with repeat_interleave
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    # Apply rotary embeddings with interleaved rotate_half
    q_embed = (q.float() * cos) + (rotate_half_ernie(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half_ernie(k).float() * sin)

    return q_embed.to(original_dtype), k_embed.to(original_dtype)


class Ernie4_5NeuronConfig(NeuronConfig):
    """
    NeuronX configuration for ERNIE-4.5 model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use our custom attention class with GLM-style RoPE
        self.attn_cls = NeuronErnie4_5Attention


class Ernie4_5InferenceConfig(InferenceConfig):
    """
    Configuration class for ERNIE-4.5 inference on NeuronX hardware.
    """

    def add_derived_config(self):
        """Add derived configuration parameters specific to ERNIE-4.5."""
        self.num_cores_per_group = 1
        self.qkv_bias = False
        self.o_bias = False

    def get_required_attributes(self) -> List[str]:
        """List of required configuration attributes for ERNIE-4.5."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Ernie4_5NeuronConfig]:
        """Returns the NeuronConfig class to use for ERNIE-4.5."""
        return Ernie4_5NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """Load configuration from a pretrained model directory."""
        import json
        import os
        import sys

        if neuron_config is None:
            neuron_config = Ernie4_5NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )

        # Load HuggingFace config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            hf_config = json.load(f)

        # Extract all relevant parameters
        config_dict = {
            'hidden_size': hf_config.get('hidden_size', 1024),
            'num_attention_heads': hf_config.get('num_attention_heads', 16),
            'num_hidden_layers': hf_config.get('num_hidden_layers', 18),
            'num_key_value_heads': hf_config.get('num_key_value_heads', 2),
            'vocab_size': hf_config.get('vocab_size', 103424),
            'max_position_embeddings': hf_config.get('max_position_embeddings', 131072),
            'rope_theta': hf_config.get('rope_theta', 500000.0),
            'rms_norm_eps': hf_config.get('rms_norm_eps', 1e-5),
            'hidden_act': hf_config.get('hidden_act', 'silu'),
            'intermediate_size': hf_config.get('intermediate_size', 3072),
            'pad_token_id': hf_config.get('pad_token_id', 0),
            'bos_token_id': hf_config.get('bos_token_id', 1),
            'eos_token_id': hf_config.get('eos_token_id', 2),
            'tie_word_embeddings': hf_config.get('tie_word_embeddings', True),
            'use_bias': hf_config.get('use_bias', False),
            'output_attentions': False,
            'output_hidden_states': False,
            'use_cache': True,
            # ERNIE uses head_dim=128 (not hidden_size // num_attention_heads = 64)
            'head_dim': hf_config.get('head_dim', 128),
        }

        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronErnie4_5Attention(NeuronAttentionBase):
    """
    ERNIE-4.5 attention mechanism with GLM-style RoPE for NeuronX hardware.

    CRITICAL: Overrides apply_rotary_embedding to use GLM-style interleaved pattern.
    """

    def __init__(self, config: Ernie4_5InferenceConfig):
        """Initialize ERNIE-4.5 attention layer."""
        head_dim = getattr(config, 'head_dim', 128)
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            rotary_emb=rotary_emb,
        )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """
        Override base class to use ERNIE-4.5's GLM-style RoPE.

        This is the CRITICAL FIX - without this override, the base class uses
        Llama-style RoPE which produces incorrect results for ERNIE-4.5.
        """
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb_ernie(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


class NeuronErnie4_5MLP(nn.Module):
    """
    ERNIE-4.5 MLP (Feed-Forward Network) with SwiGLU activation.
    """

    def __init__(self, config: Ernie4_5InferenceConfig):
        """Initialize ERNIE-4.5 MLP layer."""
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.sequence_parallel_enabled = getattr(
            config.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        # Gate and up projections
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )

        # Down projection
        from neuronx_distributed.parallel_layers.layers import RowParallelLinear
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        """Forward pass: SwiGLU activation."""
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        return output, None


class NeuronErnie4_5DecoderLayer(nn.Module):
    """ERNIE-4.5 decoder layer with Pre-Norm architecture."""

    def __init__(self, config: Ernie4_5InferenceConfig):
        """Initialize ERNIE-4.5 decoder layer."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronErnie4_5Attention(config)
        self.mlp = NeuronErnie4_5MLP(config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass of the decoder layer."""
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronErnie4_5Model(NeuronBaseModel):
    """ERNIE-4.5 base model for NeuronX hardware."""

    def setup_attr_for_model(self, config: Ernie4_5InferenceConfig):
        """Setup model attributes."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Ernie4_5InferenceConfig):
        """Initialize the model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        self.layers = nn.ModuleList(
            [NeuronErnie4_5DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronErnie4_5ForCausalLM(NeuronBaseForCausalLM):
    """ERNIE-4.5 model for causal language modeling on NeuronX hardware."""

    _model_cls = NeuronErnie4_5Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace ERNIE-4.5 model."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace ERNIE-4.5 state dict to NeuronX format.

        Simple mapping - just remove "model." prefix. No qkv_proj nesting needed.
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}

        # Remove "model." prefix
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]
                neuron_state_dict[new_key] = value
            else:
                neuron_state_dict[key] = value

        # Add rank tensors for tensor parallelism
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Update state dict for tied weights."""
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Returns the configuration class for ERNIE-4.5."""
        return Ernie4_5InferenceConfig

    def get_compiler_args(self):
        """Get compiler arguments."""
        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            "--auto-cast=none "
            "--model-type transformer "
            "-O1"
        )
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 "
            "--vectorize-strided-dma'"
        )
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args


__all__ = [
    "Ernie4_5InferenceConfig",
    "Ernie4_5NeuronConfig",
    "NeuronErnie4_5ForCausalLM",
    "NeuronErnie4_5Model",
    "NeuronErnie4_5Attention",
    "NeuronErnie4_5MLP",
    "NeuronErnie4_5DecoderLayer",
]
