# coding=utf-8
# Copyright 2026 Google Inc. and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Gemma4 model for NeuronX Distributed Inference

Gemma 4 31B key differences from Gemma 3:
- attention_k_eq_v=true on global layers (V copies K, no v_proj)
- global_head_dim=512, head_dim=256 (heterogeneous layer shapes)
- num_global_key_value_heads=4, num_key_value_heads=16
- v_norm (RMSNorm without scale) on all layers
- layer_scalar per layer (learned, multiplied at end of decoder forward)
- scaling=1.0 (no query_pre_attn_scalar)
- final_logit_softcapping=30.0
- partial_rotary_factor=0.25 for global layers
- layer_types explicit list instead of sliding_window_pattern
- rope_parameters with per-layer-type config
"""

import copy
import json
import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)
from neuronx_distributed_inference.modules.kvcache.utils import get_kv_shapes


# ====================================================================================
# Normalization
# ====================================================================================


class Gemma4RMSNorm(nn.Module):
    """
    Gemma4 RMSNorm with standard weight scaling.
    HF initializes weight to ones and applies: normed * weight
    (NOT the (1+weight) pattern from earlier Gemma versions).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)


class Gemma4VNorm(nn.Module):
    """
    Gemma4 V-norm: RMSNorm WITHOUT learnable scale (with_scale=False in HF).
    Applied to value states in attention.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.type_as(x)


def get_rmsnorm_cls():
    """Return appropriate RMSNorm for current execution context."""
    # Gemma4 uses (1 + weight) scaling which CustomRMSNorm doesn't support yet
    return Gemma4RMSNorm


# ====================================================================================
# Embeddings
# ====================================================================================


class SoftcappedLMHead(nn.Module):
    """
    Wrapper that applies final_logit_softcapping after the lm_head linear.
    Implements: cap * tanh(logits / cap)

    This is applied within the lm_head module so we don't need to override
    NeuronBaseModel.forward(). The base class does `logits.float()` after
    lm_head, but since tanh output is already in a safe range this is fine.
    """

    def __init__(self, linear: nn.Module, cap: float):
        super().__init__()
        self.linear = linear
        self.cap = cap

    def forward(self, x):
        logits = self.linear(x)
        # Apply in float32 for numerical precision
        logits = logits.float()
        logits = self.cap * torch.tanh(logits / self.cap)
        return logits

    def __getattr__(self, name):
        """Proxy attributes to the wrapped linear (e.g., pad_size, gather_output,
        tensor_parallel_group) so base class checks still work."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)


class Gemma4ScaledEmbedding(nn.Module):
    """
    Gemma4 scaled embeddings: embed * sqrt(hidden_size)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        dtype: torch.dtype,
        shard_across_embedding: bool = True,
        pad: bool = True,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()
        self.embed_scale = embedding_dim**0.5
        self.embedding = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            pad=pad,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids: torch.Tensor):
        return self.embedding(input_ids) * self.embed_scale


# ====================================================================================
# Configuration
# ====================================================================================


class Gemma4NeuronConfig(NeuronConfig):
    """NeuronConfig with Gemma4-specific attention class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Will be set per-layer in get_updated_configs()
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    """
    Configuration for Gemma4 inference on NeuronX.
    Reads from HuggingFace config.json and extracts text_config fields.
    """

    def __init__(
        self,
        neuron_config: NeuronConfig,
        fused_spec_config=None,
        load_config=None,
        **kwargs,
    ):
        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config

        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        # Gemma4 nests text params under text_config (may be dict or object)
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            # Convert dict to SimpleNamespace for attribute access
            if isinstance(text_config, dict):
                self.text_config = SimpleNamespace(**text_config)
                text_config = self.text_config
            text_attrs = [
                "hidden_size",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "rms_norm_eps",
                "sliding_window",
                "hidden_activation",
                # Gemma4-specific
                "global_head_dim",
                "num_global_key_value_heads",
                "attention_k_eq_v",
                "final_logit_softcapping",
                "layer_types",
                "rope_parameters",
            ]
            for attr in text_attrs:
                if isinstance(text_config, dict):
                    if attr in text_config:
                        setattr(self, attr, text_config[attr])
                elif hasattr(text_config, attr):
                    setattr(self, attr, getattr(text_config, attr))

        # Also convert vision_config dict to namespace if present
        vision_config = getattr(self, "vision_config", None)
        if vision_config is not None and isinstance(vision_config, dict):
            self.vision_config = SimpleNamespace(**vision_config)

        # Ensure text_config has attributes required by NeuronBaseForCausalLM._setup_func_config()
        # (model_base.py:3516). HF PretrainedConfig provides these by default, but
        # SimpleNamespace conversion loses them.
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for attr, default in [
                ("output_attentions", False),
                ("output_hidden_states", False),
                ("use_return_dict", True),
            ]:
                if not hasattr(text_config, attr):
                    setattr(text_config, attr, default)
        # Also set on self for models that use get_text_config() -> self
        for attr, default in [
            ("output_attentions", False),
            ("output_hidden_states", False),
            ("use_return_dict", True),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)

        # Defaults for attributes that may not be in config
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = 0
        if not hasattr(self, "tie_word_embeddings"):
            self.tie_word_embeddings = True
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        # hidden_act for NeuronLlamaMLP compatibility
        if hasattr(self, "hidden_activation") and not hasattr(self, "hidden_act"):
            self.hidden_act = self.hidden_activation

        self.add_derived_config()
        self.validate_config()

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "intermediate_size",
            "global_head_dim",
            "num_global_key_value_heads",
            "layer_types",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4NeuronConfig]:
        return Gemma4NeuronConfig


def get_updated_configs(config: Gemma4InferenceConfig):
    """
    Generate per-layer configs for heterogeneous SWA/global layers.

    SWA layers: head_dim=256, num_key_value_heads=16, sliding_window=1024
    Global layers: head_dim=512, num_key_value_heads=4, sliding_window=None, attention_k_eq_v=True

    NKI flash attention kernel limit is head_dim=128, so both layer types
    must use decomposed attention (attn_kernel_enabled=False).
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)
        layer_type = config.layer_types[i]

        if layer_type == "sliding_attention":
            layer_config.sliding_window = config.sliding_window
            layer_config._layer_head_dim = config.head_dim
            layer_config._layer_num_kv_heads = config.num_key_value_heads
            layer_config._layer_is_sliding = True
            layer_config._layer_k_eq_v = False
            # RoPE: default type, theta=10000
            rope_params = config.rope_parameters.get("sliding_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 10000.0)
            layer_config._layer_partial_rotary_factor = 1.0  # full rotation for SWA
        else:
            # full_attention (global)
            layer_config.sliding_window = None
            layer_config._layer_head_dim = config.global_head_dim
            layer_config._layer_num_kv_heads = config.num_global_key_value_heads
            layer_config._layer_is_sliding = False
            layer_config._layer_k_eq_v = getattr(config, "attention_k_eq_v", False)
            # RoPE: proportional type, theta=1000000, partial_rotary_factor=0.25
            rope_params = config.rope_parameters.get("full_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 1000000.0)
            layer_config._layer_partial_rotary_factor = rope_params.get(
                "partial_rotary_factor", 0.25
            )

        updated_configs.append(layer_config)

    return updated_configs


# ====================================================================================
# Attention
# ====================================================================================


class NeuronGemma4Attention(NeuronAttentionBase):
    """
    Gemma4 attention with:
    - Per-layer head_dim (256 for SWA, 512 for global)
    - Per-layer KV head count (16 for SWA, 4 for global)
    - QK normalization via (1+weight) RMSNorm
    - V normalization (RMSNorm without scale)
    - Partial rotary for global layers (0.25 factor)
    - attention_k_eq_v handled at weight level (V weights = K weights)
    """

    def __init__(self, config: Gemma4InferenceConfig):
        head_dim = config._layer_head_dim
        num_kv_heads = config._layer_num_kv_heads
        is_sliding = config._layer_is_sliding
        rope_theta = config._layer_rope_theta
        partial_rotary_factor = config._layer_partial_rotary_factor

        # RoPE dimension: for global layers with partial_rotary_factor=0.25,
        # only 25% of dims get RoPE. But RotaryEmbedding always rotates dim/2 pairs,
        # so we pass the rotary dim = head_dim * partial_rotary_factor (rounded to even).
        rotary_dim = int(head_dim * partial_rotary_factor)
        # Ensure even
        rotary_dim = rotary_dim - (rotary_dim % 2)

        rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # IMPORTANT: Pass sliding_window=None to base class for ALL layers.
        # This routes all layers through standard_causal_attention_forward instead of
        # windowed_attention_forward. The windowed_attention_forward path calls
        # get_last_kv_window() which does torch.gather with indices up to
        # sliding_window-2, but during CTE the K/V only has bucket_size positions,
        # causing OOB when bucket_size < sliding_window (Discovery #27).
        # SWA windowed behavior is enforced via local_mask at the decoder layer level.
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            sliding_window=None,
            post_transpose_layernorm=True,
        )

        # QK norms: Gemma4 uses (1+weight) RMSNorm
        self.q_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)

        # V norm: RMSNorm without learnable scale
        self.v_norm = Gemma4VNorm(dim=head_dim, eps=config.rms_norm_eps)

        # Store layer properties
        self._is_sliding = is_sliding
        self._k_eq_v = config._layer_k_eq_v
        self._head_dim = head_dim
        self._rotary_dim = rotary_dim
        self._partial_rotary_factor = partial_rotary_factor

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """
        Override to handle partial rotary embedding for global layers.

        For SWA layers: partial_rotary_factor=1.0, rotary_dim==head_dim -> full rotation (same as base).
        For global layers: partial_rotary_factor=0.25, rotary_dim=128, head_dim=512 ->
            only rotate the first 128 dims, leave the remaining 384 unchanged.
        """
        if self.rotary_emb is None:
            return Q, K, cos_cache, sin_cache

        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        if self._rotary_dim == self._head_dim:
            # Full rotation (SWA layers) - use standard path
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        else:
            # Partial rotation (global layers) - split, rotate, concatenate
            # Q, K are in BHSD layout: [batch, num_heads, seq, head_dim]
            q_rot = Q[..., : self._rotary_dim]
            q_pass = Q[..., self._rotary_dim :]
            k_rot = K[..., : self._rotary_dim]
            k_pass = K[..., self._rotary_dim :]

            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_cache, sin_cache)

            Q = torch.cat([q_rot, q_pass], dim=-1)
            K = torch.cat([k_rot, k_pass], dim=-1)

        return Q, K, cos_cache, sin_cache

    # forward() is inherited from NeuronAttentionBase -- no override needed.
    # The v_norm injection happens in prep_qkv_tensors below.

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """
        Override to apply v_norm to value states after QKV projection.

        IMPORTANT: Argument order matches base class:
        prep_qkv_tensors(position_ids, hidden_states, ...) NOT (hidden_states, position_ids, ...)
        """
        Q, K, V, cos_cache, sin_cache, residual = super().prep_qkv_tensors(
            position_ids=position_ids,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=skip_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        # Apply v_norm to V tensor (BHSD layout after move_heads_front)
        # v_norm is RMSNorm without learnable scale, applied on last dim (head_dim)
        V = self.v_norm(V)

        return Q, K, V, cos_cache, sin_cache, residual


# ====================================================================================
# MLP
# ====================================================================================


class NeuronGemma4MLP(nn.Module):
    """
    Gemma4 MLP with GELU(tanh) activation.
    gate_proj and up_proj are column-parallel, down_proj is row-parallel.
    """

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), None


# ====================================================================================
# Decoder Layer
# ====================================================================================


class NeuronGemma4DecoderLayer(nn.Module):
    """
    Gemma4 decoder layer with:
    - 4 RMSNorm layers
    - layer_scalar at the end (learned per-layer multiplicative factor)
    """

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_sliding_window_attention = config._layer_is_sliding

        self.self_attn = NeuronGemma4Attention(config)
        self.mlp = NeuronGemma4MLP(config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # layer_scalar: learned per-layer scaling factor applied at the end.
        # Must be nn.Parameter (not buffer) so NxDI's weight loading populates it
        # from the checkpoint. Buffers are baked as constants at trace time.
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Gemma4 has heterogeneous RoPE configs per layer (different theta, rotary_dim).
        # The base model loop caches cos/sin from the previous layer, but we must
        # force recomputation for each layer since configs differ.
        kwargs.pop("cos_cache", None)
        kwargs.pop("sin_cache", None)

        # Select mask: SWA layers use local_mask (windowed), global uses attention_mask
        local_mask = kwargs.pop("local_mask", None)
        mask = (
            local_mask
            if (self.is_sliding_window_attention and local_mask is not None)
            else attention_mask
        )

        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Per-layer scaling
        hidden_states = hidden_states * self.layer_scalar

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# KV Cache
# ====================================================================================


class Gemma4KVCacheManager(KVCacheManager):
    """
    KV cache manager for Gemma4 with per-layer heterogeneous shapes.

    SWA layers: (B, 4, S, 256) per K/V
    Global layers: (B, 1, S, 512) per K/V

    We create per-layer cache tensors with their actual shapes rather than
    using a uniform shape for all layers.
    """

    def __init__(
        self,
        config,
        layer_kv_configs,
        global_rank=None,
        attention_chunk_size=None,
        sliding_window=None,
        windowed_context_encoding_size=None,
        layer_to_cache_size_mapping=None,
    ):
        self._layer_kv_configs = layer_kv_configs

        # We MUST pass layer_to_cache_size_mapping to trigger the per-layer branch
        # in the base __init__ (lines 152-157). If not provided, create a uniform one.
        if layer_to_cache_size_mapping is None:
            max_len = config.neuron_config.max_length
            layer_to_cache_size_mapping = [max_len] * len(layer_kv_configs)

        max_kv_heads = max(c[0] for c in layer_kv_configs)
        super().__init__(
            config,
            num_kv_head=max_kv_heads,
            global_rank=global_rank,
            attention_chunk_size=attention_chunk_size,
            sliding_window=sliding_window,
            windowed_context_encoding_size=windowed_context_encoding_size,
            layer_to_cache_size_mapping=layer_to_cache_size_mapping,
        )

    def _init_kv_shape(self, config, layer_to_cache_size_mapping=None):
        """
        Override to create per-layer KV cache shapes based on heterogeneous configs.

        Only sets self.k_shapes, self.v_shapes, and the fallback self.k_shape/self.v_shape.
        The base class __init__ will create self.past_key_values from these shapes
        when layer_to_cache_size_mapping is provided.
        """
        max_batch_size = (
            config.neuron_config.kv_cache_batch_size
            + config.neuron_config.kv_cache_padding_size
        )
        max_len = config.neuron_config.max_length

        if (
            self.attention_chunk_size
            and self.attention_chunk_size < max_len
            and not layer_to_cache_size_mapping
        ):
            max_len = self.attention_chunk_size
        elif self.sliding_window:
            max_len = self.sliding_window

        # Determine per-layer cache sequence lengths
        if layer_to_cache_size_mapping:
            layer_seq_lens = list(layer_to_cache_size_mapping)
        else:
            layer_seq_lens = [max_len] * len(self._layer_kv_configs)

        # Create per-layer k_shapes and v_shapes
        self.k_shapes = []
        self.v_shapes = []
        self.padded_layer_ids = []
        for idx, (kv_heads_per_rank, head_dim) in enumerate(self._layer_kv_configs):
            cache_len = layer_seq_lens[idx]
            k_shape, v_shape = get_kv_shapes(
                cache_len,
                max_batch_size,
                kv_heads_per_rank,
                head_dim,
                self.k_cache_transposed,
                self.is_kv_cache_tiled,
            )
            self.k_shapes.append(k_shape)
            self.v_shapes.append(v_shape)

        # Also set the default shapes to the max for any code that uses self.k_shape
        max_kv_heads = max(c[0] for c in self._layer_kv_configs)
        max_head_dim = max(c[1] for c in self._layer_kv_configs)
        self.k_shape, self.v_shape = get_kv_shapes(
            max_len,
            max_batch_size,
            max_kv_heads,
            max_head_dim,
            self.k_cache_transposed,
            self.is_kv_cache_tiled,
        )


# ====================================================================================
# Model
# ====================================================================================


class NeuronGemma4TextModel(NeuronBaseModel):
    """Gemma4 text model: embeddings + decoder layers + final norm + lm_head."""

    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # Use the maximum KV head count (SWA = 16) for base class compatibility.
        # The actual per-layer head counts are handled via per-layer KV cache shapes.
        self.num_key_value_heads = config.num_key_value_heads  # 16 (SWA)
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def scatter_by_index_put(self, h_image, encoded_patches_proj, positions):
        """
        Scatter encoded vision patches into the text embedding tensor.
        Supports batch size >= 1.

        Args:
            h_image: [B, max_positions, embedding_dim] - text embeddings
            encoded_patches_proj: [B, num_vision_tokens, embedding_dim] - vision embeddings
            positions: [B, num_positions, 1] - scatter positions
        """
        B, max_positions, embedding_dim = h_image.shape
        h_image_new = h_image.clone()
        encoded_patches_flat = encoded_patches_proj.view(-1, embedding_dim)
        positions = positions.view(-1)

        num_updates_per_batch = positions.shape[0] // B
        batch_idx = torch.arange(B, device=h_image.device, dtype=positions.dtype)
        batch_idx = batch_idx.repeat_interleave(num_updates_per_batch)

        h_image_new.index_put_(
            (batch_idx.long(), positions.long()),
            encoded_patches_flat,
            accumulate=False,
        )
        return h_image_new

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        """Merge vision embeddings into text embeddings during context encoding."""
        return self.scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def init_model(self, config: Gemma4InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # Per-layer configs for heterogeneous SWA/global shapes
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronGemma4DecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )

        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        lm_head_linear = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        # Wrap with softcapping if configured
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping > 0
        ):
            self.lm_head = SoftcappedLMHead(
                lm_head_linear, self.final_logit_softcapping
            )
        else:
            self.lm_head = lm_head_linear

        # Mixed attention: SWA layers have sliding_window, global layers don't.
        # Set attributes needed by base model for cache management.
        self.has_mixed_attn = True
        self.sliding_window = config.sliding_window  # SWA window size (1024)

        # Compute per-layer cache sizes: ALL layers use uniform_cache_len because
        # the KV cache sequence dimension must match the attention mask dimension
        # in compute_for_token_gen's torch.where(mask, prior_scores, ...).
        max_length = config.neuron_config.max_length
        sw = config.sliding_window or max_length
        self._uniform_cache_len = max(sw, max_length)
        self.layer_to_cache_size_mapping = [
            self._uniform_cache_len
        ] * config.num_hidden_layers

    def _create_simple_attn_mask(self, attention_mask):
        """Override: global (non-SWA) mask must match uniform KV cache size.

        The base class creates a mask of shape (B, 1, 1, n_positions), but our
        KV caches are sized to uniform_cache_len = max(sliding_window, max_length).
        During token gen, compute_for_token_gen does torch.where(mask, prior_scores, ...)
        where prior_scores has last dim = cache_seq_len = uniform_cache_len.
        We pad the mask with False (masked out) so shapes match.
        """
        batch_size = attention_mask.shape[0]
        # attention_mask is (B, n_positions) -- pad to uniform_cache_len
        pad_len = self._uniform_cache_len - self.n_positions
        if pad_len > 0:
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        return (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, 1, self._uniform_cache_len)
            .to(torch.bool)
        )

    def init_inference_optimization(self, config: Gemma4InferenceConfig):
        """
        Override to create per-layer KV caches with correct heterogeneous shapes.

        Gemma4 has:
        - SWA layers: num_kv_heads=16, head_dim=256 -> per-rank: 4 heads, 256 dim
        - Global layers: num_kv_heads=4, head_dim=512 -> per-rank: 1 head, 512 dim

        The standard KVCacheManager uses one shape for all layers. We create a
        custom one that allocates per-layer shapes.
        """
        if self.on_device_sampling:
            from neuronx_distributed_inference.modules.sampling.utils import (
                create_sampler,
            )

            lm_head_tp_degree = None
            if hasattr(self, "lm_head") and hasattr(
                self.lm_head, "tensor_parallel_group"
            ):
                lm_head_tp_degree = self.lm_head.tensor_parallel_group.size()
            self.sampler = create_sampler(config.neuron_config, lm_head_tp_degree)

        # Compute per-layer KV head counts and head dims
        tp_degree = config.neuron_config.tp_degree
        layer_kv_configs = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            if layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads  # 16
                hd = config.head_dim  # 256
            else:
                kv_heads = config.num_global_key_value_heads  # 4
                hd = config.global_head_dim  # 512

            # Compute sharded KV heads per rank (bypass parallel_state check)
            gqa_strategy = determine_sharding_strategy(tp_degree, kv_heads)
            _, shardable_kv_heads = get_shardable_head_counts(
                tp_degree, config.num_attention_heads, kv_heads, gqa_strategy
            )
            kv_heads_per_rank = shardable_kv_heads // tp_degree
            layer_kv_configs.append((kv_heads_per_rank, hd))

        # Use the MAXIMUM kv cache shape across all layers so the KVCacheManager
        # uses a uniform shape. Then we'll handle the per-layer differences by
        # padding/slicing in the attention module.
        max_kv_heads_per_rank = max(c[0] for c in layer_kv_configs)
        max_head_dim = max(c[1] for c in layer_kv_configs)

        # Store per-layer configs for attention modules to use
        self._layer_kv_configs = layer_kv_configs
        self._max_kv_heads_per_rank = max_kv_heads_per_rank
        self._max_head_dim = max_head_dim

        # Create KVCacheManager with per-layer shapes
        # We'll use the layer_to_cache_size_mapping mechanism, but we also need
        # different head/dim per layer. We do this by creating a uniform cache
        # with the maximum dimensions and handling the mismatch per layer.
        self.kv_mgr = Gemma4KVCacheManager(
            config,
            layer_kv_configs=layer_kv_configs,
            global_rank=self.rank_util,
            attention_chunk_size=self.attention_chunk_size,
            sliding_window=self.sliding_window,
            windowed_context_encoding_size=self.windowed_context_encoding_size,
            layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
        )


# ====================================================================================
# Top-level Model Class
# ====================================================================================


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """
    Gemma4 causal LM for NeuronX inference.
    Handles weight loading, state dict conversion, and tied weights.
    """

    _model_cls = NeuronGemma4TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: Gemma4InferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert HuggingFace Gemma4 state dict to NeuronX format.

        Key transformations:
        1. Strip 'language_model.model.' / 'language_model.' prefixes
        2. Remap embed_tokens -> embed_tokens.embedding (for ScaledEmbedding wrapper)
        3. Remap q_norm/k_norm -> q_layernorm/k_layernorm
        4. For global layers with attention_k_eq_v: copy k_proj weights to v_proj
        5. Add v_norm weights (no learnable params, but buffer may be needed)
        6. Fuse QK scaling correction into Q weights (cancel NxDI's 1/sqrt(head_dim))
        7. Add rank_util tensors for TP
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}

        for key, weight in state_dict.items():
            new_key = key

            # Strip HF prefixes
            if new_key.startswith("language_model.model."):
                new_key = new_key[len("language_model.model.") :]
            elif new_key.startswith("language_model."):
                new_key = new_key[len("language_model.") :]
            elif new_key.startswith("model.language_model.model."):
                new_key = new_key[len("model.language_model.model.") :]
            elif new_key.startswith("model.language_model."):
                new_key = new_key[len("model.language_model.") :]

            # Skip vision tower weights (text-only for now)
            if (
                "vision_tower." in new_key
                or "multi_modal_projector." in new_key
                or "embed_vision." in new_key
            ):
                continue

            # Remap embedding key for ScaledEmbedding wrapper
            if new_key == "embed_tokens.weight":
                new_key = "embed_tokens.embedding.weight"

            # Remap QK norm keys
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            new_state_dict[new_key] = weight.detach().clone()

        # Per-layer transformations
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            is_global = layer_type == "full_attention"

            if is_global:
                hd = config.global_head_dim
                kv_heads = config.num_global_key_value_heads
            else:
                hd = config.head_dim
                kv_heads = config.num_key_value_heads

            prefix = f"layers.{i}.self_attn"

            # --- QK scaling correction ---
            # Gemma4 uses scaling=1.0 (no 1/sqrt(head_dim) in attention scores).
            # NxDI always applies 1/sqrt(head_dim) in scaled_qk.
            # We CANNOT pre-scale Q weights because q_norm (RMSNorm) is applied
            # AFTER q_proj, and RMSNorm is scale-invariant: RMSNorm(x*a) = RMSNorm(x).
            # The scaling would be absorbed by normalization.
            #
            # Instead, we scale the q_layernorm WEIGHTS by sqrt(head_dim):
            #   Q_normed = Q / rms(Q) * (w * sqrt(hd)) = RMSNorm(Q) * sqrt(hd)
            #   NxDI: (Q_normed @ K^T) / sqrt(hd) = (RMSNorm(Q)*sqrt(hd) @ K^T) / sqrt(hd)
            #        = RMSNorm(Q) @ K^T  ← matches HF's scaling=1.0
            q_norm_key = f"{prefix}.q_layernorm.weight"
            if q_norm_key in new_state_dict:
                scaling_factor = math.sqrt(float(hd))
                orig_dtype = new_state_dict[q_norm_key].dtype
                new_state_dict[q_norm_key] = (
                    new_state_dict[q_norm_key].to(torch.float32) * scaling_factor
                ).to(orig_dtype)

            # --- attention_k_eq_v: copy K weights to V for global layers ---
            if is_global and getattr(config, "attention_k_eq_v", False):
                k_key = f"{prefix}.k_proj.weight"
                v_key = f"{prefix}.v_proj.weight"
                if k_key in new_state_dict and v_key not in new_state_dict:
                    new_state_dict[v_key] = new_state_dict[k_key].detach().clone()

            # --- rank_util for TP ---
            new_state_dict[f"{prefix}.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Vocabulary parallelism rank
        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.embedding.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Base model rank
        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights: embed_tokens -> lm_head.

        When SoftcappedLMHead wraps the linear, weights live at 'lm_head.linear.weight'.
        Set both keys so it works regardless of wrapper presence.
        """
        embed_key = None
        if "embed_tokens.embedding.weight" in state_dict:
            embed_key = "embed_tokens.embedding.weight"
        elif "embed_tokens.weight" in state_dict:
            embed_key = "embed_tokens.weight"

        if embed_key is not None:
            weight = state_dict[embed_key].clone()
            # Set both possible lm_head paths
            state_dict["lm_head.weight"] = weight
            state_dict["lm_head.linear.weight"] = weight.clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig
