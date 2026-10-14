# coding=utf-8
# Copyright 2025 Google Inc. All rights reserved.
# Ported to NeuronX Distributed Inference for AWS Trainium2.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Gemma4 E2B model for NeuronX Distributed Inference (text-only path).

Key architectural features vs Gemma3:
- Per-Layer Embeddings (PLE): each decoder layer gets a unique embedding input
- Variable head_dim: sliding_attention=256, full_attention=512
- KV shared layers: last 20 layers reuse KV from earlier layers (skipped in v1)
- Double-wide MLP: KV-shared layers use 2x intermediate_size
- layer_scalar: per-layer learned scaling factor
- Hybrid attention: alternating sliding/full patterns with different RoPE configs

Design decisions for Neuron compilation:
- head_dim=512 for KV cache uniformity; sliding layer weights zero-padded from 256->512
- Q-K layernorm weights compensated for zero-padding RMS difference (sqrt(2) factor)
- Sliding window DISABLED (NKI kernel doesn't support head_dim > 128)
- KV sharing DISABLED for v1 (all 35 layers get own cache)
- PLE fully implemented
"""

import json
import math
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Layer types from config.json
SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# Default layer_types for Gemma4 E2B (35 layers)
DEFAULT_LAYER_TYPES = [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
]


# ====================================================================================
# Configuration
# ====================================================================================


class Gemma4NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    """
    Configuration for Gemma4 E2B on NeuronX.

    Loads text_config from the nested HF config and exposes Gemma4-specific
    parameters (PLE, variable head_dim, layer_types, etc.).
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1

        # Standard flags
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True

        # Gemma4 text_config fields (set defaults if missing)
        if not hasattr(self, "query_pre_attn_scalar"):
            self.query_pre_attn_scalar = 256

        # Sliding window disabled on Neuron (head_dim > 128 NKI limit)
        self.sliding_window = None

        if not hasattr(self, "rope_local_base_freq"):
            self.rope_local_base_freq = 10000.0
        if not hasattr(self, "attn_logit_softcapping"):
            self.attn_logit_softcapping = None
        if not hasattr(self, "final_logit_softcapping"):
            self.final_logit_softcapping = getattr(self, "final_logit_softcapping", 30.0)
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        # Layer types
        if not hasattr(self, "layer_types"):
            self.layer_types = DEFAULT_LAYER_TYPES[:self.num_hidden_layers]

        # PLE config
        if not hasattr(self, "hidden_size_per_layer_input"):
            self.hidden_size_per_layer_input = 256
        if not hasattr(self, "vocab_size_per_layer_input"):
            self.vocab_size_per_layer_input = self.vocab_size

        # Global head dim for full_attention layers
        if not hasattr(self, "global_head_dim"):
            self.global_head_dim = 512

        # Double-wide MLP
        if not hasattr(self, "use_double_wide_mlp"):
            self.use_double_wide_mlp = True
        if not hasattr(self, "num_kv_shared_layers"):
            self.num_kv_shared_layers = 20

        # Activation
        if not hasattr(self, "hidden_activation"):
            self.hidden_activation = "gelu_pytorch_tanh"

        # RoPE config for full attention
        if not hasattr(self, "partial_rotary_factor"):
            self.partial_rotary_factor = 0.25

        # K=V optimization for full attention layers (31B uses this)
        if not hasattr(self, "attention_k_eq_v"):
            self.attention_k_eq_v = False
        if not hasattr(self, "num_global_key_value_heads"):
            self.num_global_key_value_heads = None

        # MoE (Mixture of Experts) config
        if not hasattr(self, "enable_moe_block"):
            self.enable_moe_block = False
        if not hasattr(self, "num_experts"):
            self.num_experts = None
        if not hasattr(self, "top_k_experts"):
            self.top_k_experts = None
        if not hasattr(self, "moe_intermediate_size"):
            self.moe_intermediate_size = None

        # For KV cache: use max head_dim (global_head_dim=512) so all layers are uniform
        self.head_dim = self.global_head_dim

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4NeuronConfig]:
        return Gemma4NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Gemma4InferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            raw = json.load(f)

        # Gemma4 nests text params in text_config
        text_config = raw.get("text_config", {})
        config_dict = {}
        config_dict.update(text_config)
        # Top-level overrides
        for k in ("tie_word_embeddings", "model_type"):
            if k in raw:
                config_dict[k] = raw[k]
        config_dict.setdefault("tie_word_embeddings", True)

        # Rope parameters per attention type
        rope_params = text_config.get("rope_parameters", {})
        full_rope = rope_params.get("full_attention", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        config_dict["rope_theta"] = full_rope.get("rope_theta", 1000000.0)
        config_dict["rope_local_base_freq"] = sliding_rope.get("rope_theta", 10000.0)
        config_dict["partial_rotary_factor"] = full_rope.get("partial_rotary_factor", 0.25)

        # EOS from top-level
        if "eos_token_id" in raw:
            config_dict["eos_token_id"] = raw["eos_token_id"]

        config_dict.update(kwargs)

        if neuron_config is None:
            neuron_config = NeuronConfig()

        return cls(neuron_config=neuron_config, **config_dict)


# ====================================================================================
# Components
# ====================================================================================


class Gemma4RMSNorm(nn.Module):
    """RMSNorm with direct weight multiplication (HF Gemma4 style).
    Weight is initialized to ones and multiplied directly: norm(x) * weight.
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


class Gemma4RMSNormNoScale(nn.Module):
    """RMSNorm WITHOUT learnable scale (with_scale=False). Used for v_norm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x)


def get_rmsnorm_cls():
    return Gemma4RMSNorm


class SoftcappedLMHead(nn.Module):
    """Wraps a ColumnParallelLinear LM head with final logit softcapping.

    Applies logits = softcap * tanh(logits / softcap) after the linear projection,
    matching HF Gemma4's final_logit_softcapping behavior.
    """

    def __init__(self, linear, softcap):
        super().__init__()
        self.linear = linear
        self.softcap = softcap
        # Proxy attributes that the base class forward() checks
        for attr in ('pad_size', 'gather_output', 'tensor_parallel_group'):
            if hasattr(linear, attr):
                setattr(self, attr, getattr(linear, attr))

    def forward(self, x):
        logits = self.linear(x)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


class Gemma4ScaledEmbedding(nn.Module):
    """Token embeddings scaled by sqrt(hidden_size)."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, dtype,
                 shard_across_embedding=True, pad=True, sequence_parallel_enabled=False):
        super().__init__()
        self.embed_scale = embedding_dim ** 0.5
        self.embedding = ParallelEmbedding(
            num_embeddings, embedding_dim, padding_idx, dtype=dtype,
            shard_across_embedding=shard_across_embedding, pad=pad,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.embed_scale


class Gemma4PerLayerEmbedding(nn.Module):
    """
    Per-Layer Embedding (PLE) module.

    Computes per-layer inputs from:
    1. embed_tokens_per_layer(input_ids) -> [B, S, num_layers * ple_dim]
    2. per_layer_model_projection(main_embeds) -> [B, S, num_layers * ple_dim]
    3. Combined and normalized
    """

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input  # 256
        total_ple_dim = self.num_layers * self.ple_dim  # 8960

        # Per-layer token embeddings: [vocab, num_layers * ple_dim]
        self.embed_tokens_per_layer = ParallelEmbedding(
            config.vocab_size_per_layer_input,
            total_ple_dim,
            config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        # Project main embeddings to per-layer space
        self.per_layer_model_projection = ColumnParallelLinear(
            config.hidden_size,
            total_ple_dim,
            bias=False,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        # Normalize per-layer projections
        self.per_layer_projection_norm = get_rmsnorm_cls()(self.ple_dim, eps=config.rms_norm_eps)

        # Scales (matching HF Gemma4TextModel)
        self.embed_scale = self.ple_dim ** 0.5  # sqrt(256) = 16, for PLE token embeddings
        self.per_layer_model_projection_scale = config.hidden_size ** -0.5  # 1/sqrt(1536)
        self.per_layer_input_scale = 1.0 / math.sqrt(2.0)

    def forward(self, input_ids, main_embeds):
        """
        Returns per_layer_inputs: [B, S, num_layers, ple_dim]
        """
        # Token PLE embeddings (scaled by sqrt(ple_dim) like HF ScaledWordEmbedding)
        token_ple = self.embed_tokens_per_layer(input_ids) * self.embed_scale  # [B, S, total_ple_dim]
        token_ple = token_ple.view(*input_ids.shape, self.num_layers, self.ple_dim)

        # Project main embeddings (scaled by 1/sqrt(hidden_size))
        proj = self.per_layer_model_projection(main_embeds) * self.per_layer_model_projection_scale
        proj = proj.view(*main_embeds.shape[:-1], self.num_layers, self.ple_dim)
        proj = self.per_layer_projection_norm(proj)

        # Combine
        return (proj + token_ple) * self.per_layer_input_scale


class ProportionalRotaryEmbedding(nn.Module):
    """Proportional RoPE for Gemma4 full_attention layers.

    Matches HF's _compute_proportional_rope_parameters exactly:
    - Creates inv_freq with head_dim//2 elements: rope_angles real freqs + zeros
    - emb = cat(freqs, freqs) → head_dim elements
    - cos/sin shape [B, S, head_dim] (512)
    - Applied with standard rotate_half on full head_dim, which pairs dim i with dim i+256
    - Non-rotated dims get cos=1, sin=0 → pass through unchanged
    """

    def __init__(self, head_dim, rope_proportion, max_position_embeddings, base):
        super().__init__()
        self.head_dim = head_dim
        rope_angles = int(rope_proportion * head_dim // 2)  # 64

        # 64 real frequencies, same as HF
        real_inv_freq = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
        )
        # Pad to head_dim//2 = 256 with zeros (matching HF lines 240-249)
        inv_freq = torch.zeros(head_dim // 2, dtype=torch.float)
        inv_freq[:rope_angles] = real_inv_freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, 256]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, 512]
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _partial_apply_rotary_pos_emb(q, k, cos, sin, rope_dim, unsqueeze_dim=1):
    """Apply RoPE to only the first rope_dim dimensions, pass through the rest."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Split into rotated and pass-through parts
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]

    # Rotate half within the rope_dim portion
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot_emb = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot_emb = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    # Recombine
    q_embed = torch.cat([q_rot_emb, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_emb, k_pass], dim=-1)
    return q_embed, k_embed


def _full_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply standard full-dim RoPE (rotate_half pairs dim i with dim i+head_dim/2).

    Used for full attention layers where cos/sin are [B, S, head_dim] with
    cos=1/sin=0 for non-rotated dimensions, ensuring they pass through unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class NeuronGemma4Attention(NeuronAttentionBase):
    """
    Gemma4 attention with Q-K normalization, variable head_dim, and partial RoPE.

    For Neuron compilation, ALL layers use head_dim=512 (global_head_dim).
    Sliding layers have weights zero-padded from 256->512 in state dict conversion.

    RoPE per layer type:
    - Sliding layers: standard RoPE, rotate first 256 of 512 dims (rope_theta=10000)
    - Full layers: proportional RoPE, rotate 128 of 512 dims (partial_rotary_factor=0.25,
      rope_theta=1000000, freq denominator=head_dim=512)
    """

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int = 0):
        is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION

        # All layers use global_head_dim=512 for uniform KV cache
        head_dim = config.global_head_dim  # 512

        # RoPE config differs per layer type:
        # - Sliding: standard RoPE on all 256 original dims (padded to 512), theta=10000
        # - Full: proportional RoPE, only 128 of 512 dims rotated (partial_rotary_factor=0.25),
        #   theta=1000000, frequencies use full head_dim=512 as denominator
        if is_sliding:
            rope_theta = config.rope_local_base_freq  # 10000
            rope_dim = 256  # original sliding head_dim (rotates first 256 of 512)
            rotary_emb = RotaryEmbedding(
                dim=rope_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=rope_theta,
            )
        else:
            # Full attention: proportional RoPE rotates first 128 of 512 dims
            # (partial_rotary_factor=0.25 * head_dim=512 = 128)
            rope_dim = int(head_dim * config.partial_rotary_factor)  # 128
            rotary_emb = ProportionalRotaryEmbedding(
                head_dim=head_dim,
                rope_proportion=config.partial_rotary_factor,  # 0.25
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,  # 1000000
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            sliding_window=None,
            q_layernorm=get_rmsnorm_cls()(head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(head_dim, eps=config.rms_norm_eps),
        )

        # V normalization (HF Gemma4 applies v_norm with_scale=False - no learnable weight)
        self.v_layernorm = Gemma4RMSNormNoScale(dim=head_dim, eps=config.rms_norm_eps)

        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.is_sliding = is_sliding
        self.rope_dim = rope_dim
        # HF Gemma4 uses scaling = 1.0 (Q-K RMSNorms keep scores reasonable).
        # The NxDI framework divides QK by sqrt(head_dim=512). Pre-multiply Q
        # by sqrt(head_dim) so net scaling = Q*sqrt(512)·K / sqrt(512) = Q·K = 1.0.
        self._q_scale_factor = math.sqrt(head_dim)

        # V norm compensation for zero-padded sliding layers.
        # When V is zero-padded from 256→512 dims, the RMS is computed over
        # 512 dims (half zeros), making the normalized output sqrt(2) too large.
        # Q/K norms compensate via weight scaling, but V norm has no learnable
        # weight (with_scale=False). We apply the factor after V norm instead.
        if is_sliding:
            orig_head_dim = 256  # sliding attention original head_dim
            self._v_norm_compensation = math.sqrt(orig_head_dim / head_dim)  # sqrt(256/512) = 1/sqrt(2)
        else:
            self._v_norm_compensation = 1.0

        # KV sharing: layers >= first_kv_shared reuse K/V from source layers.
        # In HF Gemma4, shared layers skip K/V projection and use cached K/V
        # from the last non-shared layer of the same type (sliding/full).
        first_kv_shared = config.num_hidden_layers - getattr(config, 'num_kv_shared_layers', 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared > 0
        self.kv_source_idx = None
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared]
            self.kv_source_idx = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
        self.layer_idx = layer_idx
        # _kv_buffer is a model-level dict {layer_idx: (K, V)} set by model.get_model_output
        # Source layers write their K/V here; shared layers read from their source
        self._kv_buffer = None  # set to model's _kv_buffer dict by init_model

    def prep_qkv_tensors(self, position_ids, hidden_states, past_key_value, **kwargs):
        """Override to apply V normalization and handle KV sharing.

        For non-shared layers: compute Q/K/V normally, apply V norm.
        For KV source layers: also cache K/V in _kv_buffer for shared layers.
        For KV shared layers: compute Q only, use source layer's cached K/V.
        """
        if self.is_kv_shared_layer and self._kv_buffer is not None and self.kv_source_idx in self._kv_buffer:
            # Shared layer: compute Q normally, use source K/V
            # We still call super() to get Q (and produce K/V that we'll discard)
            Q, _K, _V, cos_cache, sin_cache, residual = super().prep_qkv_tensors(
                position_ids, hidden_states, past_key_value, **kwargs
            )
            # Use source layer's cached K/V (post-norm, post-RoPE)
            K, V = self._kv_buffer[self.kv_source_idx]
            return Q, K, V, cos_cache, sin_cache, residual

        # Non-shared layer: compute Q/K/V normally
        Q, K, V, cos_cache, sin_cache, residual = super().prep_qkv_tensors(
            position_ids, hidden_states, past_key_value, **kwargs
        )
        # V is [B, H, S, D] after move_heads_front - apply V norm over last dim
        V = self.v_layernorm(V)
        # Compensate V norm for zero-padding: sliding layers have V padded from
        # 256→512, so the RMS (computed over all 512 dims including zeros) is
        # sqrt(2) too small, making the normalized output sqrt(2) too large.
        if self._v_norm_compensation != 1.0:
            V = V * self._v_norm_compensation

        # Cache K/V in model-level buffer for shared layers to read
        if self._kv_buffer is not None:
            self._kv_buffer[self.layer_idx] = (K, V)
        return Q, K, V, cos_cache, sin_cache, residual

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Override to handle partial RoPE (rope_dim < head_dim).

        Always recompute cos/sin because different layer types have different rope_dim,
        and the framework passes cos_cache between layers which would have wrong shape.

        Sliding layers: cos/sin are [B,S,256], apply partial RoPE on first 256 of 512 dims.
        Full layers: cos/sin are [B,S,512] (with cos=1/sin=0 for non-rotated dims),
          apply standard full-dim rotate_half which pairs dim i with dim i+256.

        Also pre-scales Q to cancel framework's 1/sqrt(head_dim) scaling since
        HF Gemma4 uses scaling=1.0 (Q-K norms keep attention scores reasonable).
        """
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            # Always recompute - each layer type has different rope_dim
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            if self.is_sliding:
                # Sliding: partial RoPE on first 256 of 512 dims
                Q, K = _partial_apply_rotary_pos_emb(Q, K, cos_cache, sin_cache, self.rope_dim)
            else:
                # Full: standard full-dim RoPE (512-dim cos/sin handle passthrough via cos=1/sin=0)
                Q, K = _full_apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        # Cancel framework's 1/sqrt(head_dim) division: pre-scale Q by sqrt(head_dim).
        # Net attention scaling: Q*sqrt(512)·K / sqrt(512) = Q·K (scaling=1.0, matching HF).
        Q = Q * self._q_scale_factor
        return Q, K, cos_cache, sin_cache


class NeuronGemma4MLP(nn.Module):
    """Gemma4 MLP with configurable intermediate_size (double-wide for KV-shared layers)."""

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Double-wide MLP for KV-shared layers
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared = layer_idx >= first_kv_shared > 0
        if config.use_double_wide_mlp and is_kv_shared:
            self.intermediate_size = config.intermediate_size * 2
        else:
            self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype, pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False,
            gather_output=False, dtype=config.neuron_config.torch_dtype, pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size, bias=False,
            input_is_parallel=True, dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), None


class NeuronGemma4MoEBlock(nn.Module):
    """Gemma4 Mixture of Experts block with custom router and NxDI ExpertMLPs.

    Architecture (from HF Gemma4):
    - Custom router: RMSNorm(x) * scale * (1/sqrt(H)) → proj → softmax → topk → normalize → per_expert_scale
    - Expert MLPs: 128 experts, each with gated MLP (gate_up → gelu → down), top-8 active per token
    - ExpertMLPs handles TP sharding of expert weights automatically
    """

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_experts = config.num_experts
        top_k = config.top_k_experts
        moe_intermediate = config.moe_intermediate_size

        # --- Gemma4 custom router ---
        # Router norm: RMSNorm without learnable weight (not in HF checkpoint)
        self.router_norm = Gemma4RMSNormNoScale(dim=hidden_size, eps=config.rms_norm_eps)
        # Router projection: [num_experts, hidden_size]
        self.router_proj_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        # Learnable per-dimension scale
        self.router_scale = nn.Parameter(torch.ones(hidden_size))
        # Per-expert scale applied after top-k normalization
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.scalar_root_size = hidden_size ** -0.5
        self.top_k = top_k
        self.num_experts = num_experts

        # --- Expert MLPs via NxDI (handles TP sharding automatically) ---
        self.expert_mlps = ExpertMLPs(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate,
            hidden_act='gelu',  # closest to gelu_pytorch_tanh in NxDI
            glu_mlp=True,
            capacity_factor=None,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            normalize_top_k_affinities=False,  # we handle normalization in our router
            is_prefill=config.neuron_config.is_prefill_stage,
        )

    def forward(self, router_input, expert_input, seq_len):
        """
        Args:
            router_input: [B*S, H] - residual hidden states for routing decisions
            expert_input: [B*S, H] - pre_feedforward_layernorm_2(residual) for expert computation
            seq_len: actual sequence length (for ExpertMLPs internal logic)
        Returns:
            output: [B*S, H] - weighted sum of expert outputs
        """
        T = router_input.shape[0]

        # --- Custom Gemma4 Router ---
        h = self.router_norm(router_input)
        h = h * self.router_scale * self.scalar_root_size
        logits = F.linear(h, self.router_proj_weight)  # [T, E]
        probs = F.softmax(logits.float(), dim=-1)  # fp32 for routing accuracy
        top_k_weights, top_k_idx = torch.topk(probs, self.top_k, dim=-1)  # [T, K]
        top_k_idx = top_k_idx.detach().to(dtype=torch.long)
        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # Apply per-expert scale
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_idx].float()
        top_k_weights = top_k_weights.to(router_input.dtype)

        # Create full expert_affinities [T, E] with routing weights at top-k positions
        expert_affinities = torch.zeros(T, self.num_experts, device=router_input.device, dtype=router_input.dtype)
        expert_affinities = expert_affinities.scatter(1, top_k_idx, top_k_weights)

        # --- Expert computation via NxDI ExpertMLPs ---
        output = self.expert_mlps(
            hidden_states=expert_input,
            expert_affinities=expert_affinities,
            expert_index=top_k_idx,
            seq_len=seq_len,
        )

        # All-reduce across TP ranks (ExpertMLPs computes partial results per rank)
        output = reduce_from_tensor_model_parallel_region(output)

        return output


class NeuronGemma4DecoderLayer(nn.Module):
    """
    Gemma4 decoder layer with PLE integration.

    Forward flow:
    1. input_layernorm -> self_attn -> post_attention_layernorm -> residual
    2. pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual
    3. PLE gate+projection -> post_per_layer_input_norm -> residual
    4. Multiply by layer_scalar
    """

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.ple_dim = config.hidden_size_per_layer_input

        # Attention
        self.self_attn = NeuronGemma4Attention(config, layer_idx=layer_idx)

        # MLP
        self.mlp = NeuronGemma4MLP(config, layer_idx=layer_idx)

        # Norms (5 total for Gemma4 with PLE)
        self.input_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

        # MoE block (only when enable_moe_block=True; 26B-A4B uses this)
        self.enable_moe_block = getattr(config, 'enable_moe_block', False)
        if self.enable_moe_block:
            self.moe_block = NeuronGemma4MoEBlock(config)
            # Additional norms for MoE layers
            self.post_feedforward_layernorm_1 = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

        # PLE components (only when hidden_size_per_layer_input > 0; 31B has no PLE)
        if self.ple_dim > 0:
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.ple_dim, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
            self.ple_act_fn = nn.GELU(approximate="tanh")

        # Per-layer scalar - must be nn.Parameter (not buffer) so it becomes a graph
        # input during XLA tracing. Buffers are baked as constants with their init
        # values; parameters are replaceable after tracing from checkpoint.
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        # === Attention block ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # === MLP block (+ optional MoE) ===
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]

        if self.enable_moe_block:
            # MoE block: router uses residual, experts use normalized residual
            B, S, H = residual.shape
            residual_flat = residual.reshape(-1, H)
            expert_input = self.pre_feedforward_layernorm_2(residual_flat)
            moe_output = self.moe_block(residual_flat, expert_input, S)
            moe_output = moe_output.reshape(B, S, H)

            # Combine: norm(MLP_output) + norm(MoE_output)
            hidden_states = self.post_feedforward_layernorm_1(hidden_states) + \
                            self.post_feedforward_layernorm_2(moe_output)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # === PLE block ===
        # per_layer_input is stored on this layer by NeuronGemma4Model.get_model_output
        per_layer_input = getattr(self, '_per_layer_input', None)
        if per_layer_input is not None:
            residual = hidden_states
            gate = self.per_layer_input_gate(hidden_states)  # [B, S, ple_dim]
            gate = self.ple_act_fn(gate)
            gated = gate * per_layer_input  # element-wise
            projected = self.per_layer_projection(gated)  # [B, S, hidden_size]
            projected = self.post_per_layer_input_norm(projected)
            hidden_states = residual + projected

        # === Per-layer scaling ===
        hidden_states = hidden_states * self.layer_scalar

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# Model
# ====================================================================================


class NeuronGemma4Model(NeuronBaseModel):
    """Gemma4 base model for NeuronX inference."""

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Replace token positions with pre-computed audio/vision embeddings.

        Uses index_put_ to scatter embeddings at positions specified by vision_mask.
        This matches the Llama4 scatter_by_index_put pattern used by NxDI.

        Args:
            inputs_embeds: [B, seq_len, hidden_size] - text token embeddings
            vision_embeddings: [B, seq_len, hidden_size] - audio embeddings packed sequentially
            vision_mask: [B, seq_len, 1] - int32 position indices for scatter
        """
        _, max_positions, embedding_dim = inputs_embeds.shape
        result = inputs_embeds.clone()
        flat_embeds = vision_embeddings.view(-1, embedding_dim)
        positions = vision_mask.view(-1)
        num_positions = len(positions)
        flat_embeds = flat_embeds[:num_positions]
        result.view(-1, embedding_dim).index_put_((positions,), flat_embeds, accumulate=False)
        return result

    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def get_model_output(self, input_ids=None, inputs_embeds=None, **kwargs):
        """Override to compute PLE and pass per-layer inputs to decoder layers.

        The framework's get_model_output never passes per_layer_input to layers.
        We compute PLE here and store on each layer before calling super().
        Also clears the KV sharing buffer before each forward pass.
        """
        # Clear KV buffer for this forward pass
        self._kv_buffer.clear()

        # Compute embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute PLE if enabled (ple_dim > 0; 31B has no PLE)
        if self.ple_dim > 0:
            per_layer_inputs = self.ple(input_ids, inputs_embeds)
            for idx, layer in enumerate(self.layers):
                layer._per_layer_input = per_layer_inputs[:, :, idx, :]

        # Call parent with inputs_embeds pre-computed to avoid double embedding
        return super().get_model_output(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
        )

    def init_model(self, config: Gemma4InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.ple_dim = config.hidden_size_per_layer_input

        # Main token embeddings (scaled)
        self.embed_tokens = Gemma4ScaledEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True, pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # PLE module (only when hidden_size_per_layer_input > 0; 31B has no PLE)
        if self.ple_dim > 0:
            self.ple = Gemma4PerLayerEmbedding(config)

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # KV sharing buffer: a plain dict (not nn.Module) shared across attention layers.
        # Source layers write (K, V) here; shared layers read from their source index.
        self._kv_buffer = {}
        for layer in self.layers:
            layer.self_attn._kv_buffer = self._kv_buffer

        # Final norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (with final logit softcapping if configured)
        lm_head_linear = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None and softcap > 0:
            self.lm_head = SoftcappedLMHead(lm_head_linear, softcap)
        else:
            self.lm_head = lm_head_linear


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """Top-level Gemma4 causal LM for NeuronX compilation and inference."""

    _model_cls = NeuronGemma4Model
    _STATE_DICT_MODEL_PREFIX = "model.language_model."

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        # Gemma4 is not in transformers 4.57, so we don't load via AutoModel.
        # The framework's get_state_dict will call load_state_dict(model_path)
        # directly from safetensors, bypassing this method.
        raise NotImplementedError(
            "Gemma4 not supported by installed transformers. "
            "Weights are loaded directly from safetensors."
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Gemma4InferenceConfig) -> dict:
        """
        Convert HF Gemma4 state dict to NxDI format.

        Key transformations:
        - Embed tokens: embed_tokens.weight -> embed_tokens.embedding.weight
        - Q-K norm: q_norm -> q_layernorm, k_norm -> k_layernorm
        - Sliding layer attention weights: zero-pad from head_dim=256 to 512
        - Q-K norm weights: compensate for sqrt(2) RMS difference from zero-padding
        - PLE: map embed_tokens_per_layer and per_layer_model_projection
        - Rank util tensors for tensor parallelism
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        neuron_sd = {}

        global_head_dim = config.global_head_dim  # 512
        sliding_head_dim = 256  # original sliding attention head_dim
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        # Full attention layers may have fewer KV heads (31B: 4 vs 16 sliding)
        num_global_kv_heads = getattr(config, 'num_global_key_value_heads', None) or num_kv_heads
        attention_k_eq_v = getattr(config, 'attention_k_eq_v', False)
        num_layers = config.num_hidden_layers
        layer_types = config.layer_types
        ple_dim = config.hidden_size_per_layer_input
        enable_moe = getattr(config, 'enable_moe_block', False)

        # === Embeddings ===
        if "embed_tokens.weight" in state_dict:
            neuron_sd["embed_tokens.embedding.weight"] = state_dict["embed_tokens.weight"].detach().clone()

        # === PLE embeddings (only when ple_dim > 0) ===
        if ple_dim > 0:
            total_ple_dim = num_layers * ple_dim
            if "embed_tokens_per_layer.weight" in state_dict:
                w = state_dict["embed_tokens_per_layer.weight"].detach().clone()
                neuron_sd["ple.embed_tokens_per_layer.weight"] = w[:, :total_ple_dim]
            if "per_layer_model_projection.weight" in state_dict:
                w = state_dict["per_layer_model_projection.weight"].detach().clone()
                neuron_sd["ple.per_layer_model_projection.weight"] = w[:total_ple_dim, :]
            if "per_layer_projection_norm.weight" in state_dict:
                neuron_sd["ple.per_layer_projection_norm.weight"] = (
                    state_dict["per_layer_projection_norm.weight"].detach().clone()
                )

        # === Final norm ===
        if "norm.weight" in state_dict:
            neuron_sd["norm.weight"] = state_dict["norm.weight"].detach().clone()

        # === LM head (tied to embeddings) ===
        if "lm_head.weight" in state_dict:
            neuron_sd["lm_head.weight"] = state_dict["lm_head.weight"].detach().clone()

        # === Decoder layers ===
        for i in range(num_layers):
            prefix = f"layers.{i}"
            is_sliding = layer_types[i] == SLIDING_ATTENTION
            orig_head_dim = sliding_head_dim if is_sliding else global_head_dim
            # Full attention layers may use fewer KV heads (31B: 4 global vs 16 sliding)
            layer_kv_heads = num_kv_heads if is_sliding else num_global_kv_heads
            # Whether this full layer uses k_eq_v (V=K, no v_proj)
            layer_k_eq_v = attention_k_eq_v and not is_sliding

            # --- Attention Q/K/V/O weights ---
            q_key = f"{prefix}.self_attn.q_proj.weight"
            k_key = f"{prefix}.self_attn.k_proj.weight"
            v_key = f"{prefix}.self_attn.v_proj.weight"
            o_key = f"{prefix}.self_attn.o_proj.weight"

            if q_key in state_dict:
                q_w = state_dict[q_key].detach().clone()
                if is_sliding and orig_head_dim < global_head_dim:
                    q_w = _zero_pad_qo_weight(q_w, num_heads, orig_head_dim, global_head_dim)
                neuron_sd[q_key] = q_w

            if k_key in state_dict:
                k_w = state_dict[k_key].detach().clone()
                if is_sliding and orig_head_dim < global_head_dim:
                    k_w = _zero_pad_kv_weight(k_w, layer_kv_heads, orig_head_dim, global_head_dim)
                # Replicate KV heads for full layers if they have fewer heads
                if not is_sliding and layer_kv_heads < num_kv_heads:
                    k_w = _replicate_kv_heads(k_w, layer_kv_heads, num_kv_heads, global_head_dim)
                neuron_sd[k_key] = k_w

            if v_key in state_dict:
                v_w = state_dict[v_key].detach().clone()
                if is_sliding and orig_head_dim < global_head_dim:
                    v_w = _zero_pad_kv_weight(v_w, layer_kv_heads, orig_head_dim, global_head_dim)
                if not is_sliding and layer_kv_heads < num_kv_heads:
                    v_w = _replicate_kv_heads(v_w, layer_kv_heads, num_kv_heads, global_head_dim)
                neuron_sd[v_key] = v_w
            elif layer_k_eq_v and k_key in state_dict:
                # k_eq_v: no v_proj in checkpoint, create from k_proj (V=K before norms)
                neuron_sd[v_key] = neuron_sd[k_key].clone()

            if o_key in state_dict:
                o_w = state_dict[o_key].detach().clone()
                if is_sliding and orig_head_dim < global_head_dim:
                    o_w = _zero_pad_o_weight(o_w, num_heads, orig_head_dim, global_head_dim)
                neuron_sd[o_key] = o_w

            # --- Q-K norms (rename q_norm -> q_layernorm, k_norm -> k_layernorm) ---
            for src_name, dst_name in [("q_norm", "q_layernorm"), ("k_norm", "k_layernorm")]:
                src_key = f"{prefix}.self_attn.{src_name}.weight"
                dst_key = f"{prefix}.self_attn.{dst_name}.weight"
                if src_key in state_dict:
                    w = state_dict[src_key].detach().clone()
                    if is_sliding and orig_head_dim < global_head_dim:
                        w = _compensate_pad_norm_weight(w, orig_head_dim, global_head_dim)
                    neuron_sd[dst_key] = w

            # --- MLP ---
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    neuron_sd[key] = state_dict[key].detach().clone()

            # --- Layer norms ---
            norm_names = [
                "input_layernorm", "post_attention_layernorm",
                "pre_feedforward_layernorm", "post_feedforward_layernorm",
                "post_per_layer_input_norm",
            ]
            # MoE layers have additional norms
            if enable_moe:
                norm_names += [
                    "post_feedforward_layernorm_1",
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_2",
                ]
            for norm_name in norm_names:
                key = f"{prefix}.{norm_name}.weight"
                if key in state_dict:
                    neuron_sd[key] = state_dict[key].detach().clone()

            # --- PLE per-layer components ---
            for ple_name in ["per_layer_input_gate", "per_layer_projection"]:
                key = f"{prefix}.{ple_name}.weight"
                if key in state_dict:
                    neuron_sd[key] = state_dict[key].detach().clone()

            # --- Layer scalar ---
            scalar_key = f"{prefix}.layer_scalar"
            if scalar_key in state_dict:
                neuron_sd[scalar_key] = state_dict[scalar_key].detach().clone()

            # --- MoE weights ---
            if enable_moe:
                # Router weights
                router_proj_key = f"{prefix}.router.proj.weight"
                if router_proj_key in state_dict:
                    neuron_sd[f"{prefix}.moe_block.router_proj_weight"] = (
                        state_dict[router_proj_key].detach().clone()
                    )
                router_scale_key = f"{prefix}.router.scale"
                if router_scale_key in state_dict:
                    neuron_sd[f"{prefix}.moe_block.router_scale"] = (
                        state_dict[router_scale_key].detach().clone()
                    )
                per_expert_scale_key = f"{prefix}.router.per_expert_scale"
                if per_expert_scale_key in state_dict:
                    neuron_sd[f"{prefix}.moe_block.per_expert_scale"] = (
                        state_dict[per_expert_scale_key].detach().clone()
                    )

                # Expert weights: transpose from HF format to NxDI ExpertMLPs format
                # HF: gate_up_proj [E, 2*I, H] → NxDI: [E, H, 2*I]
                gate_up_key = f"{prefix}.experts.gate_up_proj"
                if gate_up_key in state_dict:
                    w = state_dict[gate_up_key].detach().clone()
                    neuron_sd[f"{prefix}.moe_block.expert_mlps.mlp_op.gate_up_proj.weight"] = (
                        w.transpose(1, 2).contiguous()
                    )
                # HF: down_proj [E, H, I] → NxDI: [E, I, H]
                down_key = f"{prefix}.experts.down_proj"
                if down_key in state_dict:
                    w = state_dict[down_key].detach().clone()
                    neuron_sd[f"{prefix}.moe_block.expert_mlps.mlp_op.down_proj.weight"] = (
                        w.transpose(1, 2).contiguous()
                    )

            # --- Rank util for attention TP ---
            neuron_sd[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # === Global rank util ===
        neuron_sd["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if neuron_config.vocab_parallel:
            neuron_sd["embed_tokens.embedding.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        return neuron_sd

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Tie embed_tokens -> lm_head.

        Handles both direct lm_head (ColumnParallelLinear) and wrapped
        lm_head (SoftcappedLMHead wrapping ColumnParallelLinear).
        """
        if "embed_tokens.embedding.weight" in state_dict:
            w = state_dict["embed_tokens.embedding.weight"].clone()
            # Direct linear: lm_head.weight
            state_dict["lm_head.weight"] = w
            # Wrapped (SoftcappedLMHead): lm_head.linear.weight
            state_dict["lm_head.linear.weight"] = w

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig


# ====================================================================================
# Weight padding helpers for sliding -> global head_dim conversion
# ====================================================================================


def _zero_pad_qo_weight(w, num_heads, orig_hd, target_hd):
    """
    Zero-pad Q or O projection weight for multi-head: [num_heads*orig_hd, hidden] -> [num_heads*target_hd, hidden]
    Pads each head's slice independently to maintain head structure.
    """
    hidden = w.shape[1]
    # Reshape to [num_heads, orig_hd, hidden]
    w = w.view(num_heads, orig_hd, hidden)
    # Pad along head_dim: [num_heads, orig_hd, hidden] -> [num_heads, target_hd, hidden]
    padded = torch.zeros(num_heads, target_hd, hidden, dtype=w.dtype, device=w.device)
    padded[:, :orig_hd, :] = w
    return padded.view(num_heads * target_hd, hidden)


def _zero_pad_kv_weight(w, num_kv_heads, orig_hd, target_hd):
    """Zero-pad K or V projection: [num_kv*orig_hd, hidden] -> [num_kv*target_hd, hidden]."""
    hidden = w.shape[1]
    w = w.view(num_kv_heads, orig_hd, hidden)
    padded = torch.zeros(num_kv_heads, target_hd, hidden, dtype=w.dtype, device=w.device)
    padded[:, :orig_hd, :] = w
    return padded.view(num_kv_heads * target_hd, hidden)


def _zero_pad_o_weight(w, num_heads, orig_hd, target_hd):
    """Zero-pad O projection: [hidden, num_heads*orig_hd] -> [hidden, num_heads*target_hd]."""
    hidden = w.shape[0]
    w = w.view(hidden, num_heads, orig_hd)
    padded = torch.zeros(hidden, num_heads, target_hd, dtype=w.dtype, device=w.device)
    padded[:, :, :orig_hd] = w
    return padded.view(hidden, num_heads * target_hd)


def _replicate_kv_heads(w, orig_kv_heads, target_kv_heads, head_dim):
    """Replicate KV heads: [orig_kv*hd, hidden] -> [target_kv*hd, hidden].

    Each original head is repeated target_kv/orig_kv times. This is used when
    full attention layers have fewer KV heads (e.g., 4) than sliding layers (e.g., 16).
    Replication preserves GQA semantics since grouped Q heads all see the same K/V.
    """
    assert target_kv_heads % orig_kv_heads == 0
    repeats = target_kv_heads // orig_kv_heads
    hidden = w.shape[1]
    w = w.view(orig_kv_heads, head_dim, hidden)
    w = w.repeat_interleave(repeats, dim=0)  # [target_kv, hd, hidden]
    return w.view(target_kv_heads * head_dim, hidden)


def _compensate_pad_norm_weight(w, orig_dim, target_dim):
    """
    Zero-pad RMSNorm weight and compensate for RMS difference.

    When zero-padding input from orig_dim to target_dim, the RMS changes by sqrt(orig/target).
    The norm output scales by 1/RMS, so it becomes sqrt(target/orig) times larger.
    To compensate: new_weight = old_weight * sqrt(orig/target)
    Padded dimensions get weight=0 (zeroed out by norm).

    For orig=256, target=512: factor = sqrt(256/512) = 1/sqrt(2)
    """
    factor = math.sqrt(orig_dim / target_dim)
    compensated = w.float() * factor
    padded = torch.zeros(target_dim, dtype=w.dtype, device=w.device)
    padded[:orig_dim] = compensated.to(w.dtype)
    return padded


# ====================================================================================
# Multimodal (Audio/Vision) Conditional Generation
# ====================================================================================


class _MultimodalConfigWrapper:
    """Wraps a Gemma4InferenceConfig to expose text_config/vision_config
    as required by NeuronBaseForImageToText."""

    def __init__(self, text_config: Gemma4InferenceConfig):
        self._text_config = text_config
        # Proxy all attribute access to text_config by default
        # text_config and vision_config are the special ones

    @property
    def text_config(self):
        return self._text_config

    @property
    def vision_config(self):
        return None  # No vision encoder on Neuron

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._text_config, name)

    def save(self, path):
        return self._text_config.save(path)


class NeuronGemma4ForConditionalGeneration(NeuronBaseForImageToText):
    """
    Gemma4 multimodal model for NeuronX.

    Text decoder runs on Neuron (compiled with ImageToTextModelWrapper input signature
    which includes vision_embeddings/vision_mask slots).

    Audio/vision encoder runs on CPU (not compiled for Neuron).
    Pre-computed audio embeddings are passed through vision_embeddings/vision_mask
    to the Neuron text decoder.
    """

    # NeuronBaseForImageToText requires these class attributes
    text_model_cls = NeuronGemma4Model
    _model_cls = NeuronGemma4Model
    text_model_wrapper = ImageToTextModelWrapper
    # We don't use a Neuron-compiled vision model (audio encoder runs on CPU)
    vision_model_cls = None
    vision_model_wrapper = None

    _STATE_DICT_MODEL_PREFIX = "model.language_model."

    def __init__(self, model_path, config):
        # Wrap config to expose text_config/vision_config interface
        wrapped_config = _MultimodalConfigWrapper(config)
        super().__init__(
            text_model_cls=NeuronGemma4Model,
            vision_model_cls=None,
            text_model_wrapper=ImageToTextModelWrapper,
            vision_model_wrapper=None,
            model_path=model_path,
            config=wrapped_config,
        )

    def enable_vision_encoder(self, **kwargs):
        """No-op: audio encoder runs on CPU, not compiled for Neuron."""
        self.vision_models = []

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        vision_embeddings,
        vision_mask,
        deepstack_vision_embeds,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        rotary_position_ids=None,
    ):
        """Override to pass empty vision tensors during token generation.

        The compiled TKG model expects vision_embeddings/vision_mask as torch.empty(0),
        while the base class passes them unchanged for both CTE and TKG.
        """
        if rotary_position_ids is None:
            rotary_position_ids = torch.empty(0)

        # For token generation, use empty vision tensors (audio only in context encoding)
        empty_ve = torch.zeros(0, dtype=self.neuron_config.torch_dtype)
        empty_vm = torch.zeros(0, dtype=torch.bool)

        # For context encoding, provide default vision tensors if None
        # Use S-1 as padding position in mask so index_put_ overwrites a harmless
        # position instead of BOS (position 0).
        if vision_embeddings is None:
            max_ctx = self.neuron_config.max_context_length
            cte_ve = torch.zeros(
                input_ids.shape[0], max_ctx,
                self.text_config.hidden_size, dtype=self.neuron_config.torch_dtype)
            cte_vm = torch.full(
                (input_ids.shape[0], max_ctx, 1),
                max_ctx - 1, dtype=torch.int32)
        else:
            cte_ve = vision_embeddings
            cte_vm = vision_mask

        from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG

        is_prefill = self._is_prefill(position_ids)

        if is_prefill:
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping
                torch.empty(0),  # active_block_table
                torch.empty(0),  # num_queries
                torch.empty(0),  # computed_context_lens
                torch.empty(0),  # tile_q_indices
                torch.empty(0),  # tile_block_tables
                torch.empty(0),  # tile_masks
                torch.empty(0),  # inputs_embeds
                torch.empty(0),  # kv_cache
                torch.empty(0),  # active_mask
                rotary_position_ids,
                cte_ve,
                cte_vm,
                None,  # deepstack_vision_embeds (filtered by wrapper)
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping
                torch.empty(0),  # active_block_table
                torch.empty(0),  # num_queries
                torch.empty(0),  # computed_context_lens
                torch.empty(0),  # tile_q_indices
                torch.empty(0),  # tile_block_tables
                torch.empty(0),  # tile_masks
                torch.empty(0),  # inputs_embeds
                torch.empty(0),  # kv_cache
                torch.empty(0),  # active_mask
                rotary_position_ids,
                empty_ve,   # no vision during token generation
                empty_vm,   # no vision during token generation
                None,  # deepstack_vision_embeds (filtered by wrapper)
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def _save_configs_to_compiler_workdir(self):
        """Override to skip vision_config.save() since audio encoder runs on CPU."""
        import os
        import copy
        from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self.config.save(base_compile_work_dir)

        text_model_compile_work_dir = os.path.join(base_compile_work_dir, "text_model")
        self.config.text_config.save(text_model_compile_work_dir)
        # Skip vision_config.save() — no vision model on Neuron

        for submodel in self.text_models:
            for bucket_rank, bucket_size in enumerate(submodel.config.neuron_config.buckets):
                specific_config = copy.deepcopy(submodel.config)
                specific_config.neuron_config.buckets = [bucket_size]
                if submodel.tag == CONTEXT_ENCODING_MODEL_TAG:
                    specific_config.neuron_config.context_encoding_buckets = specific_config.neuron_config.buckets
                else:
                    specific_config.neuron_config.token_generation_buckets = specific_config.neuron_config.buckets
                submodel_path = os.path.join(text_model_compile_work_dir, submodel.tag, f"_tp0_bk{bucket_rank}")
                specific_config.save(submodel_path)

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None, dry_run=False):
        """Compile text model only (no vision/audio encoder on Neuron)."""
        import os
        from neuronx_distributed_inference.models.application_base import (
            normalize_path, COMPILED_MODEL_FILE_NAME,
        )

        self.config.save(compiled_model_path)
        text_path = normalize_path(compiled_model_path) + "text_model/"
        os.makedirs(text_path, exist_ok=True)
        # Also create empty vision dir so load() doesn't fail
        vision_path = normalize_path(compiled_model_path) + "vision_model/"
        os.makedirs(vision_path, exist_ok=True)

        text_traced = self.get_text_builder(debug).trace(initialize_model_weights=False, dry_run=dry_run)
        if not dry_run:
            torch.jit.save(text_traced, text_path + COMPILED_MODEL_FILE_NAME)
            del text_traced

        self._save_configs_to_compiler_workdir()
        if dry_run:
            return

        self.shard_text_weights(text_path, debug, pre_shard_weights_hook)
        self.is_compiled = True

    def load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False):
        """Load text model only (no vision model on Neuron)."""
        import os, time, logging
        from neuronx_distributed_inference.models.application_base import (
            normalize_path, COMPILED_MODEL_FILE_NAME,
        )
        from safetensors.torch import load_file

        compiled_model_path = normalize_path(compiled_model_path)
        text_path = os.path.join(compiled_model_path, "text_model/")

        self.text_traced_model = torch.jit.load(os.path.join(text_path, COMPILED_MODEL_FILE_NAME))

        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        # Load text weights
        text_weights = []
        t0 = time.monotonic()
        if self.neuron_config.save_sharded_checkpoint:
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(os.path.join(text_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
                text_weights.append(ckpt)
        else:
            text_weights = self.get_text_builder().shard_checkpoint()

        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.text_traced_model.nxd_model.initialize(text_weights, start_rank_tensor)
        logging.info(f"Finished text weights loading in {time.monotonic() - t0:.1f}s")

        for model_wrapper in self.text_models:
            model_wrapper.model = self.text_traced_model

        self.is_loaded_to_neuron = True
        if not self.neuron_config.skip_warmup and not skip_warmup:
            self.warmup()

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        raise NotImplementedError("Gemma4 not in installed transformers.")

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        return NeuronGemma4ForCausalLM.convert_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        return NeuronGemma4ForCausalLM.update_state_dict_for_tied_weights(state_dict)

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig
