# coding=utf-8
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==========================================================================
# NeuronX Distributed Inference port of google/gemma-4-26B-A4B-it.
# ==========================================================================
#
# Round 2 — heavily borrowed from Jim Burtoft's PR #106 (gemma-4-31B-IT) for
# attention, KV cache, softcapping, and weight conversion. The MoE block,
# router, and decoder-layer MoE branch are 26B-A4B-specific (the 31B model
# is dense). NKI flash attention kernels (`nki_flash_attn_d256_swa.py` and
# `nki_flash_attn_large_d.py`) are taken verbatim from PR #106; head_dim
# values match (256 sliding / 512 full) so they work as-is.
#
# Differences from PR #106 worth knowing about:
#   * 26B-A4B has `enable_moe_block=True`, `num_experts=128`, `top_k=8`.
#     Each decoder layer runs the dense MLP and the MoE block in parallel
#     (HF source lines 1429-1441), then sums their outputs.
#   * `hidden_size=2816` (vs 31B's 5376), `num_attention_heads=16`,
#     `num_key_value_heads=8` for sliding, `num_global_key_value_heads=2`.
#   * `final_logit_softcapping=30.0` (same as 31B-IT).
#   * `hidden_size_per_layer_input=0` — no per-layer-embedding (PLE) on
#     26B-A4B, so the round-1 PLE code is dropped.
#
# To use the NKI kernels at runtime, callers must invoke
# `import ndxi_patch; ndxi_patch.apply_patch()` before
# constructing `NeuronGemma4ForCausalLM`.

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import mappings as _nxd_parallel_mappings

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
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

# MoE module from NxDI v2.
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

# NKI flash attention kernel for head_dim=256 SWA layers (sliding window).
try:
    try:
        from .nki_flash_attn_d256_swa import flash_attn_d256_swa as _nki_flash_attn_d256_swa  # type: ignore[import-not-found]
    except ImportError:
        # Fallback for top-level (PYTHONPATH=src) layout.
        from nki_flash_attn_d256_swa import flash_attn_d256_swa as _nki_flash_attn_d256_swa  # type: ignore[import-not-found]

    _HAS_NKI_SWA_KERNEL = True
except Exception:  # pragma: no cover - kernel optional at import time
    _HAS_NKI_SWA_KERNEL = False


# ====================================================================================
# Normalization (PR #106 pattern: Gemma4RMSNorm with weight, Gemma4VNorm without)
# ====================================================================================


class Gemma4RMSNorm(nn.Module):
    """Standard Gemma4 RMSNorm: normed * weight (weight init to ones)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)


class Gemma4VNorm(nn.Module):
    """Gemma4 v_norm: RMSNorm without learnable scale (with_scale=False)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.type_as(x)


def get_rmsnorm_cls():
    """Single source of truth for the outer norm class (matches PR #106)."""
    return Gemma4RMSNorm


# ====================================================================================
# Embeddings + softcapped LM head (verbatim from PR #106)
# ====================================================================================


class SoftcappedLMHead(nn.Module):
    """Wrap lm_head and apply final_logit_softcapping: cap * tanh(x / cap)."""

    def __init__(self, linear: nn.Module, cap: float):
        super().__init__()
        self.linear = linear
        self.cap = cap

    def forward(self, x):
        logits = self.linear(x)
        logits = logits.float()
        return self.cap * torch.tanh(logits / self.cap)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)


class Gemma4ScaledEmbedding(nn.Module):
    """Token embedding with sqrt(hidden_size) scaling (per gemma4 source)."""

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
# Configuration (PR #106 pattern, extended for MoE attributes)
# ====================================================================================


class Gemma4NeuronConfig(MoENeuronConfig):
    """NeuronConfig hard-pinning the gemma4 attention class.

    Extends `MoENeuronConfig` (not plain `NeuronConfig`) so the MoE-specific
    attributes that `initialize_moe_module` reads off `config.neuron_config`
    -- `router_config`, `blockwise_matmul_config`, `moe_ep_degree`,
    `moe_tp_degree`, `glu_mlp`, `glu_type`, `normalize_top_k_affinities`,
    `early_expert_affinity_modulation`, `is_prefill_stage`, etc. -- exist
    even on dense smoke-compile runs (default values are harmless when MoE
    is disabled).
    """

    def __init__(self, **kwargs):
        # Gemma-4-26B-A4B keeps the routed top-k weights AS GIVEN by the router
        # (the router itself already renormalizes to sum=1 and applies the
        # per-expert scale). We do NOT want NxDI to renormalize again, so set
        # the disable flag (NeuronGemma4MoEBlock pre-builds the (T,E)
        # expert_affinities tensor with zeros outside top-k).
        kwargs.setdefault("disable_normalize_top_k_affinities", True)
        super().__init__(**kwargs)
        # attn_cls is set per-layer in get_updated_configs(); this default is
        # used for the framework-level introspection only.
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    """Inference config that pulls fields from HF gemma4 config.json (text_config)."""

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

        # Gemma4 nests text params under text_config.
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
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
                # MoE-specific (26B-A4B)
                "enable_moe_block",
                "num_experts",
                "top_k_experts",
                "moe_intermediate_size",
                # PLE / KV-share (0 on 26B-A4B but kept for forward compat)
                "hidden_size_per_layer_input",
                "vocab_size_per_layer_input",
                "num_kv_shared_layers",
                "use_double_wide_mlp",
                "tie_word_embeddings",
                "pad_token_id",
            ]
            for attr in text_attrs:
                if isinstance(text_config, dict):
                    if attr in text_config:
                        setattr(self, attr, text_config[attr])
                elif hasattr(text_config, attr):
                    setattr(self, attr, getattr(text_config, attr))

        # PretrainedConfig defaults that SimpleNamespace conversion drops.
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for attr, default in [
                ("output_attentions", False),
                ("output_hidden_states", False),
                ("use_return_dict", True),
            ]:
                if not hasattr(text_config, attr):
                    setattr(text_config, attr, default)
        for attr, default in [
            ("output_attentions", False),
            ("output_hidden_states", False),
            ("use_return_dict", True),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)

        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = 0
        if not hasattr(self, "tie_word_embeddings"):
            self.tie_word_embeddings = True
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

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
    """Per-layer configs for heterogeneous SWA/global layers (PR #106 pattern)."""
    updated_configs = []
    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)
        layer_type = config.layer_types[i]

        # MoE config aliases that NxDI's `initialize_moe_module` reads off the
        # layer config. We do NOT overwrite `intermediate_size` here (the
        # dense MLP needs it); the MoE block uses `_moe_config` (set later)
        # which has the moe_intermediate_size.
        if getattr(layer_config, "enable_moe_block", False):
            layer_config.num_local_experts = layer_config.num_experts
            layer_config.num_experts_per_tok = layer_config.top_k_experts

        if layer_type == "sliding_attention":
            layer_config.sliding_window = config.sliding_window
            layer_config._layer_head_dim = config.head_dim
            layer_config._layer_num_kv_heads = config.num_key_value_heads
            layer_config._layer_is_sliding = True
            layer_config._layer_k_eq_v = False
            rope_params = config.rope_parameters.get("sliding_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 10000.0)
            layer_config._layer_partial_rotary_factor = 1.0
        else:
            layer_config.sliding_window = None
            layer_config._layer_head_dim = config.global_head_dim
            layer_config._layer_num_kv_heads = config.num_global_key_value_heads
            layer_config._layer_is_sliding = False
            layer_config._layer_k_eq_v = getattr(config, "attention_k_eq_v", False)
            rope_params = config.rope_parameters.get("full_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 1000000.0)
            layer_config._layer_partial_rotary_factor = rope_params.get(
                "partial_rotary_factor", 0.25
            )

        updated_configs.append(layer_config)
    return updated_configs


# ====================================================================================
# Attention (PR #106 verbatim, head_dim values match for 26B-A4B)
# ====================================================================================


class NeuronGemma4Attention(NeuronAttentionBase):
    """Gemma4 attention with per-layer head_dim/kv_heads, partial RoPE, v_norm.

    Borrowed wholesale from PR #106 (Jim Burtoft, gemma-4-31B-IT). 26B-A4B
    shares all the head dimensions (256 sliding / 512 global) so this works
    unchanged. The only per-config differences (kv head counts, rope theta)
    are read from the per-layer config dict produced by `get_updated_configs`.
    """

    def __init__(self, config: Gemma4InferenceConfig):
        head_dim = config._layer_head_dim
        num_kv_heads = config._layer_num_kv_heads
        is_sliding = config._layer_is_sliding
        rope_theta = config._layer_rope_theta
        partial_rotary_factor = config._layer_partial_rotary_factor

        # Partial RoPE: rotate first head_dim*factor dims, leave the rest alone.
        rotary_dim = int(head_dim * partial_rotary_factor)
        rotary_dim = rotary_dim - (rotary_dim % 2)

        rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # PR #106 Discovery #27: pass sliding_window=None to base for ALL layers
        # to avoid OOB in get_last_kv_window when bucket_size < sliding_window.
        # Windowed masking is applied at the decoder-layer level via local_mask.
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

        # QK norms: gemma4 RMSNorm with learned weight (initialized to 1).
        self.q_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)

        # V norm: RMSNorm without learnable scale.
        self.v_norm = Gemma4VNorm(dim=head_dim, eps=config.rms_norm_eps)

        self._is_sliding = is_sliding
        self._k_eq_v = config._layer_k_eq_v
        self._head_dim = head_dim
        self._rotary_dim = rotary_dim
        self._partial_rotary_factor = partial_rotary_factor

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        if self.rotary_emb is None:
            return Q, K, cos_cache, sin_cache
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        if self._rotary_dim == self._head_dim:
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        else:
            q_rot = Q[..., : self._rotary_dim]
            q_pass = Q[..., self._rotary_dim :]
            k_rot = K[..., : self._rotary_dim]
            k_pass = K[..., self._rotary_dim :]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_cache, sin_cache)
            Q = torch.cat([q_rot, q_pass], dim=-1)
            K = torch.cat([k_rot, k_pass], dim=-1)
        return Q, K, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
        """Use NKI d=256 SWA kernel for sliding layers when available."""
        if (
            _HAS_NKI_SWA_KERNEL
            and self._is_sliding
            and self._head_dim == 256
            and q_len >= 128
        ):
            q_kernel = Q.to(self.torch_dtype)
            k_kernel = K.to(self.torch_dtype)
            v_kernel = V.to(self.torch_dtype)

            n_kv_heads = K.shape[1]
            n_q_heads = Q.shape[1]
            q_h_per_kv = n_q_heads // n_kv_heads
            window_size = 1024

            out_parts = []
            for b in range(bsz):
                for kv_h in range(n_kv_heads):
                    q_slice = q_kernel[
                        b : b + 1, kv_h * q_h_per_kv : (kv_h + 1) * q_h_per_kv, :, :
                    ]
                    k_slice = k_kernel[b : b + 1, kv_h : kv_h + 1, :, :]
                    v_slice = v_kernel[b : b + 1, kv_h : kv_h + 1, :, :]
                    o_part = _nki_flash_attn_d256_swa(
                        q_slice,
                        k_slice,
                        v_slice,
                        q_h_per_k_h=q_h_per_kv,
                        n_kv_heads=1,
                        seqlen_q=q_len,
                        seqlen_kv=q_len,
                        window_size=window_size,
                    )
                    out_parts.append(o_part)
            attn_output = torch.cat(out_parts, dim=1)
            if bsz > 1:
                attn_output = attn_output.reshape(bsz, n_q_heads, q_len, self._head_dim)
            return attn_output, FlashAttentionStrategy.NONE

        return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

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
        # Apply v_norm on BHSD-laid V (last dim == head_dim).
        V = self.v_norm(V)
        return Q, K, V, cos_cache, sin_cache, residual


# ====================================================================================
# MLP (dense feed-forward)
# ====================================================================================


class NeuronGemma4MLP(nn.Module):
    """Dense SwiGLU MLP with `gelu_pytorch_tanh` activation."""

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        dtype = config.neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), None


# ====================================================================================
# Router + MoE block (26B-A4B specific — PR #106 has no MoE)
# ====================================================================================


class NeuronGemma4Router(nn.Module):
    """Top-K router with gemma4's `scale` and `per_expert_scale` learned tensors.

    Mirrors HF `Gemma4TextRouter` (modeling_gemma4.py:1334) and exposes the
    contract that NxDI's `MoE` wrapper expects from its `router` member:

        forward(hidden_states) -> (router_logits, expert_affinities, expert_index)

    where
        router_logits      : (T, E)   raw projection (used for aux losses; we
                                      return it for compatibility, never used
                                      at inference)
        expert_affinities  : (T, E)   sparse tensor with the FINAL post-softmax
                                      / post-renormalize / post-per-expert-scale
                                      weights at top-k indices, zero elsewhere.
                                      With `normalize_top_k_affinities=False`
                                      on the MoE config, NxDI's expert dispatch
                                      uses these values directly.
        expert_index       : (T, K)   top-k expert indices.

    Routing math is FP32 for numerical stability across 128 experts.

    Replicated across TP (no parallel layer): every rank computes the same
    routing decisions, so no all-reduce is needed for the indices.
    """

    REQUIRED_ATTRS = ("num_experts", "top_k", "hidden_size", "sequence_parallel_enabled", "sequence_dimension")

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size ** -0.5
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts

        # NxDI's `MoE.__init__` cross-checks (`router.num_experts`,
        # `router.top_k`, `router.hidden_size`) against the expert_mlps
        # config -- the names above already match.

        # NxDI also reads `router.sequence_parallel_enabled` and
        # `router.sequence_dimension` to decide whether to gather
        # hidden_states before / after routing. We replicate the router
        # weights across TP, so SP-on-router is False.
        self.sequence_parallel_enabled = False
        self.sequence_dimension = 1

        # No-scale RMSNorm (gemma4 source line 1342: with_scale=False).
        self.norm = Gemma4VNorm(self.hidden_size, eps=self.eps)
        # Replicated across TP — every rank must reach the same routing.
        self.proj = nn.Linear(
            self.hidden_size, self.num_experts, bias=False, dtype=torch.float32
        )
        # Learned scale parameters (gemma4 source 1344-1345).
        self.scale = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float32))
        self.per_expert_scale = nn.Parameter(
            torch.ones(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor):
        # Accept either (B, S, H) or (T, H). NxDI's MoE wrapper passes the
        # full / SP-gathered hidden states; we flatten to (T, H) for routing.
        original_shape = hidden_states.shape
        h = hidden_states.float().reshape(-1, original_shape[-1])  # (T, H)

        h = self.norm(h)
        h = h * self.scale * self.scalar_root_size

        router_logits = self.proj(h)  # (T, E) FP32
        probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        # Per-token re-normalisation (gemma4 source line 1362).
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # Apply per-expert scale (gemma4 source line 1365).
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        # Build sparse (T, E) expert_affinities: zero everywhere except the
        # top-k positions, which carry the post-renormalize, post-per-expert
        # scaled weights. With MoE config `normalize_top_k_affinities=False`,
        # the downstream ExpertMLPsV2 will use these values verbatim.
        expert_affinities = torch.zeros_like(probs)
        expert_affinities = expert_affinities.scatter(
            1, top_k_index, top_k_weights.to(expert_affinities.dtype)
        )

        # Cast to hidden_states dtype so downstream matmuls stay in bf16.
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = top_k_index.detach().to(dtype=torch.long)
        # router_logits stays in FP32; NxDI doesn't actually use it at
        # inference (return_router_logits is False by default).
        return router_logits, expert_affinities, expert_index


class NeuronGemma4MoEBlock(nn.Module):
    """Gemma-4 MoE block built on top of NxDI's `initialize_moe_module`.

    Why we don't just call NxDI's wrapper `MoE.forward(hidden_states)`:
    `MoE.forward` calls the router and the expert_mlps with the SAME
    `hidden_states`, but Gemma-4's HF source feeds the router the RAW
    pre-MLP residual (line 1434) while the experts get the
    `pre_feedforward_layernorm_2(residual)` (line 1435-1436). The two
    inputs differ by an RMSNorm with a learned scale, so collapsing them
    is numerically incorrect.

    We therefore:
      1. Use `initialize_moe_module(config=...)` to construct router +
         expert_mlps with all the correct process-group / dtype wiring.
      2. Replace the auto-built `RouterTopK` with `NeuronGemma4Router`
         (custom math: no-scale norm, scalar_root_size, learned scale +
         per_expert_scale, top-k renorm).
      3. Re-implement the wrapper's forward path — call router on
         `router_input`, expert_mlps on `expert_input`, then do the
         delayed all-reduce that `MoE.forward` would have done.

    No shared experts inside this block: Gemma-4's "shared" branch is the
    dense MLP that runs in PARALLEL with this block at the decoder-layer
    level (HF source 1427+1441 — `mlp(x) + experts(x)`), so we set
    `n_shared_experts = 0` and skip NxDI's `_apply_shared_experts` path.

    Sequence parallelism / token shuffle / EP > 1 are unsupported by this
    wrapper (we follow the simplest non-SP, non-EP path). If those are
    needed later, mirror the corresponding branches from
    `neuronx_distributed.modules.moe.model.MoE._forward_compute_bound`.
    """

    def __init__(self, config: Gemma4InferenceConfig):
        super().__init__()
        self.config = config
        # `initialize_moe_module` reads `n_shared_experts` -- Llama4 sets =1,
        # we set =0 because Gemma's "shared" branch is the dense MLP that
        # lives OUTSIDE this module.
        if not hasattr(config, "n_shared_experts"):
            config.n_shared_experts = 0

        # Build the underlying NxDI MoE wrapper to leverage its router +
        # expert_mlps construction (process groups, dtype, padding, etc.).
        self.moe = initialize_moe_module(config=config)
        # Replace the wrapper's RouterTopK with Gemma's custom router. Safe:
        # `MoE.__init__` already validated num_experts/top_k/hidden_size
        # cross-consistency, and we keep those identical on our router.
        self.moe.router = NeuronGemma4Router(config)

        # Pull tensor-parallel group from the wrapper for the explicit
        # delayed all-reduce in our forward.
        self._tp_group = self.moe.tensor_parallel_group
        self._ep_enabled = self.moe.ep_enabled

    def forward(
        self,
        router_input: torch.Tensor,
        expert_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gemma-4 MoE output.

        Arguments:
            router_input: pre-norm `residual`, shape (B, S, H) or (T, H).
                          The router applies its own (no-scale) RMSNorm
                          internally, matching HF source line 1434.
            expert_input: `pre_feedforward_layernorm_2(residual)`, same
                          leading layout as `router_input`. Fed to the
                          expert MLPs (HF source line 1436).

        Returns:
            output: same leading shape as `expert_input`, after expert
                    dispatch + post-TP all-reduce. Caller is responsible
                    for any further normalization (HF's
                    `post_feedforward_layernorm_2`) and summation with
                    the dense-MLP branch.
        """
        # Router consumes raw residual (it flattens to (T, H) internally).
        # Returns: router_logits (T,E), expert_affinities (T,E),
        #          expert_index (T, top_k).
        _router_logits, expert_affinities, expert_index = self.moe.router(router_input)

        # Mirror MoE.forward: gradient bookkeeping all-reduce on expert
        # affinities (no-op forward in pure inference, but keeps the
        # autograd graph consistent if we're ever traced for backward).
        if not self._ep_enabled:
            expert_affinities = _nxd_parallel_mappings.copy_to_tensor_model_parallel_region(
                expert_affinities, process_group=self._tp_group
            )

        # Flatten experts input to (T, H). seq_len comes from the
        # sequence_dimension axis of the un-flattened input. Caller is
        # expected to pass (B, S, H); if a 2D (T, H) tensor lands here we
        # fall back to seq_len = T (seq_len affects only path selection
        # heuristics inside ExpertMLPsV2, not correctness for full-capacity
        # routing).
        expert_shape = expert_input.shape
        if expert_input.dim() >= self.moe.sequence_dimension + 1:
            seq_len = expert_shape[self.moe.sequence_dimension]
        else:
            seq_len = expert_shape[0]
        expert_input_flat = expert_input.reshape(-1, expert_shape[-1])

        output = self.moe.expert_mlps(
            hidden_states=expert_input_flat,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len,
        )

        # Reshape back and do delayed all-reduce (TP) since expert_mlps
        # leaves the output in tensor-parallel un-reduced form.
        output = output.view(expert_shape)
        if not self._ep_enabled:
            output = _nxd_parallel_mappings.reduce_from_tensor_model_parallel_region(
                output, process_group=self._tp_group
            )
        return output


# ====================================================================================
# Decoder layer (dense MLP + parallel MoE branch)
# ====================================================================================


class NeuronGemma4DecoderLayer(nn.Module):
    """Gemma4 decoder layer with optional parallel MoE branch and layer_scalar."""

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_sliding_window_attention = config._layer_is_sliding

        self.self_attn = NeuronGemma4Attention(config)
        self.mlp = NeuronGemma4MLP(config)

        norm_cls = get_rmsnorm_cls()
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer learned scaling factor (must be Parameter, not buffer, so
        # NxDI's weight loader populates it from the checkpoint).
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

        # MoE branch can be disabled for smoke compile (set
        # `disable_moe_for_smoke_compile=True` on the InferenceConfig) to
        # validate the rest of the architecture without the routed experts.
        # When enabled, the dense MLP and the MoE block run in PARALLEL and
        # their outputs are summed (HF source 1427-1441).
        self.enable_moe_block = bool(getattr(config, "enable_moe_block", False)) and not bool(
            getattr(config, "disable_moe_for_smoke_compile", False)
        )
        if self.enable_moe_block:
            # Build a separate config view for MoE. Two things must change
            # vs the per-layer attention config:
            #   1. `intermediate_size` -> `moe_intermediate_size` (704 not
            #      2112). NxDI's `ExpertMLPsV2` reads `config.intermediate_size`
            #      for the expert intermediate dim.
            #   2. `num_local_experts` / `num_experts_per_tok` aliases for
            #      Gemma's `num_experts` / `top_k_experts` (already set at
            #      the per-layer level by `get_updated_configs`, but we set
            #      again for explicitness on the deepcopy).
            #   3. `n_shared_experts = 0` — the dense MLP plays the role of
            #      "shared expert" but lives OUTSIDE the MoE block (parallel
            #      branch summed with experts output, per HF source).
            moe_config = copy.deepcopy(config)
            moe_config.intermediate_size = config.moe_intermediate_size
            moe_config.num_local_experts = config.num_experts
            moe_config.num_experts_per_tok = config.top_k_experts
            moe_config.n_shared_experts = 0
            # ExpertMLPsV2 raises `Unknown activation: gelu_pytorch_tanh`.
            # Alias to plain `gelu` — tanh-approximation diff is <1e-4 and not
            # load-bearing for routing/topk correctness. Flag for accuracy
            # validation pass.
            hf_act = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
            moe_config.hidden_act = "gelu" if hf_act == "gelu_pytorch_tanh" else hf_act

            self.moe_block = NeuronGemma4MoEBlock(moe_config)
            self.post_feedforward_layernorm_1 = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Heterogeneous RoPE: drop cached cos/sin from the previous layer.
        kwargs.pop("cos_cache", None)
        kwargs.pop("sin_cache", None)

        # SWA layers use local_mask; global layers use attention_mask.
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

        # Feed-forward block
        residual = hidden_states
        hidden_states_pre = self.pre_feedforward_layernorm(hidden_states)
        hidden_states_dense = self.mlp(hidden_states_pre)[0]

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_dense)

            # HF source 1432-1438: router consumes the RAW residual, while
            # the experts consume `pre_feedforward_layernorm_2(residual)`.
            # The two inputs differ by a learned-scale RMSNorm and cannot
            # be collapsed, so we pass them separately to the MoE block
            # (which bypasses NxDI's MoE wrapper to honor this split).
            expert_input = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe_block(
                router_input=residual,
                expert_input=expert_input,
            )
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2
        else:
            hidden_states = hidden_states_dense

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Per-layer scalar
        hidden_states = hidden_states * self.layer_scalar

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# KV cache manager (PR #106 verbatim — handles heterogeneous SWA/global shapes)
# ====================================================================================


class Gemma4KVCacheManager(KVCacheManager):
    """KV cache manager with per-layer heterogeneous shapes.

    26B-A4B layer kv configs (per rank, after TP sharding):
      - SWA layers:    num_kv_heads=8 / TP, head_dim=256
      - Global layers: num_kv_heads=2 / TP, head_dim=512
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

        if layer_to_cache_size_mapping:
            layer_seq_lens = list(layer_to_cache_size_mapping)
        else:
            layer_seq_lens = [max_len] * len(self._layer_kv_configs)

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
# Top-level model
# ====================================================================================


class NeuronGemma4TextModel(NeuronBaseModel):
    """Gemma4 text decoder: scaled embeds + decoder layers + final norm + softcapped lm_head."""

    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # Use the maximum KV head count (SWA = 8 on 26B-A4B) for base class.
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

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

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(conf, idx) for idx, conf in enumerate(updated_configs)]
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

        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping > 0
        ):
            self.lm_head = SoftcappedLMHead(lm_head_linear, self.final_logit_softcapping)
        else:
            self.lm_head = lm_head_linear

        self.has_mixed_attn = True
        self.sliding_window = config.sliding_window

        max_length = config.neuron_config.max_length
        sw = config.sliding_window or max_length
        self._uniform_cache_len = max(sw, max_length)
        self.layer_to_cache_size_mapping = [self._uniform_cache_len] * config.num_hidden_layers

    def _create_windowed_attn_mask_tkg(self, attention_mask, window_size, position_ids):
        """SWA TKG mask must match uniform KV cache size (PR #106 fix)."""
        batch_size, _ = attention_mask.shape
        cache_len = self._uniform_cache_len

        if cache_len == window_size:
            return super()._create_windowed_attn_mask_tkg(
                attention_mask, window_size, position_ids
            )

        pos = position_ids[:, 0]
        idx = torch.arange(window_size, device=attention_mask.device).unsqueeze(0)
        base_mask = (idx < pos.unsqueeze(1)) & (idx < window_size - 1)

        full_mask = torch.ones(
            (batch_size, window_size), dtype=torch.bool, device=attention_mask.device
        )
        full_mask[:, -1] = False
        seq_less_than_window = pos < window_size - 1
        window_mask = torch.where(
            seq_less_than_window.unsqueeze(1), base_mask, full_mask
        )
        pad_len = cache_len - window_size
        padded_mask = F.pad(window_mask, (0, pad_len), value=False)
        return padded_mask[:, None, None, :]

    def _create_simple_attn_mask(self, attention_mask):
        """Global mask must match uniform KV cache size (PR #106 fix)."""
        batch_size = attention_mask.shape[0]
        pad_len = self._uniform_cache_len - self.n_positions
        if pad_len > 0:
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        return (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, 1, self._uniform_cache_len)
            .to(torch.bool)
        )

    def init_inference_optimization(self, config: Gemma4InferenceConfig):
        if self.on_device_sampling:
            try:
                from neuronx_distributed_inference.modules.generation.sampling import (
                    create_sampler,
                )
            except ImportError:
                from neuronx_distributed_inference.modules.sampling.utils import (
                    create_sampler,
                )

            lm_head_tp_degree = None
            if hasattr(self, "lm_head") and hasattr(
                self.lm_head, "tensor_parallel_group"
            ):
                lm_head_tp_degree = self.lm_head.tensor_parallel_group.size()
            self.sampler = create_sampler(config.neuron_config, lm_head_tp_degree)

        tp_degree = config.neuron_config.tp_degree
        layer_kv_configs = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            if layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads
                hd = config.head_dim
            else:
                kv_heads = config.num_global_key_value_heads
                hd = config.global_head_dim
            gqa_strategy = determine_sharding_strategy(tp_degree, kv_heads)
            _, shardable_kv_heads = get_shardable_head_counts(
                tp_degree, config.num_attention_heads, kv_heads, gqa_strategy
            )
            kv_heads_per_rank = max(1, shardable_kv_heads // tp_degree)
            layer_kv_configs.append((kv_heads_per_rank, hd))

        self._layer_kv_configs = layer_kv_configs
        self._max_kv_heads_per_rank = max(c[0] for c in layer_kv_configs)
        self._max_head_dim = max(c[1] for c in layer_kv_configs)

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
# Causal-LM wrapper + state-dict converter
# ====================================================================================


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """Gemma4 causal LM for NeuronX inference.

    Handles weight conversion from HF checkpoint to NxDI naming, including:
      * stripping `language_model.`, `model.` prefixes
      * embed_tokens -> embed_tokens.embedding (ScaledEmbedding wrapper)
      * q/k_norm -> q_layernorm/k_layernorm
      * QK scaling correction (cancel NxDI's automatic 1/sqrt(head_dim))
      * `attention_k_eq_v` -> copy k_proj weights into v_proj for global layers
      * tied lm_head (handle SoftcappedLMHead path)
      * rank_util tensors for TP

    Plus the 26B-A4B-specific MoE keys:
      * `layers.{i}.router.{proj.weight,scale,per_expert_scale}` ->
        `layers.{i}.moe_block.moe.router.{...}` (router lives inside our
        wrapper module after `initialize_moe_module`).
      * `layers.{i}.experts.gate_up_proj` (HF shape `(E, 2*I, H)`) ->
        `layers.{i}.moe_block.moe.expert_mlps.mlp_op.gate_up_proj.weight`
        (NxDI shape `(E, H, 2*I)` — last two dims transposed).
      * `layers.{i}.experts.down_proj` (HF shape `(E, H, I)`) ->
        `layers.{i}.moe_block.moe.expert_mlps.mlp_op.down_proj.weight`
        (NxDI shape `(E, I, H)` — last two dims transposed).
    """

    _model_cls = NeuronGemma4TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma4ForConditionalGeneration  # type: ignore[import-not-found]

        return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: Gemma4InferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}

        for key, weight in state_dict.items():
            new_key = key

            if new_key.startswith("language_model.model."):
                new_key = new_key[len("language_model.model."):]
            elif new_key.startswith("language_model."):
                new_key = new_key[len("language_model."):]
            elif new_key.startswith("model.language_model.model."):
                new_key = new_key[len("model.language_model.model."):]
            elif new_key.startswith("model.language_model."):
                new_key = new_key[len("model.language_model."):]
            elif new_key.startswith("model."):
                new_key = new_key[len("model."):]

            # Skip vision/audio/multimodal weights — text-only port for now.
            if (
                "vision_tower." in new_key
                or "multi_modal_projector." in new_key
                or "embed_vision." in new_key
                or "audio_tower." in new_key
                or "embed_audio." in new_key
            ):
                continue

            if new_key == "embed_tokens.weight":
                new_key = "embed_tokens.embedding.weight"

            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            new_state_dict[new_key] = weight.detach().clone()

        # Per-layer transformations
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            is_global = layer_type == "full_attention"

            if is_global:
                hd = config.global_head_dim
            else:
                hd = config.head_dim

            prefix = f"layers.{i}.self_attn"

            # QK scaling: gemma4 uses scaling=1.0 (no 1/sqrt(head_dim)). NxDI
            # always applies 1/sqrt(head_dim). Pre-scale q_layernorm.weight by
            # sqrt(head_dim) so the effects cancel after RMSNorm scale-invariance.
            q_norm_key = f"{prefix}.q_layernorm.weight"
            if q_norm_key in new_state_dict:
                scaling_factor = math.sqrt(float(hd))
                orig_dtype = new_state_dict[q_norm_key].dtype
                new_state_dict[q_norm_key] = (
                    new_state_dict[q_norm_key].to(torch.float32) * scaling_factor
                ).to(orig_dtype)

            # attention_k_eq_v: copy K weights to V for global layers (no v_proj in HF).
            if is_global and getattr(config, "attention_k_eq_v", False):
                k_key = f"{prefix}.k_proj.weight"
                v_key = f"{prefix}.v_proj.weight"
                if k_key in new_state_dict and v_key not in new_state_dict:
                    new_state_dict[v_key] = new_state_dict[k_key].detach().clone()

            new_state_dict[f"{prefix}.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # MoE remap: HF stores router/experts directly under the
            # decoder layer. Our `NeuronGemma4DecoderLayer` nests them as
            # `moe_block.moe.{router,expert_mlps.mlp_op}`. Skip silently
            # if the layer has no MoE keys (e.g. dense smoke-compile run
            # or a future variant that toggles MoE per-layer).
            moe_src_prefix = f"layers.{i}."
            moe_dst_prefix = f"layers.{i}.moe_block.moe."
            router_keys = [
                ("router.proj.weight", "router.proj.weight"),
                ("router.scale", "router.scale"),
                ("router.per_expert_scale", "router.per_expert_scale"),
            ]
            for src_suffix, dst_suffix in router_keys:
                src_key = moe_src_prefix + src_suffix
                if src_key in new_state_dict:
                    new_state_dict[moe_dst_prefix + dst_suffix] = new_state_dict.pop(
                        src_key
                    )

            gate_up_src = moe_src_prefix + "experts.gate_up_proj"
            gate_up_dst = moe_dst_prefix + "expert_mlps.mlp_op.gate_up_proj.weight"
            if gate_up_src in new_state_dict:
                # HF shape: (E, 2*I, H); NxDI shape: (E, H, 2*I).
                w = new_state_dict.pop(gate_up_src)
                new_state_dict[gate_up_dst] = w.transpose(1, 2).contiguous()

            down_src = moe_src_prefix + "experts.down_proj"
            down_dst = moe_dst_prefix + "expert_mlps.mlp_op.down_proj.weight"
            if down_src in new_state_dict:
                # HF shape: (E, H, I); NxDI shape: (E, I, H).
                w = new_state_dict.pop(down_src)
                new_state_dict[down_dst] = w.transpose(1, 2).contiguous()

        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.embedding.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Tied weights: embed_tokens -> lm_head (handle SoftcappedLMHead path)."""
        embed_key = None
        if "embed_tokens.embedding.weight" in state_dict:
            embed_key = "embed_tokens.embedding.weight"
        elif "embed_tokens.weight" in state_dict:
            embed_key = "embed_tokens.weight"

        if embed_key is not None:
            weight = state_dict[embed_key].clone()
            state_dict["lm_head.weight"] = weight
            state_dict["lm_head.linear.weight"] = weight.clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig


__all__ = [
    "Gemma4InferenceConfig",
    "Gemma4NeuronConfig",
    "Gemma4KVCacheManager",
    "Gemma4RMSNorm",
    "Gemma4ScaledEmbedding",
    "Gemma4VNorm",
    "NeuronGemma4Attention",
    "NeuronGemma4DecoderLayer",
    "NeuronGemma4ForCausalLM",
    "NeuronGemma4MLP",
    "NeuronGemma4MoEBlock",
    "NeuronGemma4Router",
    "NeuronGemma4TextModel",
    "SoftcappedLMHead",
    "get_updated_configs",
]
