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
PyTorch Gemma4 E2B model for NeuronX Distributed Inference.

Gemma4 E2B (google/gemma-4-E2B) is a 2.3B-effective-parameter model with 35 layers,
featuring Per-Layer Embeddings (PLE) and KV cache sharing.

Key differences from Gemma4 31B:
  - hidden_size: 1536 (was 5376)
  - num_hidden_layers: 35 (was 60)
  - num_attention_heads: 8 (was 32)
  - num_key_value_heads: 1 for all layers (was 16 SWA / 4 global)
  - head_dim: 256 (SWA), 512 (global) -- same as 31B
  - intermediate_size: per-layer (6144 for layers 0-14, 12288 for layers 15-34)
  - sliding_window: 512 (was 1024)

E2B-specific features:
  - Per-Layer Embeddings (PLE): additional embedding table and per-layer gate/projection
  - Per-layer MLP sizes (intermediate_size varies by layer)
  - KV cache sharing: layers 15-34 reuse KV from donor layers 13/14
  - No v_norm, no attention_k_eq_v (unlike 31B)
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
    move_heads_front,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    _rotate_half,
)


class ProportionalRotaryEmbedding(RotaryEmbedding):
    """
    RoPE variant for proportional (partial) rotation used in Gemma4 global layers.

    Standard RotaryEmbedding computes: inv_freq = 1/(base^(i/dim))
    where dim = rotary_dim (the number of rotated dimensions).

    HF Gemma4's proportional RoPE computes: inv_freq = 1/(base^(i/head_dim))
    where head_dim is the FULL head dimension (e.g. 512), even though only
    rotary_dim dims (e.g. 128) are actually rotated.

    This produces lower frequencies (slower rotation) than standard RoPE,
    which is critical for correctness.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, head_dim=None):
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )
        # head_dim is the denominator for frequency computation
        self._freq_denominator = head_dim if head_dim is not None else dim

    def get_inv_freqs(self, device=None):
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self._freq_denominator))
        factor = getattr(self, "factor", None)
        if factor is not None:
            inv_freq = inv_freq / factor
        return inv_freq


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


def get_rmsnorm_cls():
    """Return appropriate RMSNorm for current execution context.

    Uses NxDI's CustomRMSNorm (which calls AwsNeuronRmsNorm) when available,
    falling back to our Python Gemma4RMSNorm. Using the Neuron-optimized kernel
    avoids accumulated numerical divergence between Python RMSNorm and the
    compiled graph's behavior over many layers.

    Wraps CustomRMSNorm to accept both positional (dim) and keyword (hidden_size)
    first arg for API compatibility with Gemma4RMSNorm call sites.
    """
    try:
        from neuronx_distributed_inference.modules.custom_calls import (
            CustomRMSNorm as _CRN,
        )

        class CompatRMSNorm(_CRN):
            """CustomRMSNorm wrapper that accepts 'dim' as first positional arg."""

            def __init__(self, dim=None, hidden_size=None, eps=1e-6):
                actual_dim = dim if dim is not None else hidden_size
                super().__init__(hidden_size=actual_dim, eps=eps)

        return CompatRMSNorm
    except ImportError:
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
    Simple scaled token embedding: ParallelEmbedding(input_ids) * sqrt(hidden_size).

    This is the standard Gemma4 embedding used as self.embed_tokens. PLE is computed
    separately in Gemma4PLEModule and passed through get_model_output.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        dtype: torch.dtype,
        shard_across_embedding: bool = True,
        pad: bool = True,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()
        self.embed_scale = hidden_size**0.5
        self.embedding = ParallelEmbedding(
            vocab_size,
            hidden_size,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            pad=pad,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids: torch.Tensor):
        return self.embedding(input_ids) * self.embed_scale


class Gemma4PLEModule(nn.Module):
    """
    Per-Layer Embedding (PLE) module for Gemma4 E2B.

    Computes the PLE tensor from input_ids and main embeddings. Called explicitly
    from get_model_output (NOT via side-effects/caching, which don't survive
    torch.jit.trace).

    Forward:
      1. PLE lookup: embed_tokens_per_layer(input_ids) * sqrt(ple_dim) -> reshape
      2. PLE projection: per_layer_model_projection(inputs_embeds) * (1/sqrt(hidden_size))
         -> reshape -> per_layer_projection_norm -> add PLE lookup -> scale by rsqrt(2)
      3. Return combined PLE [B, S, num_hidden_layers, ple_dim]
    """

    def __init__(self, config, padding_idx: int, dtype: torch.dtype):
        super().__init__()
        hidden_size = config.hidden_size
        ple_dim = config.hidden_size_per_layer_input  # 256
        num_layers = config.num_hidden_layers  # 35
        total_ple_dim = num_layers * ple_dim  # 8960
        vocab_size_ple = config.vocab_size_per_layer_input

        self._ple_dim = ple_dim
        self._num_layers = num_layers

        # PLE embedding table: vocab_size -> L * ple_dim, scaled by sqrt(ple_dim)
        # CRITICAL: Use bf16 for PLE embedding to save ~4.6 GB HBM.
        # The full PLE embedding in f32 is 262144 x 8960 x 4 = 9.17 GB, which causes
        # CTE OOM on a single core (24 GB limit with LNC=2). bf16 halves this to 4.58 GB.
        # The lookup result is cast to f32 in forward() before further computation.
        self.ple_embed_scale = ple_dim**0.5
        self.embed_tokens_per_layer = ParallelEmbedding(
            vocab_size_ple,
            total_ple_dim,
            padding_idx,
            dtype=torch.bfloat16,
            shard_across_embedding=True,
            pad=True,
        )

        # PLE model projection: hidden_size -> L * ple_dim
        self.per_layer_model_projection = ColumnParallelLinear(
            hidden_size,
            total_ple_dim,
            bias=False,
            gather_output=True,
            dtype=dtype,
            pad=True,
        )

        # PLE projection norm: RMSNorm on ple_dim
        self.per_layer_projection_norm = get_rmsnorm_cls()(
            ple_dim, eps=config.rms_norm_eps
        )

        # PLE scaling constants
        self.register_buffer(
            "per_layer_projection_scale",
            torch.tensor(hidden_size ** (-0.5), dtype=torch.float32),
        )
        self.register_buffer(
            "per_layer_input_scale",
            torch.tensor(1.0 / math.sqrt(2.0), dtype=torch.float32),
        )

    def forward(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PLE tensor.

        Args:
            input_ids: [B, S]
            inputs_embeds: [B, S, hidden_size] — already-scaled main embeddings
        Returns:
            per_layer_inputs: [B, S, num_hidden_layers, ple_dim]

        Note: RMSNorm is computed manually inline instead of using the nn.Module
        RMSNorm, to work around NCC_ITEN404 compiler crash. The compiler has an
        internal tensorizer issue with the reduce pattern generated by RMSNorm
        on reshaped tensors in the CTE (context encoding) model.
        """
        B, S = input_ids.shape
        L = self._num_layers
        D = self._ple_dim

        # PLE embedding lookup -> [B, S, L * D] -> cast to f32 -> reshape -> [B, S, L, D]
        ple_embeds = (
            self.embed_tokens_per_layer(input_ids).float() * self.ple_embed_scale
        )
        ple_embeds = ple_embeds.view(B, S, L, D)

        # PLE projection: main_embeds -> [B, S, L * D] -> reshape -> [B, S, L, D]
        proj = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_projection_scale
        proj = proj.view(B, S, L, D)

        # Manual RMSNorm on last dimension (avoids compiler crash)
        proj_f = proj.float()
        variance = proj_f.pow(2).mean(-1, keepdim=True)
        proj_normed = proj_f * torch.rsqrt(variance + 1e-6)
        proj = (proj_normed * self.per_layer_projection_norm.weight.float()).to(
            proj.dtype
        )

        # Combine: (norm(proj) + ple_lookup) * rsqrt(2)
        return (proj + ple_embeds) * self.per_layer_input_scale


# ====================================================================================
# Configuration
# ====================================================================================


class Gemma4E2BNeuronConfig(NeuronConfig):
    """NeuronConfig with Gemma4 E2B-specific attention class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma4E2BAttention


class Gemma4E2BInferenceConfig(InferenceConfig):
    """
    Configuration for Gemma4 E2B inference on NeuronX.
    Reads from HuggingFace config.json and extracts text_config fields.

    E2B-specific config attributes:
      - hidden_size_per_layer_input: dimension of PLE vectors (typically 256)
      - num_kv_shared_layers: number of layers sharing KV cache (20)
      - layer_intermediate_sizes: per-layer intermediate_size list (computed)
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
                "final_logit_softcapping",
                "layer_types",
                "rope_parameters",
                # E2B-specific
                "hidden_size_per_layer_input",
                "num_kv_shared_layers",
                "vocab_size_per_layer_input",
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

        # E2B defaults
        if not hasattr(self, "hidden_size_per_layer_input"):
            self.hidden_size_per_layer_input = 256
        if not hasattr(self, "num_kv_shared_layers"):
            self.num_kv_shared_layers = 20
        if not hasattr(self, "global_head_dim"):
            # E2B uses same global_head_dim as 31B
            self.global_head_dim = 512
        if not hasattr(self, "vocab_size_per_layer_input"):
            self.vocab_size_per_layer_input = self.vocab_size

        # E2B has attention_k_eq_v=false and no num_global_key_value_heads distinction.
        # All layers use the same num_key_value_heads=1.
        self.attention_k_eq_v = False
        # Override any HF config value -- E2B has null for this
        self.num_global_key_value_heads = self.num_key_value_heads

        # Compute per-layer intermediate sizes.
        # E2B: layers 0 to (first_kv_shared_layer_idx - 1) use smaller intermediate_size,
        # layers from first_kv_shared_layer_idx onward use 2x intermediate_size.
        # The config.intermediate_size is the SMALLER value (6144).
        # Use use_double_wide_mlp from HF config if available; else compute from
        # num_kv_shared_layers. The boundary is always at layer 15 in the full model.
        # When testing with truncated layer counts, we use the original boundary (15)
        # so that layers 0-14 correctly get the smaller intermediate_size.
        full_num_layers = 35  # E2B always has 35 layers
        first_kv_shared = full_num_layers - self.num_kv_shared_layers  # 35 - 20 = 15
        base_intermediate = self.intermediate_size  # 6144
        self.layer_intermediate_sizes = []
        for i in range(self.num_hidden_layers):
            if i < first_kv_shared:
                self.layer_intermediate_sizes.append(base_intermediate)
            else:
                self.layer_intermediate_sizes.append(base_intermediate * 2)

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
            "layer_types",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4E2BNeuronConfig]:
        return Gemma4E2BNeuronConfig


def get_updated_configs(config: Gemma4E2BInferenceConfig):
    """
    Generate per-layer configs for heterogeneous SWA/global layers.

    E2B layers:
      SWA layers: head_dim=256, num_key_value_heads=1, sliding_window=512
      Global layers: head_dim=512, num_key_value_heads=1, sliding_window=None

    Each layer also gets its own intermediate_size from config.layer_intermediate_sizes.

    NKI flash attention kernel limit is head_dim=128, so both layer types
    must use decomposed attention (attn_kernel_enabled=False).
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)
        layer_type = config.layer_types[i]

        # Per-layer intermediate size
        layer_config._layer_intermediate_size = config.layer_intermediate_sizes[i]

        if layer_type == "sliding_attention":
            layer_config.sliding_window = config.sliding_window
            layer_config._layer_head_dim = config.head_dim
            layer_config._layer_num_kv_heads = config.num_key_value_heads
            layer_config._layer_is_sliding = True
            # RoPE: default type, theta=10000
            rope_params = config.rope_parameters.get("sliding_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 10000.0)
            layer_config._layer_partial_rotary_factor = 1.0  # full rotation for SWA
        else:
            # full_attention (global)
            layer_config.sliding_window = None
            layer_config._layer_head_dim = config.global_head_dim
            layer_config._layer_num_kv_heads = (
                config.num_key_value_heads
            )  # 1 for all E2B layers
            layer_config._layer_is_sliding = False
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


def _maybe_post_transpose_layernorm():
    """Return post_transpose_layernorm kwarg if supported by NxDI version (>= 0.8.0)."""
    import inspect

    sig = inspect.signature(NeuronAttentionBase.__init__)
    if "post_transpose_layernorm" in sig.parameters:
        return {"post_transpose_layernorm": True}
    return {}


def _move_heads_front_kwargs(post_transpose_layernorm_value):
    """Return post_transpose_layernorm kwarg for move_heads_front if supported."""
    import inspect

    sig = inspect.signature(move_heads_front)
    if "post_transpose_layernorm" in sig.parameters:
        return {"post_transpose_layernorm": post_transpose_layernorm_value}
    return {}


class NeuronGemma4E2BAttention(NeuronAttentionBase):
    """
    Gemma4 E2B attention with:
    - Per-layer head_dim (256 for SWA, 512 for global)
    - All layers: num_key_value_heads=1
    - QK normalization via RMSNorm (standard weight scaling)
    - No v_norm (unlike 31B)
    - No attention_k_eq_v (all layers have separate K and V projections)
    - Partial rotary for global layers (0.25 factor)
    """

    def __init__(self, config: Gemma4E2BInferenceConfig):
        head_dim = config._layer_head_dim
        num_kv_heads = config._layer_num_kv_heads
        is_sliding = config._layer_is_sliding
        rope_theta = config._layer_rope_theta
        partial_rotary_factor = config._layer_partial_rotary_factor

        # RoPE dimension: for global layers with partial_rotary_factor=0.25,
        # only 25% of dims get RoPE. RotaryEmbedding always rotates dim/2 pairs,
        # so we pass rotary_dim = head_dim * partial_rotary_factor (rounded to even).
        rotary_dim = int(head_dim * partial_rotary_factor)
        rotary_dim = rotary_dim - (rotary_dim % 2)  # ensure even

        if partial_rotary_factor < 1.0:
            # Global layers use proportional RoPE: frequencies are computed with
            # head_dim as denominator (not rotary_dim). This produces lower frequencies
            # matching HF Gemma4's proportional rope_type.
            # Standard: inv_freq = 1/(theta^(i/rotary_dim))  [WRONG for partial rotation]
            # Correct:  inv_freq = 1/(theta^(i/head_dim))     [proportional RoPE]
            rotary_emb = ProportionalRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=rope_theta,
                head_dim=head_dim,
            )
        else:
            # SWA layers: full rotation (rotary_dim == head_dim), standard RoPE
            rotary_emb = RotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=rope_theta,
            )

        # Pass sliding_window=None to base class for ALL layers.
        # This routes all layers through standard_causal_attention_forward instead of
        # windowed_attention_forward. The windowed_attention_forward path calls
        # get_last_kv_window() which does torch.gather with indices up to
        # sliding_window-2, but during CTE the K/V only has bucket_size positions,
        # causing OOB when bucket_size < sliding_window.
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
            **_maybe_post_transpose_layernorm(),
        )

        # QK norms: Gemma4 uses standard weight RMSNorm
        self.q_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps)

        # No v_norm in E2B (unlike 31B which has Gemma4VNorm)

        # Store layer properties
        self._is_sliding = is_sliding
        self._head_dim = head_dim
        self._rotary_dim = rotary_dim
        self._partial_rotary_factor = partial_rotary_factor

        # KV sharing: when set, skip K/V projection and use these K/V tensors instead.
        # Set by the decoder layer before calling forward().
        # Format: (K_shared, V_shared) in BHSD layout, already with RoPE applied.
        self._shared_kv = None

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """
        Override to handle partial rotary embedding for global layers.

        For SWA layers: partial_rotary_factor=1.0, rotary_dim==head_dim -> full rotation.
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
    # E2B has no v_norm, so we do NOT override prep_qkv_tensors (unlike 31B).

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """
        Override to support KV sharing.

        When self._shared_kv is set, skip K/V projection and use the donor's
        pre-computed K/V instead. Only Q is computed from the current layer's
        hidden_states.

        The shared K/V already have QK-norm and RoPE applied (from the donor
        layer's prep_qkv_tensors), so we only need to compute and prepare Q.
        """
        if self._shared_kv is not None:
            K_shared, V_shared = self._shared_kv
            self._shared_kv = None  # consume it

            bsz, q_len, _ = hidden_states.size()

            # Compute Q using the layer's own q_proj
            qkv_proj = self.get_qkv_proj()
            Q = qkv_proj.q_proj(hidden_states)

            # Reshape Q: BS(H*D) -> BHSD and apply Q layernorm
            Q = move_heads_front(
                Q,
                bsz,
                q_len,
                self.num_heads,
                self.head_dim,
                layernorm=self.q_layernorm,
                **_move_heads_front_kwargs(
                    getattr(self, "post_transpose_layernorm", False)
                ),
            )

            # Apply RoPE to Q only (K already has RoPE from donor)
            if not skip_rope and self.rotary_emb is not None:
                if cos_cache is None or sin_cache is None:
                    cos_cache, sin_cache = self.rotary_emb(Q, position_ids)

                if self._rotary_dim == self._head_dim:
                    # Full rotation (SWA layers)
                    cos_unsq = cos_cache.unsqueeze(1)
                    sin_unsq = sin_cache.unsqueeze(1)
                    Q = (Q * cos_unsq) + (_rotate_half(Q) * sin_unsq)
                else:
                    # Partial rotation (global layers)
                    cos_unsq = cos_cache.unsqueeze(1)
                    sin_unsq = sin_cache.unsqueeze(1)
                    q_rot = Q[..., : self._rotary_dim]
                    q_pass = Q[..., self._rotary_dim :]
                    q_rot = (q_rot * cos_unsq) + (_rotate_half(q_rot) * sin_unsq)
                    Q = torch.cat([q_rot, q_pass], dim=-1)

            return Q, K_shared, V_shared, cos_cache, sin_cache, residual

        # Default: compute full Q, K, V (non-shared layers)
        return super().prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=skip_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )


# ====================================================================================
# MLP
# ====================================================================================


class NeuronGemma4E2BMLP(nn.Module):
    """
    Gemma4 E2B MLP with GELU(tanh) activation.
    gate_proj and up_proj are column-parallel, down_proj is row-parallel.

    Reads intermediate_size from config._layer_intermediate_size for per-layer sizing.
    """

    def __init__(self, config: Gemma4E2BInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config._layer_intermediate_size

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


class NeuronGemma4E2BDecoderLayer(nn.Module):
    """
    Gemma4 E2B decoder layer with:
    - 4 RMSNorm layers (input, post_attention, pre_feedforward, post_feedforward)
    - Per-Layer Embeddings (PLE): gate, projection, and norm applied after MLP residual
    - layer_scalar at the end (learned per-layer multiplicative factor)
    - Per-layer intermediate_size for the MLP

    Forward flow:
      1. input_layernorm -> attention -> post_attention_layernorm -> residual add
      2. pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> residual add
      3. PLE: act_fn(gate(hidden_states)) * per_layer_input -> project -> norm -> residual add
      4. Multiply by layer_scalar
    """

    def __init__(self, config: Gemma4E2BInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_sliding_window_attention = config._layer_is_sliding

        # ======== KV cache sharing ========
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        num_layers = config.num_hidden_layers
        first_shared_idx = (
            num_layers - num_kv_shared if num_kv_shared > 0 else num_layers
        )
        self.is_kv_shared_layer = layer_idx >= first_shared_idx and num_kv_shared > 0

        if self.is_kv_shared_layer:
            # Find the donor: last non-shared layer of the same attention type
            layer_types = config.layer_types
            prev_layers = layer_types[:first_shared_idx]
            my_type = layer_types[layer_idx]
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(my_type)
            )
        else:
            self.kv_shared_layer_index = None

        self.self_attn = NeuronGemma4E2BAttention(config)
        self.mlp = NeuronGemma4E2BMLP(config)

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

        # PLE per-layer components
        ple_dim = config.hidden_size_per_layer_input  # 256
        self.per_layer_input_gate = nn.Linear(config.hidden_size, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Activation for PLE gating (GELU tanh, matching MLP)
        self.ple_act_fn = nn.GELU(approximate="tanh")

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

        # ======== KV sharing: inject donor K/V into attention module ========
        shared_kv = kwargs.pop("shared_kv", None)
        if self.is_kv_shared_layer and shared_kv is not None:
            self.self_attn._shared_kv = shared_kv

        # ======== Attention block ========
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

        # ======== MLP block ========
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ======== Per-Layer Embeddings (PLE) ========
        # PLE tensor is passed explicitly via kwargs from get_model_output.
        # Shape: [B, S, num_hidden_layers, ple_dim]
        per_layer_inputs = kwargs.get("per_layer_inputs", None)
        if per_layer_inputs is not None:
            # Slice this layer's PLE input
            per_layer_input = per_layer_inputs[
                :, :, self.layer_idx, :
            ]  # [B, S, ple_dim]

            # Gate: hidden_states -> ple_dim, apply activation
            gated = self.ple_act_fn(
                self.per_layer_input_gate(hidden_states)
            )  # [B, S, ple_dim]
            # Element-wise multiply with PLE input
            gated = gated * per_layer_input  # [B, S, ple_dim]
            # Project back to hidden_size
            projected = self.per_layer_projection(gated)  # [B, S, hidden_size]
            # Norm and residual add
            hidden_states = hidden_states + self.post_per_layer_input_norm(projected)

        # ======== Per-layer scaling ========
        hidden_states = hidden_states * self.layer_scalar

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# KV Cache
# ====================================================================================


class Gemma4E2BKVCacheManager(KVCacheManager):
    """
    KV cache manager for Gemma4 E2B with per-layer heterogeneous shapes.

    SWA layers: (B, kv_heads_per_rank, S, 256) per K/V
    Global layers: (B, kv_heads_per_rank, S, 512) per K/V

    With num_key_value_heads=1 on all layers, after TP sharding:
    - TP=1: 1 KV head per rank for all layers

    KV sharing: All 35 layers have independent cache slots. Shared layers
    (15-34) receive donor K/V from the layer loop, so their cache slots end
    up containing the donor's K/V history. This is wasteful (donor's K/V is
    stored twice) but correct. A future optimization could alias shared cache
    slots to the donor's slot.
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
        # in the base __init__. If not provided, create a uniform one.
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


class NeuronGemma4E2BTextModel(NeuronBaseModel):
    """
    Gemma4 E2B text model: embeddings + PLE + decoder layers + final norm + lm_head.

    Per-Layer Embeddings (PLE):
      E2B introduces an additional embedding table (embed_tokens_per_layer) that maps
      each token to a per-layer vector. This is combined with a projection of the main
      embeddings, then passed to each decoder layer as side input.

      PLE is computed in a separate Gemma4PLEModule and passed explicitly as
      per_layer_inputs through the decoder layer loop in get_model_output. This
      avoids side-effect caching which doesn't survive torch.jit.trace.
    """

    def setup_attr_for_model(self, config: Gemma4E2BInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads  # 1 for all E2B layers
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Gemma4E2BInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # ======== Token embeddings (scaled, no PLE) ========
        self.embed_tokens = Gemma4ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # ======== PLE module (separate from embed_tokens) ========
        self.ple_module = Gemma4PLEModule(
            config,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
        )

        # ======== Decoder layers ========
        # Per-layer configs for heterogeneous SWA/global shapes and intermediate sizes
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronGemma4E2BDecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )

        # ======== Final norm and LM head ========
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
        self.has_mixed_attn = True
        self.sliding_window = config.sliding_window  # SWA window size (512)

        # All layers use a uniform cache size = max(sliding_window, max_length).
        # Required because we pass sliding_window=None to the attention base class
        # (to avoid OOB in get_last_kv_window when bucket_size < sliding_window).
        max_length = config.neuron_config.max_length
        sw = config.sliding_window or max_length
        self._uniform_cache_len = max(sw, max_length)
        self.layer_to_cache_size_mapping = [
            self._uniform_cache_len
        ] * config.num_hidden_layers

        # ======== KV sharing: build donor map ========
        # Maps shared layer index -> donor layer index.
        # Donor = last non-shared layer of the same attention type.
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        num_layers = config.num_hidden_layers
        first_shared_idx = (
            num_layers - num_kv_shared if num_kv_shared > 0 else num_layers
        )
        self._kv_donor_map = {}  # {shared_layer_idx: donor_layer_idx}
        if num_kv_shared > 0:
            layer_types = config.layer_types
            non_shared_types = layer_types[:first_shared_idx]
            for layer_idx in range(first_shared_idx, num_layers):
                my_type = layer_types[layer_idx]
                # Find last non-shared layer of the same type
                donor_idx = (
                    len(non_shared_types) - 1 - non_shared_types[::-1].index(my_type)
                )
                self._kv_donor_map[layer_idx] = donor_idx
        self._kv_donor_indices = set(self._kv_donor_map.values())

    # ====================================================================
    # get_model_output override: compute PLE and pass to decoder layers
    # ====================================================================

    def get_model_output(
        self,
        input_ids,
        seq_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds=None,
        prev_hidden=None,
        adapter_ids=None,
        rotary_position_ids=None,
        update_cache=False,
        is_for_context_encoding=False,
        vision_embeddings=None,
        vision_mask=None,
        deepstack_vision_embeds=None,
        local_attn_mask=None,
        windowed_context_encoding_window_idx=-1,
        padding_mask=None,
        **kwargs,
    ):
        """
        Override base class to compute PLE and pass it explicitly to decoder layers.

        After computing inputs_embeds, this computes PLE via self.ple_module
        and passes per_layer_inputs=ple_tensor in kwargs to each decoder layer.
        """
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # === E2B-specific: compute PLE ===
        per_layer_inputs = self.ple_module(input_ids, inputs_embeds)

        if (vision_embeddings is not None) and (vision_mask is not None):
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(
                    self.config.neuron_config.torch_dtype
                )
            if is_for_context_encoding:
                inputs_embeds = self.encode_vision_to_input(
                    inputs_embeds, vision_embeddings, vision_mask
                )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.sequence_parallel_enabled:
            self.validate_sequence_parallel(seq_length)

        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        update_kv_per_layer = update_cache and (
            self.neuron_config.layer_boundary_markers
            or (
                self.neuron_config.attn_block_tkg_nki_kernel_cache_update
                and not is_for_context_encoding
            )
        )

        # decoder layers
        next_decoder_cache = [] if update_kv_per_layer else ()
        cos_cache = None
        sin_cache = None

        cache_size = self.n_positions
        if self.attention_chunk_size:
            cache_size = self.attention_chunk_size
        elif self.sliding_window:
            cache_size = self.sliding_window
        get_kv_per_layer = False
        active_block_table = kwargs.get("active_block_table", None)
        empty_active_block_table = (
            True if active_block_table is None else len(active_block_table.shape) == 1
        )
        may_have_prefix = (
            self.is_prefix_caching
            and is_for_context_encoding
            and not empty_active_block_table
        )
        if (
            may_have_prefix
            or not is_for_context_encoding
            or windowed_context_encoding_window_idx >= 1
        ):
            if not self.config.neuron_config.layer_boundary_markers:
                past_key_values = self.kv_mgr.get_cache(
                    seq_ids=seq_ids,
                    seq_len=cache_size,
                    is_for_context_encoding=is_for_context_encoding,
                    windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                    **kwargs,
                )
            else:
                get_kv_per_layer = True

        residual = None
        # KV sharing: track donor layers' fresh K/V output for shared layers
        donor_kv = {}  # {donor_layer_idx: (K, V)} fresh K/V from donor layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # KV sharing: if this is a shared layer, pass donor's fresh K/V
            extra_kv_kwargs = {}
            donor_idx = self._kv_donor_map.get(idx)
            if donor_idx is not None and donor_idx in donor_kv:
                extra_kv_kwargs["shared_kv"] = donor_kv[donor_idx]

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=get_kv_per_layer,
                update_kv_per_layer=update_kv_per_layer,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=residual,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                # === E2B-specific: pass PLE to each decoder layer ===
                per_layer_inputs=per_layer_inputs,
                **extra_kv_kwargs,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]

            # KV sharing: save donor layer's fresh K/V for shared layers
            # Only save for layers that are actually donors (i.e., referenced
            # in _kv_donor_map). kv is (K, V) tuple from attention.
            if idx in self._kv_donor_indices:
                donor_kv[idx] = kv

            if update_kv_per_layer:
                next_decoder_cache += kv
            else:
                next_decoder_cache += (kv,)
            cos_cache, sin_cache = layer_outputs[2:4]
            residual = layer_outputs[4]

        if update_cache and not update_kv_per_layer:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        self.full_hidden_states = None
        if self.neuron_config.enable_eagle_speculation:
            self.full_hidden_states = hidden_states

        return hidden_states, next_decoder_cache

    # ====================================================================
    # Mask overrides (same pattern as 31B, adapted for 512 SWA window)
    # ====================================================================

    def _create_windowed_attn_mask_tkg(self, attention_mask, window_size, position_ids):
        """Override: SWA TKG mask must match the uniform KV cache size.

        The base class creates a mask of shape (B, 1, 1, window_size), but our
        KV caches use _uniform_cache_len = max(sliding_window, max_length).
        When max_length > sliding_window, the SWA cache has extra slots beyond
        the window that must be masked out.
        """
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

        # Pad to cache_len with False (masked out)
        pad_len = cache_len - window_size
        padded_mask = F.pad(window_mask, (0, pad_len), value=False)

        return padded_mask[:, None, None, :]

    def _create_simple_attn_mask(self, attention_mask):
        """Override: global (non-SWA) mask must match uniform KV cache size."""
        batch_size = attention_mask.shape[0]
        pad_len = self._uniform_cache_len - self.n_positions
        if pad_len > 0:
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        return (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, 1, self._uniform_cache_len)
            .to(torch.bool)
        )

    # ====================================================================
    # Vision scatter (same as 31B, kept for multimodal compatibility)
    # ====================================================================

    def scatter_by_index_put(self, h_image, encoded_patches_proj, positions):
        """Scatter encoded vision patches into the text embedding tensor.

        Uses torch.scatter_ instead of index_put_ to generate simpler HLO
        that avoids NCC compiler bug NCC_ITEN404 with E2B's heterogeneous layers.
        """
        B, max_positions, embedding_dim = h_image.shape
        h_image_new = h_image.clone()

        # positions shape: [B, num_vision_tokens, 1] or [B*num_vision_tokens]
        # encoded_patches_proj shape: [B, num_vision_tokens, embedding_dim] or similar
        positions_flat = positions.view(B, -1)  # [B, num_vision_tokens]
        num_vision_tokens = positions_flat.shape[1]

        # Expand positions for scatter: [B, num_vision_tokens, embedding_dim]
        scatter_idx = (
            positions_flat.unsqueeze(-1)
            .expand(B, num_vision_tokens, embedding_dim)
            .long()
        )

        # Ensure vision embeddings are [B, num_vision_tokens, embedding_dim]
        vision_flat = encoded_patches_proj.view(B, num_vision_tokens, embedding_dim)

        h_image_new.scatter_(1, scatter_idx, vision_flat)
        return h_image_new

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        """Merge vision embeddings into text embeddings during context encoding."""
        return self.scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def init_inference_optimization(self, config: Gemma4E2BInferenceConfig):
        """
        Override to create per-layer KV caches with correct heterogeneous shapes.

        E2B has:
        - SWA layers: num_kv_heads=1, head_dim=256
        - Global layers: num_kv_heads=1, head_dim=512

        Per-rank after TP sharding:
        - All layers: kv_heads_per_rank depends on GQA replication strategy
        """
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

        # Compute per-layer KV head counts and head dims
        tp_degree = config.neuron_config.tp_degree
        layer_kv_configs = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            if layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads  # 1
                hd = config.head_dim  # 256
            else:
                kv_heads = config.num_key_value_heads  # 1 (same for E2B)
                hd = config.global_head_dim  # 512

            # Compute sharded KV heads per rank
            gqa_strategy = determine_sharding_strategy(tp_degree, kv_heads)
            _, shardable_kv_heads = get_shardable_head_counts(
                tp_degree, config.num_attention_heads, kv_heads, gqa_strategy
            )
            kv_heads_per_rank = shardable_kv_heads // tp_degree
            layer_kv_configs.append((kv_heads_per_rank, hd))

        # Store per-layer configs for attention modules to use
        self._layer_kv_configs = layer_kv_configs

        # Create KVCacheManager with per-layer shapes
        self.kv_mgr = Gemma4E2BKVCacheManager(
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


class NeuronGemma4E2BForCausalLM(NeuronBaseForCausalLM):
    """
    Gemma4 E2B causal LM for NeuronX inference.
    Handles weight loading, state dict conversion, and tied weights.
    """

    _model_cls = NeuronGemma4E2BTextModel

    # Note: E2B uses default compiler args (auto-cast=none with f32 or bf16 dtype).
    # Do NOT use --auto-cast=matmult with f32 dtype -- it downcasts matmuls to bf16.

    def get_compiler_args(self) -> str:
        """Return compiler args. Uses default (auto-cast=none -O1)."""
        return None

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        # Gemma4 E2B uses Gemma4ForConditionalGeneration (model_type=gemma4)
        try:
            from transformers import Gemma4ForConditionalGeneration

            return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)
        except (ImportError, OSError):
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: Gemma4E2BInferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert HuggingFace Gemma4 E2B state dict to NeuronX format.

        Key transformations:
        1. Strip 'language_model.model.' / 'language_model.' prefixes
        2. Remap embed_tokens -> embed_tokens.embedding (for Gemma4ScaledEmbedding)
        3. Remap PLE keys into ple_module.* namespace (separate PLE module)
        4. Remap q_norm/k_norm -> q_layernorm/k_layernorm
        5. Fuse QK scaling correction into Q norm weights (cancel NxDI's 1/sqrt(head_dim))
        6. Add rank_util tensors for TP
        7. No k_eq_v handling (E2B has separate K/V for all layers)
        8. No v_norm handling (E2B has no v_norm)
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}

        for key, weight in state_dict.items():
            new_key = key

            # Strip HF prefixes (handle multiple nesting patterns)
            if new_key.startswith("language_model.model."):
                new_key = new_key[len("language_model.model.") :]
            elif new_key.startswith("language_model."):
                new_key = new_key[len("language_model.") :]
            elif new_key.startswith("model.language_model.model."):
                new_key = new_key[len("model.language_model.model.") :]
            elif new_key.startswith("model.language_model."):
                new_key = new_key[len("model.language_model.") :]

            # Skip vision/audio tower weights (text-only for now)
            if (
                "vision_tower." in new_key
                or "multi_modal_projector." in new_key
                or "embed_vision" in new_key
                or "audio_tower." in new_key
                or "embed_audio" in new_key
            ):
                continue

            # Remap main embedding key: now inside combined embedding wrapper
            # HF: embed_tokens.weight -> Neuron: embed_tokens.embedding.weight
            if new_key == "embed_tokens.weight":
                new_key = "embed_tokens.embedding.weight"

            # Remap PLE embedding: into separate ple_module
            # HF: embed_tokens_per_layer.weight -> Neuron: ple_module.embed_tokens_per_layer.weight
            if new_key == "embed_tokens_per_layer.weight":
                new_key = "ple_module.embed_tokens_per_layer.weight"

            # Remap PLE model-level components into ple_module
            # HF: per_layer_model_projection.weight -> Neuron: ple_module.per_layer_model_projection.weight
            if new_key == "per_layer_model_projection.weight":
                new_key = "ple_module.per_layer_model_projection.weight"
            if new_key == "per_layer_projection_norm.weight":
                new_key = "ple_module.per_layer_projection_norm.weight"

            # Remap QK norm keys
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            # NOTE: Do NOT remap q/k/v/o_proj keys here.
            # NxDI's GroupQueryAttention_QKV and GroupQueryAttention_O modules have
            # preshard_hooks that automatically remap:
            #   self_attn.q_proj -> self_attn.qkv_proj.q_proj
            #   self_attn.k_proj -> self_attn.qkv_proj.k_proj
            #   self_attn.v_proj -> self_attn.qkv_proj.v_proj
            #   self_attn.o_proj -> self_attn.o_proj.o_proj
            # Doing it here would cause double-remapping (preshard_hook uses substring
            # replace, so pre-remapped keys get triple-nested).

            new_state_dict[new_key] = weight.detach().clone()

        # Cast PLE embedding to bf16 (model stores it in bf16 to save HBM)
        ple_embed_key = "ple_module.embed_tokens_per_layer.weight"
        if ple_embed_key in new_state_dict:
            new_state_dict[ple_embed_key] = new_state_dict[ple_embed_key].to(
                torch.bfloat16
            )

        # Per-layer transformations
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            is_global = layer_type == "full_attention"

            if is_global:
                hd = config.global_head_dim  # 512
            else:
                hd = config.head_dim  # 256

            prefix = f"layers.{i}.self_attn"

            # --- QK scaling correction ---
            # Gemma4 uses scaling=1.0 (no 1/sqrt(head_dim) in attention scores).
            # NxDI always applies 1/sqrt(head_dim) in scaled_qk.
            # We scale the q_layernorm WEIGHTS by sqrt(head_dim) to cancel:
            #   Q_normed = Q / rms(Q) * (w * sqrt(hd)) = RMSNorm(Q) * sqrt(hd)
            #   NxDI: (Q_normed @ K^T) / sqrt(hd) = RMSNorm(Q) @ K^T  (matches HF scaling=1.0)
            q_norm_key = f"{prefix}.q_layernorm.weight"
            if q_norm_key in new_state_dict:
                scaling_factor = math.sqrt(float(hd))
                orig_dtype = new_state_dict[q_norm_key].dtype
                new_state_dict[q_norm_key] = (
                    new_state_dict[q_norm_key].to(torch.float32) * scaling_factor
                ).to(orig_dtype)

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
        return Gemma4E2BInferenceConfig
