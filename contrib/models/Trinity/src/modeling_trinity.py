#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
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
Unified NeuronX Distributed Inference implementation for the Trinity model family
(AfmoeForCausalLM) from Arcee AI.

Supports all three Trinity sizes from a single codebase:
- Trinity-Nano-Preview (~6B total, ~1B active)
- Trinity-Mini (~26B total, ~4.5B active)
- Trinity-Large-Preview (~250B total, ~15B active)

Architecture (shared across all sizes):
- AfmoeForCausalLM: Arcee Foundation Mixture of Experts
- Mixed attention: sliding_attention + full_attention (every 4th layer)
- Gated attention: gate_proj + sigmoid on attention output
- QK normalization: RMSNorm on Q and K per head
- Dual layer norms: pre/post for both attention and MLP (4 per layer)
- muP scaling: hidden_size**0.5 on input embeddings
- Sigmoid routing with normalization
- SiLU gated MLP (gate_proj, up_proj, down_proj)
- Expert bias on routing scores

Key porting decisions:
- glu_type="glu" (NOT "swiglu") -- Trinity uses SiLU(gate)*up, which is NxDI's "glu"
- route_scale baked into routed expert down_proj weights (NxDI MoE v2 doesn't support it)
- muP scaling baked into embedding weights during conversion
- expert_bias handled via custom RouterTopKWithBias subclass
- Gated attention handled via inline override of attention forward methods
- Gate weight padding uses interleaved layout matching Q projection (for high TP)
"""

import json
import os
import math
import logging
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.kvcache.utils import (
    dynamic_update_slice,
    fill_prefix,
)
from neuronx_distributed_inference.modules.generation.sampling import Sampler

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers import utils as nxd_utils
from neuronx_distributed.utils import cpu_mode

logger = logging.getLogger(__name__)

# MoE v2 module (required for MoE layers)
try:
    from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
    from neuronx_distributed.modules.moe.routing import RouterTopK

    MOE_V2_AVAILABLE = True
except ImportError:
    MOE_V2_AVAILABLE = False
    logger.warning("moe_v2 not available, MoE layers will not work")


def _patch_fused_tkg_for_sigmoid():
    """Patch MoEFusedTKG kernel to use ISA router fallback for sigmoid routing.

    The SDK 2.28 fused MoE TKG NKI kernel's router_topk_kernel_nki only supports
    softmax activation. Trinity uses sigmoid routing. The kernel also has an ISA
    router fallback (router_topk_isa_kernel) that supports both sigmoid and softmax.

    This patch wraps the selective-load kernel call to force
    use_router_topk_nki_kernel=False, which uses the ISA router fallback.

    NOTE: The fused TKG kernel supports expert_bias when using the patched
    nki-library, neuronx-distributed, and neuronx-distributed-inference
    libraries from the feature/expert-bias-support branches. The ISA router
    fallback is still needed for sigmoid routing (NKI kernel only supports
    softmax), but expert_bias is correctly passed through and applied in the
    ISA path.

    Must be called before model.compile().
    """
    try:
        import neuronx_distributed.modules.moe.moe_fused_tkg as fused_tkg_mod

        original_kernel = fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call
        if original_kernel is None:
            logger.warning(
                "Fused TKG selective load kernel not available, skipping patch"
            )
            return

        # The NKI kernel call object supports [grid](**kwargs) invocation.
        # We wrap it to inject use_router_topk_nki_kernel=False.
        class _PatchedKernelCall:
            """Wrapper that injects use_router_topk_nki_kernel=False into kernel calls."""

            def __init__(self, original):
                self._original = original

            def __getitem__(self, grid):
                original_grid_call = self._original[grid]

                def patched_call(*args, **kwargs):
                    kwargs["use_router_topk_nki_kernel"] = False
                    return original_grid_call(*args, **kwargs)

                return patched_call

        fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call = (
            _PatchedKernelCall(original_kernel)
        )

        # Also patch the forward-all-experts kernel if it has the same issue
        original_all = fused_tkg_mod._moe_tkg_forward_all_experts_nki_call
        if original_all is not None:
            fused_tkg_mod._moe_tkg_forward_all_experts_nki_call = _PatchedKernelCall(
                original_all
            )

        logger.warning(
            "Patched MoEFusedTKG for sigmoid routing (ISA fallback). "
            "expert_bias is supported via patched libraries (feature/expert-bias-support branches)."
        )
    except ImportError:
        logger.info("moe_fused_tkg module not available (SDK < 2.28), skipping patch")
    except Exception as e:
        logger.warning("Failed to patch MoEFusedTKG for sigmoid: %s", e)


# ---------------------------------------------------------------------------
# TrinityKVCacheManager: Per-layer KV cache sizing for mixed attention
# ---------------------------------------------------------------------------
# Adapted from GptOssKVCacheManager in NxDI.  Trinity has mixed attention:
# most layers use sliding-window attention (KV cache = sliding_window - 1),
# while every 4th layer uses full attention (KV cache = max_length).
#
# The standard KVCacheManager applies a single sliding_window modulation to
# ALL layers in _get_index_to_update_new_position, which causes OOB when
# full-attention layers have larger KV cache buffers.  This custom manager
# creates per-layer cache buffers and applies per-layer scatter modulation.
# ---------------------------------------------------------------------------


def _slice_kv_cacheline(padding_side, seq_len, cache, transposed):
    """Slice KV cache to seq_len along the sequence dimension."""
    seqlen_dim = 3 if transposed else 2
    if padding_side == "right":
        return torch.ops.aten.slice(cache, dim=seqlen_dim, start=0, end=seq_len)
    max_idx = cache.shape[seqlen_dim]
    return torch.ops.aten.slice(
        cache, dim=seqlen_dim, start=max_idx - seq_len, end=max_idx
    )


class TrinityKVCacheManager(nn.Module):
    """Per-layer KV cache manager for Trinity's mixed attention.

    Sliding-window layers get a smaller KV cache (sliding_window - 1 positions).
    Full-attention layers get the full max_length cache.  Each layer's scatter
    index is modulated correctly to stay within its own buffer.
    """

    def __init__(
        self, config, num_kv_head, layer_types, sliding_window, global_rank=None
    ):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.padding_side = config.neuron_config.padding_side
        self.is_continuous_batching = config.neuron_config.is_continuous_batching
        self.num_kv_head = num_kv_head
        self.batch_size = config.neuron_config.max_batch_size
        self.k_cache_transposed = config.neuron_config.k_cache_transposed
        self.global_rank = global_rank
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.tp_degree = config.neuron_config.tp_degree
        self.dtype = (
            config.neuron_config.attention_dtype
            if config.neuron_config.attention_dtype is not None
            else config.neuron_config.torch_dtype
        )

        # Per-layer attention type
        self.layer_types = layer_types
        # Use sliding_window directly as the cache size.  GptOss uses
        # sliding_window - 1 because its attention kernel convention differs,
        # but Trinity's windowed_attention_forward creates TKG masks of size
        # sliding_window, so the cache must match exactly.
        self.sliding_window = sliding_window

        self._init_kv_shape()

        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(shape, dtype=self.dtype), requires_grad=False)
                for layer_idx in range(self.num_layers)
                for shape in [self.k_shapes[layer_idx], self.v_shapes[layer_idx]]
            ]
        )

    def _get_num_kv_heads_per_rank(self):
        gqa_sharding_strategy = determine_sharding_strategy(
            self.tp_degree, self.num_kv_head
        )
        _, num_key_value_heads = get_shardable_head_counts(
            self.tp_degree,
            self.num_attention_heads,
            self.num_kv_head,
            gqa_sharding_strategy,
        )
        if parallel_state.model_parallel_is_initialized():
            return nxd_utils.divide(num_key_value_heads, self.tp_degree)
        return num_key_value_heads

    def _init_kv_shape(self):
        self.k_shapes = []
        self.v_shapes = []
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank()
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        max_length = self.config.neuron_config.max_length

        # All layers get max_length cache.  During CTE, fill_prefix writes
        # the full context (up to max_length tokens) into the cache.  A
        # smaller sliding-window cache would cause OOB when the CTE bucket
        # exceeds sliding_window.  The sliding-window optimization is applied
        # only at TKG time via _get_index_to_update_new_position (wrapping
        # position_ids modulo sliding_window for sliding layers) and via
        # get_kv_by_layer_id (slicing the cache to sliding_window during read).
        for layer_idx in range(self.num_layers):
            shape = (self.batch_size, num_kv_heads_per_rank, max_length, head_dim)
            self.k_shapes.append(shape)
            self.v_shapes.append(shape)

    def _fetch_cache(self, idx, kvcache_buffer=None):
        if kvcache_buffer is not None:
            if (
                len(kvcache_buffer) == len(self.past_key_values) // 2
                and len(kvcache_buffer[0]) == 2
            ):
                return kvcache_buffer[idx][0], kvcache_buffer[idx][1]
            elif len(kvcache_buffer) == len(self.past_key_values):
                return kvcache_buffer[2 * idx], kvcache_buffer[2 * idx + 1]
            else:
                raise ValueError(
                    f"kvcache_buffer length {len(kvcache_buffer)} not recognized"
                )
        return self.past_key_values[2 * idx], self.past_key_values[2 * idx + 1]

    def get_kv_by_layer_id(
        self,
        idx,
        seq_len,
        skip_slice=False,
        kvcache_buffer=None,
        **kwargs,
    ):
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        # Override seq_len with the per-layer effective size:
        # - Sliding layers: min(sliding_window, max_length) — only read the
        #   last sliding_window positions during TKG.
        # - Full-attention layers: max_length — read everything.
        # During CTE, seq_len from the caller equals n_positions (the bucket);
        # we still override to ensure the slice matches what attention expects.
        if hasattr(self, "v_shapes"):
            is_sliding = self.layer_types[idx] == "sliding_attention"
            max_len = self.v_shapes[idx][2]  # always max_length
            if is_sliding and self.sliding_window and self.sliding_window < max_len:
                seq_len = self.sliding_window
            else:
                seq_len = max_len

        if not skip_slice:
            k_cache = _slice_kv_cacheline(
                self.padding_side, seq_len, k_cache, self.k_cache_transposed
            )
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache, False)
        return k_cache, v_cache

    def get_cache(self, seq_len, skip_slice=False, kvcache_buffer=None, **kwargs):
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            k_cache, v_cache = self.get_kv_by_layer_id(
                idx=idx,
                seq_len=seq_len,
                skip_slice=skip_slice,
                kvcache_buffer=kvcache_buffer,
                **kwargs,
            )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding,
        seq_ids,
        position_ids,
        new_key_values,
        seq_len,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            k_cache, v_cache = self.update_kv_by_layer_id(
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=kv_per_layer,
                seq_len=seq_len,
                scatter_index=scatter_index,
                kv_active_mask=kv_active_mask,
                kvcache_buffer=kvcache_buffer,
            )
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)
        return updated_kv_cache

    def update_kv_by_layer_id(
        self,
        idx,
        is_for_context_encoding,
        seq_ids,
        position_ids,
        kv_per_layer,
        seq_len,
        scatter_index=None,
        kv_active_mask=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

        if is_for_context_encoding:
            if self.is_continuous_batching:
                assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1
                if self.k_cache_transposed:
                    cache_idx = seq_ids
                    indices = [cache_idx] + [
                        torch.zeros(1, device=seq_ids.device)
                        for _ in range(k_cache.dim() - 1)
                    ]
                    indices = [t.squeeze().to(torch.int32) for t in indices]
                    k_cache = dynamic_update_slice(k_cache, latest_k, indices)
                    v_cache = dynamic_update_slice(v_cache, latest_v, indices)
                else:
                    from neuronx_distributed_inference.modules.kvcache.utils import (
                        update_cache_const_indices,
                    )

                    k_cache = update_cache_const_indices(k_cache, latest_k, seq_ids)
                    v_cache = update_cache_const_indices(v_cache, latest_v, seq_ids)
            else:
                k_cache = fill_prefix(k_cache, latest_k)
                v_cache = fill_prefix(v_cache, latest_v)
        else:
            # Token generation: scatter new KV into the correct position.
            # Per-layer modulation keeps indices within the buffer bounds.
            scatter_index_k = self._get_index_to_update_new_position(
                scatter_index, position_ids, latest_k, self.k_cache_transposed, idx
            )
            scatter_index_v = self._get_index_to_update_new_position(
                scatter_index, position_ids, latest_v, False, idx
            )
            k_cache = torch.scatter(
                input=k_cache,
                dim=(2 if not self.k_cache_transposed else 3),
                index=scatter_index_k,
                src=latest_k,
            )
            v_cache = torch.scatter(
                input=v_cache, dim=2, index=scatter_index_v, src=latest_v
            )
        return k_cache, v_cache

    def _get_index_to_update_new_position(
        self, scatter_index, position_ids, full_k, transposed, layer_idx
    ):
        """Per-layer scatter index modulation.

        Sliding-window layers: position_ids % sliding_window  (wraps within window)
        Full-attention layers: position_ids as-is              (no modulation needed)
        """
        is_sliding = self.layer_types[layer_idx] == "sliding_attention"
        if is_sliding and self.sliding_window:
            position_ids = position_ids % self.sliding_window
        index = position_ids
        view_shape = (
            (-1, 1, index.shape[-1], 1)
            if not transposed
            else (-1, 1, 1, index.shape[-1])
        )
        return index.view(*view_shape).expand_as(full_k)


class RouterTopKWithBias(RouterTopK):
    """RouterTopK with expert_bias support for Trinity.

    Trinity uses expert_bias to influence which experts are selected:
    - Sigmoid scores are computed: scores = sigmoid(logits)
    - For top-k selection: topk(scores + expert_bias)
    - For actual routing weights: gather scores at selected indices (no bias)

    The bias only affects WHICH experts are selected, not their weights.
    """

    def __init__(self, expert_bias_size, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "expert_bias",
            torch.zeros(expert_bias_size, dtype=torch.float32),
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        # Top-k selection with expert_bias added to scores.
        scores_for_selection = expert_affinities.float() + self.expert_bias.float()
        _, expert_index = torch.topk(scores_for_selection, self.top_k)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def initialize_moe_with_expert_bias(config, init_tkg_module=False, rmsnorm=None):
    """Initialize MoE module with expert_bias support.

    Args:
        config: TrinityInferenceConfig
        init_tkg_module: If True, enable fused MoE TKG NKI kernel path.
            Requires SDK 2.28+ and moe_intermediate_size/tp % 128 == 0.
        rmsnorm: RMSNorm module to fuse into MoE (required when init_tkg_module=True).
            The fused kernel applies this norm internally, so the caller must skip it.
    """
    if init_tkg_module:
        try:
            moe = initialize_moe_module(
                config=config, init_tkg_module=True, rmsnorm=rmsnorm
            )
        except TypeError:
            # SDK 2.27 or older: initialize_moe_module doesn't accept these args
            logger.warning(
                "Fused MoE TKG not supported by this SDK version. "
                "Falling back to standard path."
            )
            moe = initialize_moe_module(config=config)
    else:
        moe = initialize_moe_module(config=config)

    old_router = moe.router
    new_router = RouterTopKWithBias(
        expert_bias_size=config.num_local_experts,
        num_experts=old_router.num_experts,
        top_k=old_router.top_k,
        hidden_size=old_router.hidden_size,
        dtype=old_router.dtype,
        device=old_router.device,
        act_fn=old_router.act_fn,
        sequence_parallel_enabled=old_router.sequence_parallel_enabled,
        sequence_dimension=old_router.sequence_dimension,
        bias=old_router.bias,
        apply_act_fn_over_topk=old_router.apply_act_fn_over_topk,
        store_transposed_weights=old_router.store_transposed_weights,
    )
    new_router.linear_router = old_router.linear_router
    if hasattr(old_router, "weight_T"):
        new_router.weight_T = old_router.weight_T

    moe.router = new_router
    moe.eval()
    return moe


def get_rmsnorm_cls():
    """Get the appropriate RMSNorm class based on execution mode."""
    if cpu_mode():

        class StandardRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                return (self.weight * hidden_states).to(input_dtype)

        return StandardRMSNorm
    else:
        return CustomRMSNorm


class TrinityInferenceConfig(InferenceConfig):
    """Configuration for Trinity (AfmoeForCausalLM) inference.

    Handles all Trinity model sizes (Nano, Mini, Large) via config-driven values.

    IMPORTANT: initialize_moe_module reads config.intermediate_size for expert MLP
    dimensions. Trinity has two different intermediate sizes:
    - intermediate_size: used for dense MLP layers (first num_dense_layers)
    - moe_intermediate_size: used for MoE expert MLPs

    We store the dense size as dense_intermediate_size and set intermediate_size to
    moe_intermediate_size so that initialize_moe_module gets the correct value.
    """

    def __init__(self, neuron_config=None, **kwargs):
        # Model architecture parameters from AfmoeConfig
        self.vocab_size = kwargs.pop("vocab_size", 200192)
        self.hidden_size = kwargs.pop("hidden_size", 2048)

        # CRITICAL: intermediate_size must be the MoE intermediate size for initialize_moe_module
        dense_intermediate = kwargs.pop("intermediate_size", 6144)
        moe_intermediate = kwargs.pop("moe_intermediate_size", 1024)
        self.dense_intermediate_size = dense_intermediate
        self.intermediate_size = moe_intermediate
        self.moe_intermediate_size = moe_intermediate

        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 32)
        self.num_dense_layers = kwargs.pop("num_dense_layers", 2)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 4)
        self.head_dim = kwargs.pop("head_dim", 128)
        self.hidden_act = kwargs.pop("hidden_act", "silu")
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 131072)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-5)
        self.use_cache = kwargs.pop("use_cache", True)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)

        # MoE parameters
        self.num_experts = kwargs.pop("num_experts", 128)
        self.num_local_experts = kwargs.pop("num_local_experts", None)
        if self.num_local_experts is None:
            self.num_local_experts = self.num_experts
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", 8)
        self.num_shared_experts = kwargs.pop("num_shared_experts", 1)
        # IMPORTANT: Set n_shared_experts=0 for initialize_moe_module so the NxDI MoE
        # module does NOT create its own SharedExperts. We handle shared experts ourselves
        # in NeuronTrinityDecoderLayer to ensure proper weight loading.
        self.n_shared_experts = 0
        self.num_expert_groups = kwargs.pop("num_expert_groups", 1)
        self.num_limited_groups = kwargs.pop("num_limited_groups", 1)
        self.score_func = kwargs.pop("score_func", "sigmoid")
        self.route_norm = kwargs.pop("route_norm", True)
        self.route_scale = kwargs.pop("route_scale", 1.0)
        self.n_group = kwargs.pop("n_group", 1)
        self.topk_group = kwargs.pop("topk_group", 1)
        self.load_balance_coeff = kwargs.pop("load_balance_coeff", 0.001)

        # Attention patterns
        self.global_attn_every_n_layers = kwargs.pop("global_attn_every_n_layers", 4)
        self.sliding_window = kwargs.pop("sliding_window", 2048)
        self.layer_types = kwargs.pop("layer_types", None)

        # Clamp sliding_window to seq_len if seq_len < sliding_window.
        # The KV cache is sized by seq_len (via n_positions), and sliding window
        # attention creates masks of size sliding_window. These must match.
        if neuron_config is not None and hasattr(neuron_config, "seq_len"):
            if neuron_config.seq_len < self.sliding_window:
                logger.info(
                    "Clamping sliding_window from %d to %d to match seq_len",
                    self.sliding_window,
                    neuron_config.seq_len,
                )
                self.sliding_window = neuron_config.seq_len

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if bool((i + 1) % self.global_attn_every_n_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # muP
        self.mup_enabled = kwargs.pop("mup_enabled", True)

        # Standard attributes
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.torch_dtype = kwargs.pop("torch_dtype", "bfloat16")
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)

        # Pop HF-specific keys not used by our config
        kwargs.pop("auto_map", None)
        kwargs.pop("architectures", None)
        kwargs.pop("model_type", None)
        kwargs.pop("transformers_version", None)
        kwargs.pop("dtype", None)
        kwargs.pop("use_grouped_mm", None)

        super().__init__(neuron_config=neuron_config, **kwargs)

        # Adjust num_local_experts for expert parallelism
        if hasattr(self, "neuron_config") and self.neuron_config is not None:
            ep_degree = getattr(self.neuron_config, "ep_degree", 1)
            if ep_degree > 1:
                self.num_local_experts = self.num_experts // ep_degree

        # Set MoE neuron config parameters
        if hasattr(self, "neuron_config") and self.neuron_config is not None:
            if not hasattr(self.neuron_config, "glu_mlp"):
                self.neuron_config.glu_mlp = True
            # Trinity uses SiLU(gate)*up which is NxDI's "glu" type,
            # NOT "swiglu" which computes gate*SiLU(gate)*up
            self.neuron_config.glu_type = "glu"
            # Trinity uses sigmoid routing (not softmax)
            if hasattr(self.neuron_config, "router_config"):
                self.neuron_config.router_config.act_fn = "sigmoid"

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1

        # Enable fused MoE TKG kernel if alignment constraint is met.
        # The fused NKI kernel requires intermediate_size_per_tp % 128 == 0.
        # This is auto-validated; if the constraint fails, we fall back to
        # the standard blockwise matmul path silently.
        self._enable_fused_moe_tkg()

    def _enable_fused_moe_tkg(self):
        """Check and enable fused MoE TKG NKI kernel (SDK 2.28+).

        The fused kernel combines RMSNorm + Router TopK + Expert MLP into a
        single NKI kernel launch, reducing HBM round-trips during token gen.

        Requires: moe_intermediate_size / moe_tp_degree % 128 == 0.
        """
        MOE_TKG_MK_INTERMEDIATE_PER_TP = 128
        if not hasattr(self, "neuron_config") or self.neuron_config is None:
            return

        # Check if user explicitly requested fused kernel
        fused_requested = getattr(
            self.neuron_config, "moe_fused_nki_kernel_enabled", None
        )
        if fused_requested is None:
            return  # Not requested, don't enable

        moe_tp = getattr(self.neuron_config, "moe_tp_degree", None)
        if moe_tp is None:
            moe_tp = getattr(self.neuron_config, "tp_degree", 1)

        i_per_tp = self.moe_intermediate_size // moe_tp
        if i_per_tp % MOE_TKG_MK_INTERMEDIATE_PER_TP != 0:
            logger.warning(
                "Cannot enable fused MoE TKG kernel: "
                "moe_intermediate_size/tp (%d/%d=%d) is not divisible by %d. "
                "Falling back to standard blockwise matmul path.",
                self.moe_intermediate_size,
                moe_tp,
                i_per_tp,
                MOE_TKG_MK_INTERMEDIATE_PER_TP,
            )
            self.neuron_config.moe_fused_nki_kernel_enabled = None
            self.moe_fused_nki_kernel_enabled = False
        else:
            self.moe_fused_nki_kernel_enabled = True
            logger.info(
                "Fused MoE TKG NKI kernel enabled (intermediate_per_tp=%d)", i_per_tp
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "num_local_experts",
            "num_experts_per_tok",
            "intermediate_size",
            "head_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "TrinityInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronTrinityAttention(NeuronAttentionBase):
    """Trinity attention with QK norms, conditional RoPE, and gated output.

    Key differences from standard attention:
    1. QK norms: RMSNorm applied to Q and K per head before attention
    2. Conditional RoPE: Only applied for sliding_attention layers, not full_attention
    3. Gated output: output = o_proj(attn_out * sigmoid(gate_proj(input)))

    Gating strategy (inline override):
    The Neuron tracer cannot follow tensor flow through mutable state, closures,
    or dynamic method replacement. The ONLY working approach is to have the gate
    computation INLINE in the same method that calls o_proj.

    We override standard_causal_attention_forward and windowed_attention_forward
    to insert gate_values = sigmoid(attn_gate_proj(original_hidden_states))
    and apply attn_output = attn_output * gate_values before the o_proj call.
    """

    def __init__(self, config: TrinityInferenceConfig, layer_idx: int):
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        # RoPE only for sliding attention layers
        if is_sliding:
            rotary_emb = RotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            rotary_emb = None

        # Set sliding_window on sliding layers so the base class dispatches to
        # windowed_attention_forward (which uses the local_mask from the
        # framework's mixed-attention flow).  Full-attention layers get None.
        sliding_window = config.sliding_window if is_sliding else None

        # Per-head QK norm
        rmsnorm_cls = get_rmsnorm_cls()
        q_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)
        k_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta if is_sliding else None,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_norm,
            k_layernorm=k_norm,
            sliding_window=sliding_window,
        )

        self.layer_idx = layer_idx
        self.is_sliding = is_sliding

        # Gated attention: gate_proj applied before o_proj.
        # Must match the actual per-rank attention output size from NxDI.
        # When num_attention_heads is not divisible by TP, NxDI pads to
        # ceil(num_heads/tp) heads per rank. We must match that padding.
        tp_degree = config.neuron_config.tp_degree
        heads_per_rank = math.ceil(config.num_attention_heads / tp_degree)
        padded_total_heads = heads_per_rank * tp_degree
        gate_output_size = padded_total_heads * config.head_dim

        self.attn_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            gate_output_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

    def _apply_gated_o_proj(self, attn_output, gate_hidden_states, adapter_ids=None):
        """Apply gating then o_proj, all inline for Neuron tracing.

        This method MUST be called from within the same forward pass where
        gate_hidden_states is a live tensor in the traced graph.
        """
        gate_values = torch.sigmoid(self.attn_gate_proj(gate_hidden_states))
        attn_output = attn_output * gate_values
        return self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

    def standard_causal_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        rotary_position_ids=None,
        kv_mgr=None,
        get_kv_per_layer=False,
        update_kv_per_layer=False,
        residual=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        """Override base class to insert gating before o_proj.

        Copied from NeuronAttentionBase.standard_causal_attention_forward (NxDI 0.8.0)
        with one change: the o_proj call is replaced with _apply_gated_o_proj.
        """
        from neuronx_distributed_inference.modules.attention.attention_base import (
            NeuronAttentionBaseOutput,
        )

        use_polar_compatible_rope = kwargs.get("use_polar_compatible_rope", False)

        # Save original hidden_states for gate computation BEFORE dtype conversion
        gate_hidden_states = hidden_states

        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch_dtype)
        seq_ids = kwargs.get("seq_ids")
        is_context_parallel = past_key_value is None and self.cp_degree > 1
        is_data_parallel = past_key_value is not None and self.dp_degree > 1
        if is_context_parallel:
            attention_mask, hidden_states, position_ids, cos_cache, sin_cache = (
                self._split_inputs_for_context_parallel(
                    attention_mask, hidden_states, position_ids, cos_cache, sin_cache
                )
            )

        if is_data_parallel:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                get_dp_rank,
                split_along_dim,
                get_data_parallel_attention_dp_group,
                gather_from_tensor_model_parallel_region_with_dim,
            )

            dp_rank = get_dp_rank(
                self.rank_util.get_rank(),
                self.tp_degree,
                self.dp_degree,
                self.neuron_config.switch_cc,
            )
            hidden_states = split_along_dim(
                hidden_states, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            attention_mask = split_along_dim(
                attention_mask, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None

        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False

        if self.neuron_config.is_prefix_caching:
            is_token_gen = is_token_gen and q_len < 128

        # NKI kernel paths -- delegate to base class (no custom gating in fused kernels)
        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            return super().standard_causal_attention_forward(
                gate_hidden_states.to(self.torch_dtype)
                if is_context_parallel or is_data_parallel
                else gate_hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

        if (
            self.attn_block_cte_nki_kernel_enabled
            and not is_token_gen
            and not self.neuron_config.is_prefix_caching
        ):
            return super().standard_causal_attention_forward(
                gate_hidden_states.to(self.torch_dtype)
                if is_context_parallel or is_data_parallel
                else gate_hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

        tkg_attn_kernel_fused_rope = (
            is_token_gen and self.attn_tkg_builtin_kernel_enabled
        )

        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        if is_token_gen:
            if tkg_attn_kernel_fused_rope:
                attn_output, K = self.attention_tokengen_kernel_builtin(
                    Q,
                    K,
                    V,
                    position_ids,
                    past_key_value,
                    attention_mask,
                    active_mask,
                    rotary_position_ids,
                )
            else:
                attn_output = self.attention_tokengen(
                    Q,
                    K,
                    V,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    active_mask,
                    **kwargs,
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # *** GATED ATTENTION: apply gate BEFORE o_proj, all inline ***
        attn_output = self._apply_gated_o_proj(
            attn_output, gate_hidden_states, adapter_ids=adapter_ids
        )

        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)

        kv = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        if is_context_parallel and not self.sequence_parallel_enabled:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                gather_from_tensor_model_parallel_region_with_dim,
                get_context_parallel_attention_cp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=1,
                process_group=get_context_parallel_attention_cp_group(),
            )

        if is_data_parallel:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                gather_from_tensor_model_parallel_region_with_dim,
                get_data_parallel_attention_dp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=0,
                process_group=get_data_parallel_attention_dp_group(),
            )

        attn_output = attn_output.to(original_dtype)

        return NeuronAttentionBaseOutput(
            attn_output, kv, cos_cache, sin_cache, residual
        )

    def windowed_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        rotary_position_ids=None,
        kv_mgr=None,
        get_kv_per_layer=False,
        update_kv_per_layer=False,
        residual=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        """Override base class to insert gating before o_proj.

        Copied from NeuronAttentionBase.windowed_attention_forward (NxDI 0.8.0)
        with one change: the o_proj call is replaced with _apply_gated_o_proj.
        """
        from neuronx_distributed_inference.modules.attention.attention_base import (
            NeuronAttentionBaseOutput,
            get_last_kv_window,
        )

        # Save original hidden_states for gate computation BEFORE any modifications
        gate_hidden_states = hidden_states

        is_context_parallel = past_key_value is None and self.cp_degree > 1
        is_data_parallel = past_key_value is not None and self.dp_degree > 1

        full_position_ids = position_ids.clone()

        if is_context_parallel:
            attention_mask, hidden_states, position_ids, cos_cache, sin_cache = (
                self._split_inputs_for_context_parallel(
                    attention_mask, hidden_states, position_ids, cos_cache, sin_cache
                )
            )

        if is_data_parallel:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                get_dp_rank,
                split_along_dim,
            )

            dp_rank = get_dp_rank(
                self.rank_util.get_rank(),
                self.tp_degree,
                self.dp_degree,
                self.neuron_config.switch_cc,
            )
            hidden_states = split_along_dim(
                hidden_states, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            attention_mask = split_along_dim(
                attention_mask, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None

        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False

        # NKI kernel path -- delegate to base class (no gating)
        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            return super().windowed_attention_forward(
                gate_hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

        tkg_attn_kernel_fused_rope = (
            is_token_gen and self.attn_tkg_builtin_kernel_enabled
        )

        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
        )

        if is_token_gen:
            attn_output = self.attention_tokengen(
                Q,
                K,
                V,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                **kwargs,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode_windowed_attention(
                Q,
                K,
                V,
                q_len,
                bsz,
                attention_mask,
                self.sliding_window,
                past_key_value,
                active_mask,
            )
            K, V = get_last_kv_window(
                self.sliding_window,
                full_position_ids,
                K,
                V,
                windowed_context_encoding_window_idx,
                self.neuron_config.speculation_length,
            )

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # *** GATED ATTENTION: apply gate BEFORE o_proj, all inline ***
        attn_output = self._apply_gated_o_proj(
            attn_output, gate_hidden_states, adapter_ids=adapter_ids
        )

        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)

        kv = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        if is_context_parallel and not self.sequence_parallel_enabled:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                gather_from_tensor_model_parallel_region_with_dim,
                get_context_parallel_attention_cp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=1,
                process_group=get_context_parallel_attention_cp_group(),
            )

        return NeuronAttentionBaseOutput(
            attn_output, kv, cos_cache, sin_cache, residual
        )


class NeuronTrinityMLP(nn.Module):
    """Dense MLP for non-MoE layers (first num_dense_layers layers).

    Uses dense_intermediate_size, NOT the MoE intermediate_size.
    """

    def __init__(self, config: TrinityInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        intermediate = config.dense_intermediate_size

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronTrinitySharedExpert(nn.Module):
    """Shared expert MLP for MoE layers.

    Trinity has num_shared_experts=1. Each MoE layer has a shared expert whose
    output is added to the routed expert output for every token. Uses the same
    SiLU-gated MLP architecture as the dense layers but with moe_intermediate_size.

    Implemented as a standalone module (separate from NxDI's MoE SharedExperts)
    to ensure reliable weight loading via standard ColumnParallelLinear/RowParallelLinear.
    """

    def __init__(self, config: TrinityInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        intermediate = config.moe_intermediate_size

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronTrinityDecoderLayer(nn.Module):
    """Trinity decoder layer with dual layer norms and conditional MoE.

    Structure:
    - input_layernorm -> attention -> post_attention_layernorm -> residual
    - pre_mlp_layernorm -> MLP/MoE -> post_mlp_layernorm -> residual
    """

    def __init__(self, config: TrinityInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = NeuronTrinityAttention(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_mlp_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE for layers >= num_dense_layers, dense MLP otherwise
        self.moe_enabled = layer_idx >= config.num_dense_layers
        self.moe_fused_tkg = getattr(config, "moe_fused_nki_kernel_enabled", False)
        if self.moe_enabled and MOE_V2_AVAILABLE:
            self.mlp = initialize_moe_with_expert_bias(
                config=config,
                init_tkg_module=self.moe_fused_tkg,
                # Pass rmsnorm=None so MoE's _forward_compute_bound does NOT
                # re-normalize during CTE (we normalize in the decoder forward).
                # For the TKG fused kernel, we provide a separate RMSNorm
                # instance below so the kernel can access gamma/eps.
                rmsnorm=None,
            )
            # For fused TKG: the kernel needs gamma/eps for its internal
            # RMSNorm. Since we passed rmsnorm=None above, we must provide
            # a separate (non-shared) RMSNorm instance on MoEFusedTKG.
            # This avoids the shared-module aliasing issue that corrupted
            # weight loading in the CTE path.
            if self.moe_fused_tkg and hasattr(self.mlp, "moe_fused_tkg"):
                fused_tkg = self.mlp.moe_fused_tkg
                if fused_tkg is not None:
                    moe_rmsnorm = rmsnorm_cls(
                        config.hidden_size, eps=config.rms_norm_eps
                    )
                    fused_tkg.post_attention_layernorm = moe_rmsnorm
            # Shared expert: handled outside NxDI MoE to ensure reliable weight loading
            if config.num_shared_experts > 0:
                self.shared_expert = NeuronTrinitySharedExpert(config)
            else:
                self.shared_expert = None
        else:
            self.mlp = NeuronTrinityMLP(config)
            self.shared_expert = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # Mixed attention mask selection (matches Llama4 pattern):
        # - Sliding layers use local_mask (windowed, sized to sliding_window)
        # - Full-attention layers use attention_mask (global, sized to n_positions;
        #   padded by apply_seq_ids_mask in compute_for_token_gen when KV cache
        #   is larger than the mask)
        local_mask = kwargs.pop("local_mask", None)
        if self.attention_type == "sliding_attention" and local_mask is not None:
            mask = local_mask
        else:
            mask = attention_mask

        attn_output, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=normed,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output

        # MLP with dual norms
        residual = hidden_states
        # Normalization strategy for fused MoE TKG:
        # - CTE (seq_len > 1): Decoder applies pre_mlp_layernorm.
        #   MoE's _forward_compute_bound skips norm (rmsnorm=None).
        # - TKG (seq_len == 1): Decoder skips pre_mlp_layernorm.
        #   Fused kernel applies norm internally using its own RMSNorm.
        #   MoEFusedTKG fallback also skips norm (post_attn_layernorm
        #   handles it when kernel is disabled).
        # When fused TKG is not enabled, decoder always applies norm.
        is_tkg = self.moe_fused_tkg and hidden_states.shape[1] == 1
        if not is_tkg:
            hidden_states = self.pre_mlp_layernorm(hidden_states)

        if self.moe_enabled and MOE_V2_AVAILABLE:
            mlp_output = self.mlp(hidden_states, padding_mask)[0]
            # Add shared expert output (applied to every token)
            if self.shared_expert is not None:
                # In TKG mode, hidden_states is un-normed (fused kernel
                # handles norm internally). Shared expert needs normed input.
                shared_input = (
                    self.pre_mlp_layernorm(hidden_states) if is_tkg else hidden_states
                )
                shared_output = self.shared_expert(shared_input)
                mlp_output = mlp_output + shared_output
            hidden_states = mlp_output
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronTrinityModel(NeuronBaseModel):
    """NeuronX Trinity base model (all sizes)."""

    def setup_attr_for_model(self, config: TrinityInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = getattr(config.neuron_config, "buckets", None)

        # Mixed attention: set has_mixed_attn so the framework creates both
        # global and local masks.  Set self.sliding_window so the framework
        # generates local_attn_mask via _create_windowed_attn_mask_tkg.
        # Full-attention layers use the global mask (padded by apply_seq_ids_mask
        # in compute_for_token_gen when KV cache > mask size).
        # Sliding layers use local_mask (sized to sliding_window).
        self.sliding_window = getattr(config, "sliding_window", None)
        self.has_mixed_attn = True

        # Store layer_types and raw sliding_window for the custom KV cache manager.
        self._layer_types = config.layer_types
        self._config_sliding_window = getattr(config, "sliding_window", None)

        # Patch fused MoE TKG kernel for sigmoid routing (must happen before compile).
        # Trinity uses sigmoid routing but the fused NKI kernel's router_topk_kernel_nki
        # only supports softmax. This forces the ISA router fallback which supports both.
        if getattr(config, "moe_fused_nki_kernel_enabled", False):
            _patch_fused_tkg_for_sigmoid()

    def init_inference_optimization(self, config: TrinityInferenceConfig):
        """Override to use TrinityKVCacheManager for per-layer KV cache sizing.

        The standard KVCacheManager cannot handle mixed attention with bucketing:
        its _get_index_to_update_new_position applies uniform sliding_window
        modulation to ALL layers, causing OOB for full-attention layers.

        TrinityKVCacheManager creates per-layer cache buffers and applies
        per-layer scatter modulation (sliding layers: position % window,
        full-attention layers: no modulation).
        """
        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        self.kv_mgr = TrinityKVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            layer_types=self._layer_types,
            sliding_window=self._config_sliding_window,
            global_rank=self.rank_util,
        )

    def init_model(self, config: TrinityInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.mup_enabled = getattr(config, "mup_enabled", False)
        self.mup_scale = math.sqrt(config.hidden_size) if self.mup_enabled else 1.0

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList(
            [
                NeuronTrinityDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)

        # Pad vocab_size to be divisible by TP degree for ColumnParallelLinear
        tp_degree = config.neuron_config.tp_degree
        padded_vocab = config.vocab_size
        if padded_vocab % tp_degree != 0:
            padded_vocab = ((padded_vocab // tp_degree) + 1) * tp_degree
        self.padded_vocab_size = padded_vocab
        self.actual_vocab_size = config.vocab_size

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            padded_vocab,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )


class NeuronTrinityForCausalLM(NeuronBaseForCausalLM):
    """NeuronX wrapper for Trinity causal language models (all sizes).

    Supports:
    - arcee-ai/Trinity-Nano-Preview (~6B total, ~1B active)
    - arcee-ai/Trinity-Mini (~26B total, ~4.5B active)
    - arcee-ai/Trinity-Large-Preview (~250B total, ~15B active)
    """

    _model_cls = NeuronTrinityModel

    @classmethod
    def get_config_cls(cls):
        return TrinityInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """Convert HuggingFace AfmoeForCausalLM state dict to NeuronX format.

        Key transformations:
        1. Remove 'model.' prefix from HF keys
        2. Rename QK norms: q_norm -> q_layernorm, k_norm -> k_layernorm
        3. Map attention gate_proj to attn_gate_proj (gated attention)
        4. Stack per-expert weights into [E, H, 2*I] gate_up_proj format
        5. Map router: router.gate.weight -> router.linear_router.weight
        6. Map shared expert weights to standalone shared_expert module
        7. Bake muP scaling into embedding weights
        8. Bake route_scale into routed expert down_proj weights
        9. Pad gate_proj weights with interleaved layout (when num_heads % TP != 0)
        10. Pad lm_head weights (when vocab_size % TP != 0)
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        target_dtype = torch.bfloat16

        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

        def strip_prefix(key):
            if has_model_prefix and key.startswith("model."):
                return key[6:]
            return key

        # Direct mappings: embeddings, final norm, lm_head
        for key, value in state_dict.items():
            stripped = strip_prefix(key)

            if stripped == "embed_tokens.weight":
                embed_weight = value.to(target_dtype)
                mup_enabled = getattr(config, "mup_enabled", False)
                if mup_enabled:
                    mup_scale = math.sqrt(config.hidden_size)
                    embed_weight = embed_weight * mup_scale
                neuron_state_dict["embed_tokens.weight"] = embed_weight
                continue
            if stripped == "norm.weight":
                neuron_state_dict["norm.weight"] = value.to(target_dtype)
                continue
            if key == "lm_head.weight":
                lm_weight = value.to(target_dtype)
                # Pad lm_head to be divisible by TP degree
                tp_degree = neuron_config.tp_degree
                vocab_size = lm_weight.shape[0]
                if vocab_size % tp_degree != 0:
                    padded_vocab = ((vocab_size // tp_degree) + 1) * tp_degree
                    pad_rows = padded_vocab - vocab_size
                    lm_weight = torch.cat(
                        [
                            lm_weight,
                            torch.zeros(
                                pad_rows, lm_weight.shape[1], dtype=target_dtype
                            ),
                        ],
                        dim=0,
                    )
                neuron_state_dict["lm_head.weight"] = lm_weight
                continue

        # Layer-by-layer conversion
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
        moe_intermediate = config.moe_intermediate_size
        hidden_size = config.hidden_size
        num_dense_layers = getattr(config, "num_dense_layers", 2)

        for layer_idx in range(num_layers):
            if has_model_prefix:
                hf_prefix = f"model.layers.{layer_idx}"
            else:
                hf_prefix = f"layers.{layer_idx}"
            neuron_prefix = f"layers.{layer_idx}"

            # Layer norms (4 per layer)
            for norm_name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_mlp_layernorm",
                "post_mlp_layernorm",
            ]:
                hf_key = f"{hf_prefix}.{norm_name}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[f"{neuron_prefix}.{norm_name}.weight"] = (
                        state_dict[hf_key].to(target_dtype)
                    )

            # Attention Q, K, V projections
            for proj in ["q_proj", "k_proj", "v_proj"]:
                hf_key = f"{hf_prefix}.self_attn.{proj}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[
                        f"{neuron_prefix}.self_attn.qkv_proj.{proj}.weight"
                    ] = state_dict[hf_key].to(target_dtype)

            # O projection
            hf_key = f"{hf_prefix}.self_attn.o_proj.weight"
            if hf_key in state_dict:
                neuron_state_dict[f"{neuron_prefix}.self_attn.o_proj.weight"] = (
                    state_dict[hf_key].to(target_dtype)
                )

            # QK norm weights: q_norm -> q_layernorm, k_norm -> k_layernorm
            for hf_norm, neuron_norm in [
                ("q_norm", "q_layernorm"),
                ("k_norm", "k_layernorm"),
            ]:
                hf_key = f"{hf_prefix}.self_attn.{hf_norm}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[
                        f"{neuron_prefix}.self_attn.{neuron_norm}.weight"
                    ] = state_dict[hf_key].to(target_dtype)

            # Attention gate_proj (gated attention, Trinity-specific)
            # CRITICAL: Must use INTERLEAVED padding matching Q projection layout.
            # NxDI pads Q with maybe_pad_interleaved (REPLICATE_TO_TP_DEGREE),
            # inserting zero heads between KV groups. The gate_proj output is
            # element-wise multiplied with the attention output (which follows
            # the Q head layout), so gate_proj MUST use the same interleaved
            # padding pattern. Using tail padding causes cores to apply gate
            # weights from the wrong head.
            hf_key = f"{hf_prefix}.self_attn.gate_proj.weight"
            if hf_key in state_dict:
                gate_weight = state_dict[hf_key].to(target_dtype)
                tp_degree = neuron_config.tp_degree
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                if num_heads % tp_degree != 0:
                    # Use interleaved padding matching Q layout.
                    # Gate weight is (num_heads, hidden_size) -- one row per head.
                    # Split into KV groups, pad each group with zero rows,
                    # then concatenate back to (padded_total_heads, hidden_size).
                    padded_total_heads = math.ceil(num_heads / tp_degree) * tp_degree
                    group_size = num_heads // num_kv_heads  # Q heads per KV group
                    groups = gate_weight.split(group_size, dim=0)
                    pad_per_group = (padded_total_heads - num_heads) // num_kv_heads
                    interleaved = []
                    for group in groups:
                        interleaved.append(group)
                        interleaved.append(
                            torch.zeros(
                                pad_per_group,
                                gate_weight.shape[1],
                                dtype=target_dtype,
                            )
                        )
                    gate_weight = torch.cat(interleaved, dim=0)
                neuron_state_dict[
                    f"{neuron_prefix}.self_attn.attn_gate_proj.weight"
                ] = gate_weight

            # MLP weights
            if layer_idx < num_dense_layers:
                # Dense layers (uses dense_intermediate_size)
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    hf_key = f"{hf_prefix}.mlp.{proj_name}.weight"
                    if hf_key in state_dict:
                        neuron_state_dict[f"{neuron_prefix}.mlp.{proj_name}.weight"] = (
                            state_dict[hf_key].to(target_dtype)
                        )
            else:
                # MoE layers
                # Router: router.gate.weight -> router.linear_router.weight
                hf_router_key = f"{hf_prefix}.mlp.router.gate.weight"
                if hf_router_key in state_dict:
                    neuron_state_dict[
                        f"{neuron_prefix}.mlp.router.linear_router.weight"
                    ] = state_dict[hf_router_key].to(target_dtype)

                # Expert bias (Trinity-specific routing parameter)
                hf_bias_key = f"{hf_prefix}.mlp.expert_bias"
                if hf_bias_key in state_dict:
                    neuron_state_dict[f"{neuron_prefix}.mlp.router.expert_bias"] = (
                        state_dict[hf_bias_key].to(torch.float32)
                    )

                # Stack expert weights for NxDI MoE v2 format
                gate_up_proj = torch.empty(
                    num_experts, hidden_size, 2 * moe_intermediate, dtype=target_dtype
                )
                down_proj = torch.empty(
                    num_experts, moe_intermediate, hidden_size, dtype=target_dtype
                )

                all_experts_found = True
                for e in range(num_experts):
                    gate_key = f"{hf_prefix}.mlp.experts.{e}.gate_proj.weight"
                    up_key = f"{hf_prefix}.mlp.experts.{e}.up_proj.weight"
                    down_key = f"{hf_prefix}.mlp.experts.{e}.down_proj.weight"

                    if (
                        gate_key in state_dict
                        and up_key in state_dict
                        and down_key in state_dict
                    ):
                        gate_w = state_dict[gate_key].to(target_dtype)
                        up_w = state_dict[up_key].to(target_dtype)
                        down_w = state_dict[down_key].to(target_dtype)

                        gate_up_concat = torch.cat([gate_w, up_w], dim=0)
                        gate_up_proj[e] = gate_up_concat.T
                        down_proj[e] = down_w.T
                    else:
                        all_experts_found = False
                        break

                if all_experts_found:
                    # Bake route_scale into routed expert down_proj weights.
                    # NxDI MoE v2 does NOT support route_scale natively.
                    # Shared experts are NOT scaled.
                    route_scale = getattr(config, "route_scale", 1.0)
                    if route_scale != 1.0:
                        down_proj = down_proj * route_scale

                    neuron_state_dict[
                        f"{neuron_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                    ] = gate_up_proj
                    neuron_state_dict[
                        f"{neuron_prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                    ] = down_proj

                # Shared expert weights (mapped to standalone NeuronTrinitySharedExpert)
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    hf_key = f"{hf_prefix}.mlp.shared_experts.{proj_name}.weight"
                    if hf_key in state_dict:
                        neuron_state_dict[
                            f"{neuron_prefix}.shared_expert.{proj_name}.weight"
                        ] = state_dict[hf_key].to(target_dtype)

                # Fused MoE TKG aliased weights.
                # When init_tkg_module=True, the MoE module stores the
                # pre_mlp_layernorm as moe.rmsnorm and also inside
                # moe.moe_fused_tkg.post_attention_layernorm. These are the
                # same Python object (aliased), but appear as separate keys in
                # the state dict. We must provide both so the framework loads
                # them correctly.
                # Similarly, the router stores a transposed weight (weight_T)
                # alongside linear_router.weight.
                if getattr(config, "moe_fused_nki_kernel_enabled", False):
                    # MoEFusedTKG has a separate (non-shared) RMSNorm that
                    # needs the same weights as pre_mlp_layernorm. This is
                    # a distinct module (not aliased) so we copy the weight.
                    # Note: moe.rmsnorm is None (not a module), so we do NOT
                    # provide mlp.rmsnorm.weight.
                    pre_mlp_key = f"{neuron_prefix}.pre_mlp_layernorm.weight"
                    if pre_mlp_key in neuron_state_dict:
                        pre_mlp_w = neuron_state_dict[pre_mlp_key]
                        neuron_state_dict[
                            f"{neuron_prefix}.mlp.moe_fused_tkg.post_attention_layernorm.weight"
                        ] = pre_mlp_w.clone()

                    # Router transposed weight (generated by preshard_hook,
                    # but we provide it here too for completeness)
                    router_key = f"{neuron_prefix}.mlp.router.linear_router.weight"
                    if router_key in neuron_state_dict:
                        neuron_state_dict[f"{neuron_prefix}.mlp.router.weight_T"] = (
                            neuron_state_dict[router_key].detach().T.clone()
                        )

        # Rank utilities for tensor parallel
        tp_degree = neuron_config.tp_degree
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        return neuron_state_dict

    def get_compiler_args(self):
        """Get compiler arguments for Trinity models."""
        return "--model-type=transformer -O1"
