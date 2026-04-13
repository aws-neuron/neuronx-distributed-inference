# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M2 model for NeuronX Distributed Inference.

Architecture: 229B total, ~10B active. 62 decoder layers, 256 MoE experts (top-8),
sigmoid routing with e_score_correction_bias, partial RoPE (64/128 head dim),
QK normalization (RMSNorm before reshape), GQA 48Q/8KV heads, SwiGLU experts.

Based on Henan's (whn09) implementation with SDK 2.28 improvements:
- Fused MoE NKI kernels (router_topk, moe_cte, moe_tkg)
- ModuleMarker wrappers for compiler optimization
- Fused QKV support
- Shard-on-intermediate padding for blockwise matmul
- RouterTopKWithBias preserving e_score_correction_bias for accuracy
"""

import gc
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)

# nki-library attention block kernel (partial RoPE support)
try:
    from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
    from nkilib.core.utils.common_types import (
        QuantizationType as NkilibQuantizationType,
    )

    _HAS_NKILIB_ATTN_BLOCK = True
except ImportError:
    _HAS_NKILIB_ATTN_BLOCK = False
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import (
    initialize_moe_process_group,
)

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_rmsnorm_cls():
    """Return the appropriate RMSNorm class for the execution environment."""
    if cpu_mode():

        class SimpleRMSNorm(nn.Module):
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
                return self.weight * hidden_states.to(input_dtype)

        return SimpleRMSNorm
    return CustomRMSNorm


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


# ---------------------------------------------------------------------------
# Fused QKV helpers
# ---------------------------------------------------------------------------


def _helper_concat_and_delete_qkv(
    state_dict: Dict[str, Any], layer_num: int, attr: str
):
    """Concatenate Q/K/V into fused Wqkv for a single attribute (weight or scale).

    The fused key uses the ``qkv_proj.Wqkv`` path because the NxDI model nests
    the Wqkv linear layer under ``self_attn.qkv_proj`` (a GroupQueryAttention_QKV module).
    """
    state_dict[f"layers.{layer_num}.self_attn.qkv_proj.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict: Dict[str, Any], cfg: InferenceConfig):
    """Fuse separate Q/K/V weights into a single Wqkv tensor per layer."""
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config) or []
    for layer_idx in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, layer_idx, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled
            or cfg.neuron_config.quantized
        ) and f"layers.{layer_idx}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, layer_idx, "scale")
    gc.collect()
    return state_dict


def maybe_dequantize_layer(neuron_state_dict: dict, config):
    """Dequantize FP8 layers (weight_scale_inv) to the configured torch dtype."""
    scale_layers = []
    for layer_key in list(neuron_state_dict.keys()):
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = scales.repeat_interleave(
                block_size[0], dim=0
            ).repeat_interleave(block_size[1], dim=1)
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(
                torch.float32
            )
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(
                config.neuron_config.torch_dtype
            )
    for key in scale_layers:
        del neuron_state_dict[key]


# ---------------------------------------------------------------------------
# MiniMax-M2 specific modules
# ---------------------------------------------------------------------------


class MiniMaxM2QKNorm(nn.Module):
    """
    QK normalization for MiniMax-M2 using Neuron's fused RmsNorm custom call.

    MiniMax-M2 applies RMSNorm on the Q/K projection output before reshape.
    This implementation uses the Neuron-native AwsNeuronRmsNorm custom call
    (via RmsNorm.apply) which is validated for both context encoding and token
    generation NEFFs. Hand-rolled PyTorch RMSNorm (pow/mean/rsqrt) compiles
    into different HLO in CE vs TG and produces incorrect TG results.

    Normalization is computed per-rank (no all-reduce) on the flat projection
    output [B, S, per_rank_dim]. The per-element weight is selected dynamically
    by SPMD rank from a padded weight tensor.

    Args:
        hidden_size: Per-rank hidden dimension (num_heads_per_rank * head_dim)
        eps: Epsilon for numerical stability
        tp_degree: Tensor parallelism degree
        padded_hidden_size: Total weight storage size (tp_degree * per_rank_size)
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        tp_degree=1,
        padded_hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.tp_degree = tp_degree
        self.padded_hidden_size = (
            padded_hidden_size
            if padded_hidden_size is not None
            else (hidden_size * tp_degree)
        )
        # Weight stored at full padded size for SPMD rank-based selection
        self.weight = nn.Parameter(torch.ones(self.padded_hidden_size))

    def forward(self, hidden_states, rank_util=None):
        """
        Apply Neuron-native RMSNorm on flat Q or K tensor (no all-reduce).

        Args:
            hidden_states: [B, S, per_rank_dim] — flat projection output
            rank_util: SPMDRank for dynamic weight slice selection
        """
        from neuronx_distributed_inference.modules.custom_calls import RmsNorm

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Dynamically select weight slice by SPMD rank (XLA-compatible)
        if rank_util is not None and self.tp_degree > 1:
            weight_reshaped = self.weight.view(self.tp_degree, self.hidden_size)
            rank_index = rank_util.rank[:1]
            local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)
        else:
            local_weight = self.weight[: self.hidden_size]

        # Use Neuron-native fused RmsNorm (AwsNeuronRmsNorm custom call)
        dim = len(hidden_states.shape) - 1
        result = RmsNorm.apply(hidden_states, local_weight, self.variance_epsilon, dim)

        return result.to(input_dtype)


class RouterTopKWithBias(RouterTopK):
    """
    RouterTopK with e_score_correction_bias for MiniMax-M2 sigmoid routing.

    MiniMax-M2 applies sigmoid to router logits to obtain expert affinities, then
    adds a learned per-expert bias before top-K selection. The bias influences which
    experts are chosen but does NOT affect the affinity weights passed to experts.

    The bias MUST be an nn.Parameter (not a buffer) because:
    - XLA tracing bakes register_buffer values as constants in the NEFF
    - shard_children only processes nn.Parameter in supported modules
    - replace_weights only loads tensors present in the traced model's separated weights
    Using nn.Parameter ensures the bias is separated during tracing and loaded from
    the checkpoint at inference time.

    Dropping the bias (as v3 does for XLA simplicity) causes ~75% wrong expert selection
    because bias values (~8.0-9.5) dominate sigmoid scores (0-1).
    """

    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        # nn.Parameter so it gets separated from NEFF and loaded from checkpoint.
        # requires_grad=False since this is inference-only.
        # CRITICAL: Initialize with non-uniform values to prevent XLA graph optimization
        # from eliminating the add-bias operation. Uniform values (zeros, ones) don't
        # change relative ordering in topk, so XLA can prove the add is a no-op and
        # eliminate it — removing the bias parameter from the HLO entirely and making it
        # impossible to load the real bias values at inference time.
        # Using arange produces distinct per-expert values that genuinely affect topk
        # ordering, forcing the compiler to keep the bias as a runtime parameter.
        # IMPORTANT: Initialize as bfloat16 to match the dtype that _cast_helper
        # will produce from the checkpoint (FP32 → BF16). If the NEFF expects FP32
        # but the checkpoint provides BF16, the LayoutTransformation silently
        # ignores the weight and leaves the trace-time values in place.
        self.e_score_correction_bias = nn.Parameter(
            torch.arange(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)

        # Add bias for expert selection only (MiniMax-M2 specific).
        # sigmoid(logits) + bias determines WHICH experts are selected,
        # but the un-biased sigmoid scores are used as affinity weights.
        scores_for_choice = (
            expert_affinities.float() + self.e_score_correction_bias.unsqueeze(0)
        )
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


# ---------------------------------------------------------------------------
# MoE initialization
# ---------------------------------------------------------------------------


def initialize_minimax_m2_moe_module(
    config: InferenceConfig, rmsnorm=None, init_tkg_module=False
):
    """
    Create the MoE module for MiniMax-M2 with e_score_correction_bias.

    Instead of wrapping the standard MoE, we inject a RouterTopKWithBias directly
    as the router. This ensures the bias is an nn.Parameter that gets:
    1. Separated from the NEFF during XLA tracing (not baked as a constant)
    2. Loaded from the checkpoint via replace_weights at inference time

    The bias values (~8.0-9.5) dominate sigmoid scores (0-1) and are critical
    for correct expert selection. Without them, ~75% of experts are wrong.
    """
    from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
    from neuronx_distributed.modules.moe.model import MoE
    from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_expert_model_parallel_size,
        get_tensor_model_parallel_group,
        get_world_group,
    )

    from neuronx_distributed_inference.modules.moe_v2 import (
        initialize_moe_process_group,
    )

    enabled_hybrid_sharding = config.neuron_config.hybrid_sharding_config is not None
    (
        moe_tkg_tensor_model_parallel_group,
        moe_tkg_expert_model_parallel_group,
        moe_cte_tensor_model_parallel_group,
        moe_cte_expert_model_parallel_group,
    ) = initialize_moe_process_group(config, enabled_hybrid_sharding)

    # Use RouterTopKWithBias instead of standard RouterTopK
    router = RouterTopKWithBias(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        bias=False,  # no linear bias; we use e_score_correction_bias instead
        apply_act_fn_over_topk=False,
        store_transposed_weights=init_tkg_module,
    )

    hidden_size_actual = getattr(config, "original_hidden_size", None)
    intermediate_size_actual = getattr(config, "original_intermediate_size", None)

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_size_actual=hidden_size_actual,
            intermediate_size_actual=intermediate_size_actual,
            is_hidden_dim_shuffled=config.neuron_config.is_hidden_dim_shuffled,
            is_intermediate_dim_shuffled=config.neuron_config.is_intermediate_dim_shuffled,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            bias=False,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            use_index_calc_kernel=config.neuron_config.use_index_calc_kernel,
            gate_clamp_upper_limit=config.neuron_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=config.neuron_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=config.neuron_config.up_clamp_upper_limit,
            up_clamp_lower_limit=config.neuron_config.up_clamp_lower_limit,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping,
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        enabled_hybrid_sharding=enabled_hybrid_sharding,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tensor_model_parallel_group,
        cte_expert_model_parallel_group=moe_cte_expert_model_parallel_group,
        tkg_tensor_model_parallel_group=moe_tkg_tensor_model_parallel_group,
        tkg_expert_model_parallel_group=moe_tkg_expert_model_parallel_group,
    )

    if init_tkg_module:
        from neuronx_distributed.modules.moe.model import MoEFusedTKGConfig

        tkg_config = MoEFusedTKGConfig(
            quantized=config.neuron_config.quantized,
            moe_fused_kernel_enabled=config.neuron_config.moe_fused_nki_kernel_enabled,
            router_topk_kernel_enabled=config.neuron_config.router_topk_nki_kernel_enabled,
            expert_mlp_kernel_enabled=config.neuron_config.expert_mlp_nki_kernel_enabled,
            shared_mlp_kernel_enabled=config.neuron_config.shared_mlp_nki_kernel_enabled,
            norm_topk_prob=config.neuron_config.normalize_top_k_affinities,
            is_mxfp4_compute=config.neuron_config.is_mxfp4_compute,
            router_mm_dtype=config.neuron_config.router_config.dtype,
        )
    else:
        tkg_config = None

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=None,  # MiniMax-M2 has no shared experts
        rmsnorm=rmsnorm,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits,
        sequence_dimension=1,
        init_tkg_module=init_tkg_module,
        tkg_config=tkg_config,
    )

    moe.eval()
    return moe


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def convert_minimax_m2_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: "MiniMaxM2InferenceConfig",
) -> Dict[str, Any]:
    """
    Convert a HuggingFace MiniMax-M2 checkpoint to the NxDI-compatible format.

    Key transformations:
    1. Stack per-expert w1/w3 into gate_up_proj, w2 into down_proj
    2. Rename router gate -> router.linear_router (or router.e_score_correction_bias)
    3. Pad QK norm weights to match TP sharding (interleaved for Q, replicated for K)
    4. Optionally pad intermediate_size for shard-on-I blockwise matmul
    5. Optionally fuse QKV into Wqkv
    """
    from neuronx_distributed_inference.modules.attention.gqa import (
        GQA,
        _maybe_pad_interleaved,
        get_shardable_head_counts,
    )

    assert config.neuron_config.glu_mlp is True, (
        "MiniMax-M2 requires glu_mlp=True (SwiGLU)"
    )

    # Dequantize FP8 weights if present
    maybe_dequantize_layer(neuron_state_dict, config)

    with torch.no_grad():
        tp_degree = config.neuron_config.tp_degree
        head_dim = config.head_dim
        has_qk_norm = getattr(config, "use_qk_norm", True)

        # Rank utility tensor for SPMD operations (int32 for NKI compatibility)
        rank_tensor = torch.arange(0, tp_degree, dtype=torch.int32)
        neuron_state_dict["rank_util.rank"] = rank_tensor

        # Pre-compute sharded head counts for QK norm padding
        sharding_strategy = GQA.REPLICATE_TO_TP_DEGREE
        padded_num_attention_heads, padded_num_kv_heads = get_shardable_head_counts(
            tp_degree,
            config.num_attention_heads,
            config.num_key_value_heads,
            sharding_strategy,
        )

        gc_interval = 64  # GC every N experts to control memory

        for layer_idx in range(config.num_hidden_layers):
            # Per-layer rank tensor for attention SPMD
            neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = (
                rank_tensor.clone()
            )

            # --- QK norm weight padding ---
            if has_qk_norm:
                # Q norm: interleaved padding (48 -> padded heads)
                q_norm_key = f"layers.{layer_idx}.self_attn.q_norm.weight"
                if q_norm_key in neuron_state_dict:
                    q_norm_full = neuron_state_dict[q_norm_key]
                    source_group_size = (
                        config.num_attention_heads // config.num_key_value_heads
                    )
                    q_norm_padded = _maybe_pad_interleaved(
                        q_norm_full.unsqueeze(0),
                        pad_dim=1,
                        source_heads=config.num_attention_heads,
                        target_heads=padded_num_attention_heads,
                        source_group_size=source_group_size,
                    ).squeeze(0)
                    neuron_state_dict[q_norm_key] = q_norm_padded

                # K norm: replicate from original KV heads to padded KV heads
                k_norm_key = f"layers.{layer_idx}.self_attn.k_norm.weight"
                if k_norm_key in neuron_state_dict:
                    k_norm_full = neuron_state_dict[k_norm_key]
                    k_norm_reshaped = k_norm_full.reshape(
                        config.num_key_value_heads, head_dim
                    )
                    repeats = padded_num_kv_heads // config.num_key_value_heads
                    k_norm_replicated = k_norm_reshaped.repeat_interleave(
                        repeats, dim=0
                    )
                    neuron_state_dict[k_norm_key] = k_norm_replicated.reshape(-1)

            # --- Router weights ---
            gate_key = f"layers.{layer_idx}.block_sparse_moe.gate.weight"
            router_key = (
                f"layers.{layer_idx}.block_sparse_moe.router.linear_router.weight"
            )
            neuron_state_dict[router_key] = neuron_state_dict.pop(gate_key)

            # e_score_correction_bias: map to RouterTopKWithBias.e_score_correction_bias
            # This is an nn.Parameter in the router, so it will be separated from the
            # NEFF during tracing and loaded via replace_weights at inference time.
            bias_src_key = (
                f"layers.{layer_idx}.block_sparse_moe.e_score_correction_bias"
            )
            bias_dst_key = (
                f"layers.{layer_idx}.block_sparse_moe.router.e_score_correction_bias"
            )
            if bias_src_key in neuron_state_dict:
                neuron_state_dict[bias_dst_key] = neuron_state_dict.pop(bias_src_key)

            # --- Expert weight stacking ---
            w1_key = f"layers.{layer_idx}.block_sparse_moe.experts.0.w1.weight"
            intermediate_size, hidden_size = neuron_state_dict[w1_key].shape
            device = neuron_state_dict[w1_key].device
            dtype = neuron_state_dict[w1_key].dtype

            # Stack gate (w1) + up (w3) into gate_up_proj: [E, H, 2*I]
            gate_up_proj = torch.empty(
                config.num_local_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=dtype,
                device=device,
            )
            for expert_idx in range(config.num_local_experts):
                ew1 = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
                ew3 = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"

                gate_up_slice = torch.narrow(gate_up_proj, 0, expert_idx, 1)
                torch.narrow(gate_up_slice, 2, 0, intermediate_size).copy_(
                    neuron_state_dict[ew1].T
                )
                torch.narrow(
                    gate_up_slice, 2, intermediate_size, intermediate_size
                ).copy_(neuron_state_dict[ew3].T)
                del neuron_state_dict[ew1], neuron_state_dict[ew3]
                if (expert_idx + 1) % gc_interval == 0:
                    gc.collect()

            # Pad gate_up_proj intermediate dimension if needed for shard-on-I
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                gate_up_proj = gate_up_proj.reshape(
                    config.num_local_experts, hidden_size, 2, -1
                )
                gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(
                    config.num_local_experts, hidden_size, -1
                )

            neuron_state_dict[
                f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"
            ] = gate_up_proj

            # Stack down (w2) into down_proj: [E, I, H]
            down_proj = torch.empty(
                config.num_local_experts,
                intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            for expert_idx in range(config.num_local_experts):
                ew2 = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
                torch.narrow(down_proj, 0, expert_idx, 1).copy_(
                    neuron_state_dict[ew2].T
                )
                del neuron_state_dict[ew2]
                if (expert_idx + 1) % gc_interval == 0:
                    gc.collect()

            if pad_size > 0:
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))

            neuron_state_dict[
                f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"
            ] = down_proj

            gc.collect()

        # Fuse QKV if configured (must run BEFORE the rename below, since
        # convert_state_dict_to_fused_qkv expects layers.X.self_attn.q_proj.weight)
        if config.neuron_config.fused_qkv:
            neuron_state_dict = convert_state_dict_to_fused_qkv(
                neuron_state_dict, config
            )

        # --- Attention projection key renaming ---
        # The NxDI traced model uses nested module names for attention projections:
        #   self_attn.qkv_proj.q_proj.weight  (not self_attn.q_proj.weight)
        #   self_attn.qkv_proj.k_proj.weight  (not self_attn.k_proj.weight)
        #   self_attn.qkv_proj.v_proj.weight  (not self_attn.v_proj.weight)
        #   self_attn.o_proj.o_proj.weight     (not self_attn.o_proj.weight)
        # The preshard hook in RowParallelLinear handles the o_proj rename
        # (o_proj.weight -> o_proj.o_proj.weight), so we only rename Q/K/V here.
        # When fused_qkv=True, Q/K/V are already merged into Wqkv above.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            # Q/K/V projections -> nested under qkv_proj
            for proj in ("q_proj", "k_proj", "v_proj"):
                old_key = f"{prefix}.{proj}.weight"
                new_key = f"{prefix}.qkv_proj.{proj}.weight"
                if old_key in neuron_state_dict:
                    neuron_state_dict[new_key] = neuron_state_dict.pop(old_key)

    return neuron_state_dict


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------


class MiniMaxM2InferenceConfig(InferenceConfig):
    """
    Inference configuration for MiniMax-M2.

    Extends InferenceConfig with MoE-specific setup:
    - Sigmoid routing with FP32 router precision
    - Intermediate-size padding for shard-on-I blockwise matmul
    - Fused MoE NKI kernel enablement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MiniMax-M2 has no shared experts
        self.n_shared_experts = 0

        # Store MoE intermediate size before any padding
        self.moe_intermediate_size = self.intermediate_size

        # Pad intermediate for shard-on-I compatibility
        self.moe_intermediate_pad_size = 0
        self._maybe_pad_intermediate()

        # Enable fused MoE NKI kernels where dimensions allow
        self._enable_moe_fused_nki_kernel()

        # Router config: MiniMax-M2 uses sigmoid routing with FP32 precision
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "sigmoid"

        # MiniMax-M2 normalizes top-K affinities
        self.neuron_config.normalize_top_k_affinities = True

        # Disable numeric CC token for MoE stability
        self.neuron_config.disable_numeric_cc_token = True

    def _maybe_pad_intermediate(self):
        """Pad intermediate_size so shard-on-I blockwise matmul kernels tile correctly."""
        moe_tp_degree = self.neuron_config.moe_tp_degree
        i_tp = self.intermediate_size // moe_tp_degree
        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if i_tp % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(i_tp / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(padded - self.intermediate_size, 0)
                self.intermediate_size = padded

    def _enable_moe_fused_nki_kernel(self):
        """Enable fused MoE NKI kernel if the per-TP intermediate dimension is aligned."""
        i_tp = self.intermediate_size // self.neuron_config.moe_tp_degree
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False):
            if i_tp % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
                self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_local_experts",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
            "use_qk_norm",
            "rotary_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NeuronMiniMaxM2Attention(NeuronAttentionBase):
    """
    MiniMax-M2 attention with two non-standard features:

    1. QK normalization applied BEFORE reshape to per-head layout (on the full
       Q/K projection output). Uses MiniMaxM2QKNorm with distributed all-reduce.
    2. Partial RoPE: rotary embeddings applied to only the first ``rotary_dim``
       dimensions of each head (64 out of 128).
    """

    def __init__(self, config: MiniMaxM2InferenceConfig):
        self.rotary_dim = getattr(config, "rotary_dim", config.head_dim)

        # RotaryEmbedding sized to rotary_dim (64), not head_dim (128)
        rotary_emb = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,  # handled by MiniMaxM2QKNorm below
        )

        # --- QK normalization (local per-rank, no all-reduce) ---
        self.use_minimax_qk_norm = getattr(config, "use_qk_norm", True)
        tp_degree = config.neuron_config.tp_degree

        if self.use_minimax_qk_norm:
            q_per_rank = self.num_heads * self.head_dim
            k_per_rank = self.num_key_value_heads * self.head_dim

            # Weight storage: padded to tp_degree * per_rank for SPMD selection
            padded_q = self.num_heads * tp_degree * config.head_dim
            padded_kv = self.num_key_value_heads * tp_degree
            padded_k = padded_kv * config.head_dim

            self.q_norm = MiniMaxM2QKNorm(
                q_per_rank,
                eps=config.rms_norm_eps,
                tp_degree=tp_degree,
                padded_hidden_size=padded_q,
            )
            self.k_norm = MiniMaxM2QKNorm(
                k_per_rank,
                eps=config.rms_norm_eps,
                tp_degree=tp_degree,
                padded_hidden_size=padded_k,
            )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM2Attention requires an initialized distributed environment. "
                "Use neuronx_distributed to initialize."
            )

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
        """Apply local QK norm on flat projection, reshape to heads, then partial RoPE."""
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states,
            rmsnorm=rmsnorm,
            adapter_ids=adapter_ids,
            residual=residual,
        )

        # QK norm on flat per-rank projection output BEFORE reshape (no all-reduce)
        if self.use_minimax_qk_norm:
            Q = self.q_norm(Q, self.rank_util)
            K = self.k_norm(K, self.rank_util)

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        # Reshape to [B, S, num_heads, head_dim] then transpose to [B, H, S, D]
        Q = (
            Q.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            K.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            V.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if not skip_rope:
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        return Q, K, V, cos_cache, sin_cache, residual

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Apply partial rotary embeddings (first rotary_dim dimensions only)."""
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            if self.rotary_dim < self.head_dim:
                Q_rot, Q_pass = Q[..., : self.rotary_dim], Q[..., self.rotary_dim :]
                K_rot, K_pass = K[..., : self.rotary_dim], K[..., self.rotary_dim :]
                Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)
                Q = torch.cat([Q_rot, Q_pass], dim=-1)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, cos_cache, sin_cache

    def attention_block_tokengen_nki_kernel(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        rotary_position_ids=None,
        update_kv_per_layer=True,
        active_block_table=None,
        use_polar_compatible_rope=False,
    ):
        """
        Override base class to use nki-library attention_block_tkg kernel with
        partial RoPE support (rotary_dim < head_dim).

        Uses the nki-library kernel instead of the compiler's private kernel.
        QK norm is fused into the kernel via the flat QK RMSNorm feature, which
        normalizes across all Q (or K) heads concatenated before head splitting.
        """
        assert _HAS_NKILIB_ATTN_BLOCK, (
            "nki-library attention_block_tkg not available. "
            "Install the nki-library fork with partial RoPE support."
        )

        from neuronx_distributed.parallel_layers.mappings import (
            gather_from_sequence_parallel_region,
            reduce_from_tensor_model_parallel_region,
            reduce_scatter_to_sequence_parallel_region,
            gather_from_tensor_model_parallel_region_with_dim,
            reduce_scatter_to_tensor_model_parallel_region_with_dim,
        )
        from neuronx_distributed_inference.modules.attention.attention_base import (
            EPDispatchOption,
            get_data_parallel_attention_dp_group,
        )
        # NKI 0.3.0: use kernel[lnc_int] instead of kernel[(nc(lnc),)]

        if (
            self.sequence_parallel_enabled
            and self.tensor_model_parallel_group is not None
        ):
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        # Get shapes
        bsz, s_tkg, h = hidden_states.shape
        h_out = h // 2 if self.is_eagle3_draft else h
        num_q_heads = self.num_heads

        # Prepare rmsnorm params
        rmsnorm_enabled = rmsnorm is not None
        W_gamma = rmsnorm.weight.data.unsqueeze(0) if rmsnorm is not None else None

        # Prepare RoPE params
        rope_contiguous_layout = not use_polar_compatible_rope

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(
                    hidden_states, rotary_position_ids
                )
                # Take first half and reshape to [dim//2, batch_size, seq_len]
                cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
                sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)
        elif use_polar_compatible_rope:
            from neuronx_distributed.modules.attention.utils import precompute_freqs_cis

            rotary_freqs = precompute_freqs_cis(
                self.head_dim,
                self.neuron_config.max_context_length * 2,
                self.rope_theta,
                self.use_scaled_rope,
                device=hidden_states.device,
            )
            rotary_freqs = rotary_freqs[position_ids]
            cos_cache = rotary_freqs.cos().permute(2, 0, 1)
            sin_cache = rotary_freqs.sin().permute(2, 0, 1)
        else:
            cos_cache = None
            sin_cache = None

        # Prepare attention mask: merge active_mask and transpose for kernel layout
        attention_mask = attention_mask.expand(-1, num_q_heads, -1, -1)
        expected_active_mask_shape = (bsz, 1, s_tkg, s_tkg)
        if s_tkg == 1:
            active_mask = torch.ones(
                expected_active_mask_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            assert active_mask.shape == expected_active_mask_shape, (
                f"{active_mask.shape} != {expected_active_mask_shape}"
            )
        active_mask = active_mask.expand(-1, num_q_heads, -1, -1)
        attention_mask[:, :, :, -s_tkg:] = active_mask
        # Transpose to [S_ctx, B, q_heads, S_tkg] for nki-library kernel
        attention_mask = attention_mask.permute(3, 0, 1, 2)

        # Prepare KV cache
        K_prior, V_prior = past_key_value[:2]
        K_prior = K_prior.data
        V_prior = V_prior.data
        update_cache_in_kernel = (
            update_kv_per_layer and self.attn_block_tkg_nki_kernel_cache_update
        )
        sink = (
            self.get_learned_sinks().data.unsqueeze(-1)
            if self.learned_sinks_size is not None
            else None
        )
        kv_cache_update_idx = position_ids[:, :1].to(torch.int32)

        # Prepare output projection
        W_out = self.get_o_proj().o_proj.weight.data
        if self.o_bias:
            W_out_bias = (
                self.get_o_proj().o_proj.bias.data / self.tp_degree
            ).unsqueeze(0)
        else:
            W_out_bias = None

        # Prepare QKV projection
        W_qkv = self.get_qkv_proj().Wqkv.weight.data
        bias_qkv = (
            self.get_qkv_proj().Wqkv.bias.data.unsqueeze(0) if self.qkv_bias else None
        )

        grid = self.logical_nc_config

        # Prepare flat QK norm weights (per-rank slice via SPMD rank selection)
        # The kernel expects [1, per_rank_width] weights for each of Q and K.
        flat_qk_norm_enabled = self.use_minimax_qk_norm
        flat_qk_W_Q = None
        flat_qk_W_K = None
        if flat_qk_norm_enabled:
            # Q norm: select per-rank slice from padded weight
            q_norm_weight = self.q_norm.weight.data  # [padded_q_hidden_size]
            q_per_rank = self.q_norm.hidden_size
            if self.q_norm.tp_degree > 1:
                q_w_reshaped = q_norm_weight.view(self.q_norm.tp_degree, q_per_rank)
                rank_index = self.rank_util.rank[:1]
                flat_qk_W_Q = torch.index_select(
                    q_w_reshaped, 0, rank_index
                )  # [1, q_per_rank]
            else:
                flat_qk_W_Q = q_norm_weight[:q_per_rank].unsqueeze(0)  # [1, q_per_rank]

            # K norm: select per-rank slice from padded weight
            k_norm_weight = self.k_norm.weight.data  # [padded_k_hidden_size]
            k_per_rank = self.k_norm.hidden_size
            if self.k_norm.tp_degree > 1:
                k_w_reshaped = k_norm_weight.view(self.k_norm.tp_degree, k_per_rank)
                rank_index = self.rank_util.rank[:1]
                flat_qk_W_K = torch.index_select(
                    k_w_reshaped, 0, rank_index
                )  # [1, k_per_rank]
            else:
                flat_qk_W_K = k_norm_weight[:k_per_rank].unsqueeze(0)  # [1, k_per_rank]

        attn_output, K, V = attention_block_tkg[grid](
            # -- input
            X=hidden_states,
            X_hidden_dim_actual=getattr(self.config, "original_hidden_size", None),
            # -- rmsnorm X
            rmsnorm_X_enabled=rmsnorm_enabled,
            rmsnorm_X_eps=self.rms_norm_eps,
            rmsnorm_X_gamma=W_gamma,
            # -- qkv projections
            W_qkv=W_qkv,
            bias_qkv=bias_qkv,
            quantization_type_qkv=NkilibQuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            # -- Q/K processing: flat QK RMSNorm (before head split)
            rmsnorm_QK_flat_enabled=flat_qk_norm_enabled,
            rmsnorm_QK_flat_eps=self.rms_norm_eps if flat_qk_norm_enabled else 0.0,
            rmsnorm_QK_flat_W_Q=flat_qk_W_Q,
            rmsnorm_QK_flat_W_K=flat_qk_W_K,
            # -- Q/K processing: per-head pre-RoPE RMSNorm (disabled)
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_pre_rope_eps=0.0,
            rmsnorm_QK_pre_rope_W_Q=None,
            rmsnorm_QK_pre_rope_W_K=None,
            # -- Q/K processing: RoPE with partial rotary_dim
            cos=cos_cache,
            sin=sin_cache,
            rope_contiguous_layout=rope_contiguous_layout,
            rotary_dim=self.rotary_dim,
            # -- Q/K processing: post-RoPE RMSNorm (disabled)
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=0.0,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            # -- attention
            K_cache_transposed=self.k_cache_transposed,
            active_blocks_table=(
                active_block_table.to(torch.uint32)
                if active_block_table is not None
                else None
            ),
            K_cache=K_prior,
            V_cache=V_prior,
            attention_mask=attention_mask,
            sink=sink,
            softmax_scale=None,
            # -- KV cache update
            update_cache=update_cache_in_kernel,
            kv_cache_update_idx=kv_cache_update_idx,
            # -- output projection
            W_out=W_out,
            bias_out=W_out_bias,
            quantization_type_out=NkilibQuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=False,
            # -- output
            out_in_sb=False,
        )

        # Reshape and reduce output
        attn_output = attn_output.reshape((bsz, s_tkg, h_out))
        if self.sequence_parallel_enabled:
            attn_output = reduce_scatter_to_sequence_parallel_region(
                attn_output, 1, process_group=self.tensor_model_parallel_group
            )
        else:
            if self.ep_dispatch_cc_option == EPDispatchOption.AR_AG:
                attn_output = reduce_from_tensor_model_parallel_region(
                    attn_output, process_group=self.tensor_model_parallel_group
                )
            elif self.ep_dispatch_cc_option == EPDispatchOption.RS_AG:
                attn_output = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                    attn_output,
                    partition_dim=0,
                    process_group=self.tensor_model_parallel_group,
                )
            elif self.ep_dispatch_cc_option == EPDispatchOption.AG_AR:
                attn_output = gather_from_tensor_model_parallel_region_with_dim(
                    attn_output,
                    gather_dim=0,
                    process_group=get_data_parallel_attention_dp_group(),
                )
            else:
                raise ValueError(
                    f"Unknown EPDispatchOption: {self.ep_dispatch_cc_option}"
                )

        # KV cache handling
        if update_cache_in_kernel:
            KV = past_key_value
        else:
            # Reshape K/V from kernel output layout to the rank-4 [B, N, S, D]
            # layout expected by kv_cache_manager.update_kv_by_layer_id.
            # K from kernel: [head_dim, bsz, q_len] (dBS)
            # V from kernel: [bsz, q_len, head_dim] (BSd)
            # Target: [B, 1, S, D] (BNSd) or [B, 1, D, S] (BNdS) for transposed K
            K = K.permute(1, 0, 2) if self.k_cache_transposed else K.permute(1, 2, 0)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
            KV = (K, V)

        return attn_output, KV, cos_cache, sin_cache


class NeuronMiniMaxM2DecoderLayer(nn.Module):
    """MiniMax-M2 decoder layer: attention + MoE with ModuleMarker wrappers."""

    def __init__(self, config: MiniMaxM2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniMaxM2Attention(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(
            config, "moe_fused_nki_kernel_enabled", False
        )

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Fused MoE kernel absorbs post-attention layernorm
        if self.moe_fused_nki_kernel_enabled:
            self.block_sparse_moe = initialize_minimax_m2_moe_module(
                config=config,
                rmsnorm=self.post_attention_layernorm,
                init_tkg_module=True,
            )
        else:
            self.block_sparse_moe = initialize_minimax_m2_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronMiniMaxM2Model(NeuronBaseModel):
    """Traceable MiniMax-M2 base model."""

    def setup_attr_for_model(self, config: MiniMaxM2InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MiniMaxM2InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronMiniMaxM2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class NeuronMiniMaxM2ForCausalLM(NeuronBaseForCausalLM):
    """MiniMax-M2 causal language model for NxDI inference."""

    _model_cls = NeuronMiniMaxM2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return None

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: MiniMaxM2InferenceConfig
    ) -> dict:
        return convert_minimax_m2_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        """Compiler arguments tuned for MiniMax-M2 MoE.

        Uses -O1 by default. -O2 was tested but provides no scratchpad memory
        savings vs -O1 (identical 22 GB tensor allocation at 62 layers TP=32).
        """
        if self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            opt_level = "-O1"
        else:
            opt_level = "-O1"

        args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {opt_level}"
        args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        )
        args += " --auto-cast=none"
        args += " --internal-enable-dge-levels vector_dynamic_offsets"
        args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"
            )

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention, (
                "attn_block_tkg_nki_kernel_enabled requires attn_block_tkg_nki_kernel_cascaded_attention"
            )
            self.neuron_config.pre_rope_rmsnorm = True
            args += " --internal-max-instruction-limit=15000000"

        return args

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        """Load and convert state dict from a HuggingFace safetensors checkpoint."""
        import json
        import os

        from safetensors import safe_open

        if os.path.isdir(model_name_or_path):
            index_path = os.path.join(
                model_name_or_path, "model.safetensors.index.json"
            )
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index = json.load(f)

                model_sd: Dict[str, Any] = {}
                shard_files = sorted(set(index["weight_map"].values()))
                for i, shard_file in enumerate(shard_files):
                    if i % 20 == 0:
                        print(
                            f"  Loading shard {i + 1}/{len(shard_files)}: {shard_file}"
                        )
                    shard_path = os.path.join(model_name_or_path, shard_file)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            model_sd[key] = f.get_tensor(key)

                print(
                    f"  Loaded {len(model_sd)} parameters from {len(shard_files)} shards"
                )

                # Strip model. prefix
                for param_name in list(model_sd.keys()):
                    if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                        new_name = param_name.replace(
                            cls._STATE_DICT_MODEL_PREFIX,
                            cls._NEW_STATE_DICT_MODEL_PREFIX,
                            1,
                        )
                        model_sd[new_name] = model_sd.pop(param_name)

                model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)

                if getattr(config, "tie_word_embeddings", False):
                    cls.update_state_dict_for_tied_weights(model_sd)

                if cls._FUSED_PREFIX:
                    for param_name in list(model_sd.keys()):
                        model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd.pop(
                            param_name
                        )

                return model_sd
            else:
                from neuronx_distributed_inference.modules.checkpoint import (
                    load_state_dict,
                )

                return load_state_dict(model_name_or_path)
        else:
            return super().get_state_dict(model_name_or_path, config)
