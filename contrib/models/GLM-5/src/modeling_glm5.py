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
NeuronX Distributed Inference implementation for GLM-5 (zai-org/GLM-5).

Architecture:
- GLM-5: 754B MoE (40B active), 78 layers (3 dense + 75 MoE)
- MLA (Multi-head Latent Attention) with compressed KV cache (576 values/token)
- 256 routed experts, top-8 sigmoid routing with e_score_correction_bias
- 1 shared expert per MoE layer
- routed_scaling_factor = 2.5
- GLM-5 is architecturally identical to DeepSeek-V3 (vLLM: empty subclass)
- DSA (DeepSeek Sparse Attention) indexer SKIPPED (full-attention fallback)
- MTP (Multi-Token Prediction) layer SKIPPED (training-only)

Key differences from DeepSeek-V3:
- qk_nope_head_dim=192 (vs 128), v_head_dim=256 (vs 128), head_dim=64 (vs 128)
- q_lora_rank=2048 (vs 1536), hidden_size=6144 (vs 7168)
- 78 layers with 3 dense (vs 61 layers with 1 dense)
- rope_theta=1M (vs 10M), no YaRN scaling
- vocab_size=154880 (vs 129280)

Target: trn2.48xlarge, TP=64, EP=1, LNC=2, FP8 weights
"""

import copy
import gc
import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    manual_softmax,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import (
    calculate_num_cores_per_group,
)
from neuronx_distributed_inference.modules.generation.sampling import create_sampler
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.utils.distributed import get_tp_group

# MoE v2 module (required for MoE layers)
try:
    from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

    MOE_V2_AVAILABLE = True
except ImportError:
    MOE_V2_AVAILABLE = False

# DS-V3 RoPE utilities (reused for GLM-5 MLA)
from neuronx_distributed_inference.models.deepseek.rope_util import (
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
)

logger = logging.getLogger("Neuron")


# ---------------------------------------------------------------------------
# FP8 NaN clamping constants
# ---------------------------------------------------------------------------
# PyTorch float8_e4m3fn max = 448, but Neuron hardware treats exponent-15
# bytes as NaN. Must clamp to 240. Affects ~1.4-2.2% of bytes in practice.
FP8_E4M3_NEURON_MAX = 240.0


# ---------------------------------------------------------------------------
# Fused MoE TKG kernel patch for GLM-5 routing (Task 013)
# ---------------------------------------------------------------------------
# GLM-5 uses sigmoid routing with selection_bias (e_score_correction_bias)
# and routed_scaling_factor=2.5. The open-source nkilib (pip install -e)
# overrides the bundled nkilib in neuronx-cc via the sys.modules swap in
# nkilib/__init__.py. Our modified router_topk.py and moe_block_tkg.py
# add selection_bias and routed_scaling_factor support.
#
# We replace MoEFusedTKG._moe_fused_tkg_kernel entirely (per-instance) to:
# 1. Inject selection_bias, routed_scaling_factor, norm_topk_prob=True
# 2. Handle per_tensor_symmetric scalar FP8 scales (expand [1,1,1] -> [E,2,I]/[E,H])
#    without mutating parameters (which would break XLA tracing)


def _patch_fused_tkg_with_nkilib(moe_layers, config):
    """
    Replace MoEFusedTKG._moe_fused_tkg_kernel on each MoE layer to inject
    GLM-5 routing params and handle scalar FP8 scales.

    Args:
        moe_layers: List of (layer_idx, glm5_moe) tuples for MoE decoder layers.
        config: GLM5InferenceConfig
    """
    import types
    from neuronx_distributed.modules.moe.moe_fused_tkg import (
        moe_block_tkg_kernel,
        _convert_torch_dtype_to_nki_dtype,
        ExpertAffinityScaleMode,
        ROUTER_ACT_FN_MAPPING,
        get_kernel_activation_func_id,
        ACTFunc,
        ActFnType,
        DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    )

    patched_count = 0
    for layer_idx, glm5_moe in moe_layers:
        moe_module = glm5_moe.moe  # NxDI MoE wrapper
        fused_tkg = getattr(moe_module, "moe_fused_tkg", None)
        if fused_tkg is None or not hasattr(fused_tkg, "_moe_fused_tkg_kernel"):
            logger.warning(
                "Layer %d: No moe_fused_tkg._moe_fused_tkg_kernel, skipping",
                layer_idx,
            )
            continue

        # Capture GLM-5 routing params
        bias_buffer = glm5_moe.e_score_correction_bias
        scaling_factor = glm5_moe.routed_scaling_factor

        # Register selection_bias as a buffer on fused_tkg so it gets moved to
        # XLA device during tracing (closure-captured CPU tensors cause errors)
        fused_tkg.register_buffer("glm5_selection_bias", bias_buffer.data.clone())
        # Store scaling_factor as a Python float (not a tensor, no device issue)
        fused_tkg.glm5_routed_scaling_factor = float(scaling_factor)

        def _make_replacement_method():
            """Create a complete replacement for _moe_fused_tkg_kernel."""

            def replacement_moe_fused_tkg_kernel(self, hidden_states, residual=None):
                """
                Complete replacement for NxDI's _moe_fused_tkg_kernel that:
                1. Handles per_tensor_symmetric scalar scales
                2. Injects GLM-5 routing params (selection_bias, routed_scaling_factor)
                3. Overrides norm_topk_prob=True

                Based on NxDI 0.9.17334 MoEFusedTKG._moe_fused_tkg_kernel.
                """
                hidden_states_shape = hidden_states.shape
                router_mm_dtype = _convert_torch_dtype_to_nki_dtype(
                    self.config.router_mm_dtype
                )
                if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
                    expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
                else:
                    expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
                local_rank = self.expert_mlps.spmd_rank.get_rank()
                local_ep_rank = (
                    local_rank
                    // self.expert_mlps.moe_tensor_model_parallel_group.size()
                )
                grid = self.logical_nc_config
                (
                    shared_experts_gate_proj_weight,
                    shared_experts_up_proj_weight,
                    shared_experts_down_proj_weight,
                ) = self._slice_shared_experts_weights()

                def get_data(t):
                    return t.data if t is not None and hasattr(t, "data") else t

                router_mm_dtype = _convert_torch_dtype_to_nki_dtype(
                    self.router.weight_T.dtype
                )

                # Handle FP8 scales: expand scalar [1,1,1] to expected shapes
                # without mutating the parameter (create new tensors instead)
                gate_up_scale = None
                down_scale = None
                if self.config.quantized:
                    raw_gu_scale = self.expert_mlps.mlp_op.gate_up_proj.scale
                    raw_dn_scale = self.expert_mlps.mlp_op.down_proj.scale
                    E = self.num_local_experts

                    if raw_gu_scale is not None:
                        if raw_gu_scale.numel() == 1:
                            # Per-tensor symmetric: scalar -> [E, 2, I]
                            gu_weight = self.expert_mlps.mlp_op.gate_up_proj.weight
                            I = gu_weight.shape[-1] // 2
                            gate_up_scale = (
                                get_data(raw_gu_scale)
                                .flatten()[0]
                                .expand(E, 2, I)
                                .contiguous()
                            )
                        else:
                            gate_up_scale = get_data(raw_gu_scale.view(E, 2, -1))

                    if raw_dn_scale is not None:
                        if raw_dn_scale.numel() == 1:
                            # Per-tensor symmetric: scalar -> [E, H]
                            H = self.hidden_size
                            down_scale = (
                                get_data(raw_dn_scale)
                                .flatten()[0]
                                .expand(E, H)
                                .contiguous()
                            )
                        else:
                            down_scale = get_data(raw_dn_scale.view(E, -1))

                common_args = dict(
                    inp=get_data(hidden_states),
                    gamma=get_data(self.post_attention_layernorm.weight.unsqueeze(0)),
                    router_weights=get_data(self.router.weight_T),
                    shared_expert_gate_w=get_data(shared_experts_gate_proj_weight),
                    shared_expert_up_w=get_data(shared_experts_up_proj_weight),
                    shared_expert_down_w=get_data(shared_experts_down_proj_weight),
                    expert_gate_up_weights=get_data(
                        self.expert_mlps.mlp_op.gate_up_proj.weight.view(
                            self.num_local_experts, self.hidden_size, 2, -1
                        )
                    ),
                    expert_down_weights=get_data(
                        self.expert_mlps.mlp_op.down_proj.weight
                    ),
                    expert_gate_up_weights_scale=gate_up_scale,
                    expert_down_weights_scale=down_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    top_k=self.num_experts_per_tok,
                    router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    router_mm_dtype=router_mm_dtype,
                )

                if (
                    self.expert_mlps.routed_experts_mlp_config.hidden_size_actual
                    is not None
                ):
                    common_args["hidden_actual"] = (
                        self.expert_mlps.routed_experts_mlp_config.hidden_size_actual
                    )

                total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
                perc_experts_loaded = (
                    total_tokens * self.num_experts_per_tok / self.num_local_experts
                )

                kernel_call = moe_block_tkg_kernel
                is_all_expert = (
                    perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD
                )
                if is_all_expert:
                    logger.info(
                        "Percentage of experts loaded >= selective loading threshold, run forward all experts kernel"
                    )
                else:
                    logger.info("Run selective loading kernel")

                if kernel_call:
                    routed_experts_mlp_config = (
                        self.expert_mlps.routed_experts_mlp_config
                    )
                    kernel_activation_func_id = get_kernel_activation_func_id(
                        ACTFunc.validate(routed_experts_mlp_config.hidden_act),
                        routed_experts_mlp_config.glu_type,
                    )
                    optional_kwargs = {}
                    if routed_experts_mlp_config.gate_clamp_upper_limit is not None:
                        optional_kwargs["gate_clamp_upper_limit"] = (
                            routed_experts_mlp_config.gate_clamp_upper_limit
                        )
                    if routed_experts_mlp_config.gate_clamp_lower_limit is not None:
                        optional_kwargs["gate_clamp_lower_limit"] = (
                            routed_experts_mlp_config.gate_clamp_lower_limit
                        )
                    if routed_experts_mlp_config.up_clamp_upper_limit is not None:
                        optional_kwargs["up_clamp_upper_limit"] = (
                            routed_experts_mlp_config.up_clamp_upper_limit
                        )
                    if routed_experts_mlp_config.up_clamp_lower_limit is not None:
                        optional_kwargs["up_clamp_lower_limit"] = (
                            routed_experts_mlp_config.up_clamp_lower_limit
                        )

                    if is_all_expert:
                        optional_kwargs["rank_id"] = get_data(
                            local_ep_rank.reshape(1, 1)
                        )

                    # --- GLM-5 routing params ---
                    # Controlled by env var for isolation testing
                    if not os.environ.get("GLM5_SKIP_ROUTING_PARAMS"):
                        # selection_bias is registered as a buffer on self (fused_tkg)
                        # so it gets moved to XLA device during tracing
                        sel_bias = get_data(self.glm5_selection_bias)
                        optional_kwargs["selection_bias"] = sel_bias.unsqueeze(
                            0
                        )  # [E] -> [1, E]
                        optional_kwargs["routed_scaling_factor"] = (
                            self.glm5_routed_scaling_factor
                        )

                    out, router_logits = kernel_call[grid](
                        **common_args,
                        router_bias=get_data(self.router.linear_router.bias)
                        if self.router.bias
                        else None,
                        expert_gate_up_bias=get_data(
                            self.expert_mlps.mlp_op.gate_up_proj.bias.view(
                                self.num_local_experts, 2, -1
                            )
                        )
                        if routed_experts_mlp_config.bias
                        else None,
                        expert_down_bias=get_data(
                            self.expert_mlps.mlp_op.down_proj.bias
                        )
                        if routed_experts_mlp_config.bias
                        else None,
                        shared_expert_gate_bias=None,
                        shared_expert_up_bias=None,
                        shared_expert_down_bias=None,
                        router_pre_norm=not self.router.apply_act_fn_over_topk,
                        hidden_act_fn=ActFnType(kernel_activation_func_id),
                        hidden_act_scale_factor=None,
                        hidden_act_bias=None,
                        norm_topk_prob=True
                        if not os.environ.get("GLM5_SKIP_ROUTING_PARAMS")
                        else self.config.norm_topk_prob,  # GLM-5 override
                        is_all_expert=is_all_expert,
                        **optional_kwargs,
                    )

                return out.view(hidden_states_shape), router_logits.to(
                    hidden_states.dtype
                )

            return replacement_moe_fused_tkg_kernel

        # Bind the replacement method
        fused_tkg._moe_fused_tkg_kernel = types.MethodType(
            _make_replacement_method(),
            fused_tkg,
        )
        patched_count += 1
        logger.info(
            "Layer %d: Replaced _moe_fused_tkg_kernel with GLM-5 version "
            "(selection_bias + routed_scaling_factor=%.1f + scalar scale handling)",
            layer_idx,
            fused_tkg.glm5_routed_scaling_factor,
        )

    logger.info("Patched %d MoE layers with GLM-5 fused kernel", patched_count)


# ---------------------------------------------------------------------------
# Monkey-patch: Fix QuantizedExpertFused scale shapes for per_tensor_symmetric
# ---------------------------------------------------------------------------
# NxDI's QuantizedExpertFusedColumnParallel/RowParallel inherit _setup_for_scale
# from the base QuantizedColumnParallel/RowParallel. For PER_TENSOR_SYMMETRIC,
# scale is initialized as shape [1]. But the forward_selective_loading path in
# expert_mlps_v2.py indexes self.scale[expert_indices, :, :] (3D), which fails
# on a 1D tensor. Fix: reshape scale to [1, 1, 1] so 3D indexing works.
# This is safe because per-tensor symmetric uses a single scalar for all elements,
# and [1, 1, 1] broadcasts correctly in matmul dequantization.
def _patch_expert_fused_quantized_scale_shapes():
    """Patch QuantizedExpertFused* to create 3D scales for per_tensor_symmetric."""
    try:
        from neuronx_distributed.quantization.quantization_layers import (
            QuantizedExpertFusedColumnParallel,
            QuantizedExpertFusedRowParallel,
        )

        # Save original __init__ methods
        _orig_col_init = QuantizedExpertFusedColumnParallel.__init__
        _orig_row_init = QuantizedExpertFusedRowParallel.__init__

        def _patched_col_init(self, *args, **kwargs):
            _orig_col_init(self, *args, **kwargs)
            # After init, if scale is 1D and we have expert-fused 3D weights, reshape
            if (
                hasattr(self, "scale")
                and self.scale is not None
                and self.scale.dim() == 1
                and hasattr(self, "weight")
                and self.weight is not None
                and self.weight.dim() == 3
            ):
                old_scale = self.scale
                with torch.no_grad():
                    new_scale = nn.Parameter(
                        old_scale.data.view(1, 1, 1), requires_grad=False
                    )
                # Copy critical custom attributes set by NxDI's _setup_for_scale
                for attr_name in [
                    "get_tensor_from_state_dict",
                    "set_tensor_to_state_dict",
                    "tensor_model_parallel",
                    "partition_dim",
                    "partition_stride",
                    "num_partitions",
                    "rank_ordering",
                ]:
                    if hasattr(old_scale, attr_name):
                        setattr(new_scale, attr_name, getattr(old_scale, attr_name))
                self.scale = new_scale
                logger.info(
                    "Patched QuantizedExpertFusedColumnParallel scale: [1] -> [1, 1, 1]"
                )

        def _patched_row_init(self, *args, **kwargs):
            _orig_row_init(self, *args, **kwargs)
            if (
                hasattr(self, "scale")
                and self.scale is not None
                and self.scale.dim() == 1
                and hasattr(self, "weight")
                and self.weight is not None
                and self.weight.dim() == 3
            ):
                old_scale = self.scale
                with torch.no_grad():
                    new_scale = nn.Parameter(
                        old_scale.data.view(1, 1, 1), requires_grad=False
                    )
                for attr_name in [
                    "get_tensor_from_state_dict",
                    "set_tensor_to_state_dict",
                    "tensor_model_parallel",
                    "partition_dim",
                    "partition_stride",
                    "num_partitions",
                    "rank_ordering",
                ]:
                    if hasattr(old_scale, attr_name):
                        setattr(new_scale, attr_name, getattr(old_scale, attr_name))
                self.scale = new_scale
                logger.info(
                    "Patched QuantizedExpertFusedRowParallel scale: [1] -> [1, 1, 1]"
                )

        QuantizedExpertFusedColumnParallel.__init__ = _patched_col_init
        QuantizedExpertFusedRowParallel.__init__ = _patched_row_init
        logger.info("Monkey-patched QuantizedExpertFused* __init__ for 3D scale shapes")
    except ImportError as ie:
        logger.warning(
            "Could not import QuantizedExpertFused* classes, scale patch skipped: %s",
            ie,
        )
    except Exception as e:
        logger.warning("Failed to patch expert fused scale shapes: %s", e)


# Apply the patch at import time so it takes effect before convert() runs
_patch_expert_fused_quantized_scale_shapes()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_lm_head_pad_config(
    vocab_size: int,
    tp_degree: int,
    lm_head_pad_alignment_size: int = 1,
    skip_lm_head_pad: bool = False,
):
    """Check if lm_head padding is necessary for proper sharding."""
    if vocab_size % (tp_degree * lm_head_pad_alignment_size) == 0 or skip_lm_head_pad:
        return False, 1
    return True, lm_head_pad_alignment_size


def preshard_hook_fn(
    module: torch.nn.Module, model_state_dict: dict, prefix: str
) -> bool:
    from neuronx_distributed_inference.modules.attention.gqa import (
        BaseGroupQueryAttention,
    )

    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)
    return False


def get_rmsnorm_cls():
    """Return appropriate RMSNorm: CustomRMSNorm on Neuron, CPU fallback otherwise."""
    return GLM5RMSNorm if cpu_mode() else CustomRMSNorm


class GLM5RMSNorm(nn.Module):
    """CPU-compatible RMSNorm for GLM-5."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# FP8 Dequantization with NaN clamping
# ---------------------------------------------------------------------------


def _dequantize_fp8_blockwise(fp8_tensor, scales, block_size, target_dtype):
    """
    Dequantize a single FP8 blockwise-quantized tensor to target_dtype.

    Args:
        fp8_tensor: float8_e4m3fn weight tensor
        scales: float32 per-block scale_inv tensor
        block_size: [block_rows, block_cols]
        target_dtype: output dtype (e.g. torch.bfloat16)

    Returns:
        Dequantized tensor in target_dtype
    """
    # NaN clamp: clamp FP8 values to Neuron-safe range before dequant
    fp8_float = fp8_tensor.to(torch.float32)
    fp8_float = fp8_float.clamp(-FP8_E4M3_NEURON_MAX, FP8_E4M3_NEURON_MAX)

    # Expand block scales to match weight dimensions
    scales_expanded = scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(
        block_size[1], dim=1
    )

    # Truncate expanded scales if they exceed weight dimensions
    # (last block may be partial)
    if scales_expanded.shape[0] > fp8_float.shape[0]:
        scales_expanded = scales_expanded[: fp8_float.shape[0]]
    if scales_expanded.shape[1] > fp8_float.shape[1]:
        scales_expanded = scales_expanded[:, : fp8_float.shape[1]]

    # Dequantize: weight = fp8_value * scale
    dequantized = fp8_float * scales_expanded.to(torch.float32)
    return dequantized.to(target_dtype)


def _rescale_fp8_for_neuron(fp8_tensor, scale):
    """
    Rescale FP8 tensor from OCP E4M3 range (max 448) to Neuron E4M3 range (max 240).

    Following Llama 4 FP8 preprocessing pattern:
    1. Convert FP8 to BF16 intermediate
    2. Divide by FP8_SCALING_FACTOR = 448/240
    3. Re-cast to float8_e4m3fn
    4. Multiply scale by FP8_SCALING_FACTOR to compensate

    Args:
        fp8_tensor: float8_e4m3fn weight tensor
        scale: float32 scale tensor

    Returns:
        (rescaled_fp8, rescaled_scale) tuple
    """
    FP8_SCALING_FACTOR = 448.0 / 240.0
    fp8_bf16 = fp8_tensor.to(torch.bfloat16)
    rescaled_bf16 = fp8_bf16 / FP8_SCALING_FACTOR
    rescaled_fp8 = rescaled_bf16.to(torch.float8_e4m3fn)
    rescaled_scale = scale * FP8_SCALING_FACTOR
    return rescaled_fp8, rescaled_scale


def maybe_dequantize_fp8_with_nan_clamp(neuron_state_dict: dict, config):
    """
    Dequantize FP8 blockwise-quantized NON-EXPERT weights to BF16/FP32.

    Expert weights are handled separately in convert_hf_to_neuron_state_dict
    (kept as FP8 with per-expert scales for NxDI's quantized MoE path).

    This function only dequantizes:
    - Attention weights (q_a_proj, q_b_proj, kv_a/b_proj, o_proj)
    - Dense MLP weights (layers 0-2)
    - Shared expert weights
    - Other non-expert linear layers

    Expert weights (*.experts.*.{gate,up,down}_proj*) are skipped.

    Args:
        neuron_state_dict: State dict (modified in place)
        config: InferenceConfig with quantization_config
    """
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return

    block_size = quant_config.get("weight_block_size", None)
    if block_size is None:
        return

    target_dtype = config.neuron_config.torch_dtype
    scale_layers_to_delete = []

    for layer_key in list(neuron_state_dict.keys()):
        if not layer_key.endswith("_scale_inv"):
            continue

        fp8_layer_name = layer_key.replace("_scale_inv", "")
        if fp8_layer_name not in neuron_state_dict:
            continue

        # Skip expert weights -- they are handled separately (kept as FP8)
        if ".experts." in fp8_layer_name:
            continue

        fp8_tensor = neuron_state_dict[fp8_layer_name]
        scales = neuron_state_dict[layer_key]

        dequantized = _dequantize_fp8_blockwise(
            fp8_tensor, scales, block_size, target_dtype
        )
        neuron_state_dict[fp8_layer_name] = dequantized
        scale_layers_to_delete.append(layer_key)

    # Remove scale tensors for dequantized layers
    for key in scale_layers_to_delete:
        del neuron_state_dict[key]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GLM5InferenceConfig(InferenceConfig):
    """
    Inference config for GLM-5 (zai-org/GLM-5, model_type=glm_moe_dsa).

    Maps GLM-5 HF config fields to NxDI expectations. Handles:
    - MLA dimension fields (q_lora_rank, kv_lora_rank, qk_nope_head_dim, etc.)
    - MoE config (n_routed_experts, moe_intermediate_size, first_k_dense_replace)
    - Dense vs MoE layer dispatch
    - Sigmoid routing with e_score_correction_bias
    - routed_scaling_factor=2.5
    - No YaRN RoPE (simple RoPE with theta=1M)
    """

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    def __init__(self, *args, **kwargs):
        # NOTE: super().__init__() calls load_config -> add_derived_config -> validate_config
        # in that order. All field mappings that validation depends on MUST go in
        # add_derived_config() (not here), because this __init__ body runs AFTER
        # super().__init__() returns.
        super().__init__(*args, **kwargs)

        # --- Router and MoE config for NxDI ---
        # These neuron_config settings are NOT checked by validate_config,
        # so they can safely live here (after super().__init__()).
        self.neuron_config.glu_mlp = True
        self.neuron_config.glu_type = "glu"
        self.neuron_config.router_config.act_fn = "sigmoid"
        self.neuron_config.router_config.dtype = torch.bfloat16

        # No clamping, no scaling/bias on hidden activations
        self.neuron_config.hidden_act_scaling_factor = 1.0
        self.neuron_config.hidden_act_bias = 0
        self.neuron_config.gate_clamp_upper_limit = None
        self.neuron_config.gate_clamp_lower_limit = None
        self.neuron_config.up_clamp_upper_limit = None
        self.neuron_config.up_clamp_lower_limit = None
        # Do NOT normalize inside NxDI -- we handle normalization + scaling
        # in the patched router forward (need to apply routed_scaling_factor=2.5
        # AFTER normalization, which is impossible if NxDI normalizes internally)
        self.neuron_config.normalize_top_k_affinities = False
        self.neuron_config.transpose_shared_experts_weights = False
        self.neuron_config.early_expert_affinity_modulation = False

        # --- FP8 Quantization ---
        # CRITICAL: GLM-5 at BF16 has 26.67 GB NEFF I/O (78 layers, 256 experts)
        # which exceeds the 24 GB per-core HBM limit at LNC=2. By enabling NxDI's
        # native FP8 quantization, expert weights are stored as float8_e4m3fn (1 byte
        # each) instead of bfloat16 (2 bytes), reducing MoE I/O from 22.65 GB to
        # 11.33 GB and total to ~15 GB. The convert() function replaces:
        # - ExpertFusedColumnParallelLinear -> QuantizedExpertFusedColumnParallel
        # - ExpertFusedRowParallelLinear -> QuantizedExpertFusedRowParallel
        # Non-expert layers are excluded via modules_to_not_convert (kept BF16).
        # ModelWrapper also adds --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 to
        # compiler args when quantized=True + quantization_dtype=f8e4m3.
        if (
            not hasattr(self.neuron_config, "quantized")
            or not self.neuron_config.quantized
        ):
            self.neuron_config.quantized = True
            self.neuron_config.quantization_dtype = "f8e4m3"

        # --- Modules to NOT quantize ---
        # Only MoE expert-fused layers need FP8. All other parallel layers
        # (attention projections, dense MLP, shared experts, lm_head) stay BF16.
        # The GLM-5-FP8 checkpoint has blockwise FP8 for all linear weights,
        # but convert_hf_to_neuron_state_dict dequantizes non-expert weights
        # back to BF16. If we don't exclude these from convert(), the
        # QuantizedColumnParallel/RowParallel layers expect .scale tensors
        # that don't exist in the state dict (RuntimeError: Cannot find
        # lm_head.scale in state_dict).
        # Uses substring matching: "self_attn" matches layers.*.self_attn.*.
        # "mlp" would also match expert_mlps inside MoE, so we use specific
        # layer indices for the 3 dense layers.
        if not getattr(self.neuron_config, "modules_to_not_convert", None):
            first_k = getattr(self, "first_k_dense_replace", 3)
            if not hasattr(self, "first_k_dense_replace"):
                # Before add_derived_config runs, try raw HF config
                first_k = 3
            self.neuron_config.modules_to_not_convert = [
                "lm_head",
                "self_attn",
                "shared_expert",
            ] + [f"layers.{i}.mlp" for i in range(first_k)]

        # --- Blockwise matmul config ---
        # CRITICAL: At TP=64, expert intermediate_size/TP = 2048/64 = 32, which is
        # smaller than the minimum blockwise matmul block_size of 128. The blockwise
        # NKI kernel in ExpertMLPsV2.forward_blockwise() asserts block_size in [128,256].
        # Force block_size to a very large value to bypass forward_blockwise entirely
        # and use forward_all_experts instead for context encoding.
        if hasattr(self.neuron_config, "blockwise_matmul_config"):
            self.neuron_config.blockwise_matmul_config.block_size = 2**30

    def add_derived_config(self):
        """
        Called by super().__init__() AFTER load_config but BEFORE validate_config.
        All field mappings and defaults that validation depends on go here.
        """
        # --- Flash decoding ---
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                self.num_attention_heads,
                # For MLA, KV heads = num_attention_heads (all heads share compressed KV)
                self.num_attention_heads,
                self.neuron_config.tp_degree,
            )

        # --- MLA dimensions ---
        # These come directly from HF config (glm_moe_dsa).
        # Use getattr with defaults in case any are missing.
        self.q_lora_rank = getattr(self, "q_lora_rank", 2048)
        self.kv_lora_rank = getattr(self, "kv_lora_rank", 512)
        self.qk_nope_head_dim = getattr(self, "qk_nope_head_dim", 192)
        self.qk_rope_head_dim = getattr(self, "qk_rope_head_dim", 64)
        self.v_head_dim = getattr(self, "v_head_dim", 256)
        # --- DSA (DeepSeek Sparse Attention) config ---
        self.index_n_heads = getattr(self, "index_n_heads", 32)
        self.index_head_dim = getattr(self, "index_head_dim", 128)
        self.index_topk = getattr(self, "index_topk", 2048)
        self.indexer_rope_interleave = getattr(self, "indexer_rope_interleave", True)
        # DSA enabled by default when index_topk > 0
        if not hasattr(self, "dsa_enabled"):
            self.dsa_enabled = self.index_topk > 0

        # head_dim controls KV cache shape via _get_hidden_dim_per_head().
        # For MLA, KV cache stores concatenated [k_pe | compressed_kv] per token,
        # so head_dim = kv_lora_rank + qk_rope_head_dim = 576.
        # When DSA is enabled, we also store the indexer key (index_head_dim=128)
        # in the same cache slot: head_dim = 576 + 128 = 704.
        # This overrides the HF config's head_dim=64 (which is the output head dim).
        mla_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 512 + 64 = 576
        if self.dsa_enabled:
            self.head_dim = mla_cache_dim + self.index_head_dim  # 576 + 128 = 704
        else:
            self.head_dim = mla_cache_dim  # 576

        # --- Layer structure ---
        self.first_k_dense_replace = getattr(self, "first_k_dense_replace", 3)
        # dense_intermediate_size: the intermediate size for dense MLP layers (0-2).
        # CRITICAL: Do NOT derive this from self.intermediate_size because:
        # - At compile time: intermediate_size=12288 (from HF config), then we
        #   overwrite it to 2048 (MoE) below. So reading it here gives 12288. OK.
        # - At load from JSON: intermediate_size=2048 (already overwritten in
        #   serialized config). Reading it here gives 2048. WRONG!
        # Solution: only set dense_intermediate_size if not already set (e.g. from
        # JSON deserialization). If it needs to be derived, use the HF-original value
        # which is available as 'intermediate_size' before we overwrite it at compile
        # time, or from 'dense_intermediate_size' in the JSON at load time.
        if (
            not hasattr(self, "dense_intermediate_size")
            or self.dense_intermediate_size is None
        ):
            # First compile: intermediate_size is still the HF original (12288)
            self.dense_intermediate_size = getattr(self, "intermediate_size", 12288)
        # else: already set from JSON deserialization or previous call

        # --- MoE config ---
        # Map HF field names to NxDI expected names
        if not hasattr(self, "num_local_experts"):
            self.num_local_experts = getattr(self, "n_routed_experts", 256)
        if not hasattr(self, "num_experts_per_tok"):
            self.num_experts_per_tok = getattr(self, "num_experts_per_tok", 8)

        # MoE intermediate size: NxDI reads config.intermediate_size for expert MLP
        moe_intermediate = getattr(self, "moe_intermediate_size", 2048)
        self.intermediate_size = moe_intermediate
        self.moe_intermediate_size = moe_intermediate

        # Shared experts: disable NxDI's built-in handling, we manage it ourselves.
        # CRITICAL: Guard with hasattr — at load-from-JSON time, num_shared_experts_actual
        # is already deserialized (=1) from neuron_config.json. Without the guard,
        # getattr(self, "n_shared_experts", 1) returns 0 (also from JSON) and overwrites it.
        # Same pattern as dense_intermediate_size fix (discovery #31).
        if not hasattr(self, "num_shared_experts_actual"):
            self.num_shared_experts_actual = getattr(self, "n_shared_experts", 1)
        self.n_shared_experts = 0

        # Routing config
        self.routed_scaling_factor = getattr(self, "routed_scaling_factor", 2.5)

        # --- RoPE ---
        # GLM-5: simple RoPE with theta=1M, no YaRN.
        # CRITICAL: rope_theta is nested inside rope_parameters in HF config.json,
        # NOT a top-level key. The load_config lambda only sets top-level keys,
        # so we must extract it from the nested dict.
        if not hasattr(self, "rope_theta"):
            rope_params = getattr(self, "rope_parameters", None)
            if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                self.rope_theta = rope_params["rope_theta"]
            else:
                self.rope_theta = 1000000  # GLM-5 default
                logger.warning(
                    "rope_theta not found in config or rope_parameters, "
                    "using default 1000000"
                )

        # --- Misc defaults ---
        self.rms_norm_eps = getattr(self, "rms_norm_eps", 1e-05)
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        self.attention_bias = getattr(self, "attention_bias", False)

        # Standard HF config attributes expected by NeuronBaseModel.forward()
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True
        if not hasattr(self, "return_dict"):
            self.return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            "num_hidden_layers",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "hidden_size",
            "moe_intermediate_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "rope_theta",
            "pad_token_id",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
        ]

    def validate_config(self):
        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        return json.dumps(config_dict, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# DSA (DeepSeek Sparse Attention) Indexer
# ---------------------------------------------------------------------------


class GLM5DSAIndexer(nn.Module):
    """
    DeepSeek Sparse Attention Indexer for GLM-5.

    Computes importance scores for each KV position using lightweight side-channel
    attention with 32 index heads (dim=128). Selects top-2048 positions per query
    token, producing a sparse attention mask for the main MLA attention.

    Architecture:
    - wq_b: projects Q LoRA output (2048) -> 32 * 128 = 4096 (index query heads)
    - wk: projects hidden_states (6144) -> 128 (shared index key)
    - weights_proj: projects hidden_states (6144) -> 32 (per-head learned weights)
    - k_norm: LayerNorm(128) on index keys

    Scoring formula:
        score[b,s,t] = sum_h( weight[b,s,h] * softmax_scale * ReLU(q[b,s,h] . k[b,t]) )
        final_score = score * n_heads^{-0.5}
        mask = top-k(final_score, k=2048) -> 0.0 at selected, -inf elsewhere

    The indexer key cache is embedded in the main MLA KV cache (last 128 dims).
    """

    def __init__(self, config: "GLM5InferenceConfig"):
        super().__init__()
        self.n_heads = config.index_n_heads  # 32
        self.head_dim = config.index_head_dim  # 128
        self.topk = config.index_topk  # 2048
        self.q_lora_rank = config.q_lora_rank  # 2048
        self.hidden_size = config.hidden_size  # 6144
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64

        self.softmax_scale = self.head_dim ** (-0.5)  # 128^{-0.5}
        self.head_scale = self.n_heads ** (-0.5)  # 32^{-0.5}

        dtype = config.neuron_config.torch_dtype

        # Index Q projection: q_lora_rank -> n_heads * head_dim
        # Input: output of q_a_layernorm (shared with main Q path)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wq_b.weight = nn.Parameter(
            torch.zeros(self.n_heads * self.head_dim, self.q_lora_rank, dtype=dtype)
        )

        # Index K projection: hidden_size -> head_dim
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.wk.weight = nn.Parameter(
            torch.zeros(self.head_dim, self.hidden_size, dtype=dtype)
        )

        # Per-head weight projection: hidden_size -> n_heads
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.weights_proj.weight = nn.Parameter(
            torch.zeros(self.n_heads, self.hidden_size, dtype=dtype)
        )

        # Key normalization (LayerNorm with bias, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        # RoPE for indexer (uses split-half / NeoX style)
        # The indexer RoPE uses the same theta as the main model but only
        # over the first qk_rope_head_dim (64) dimensions of the 128-dim key/query.
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora_output: torch.Tensor,
        position_ids: torch.Tensor,
        cached_index_keys: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DSA sparse attention mask.

        Args:
            hidden_states: [B, S, 6144] - pre-norm hidden states
            q_lora_output: [B, S, 2048] - output of q_a_layernorm (shared with main Q)
            position_ids: [B, S] - position indices
            cached_index_keys: [B, 1, cache_len, 128] - cached indexer keys (from KV cache)
                               or None for prefill
            attention_mask: [B, 1, S, T] - causal attention mask (True=attend, for prefill)
                           or [B, 1, 1, T] for decode
            cos_cache: pre-computed cos for RoPE
            sin_cache: pre-computed sin for RoPE

        Returns:
            index_key: [B, S, 128] - new indexer keys to cache
            dsa_mask: [B, 1, S, T] - sparse mask (0.0 for selected positions, -inf for masked)
                      Returns None if seq_len <= topk (all positions selected = no sparsity)
        """
        bsz, q_len, _ = hidden_states.shape

        # --- Index Key (always needed: stored in KV cache) ---
        index_k = self.wk(hidden_states)  # [B, S, 128]
        index_k = self.k_norm(index_k)  # [B, S, 128]

        # Split K into rope part and non-rope part
        k_pe = index_k[:, :, : self.qk_rope_head_dim]  # [B, S, 64]
        k_nope = index_k[:, :, self.qk_rope_head_dim :]  # [B, S, 64]

        # Apply RoPE to K_pe
        k_pe_4d = k_pe.unsqueeze(1)  # [B, 1, S, 64]

        seq_len = q_len
        if cached_index_keys is not None:
            seq_len = cached_index_keys.shape[2] + q_len

        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe_4d, seq_len)
        k_pe_4d = apply_rotary_pos_emb(k_pe_4d, cos_cache, sin_cache, position_ids)
        k_pe = k_pe_4d.squeeze(1)  # [B, S, 64]

        # Reassemble K with RoPE applied to positional part
        # index_k_new: [B, S, 128] with [rope(64) | nope(64)]
        index_k_new = torch.cat([k_pe, k_nope], dim=-1)  # [B, S, 128]

        # --- Build full key sequence (cache + new) ---
        if cached_index_keys is not None:
            # Decode: cached_index_keys [B, 1, cache_len, 128]
            cached_k = cached_index_keys.squeeze(1)  # [B, cache_len, 128]
            all_keys = torch.cat([cached_k, index_k_new], dim=1)  # [B, T, 128]
        else:
            # Prefill: no cache, all keys are from current input
            all_keys = index_k_new  # [B, S, 128]

        total_len = all_keys.shape[1]  # T

        # Early return: if total sequence length <= topk, all positions are selected.
        # This avoids tracing the Q projection, score matmul, and weight projection
        # into the XLA graph when they would be dead code.
        # At seq_len=2048 with topk=2048, this is always True (no sparsity).
        if total_len <= self.topk:
            return index_k_new, None

        # --- Index Query (only needed for scoring) ---
        # q_lora_output is already normalized (shared path with main Q)
        index_q = self.wq_b(q_lora_output)  # [B, S, 4096]
        index_q = index_q.view(bsz, q_len, self.n_heads, self.head_dim)
        index_q = index_q.transpose(1, 2)  # [B, 32, S, 128]

        # Split Q into rope part and non-rope part
        q_pe = index_q[:, :, :, : self.qk_rope_head_dim]  # [B, 32, S, 64]
        q_nope = index_q[:, :, :, self.qk_rope_head_dim :]  # [B, 32, S, 64]

        # Apply RoPE to Q_pe
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)

        # Reassemble Q with RoPE
        index_q = torch.cat([q_pe, q_nope], dim=-1)  # [B, 32, S, 128]

        # --- Per-head weights (only needed for scoring) ---
        weights = self.weights_proj(hidden_states)  # [B, S, 32]

        # --- Compute per-head scores ---
        # Q: [B, 32, S, 128], K: [B, T, 128] -> scores: [B, 32, S, T]
        # Expand K to broadcast over heads: [B, 1, T, 128]
        all_keys_4d = all_keys.unsqueeze(1)  # [B, 1, T, 128]
        scores = torch.matmul(index_q, all_keys_4d.transpose(2, 3))  # [B, 32, S, T]
        scores = scores * self.softmax_scale  # scale by 128^{-0.5}
        scores = torch.relu(scores)  # ReLU activation

        # --- Weighted sum across heads ---
        # weights: [B, S, 32] -> [B, S, 32, 1] for broadcasting
        weights_4d = weights.unsqueeze(-1)  # [B, S, 32, 1]
        # scores: [B, 32, S, T] -> [B, S, 32, T] for element-wise multiply
        scores_transposed = scores.permute(0, 2, 1, 3)  # [B, S, 32, T]
        # Weighted sum: [B, S, 32, T] * [B, S, 32, 1] -> sum over heads -> [B, S, T]
        index_scores = (scores_transposed * weights_4d).sum(dim=2)  # [B, S, T]
        index_scores = index_scores * self.head_scale  # scale by 32^{-0.5}

        # --- Top-k selection and mask construction ---

        # Select top-k positions per query
        _, topk_indices = torch.topk(index_scores, k=self.topk, dim=-1)  # [B, S, 2048]

        # Build sparse mask: -inf everywhere, then 0.0 at selected positions
        dsa_mask = torch.full(
            (bsz, q_len, total_len),
            float("-inf"),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dsa_mask.scatter_(-1, topk_indices, 0.0)

        # Expand to [B, 1, S, T] for broadcasting with attention heads
        dsa_mask = dsa_mask.unsqueeze(1)  # [B, 1, S, T]

        return index_k_new, dsa_mask


# ---------------------------------------------------------------------------
# MLA Attention (adapted from DeepseekV3Attention)
# ---------------------------------------------------------------------------


class GLM5Attention(NeuronAttentionBase):
    """
    Multi-head Latent Attention for GLM-5.

    Adapted from NxDI DeepseekV3Attention with GLM-5 dimensions:
    - qk_nope_head_dim=192 (vs DS-V3: 128)
    - v_head_dim=256 (vs DS-V3: 128)
    - q_lora_rank=2048 (vs DS-V3: 1536)
    - head_dim=64 (output, vs DS-V3: 128)
    - hidden_size=6144 (vs DS-V3: 7168)
    - Simple RoPE with theta=1M (no YaRN)

    Uses weight absorption for efficient MLA:
    - q_nope absorbed with kv_b_proj[:qk_nope_head_dim] to avoid materializing k_nope
    - v absorbed with kv_b_proj[qk_nope_head_dim:] to compute output directly from compressed KV
    - KV cache stores only 576 values per token (512 compressed + 64 rope)
    """

    def __init__(
        self,
        config: GLM5InferenceConfig,
        layer_idx: Optional[int] = None,
        tensor_model_parallel_group=None,
    ):
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            # For MLA, set num_key_value_heads = num_attention_heads
            # (not applicable, compressed KV is shared across all heads)
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.v_head_dim,  # Output dimension per head
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Simple RoPE (no YaRN) with theta=1M
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Override qkv_proj from base class (MLA uses separate projections)
        self.qkv_proj = None
        self.bias = getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        assert layer_idx is not None, "layer_idx required for GLM5Attention"

        self.attention_dropout = (
            config.attention_dropout if hasattr(config, "attention_dropout") else 0.0
        )
        self.num_total_heads = config.num_attention_heads
        assert self.num_total_heads % self.tp_degree == 0, (
            f"num_attention_heads ({self.num_total_heads}) must be divisible by tp_degree ({self.tp_degree})"
        )
        if cpu_mode():
            self.num_heads = self.num_total_heads
        else:
            self.num_heads = self.num_total_heads // self.config.neuron_config.tp_degree

        # MLA dimensions
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = (
            config.qk_nope_head_dim + config.qk_rope_head_dim
        )  # 192 + 64 = 256

        # head_dim for output projection is v_head_dim (256)
        self.head_dim = self.v_head_dim

        self.is_causal = True
        self.init_mla_properties()

        # Softmax scale based on q_head_dim (256)
        # GLM-5 has no YaRN mscale, just simple 1/sqrt(q_head_dim)
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # DSA Indexer
        self.dsa_enabled = getattr(config, "dsa_enabled", False)
        if self.dsa_enabled:
            self.indexer = GLM5DSAIndexer(config)
            self.index_head_dim = config.index_head_dim  # 128
        else:
            self.indexer = None
            self.index_head_dim = 0

    def init_mla_properties(self):
        """Initialize MLA-specific projections (Q LoRA, KV compression, output)."""
        config = self.config
        dtype = self.torch_dtype

        # Q path: x -> q_a_proj (down) -> RMSNorm -> q_b_proj (up to heads*q_head_dim)
        # q_lora_rank is always set for GLM-5 (2048)
        if self.q_lora_rank is None:
            # Fallback: direct projection (not used for GLM-5 but kept for robustness)
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_total_heads * self.q_head_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                dtype=dtype,
            )
            self.q_a_layernorm = get_rmsnorm_cls()(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank,
                self.num_total_heads * self.q_head_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )

        # KV path: x -> kv_a_proj_with_mqa (down to kv_lora_rank + qk_rope_head_dim)
        # -> split into compressed_kv and k_pe
        # -> kv_b_proj expands compressed_kv to heads*(qk_nope_head_dim + v_head_dim)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.kv_a_layernorm = get_rmsnorm_cls()(config.kv_lora_rank)

        # kv_b_proj output: per head, qk_nope_head_dim (for K) + v_head_dim (for V)
        # = 192 + 256 = 448 per head, * 64 heads = 28672 total
        kv_b_out_dim = self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if self.tensor_model_parallel_group is not None:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                kv_b_out_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
        else:
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank,
                kv_b_out_dim,
                bias=False,
            )

        # Output projection: v_head_dim * num_heads -> hidden_size
        # Note: head_dim for o_proj is v_head_dim (256), NOT config.head_dim (64)
        if self.tensor_model_parallel_group is not None:
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.v_head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=True,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
                reduce_dtype=self.rpl_reduce_dtype,
            )
        else:
            self.o_proj = nn.Linear(
                self.num_attention_heads * self.v_head_dim,
                self.hidden_size,
                bias=self.bias,
            )

        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: torch.Tensor = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        MLA forward pass with weight absorption and optional DSA sparse attention.

        Weight absorption avoids materializing the full K/V tensors:
        - Instead of: Q_nope @ K_nope^T, we do: (Q_nope @ W_kv_b_k^T) @ compressed_kv^T
        - Instead of: softmax @ V, we do: (softmax @ compressed_kv) @ W_kv_b_v

        When DSA is enabled:
        - Computes sparse attention mask via the indexer (top-2048 positions)
        - DSA mask is combined with the causal mask before softmax
        - Indexer keys are stored in the last 128 dims of the KV cache

        Supports per-layer KV cache management (layer_boundary_markers mode):
        - get_kv_per_layer: fetch past_key_value from kv_mgr for this layer
        - update_kv_per_layer: store new KV into kv_mgr after attention
        """
        # Per-layer KV cache support (for layer_boundary_markers=True)
        get_kv_per_layer = kwargs.get("get_kv_per_layer", False)
        update_kv_per_layer = kwargs.get("update_kv_per_layer", False)
        kv_mgr = kwargs.get("kv_mgr", None)

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        if (
            self.sequence_parallel_enabled
            and self.tensor_model_parallel_group is not None
        ):
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        bsz, q_len, _ = hidden_states.size()

        # MLA cache dimension (without indexer keys)
        mla_cache_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 64 + 512 = 576

        # Weight matrix absorption: extract K-nope and V absorption matrices from kv_b_proj
        # wkv_b per-head layout: [k_nope(qk_nope_head_dim) | value(v_head_dim)]
        # Reference: HF DeepSeek-V3 kv_b_proj splits as [k_nope, value]
        # See: test_helper/reference_model.py lines 248, 258, 272
        #
        # IMPORTANT: The NxDI DS-V3 code uses wkv_b[:, :qk_nope_head_dim] and
        # wkv_b[:, v_head_dim:] which only works when qk_nope_head_dim == v_head_dim
        # (both 128 in DS-V3). For GLM-5 (192 != 256) we use the correct slicing:
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
        # [H, qk_nope_head_dim + v_head_dim, kv_lora_rank]

        q_absorb = wkv_b[:, : self.qk_nope_head_dim, :]  # [H, 192, C] -- K-nope weights
        v_absorb = wkv_b[:, self.qk_nope_head_dim :, :]  # [H, 256, C] -- V weights

        # Q projection (also produces q_lora_output for DSA indexer)
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
            q_lora_output = None
        else:
            q_a = self.q_a_proj(hidden_states)
            q_lora_output = self.q_a_layernorm(q_a)  # shared with DSA indexer
            q = self.q_b_proj(q_lora_output)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # KV compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        q_nope, q_pe = torch.tensor_split(q, (self.qk_nope_head_dim,), dim=-1)
        compressed_kv, k_pe = torch.tensor_split(
            compressed_kv, (self.kv_lora_rank,), dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Q absorption: transform q_nope from qk_nope space to kv_lora_rank space
        q_nope = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

        # RoPE
        seq_len = self.neuron_config.seq_len
        if sin_cache is None and cos_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos_cache, sin_cache, position_ids)

        # --- DSA Indexer: compute sparse mask ---
        index_key_new = None
        dsa_mask = None
        if self.dsa_enabled and self.indexer is not None and q_lora_output is not None:
            # Extract cached indexer keys from past_key_value (if decode)
            cached_index_keys = None
            if past_key_value is not None:
                cached_kv_full = past_key_value[0]  # [B, 1, cache_len, 704]
                # Indexer keys are the last index_head_dim (128) dims
                cached_index_keys = cached_kv_full[
                    :, :, :, mla_cache_dim:
                ]  # [B, 1, cache_len, 128]

            index_key_new, dsa_mask = self.indexer(
                hidden_states=hidden_states,
                q_lora_output=q_lora_output,
                position_ids=position_ids,
                cached_index_keys=cached_index_keys,
                attention_mask=attention_mask,
            )
            # index_key_new: [B, q_len, 128]
            # dsa_mask: [B, 1, q_len, T] or None (if T <= topk)

        # Attention scores: rope part + nope part (absorbed)
        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum(
            "bhqc,blc->bhql", q_nope, compressed_kv
        )
        active_scores *= self.softmax_scale

        if past_key_value is None:
            # Context encoding (prefill)
            # Apply DSA mask to attention scores if available
            if dsa_mask is not None:
                # Combine causal mask with DSA mask: positions must pass BOTH
                # attention_mask is True where attend is allowed (bool mask for prefill)
                # dsa_mask is 0.0 for selected, -inf for masked (additive mask)
                # Convert bool mask to additive, combine with dsa_mask, then apply
                causal_additive = torch.where(
                    attention_mask,
                    torch.zeros_like(active_scores),
                    torch.full_like(
                        active_scores, torch.finfo(active_scores.dtype).min
                    ),
                )
                combined_mask = causal_additive + dsa_mask
                active_scores = active_scores + combined_mask
            else:
                active_scores = torch.where(
                    attention_mask,
                    active_scores,
                    torch.finfo(active_scores.dtype).min,
                )
            active_scores = nn.functional.softmax(
                active_scores, dim=-1, dtype=torch.float32
            ).to(k_pe.dtype)

            # V absorption: compressed_kv -> v_head_dim space
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, v_absorb)
        else:
            # Token generation (decode) with KV cache
            # past_key_value is [k_cache, v_cache] from KVCacheManager.
            # k_cache: [B, 1, cache_len, 704] = [k_pe(64) | compressed_kv(512) | index_key(128)]
            # v_cache: [B, 1, cache_len, 704] = dummy (unused for MLA)
            cached_kv_full = past_key_value[0]  # [B, 1, cache_len, 704]
            # Split: MLA part (first 576 dims) and indexer part (last 128 dims, already extracted)
            cached_mla = cached_kv_full[
                :, :, :, :mla_cache_dim
            ]  # [B, 1, cache_len, 576]
            k_pe_prior = cached_mla[
                :, :, :, : self.qk_rope_head_dim
            ]  # [B, 1, cache_len, 64]
            compressed_kv_prior = cached_mla[
                :, :, :, self.qk_rope_head_dim :
            ]  # [B, 1, cache_len, 512]
            # Squeeze the KV head dim for einsum compatibility
            compressed_kv_prior = compressed_kv_prior.squeeze(1)  # [B, cache_len, 512]

            # Scores for prior (cached) tokens
            prior_scores = torch.matmul(
                q_pe, k_pe_prior.transpose(2, 3)
            ) + torch.einsum("bhqc,blc->bhql", q_nope, compressed_kv_prior)
            prior_scores *= self.softmax_scale

            # Apply DSA mask to prior scores (if available)
            if dsa_mask is not None:
                # dsa_mask: [B, 1, 1, T] where T = cache_len + 1
                # We only need the cache_len part for prior_scores
                dsa_mask_prior = dsa_mask[:, :, :, : prior_scores.shape[-1]]
                # Combine: attention_mask handles causal/padding, dsa_mask adds sparsity
                prior_scores = torch.where(
                    attention_mask,
                    prior_scores + dsa_mask_prior,
                    torch.finfo(prior_scores.dtype).min,
                )
            else:
                prior_scores = torch.where(
                    attention_mask,
                    prior_scores,
                    torch.finfo(prior_scores.dtype).min,
                )
            prior_scores = prior_scores.to(torch.float32)

            softmax_prior, softmax_active = manual_softmax(
                prior_scores, active_scores, is_speculation=False
            )
            softmax_prior = softmax_prior.to(k_pe.dtype)
            softmax_active = softmax_active.to(k_pe.dtype)

            # V absorption for active and prior
            x = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
            attn_active = torch.einsum("bhqc,hdc->bhqd", x, v_absorb)

            x = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
            attn_prior = torch.einsum("bhqc,hdc->bhqd", x, v_absorb)

            attn_output = attn_prior + attn_active

        # Reshape: BHSD -> BSHD -> BS(H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # KV cache return: concatenate [k_pe | compressed_kv | index_key] into 4D format
        # for KVCacheManager. Shape: [B, 1, q_len, 704] (or 576 if DSA disabled)
        # k_pe: [B, num_heads, q_len, qk_rope_head_dim] -> [B, q_len, qk_rope_head_dim]
        k_pe_flat = (
            k_pe.squeeze(1) if k_pe.dim() == 4 and k_pe.shape[1] == 1 else k_pe[:, 0]
        )  # [B, q_len, 64]
        concat_kv = torch.cat([k_pe_flat, compressed_kv], dim=-1)  # [B, q_len, 576]

        # Append indexer keys to cache if DSA enabled
        if self.dsa_enabled and index_key_new is not None:
            concat_kv = torch.cat([concat_kv, index_key_new], dim=-1)  # [B, q_len, 704]
        elif self.dsa_enabled:
            # DSA enabled but no indexer keys computed (shouldn't happen normally)
            # Pad with zeros to maintain consistent cache shape
            pad = torch.zeros(
                bsz,
                q_len,
                self.index_head_dim,
                dtype=concat_kv.dtype,
                device=concat_kv.device,
            )
            concat_kv = torch.cat([concat_kv, pad], dim=-1)  # [B, q_len, 704]

        concat_kv_4d = concat_kv.unsqueeze(1)  # [B, 1, q_len, 704]
        # Dummy V cache (same shape, will be ignored on read)
        dummy_v = torch.zeros_like(concat_kv_4d)
        past_key_value = (concat_kv_4d, dummy_v)

        # Per-layer KV cache update (for layer_boundary_markers=True)
        if update_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=past_key_value,
                position_ids=position_ids,
                **kwargs,
            )

        return attn_output, past_key_value, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Dense MLP (for layers 0 to first_k_dense_replace-1)
# ---------------------------------------------------------------------------


class GLM5DenseMLP(nn.Module):
    """
    Standard SwiGLU MLP for dense layers (layers 0, 1, 2 in GLM-5).

    Uses the dense_intermediate_size (12288), not the MoE intermediate_size (2048).
    """

    def __init__(self, config: GLM5InferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.dense_intermediate_size

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Shared Expert
# ---------------------------------------------------------------------------


class GLM5SharedExpert(nn.Module):
    """
    Shared expert for GLM-5 MoE layers.

    Uses moe_intermediate_size * n_shared_experts = 2048 * 1 = 2048 intermediate.
    Separate gate/up/down projections with SwiGLU activation.
    """

    def __init__(self, config: GLM5InferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_shared = getattr(config, "num_shared_experts_actual", 1)
        intermediate_size = config.moe_intermediate_size * num_shared

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# MoE Module
# ---------------------------------------------------------------------------


class GLM5MoE(nn.Module):
    """
    GLM-5 MoE module wrapping NxDI's initialize_moe_module.

    Key behaviors:
    - Sigmoid routing with e_score_correction_bias applied POST-sigmoid for selection
    - routed_scaling_factor=2.5 applied to normalized expert weights
    - Bias used for top-k selection but NOT for the actual expert weights
    - Same pattern as Solar-Open contrib
    """

    def __init__(
        self, config: GLM5InferenceConfig, rmsnorm: Optional[nn.Module] = None
    ):
        super().__init__()

        assert MOE_V2_AVAILABLE, "MoE v2 module required for GLM-5"

        self.routed_scaling_factor = config.routed_scaling_factor

        self.moe = initialize_moe_module(
            config=config,
            rmsnorm=rmsnorm,
            init_tkg_module=not config.neuron_config.on_cpu,
            router_bias=False,  # No bias in linear -- we handle it post-sigmoid
            experts_bias=False,  # GLM-5 experts have no bias
            apply_act_fn_over_topk=False,
        )

        # e_score_correction_bias buffer (loaded during weight conversion)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )

        # Patch the router to apply bias post-sigmoid for selection + scaling factor
        self._patch_router()

    def _patch_router(self):
        """
        Patch MoE router for GLM-5 routing logic.

        HF GLM-5 routing:
        1. router_logits = W @ x (no bias)
        2. affinities = sigmoid(router_logits)
        3. selection_scores = affinities + e_score_correction_bias
        4. top_k on selection_scores
        5. weights = affinities[top_k_indices] (un-biased)
        6. normalize: weights /= sum(weights) + 1e-20
        7. scale: weights *= routed_scaling_factor (2.5)

        We set normalize_top_k_affinities=False in the config and handle
        normalization + scaling entirely here. The NxDI ExpertMLPs module
        will use the expert_affinities directly as weights.

        The key insight: we return full expert_affinities (all experts), and
        the NxDI module gathers at expert_index internally. So we need to
        pre-compute the weights such that when NxDI gathers at the selected
        indices, the values are already normalized and scaled.

        Since NxDI gathers affinities[expert_index] to get per-token weights,
        we cannot normalize per-token here (we'd need to know which experts
        are selected). But expert_index IS computed here. So we compute the
        correct per-token normalized+scaled weights and scatter them back into
        the full affinity tensor.
        """
        router = self.moe.router
        moe_module = self

        def patched_router_forward(hidden_states):
            # Step 1: Raw logits (no bias)
            router_logits = router.get_router_logits(hidden_states)

            # Step 2: Sigmoid affinities
            expert_affinities = torch.sigmoid(router_logits)

            # Step 3: Add bias for selection only
            selection_scores = (
                expert_affinities
                + moe_module.e_score_correction_bias.to(expert_affinities.dtype)
            )

            # Step 4: Top-k selection on biased scores
            _, expert_index = torch.topk(selection_scores, router.top_k)

            # Step 5-7: Gather un-biased affinities, normalize, scale
            # expert_index: [batch*seq, top_k]
            selected_affinities = torch.gather(
                expert_affinities, dim=-1, index=expert_index
            )
            # Normalize selected weights
            weight_sum = selected_affinities.sum(dim=-1, keepdim=True) + 1e-20
            normalized_weights = selected_affinities / weight_sum
            # Apply routed_scaling_factor
            scaled_weights = normalized_weights * moe_module.routed_scaling_factor

            # Scatter back into full affinity tensor so NxDI's gather retrieves
            # the correct pre-computed weights
            expert_affinities = torch.zeros_like(expert_affinities)
            expert_affinities.scatter_(-1, expert_index, scaled_weights)

            # Cast to required dtype
            expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
            expert_index = expert_index.detach().to(dtype=torch.long)

            return router_logits, expert_affinities, expert_index

        router.forward = patched_router_forward

    def forward(self, hidden_states, is_speculative_decoding=False, residual=None):
        result = self.moe(
            hidden_states,
            is_speculative_decoding=is_speculative_decoding,
            residual=residual,
        )
        hidden_states = result[0]
        router_logits = result[1] if self.moe.return_router_logits else None
        expert_index = (
            result[-2]
            if (self.moe.return_expert_index and residual is not None)
            else (result[-1] if self.moe.return_expert_index else None)
        )
        residual_out = result[-1] if residual is not None else None

        return tuple(
            x
            for x in (hidden_states, router_logits, expert_index, residual_out)
            if x is not None
        )


# ---------------------------------------------------------------------------
# Decoder Layers
# ---------------------------------------------------------------------------


class GLM5DenseDecoderLayer(nn.Module):
    """
    Dense decoder layer for GLM-5 (layers 0, 1, 2).

    Standard pre-norm transformer block with MLA attention and SwiGLU MLP.
    No MoE routing.
    """

    def __init__(self, config: GLM5InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = GLM5Attention(
            config=config,
            layer_idx=layer_idx,
            tensor_model_parallel_group=(
                get_tp_group(config)
                if parallel_state.model_parallel_is_initialized()
                else None
            ),
        )

        # Norms
        if cpu_mode():
            self.input_layernorm = GLM5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = GLM5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )

        # Dense MLP
        self.mlp = GLM5DenseMLP(config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        cos_cache = kwargs.pop("cos_cache", None)
        sin_cache = kwargs.pop("sin_cache", None)

        residual = hidden_states.clone()

        # Pre-norm
        if not self.qkv_kernel_enabled or self.sequence_parallel_enabled:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )

        # Residual + attention output
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class GLM5MoEDecoderLayer(nn.Module):
    """
    MoE decoder layer for GLM-5 (layers 3-77).

    Pre-norm transformer block with MLA attention and MoE feed-forward.
    Includes shared expert added to routed output.
    """

    def __init__(self, config: GLM5InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.num_shared_experts = getattr(config, "num_shared_experts_actual", 1)

        # Attention
        self.self_attn = GLM5Attention(
            config=config,
            layer_idx=layer_idx,
            tensor_model_parallel_group=(
                get_tp_group(config)
                if parallel_state.model_parallel_is_initialized()
                else None
            ),
        )

        # Norms
        if cpu_mode():
            self.input_layernorm = GLM5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = GLM5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )

        # MoE feed-forward with post-attention layernorm fused
        self.feed_forward = GLM5MoE(config, rmsnorm=self.post_attention_layernorm)

        # Shared expert
        if self.num_shared_experts > 0:
            self.shared_expert = GLM5SharedExpert(config)
        else:
            self.shared_expert = None

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        cos_cache = kwargs.pop("cos_cache", None)
        sin_cache = kwargs.pop("sin_cache", None)

        residual = hidden_states.clone()

        # Pre-norm
        if not self.qkv_kernel_enabled or self.sequence_parallel_enabled:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )

        # MoE with fused residual
        is_speculative_decoding = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        moe_result = self.feed_forward(hidden_states, is_speculative_decoding, residual)
        moe_hidden_states = moe_result[0]
        # fused_residual = original_hidden_states + attn_output
        fused_residual = (
            moe_result[-1] if len(moe_result) > 1 else (residual + hidden_states)
        )

        # Shared expert: applied to post-norm of (residual + attn_output)
        if self.shared_expert is not None:
            shared_input = self.post_attention_layernorm(fused_residual)
            shared_output = self.shared_expert(shared_input)
            moe_hidden_states = moe_hidden_states + shared_output

        # Final: fused_residual + routed_output + shared_output
        hidden_states = fused_residual + moe_hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronGLM5Model(NeuronBaseModel):
    """
    GLM-5 model for NxDI inference.

    Dispatches between dense layers (0 to first_k_dense_replace-1) and
    MoE layers (first_k_dense_replace to num_hidden_layers-1).
    """

    def setup_attr_for_model(self, config: GLM5InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # MLA KV cache: single compressed KV "head" per layer.
        # head_dim = kv_lora_rank + qk_rope_head_dim = 576 (set in add_derived_config).
        # num_key_value_heads = 1 so the cache stores [B, 1, S, 576].
        # The compressed KV is NOT sharded across heads — it's a global representation.
        self.num_key_value_heads = 1
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: GLM5InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        first_k_dense = getattr(config, "first_k_dense_replace", 3)

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            should_pad_lm_head, lm_head_pad_alignment_size = get_lm_head_pad_config(
                vocab_size=config.vocab_size,
                tp_degree=config.neuron_config.tp_degree,
                lm_head_pad_alignment_size=(
                    config.neuron_config.lm_head_pad_alignment_size
                    * config.neuron_config.logical_nc_config
                ),
                skip_lm_head_pad=not config.neuron_config.lm_head_pad,
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=should_pad_lm_head,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size,
                keep_padded_output=should_pad_lm_head,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Build layers: dense for 0..first_k_dense-1, MoE for first_k_dense..num_hidden_layers-1
        # Only use num_hidden_layers=78 (skip MTP layer 78 which is layer index 78)
        layers = []
        for i in range(config.num_hidden_layers):
            if i < first_k_dense:
                layers.append(GLM5DenseDecoderLayer(config, layer_idx=i))
            else:
                layers.append(GLM5MoEDecoderLayer(config, layer_idx=i))
        self.layers = nn.ModuleList(layers)

        if cpu_mode():
            self.norm = GLM5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = CustomRMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )

        # Patch fused MoE TKG kernel for GLM-5 routing
        # The nkilib override mechanism (pip install -e nki-lib) ensures that
        # NxDI's MoEFusedTKG calls our modified nkilib kernel. We just need to
        # inject selection_bias and routed_scaling_factor into the kernel call.
        if getattr(config.neuron_config, "moe_fused_nki_kernel_enabled", False):
            moe_layers = []
            first_k = getattr(config, "first_k_dense_replace", 3)
            for layer_idx in range(first_k, config.num_hidden_layers):
                layer = self.layers[layer_idx]
                if hasattr(layer, "feed_forward"):
                    moe_layers.append((layer_idx, layer.feed_forward))
            _patch_fused_tkg_with_nkilib(moe_layers, config)

    def init_inference_optimization(self, config: GLM5InferenceConfig):
        if self.on_device_sampling:
            lm_head_tp_degree = None
            if hasattr(self, "lm_head") and hasattr(
                self.lm_head, "tensor_parallel_group"
            ):
                lm_head_tp_degree = self.lm_head.tensor_parallel_group.size()
            self.sampler = create_sampler(config.neuron_config, lm_head_tp_degree)

        # KV cache manager (MLA compressed cache)
        # For MLA, each token stores kv_lora_rank + qk_rope_head_dim = 576 values
        # The KV cache manager uses num_kv_heads to compute cache size.
        # With MLA, we set num_kv_heads = num_attention_heads since each head
        # operates on the shared compressed KV.
        self.kv_mgr = KVCacheManager(
            config, num_kv_head=self.num_key_value_heads, global_rank=self.rank_util
        )


# ---------------------------------------------------------------------------
# ForCausalLM (top-level entry point)
# ---------------------------------------------------------------------------


class NeuronGLM5ForCausalLM(NeuronBaseForCausalLM):
    """
    Top-level entry point for GLM-5 inference on Neuron.

    Usage:
        config = GLM5InferenceConfig.from_pretrained("zai-org/GLM-5-FP8", neuron_config=neuron_config)
        model = NeuronGLM5ForCausalLM(config)
        model.compile()
        model.generate(...)
    """

    _model_cls = NeuronGLM5Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: GLM5InferenceConfig
    ) -> dict:
        """
        Convert GLM-5 HuggingFace state dict to NxDI format.

        Handles:
        - FP8 dequantization with NaN clamping
        - MLA attention weights (pass through -- names match NxDI's DeepseekV3Attention)
        - Dense MLP weights for layers 0-2
        - MoE expert weights: per-expert -> fused stacked format
        - Router weights and e_score_correction_bias
        - Shared expert weights
        - DSA indexer weights (SKIPPED -- removed from state dict)
        - MTP layer 78 weights (SKIPPED -- removed from state dict)
        - Fused TKG NKI kernel weight duplication
        - LM head padding
        - Fused QKV (if enabled)
        - Rank utilities

        Note: 'model.' prefix is already stripped by NeuronBaseForCausalLM.get_state_dict().
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        first_k_dense = getattr(config, "first_k_dense_replace", 3)
        target_dtype = neuron_config.torch_dtype

        # --- FP8 dequantization ---
        maybe_dequantize_fp8_with_nan_clamp(state_dict, config)

        # --- Remove DSA indexer weights (if DSA disabled) or keep them ---
        dsa_enabled = getattr(config, "dsa_enabled", False)
        if not dsa_enabled:
            keys_to_remove = [
                k
                for k in list(state_dict.keys())
                if ".indexer." in k or ".indexers_proj." in k
            ]
            for k in keys_to_remove:
                del state_dict[k]
                logger.info("Removed DSA indexer weight (DSA disabled): %s", k)
        else:
            # DSA enabled: indexer weights stay in state_dict.
            # Weight names in checkpoint match our module structure exactly:
            #   layers.{i}.self_attn.indexer.wq_b.weight  [4096, 2048] FP8 -> dequanted to BF16
            #   layers.{i}.self_attn.indexer.wk.weight    [128, 6144]  FP8 -> dequanted to BF16
            #   layers.{i}.self_attn.indexer.weights_proj.weight [32, 6144] BF16 (not quantized)
            #   layers.{i}.self_attn.indexer.k_norm.weight [128] BF16
            #   layers.{i}.self_attn.indexer.k_norm.bias  [128] BF16
            #
            # FP8 dequantization already handled by maybe_dequantize_fp8_with_nan_clamp()
            # above (wq_b and wk have _scale_inv tensors, not in ".experts." path).
            # Just ensure all weights are cast to target dtype.
            for layer in range(num_layers):
                prefix = f"layers.{layer}"
                for sub_key in [
                    f"{prefix}.self_attn.indexer.wq_b.weight",
                    f"{prefix}.self_attn.indexer.wk.weight",
                    f"{prefix}.self_attn.indexer.weights_proj.weight",
                    f"{prefix}.self_attn.indexer.k_norm.weight",
                    f"{prefix}.self_attn.indexer.k_norm.bias",
                ]:
                    if sub_key in state_dict:
                        state_dict[sub_key] = state_dict[sub_key].to(target_dtype)

        # --- Remove MTP layer weights (layer 78+) ---
        keys_to_remove = [
            k
            for k in list(state_dict.keys())
            if any(f"layers.{i}." in k for i in range(num_layers, num_layers + 10))
        ]
        for k in keys_to_remove:
            del state_dict[k]
            logger.info("Removed MTP layer weight: %s", k)

        # --- Process each layer ---
        for layer in range(num_layers):
            prefix = f"layers.{layer}"

            if layer < first_k_dense:
                # Dense layer: rename mlp weights
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"{prefix}.mlp.{proj}.weight"
                    if key in state_dict:
                        state_dict[key] = state_dict[key].to(target_dtype)
            else:
                # MoE layer: convert expert weights to fused format

                # --- Router ---
                router_weight_key = f"{prefix}.mlp.gate.weight"
                if router_weight_key in state_dict:
                    state_dict[
                        f"{prefix}.feed_forward.moe.router.linear_router.weight"
                    ] = state_dict.pop(router_weight_key).to(target_dtype)

                router_bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
                if router_bias_key in state_dict:
                    state_dict[f"{prefix}.feed_forward.e_score_correction_bias"] = (
                        state_dict.pop(router_bias_key).to(torch.float32)
                    )

                # --- Expert weights: per-expert -> fused stacked ---
                # For FP8 quantization: keep experts as FP8 with per-expert scales
                # following the Llama 4 FP8 preprocessing pattern. This keeps
                # expert weights as 1-byte FP8 in the NEFF, halving their I/O from
                # 22.65 GB to 11.33 GB (total from 26.67 GB to ~15 GB).
                #
                # HF GLM-5-FP8 format: per-expert gate/up/down as float8_e4m3fn
                # with *_weight_scale_inv (per-block 128x128 scales).
                # NxDI quantized format: fused [E, H, 2I] gate_up + [E, I, H] down
                # as float8_e4m3fn with per-expert-channel scales.
                first_expert_key = f"{prefix}.mlp.experts.0.gate_proj.weight"
                first_expert_scale_key = (
                    f"{prefix}.mlp.experts.0.gate_proj.weight_scale_inv"
                )
                is_fp8_experts = first_expert_scale_key in state_dict

                if first_expert_key in state_dict:
                    num_experts = config.num_local_experts
                    gate_w = state_dict[first_expert_key]
                    intermediate_size, hidden_size = gate_w.shape  # [I, H]

                    quant_config = getattr(config, "quantization_config", None)
                    block_size = (
                        quant_config.get("weight_block_size", [128, 128])
                        if quant_config
                        else [128, 128]
                    )

                    if is_fp8_experts:
                        # FP8 path: dequant from block-wise FP8 to FP32, fuse gate+up,
                        # then re-quantize ALL experts with a SINGLE global scale.
                        #
                        # CRITICAL: per_tensor_symmetric means ONE scale for ALL experts.
                        # Each expert must be quantized with that same global scale.
                        # The global scale = max_abs_across_all_experts / 240.
                        # This ensures dequant (weight * scale) recovers correct values.
                        W_DTYPE = torch.float8_e4m3fn
                        S_DTYPE = torch.float32

                        # Pass 1: Dequant all experts to FP32 and fuse gate+up.
                        # Track global max abs for the unified scale.
                        all_gate_up_f32 = []  # [E] list of [H, 2I] FP32 tensors
                        all_down_f32 = []  # [E] list of [I, H] FP32 tensors
                        gate_up_global_max = torch.tensor(0.0)
                        down_global_max = torch.tensor(0.0)

                        for e in range(num_experts):
                            g_key = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                            u_key = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                            d_key = f"{prefix}.mlp.experts.{e}.down_proj.weight"
                            g_scale_key = (
                                f"{prefix}.mlp.experts.{e}.gate_proj.weight_scale_inv"
                            )
                            u_scale_key = (
                                f"{prefix}.mlp.experts.{e}.up_proj.weight_scale_inv"
                            )
                            d_scale_key = (
                                f"{prefix}.mlp.experts.{e}.down_proj.weight_scale_inv"
                            )

                            gate_dq = _dequantize_fp8_blockwise(
                                state_dict.pop(g_key),
                                state_dict.pop(g_scale_key),
                                block_size,
                                torch.float32,
                            )  # [I, H]
                            up_dq = _dequantize_fp8_blockwise(
                                state_dict.pop(u_key),
                                state_dict.pop(u_scale_key),
                                block_size,
                                torch.float32,
                            )  # [I, H]
                            down_dq = _dequantize_fp8_blockwise(
                                state_dict.pop(d_key),
                                state_dict.pop(d_scale_key),
                                block_size,
                                torch.float32,
                            )  # [H, I]

                            # Fuse gate+up: cat [I, H] + [I, H] -> [2I, H], T -> [H, 2I]
                            gate_up_fused = torch.cat(
                                [gate_dq, up_dq], dim=0
                            ).T  # [H, 2I]
                            down_fused = down_dq.T  # [I, H]

                            gate_up_global_max = torch.max(
                                gate_up_global_max, gate_up_fused.abs().max()
                            )
                            down_global_max = torch.max(
                                down_global_max, down_fused.abs().max()
                            )

                            all_gate_up_f32.append(gate_up_fused)
                            all_down_f32.append(down_fused)

                        # Compute the single global scale for all experts
                        gate_up_scale = (
                            gate_up_global_max / FP8_E4M3_NEURON_MAX
                        ).clamp(min=1e-12)
                        down_scale = (down_global_max / FP8_E4M3_NEURON_MAX).clamp(
                            min=1e-12
                        )

                        # Pass 2: Requantize all experts with the global scale
                        gate_up_weights = []
                        down_weights = []

                        for e in range(num_experts):
                            gate_up_fp8 = (
                                (all_gate_up_f32[e] / gate_up_scale)
                                .clamp(-FP8_E4M3_NEURON_MAX, FP8_E4M3_NEURON_MAX)
                                .to(W_DTYPE)
                            )
                            down_fp8 = (
                                (all_down_f32[e] / down_scale)
                                .clamp(-FP8_E4M3_NEURON_MAX, FP8_E4M3_NEURON_MAX)
                                .to(W_DTYPE)
                            )
                            gate_up_weights.append(gate_up_fp8)
                            down_weights.append(down_fp8)

                        # Free FP32 tensors
                        del all_gate_up_f32, all_down_f32

                        # Stack into [E, H, 2I] and [E, I, H]
                        gate_up_proj = torch.stack(gate_up_weights, dim=0)
                        down_proj = torch.stack(down_weights, dim=0)
                        del gate_up_weights, down_weights

                        # Scale: per_tensor_symmetric single scalar [1, 1, 1]
                        gate_up_proj_scale = gate_up_scale.view(1, 1, 1)
                        down_proj_scale = down_scale.view(1, 1, 1)

                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight"
                        ] = gate_up_proj
                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.scale"
                        ] = gate_up_proj_scale
                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight"
                        ] = down_proj
                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.down_proj.scale"
                        ] = down_proj_scale

                        logger.info(
                            f"Layer {layer}: Converted experts to FP8 "
                            f"gate_up={gate_up_proj.shape} down={down_proj.shape}"
                        )
                    else:
                        # BF16 path: standard fused expert weights
                        gate_up_proj = torch.empty(
                            num_experts,
                            hidden_size,
                            2 * intermediate_size,
                            dtype=target_dtype,
                            device="cpu",
                        )
                        down_proj = torch.empty(
                            num_experts,
                            intermediate_size,
                            hidden_size,
                            dtype=target_dtype,
                            device="cpu",
                        )

                        for e in range(num_experts):
                            g_key = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                            u_key = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                            d_key = f"{prefix}.mlp.experts.{e}.down_proj.weight"

                            gate_w = state_dict.pop(g_key).to(target_dtype)
                            up_w = state_dict.pop(u_key).to(target_dtype)
                            down_w = state_dict.pop(d_key).to(target_dtype)

                            gate_up_proj[e] = torch.cat([gate_w, up_w], dim=0).T
                            down_proj[e] = down_w.T

                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight"
                        ] = gate_up_proj
                        state_dict[
                            f"{prefix}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight"
                        ] = down_proj

                # --- Shared expert ---
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    shared_key = f"{prefix}.mlp.shared_experts.{proj_name}.weight"
                    if shared_key in state_dict:
                        state_dict[f"{prefix}.shared_expert.{proj_name}.weight"] = (
                            state_dict.pop(shared_key).to(target_dtype)
                        )

                # --- Fused MoE TKG: duplicate RMSNorm + transpose router weight ---
                if neuron_config.moe_fused_nki_kernel_enabled:
                    post_norm_key = f"{prefix}.post_attention_layernorm.weight"
                    if post_norm_key in state_dict:
                        state_dict[
                            f"{prefix}.feed_forward.moe.moe_fused_tkg.post_attention_layernorm.weight"
                        ] = state_dict[post_norm_key].clone()

                    router_w_key = (
                        f"{prefix}.feed_forward.moe.router.linear_router.weight"
                    )
                    if router_w_key in state_dict:
                        state_dict[f"{prefix}.feed_forward.moe.router.weight_T"] = (
                            state_dict[router_w_key].T.contiguous()
                        )

            gc.collect()

        # --- LM Head padding ---
        should_pad_lm_head, _ = get_lm_head_pad_config(
            vocab_size=config.vocab_size,
            tp_degree=neuron_config.tp_degree,
            lm_head_pad_alignment_size=(
                neuron_config.lm_head_pad_alignment_size
                * neuron_config.logical_nc_config
            ),
            skip_lm_head_pad=not neuron_config.lm_head_pad,
        )
        if should_pad_lm_head:
            state_dict["lm_head.bias"] = torch.zeros(
                state_dict["lm_head.weight"].shape[0], dtype=torch.float32
            )

        # --- Fused QKV ---
        # MLA doesn't use standard Q/K/V projections, so fused_qkv is NOT applicable.
        # The q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj are kept separate.
        # However, if fused_qkv is somehow enabled, we skip it for MLA layers.

        # --- Vocab parallel rank utility ---
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # --- Rank utilities ---
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        gc.collect()
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @classmethod
    def get_config_cls(cls):
        return GLM5InferenceConfig

    @staticmethod
    def get_compiler_args() -> str:
        """
        Compiler args for GLM-5.

        Returns None to use ModelWrapper's default compiler args, which:
        - Handles layer_boundary_markers (adds --recursive-layer-det=false)
        - Uses --auto-cast=none (appropriate for BF16 weights)
        - Uses -O2 for TKG, -O1 for CTE (standard NxDI defaults)
        - Adds cc-pipeline-tiling and vectorize-strided-dma

        GLM-5 requires layer_boundary_markers=True because the 78-layer
        model's weights (26.67 GB in BF16) exceed the 24 GB per-core
        HBM limit at LNC=2 for a single NEFF.

        NOTE: We return None so ModelWrapper handles marker flags and
        the --enable-verifier=false flag (needed to bypass NCC_EVRF009
        pre-flight I/O check — the check runs before marker-based splitting).
        """
        return None
