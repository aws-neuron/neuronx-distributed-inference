# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# coding=utf-8
# Adapted from DeepSeek-V3 NxDI contrib for GLM-4.7-Flash (glm4_moe_lite).
# GLM-4.7-Flash uses the same MLA + MoE architecture as DeepSeek-V3 but at
# smaller scale (30B-A3B, 47 layers, 64 experts top-4, no YaRN).
# Supports FP8 E4M3 quantization of MoE expert weights (EXPERT_WISE_PER_CHANNEL_SYMMETRIC).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import gc
import logging
import os
from typing import List, Optional, Tuple, Type

import warnings
import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.utils import cpu_mode
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from src.rope_util import (
    Glm4MoeLiteRotaryEmbedding,
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter, RouterTopK
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# NKI MLA attention kernel disabled: research showed 1.84x slower than XLA
# baseline due to graph fusion barriers from custom op boundaries.
_nki_mla_attention = None
_USE_NKI_MLA = False

logger = logging.getLogger(__name__)


def convert_glm4_moe_lite_hf_to_neuron_state_dict(
    state_dict: dict, config: "Glm4MoeLiteInferenceConfig"
) -> dict:
    """
    Convert HuggingFace GLM-4.7-Flash (glm4_moe_lite) state dict to Neuron-compatible format.

    Transformations:
    1. Add rank utility tensors for TP sharding
    2. Rename router weights: gate.weight -> router.linear_router.weight
    3. Rename e_score_correction_bias -> router.e_score_correction_bias
    4. Fuse gate_proj + up_proj into gate_up_proj for each expert
    5. Stack down_proj weights across experts
    6. Skip dense layers (first_k_dense_replace layers, only layer 0 for GLM)
    7. Skip MTP layer weights (layer 47 embed_tokens) for initial bring-up

    When loading pre-quantized checkpoints (already in NxDI format with FP8 weights
    and scale tensors), the expert weight fusion steps (4-5) are skipped since the
    checkpoint already contains fused gate_up_proj and stacked down_proj in FP8.
    """
    num_hidden_layers = config.num_hidden_layers
    num_local_experts = config.num_local_experts
    tp_degree = getattr(config.neuron_config, "tp_degree", 1)
    first_k_dense = getattr(config, "first_k_dense_replace", 1)

    # Detect pre-quantized checkpoint: if the state dict already has fused expert
    # weight keys with scale tensors, skip the fusion step.
    _sample_fused_key = (
        f"layers.{first_k_dense}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
    )
    _sample_scale_key = (
        f"layers.{first_k_dense}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
    )
    is_prequantized = (
        _sample_fused_key in state_dict and _sample_scale_key in state_dict
    )

    if is_prequantized:
        logger.info(
            "Detected pre-quantized checkpoint (FP8 expert weights with scales). "
            "Skipping expert weight fusion."
        )
        # FP8 mode: shared_experts are moved from inside MoE module to decoder layer.
        # Rename: layers.X.mlp.shared_experts.* -> layers.X.shared_experts.*
        shared_expert_renames = {}
        for k in list(state_dict.keys()):
            if ".mlp.shared_experts." in k:
                new_key = k.replace(".mlp.shared_experts.", ".shared_experts.")
                shared_expert_renames[k] = new_key
        for old_key, new_key in shared_expert_renames.items():
            state_dict[new_key] = state_dict.pop(old_key)
        if shared_expert_renames:
            logger.info(
                f"Renamed {len(shared_expert_renames)} shared_expert keys "
                "(moved from MoE module to decoder layer for FP8 mode)"
            )

    # Add rank utilities for TP
    state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    # Remove MTP layer weights (layer 47) — not used in initial bring-up
    mtp_keys = [k for k in state_dict if k.startswith(f"layers.{num_hidden_layers}.")]
    for k in mtp_keys:
        del state_dict[k]

    for layer_idx in range(num_hidden_layers):
        # Add rank utility for attention
        state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        # Skip dense layers (no MoE conversion needed)
        if layer_idx < first_k_dense:
            continue

        # Rename router weights: gate.weight -> router.linear_router.weight
        router_key = f"layers.{layer_idx}.mlp.gate.weight"
        if router_key in state_dict:
            router_weight = state_dict[router_key].detach().clone()
            state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
                router_weight
            )
            del state_dict[router_key]

        # MoEFusedTKG requires transposed router weights (weight_T)
        # Generate it from linear_router.weight (works for both fresh and pre-quantized)
        router_linear_key = f"layers.{layer_idx}.mlp.router.linear_router.weight"
        if is_prequantized and router_linear_key in state_dict:
            state_dict[f"layers.{layer_idx}.mlp.moe_fused_tkg.router.weight_T"] = (
                state_dict[router_linear_key].detach().T.contiguous()
            )

        # Rename e_score_correction_bias for GroupLimitedRouter
        bias_key = f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        if bias_key in state_dict:
            bias_tensor = state_dict[bias_key].detach().clone()
            state_dict[f"layers.{layer_idx}.mlp.router.e_score_correction_bias"] = (
                bias_tensor
            )
            # Also provide at moe_fused_tkg.correction_bias path for XLA weight loading.
            # During SPMD tracing, the bias is accessed via self.correction_bias on
            # MoEFusedTKG, so the compiled model expects it at this path.
            state_dict[f"layers.{layer_idx}.mlp.moe_fused_tkg.correction_bias"] = (
                bias_tensor
            )
            del state_dict[bias_key]

        # For pre-quantized checkpoints, the bias is already at mlp.router path.
        # Add the moe_fused_tkg.correction_bias duplicate if not present.
        if is_prequantized:
            router_bias_key = f"layers.{layer_idx}.mlp.router.e_score_correction_bias"
            tkg_bias_key = f"layers.{layer_idx}.mlp.moe_fused_tkg.correction_bias"
            if router_bias_key in state_dict and tkg_bias_key not in state_dict:
                state_dict[tkg_bias_key] = state_dict[router_bias_key]

        # If pre-quantized checkpoint, expert weights are already fused — skip fusion
        if is_prequantized:
            continue

        # Check if expert weights exist for this layer
        expert_gate_key = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if expert_gate_key not in state_dict:
            continue

        intermediate_size, hidden_size = state_dict[expert_gate_key].shape
        device = state_dict[expert_gate_key].device
        dtype = state_dict[expert_gate_key].dtype

        # Fuse gate_proj + up_proj into gate_up_proj for all experts
        gate_up_proj = torch.empty(
            num_local_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_local_experts):
            gate_key = f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            up_key = f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"

            if gate_key in state_dict and up_key in state_dict:
                gate_proj_weights = state_dict[gate_key].T.detach().clone()
                up_proj_weights = state_dict[up_key].T.detach().clone()

                gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
                torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size).copy_(
                    gate_proj_weights
                )
                torch.narrow(
                    gate_up_proj_slice, 2, intermediate_size, intermediate_size
                ).copy_(up_proj_weights)

                del state_dict[gate_key]
                del state_dict[up_key]

        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = (
            gate_up_proj
        )

        # Stack down_proj weights across all experts
        down_proj = torch.empty(
            num_local_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_local_experts):
            down_key = f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            if down_key in state_dict:
                down_proj_weights = state_dict[down_key].T.detach().clone()
                torch.narrow(down_proj, 0, e, 1).copy_(down_proj_weights)
                del state_dict[down_key]

        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
            down_proj
        )

        gc.collect()

    return state_dict


class Glm4MoeLiteNeuronConfig(MoENeuronConfig):
    """Neuron hardware configuration for GLM-4.7-Flash MoE model."""

    pass


class Glm4MoeLiteRouter(RouterTopK):
    """
    Custom router for GLM-4.7-Flash using sigmoid activation + e_score_correction_bias.

    GLM-4.7-Flash uses n_group=1, topk_group=1 which makes GroupLimitedRouter's
    group selection a complete no-op. We use RouterTopK (simple torch.topk) instead,
    which produces a simpler computation graph that avoids the NCC_IBIR297 compiler
    bug in the tensorizer's ModuleForkPass at small TP degrees.

    After top-k selection, the selected affinities are L1-normalized and then
    scaled by routed_scaling_factor (1.8). This replaces the
    normalize_top_k_affinities step in ExpertMLPsV2, so the config must set
    normalize_top_k_affinities=False.
    """

    def __init__(
        self,
        routed_scaling_factor: float = 1.8,
        n_group: int = 1,
        topk_group: int = 1,
        **kwargs,
    ):
        # RouterTopK doesn't accept n_group/topk_group, so we pop them
        super().__init__(**kwargs)
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        # e_score_correction_bias is a trained parameter loaded from checkpoint.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        # Add correction bias for top-k selection (DS-V3 e_score_correction_bias)
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(
            0
        )

        # Simple top-k (no group logic since n_group=1, topk_group=1)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k)
        topk_idx = topk_idx.detach().to(dtype=torch.long)

        # Gather ORIGINAL affinities (without bias) for selected experts
        topk_weights = expert_affinities.gather(1, topk_idx)  # (T, top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Scatter back to dense (T, E) layout for ExpertMLPsV2
        expert_affinities_scaled = torch.zeros_like(expert_affinities)
        expert_affinities_scaled.scatter_(1, topk_idx, topk_weights)

        return router_logits, expert_affinities_scaled, topk_idx


class Glm4MoeLiteInferenceConfig(InferenceConfig):
    """
    Inference configuration for GLM-4.7-Flash (glm4_moe_lite).

    Handles MLA attention parameters, MoE routing config, dense/MoE layer
    distinction, and KV cache shape overrides for MLA's compressed cache format.

    Differences from DeepSeek-V3:
    - No YaRN RoPE (standard RoPE)
    - first_k_dense_replace = 1 (only layer 0 is dense)
    - routed_scaling_factor = 1.8 (vs 2.5)
    - n_group = 1, topk_group = 1 (no group selection)
    - 64 experts, top-4 (vs 256 experts, top-8)

    FP8 Quantization:
    - Supports EXPERT_WISE_PER_CHANNEL_SYMMETRIC quantization of MoE expert weights
    - Set neuron_config.quantized=True and provide quantized_checkpoints_path
    - Only expert gate_up_proj and down_proj are quantized; attention/dense layers stay BF16
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Standard HF config attributes expected by model_base.py
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "return_dict"):
            self.return_dict = True

        # GLM-4.7-Flash stores rope_theta inside rope_parameters dict
        if not hasattr(self, "rope_theta"):
            rope_params = getattr(self, "rope_parameters", None)
            if rope_params is not None:
                if isinstance(rope_params, dict):
                    self.rope_theta = rope_params.get("rope_theta", 1000000)
                else:
                    self.rope_theta = getattr(rope_params, "rope_theta", 1000000)
            else:
                self.rope_theta = 1000000

        # GLM-4.7-Flash uses attention_dropout=0.0
        if not hasattr(self, "attention_dropout"):
            self.attention_dropout = 0.0

        # Map HF config names to NXDI MoE names
        self.num_local_experts = getattr(
            self, "n_routed_experts", getattr(self, "num_experts", 0)
        )
        self.n_shared_experts = getattr(self, "n_shared_experts", 0)
        self.num_experts_per_tok = getattr(self, "num_experts_per_tok", 0)

        # Store dense layer intermediate size before overriding with MoE size.
        # HF config uses "intermediate_size" for the dense FFN (10240).
        if not hasattr(self, "dense_intermediate_size"):
            self.dense_intermediate_size = getattr(self, "intermediate_size", 0)

        # ExpertMLPsV2 reads config.intermediate_size for MoE expert size
        if getattr(self, "moe_intermediate_size", None) is not None:
            self.intermediate_size = self.moe_intermediate_size

        # Activation function
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"

        # Number of dense (non-MoE) layers at the start
        if not hasattr(self, "first_k_dense_replace"):
            self.first_k_dense_replace = 1

        # MoE routing config (only when MoENeuronConfig is used)
        if hasattr(self.neuron_config, "router_config"):
            self.neuron_config.router_config.dtype = torch.float32
            self.neuron_config.router_config.act_fn = "sigmoid"
            # Normalization + scaling is handled by Glm4MoeLiteRouter, not ExpertMLPsV2
            self.neuron_config.normalize_top_k_affinities = False

        # MoE kernel selection: use NKI shard-on-block kernel for CTE path.
        # This is the preferred kernel for GLM-4.7-Flash because:
        # - I_TP=384 satisfies shard_on_block constraint (I_TP % 16 == 0)
        # - shard_on_intermediate requires I_TP % 256 == 0 (would need padding to 512)
        # - shard_on_block has dynamic while loop for early exit on empty blocks
        # - Requires PING_PONG sharding strategy
        if hasattr(self.neuron_config, "blockwise_matmul_config"):
            from src.compat import _patched as _nki_kernel_available

            self.neuron_config.blockwise_matmul_config.use_torch_block_wise = False
            self.neuron_config.blockwise_matmul_config.use_shard_on_block_dynamic_while = True
            from neuronx_distributed.modules.moe.blockwise import BlockShardStrategy

            self.neuron_config.blockwise_matmul_config.block_sharding_strategy = (
                BlockShardStrategy.PING_PONG
            )
            logger.info(
                "NKI shard-on-block kernel enabled for MoE CTE blockwise matmul"
            )

            # Also keep shard_hidden patch as fallback (if shard_on_block fails)
            if not _nki_kernel_available:
                logger.warning(
                    "NKI compat patches not applied - shard_hidden fallback unavailable. "
                    "shard_on_block kernel should still work via nkilib."
                )

        # Disable numeric CC token (workaround for all-gather/reduce-scatter)
        self.neuron_config.disable_numeric_cc_token = True

        # FP8 quantization support for MoE expert weights
        if getattr(self.neuron_config, "quantized", False):
            # Set modules_to_not_convert: everything except MoE expert gate_up/down_proj.
            # CRITICAL: EXPERT_WISE_PER_CHANNEL_SYMMETRIC has per_channel_axis=None which
            # causes QuantizedColumnParallel to assert. We must exclude ALL non-expert-fused
            # linear layers from conversion.
            if not getattr(self.neuron_config, "modules_to_not_convert", None):
                self.neuron_config.modules_to_not_convert = [
                    "lm_head",
                    "embed_tokens",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                    "norm",
                    "layers.0.mlp",  # Dense MLP layer (not MoE)
                    "shared_experts",  # Shared expert MLP in MoE layers (not fused)
                    "router",
                    "rmsnorm",
                ]
            # Set the UNSAFE_FP8FNCAST env var required by the Neuron compiler
            os.environ["UNSAFE_FP8FNCAST"] = "1"
            # FP8 strategy: Use MoEFusedTKG for routed experts (it handles FP8 scales
            # natively), but the TKG kernel doesn't support shared_experts yet.
            # Solution: Set n_shared_experts=0 so initialize_moe_module doesn't include
            # shared experts in the MoE module. Instead, we handle shared experts as a
            # separate BF16 MLP in the decoder layer forward.
            if getattr(self, "n_shared_experts", 0) > 0:
                logger.info(
                    f"FP8 mode: Moving shared_experts (n={self.n_shared_experts}) out of MoE module "
                    "into separate BF16 MLP (MoEFusedTKG doesn't support shared_experts)."
                )
                # Store original value for decoder layer to create separate shared expert MLP
                self._fp8_shared_expert_intermediate_size = getattr(
                    self, "shared_expert_intermediate_size", None
                ) or (self.moe_intermediate_size * self.n_shared_experts)
                self.n_shared_experts = 0
            # Enable MoEFusedTKG for FP8 (now safe without shared_experts)
            if not self.neuron_config.moe_fused_nki_kernel_enabled:
                self.neuron_config.moe_fused_nki_kernel_enabled = True
                logger.info(
                    "Enabled moe_fused_nki_kernel for FP8 quantized path "
                    "(MoEFusedTKG handles scale tensor passing natively)"
                )
            logger.info(
                "FP8 quantization enabled for MoE experts. "
                f"modules_to_not_convert={self.neuron_config.modules_to_not_convert}"
            )

        # MLA KV cache: override head_dim and num_key_value_heads so the
        # KVCacheManager allocates (bsz, 1, max_len, rope_dim + kv_lora_rank)
        # instead of standard GQA layout.
        # For GLM-4.7-Flash: 64 (qk_rope_head_dim) + 512 (kv_lora_rank) = 576
        #
        # CRITICAL: The HF Glm4MoeLiteConfig has attribute_map={'head_dim': 'qk_rope_head_dim'}
        # which means setting self.head_dim would actually modify qk_rope_head_dim.
        # We must bypass this by writing directly to __dict__.
        self.__dict__["head_dim"] = self.qk_rope_head_dim + self.kv_lora_rank
        self.__dict__["num_key_value_heads"] = 1
        # Remove the head_dim alias from attribute_map to prevent KVCacheManager confusion
        if hasattr(self, "attribute_map") and isinstance(self.attribute_map, dict):
            self.attribute_map.pop("head_dim", None)

    def add_derived_config(self):
        self.num_cores_per_group = 1

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Glm4MoeLiteNeuronConfig

    def get_required_attributes(self) -> List[str]:
        return [
            # MLA (Multi-head Latent Attention) parameters
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            # MoE parameters
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
        ]


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def custom_compiler_args(quantized=False):
    """
    Compiler flags for GLM-4.7-Flash on Neuron.
    Same as DeepSeek-V3 except no --verify-hlo (debug only).
    When quantized=True, adds FP8 E4M3 cast flag.
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    # Removed: --enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 (causes NCC_IXCG967 at T=4096)
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    if quantized:
        # Enable unsafe FP8 E4M3 cast for Neuron hardware
        compiler_args += " --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3'"
    return compiler_args


class Glm4MoeLiteDenseMLP(nn.Module):
    """
    Dense MLP for GLM-4.7-Flash layer 0 (first_k_dense_replace=1).

    Uses SiLU-gated architecture: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses dense_intermediate_size (10240) instead of moe_intermediate_size (1536).
    """

    def __init__(self, config: Glm4MoeLiteInferenceConfig):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.dense_intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        output = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return (output,)


class Glm4MoeLiteSharedExpertMLP(nn.Module):
    """
    Separate shared expert MLP for FP8 mode.

    When FP8 quantization is enabled, the MoEFusedTKG kernel handles routed experts
    but doesn't support shared_experts. This module runs the shared expert computation
    separately in BF16, then its output is added to the routed expert output.

    Uses the same SiLU-gated architecture as the dense MLP:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    But with moe_intermediate_size (1536) instead of dense_intermediate_size (10240).
    """

    def __init__(self, config: "Glm4MoeLiteInferenceConfig"):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        # Use the stored shared expert intermediate size from config
        intermediate_size = getattr(
            config, "_fp8_shared_expert_intermediate_size", None
        )
        if intermediate_size is None:
            intermediate_size = config.moe_intermediate_size * getattr(
                config, "n_shared_experts", 1
            )

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Glm4MoeLiteAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) for GLM-4.7-Flash.

    Key differences from DeepSeek-V3:
    - qk_nope_head_dim=192 (vs 128), v_head_dim=256 (vs 128) -- requires
      corrected wkv_b split using qk_nope_head_dim instead of v_head_dim
    - Standard RoPE (no YaRN scaling)
    - q_lora_rank=768 (vs 1536)
    - 20 attention heads (vs 128)
    """

    def __init__(
        self,
        config: Glm4MoeLiteInferenceConfig,
        layer_idx: Optional[int] = None,
        tensor_model_parallel_group=None,
    ):
        super().__init__()

        # Config
        self.config = config
        self.neuron_config = config.neuron_config

        # Tensor parallelism
        self.tp_degree = config.neuron_config.tp_degree
        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        else:
            try:
                from neuronx_distributed.parallel_layers import parallel_state

                self.tensor_model_parallel_group = (
                    parallel_state.get_tensor_model_parallel_group()
                )
            except Exception:
                self.tensor_model_parallel_group = None
        self.rank_util = SPMDRank(world_size=self.tp_degree)

        # Data types
        self.torch_dtype = (
            getattr(config.neuron_config, "attention_dtype", None)
            or config.neuron_config.torch_dtype
        )
        self.rpl_reduce_dtype = getattr(config.neuron_config, "rpl_reduce_dtype", None)

        # Sequence parallelism
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Standard RoPE (no YaRN)
        self.rotary_emb = Glm4MoeLiteRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.bias = getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        assert layer_idx is not None, (
            "Please make sure to provide a `layer_idx` when creating this class."
        )

        self.attention_dropout = config.attention_dropout
        self.num_total_heads = config.num_attention_heads
        assert self.num_attention_heads % self.tp_degree == 0, (
            "Number of attention heads must be a multiple of tp degree."
        )
        if cpu_mode():
            self.num_heads = self.num_total_heads
        else:
            self.num_heads = self.num_total_heads // self.tp_degree

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.head_dim = self.v_head_dim

        self.is_causal = True
        self.init_mla_properties()

        # Standard softmax scale (no mscale adjustment for standard RoPE)
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def init_mla_properties(self):
        config = self.config
        dtype = self.torch_dtype
        if self.q_lora_rank is None:
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

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.kv_a_layernorm = get_rmsnorm_cls()(config.kv_lora_rank)
        if self.tensor_model_parallel_group is not None:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
            )
        else:
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )

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
        """Implements each layer's forward pass for the attention block."""
        # On decode, past_key_value comes from KVCacheManager as [k_cache, v_cache]
        # each shaped (bsz, 1, seq_len, qk_rope_head_dim + kv_lora_rank).
        # Convert to the single concatenated tensor that the decode path expects.
        if past_key_value is not None and isinstance(past_key_value, (list, tuple)):
            combined = past_key_value[0].squeeze(
                1
            )  # (bsz, seq_len, rope_dim + kv_lora_rank)
            past_key_value = combined

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

        # Weight matrix absorption
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
        # CRITICAL FIX: Split by qk_nope_head_dim, NOT v_head_dim.
        # Layout in kv_b_proj output: [K_nope (qk_nope_head_dim) | V (v_head_dim)] per head.
        # DS-V3 used v_head_dim which only worked because nope==v==128.
        # GLM-4.7-Flash: nope=192, v=256, so we must use qk_nope_head_dim.
        out_absorb = wkv_b[
            :, self.qk_nope_head_dim :, :
        ]  # V absorption: (num_heads, v_head_dim, kv_lora_rank)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        q_nope, q_pe = torch.tensor_split(q, (self.qk_nope_head_dim,), dim=-1)
        compressed_kv, k_pe = torch.tensor_split(
            compressed_kv, (self.kv_lora_rank,), dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Q_nope absorption: project Q_nope into compressed space using K_nope weights
        q_absorb = wkv_b[
            :, : self.qk_nope_head_dim
        ]  # K_nope absorption: (num_heads, qk_nope_head_dim, kv_lora_rank)
        q_nope = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

        seq_len = self.neuron_config.seq_len
        if sin_cache is None and cos_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos_cache, sin_cache, position_ids)

        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum(
            "bhqc,blc->bhql", q_nope, compressed_kv
        )
        active_scores *= self.softmax_scale

        if past_key_value is None:
            active_scores = torch.where(
                attention_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
            active_scores = nn.functional.softmax(
                active_scores, dim=-1, dtype=torch.float32
            ).to(k_pe.dtype)

            # Attention result with V absorb
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
        else:
            if _USE_NKI_MLA and _nki_mla_attention is not None:
                # === NKI FUSED MLA ATTENTION (TKG decode) ===
                # The kernel fuses: score computation + online softmax + V multiplication
                # for the prior KV cache AND combines with active token in one pass.
                #
                # Returns fully normalized output in compressed space [B, H, kv_lora_rank].
                seq_len_prior = past_key_value.shape[1]

                # q_nope: [B, H, 1, kv_lora_rank] -> [B, H, kv_lora_rank] (squeeze S_q=1)
                # q_pe: [B, H, 1, rope_dim] -> [B, H, rope_dim]
                q_nope_squeezed = q_nope.squeeze(2)  # [B, H, kv_lora_rank]
                q_pe_squeezed = q_pe.squeeze(2)  # [B, H, rope_dim]

                # Active token score and V for combining inside kernel
                # active_scores is [B, H, 1, 1] -- squeeze last dim to [B, H, 1] float32
                active_scores_for_kernel = active_scores.squeeze(
                    -1
                ).float()  # [B, H, 1]

                # Active V: compressed_kv is [B, 1, kv_lora_rank] -- shared across heads
                # Expand to [B, H, kv_lora_rank]
                active_v_for_kernel = (
                    compressed_kv.squeeze(1)
                    .unsqueeze(1)
                    .expand(bsz, self.num_heads, self.kv_lora_rank)
                    .contiguous()
                )

                # Construct additive attention mask for NKI kernel: [B, S, 1]
                # attention_mask is [B, 1, 1, S] bool (True=valid, False=invalid)
                # Convert to [B, S, 1] float32: 0.0 for valid, -9984.0 for invalid
                nki_mask = torch.where(
                    attention_mask.squeeze(1).squeeze(1).unsqueeze(-1),  # [B, S, 1]
                    torch.zeros(1, dtype=torch.float32, device=attention_mask.device),
                    torch.full(
                        (1,), -9984.0, dtype=torch.float32, device=attention_mask.device
                    ),
                )

                # KV cache is stored as [k_pe(64) | compressed_kv(512)] -- kernel accepts this order directly
                v_compressed = _nki_mla_attention[2](
                    q_nope_squeezed,
                    q_pe_squeezed,
                    past_key_value,
                    active_scores_for_kernel,
                    active_v_for_kernel,
                    nki_mask,
                    softmax_scale=self.softmax_scale,
                    batch_size=bsz,
                    num_heads=self.num_heads,
                    seq_len=seq_len_prior,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                )
                # v_compressed: [B, H, kv_lora_rank] BF16 (fully normalized)

                # Apply out_absorb: [B, H, kv_lora_rank] @ [H, v_head_dim, kv_lora_rank]^T -> [B, H, v_head_dim]
                attn_output = (
                    torch.einsum("bhc,hdc->bhd", v_compressed.float(), out_absorb)
                    .to(q_nope.dtype)
                    .unsqueeze(2)
                )  # [B, H, 1, v_head_dim] to match expected shape

            else:
                # === ORIGINAL PyTorch MLA ATTENTION (fallback) ===
                k_pe_prior, compressed_kv_prior = torch.tensor_split(
                    past_key_value,
                    [
                        self.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
                k_pe_prior = k_pe_prior.reshape(
                    bsz, 1, compressed_kv_prior.shape[1], self.qk_rope_head_dim
                )

                # I. Scores and softmax
                prior_scores = torch.matmul(
                    q_pe, k_pe_prior.transpose(2, 3)
                ) + torch.einsum("bhqc,blc->bhql", q_nope, compressed_kv_prior)
                prior_scores *= self.softmax_scale
                prior_scores = torch.where(
                    attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
                )
                prior_scores = prior_scores.to(torch.float32)

                softmax_prior, softmax_active = manual_softmax(
                    prior_scores, active_scores, is_speculation=False
                )
                softmax_prior, softmax_active = (
                    softmax_prior.to(k_pe.dtype),
                    softmax_active.to(k_pe.dtype),
                )

                # II. Attention result with V absorb
                x = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
                attn_active = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

                x = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
                attn_prior = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

                attn_output = attn_prior + attn_active

        # Transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)

        # Concatenate k_pe and compressed_kv into combined format for KVCacheManager.
        # KVCacheManager expects (key, value) tuple each shaped (bsz, 1, seq_len, head_dim).
        # For MLA, we store [k_pe | compressed_kv] in both slots (V is duplicate).
        combined = torch.cat([k_pe.squeeze(1), compressed_kv], dim=-1).unsqueeze(1)
        past_key_value = (combined, combined)

        return attn_output, past_key_value, cos_cache, sin_cache


class NeuronGlm4MoeLiteDecoderLayer(nn.Module):
    """
    GLM-4.7-Flash decoder layer with MLA attention and Dense MLP or MoE.

    Layer 0 uses a dense MLP; layers 1-46 use Mixture-of-Experts (MoE).
    """

    def __init__(self, config: Glm4MoeLiteInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx < getattr(config, "first_k_dense_replace", 1)

        self.self_attn = Glm4MoeLiteAttention(config=config, layer_idx=layer_idx)
        self.moe_fused_nki_kernel_enabled = getattr(
            config.neuron_config, "moe_fused_nki_kernel_enabled", False
        )

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if self.is_dense_layer:
            self.mlp = Glm4MoeLiteDenseMLP(config)
        elif self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config,
                rmsnorm=self.post_attention_layernorm,
                init_tkg_module=True,
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        # Swap in Glm4MoeLiteRouter (GroupLimitedRouter + routed_scaling_factor)
        if not self.is_dense_layer:
            self.mlp.router = Glm4MoeLiteRouter(
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.8),
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                n_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                dtype=config.neuron_config.router_config.dtype,
                act_fn=config.neuron_config.router_config.act_fn,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=1,
                # MoEFusedTKG requires transposed router weights
                store_transposed_weights=self.moe_fused_nki_kernel_enabled,
            )
            # Also update the router reference in MoEFusedTKG (if present)
            if (
                hasattr(self.mlp, "moe_fused_tkg")
                and self.mlp.moe_fused_tkg is not None
            ):
                self.mlp.moe_fused_tkg.router = self.mlp.router
                # Register correction bias directly on MoEFusedTKG as a parameter
                # so that XLA tracing captures it as a weight input (not a constant).
                # Accessing it via self.router.e_score_correction_bias inside the NKI
                # kernel call doesn't get captured by XLA's weight tracking.
                if hasattr(self.mlp.router, "e_score_correction_bias"):
                    self.mlp.moe_fused_tkg.correction_bias = (
                        self.mlp.router.e_score_correction_bias
                    )

        # FP8 mode: create separate shared expert MLP (BF16) since MoEFusedTKG
        # doesn't support shared_experts in the fused kernel.
        self.has_separate_shared_expert = not self.is_dense_layer and hasattr(
            config, "_fp8_shared_expert_intermediate_size"
        )
        if self.has_separate_shared_expert:
            self.shared_experts = Glm4MoeLiteSharedExpertMLP(config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states: input to the layer of shape (batch, seq_len, embed_dim)
            attention_mask: mask of size (batch_size, 1, query_seq_len, key_seq_len)
            position_ids: position ids of size (batch_size, sequence_length)
            past_key_value: cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Please use `attention_mask` instead."
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP (Dense for layer 0, MoE for rest)
        residual = hidden_states
        if self.is_dense_layer:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            if not self.moe_fused_nki_kernel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            # Save post-layernorm input for shared expert (FP8 mode)
            # In fused TKG mode, the rmsnorm is fused into the MoE module,
            # so we need the pre-norm input for the separate shared expert.
            if self.has_separate_shared_expert:
                if self.moe_fused_nki_kernel_enabled:
                    # In fused mode, post_attention_layernorm is passed to MoE as rmsnorm.
                    # The shared expert needs the normalized input.
                    shared_expert_input = self.post_attention_layernorm(hidden_states)
                else:
                    shared_expert_input = hidden_states
            is_speculative_decoding = (
                self.config.neuron_config.enable_fused_speculation
                and (not self.config.neuron_config.is_prefill_stage)
            )
            hidden_states = self.mlp(
                hidden_states,
                padding_mask,
                is_speculative_decoding=is_speculative_decoding,
            )[0]
            # Add shared expert output (FP8 mode: separate BF16 computation)
            if self.has_separate_shared_expert:
                hidden_states = hidden_states + self.shared_experts(shared_expert_input)
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronGlm4MoeLiteModel(NeuronBaseModel):
    """
    NeuronGlm4MoeLiteModel extends the GLM-4.7-Flash model to be traceable.
    The forward function of this class is traced by NxDI.
    """

    def setup_attr_for_model(self, config: Glm4MoeLiteInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Glm4MoeLiteInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
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
                NeuronGlm4MoeLiteDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronGlm4MoeLiteForCausalLM(NeuronBaseForCausalLM):
    """
    NxDI CausalLM wrapper for GLM-4.7-Flash (glm4_moe_lite).
    """

    _model_cls = NeuronGlm4MoeLiteModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return Glm4MoeLiteInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: Glm4MoeLiteInferenceConfig
    ) -> dict:
        return convert_glm4_moe_lite_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        """Return compiler args for GLM-4.7-Flash on Neuron."""
        quantized = getattr(self.config.neuron_config, "quantized", False)
        args = custom_compiler_args(quantized=quantized)
        args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return args


class Glm4MoeLiteGenerationAdapter(HuggingFaceGenerationAdapter):
    """Generation adapter with position_ids fix for transformers 5.x.

    In transformers >= 5.0, _update_model_kwargs_for_generation appends to
    position_ids and passes them back via kwargs on subsequent decode steps.
    However, NxDI's HuggingFaceGenerationAdapter.prepare_inputs_for_generation
    only recomputes position_ids when they are None in kwargs. When they are
    present (stale, growing), it passes them unchanged — leading to incorrect
    RoPE and KV cache positioning during autoregressive decode.

    Fix: Remove stale position_ids from kwargs so the base class recomputes
    them correctly from attention_mask each step.
    """

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        sampling_params=None,
        adapter_ids=None,
        divergence_idx=None,
        **kwargs,
    ):
        # Remove stale position_ids so base class recomputes from attention_mask
        kwargs.pop("position_ids", None)
        return super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            sampling_params=sampling_params,
            adapter_ids=adapter_ids,
            divergence_idx=divergence_idx,
            **kwargs,
        )
