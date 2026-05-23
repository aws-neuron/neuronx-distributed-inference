# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NKI kernel compatibility patch for GLM-4.7-Flash on SDK 2.29.

SDK 2.29 removed the `neuronxcc.nki._private.blockwise_mm` module, leaving
`_call_shard_hidden_kernel` in NxD as a stub. This patch restores it using
the nkilib `blockwise_mm_baseline_shard_hidden` kernel (BF16 path only).

Usage:
    import src.compat  # patches are applied on import

    # Then set use_torch_block_wise=False in your config to use the NKI kernel:
    # config.neuron_config.blockwise_matmul_config.use_torch_block_wise = False

This enables the NKI-optimized MoE CTE (context encoding) path, which should
provide significantly better performance than the torch fallback for blockwise
MoE computation.

Based on the patching pattern from MiniMax-M2 contrib (compat.py).
"""

import importlib
import logging

import torch

logger = logging.getLogger(__name__)


def _patch_blockwise_shard_hidden():
    """Patch NxD blockwise.py _call_shard_hidden_kernel from nkilib.

    GLM-4.7-Flash uses native BF16 weights — no FP8 dequant needed.
    We only restore the shard_hidden kernel call for the CTE path.
    """
    try:
        import neuronx_distributed.modules.moe.blockwise as bw
    except ImportError:
        logger.debug(
            "neuronx_distributed.modules.moe.blockwise not available, skipping patch"
        )
        return False

    # Check if the function is a stub (raises NotImplementedError)
    try:
        bw._call_shard_hidden_kernel(None)
    except NotImplementedError:
        pass  # Confirmed stub, proceed with patch
    except (TypeError, AttributeError):
        logger.debug("_call_shard_hidden_kernel appears functional, skipping patch")
        return False

    try:
        mod = importlib.import_module("nkilib.experimental.moe.forward.bwmm_shard_on_H")
        kernel_fn = getattr(mod, "blockwise_mm_baseline_shard_hidden")

        import nki

        wrapped_kernel = nki.jit(kernel_fn)
        bw._blockwise_mm_baseline_shard_hidden_nki_call = wrapped_kernel

        def _call_shard_hidden_kernel_patched(args):
            """Call the nkilib shard_hidden kernel for blockwise matmul (BF16)."""
            output = wrapped_kernel[2](
                hidden_states=args.hidden_states,
                expert_affinities_masked=args.expert_affinities_masked,
                gate_up_proj_weight=args.gate_up_proj_weight,
                down_proj_weight=args.down_proj_weight,
                block_size=args.block_size,
                token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
                block_to_expert=args.block_to_expert.to(dtype=torch.int32),
                gate_up_activations_T=args.gate_up_activations_T,
                down_activations=args.down_activations,
                skip_dma=args.skip_dma,
                is_tensor_update_accumulating=args.is_tensor_update_accumulating,
                expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
            )
            return output, args.gate_up_activations_T, args.down_activations

        bw._call_shard_hidden_kernel = _call_shard_hidden_kernel_patched
        logger.info(
            "Patched NxD blockwise._call_shard_hidden_kernel with nkilib kernel"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to patch blockwise._call_shard_hidden_kernel: {e}")
        return False


# Apply patch on import
_patched = _patch_blockwise_shard_hidden()
if _patched:
    logger.info("NKI blockwise MoE shard_hidden kernel enabled")
else:
    logger.warning(
        "NKI blockwise MoE shard_hidden kernel NOT enabled — "
        "falling back to torch blockwise (use_torch_block_wise=True)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: MoE TKG selective-expert NKI kernel
# ─────────────────────────────────────────────────────────────────────────────


def _patch_moe_tkg_selective_loading():
    """Patch ExpertMLPsV2.forward_selective_loading with nkilib moe_tkg kernel.

    The default forward_selective_loading loops over tokens in Python —
    this replaces it with a single NKI kernel call that processes all tokens
    in parallel using the selective-expert path (only top-K experts loaded).

    Weight layout follows MoEFusedTKG convention:
        gate_up_proj.weight.view(E_L, H, 2, -1)  -> [E_L, H, 2, I]
        down_proj.weight                          -> [E_L, I, H]

    GLM-4.7-Flash: T=4, H=2048, I=384 (TP=4), E_L=64, K=4, BF16.
    """
    try:
        from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
    except ImportError:
        logger.debug("ExpertMLPsV2 not available, skipping TKG patch")
        return False

    try:
        mod = importlib.import_module("nkilib.core.moe.moe_tkg.moe_tkg")
        moe_tkg_fn = getattr(mod, "moe_tkg")
        types_mod = importlib.import_module("nkilib.core.utils.common_types")
        ExpertAffinityScaleMode = getattr(types_mod, "ExpertAffinityScaleMode")
        ActFnType = getattr(types_mod, "ActFnType")

        import nki

        wrapped_moe_tkg = nki.jit(moe_tkg_fn)
    except Exception as e:
        logger.warning(f"Failed to import nkilib moe_tkg: {e}")
        return False

    # Save original for fallback
    _original_forward_selective_loading = ExpertMLPsV2.forward_selective_loading

    def _forward_selective_loading_nki(
        self, hidden_states, expert_affinities, expert_index
    ):
        """NKI moe_tkg selective-expert replacement for forward_selective_loading.

        Follows the exact weight access pattern from MoEFusedTKG:
            gate_up_proj.weight.view(num_local_experts, hidden_size, 2, -1)
            down_proj.weight  # already [E_L, I, H]

        Args:
            hidden_states: [T, H] (already 2D from MoE dispatcher)
            expert_affinities: [T, E] (dense, with scaled top-K values, zeros elsewhere)
            expert_index: [T, K] (top-k expert indices per token, int64)
        """
        H = hidden_states.shape[1]
        mlp_op = self.get_mlp_op()

        # Access weights exactly as MoEFusedTKG does (line 225-228 of moe_fused_tkg.py)
        # gate_up_proj.weight: flat tensor, reshape to [E_L, H, 2, I]
        E_L = mlp_op.gate_up_proj._n_local_experts
        gate_up_reshaped = mlp_op.gate_up_proj.weight.view(E_L, H, 2, -1)
        # down_proj.weight: already [E_L, I, H]
        down_w = mlp_op.down_proj.weight

        # Call NKI kernel: selective-expert mode (is_all_expert=False)
        # POST_SCALE: kernel extracts affinities at expert_index positions
        # and multiplies expert outputs by them (matching our router's pre-scaling)
        # NOTE: expert_affinities MUST be float32 — the kernel's tensor_scalar op
        # for affinity scaling requires float32 operand (MLIR verification fails on bf16)
        output = wrapped_moe_tkg[2](
            hidden_input=hidden_states,  # [T, H]
            expert_gate_up_weights=gate_up_reshaped,  # [E_L, H, 2, I]
            expert_down_weights=down_w,  # [E_L, I, H]
            expert_affinities=expert_affinities.to(torch.float32),  # [T, E] float32
            expert_index=expert_index.to(torch.int32),  # [T, K]
            is_all_expert=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )
        return output

    ExpertMLPsV2.forward_selective_loading = _forward_selective_loading_nki
    logger.info(
        "Patched ExpertMLPsV2.forward_selective_loading with nkilib moe_tkg kernel"
    )
    return True


_patched_moe_tkg = _patch_moe_tkg_selective_loading()
if _patched_moe_tkg:
    logger.info("NKI MoE TKG selective-expert kernel enabled")
else:
    logger.warning(
        "NKI MoE TKG selective-expert kernel NOT enabled — "
        "using default forward_selective_loading (torch loop)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: MoE TKG all-expert NKI kernel (for larger batch sizes)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_moe_tkg_all_experts():
    """Patch ExpertMLPsV2.forward_all_experts with nkilib moe_tkg kernel (all-expert mode).

    When batch_size is large enough that T*K/E >= 1.0, NxD switches to forward_all_experts
    which broadcasts all tokens through ALL experts. The NKI kernel with is_all_expert=True
    does this in a single fused kernel call.

    Requires rank_id for affinity slicing in all-expert mode.
    """
    try:
        from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
    except ImportError:
        return False

    try:
        mod = importlib.import_module("nkilib.core.moe.moe_tkg.moe_tkg")
        moe_tkg_fn = getattr(mod, "moe_tkg")
        types_mod = importlib.import_module("nkilib.core.utils.common_types")
        ExpertAffinityScaleMode = getattr(types_mod, "ExpertAffinityScaleMode")
        ActFnType = getattr(types_mod, "ActFnType")

        import nki

        wrapped_moe_tkg = nki.jit(moe_tkg_fn)
    except Exception as e:
        logger.warning(f"Failed to import nkilib moe_tkg for all-expert patch: {e}")
        return False

    _original_forward_all_experts = ExpertMLPsV2.forward_all_experts

    def _forward_all_experts_nki(
        self, hidden_states, expert_affinities, expert_index, chosen_expert_indices=None
    ):
        """NKI moe_tkg all-expert replacement for forward_all_experts.

        All tokens go through ALL local experts. The kernel handles masking/scaling internally.

        Args:
            hidden_states: [T, H]
            expert_affinities: [T, E] (dense, pre-scaled by router)
            expert_index: [T, K] (top-k indices)
            chosen_expert_indices: ignored (used by some EP paths)
        """
        H = hidden_states.shape[1]
        mlp_op = self.get_mlp_op()

        E_L = mlp_op.gate_up_proj._n_local_experts
        gate_up_reshaped = mlp_op.gate_up_proj.weight.view(E_L, H, 2, -1)
        down_w = mlp_op.down_proj.weight

        # All-expert mode requires rank_id for affinity slicing
        # With no EP (single rank), rank_id = 0
        # Must be on the same device as other tensors (XLA device during tracing)
        rank_id = torch.zeros(1, 1, dtype=torch.int32, device=hidden_states.device)

        output = wrapped_moe_tkg[2](
            hidden_input=hidden_states,  # [T, H]
            expert_gate_up_weights=gate_up_reshaped,  # [E_L, H, 2, I]
            expert_down_weights=down_w,  # [E_L, I, H]
            expert_affinities=expert_affinities.to(torch.float32),  # [T, E]
            expert_index=expert_index.to(torch.int32),  # [T, K]
            is_all_expert=True,
            rank_id=rank_id,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )
        return output

    ExpertMLPsV2.forward_all_experts = _forward_all_experts_nki
    logger.info(
        "Patched ExpertMLPsV2.forward_all_experts with nkilib moe_tkg kernel (all-expert mode)"
    )
    return True


# NOTE: All-expert NKI patch — DGE OOB was fixed in SDK 2.29.1 (neuronx-cc 2.24.8799).
# Re-enabling the NKI all-expert kernel for BS>=16 where T*K/E >= 1.0 triggers
# forward_all_experts mode. The kernel provides fused expert computation which may
# improve TPOT at high batch sizes.
_patched_moe_tkg_all = _patch_moe_tkg_all_experts()


# ─────────────────────────────────────────────────────────────────────────────
# Patch 4: Replace MoEFusedTKG kernel with nkilib moe_block_tkg for
#           e_score_correction_bias support
# ─────────────────────────────────────────────────────────────────────────────


def _patch_fused_tkg_for_correction_bias():
    """Replace MoEFusedTKG._moe_fused_tkg_kernel to use nkilib's moe_block_tkg.

    The pre-prod kernels (moe_token_gen_*) that NxDI calls internally do NOT
    support e_score_correction_bias. However, the open-source nkilib
    moe_block_tkg kernel does (we added router_correction_bias/scale params).

    This patch replaces the entire _moe_fused_tkg_kernel method to call
    nkilib's moe_block_tkg kernel directly, passing the correction bias
    and scaling factor from the GLM-4.7-Flash router.

    The nkilib kernel handles correction bias by:
    1. Adding bias to sigmoid affinities for top-K index selection
    2. Gathering ORIGINAL (unbiased) affinities at selected indices
    3. L1-normalizing and scaling by router_correction_scale

    Must be called before model.compile().
    """
    try:
        import neuronx_distributed.modules.moe.moe_fused_tkg as fused_tkg_mod
        from neuronx_distributed.modules.moe.model_utils import (
            ACTFunc,
            DEFAULT_SELECTIVE_LOADING_THRESHOLD,
            get_kernel_activation_func_id,
        )

        # Import nkilib moe_block_tkg kernel
        try:
            from nkilib_src.nkilib.core.moe_block.moe_block_tkg import moe_block_tkg
            from nkilib_src.nkilib.core.utils.common_types import (
                ActFnType as NkilibActFnType,
                RouterActFnType as NkilibRouterActFnType,
                ExpertAffinityScaleMode as NkilibExpertAffinityScaleMode,
            )
        except ImportError:
            from nkilib.core.moe_block.moe_block_tkg import moe_block_tkg
            from nkilib.core.utils.common_types import (
                ActFnType as NkilibActFnType,
                RouterActFnType as NkilibRouterActFnType,
                ExpertAffinityScaleMode as NkilibExpertAffinityScaleMode,
            )

        # Map NxDI's act_fn string names to nkilib enum values
        NKILIB_ROUTER_ACT_FN_MAP = {
            "sigmoid": NkilibRouterActFnType.SIGMOID,
            "softmax": NkilibRouterActFnType.SOFTMAX,
        }
        NKILIB_ACT_FN_MAP = {
            "silu": NkilibActFnType.SiLU,
            "gelu": NkilibActFnType.GELU,
        }

        def _replacement_moe_fused_tkg_kernel(self, hidden_states):
            """Replacement _moe_fused_tkg_kernel using nkilib moe_block_tkg.

            Calls the nkilib kernel which supports router_correction_bias,
            instead of the pre-prod moe_token_gen_* kernels which don't.
            """
            hidden_states_shape = hidden_states.shape

            # Determine expert affinity scaling mode
            if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
                scaling_mode = NkilibExpertAffinityScaleMode.PRE_SCALE
            else:
                scaling_mode = NkilibExpertAffinityScaleMode.POST_SCALE

            # Determine if we should use all-expert mode
            total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
            perc_experts_loaded = (
                total_tokens * self.num_experts_per_tok / self.num_local_experts
            )
            use_all_expert = perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD

            # LNC config for nkilib kernel (integer: 1 or 2)
            lnc = self.logical_nc_config

            # Get shared expert weights (will be None in FP8 mode)
            (
                shared_expert_gate_w,
                shared_expert_up_w,
                shared_expert_down_w,
            ) = self._slice_shared_experts_weights()

            # Get activation function
            routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
            kernel_activation_func_id = get_kernel_activation_func_id(
                ACTFunc.validate(routed_experts_mlp_config.hidden_act),
                routed_experts_mlp_config.glu_type,
            )

            # Build kernel kwargs
            kernel_kwargs = dict(
                inp=hidden_states,  # [B, S, H]
                gamma=self.post_attention_layernorm.weight.unsqueeze(0),  # [1, H]
                router_weights=self.router.weight_T,  # [H, E]
                expert_gate_up_weights=self.expert_mlps.mlp_op.gate_up_proj.weight.view(
                    self.num_local_experts, self.hidden_size, 2, -1
                ),  # [E, H, 2, I]
                expert_down_weights=self.expert_mlps.mlp_op.down_proj.weight,  # [E, I, H]
                shared_expert_gate_w=shared_expert_gate_w,
                shared_expert_up_w=shared_expert_up_w,
                shared_expert_down_w=shared_expert_down_w,
                expert_gate_up_weights_scale=(
                    self.expert_mlps.mlp_op.gate_up_proj.scale.view(
                        self.num_local_experts, 2, -1
                    )
                    if self.config.quantized
                    else None
                ),
                expert_down_weights_scale=(
                    self.expert_mlps.mlp_op.down_proj.scale.view(
                        self.num_local_experts, -1
                    )
                    if self.config.quantized
                    else None
                ),
                router_bias=(
                    self.router.linear_router.bias if self.router.bias else None
                ),
                expert_gate_up_bias=(
                    self.expert_mlps.mlp_op.gate_up_proj.bias.view(
                        self.num_local_experts, 2, -1
                    )
                    if routed_experts_mlp_config.bias
                    else None
                ),
                expert_down_bias=(
                    self.expert_mlps.mlp_op.down_proj.bias
                    if routed_experts_mlp_config.bias
                    else None
                ),
                eps=self.post_attention_layernorm.variance_epsilon,
                top_k=self.num_experts_per_tok,
                router_act_fn=NKILIB_ROUTER_ACT_FN_MAP[self.router.act_fn],
                router_pre_norm=not self.router.apply_act_fn_over_topk,
                norm_topk_prob=self.config.norm_topk_prob,
                expert_affinities_scaling_mode=scaling_mode,
                hidden_act_fn=NkilibActFnType(kernel_activation_func_id),
                is_all_expert=use_all_expert,
            )

            # Optional: hidden_actual for padded dimensions
            if routed_experts_mlp_config.hidden_size_actual is not None:
                kernel_kwargs["hidden_actual"] = (
                    routed_experts_mlp_config.hidden_size_actual
                )

            # Optional: clamping limits
            if routed_experts_mlp_config.gate_clamp_upper_limit is not None:
                kernel_kwargs["gate_clamp_upper_limit"] = (
                    routed_experts_mlp_config.gate_clamp_upper_limit
                )
            if routed_experts_mlp_config.gate_clamp_lower_limit is not None:
                kernel_kwargs["gate_clamp_lower_limit"] = (
                    routed_experts_mlp_config.gate_clamp_lower_limit
                )
            if routed_experts_mlp_config.up_clamp_upper_limit is not None:
                kernel_kwargs["up_clamp_upper_limit"] = (
                    routed_experts_mlp_config.up_clamp_upper_limit
                )
            if routed_experts_mlp_config.up_clamp_lower_limit is not None:
                kernel_kwargs["up_clamp_lower_limit"] = (
                    routed_experts_mlp_config.up_clamp_lower_limit
                )

            # For all-expert mode, provide rank_id
            if use_all_expert:
                local_rank = self.expert_mlps.spmd_rank.get_rank()
                local_ep_rank = (
                    local_rank
                    // self.expert_mlps.moe_tensor_model_parallel_group.size()
                )
                kernel_kwargs["rank_id"] = local_ep_rank.reshape(1, 1)

            # Inject router_correction_bias and router_correction_scale
            # NOTE: Access correction_bias directly from self (MoEFusedTKG) where it's
            # registered as a parameter. This ensures XLA tracing captures it as a
            # weight input to the NEFF (not inlined as a compile-time constant).
            # Accessing via self.router.e_score_correction_bias doesn't work because
            # XLA's parameter tracking doesn't follow nested module access inside NKI calls.
            if hasattr(self, "correction_bias"):
                bias = self.correction_bias
                bias = bias.to(torch.float32)
                if bias.dim() == 1:
                    bias = bias.unsqueeze(0)  # [1, E]
                kernel_kwargs["router_correction_bias"] = bias
            if hasattr(self, "router") and hasattr(
                self.router, "routed_scaling_factor"
            ):
                kernel_kwargs["router_correction_scale"] = (
                    self.router.routed_scaling_factor
                )

            # Call nkilib moe_block_tkg kernel
            out, router_logits = moe_block_tkg[lnc](**kernel_kwargs)

            return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

        # Replace the method
        fused_tkg_mod.MoEFusedTKG._moe_fused_tkg_kernel = (
            _replacement_moe_fused_tkg_kernel
        )
        logger.info(
            "Patched MoEFusedTKG._moe_fused_tkg_kernel to use nkilib "
            "moe_block_tkg kernel with router_correction_bias support"
        )
        return True

    except ImportError as e:
        logger.warning(
            "Failed to import nkilib moe_block_tkg kernel: %s. "
            "Correction bias will NOT be applied in fused TKG path.",
            e,
        )
        return False
    except Exception as e:
        logger.warning("Failed to patch MoEFusedTKG for correction bias: %s", e)
        return False


_patched_correction_bias = _patch_fused_tkg_for_correction_bias()
