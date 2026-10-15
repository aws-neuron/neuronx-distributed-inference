# coding=utf-8
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
"""Sarvam-30B MoE model for NXD inference.

Architecture: sarvamai/sarvam-30b (SarvamMoEForCausalLM)
  - 19 layers, 32B total / 2.4B active
  - GQA: 64 Q heads, 4 KV heads, head_dim=64, QK norm (RMSNorm)
  - MoE: 128 routed experts + 1 shared, top-6 sigmoid routing
  - first_k_dense_replace=1: layer 0 is dense FFN (intermediate=8192)
  - layers 1-18: MoE (moe_intermediate_size=1024 per expert)
  - RoPE: theta=8M, no scaling, full head_dim
  - Routed scaling factor: 2.5, norm_topk_prob, expert bias
  - Custom architecture (trust_remote_code=True)

Based on solar_open contrib (same GQA + sigmoid MoE pattern).
Key differences from solar_open:
  - QK norm enabled (use_qk_norm=True)
  - Layer 0 is dense (first_k_dense_replace=1)
  - Fused QKV projection (query_key_value) instead of separate Q/K/V
  - head_dim=64 (smaller than typical 128)
  - No YaRN RoPE (plain RoPE with theta=8M)
"""

import gc
import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch_neuronx.xla_impl.ops import nki_jit

try:
    from transformers.generation import (
        GenerateDecoderOnlyOutput as SampleDecoderOnlyOutput,
        GenerateEncoderDecoderOutput as SampleEncoderDecoderOutput,
    )
except ImportError:
    from transformers.generation import (
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
    )

# MoE infrastructure
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from neuronx_distributed.modules.moe.moe_configs import (
    RoutedExpertsMLPOpsConfig,
    MoEFusedTKGConfig,
)
from neuronx_distributed.modules.moe.moe_process_group import (
    init_tensor_expert_parallel_moe_process_groups,
    get_moe_tp_ep_group,
    get_moe_ep_group,
)

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid routing patch for fused TKG kernel
# ---------------------------------------------------------------------------


def _patch_fused_tkg_for_sigmoid():
    """Patch MoEFusedTKG kernel to use ISA router fallback for sigmoid routing."""
    try:
        import neuronx_distributed.modules.moe.moe_fused_tkg as fused_tkg_mod

        class _PatchedKernelCall:
            def __init__(self, original):
                self._original = original

            def __getitem__(self, grid):
                original_grid_call = self._original[grid]

                def patched_call(*args, **kwargs):
                    kwargs["use_router_topk_nki_kernel"] = False
                    return original_grid_call(*args, **kwargs)

                return patched_call

        patched_any = False

        # Patch selective-load kernel
        if hasattr(fused_tkg_mod, "_moe_token_gen_selective_load_kernel_nki_call"):
            original_kernel = (
                fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call
            )
            if original_kernel is not None:
                fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call = (
                    _PatchedKernelCall(original_kernel)
                )
                patched_any = True

        # Patch all-experts kernel
        if hasattr(fused_tkg_mod, "_moe_tkg_forward_all_experts_nki_call"):
            original_all = fused_tkg_mod._moe_tkg_forward_all_experts_nki_call
            if original_all is not None:
                fused_tkg_mod._moe_tkg_forward_all_experts_nki_call = (
                    _PatchedKernelCall(original_all)
                )
                patched_any = True

        if patched_any:
            logger.warning("Patched MoEFusedTKG for sigmoid routing (ISA fallback)")
        else:
            logger.warning(
                "No fused TKG kernels found to patch (SDK 2.29 may use different API). "
                "Sigmoid routing will use ISA fallback via config."
            )
    except ImportError:
        logger.info("moe_fused_tkg module not available, skipping patch")
    except Exception as e:
        logger.warning("Failed to patch MoEFusedTKG for sigmoid: %s", e)


_patch_fused_tkg_for_sigmoid()


# ---------------------------------------------------------------------------
# RMSNorm helpers
# ---------------------------------------------------------------------------


def _rms_norm_cls():
    if cpu_mode():
        return _SimpleRMSNorm
    return CustomRMSNorm


class _SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Router: sigmoid routing with bias correction (same as solar_open/GLM-5)
# ---------------------------------------------------------------------------


class NeuronSarvamMoERouter(GroupLimitedRouter):
    """
    Sarvam-30B MoE router:
    - Sigmoid score function with norm_topk_prob and routed_scaling_factor
    - e_score_correction_bias (expert_bias in HF checkpoint)
    - n_group=1, topk_group=1 (no group routing constraint)
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group=None,
        jitter_eps: float = 0.0,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            n_group=n_group,
            topk_group=topk_group,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            dtype=dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
            jitter_eps=jitter_eps,
        )
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    def noaux_tc_top_k(self, scores):
        batch_size, num_experts = scores.shape

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        group_scores = self._calculate_group_scores(scores_for_choice, batch_size)
        group_idx = torch.topk(group_scores, k=self.topk_group)[1]
        group_mask = self._create_group_mask(group_scores, group_idx)
        score_mask = self._expand_group_mask(group_mask, batch_size)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        _, topk_idx = torch.topk(masked_scores, k=self.top_k)

        topk_weights = scores.gather(1, topk_idx)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        full_affinities = torch.zeros_like(scores)
        full_affinities.scatter_(1, topk_idx, topk_weights)

        return topk_idx, full_affinities

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        topk_idx, full_affinities = self.noaux_tc_top_k(expert_affinities)
        topk_idx = topk_idx.detach().to(dtype=torch.long)

        return router_logits, full_affinities, topk_idx


# ---------------------------------------------------------------------------
# MoE module initializer
# ---------------------------------------------------------------------------


def initialize_sarvam_moe_module(
    config: "SarvamMoEInferenceConfig",
    init_tkg_module: bool = False,
) -> MoE:
    """Initialize MoE module for Sarvam-30B (layers 1-18).

    Shared experts are NOT included in the MoE module — they are handled as
    a standalone NeuronSarvamSharedExpert on the decoder layer (Trinity pattern).
    Config sets num_shared_experts=0 to suppress NxDI's internal SharedExperts.

    Args:
        config: Model configuration.
        init_tkg_module: If True, enable fused MoE TKG NKI kernel path.
            Requires moe_intermediate_size / moe_tp_degree % 128 == 0.
    """
    if config.neuron_config.moe_ep_degree > 1:
        moe_ep_degree = config.neuron_config.moe_ep_degree
        moe_tp_degree = config.neuron_config.moe_tp_degree
        init_tensor_expert_parallel_moe_process_groups(
            moe_tp_degree, moe_ep_degree, moe_tp_degree, moe_ep_degree
        )
        moe_tkg_tp_group = get_moe_tp_ep_group(prefill=False)
        moe_tkg_ep_group = get_moe_ep_group(prefill=False)
        moe_cte_tp_group = get_moe_tp_ep_group(prefill=True)
        moe_cte_ep_group = get_moe_ep_group(prefill=True)
    else:
        moe_tkg_tp_group = parallel_state.get_tensor_model_parallel_group()
        moe_tkg_ep_group = parallel_state.get_expert_model_parallel_group()
        moe_cte_tp_group = parallel_state.get_tensor_model_parallel_group()
        moe_cte_ep_group = parallel_state.get_expert_model_parallel_group()

    router = NeuronSarvamMoERouter(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
        dtype=config.neuron_config.router_config.dtype,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
    )

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_size_actual=getattr(config, "original_hidden_size", None),
            intermediate_size_actual=getattr(
                config, "original_intermediate_size", None
            ),
            is_hidden_dim_shuffled=config.neuron_config.is_hidden_dim_shuffled,
            is_intermediate_dim_shuffled=config.neuron_config.is_intermediate_dim_shuffled,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            use_index_calc_kernel=config.neuron_config.use_index_calc_kernel,
            gate_clamp_upper_limit=config.neuron_config.gate_clamp_upper_limit,
            gate_clamp_lower_limit=config.neuron_config.gate_clamp_lower_limit,
            up_clamp_upper_limit=config.neuron_config.up_clamp_upper_limit,
            up_clamp_lower_limit=config.neuron_config.up_clamp_lower_limit,
            normalize_top_k_affinities=False,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping,
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tp_group,
        cte_expert_model_parallel_group=moe_cte_ep_group,
        tkg_tensor_model_parallel_group=moe_tkg_tp_group,
        tkg_expert_model_parallel_group=moe_tkg_ep_group,
    )

    # Shared experts are handled outside the MoE module as a standalone
    # NeuronSarvamSharedExpert on the decoder layer. num_shared_experts=0
    # on config ensures MoE won't create its own SharedExperts.

    # Build fused TKG config when requested
    if init_tkg_module:
        tkg_config = MoEFusedTKGConfig(
            quantized=config.neuron_config.quantized,
            moe_fused_kernel_enabled=config.neuron_config.moe_fused_nki_kernel_enabled,
            router_topk_kernel_enabled=config.neuron_config.router_topk_nki_kernel_enabled,
            expert_mlp_kernel_enabled=config.neuron_config.expert_mlp_nki_kernel_enabled,
            shared_mlp_kernel_enabled=config.neuron_config.shared_mlp_nki_kernel_enabled,
            norm_topk_prob=config.neuron_config.normalize_top_k_affinities,
            is_mxfp4_compute=getattr(config.neuron_config, "is_mxfp4_compute", None),
            router_mm_dtype=config.neuron_config.router_config.dtype,
        )
    else:
        tkg_config = None

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits,
        sequence_dimension=1,
        init_tkg_module=init_tkg_module,
        tkg_config=tkg_config,
    )

    # For fused TKG: register transposed router weights (needed by the kernel).
    # GroupLimitedRouter doesn't support store_transposed_weights, so we add
    # weight_T manually after construction.
    if init_tkg_module and hasattr(router, "linear_router"):
        router.store_transposed_weights = True
        router.weight_T = nn.Parameter(
            router.linear_router.weight.detach().T.clone(),
            requires_grad=False,
        )

    moe.eval()
    return moe


# ---------------------------------------------------------------------------
# Dense MLP for layer 0 (first_k_dense_replace=1)
# ---------------------------------------------------------------------------


class NeuronSarvamDenseMLP(nn.Module):
    """Dense MLP for layer 0 (gate_proj + up_proj -> SiLU -> down_proj).

    Uses _dense_intermediate_size (8192) instead of intermediate_size
    which is overridden to moe_intermediate_size (1024) for MoE layers.
    """

    def __init__(self, config: "SarvamMoEInferenceConfig"):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        # Use the original dense intermediate_size, not the MoE one
        dense_intermediate_size = getattr(
            config, "_dense_intermediate_size", config.intermediate_size
        )

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            dense_intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, x, padding_mask=None):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        # Return tuple to match MoE output format (hidden_states, ...)
        return (out,)


# ---------------------------------------------------------------------------
# Shared expert (standalone, outside NxDI MoE module)
# ---------------------------------------------------------------------------


class NeuronSarvamSharedExpert(nn.Module):
    """Standalone shared expert MLP for MoE layers.

    Sarvam-30B has num_shared_experts=1. Each MoE layer (1-18) has a shared
    expert whose output is added to the routed expert output for every token.
    Uses SiLU-gated MLP with moe_intermediate_size (same arch as routed experts).

    Implemented outside NxDI's MoE module (following Trinity's pattern) to:
    1. Avoid SDK 2.29 fused TKG kernel limitation (no shared expert support)
    2. Ensure reliable weight loading via standard parallel layers
    """

    def __init__(self, config: "SarvamMoEInferenceConfig"):
        super().__init__()
        from neuronx_distributed.parallel_layers.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

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

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


# ---------------------------------------------------------------------------
# Attention: GQA with QK norm, full RoPE, no bias
# ---------------------------------------------------------------------------


class NeuronSarvamMoEAttention(NeuronAttentionBase):
    """
    Sarvam-30B attention:
    - GQA: 64 Q heads, 4 KV heads, head_dim=64
    - QK norm: RMSNorm on Q and K before RoPE (separate q_norm and k_norm)
    - Full RoPE: rotary_dim = head_dim = 64, theta=8M
    - No QKV bias
    """

    def __init__(self, config: "SarvamMoEInferenceConfig"):
        rotary_dim = config.head_dim
        rotary_emb = RotaryEmbedding(
            rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Create separate q/k layernorms with proper hidden_size so weights
        # are initialized before trace (avoids "unexpected new parameter" error).
        # NeuronAttentionBase applies these per-head inside move_heads_front().
        q_layernorm = _rms_norm_cls()(config.head_dim, config.rms_norm_eps)
        k_layernorm = _rms_norm_cls()(config.head_dim, config.rms_norm_eps)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,  # Don't use shared qk_norm (creates weight=None)
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
            qkv_bias=False,
        )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronSarvamMoEAttention must be initialized in a distributed env."
            )


# ---------------------------------------------------------------------------
# Decoder layer: handles both dense (layer 0) and MoE (layers 1-18)
# ---------------------------------------------------------------------------


class NeuronSarvamMoEDecoderLayer(nn.Module):
    def __init__(self, config: "SarvamMoEInferenceConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = NeuronSarvamMoEAttention(config=config)

        self.input_layernorm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rms_norm_cls()(
            config.hidden_size, config.rms_norm_eps
        )

        # Fused TKG: enabled for MoE layers when user requests it
        self.moe_fused_tkg = getattr(config, "moe_fused_nki_kernel_enabled", False)

        # Layer 0 is dense, layers 1-18 are MoE
        if layer_idx < config.first_k_dense_replace:
            self.mlp = NeuronSarvamDenseMLP(config)
            self.moe_fused_tkg = False  # Dense layers never use fused TKG
            self.shared_expert = None
        else:
            self.mlp = initialize_sarvam_moe_module(
                config, init_tkg_module=self.moe_fused_tkg
            )
            # For fused TKG: attach a separate RMSNorm to the fused TKG module.
            # The fused kernel needs its own gamma/eps for the internal norm.
            # We pass rmsnorm=None to MoE (so CTE doesn't double-norm), then
            # attach a separate non-aliased instance for the TKG kernel.
            if self.moe_fused_tkg and hasattr(self.mlp, "moe_fused_tkg"):
                fused_tkg = self.mlp.moe_fused_tkg
                if fused_tkg is not None:
                    moe_rmsnorm = _rms_norm_cls()(
                        config.hidden_size, config.rms_norm_eps
                    )
                    fused_tkg.post_attention_layernorm = moe_rmsnorm

            # Shared expert: standalone module outside MoE (Trinity pattern).
            # num_shared_experts is zeroed in config to suppress NxDI's internal
            # SharedExperts; the actual count is in _num_shared_experts_actual.
            num_shared = getattr(config, "_num_shared_experts_actual", 0)
            if num_shared > 0:
                self.shared_expert = NeuronSarvamSharedExpert(config)
            else:
                self.shared_expert = None

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
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)
                qkv_fused_rmsnorm = None
        else:
            qkv_fused_rmsnorm = None

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        # Normalization strategy for fused MoE TKG:
        # - CTE (seq_len > 1): Decoder applies post_attention_layernorm as usual.
        # - TKG (seq_len == 1) with fused kernel: Decoder skips norm; the fused
        #   kernel applies it internally via moe_fused_tkg.post_attention_layernorm.
        is_tkg = self.moe_fused_tkg and hidden_states.shape[1] == 1
        if not is_tkg:
            hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states, padding_mask)[0]

        # Add shared expert output (applied to every token, outside MoE module).
        if self.shared_expert is not None:
            # In TKG mode, hidden_states is un-normed (fused kernel handles
            # norm internally for routed experts). Shared expert needs normed input.
            shared_input = (
                self.post_attention_layernorm(hidden_states)
                if is_tkg
                else hidden_states
            )
            shared_output = self.shared_expert(shared_input)
            mlp_output = mlp_output + shared_output

        hidden_states = residual + mlp_output

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronSarvamMoEModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: "SarvamMoEInferenceConfig"):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: "SarvamMoEInferenceConfig"):
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
                NeuronSarvamMoEDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class NeuronSarvamMoEForCausalLM(NeuronBaseForCausalLM):
    """Sarvam-30B MoE CausalLM for NXD inference."""

    _model_cls = NeuronSarvamMoEModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load Sarvam-30B using trust_remote_code (custom architecture)."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return SarvamMoEInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: "SarvamMoEInferenceConfig"
    ) -> dict:
        return convert_sarvam_moe_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def _construct_output(self, logits_or_next_tokens):
        if (
            isinstance(logits_or_next_tokens, (list, tuple))
            and len(logits_or_next_tokens) > 0
        ):
            logits_or_next_tokens = logits_or_next_tokens[0]
        return super()._construct_output(logits_or_next_tokens)

    def get_compiler_args(self):
        optimization_level = "-O1"
        compiler_args = (
            f"--enable-saturate-infinity --enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level}"
        )
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        if self.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "
        return compiler_args


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_sarvam_moe_config(model_path: str):
    """Return a load_config hook for SarvamMoEInferenceConfig."""
    from neuronx_distributed_inference.models.config import to_torch_dtype

    def load_config(self: "SarvamMoEInferenceConfig"):
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_dict = hf_config.to_dict()

        # Remove transformers-internal keys
        for key in (
            "model_type",
            "transformers_version",
            "architectures",
            "_attn_implementation",
            "id2label",
            "label2id",
            "problem_type",
            "return_dict",
            "auto_map",
        ):
            config_dict.pop(key, None)

        config_dict.setdefault("rope_scaling", None)

        # Handle dtype
        hf_dtype = config_dict.pop("torch_dtype", config_dict.pop("dtype", None))
        if hf_dtype is not None and self.neuron_config is not None:
            if not self.neuron_config.overrides_torch_dtype:
                self.neuron_config.torch_dtype = (
                    to_torch_dtype(hf_dtype) if isinstance(hf_dtype, str) else hf_dtype
                )

        self.__dict__.update(config_dict)
        self._name_or_path = model_path

    return load_config


class SarvamMoEInferenceConfig(InferenceConfig):
    """InferenceConfig for Sarvam-30B MoE model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "is_encoder_decoder"):
            self.is_encoder_decoder = False
        if not hasattr(self, "transformers_version"):
            self.transformers_version = "5.0.0"

        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        if not hasattr(self, "n_group"):
            self.n_group = getattr(self, "n_group", 1)
        if not hasattr(self, "topk_group"):
            self.topk_group = getattr(self, "topk_group", 1)
        if not hasattr(self, "first_k_dense_replace"):
            self.first_k_dense_replace = 1

        # Sarvam uses num_experts; NxDI expects num_local_experts
        self.num_local_experts = self.num_experts

        # For MoE layers, use moe_intermediate_size
        # Keep intermediate_size for dense layer 0
        # Override for ExpertMLPsV2
        self._dense_intermediate_size = self.intermediate_size
        self.intermediate_size = self.moe_intermediate_size

        # Router config
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.normalize_top_k_affinities = False
        self.neuron_config.disable_numeric_cc_token = True
        # Sigmoid routing: disable NKI router top-k kernel (only supports softmax)
        self.neuron_config.router_topk_nki_kernel_enabled = False
        # HF checkpoint has fused QKV (query_key_value); tell NxDI GQA to use Wqkv layout
        self.neuron_config.fused_qkv = True

        # Shared expert config — handled as a standalone module on the decoder
        # layer (NeuronSarvamSharedExpert), NOT inside NxDI's MoE module.
        # This follows Trinity's pattern and avoids the SDK 2.29 limitation
        # where the fused TKG kernel doesn't support shared experts.
        # Store real value for decoder layer to use, then zero out so
        # initialize_sarvam_moe_module won't create SharedExperts.
        self._num_shared_experts_actual = self.num_shared_experts
        self.num_shared_experts = 0
        self.neuron_config.fused_shared_experts = False
        self.neuron_config.transpose_shared_experts_weights = False
        self.neuron_config.shared_experts_sequence_parallel_enabled = False

        # SDK 2.29: shard_hidden kernel is missing from nkilib.
        # Force shard_on_intermediate path to avoid NotImplementedError.
        self.neuron_config.blockwise_matmul_config.use_shard_on_intermediate_dynamic_while = True

        self.maybe_pad_intermediate()
        self._enable_fused_moe_tkg()

    def maybe_pad_intermediate(self):
        from neuronx_distributed_inference.models.config import (
            SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
        )

        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(
                    padded - self.moe_intermediate_size, 0
                )
                self.moe_intermediate_size = padded

    def _enable_fused_moe_tkg(self):
        """Check and enable fused MoE TKG NKI kernel.

        The fused kernel combines RMSNorm + Router TopK + Expert MLP into a
        single NKI kernel launch, reducing HBM round-trips during token gen.

        Requires: moe_intermediate_size / moe_tp_degree % 128 == 0.
        """
        MOE_TKG_MK_INTERMEDIATE_PER_TP = 128

        if not hasattr(self, "neuron_config") or self.neuron_config is None:
            return

        fused_requested = getattr(
            self.neuron_config, "moe_fused_nki_kernel_enabled", None
        )
        if fused_requested is None:
            self.moe_fused_nki_kernel_enabled = False
            return

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
            "first_k_dense_replace",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_shared_experts",
            "norm_topk_prob",
            "num_attention_heads",
            "rms_norm_eps",
            "rope_theta",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ---------------------------------------------------------------------------
# State dict conversion: HF sarvam_moe -> NxDI
# ---------------------------------------------------------------------------


def convert_sarvam_moe_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: "SarvamMoEInferenceConfig",
) -> Dict[str, Any]:
    """
    Convert Sarvam-30B HF state dict to NxDI format.

    HF weight names (under 'model.' prefix, stripped by NxDI loader):
      - layers.{l}.attention.query_key_value.weight  (fused QKV)
      - layers.{l}.attention.dense.weight             (output proj)
      - layers.{l}.attention.query_layernorm.weight   (QK norm)
      - layers.{l}.attention.key_layernorm.weight     (QK norm)
      - layers.{l}.mlp.gate.weight                    (router, MoE layers)
      - layers.{l}.mlp.gate.expert_bias               (router bias, MoE layers)
      - layers.{l}.mlp.experts.{e}.gate_proj.weight   (MoE experts)
      - layers.{l}.mlp.experts.{e}.up_proj.weight
      - layers.{l}.mlp.experts.{e}.down_proj.weight
      - layers.{l}.mlp.shared_experts.gate_proj.weight (shared expert)
      - layers.{l}.mlp.shared_experts.up_proj.weight
      - layers.{l}.mlp.shared_experts.down_proj.weight
      - layers.{l}.mlp.gate_proj.weight               (dense MLP, layer 0)
      - layers.{l}.mlp.up_proj.weight
      - layers.{l}.mlp.down_proj.weight
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Rename word_embeddings -> embed_tokens (Sarvam uses word_embeddings, NxDI expects embed_tokens)
    if "word_embeddings.weight" in neuron_state_dict:
        neuron_state_dict["embed_tokens.weight"] = neuron_state_dict.pop(
            "word_embeddings.weight"
        )

    pad_size = getattr(config, "moe_intermediate_pad_size", 0)
    num_moe_experts = config.num_experts

    for l in range(config.num_hidden_layers):
        # Per-layer rank_util
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # ---- Rename attention keys ----
        # HF: layers.{l}.attention.query_key_value -> NxDI: layers.{l}.self_attn.Wqkv
        qkv_key = f"layers.{l}.attention.query_key_value.weight"
        if qkv_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = (
                neuron_state_dict.pop(qkv_key)
            )

        # HF: layers.{l}.attention.dense -> NxDI: layers.{l}.self_attn.o_proj
        dense_key = f"layers.{l}.attention.dense.weight"
        if dense_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"] = (
                neuron_state_dict.pop(dense_key)
            )

        # QK norm: query_layernorm -> q_layernorm, key_layernorm -> k_layernorm
        q_norm_key = f"layers.{l}.attention.query_layernorm.weight"
        if q_norm_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
                neuron_state_dict.pop(q_norm_key)
            )
        k_norm_key = f"layers.{l}.attention.key_layernorm.weight"
        if k_norm_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
                neuron_state_dict.pop(k_norm_key)
            )

        # ---- Dense layer 0 ----
        if l < config.first_k_dense_replace:
            # Dense MLP weights stay as-is (gate_proj, up_proj, down_proj)
            gc.collect()
            continue

        # ---- MoE layers (1-18) ----

        # Router: mlp.gate.weight -> mlp.router.linear_router.weight
        gate_weight_key = f"layers.{l}.mlp.gate.weight"
        if gate_weight_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict.pop(gate_weight_key)
            )

        # Router bias: mlp.gate.expert_bias -> mlp.router.e_score_correction_bias
        bias_key = f"layers.{l}.mlp.gate.expert_bias"
        if bias_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.e_score_correction_bias"] = (
                neuron_state_dict.pop(bias_key).to(torch.float32)
            )

        # Routed experts: fuse gate+up -> [E, H, 2I], transpose down -> [E, I, H]
        expert_0_key = f"layers.{l}.mlp.experts.0.gate_proj.weight"
        if expert_0_key in neuron_state_dict:
            gate_proj_0 = neuron_state_dict[expert_0_key]
            intermediate_size_e, hidden_size = gate_proj_0.shape
            device = gate_proj_0.device
            dtype = gate_proj_0.dtype

            gate_up_proj = torch.empty(
                num_moe_experts,
                hidden_size,
                2 * intermediate_size_e,
                dtype=dtype,
                device=device,
            )
            down_proj = torch.empty(
                num_moe_experts,
                intermediate_size_e,
                hidden_size,
                dtype=dtype,
                device=device,
            )

            for e in range(num_moe_experts):
                gate_w = (
                    neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                    .T.detach()
                    .clone()
                )
                up_w = (
                    neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                    .T.detach()
                    .clone()
                )
                down_w = (
                    neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
                    .T.detach()
                    .clone()
                )

                gate_up_slice = torch.narrow(gate_up_proj, 0, e, 1)
                torch.narrow(gate_up_slice, 2, 0, intermediate_size_e).copy_(gate_w)
                torch.narrow(
                    gate_up_slice, 2, intermediate_size_e, intermediate_size_e
                ).copy_(up_w)

                down_slice = torch.narrow(down_proj, 0, e, 1)
                down_slice.copy_(down_w)

                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]

            if pad_size > 0:
                gate_up_proj = gate_up_proj.reshape(num_moe_experts, hidden_size, 2, -1)
                gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(num_moe_experts, hidden_size, -1)
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))

            neuron_state_dict[
                f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
            ] = gate_up_proj
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
                down_proj
            )

        # Shared expert weights: remap from mlp.shared_experts.* to standalone
        # shared_expert.* on the decoder layer (sibling of mlp, not inside it).
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            hf_key = f"layers.{l}.mlp.shared_experts.{proj_name}.weight"
            if hf_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.shared_expert.{proj_name}.weight"] = (
                    neuron_state_dict.pop(hf_key)
                )

        # Fused MoE TKG aliased weights:
        # When moe_fused_nki_kernel_enabled, the fused TKG kernel needs:
        #   1. post_attention_layernorm.weight -> mlp.moe_fused_tkg.post_attention_layernorm.weight
        #   2. router.linear_router.weight -> router.weight_T (transposed)
        if getattr(config, "moe_fused_nki_kernel_enabled", False):
            post_attn_key = f"layers.{l}.post_attention_layernorm.weight"
            if post_attn_key in neuron_state_dict:
                neuron_state_dict[
                    f"layers.{l}.mlp.moe_fused_tkg.post_attention_layernorm.weight"
                ] = neuron_state_dict[post_attn_key].clone()

            router_key = f"layers.{l}.mlp.router.linear_router.weight"
            if router_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.mlp.router.weight_T"] = (
                    neuron_state_dict[router_key].detach().T.clone()
                )

        gc.collect()

    return neuron_state_dict
