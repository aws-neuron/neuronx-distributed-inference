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
"""GLM-4.5 MoE model for NXD inference.

Architecture differences from Qwen3MoE:
  - partial_rotary_factor=0.5: RoPE applied to only half of head_dim
  - attention_bias=True: QKV projections have bias
  - use_qk_norm (configurable): QK normalization
  - first_k_dense_replace: first N layers use dense MLP instead of MoE
  - n_shared_experts=1: shared expert alongside routed experts
  - Router: sigmoid + group selection + e_score_correction_bias + routed_scaling_factor
"""

import gc
import warnings
import math
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Glm4MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRMSNorm

# MoE infrastructure
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
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
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


# ---------------------------------------------------------------------------
# RMSNorm helpers
# ---------------------------------------------------------------------------


def get_rmsnorm_cls():
    """Return appropriate RMSNorm class for CPU vs Neuron execution."""
    return Glm4MoeRMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# Custom router: sigmoid + group routing + e_score_correction_bias + scaling
# ---------------------------------------------------------------------------


class NeuronGlm4MoeRouter(GroupLimitedRouter):
    """
    GLM-4.5 MoE router extending GroupLimitedRouter with:
    - e_score_correction_bias buffer (initialized to zeros, loaded from checkpoint)
    - norm_topk_prob: normalize top-k weights before applying scaling
    - routed_scaling_factor: scale final expert weights

    The forward returns (router_logits, expert_affinities_full, topk_idx) where
    expert_affinities_full has normalized+scaled weights for selected experts and
    zeros elsewhere, so ExpertMLPsV2 with normalize_top_k_affinities=False uses
    them directly.
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
        # Initialize e_score_correction_bias as FP32 buffer (loaded from checkpoint)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    def noaux_tc_top_k(self, scores):
        """
        Group-limited top-k selection with normalization and scaling.

        Args:
            scores: sigmoid-activated expert affinities [batch_size, num_experts]

        Returns:
            (topk_idx, full_affinities) where full_affinities has normalized+scaled
            weights at selected positions and zeros elsewhere.
        """
        batch_size, num_experts = scores.shape

        # Add correction bias for routing decision (not for final weights)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group-based selection
        group_scores = self._calculate_group_scores(scores_for_choice, batch_size)
        group_idx = torch.topk(group_scores, k=self.topk_group)[1]
        group_mask = self._create_group_mask(group_scores, group_idx)
        score_mask = self._expand_group_mask(group_mask, batch_size)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        # Select top-k experts
        _, topk_idx = torch.topk(masked_scores, k=self.top_k)

        # Get weights from ORIGINAL sigmoid scores (not bias-corrected)
        topk_weights = scores.gather(1, topk_idx)

        # Normalize
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # Apply routed scaling factor
        topk_weights = topk_weights * self.routed_scaling_factor

        # Scatter back into full-size tensor (zeros for non-selected)
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
# MoE module initializer for GLM-4.5
# ---------------------------------------------------------------------------


def initialize_glm4_moe_module(config: "Glm4MoeInferenceConfig") -> MoE:
    """
    Initialize the GLM-4.5 MoE module with GroupLimitedRouter + SharedExperts.
    """
    # Set up process groups
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

    # Router
    router = NeuronGlm4MoeRouter(
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

    # Expert MLPs
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
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
            normalize_top_k_affinities=False,  # router handles normalization+scaling
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

    # Shared experts (always on, parallel to routed experts)
    shared_experts = None
    if config.n_shared_experts:
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_shared_experts=config.n_shared_experts,
            hidden_act=config.hidden_act,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            fused_gate_up_projection=config.neuron_config.fused_shared_experts,
            sequence_parallel_enabled=config.neuron_config.shared_experts_sequence_parallel_enabled,
            transpose_weights=config.neuron_config.transpose_shared_experts_weights,
        )

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=shared_experts,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        return_router_logits=config.neuron_config.return_router_logits,
        sequence_dimension=1,
    )

    moe.eval()
    return moe


# ---------------------------------------------------------------------------
# Dense MLP for first_k_dense_replace layers
# ---------------------------------------------------------------------------


class NeuronGlm4MoeDenseMLP(nn.Module):
    """Standard GLU MLP (SiLU activation) used for dense layers (first_k_dense_replace)."""

    def __init__(self, config: "Glm4MoeInferenceConfig"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.dense_intermediate_size
        )  # full intermediate size

        # Gate and up projection (column parallel - split output)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
        )
        # Down projection (row parallel - reduce across TP)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention with partial RoPE and attention bias
# ---------------------------------------------------------------------------


class NeuronGlm4MoeAttention(NeuronAttentionBase):
    """
    GLM-4.5 MoE attention with:
    - partial_rotary_factor=0.5: RoPE applied to first half of head_dim only
    - attention_bias=True: bias in q/k/v projections
    - use_qk_norm: optional QK normalization
    """

    def __init__(self, config: "Glm4MoeInferenceConfig"):
        # Partial RoPE: use rotary_dim = int(head_dim * partial_rotary_factor)
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rotary_emb = RotaryEmbedding(
            rotary_dim,
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
            use_qk_norm=False,  # we handle qk_norm manually
            qkv_bias=config.attention_bias,
        )

        # Store rotary_dim for partial RoPE apply
        self.rotary_dim = rotary_dim

        # Optional QK norm (Glm4Moe applies RMSNorm per head after projection)
        if config.use_qk_norm:
            self.q_layernorm = get_rmsnorm_cls()(config.head_dim, config.rms_norm_eps)
            self.k_layernorm = get_rmsnorm_cls()(config.head_dim, config.rms_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronGlm4MoeAttention must be initialized in a distributed env. "
                "Please use neuronx_distributed module to initialize a distributed env."
            )

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Override to implement partial RoPE (apply to first rotary_dim dims only)."""
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            # Partial RoPE: split Q and K into rotary and pass-through portions
            # Q, K shape: [batch, heads, seq, head_dim]
            # cos_cache, sin_cache shape: [batch, seq, rotary_dim]
            rotary_dim = cos_cache.shape[-1]

            Q_rot = Q[..., :rotary_dim]
            Q_pass = Q[..., rotary_dim:]
            K_rot = K[..., :rotary_dim]
            K_pass = K[..., rotary_dim:]

            Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)

            Q = torch.cat([Q_rot, Q_pass], dim=-1)
            K = torch.cat([K_rot, K_pass], dim=-1)

        return Q, K, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class NeuronGlm4MoeDecoderLayer(nn.Module):
    """
    GLM-4.5 MoE decoder layer.

    - Layers 0..first_k_dense_replace-1: dense MLP
    - Layers first_k_dense_replace..num_hidden_layers-1: MoE block
    """

    def __init__(self, config: "Glm4MoeInferenceConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_layer = layer_idx >= config.first_k_dense_replace

        self.self_attn = NeuronGlm4MoeAttention(config=config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.is_moe_layer:
            self.mlp = initialize_glm4_moe_module(config)
        else:
            self.mlp = NeuronGlm4MoeDenseMLP(config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

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

        # MLP / MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronGlm4MoeModel(NeuronBaseModel):
    """NeuronGlm4MoeModel extends the GLM-4.5 MoE model to be traceable."""

    def setup_attr_for_model(self, config: "Glm4MoeInferenceConfig"):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: "Glm4MoeInferenceConfig"):
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
                NeuronGlm4MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class NeuronGlm4MoeForCausalLM(NeuronBaseForCausalLM):
    """
    GLM-4.5 MoE CausalLM for NXD inference.
    """

    _model_cls = NeuronGlm4MoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Glm4MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Glm4MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: "Glm4MoeInferenceConfig"
    ) -> dict:
        return convert_glm4_moe_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        else:
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
# InferenceConfig
# ---------------------------------------------------------------------------


class Glm4MoeInferenceConfig(InferenceConfig):
    """
    InferenceConfig for GLM-4.5 MoE model.

    Key adaptations from Qwen3MoeInferenceConfig:
    - Maps n_routed_experts -> num_local_experts
    - Sets n_shared_experts from HF config
    - Configures GLM-4.5-specific router (sigmoid, group routing, scaling factor)
    - Handles dense layers (first_k_dense_replace)
    - Handles partial RoPE (partial_rotary_factor)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GLM-4.5 uses n_routed_experts; neuronx expects num_local_experts
        self.num_local_experts = self.n_routed_experts

        # Store the dense-layer intermediate size (for NeuronGlm4MoeDenseMLP)
        # The HF config has both intermediate_size (dense) and moe_intermediate_size (MoE)
        self.dense_intermediate_size = self.intermediate_size

        # Set intermediate_size to moe_intermediate_size for MoE layers
        # (ExpertMLPsV2 and SharedExperts read config.intermediate_size)
        self.intermediate_size = self.moe_intermediate_size

        # Shared experts: n_shared_experts comes directly from HF config
        # (already set via load_pretrained_config)

        # Router configuration for GLM-4.5 MoE
        self.neuron_config.router_config.dtype = torch.float32  # router in FP32
        # act_fn is handled inside NeuronGlm4MoeRouter (always sigmoid)

        # Disable the standard normalize_top_k_affinities since our router handles it
        self.neuron_config.normalize_top_k_affinities = False

        # Set DISABLE_NUMERIC_CC_TOKEN for MoE
        self.neuron_config.disable_numeric_cc_token = True

        # Shared expert config
        self.neuron_config.fused_shared_experts = False
        self.neuron_config.transpose_shared_experts_weights = False
        self.neuron_config.shared_experts_sequence_parallel_enabled = False

        # Check if moe_intermediate_pad_size is needed
        self.maybe_pad_intermediate()

    def maybe_pad_intermediate(self):
        """Pad moe_intermediate_size if needed for blockwise matmul alignment."""
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

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "n_group",
            "n_routed_experts",
            "n_shared_experts",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "partial_rotary_factor",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "topk_group",
            "use_qk_norm",
            "vocab_size",
            "first_k_dense_replace",
            "attention_bias",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# ---------------------------------------------------------------------------
# State dict conversion: HF -> Neuronx
# ---------------------------------------------------------------------------


def _helper_concat_and_delete_qkv(
    state_dict: Dict[str, Any], layer_num: int, key_type: str
):
    """Concatenate Q/K/V weights (or biases) for fused QKV."""
    q_key = f"layers.{layer_num}.self_attn.q_proj.{key_type}"
    k_key = f"layers.{layer_num}.self_attn.k_proj.{key_type}"
    v_key = f"layers.{layer_num}.self_attn.v_proj.{key_type}"

    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{key_type}"] = torch.cat(
        [
            state_dict[q_key],
            state_dict[k_key],
            state_dict[v_key],
        ]
    )
    del state_dict[q_key]
    del state_dict[k_key]
    del state_dict[v_key]


def convert_glm4_moe_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: Glm4MoeInferenceConfig,
) -> Dict[str, Any]:
    """
    Convert HF GLM-4.5 MoE state dict to neuronx format.

    Transformations:
    1. Add rank_util tensors
    2. Rename q_norm/k_norm -> q_layernorm/k_layernorm
    3. Fuse QKV weights and biases
    4. For dense layers: no MoE weight transformation needed
    5. For MoE layers:
       - Rename router weight: mlp.gate.weight -> mlp.router.linear_router.weight
       - Copy correction bias: mlp.gate.e_score_correction_bias -> mlp.router.e_score_correction_bias
       - Fuse expert weights: per-expert gate_proj + up_proj -> [E, H, 2I] gate_up_proj
       - Copy shared expert weights (renamed to match SharedExperts structure)
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Add rank_util tensor for distributed inference
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    num_moe_experts = config.num_local_experts
    moe_intermediate_size = config.moe_intermediate_size
    pad_size = getattr(config, "moe_intermediate_pad_size", 0)

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Add per-layer rank_util
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Rename q_norm/k_norm -> q_layernorm/k_layernorm
        if f"layers.{l}.self_attn.q_norm.weight" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
                neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]
                .detach()
                .clone()
            )
            del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        if f"layers.{l}.self_attn.k_norm.weight" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
                neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]
                .detach()
                .clone()
            )
            del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]

        is_moe_layer = l >= config.first_k_dense_replace

        if is_moe_layer:
            # ---- Router ----
            # Rename: mlp.gate.weight -> mlp.router.linear_router.weight
            gate_weight_key = f"layers.{l}.mlp.gate.weight"
            if gate_weight_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                    neuron_state_dict[gate_weight_key].detach().clone()
                )
                del neuron_state_dict[gate_weight_key]

            # Copy e_score_correction_bias
            bias_key = f"layers.{l}.mlp.gate.e_score_correction_bias"
            if bias_key in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.mlp.router.e_score_correction_bias"] = (
                    neuron_state_dict[bias_key].detach().clone().to(torch.float32)
                )
                del neuron_state_dict[bias_key]

            # ---- Routed Expert weights ----
            # Get shape info from first expert
            gate_proj_0 = neuron_state_dict[
                f"layers.{l}.mlp.experts.0.gate_proj.weight"
            ]
            intermediate_size_e, hidden_size = gate_proj_0.shape
            device = gate_proj_0.device
            dtype = gate_proj_0.dtype

            # Fuse gate_proj + up_proj -> gate_up_proj: [E, H, 2I]
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

            # Pad intermediate size if needed
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

            # ---- Shared Expert weights ----
            # SharedExperts with fused_gate_up_projection=False expects separate gate_proj, up_proj, down_proj
            # Keys: mlp.shared_experts.gate_proj.weight -> mlp.shared_experts.gate_proj.weight (no rename needed
            #       IF SharedExperts uses the same naming)
            # However, we need to check SharedExperts's weight key names.
            # SharedExperts stores weights as gate_proj and up_proj (separate) or gate_up_proj (fused).
            # With fused_gate_up_projection=False and transpose_weights=False:
            # The keys should remain as-is: mlp.shared_experts.{gate/up/down}_proj.weight
            # No transformation needed - keys already match.

        gc.collect()

    # Fuse QKV weights (and biases if attention_bias=True)
    if config.neuron_config.fused_qkv:
        for l in range(config.num_hidden_layers):  # noqa: E741
            _helper_concat_and_delete_qkv(neuron_state_dict, l, "weight")
            if config.attention_bias:
                _helper_concat_and_delete_qkv(neuron_state_dict, l, "bias")

    return neuron_state_dict
