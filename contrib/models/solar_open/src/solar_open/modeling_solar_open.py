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
"""Solar Open MoE model for NXD inference.

Architecture notes vs GLM-4.5 MoE (which is the primary template):
  - partial_rotary_factor=1.0: full RoPE (no partial RoPE; no split/pass-through)
  - attention_bias=False: no bias in QKV projections
  - use_qk_norm=False: no QK normalization
  - first_k_dense_replace=0: ALL layers are MoE (no dense branch)
  - Expert weights in HF checkpoint (per-expert format, same as GLM-4.5):
      mlp.experts.{e}.gate_proj.weight  [I, H]
      mlp.experts.{e}.up_proj.weight    [I, H]
      mlp.experts.{e}.down_proj.weight  [H, I]
    Conversion: fuse gate+up → [E, H, 2I], transpose down → [E, I, H]
  - rope_scaling: None → plain RotaryEmbedding; {"type":"yarn"} → YaRN RoPE
  - Router: same sigmoid + group routing + e_score_correction_bias + routed_scaling_factor
    as GLM-4.5 (NeuronGlm4MoeRouter is reused directly)
  - solar_open is not in transformers stable releases (≤4.56.2) but has been merged
    into transformers main; load_hf_model loads safetensors directly for now
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

# transformers >= 5.0 renamed Sample*Output → Generate*Output; support both
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
)
from neuronx_distributed_inference.models.deepseek.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
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


def _rms_norm_cls():
    """Return appropriate RMSNorm class for CPU vs Neuron execution."""
    # Use a simple nn.Module RMSNorm when in CPU mode; CustomRMSNorm for Neuron.
    if cpu_mode():
        return _SimpleRMSNorm
    return CustomRMSNorm


class _SimpleRMSNorm(nn.Module):
    """Minimal RMSNorm for CPU reference / testing."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Router: reuse GLM-4.5 sigmoid router (identical logic)
# ---------------------------------------------------------------------------


class NeuronSolarOpenRouter(GroupLimitedRouter):
    """
    Solar Open MoE router extending GroupLimitedRouter with:
    - e_score_correction_bias buffer (initialized to zeros, loaded from checkpoint)
    - norm_topk_prob: normalize top-k weights before applying scaling
    - routed_scaling_factor: scale final expert weights

    Identical to NeuronGlm4MoeRouter — only the class name differs.
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

        # Bias-corrected scores for routing decision
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group-based selection
        group_scores = self._calculate_group_scores(scores_for_choice, batch_size)
        group_idx = torch.topk(group_scores, k=self.topk_group)[1]
        group_mask = self._create_group_mask(group_scores, group_idx)
        score_mask = self._expand_group_mask(group_mask, batch_size)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        _, topk_idx = torch.topk(masked_scores, k=self.top_k)

        # Weights from ORIGINAL sigmoid scores (not bias-corrected)
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
# MoE module initializer for Solar Open
# ---------------------------------------------------------------------------


def initialize_solar_open_moe_module(config: "SolarOpenInferenceConfig") -> MoE:
    """
    Initialize the Solar Open MoE module with GroupLimitedRouter + SharedExperts.
    All layers are MoE (first_k_dense_replace=0).
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

    router = NeuronSolarOpenRouter(
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
# YaRN RoPE wrapper (adapts DeepseekV3YarnRotaryEmbedding to position_ids interface)
# ---------------------------------------------------------------------------


class SolarOpenYarnRotaryEmbedding(nn.Module):
    """
    Wrapper that adapts DeepseekV3YarnRotaryEmbedding to the position_ids-based
    interface expected by NeuronAttentionBase.

    Standard RotaryEmbedding.forward(x, position_ids) returns (cos, sin) of shape
    [batch, seq, rotary_dim].

    DeepseekV3YarnRotaryEmbedding.forward(x, seq_len) returns (cos, sin) of shape
    [seq_len, rotary_dim] (not batched) — this wrapper indexes by position_ids.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
    ):
        super().__init__()
        self._yarn = DeepseekV3YarnRotaryEmbedding(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=32,
            beta_slow=1,
            mscale=1,
            mscale_all_dim=0,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]
        Returns:
            cos, sin: [batch, seq_len, dim]
        """
        seq_len = x.shape[2]
        max_pos = int(position_ids.max().item()) + 1
        needed_len = max(seq_len, max_pos)

        cos, sin = self._yarn(x, seq_len=needed_len)  # [needed_len, dim]

        # Index by position_ids to get [batch, seq_len, dim]
        cos = cos[position_ids]
        sin = sin[position_ids]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Attention: full RoPE, no bias, no QK norm
# ---------------------------------------------------------------------------


class NeuronSolarOpenAttention(NeuronAttentionBase):
    """
    Solar Open attention with:
    - Full RoPE (partial_rotary_factor=1.0): RotaryEmbedding with dim=head_dim
    - YaRN RoPE if rope_scaling.type == "yarn"
    - No attention bias (qkv_bias=False)
    - No QK normalization
    """

    def __init__(self, config: "SolarOpenInferenceConfig"):
        # Full RoPE: rotary_dim = head_dim (partial_rotary_factor=1.0)
        rotary_dim = config.head_dim
        rope_scaling = getattr(config, "rope_scaling", None)

        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            rotary_emb = SolarOpenYarnRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling[
                    "original_max_position_embeddings"
                ],
            )
        else:
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
            use_qk_norm=False,
            qkv_bias=False,
        )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronSolarOpenAttention must be initialized in a distributed env. "
                "Please use neuronx_distributed module to initialize a distributed env."
            )


# ---------------------------------------------------------------------------
# Decoder layer (always MoE — first_k_dense_replace=0)
# ---------------------------------------------------------------------------


class NeuronSolarOpenDecoderLayer(nn.Module):
    """
    Solar Open decoder layer. All layers are MoE (first_k_dense_replace=0).
    """

    def __init__(self, config: "SolarOpenInferenceConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = NeuronSolarOpenAttention(config=config)

        self.input_layernorm = _rms_norm_cls()(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rms_norm_cls()(
            config.hidden_size, config.rms_norm_eps
        )

        # All layers are MoE
        self.mlp = initialize_solar_open_moe_module(config)

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

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronSolarOpenModel(NeuronBaseModel):
    """NeuronSolarOpenModel extends Solar Open MoE model to be traceable."""

    def setup_attr_for_model(self, config: "SolarOpenInferenceConfig"):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: "SolarOpenInferenceConfig"):
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
                NeuronSolarOpenDecoderLayer(config, layer_idx)
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


class NeuronSolarOpenForCausalLM(NeuronBaseForCausalLM):
    """Solar Open MoE CausalLM for NXD inference."""

    _model_cls = NeuronSolarOpenModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load Solar Open using transformers SolarOpenForCausalLM (available since 5.0.0)."""
        from transformers import SolarOpenForCausalLM

        return SolarOpenForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return SolarOpenInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: "SolarOpenInferenceConfig"
    ) -> dict:
        return convert_solar_open_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def _construct_output(self, logits_or_next_tokens):
        """Override to ensure logits is always a tensor, not a list.

        NxDI's base _construct_output only unwraps list->tensor when async_mode=True.
        Solar Open uses sync mode, so logits can arrive as a list of per-bucket tensors
        from the Neuron runtime.  Unwrap here so that HuggingFaceGenerationAdapter can
        slice ``outputs.logits[:, -1, :]`` without a TypeError.
        """
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
# Config loader (transformers >= 5.0.0 includes solar_open)
# ---------------------------------------------------------------------------


def load_solar_open_config(model_path: str):
    """
    Return a load_config hook for SolarOpenInferenceConfig.

    Uses transformers.SolarOpenConfig.from_pretrained (available since transformers 5.0.0).
    Converts rope_parameters → rope_theta/rope_scaling for NxDI compatibility and
    sets fields that NxDI's InferenceConfig requires but SolarOpenConfig does not expose.
    """
    from neuronx_distributed_inference.models.config import to_torch_dtype

    def load_config(self: "SolarOpenInferenceConfig"):
        from transformers import SolarOpenConfig

        hf_config = SolarOpenConfig.from_pretrained(model_path)
        config_dict = hf_config.to_dict()

        # rope_parameters → rope_theta / rope_scaling (NxDI uses these fields)
        rope_params = config_dict.pop("rope_parameters", None)
        if isinstance(rope_params, dict):
            config_dict.setdefault(
                "rope_theta", rope_params.get("rope_theta", 1_000_000.0)
            )
            rope_type = rope_params.get("rope_type", "default")
            if rope_type != "default":
                config_dict["rope_scaling"] = {"type": rope_type}
            else:
                config_dict.setdefault("rope_scaling", None)
        else:
            config_dict.setdefault("rope_theta", 1_000_000.0)
            config_dict.setdefault("rope_scaling", None)

        # Remove transformers-internal keys that InferenceConfig doesn't need
        for key in (
            "model_type",
            "transformers_version",
            "architectures",
            "_attn_implementation",
            "id2label",
            "label2id",
            "problem_type",
            "return_dict",
        ):
            config_dict.pop(key, None)

        # Handle dtype
        hf_dtype = config_dict.pop("torch_dtype", config_dict.pop("dtype", None))
        if hf_dtype is not None and self.neuron_config is not None:
            if not self.neuron_config.overrides_torch_dtype:
                self.neuron_config.torch_dtype = (
                    to_torch_dtype(hf_dtype) if isinstance(hf_dtype, str) else hf_dtype
                )

        self.__dict__.update(config_dict)

        # Set _name_or_path so checkpoint_loader_fn can find the safetensors
        self._name_or_path = model_path

    return load_config


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------


class SolarOpenInferenceConfig(InferenceConfig):
    """
    InferenceConfig for Solar Open MoE model.

    Key differences from Glm4MoeInferenceConfig:
    - No first_k_dense_replace (always 0; all layers MoE)
    - No attention_bias (always False)
    - No use_qk_norm (always False)
    - No partial_rotary_factor (always 1.0 → full RoPE)
    - Expert weights are pre-fused in HF checkpoint (no per-expert separate modules)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set transformers PretrainedConfig defaults if not already present.
        # Solar Open has been merged into transformers main but is not yet available
        # in the current stable release, so AutoConfig does not set these fields.
        # Note: use_return_dict is a property on PretrainedConfig, skip it here
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "is_encoder_decoder"):
            self.is_encoder_decoder = False
        # HuggingFaceGenerationAdapter copies this into generation_config.transformers_version.
        # Without it, transformers' _prepare_generation_config raises TypeError on version.parse(None).
        if not hasattr(self, "transformers_version"):
            self.transformers_version = "5.0.0"

        # Fields that may be absent from upstage/Solar-Open-100B config.json → apply defaults
        # hidden_act: Solar Open uses SiLU gating (standard for SwiGLU-style MoE)
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        # n_group / topk_group: group-limited routing; default 1 = no group constraint
        if not hasattr(self, "n_group"):
            self.n_group = 1
        if not hasattr(self, "topk_group"):
            self.topk_group = 1

        # solar_open uses n_routed_experts; neuronx expects num_local_experts
        self.num_local_experts = self.n_routed_experts

        # intermediate_size in the HF config refers to a (unused) dense MLP size.
        # All layers use moe_intermediate_size for the MoE experts.
        # Override intermediate_size so ExpertMLPsV2 and SharedExperts use the right value.
        self.intermediate_size = self.moe_intermediate_size

        # Router configuration: sigmoid activation, FP32 router
        self.neuron_config.router_config.dtype = torch.float32

        # Disable standard normalize_top_k_affinities since our router handles it
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
            "n_routed_experts",
            "n_shared_experts",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
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
# State dict conversion: HF solar_open -> Neuronx
# ---------------------------------------------------------------------------


def _helper_concat_and_delete_qkv(
    state_dict: Dict[str, Any], layer_num: int, key_type: str
):
    """Concatenate Q/K/V weights for fused QKV."""
    q_key = f"layers.{layer_num}.self_attn.q_proj.{key_type}"
    k_key = f"layers.{layer_num}.self_attn.k_proj.{key_type}"
    v_key = f"layers.{layer_num}.self_attn.v_proj.{key_type}"

    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{key_type}"] = torch.cat(
        [state_dict[q_key], state_dict[k_key], state_dict[v_key]]
    )
    del state_dict[q_key]
    del state_dict[k_key]
    del state_dict[v_key]


def convert_solar_open_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: "SolarOpenInferenceConfig",
) -> Dict[str, Any]:
    """
    Convert Solar Open HF state dict to neuronx format.

    Supports two HF checkpoint formats:

    Format A — Per-expert (actual upstage/Solar-Open-* HF checkpoints, same as GLM-4.5):
      mlp.experts.{e}.gate_proj.weight [I, H]
      mlp.experts.{e}.up_proj.weight   [I, H]
      mlp.experts.{e}.down_proj.weight [H, I]
      → fuse gate+up: [E, H, 2I], transpose down: [E, I, H]

    Format B — Pre-fused 3D (legacy test models):
      mlp.experts.gate_up_proj [E, 2*I, H]   (no .weight suffix)
      mlp.experts.down_proj    [E, H, I]      (no .weight suffix)
      → permute(0,2,1): [E, H, 2I] and [E, I, H]

    The format is auto-detected from the state dict keys.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Auto-detect expert format from first available layer
    _per_expert_format = f"layers.0.mlp.experts.0.gate_proj.weight" in neuron_state_dict

    # Add rank_util tensor for distributed inference
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    pad_size = getattr(config, "moe_intermediate_pad_size", 0)
    num_moe_experts = config.n_routed_experts

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Add per-layer rank_util
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

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
        if _per_expert_format:
            # Format A: per-expert separate projections (actual HF model)
            gate_proj_0 = neuron_state_dict[
                f"layers.{l}.mlp.experts.0.gate_proj.weight"
            ]
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

        else:
            # Format B: pre-fused 3D tensors (legacy tiny_random models)
            # HF: gate_up_proj [E, 2*I, H] → Neuron: [E, H, 2*I]  (permute(0,2,1))
            gate_up_key = f"layers.{l}.mlp.experts.gate_up_proj"
            if gate_up_key in neuron_state_dict:
                gate_up = neuron_state_dict[gate_up_key]  # [E, 2*I, H]
                gate_up_neuron = (
                    gate_up.permute(0, 2, 1).detach().clone()
                )  # [E, H, 2*I]

                if pad_size > 0:
                    E, H, two_I = gate_up_neuron.shape
                    I = two_I // 2
                    gate_up_neuron = gate_up_neuron.reshape(E, H, 2, I)
                    gate_up_neuron = torch.nn.functional.pad(
                        gate_up_neuron, (0, pad_size)
                    )
                    gate_up_neuron = gate_up_neuron.reshape(E, H, -1)

                neuron_state_dict[
                    f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                ] = gate_up_neuron
                del neuron_state_dict[gate_up_key]

            # HF: down_proj [E, H, I] → Neuron: [E, I, H]  (permute(0,2,1))
            down_key = f"layers.{l}.mlp.experts.down_proj"
            if down_key in neuron_state_dict:
                down = neuron_state_dict[down_key]  # [E, H, I]
                down_neuron = down.permute(0, 2, 1).detach().clone()  # [E, I, H]

                if pad_size > 0:
                    down_neuron = torch.nn.functional.pad(
                        down_neuron, (0, 0, 0, pad_size)
                    )

                neuron_state_dict[
                    f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"
                ] = down_neuron
                del neuron_state_dict[down_key]

        # ---- Shared Expert weights ----
        # Keys: mlp.shared_experts.{gate/up/down}_proj.weight — no rename needed

        gc.collect()

    # Fuse QKV weights (solar_open has no attention bias, so only weights)
    if config.neuron_config.fused_qkv:
        for l in range(config.num_hidden_layers):  # noqa: E741
            _helper_concat_and_delete_qkv(neuron_state_dict, l, "weight")

    return neuron_state_dict
