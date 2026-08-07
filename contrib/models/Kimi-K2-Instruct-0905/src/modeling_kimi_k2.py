# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi-K2 (moonshotai/Kimi-K2-Instruct-0905) on Neuron via NxDI.
#
# Architecture: DeepseekV3ForCausalLM variant with MLA attention + MoE
#   - 1T total parameters, 32B active per token
#   - 384 routed experts (8 active per token) + 1 shared expert
#   - Multi-Latent Attention (MLA) with compressed KV cache
#   - Sigmoid routing with e_score_correction_bias + normalized top-K
#   - Blockwise FP8 quantization (e4m3, 128x128 blocks) in HF checkpoint
#   - Per-channel FP8 re-quantization for NKI TKG megakernel
#   - YaRN RoPE (factor=64, max_position_embeddings=262144)
#
# Supported configuration:
#   - trn2.48xlarge: TP=64, EP=1, LNC=2 (64 logical cores)
#   - Per-channel FP8 for routed expert weights (expert_wise_per_channel_symmetric)
#   - CPU greedy sampling (no on-device sampling)
#
# References:
#   - NxDI DeepseekV3Attention: models/deepseek/modeling_deepseek.py
#   - Qwen3-MoE reference: models/qwen3_moe/modeling_qwen3_moe.py

import gc
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
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
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from neuronx_distributed.modules.moe.routing import RouterTopK

from neuronx_distributed_inference.models.deepseek.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
    yarn_get_mscale,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def get_rmsnorm_cls():
    return KimiK2RMSNorm if cpu_mode() else CustomRMSNorm


class KimiK2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
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
# MoE initialization
# ---------------------------------------------------------------------------


def initialize_kimi_k2_moe_module(config: "KimiK2InferenceConfig"):
    """Initialize MoE module with sigmoid routing + e_score_correction_bias.

    Uses standard RouterTopK with bias=True. The e_score_correction_bias is
    loaded as the router linear layer's bias (pre-sigmoid), which is a slight
    semantics change from the original post-sigmoid application. The biases are
    small (~0.03-0.1) so the approximation is acceptable.
    """
    from neuronx_distributed_inference.modules.moe_v2 import (
        initialize_moe_process_group,
    )
    from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
    from neuronx_distributed.modules.moe.model import MoE
    from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig

    enabled_hybrid_sharding = config.neuron_config.hybrid_sharding_config is not None
    (
        moe_tkg_tensor_model_parallel_group,
        moe_tkg_expert_model_parallel_group,
        moe_cte_tensor_model_parallel_group,
        moe_cte_expert_model_parallel_group,
    ) = initialize_moe_process_group(config, enabled_hybrid_sharding)

    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        bias=True,
        apply_act_fn_over_topk=False,
        store_transposed_weights=False,
    )

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            bias=False,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping,
            capacity_factor=config.neuron_config.capacity_factor,
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

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=None,
        rmsnorm=None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        sequence_dimension=1,
        init_tkg_module=False,
        tkg_config=None,
    )
    moe.eval()
    return moe


# ---------------------------------------------------------------------------
# Shared Expert MLP (dense, always active)
# ---------------------------------------------------------------------------


class KimiK2SharedExpertMLP(nn.Module):
    """Shared expert MLP (SwiGLU) for Kimi-K2. Always active, not routed."""

    def __init__(self, config: "KimiK2InferenceConfig"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.shared_intermediate_size = config.moe_intermediate_size

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            self.shared_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            self.shared_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            self.shared_intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Dense MLP (for first_k_dense_replace layers, i.e. layer 0)
# ---------------------------------------------------------------------------


class KimiK2DenseMLP(nn.Module):
    """Standard SwiGLU MLP for the dense layers (first_k_dense_replace = 1)."""

    def __init__(self, config: "KimiK2InferenceConfig"):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MLA Attention (Multi-Latent Attention)
# ---------------------------------------------------------------------------


class KimiK2Attention(NeuronAttentionBase):
    """
    Multi-Latent Attention (MLA) for Kimi-K2.

    KV cache format: stores (k_pe, compressed_kv) concatenated.
    - K cache: [batch, 1, seq, qk_rope_head_dim + kv_lora_rank]
    - V cache: same (placeholder, only K is read during decode)
    """

    def __init__(
        self,
        config: "KimiK2InferenceConfig",
        layer_idx: int,
        tensor_model_parallel_group=None,
    ):
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.v_head_dim,
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.layer_idx = layer_idx
        self.bias = getattr(config, "attention_bias", False)
        self.attention_dropout = config.attention_dropout
        self.num_total_heads = config.num_attention_heads

        if cpu_mode():
            self.num_heads = self.num_total_heads
        else:
            self.num_heads = self.num_total_heads // config.neuron_config.tp_degree

        # MLA dimensions
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.head_dim = self.v_head_dim

        # MLA doesn't use fused QKV
        self.qkv_proj = None

        # YaRN RoPE
        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=config.rope_scaling["factor"],
            base=config.rope_theta,
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
        )

        # Softmax scale with mscale adjustment
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.is_causal = True
        self._init_mla_projections(config)

    def _init_mla_projections(self, config):
        dtype = self.torch_dtype
        tp_group = self.tensor_model_parallel_group

        # Query projection (LoRA: hidden -> q_lora_rank -> num_heads * q_head_dim)
        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_total_heads * self.q_head_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
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
                tensor_model_parallel_group=tp_group,
            )

        # KV projection (compressed: hidden -> kv_lora_rank + qk_rope_head_dim)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.kv_a_layernorm = get_rmsnorm_cls()(config.kv_lora_rank)

        # kv_b_proj: decompresses latent -> per-head qk_nope + v
        if tp_group is not None:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )

        # Output projection
        if tp_group is not None:
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=True,
                dtype=self.torch_dtype,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=tp_group,
                reduce_dtype=self.rpl_reduce_dtype,
            )
        else:
            self.o_proj = nn.Linear(
                self.num_attention_heads * self.head_dim,
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

        # Weight absorption: precompute from kv_b_proj weights
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
        out_absorb = wkv_b[:, self.qk_nope_head_dim :, :]  # V absorption
        q_absorb = wkv_b[:, : self.qk_nope_head_dim, :]  # Q nope absorption

        # Query projection
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # KV compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        q_nope, q_pe = torch.tensor_split(q, (self.qk_nope_head_dim,), dim=-1)
        compressed_kv, k_pe = torch.tensor_split(
            compressed_kv, (self.kv_lora_rank,), dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Q nope absorption
        q_nope = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

        # RoPE
        seq_len = self.neuron_config.seq_len
        if sin_cache is None and cos_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos_cache, sin_cache, position_ids)

        # Attention scores
        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum(
            "bhqc,blc->bhql", q_nope, compressed_kv
        )
        active_scores *= self.softmax_scale

        if past_key_value is None:
            # Context encoding (prefill)
            active_scores = torch.where(
                attention_mask,
                active_scores,
                torch.finfo(active_scores.dtype).min,
            )
            active_scores = nn.functional.softmax(
                active_scores, dim=-1, dtype=torch.float32
            ).to(k_pe.dtype)
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
        else:
            # Token generation (decode) -- split prior cache
            cached_kv = past_key_value[0]
            if cached_kv.dim() == 4:
                cached_kv = cached_kv.squeeze(1)
            k_pe_prior, compressed_kv_prior = torch.tensor_split(
                cached_kv,
                [self.qk_rope_head_dim],
                dim=-1,
            )
            k_pe_prior = k_pe_prior.reshape(
                bsz,
                1,
                compressed_kv_prior.shape[1],
                self.qk_rope_head_dim,
            )

            prior_scores = torch.matmul(
                q_pe, k_pe_prior.transpose(2, 3)
            ) + torch.einsum("bhqc,blc->bhql", q_nope, compressed_kv_prior)
            prior_scores *= self.softmax_scale
            prior_scores = torch.where(
                attention_mask,
                prior_scores,
                torch.finfo(prior_scores.dtype).min,
            )
            prior_scores = prior_scores.to(torch.float32)

            softmax_prior, softmax_active = manual_softmax(
                prior_scores,
                active_scores,
                is_speculation=False,
            )
            softmax_prior = softmax_prior.to(k_pe.dtype)
            softmax_active = softmax_active.to(k_pe.dtype)

            x = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
            attn_active = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            x = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
            attn_prior = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            attn_output = attn_prior + attn_active

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        # KV cache: concatenate k_pe and compressed_kv
        k_pe_flat = k_pe.squeeze(1)
        kv_for_cache = torch.cat([k_pe_flat, compressed_kv], dim=-1)
        kv_for_cache = kv_for_cache.unsqueeze(1)

        # Store same data in both K and V cache slots (only K is read during decode)
        past_key_value = (kv_for_cache, kv_for_cache)

        return attn_output, past_key_value, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class KimiK2DecoderLayer(nn.Module):
    """Decoder layer: MLA attention + (MoE or Dense MLP) + optional shared expert."""

    def __init__(self, config: "KimiK2InferenceConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = KimiK2Attention(config=config, layer_idx=layer_idx)

        # MLP: dense for layer 0 (first_k_dense_replace=1), MoE for the rest
        self.is_moe_layer = layer_idx >= config.first_k_dense_replace

        if self.is_moe_layer:
            self.mlp = initialize_kimi_k2_moe_module(config=config)
            if config.n_shared_experts > 0:
                self.shared_experts = KimiK2SharedExpertMLP(config)
            else:
                self.shared_experts = None
        else:
            self.mlp = KimiK2DenseMLP(config)
            self.shared_experts = None

        # Routed expert scaling factor (DeepSeek-V3 / Kimi-K2 pattern)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            moe_output = self.mlp(hidden_states, padding_mask)[0]
            moe_output = moe_output * self.routed_scaling_factor
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
                hidden_states = moe_output + shared_output
            else:
                hidden_states = moe_output
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class KimiK2InferenceConfig(InferenceConfig):
    """Inference config for Kimi-K2 (DeepSeek-V3 architecture)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_local_experts = self.n_routed_experts  # 384
        self.first_k_dense_replace = getattr(self, "first_k_dense_replace", 1)
        self.n_shared_experts = getattr(self, "n_shared_experts", 1)

        # Router config: sigmoid with normalized top-k
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "sigmoid"
        self.neuron_config.normalize_top_k_affinities = True

        # GLU MLP for SwiGLU
        self.neuron_config.glu_mlp = True

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_bias",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "kv_lora_rank",
            "max_position_embeddings",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "scoring_func",
            "v_head_dim",
            "vocab_size",
        ]

    def add_derived_config(self):
        self.num_cores_per_group = 1

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MoENeuronConfig]:
        return MoENeuronConfig


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronKimiK2Model(NeuronBaseModel):
    def setup_attr_for_model(self, config: KimiK2InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # MLA KV cache: 1 "head" with dim = qk_rope_head_dim + kv_lora_rank
        self.num_key_value_heads = 1
        config.head_dim = (
            config.qk_rope_head_dim + config.kv_lora_rank
        )  # 64 + 512 = 576

        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: KimiK2InferenceConfig):
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
                KimiK2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


# ---------------------------------------------------------------------------
# Block FP8 Dequantization Utilities
# ---------------------------------------------------------------------------

# FP8 E4M3 max representable value.
# PyTorch's e4m3fn has max=448 (no NaN encoding), but with
# --experimental-unsafe-fp8e4m3fn-as-fp8e4m3, exponent-15 values (>240) become NaN.
# Use 240.0 to ensure all quantized values stay in the e4m3-safe range.
_FP8_E4M3_MAX = 240.0


def _dequant_block_fp8_to_fp32(
    fp8_weight: Tensor, block_scales: Tensor, block_size: List[int]
) -> Tensor:
    """Dequantize block-FP8 (e4m3, 128x128 blocks) weight to FP32."""
    se = block_scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(
        block_size[1], dim=1
    )
    if se.shape != fp8_weight.shape:
        se = se[: fp8_weight.shape[0], : fp8_weight.shape[1]]
    return fp8_weight.to(torch.float32) * se.to(torch.float32)


def _clamp_fp8_exponent15(fp8_weight: Tensor) -> Tensor:
    """Clamp FP8 e4m3fn bytes that have exponent=15, which become NaN in e4m3.

    On Neuron hardware with --experimental-unsafe-fp8e4m3fn-as-fp8e4m3,
    bytes 0x78-0x7E and 0xF8-0xFE become NaN. Clamp to max safe values.
    """
    raw = fp8_weight.view(torch.uint8)
    pos_exp15 = (raw >= 0x78) & (raw <= 0x7E)
    neg_exp15 = (raw >= 0xF8) & (raw <= 0xFE)
    clamped = raw.clone()
    clamped[pos_exp15] = 0x77  # +240.0
    clamped[neg_exp15] = 0xF7  # -240.0
    return clamped.view(torch.float8_e4m3fn)


def _pack_experts_blockwise_fp8(
    expert_fp8_weights: List[Tensor],
    expert_block_scales: List[Tensor],
    block_size: List[int],
    tp_degree: int,
    layout: str = "gate_up",
) -> Tuple[Tensor, Tensor]:
    """Pack per-expert FP8 weights and block scales into fused tensors.

    Preserves original FP8 bytes (with exponent-15 clamping) and packs
    block-wise scales into the matching layout. This avoids the lossy
    FP8->FP32->FP8 re-quantization path.

    For gate_up (ColumnParallel, stride=2): packs [gate_w.T, up_w.T] -> [E, H, 2*I]
    For down (RowParallel, stride=1): packs [down_w.T] -> [E, I, H]
    """
    n_experts = len(expert_fp8_weights)
    bs0, bs1 = block_size

    if layout == "gate_up":
        gate0, up0 = expert_fp8_weights[0]
        I, H = gate0.shape
        packed_w = torch.empty(n_experts, H, 2 * I, dtype=torch.float8_e4m3fn)

        gs0, us0 = expert_block_scales[0]
        sI, sH = gs0.shape
        raw_scale = torch.empty(n_experts, sH, 2 * sI, dtype=torch.float32)

        for e in range(n_experts):
            g_fp8, u_fp8 = expert_fp8_weights[e]
            g_scale, u_scale = expert_block_scales[e]
            g_fp8 = _clamp_fp8_exponent15(g_fp8)
            u_fp8 = _clamp_fp8_exponent15(u_fp8)
            packed_w[e, :, :I] = g_fp8.T
            packed_w[e, :, I:] = u_fp8.T
            raw_scale[e, :, :sI] = g_scale.T
            raw_scale[e, :, sI:] = u_scale.T

        out_dim = 2 * I
        per_tp = out_dim // tp_degree
        repeat_factor = bs1 // per_tp if per_tp < bs1 else 1
        if repeat_factor > 1:
            expanded_scale = raw_scale.repeat_interleave(repeat_factor, dim=2)
        else:
            expanded_scale = raw_scale

        return packed_w, expanded_scale

    elif layout == "down":
        d0 = expert_fp8_weights[0]
        H_orig, I = d0.shape
        packed_w = torch.empty(n_experts, I, H_orig, dtype=torch.float8_e4m3fn)

        ds0 = expert_block_scales[0]
        sH, sI = ds0.shape
        raw_scale = torch.empty(n_experts, sI, sH, dtype=torch.float32)

        for e in range(n_experts):
            d_fp8 = expert_fp8_weights[e]
            d_scale = expert_block_scales[e]
            d_fp8 = _clamp_fp8_exponent15(d_fp8)
            packed_w[e] = d_fp8.T
            raw_scale[e] = d_scale.T

        in_dim = I
        per_tp = in_dim // tp_degree
        repeat_factor = bs0 // per_tp if per_tp < bs0 else 1
        if repeat_factor > 1:
            expanded_scale = raw_scale.repeat_interleave(repeat_factor, dim=1)
        else:
            expanded_scale = raw_scale

        return packed_w, expanded_scale

    else:
        raise ValueError(f"Unknown layout: {layout}")


def _requantize_per_channel_fp8(bf16_weight: Tensor) -> Tuple[Tensor, Tensor]:
    """Re-quantize a BF16 weight to per-expert per-channel FP8 E4M3.

    Per-expert per-channel means one scale per output column PER EXPERT.
    This matches NxDI's EXPERT_WISE_PER_CHANNEL_SYMMETRIC quantization type
    which uses scale shape [E, 1, output_per_tp].

    Args:
        bf16_weight: [E, H, W] bfloat16 tensor (experts x input x output)
    Returns:
        (fp8_weight, per_expert_per_channel_scale) where:
            fp8_weight: [E, H, W] float8_e4m3fn
            per_expert_per_channel_scale: [E, 1, W] float32 (per-expert per-channel)
    """
    fp32_weight = bf16_weight.float()
    # max abs per output column, PER EXPERT (reduce dim 1 = input rows only)
    amax = fp32_weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [E, 1, W]
    scale = amax / _FP8_E4M3_MAX
    scaled = (fp32_weight / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    fp8_weight = scaled.to(torch.float8_e4m3fn)

    # Clamp exponent-15 bytes (0x78-0x7E, 0xF8-0xFE become NaN in e4m3)
    raw = fp8_weight.view(torch.uint8)
    pos_exp15 = (raw >= 0x78) & (raw <= 0x7E)
    neg_exp15 = (raw >= 0xF8) & (raw <= 0xFE)
    raw = torch.where(pos_exp15, torch.tensor(0x77, dtype=torch.uint8), raw)
    raw = torch.where(neg_exp15, torch.tensor(0xF7, dtype=torch.uint8), raw)
    fp8_weight = raw.view(torch.float8_e4m3fn)

    return fp8_weight, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# State Dict Conversion
# ---------------------------------------------------------------------------


def convert_kimi_k2_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: KimiK2InferenceConfig,
) -> Dict[str, Any]:
    """Convert HuggingFace Kimi-K2 / DeepSeek-V3 weights to NxDI format.

    Key conversions:
    1. Dequantize block-FP8 weights to BF16 (if FP8 weights detected)
    2. Rename router weights (gate.weight -> router.linear_router.weight)
    3. Concatenate per-expert weights into packed tensors for ExpertMLPsV2
    4. Handle shared expert weight naming
    5. Handle e_score_correction_bias loading as router bias
    """
    # Check if weights are already pre-converted
    has_packed_experts = any(
        "expert_mlps.mlp_op.gate_up_proj.weight" in k for k in neuron_state_dict
    )
    has_per_expert = any(
        "mlp.experts.0.gate_proj.weight" in k for k in neuron_state_dict
    )

    if has_packed_experts and not has_per_expert:
        logger.info("Weights already pre-converted. Adding rank_util only.")
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0,
            config.neuron_config.tp_degree,
            dtype=torch.int32,
        )
        for layer_idx in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = (
                torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)
            )
        return neuron_state_dict

    # Dequantize FP8 weights
    _maybe_dequantize_fp8(neuron_state_dict, config)

    # Add rank utility tensors
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0,
        config.neuron_config.tp_degree,
        dtype=torch.int32,
    )

    for layer_idx in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = (
            torch.arange(0, config.neuron_config.tp_degree, dtype=torch.int32)
        )

        is_moe_layer = layer_idx >= config.first_k_dense_replace
        if not is_moe_layer:
            continue

        # Router weights
        gate_key = f"layers.{layer_idx}.mlp.gate.weight"
        if gate_key in neuron_state_dict:
            neuron_state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
                neuron_state_dict[gate_key].detach().clone()
            )
            del neuron_state_dict[gate_key]

        # e_score_correction_bias
        bias_key = f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        if bias_key in neuron_state_dict:
            neuron_state_dict[
                f"layers.{layer_idx}.mlp.router.e_score_correction_bias"
            ] = neuron_state_dict[bias_key].detach().clone()
            del neuron_state_dict[bias_key]

        # Expert weights: per-expert -> packed format
        expert_0_gate = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if expert_0_gate not in neuron_state_dict:
            continue

        intermediate_size, hidden_size = neuron_state_dict[expert_0_gate].shape
        device = neuron_state_dict[expert_0_gate].device
        dtype = neuron_state_dict[expert_0_gate].dtype
        num_experts = config.n_routed_experts

        # Concatenate gate + up projections
        gate_up_proj = torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )
        for e in range(num_experts):
            gate_w = (
                neuron_state_dict[
                    f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
                ]
                .T.detach()
                .clone()
            )
            up_w = (
                neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"]
                .T.detach()
                .clone()
            )
            gate_up_proj[e, :, :intermediate_size] = gate_w
            gate_up_proj[e, :, intermediate_size:] = up_w
            del neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            ]
            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"]

        neuron_state_dict[
            f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        ] = gate_up_proj

        # Down projections
        down_proj = torch.empty(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for e in range(num_experts):
            down_w = (
                neuron_state_dict[
                    f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
                ]
                .T.detach()
                .clone()
            )
            down_proj[e] = down_w
            del neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            ]

        neuron_state_dict[
            f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"
        ] = down_proj

        # Shared expert rename
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            hf_key = f"layers.{layer_idx}.mlp.shared_experts.{proj}.weight"
            nxdi_key = f"layers.{layer_idx}.shared_experts.{proj}.weight"
            if hf_key in neuron_state_dict:
                neuron_state_dict[nxdi_key] = neuron_state_dict[hf_key]
                del neuron_state_dict[hf_key]

        gc.collect()

    return neuron_state_dict


def _maybe_dequantize_fp8(
    neuron_state_dict: Dict[str, Any],
    config: "KimiK2InferenceConfig",
):
    """Dequantize block-FP8 weights to BF16."""
    scale_layers = []

    for layer_key in list(neuron_state_dict.keys()):
        if "_scale_inv" in layer_key or "weight_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)

            if layer_key.endswith(".weight_scale_inv"):
                fp8_layer_name = layer_key.replace(".weight_scale_inv", ".weight")
            elif "_scale_inv" in layer_key:
                fp8_layer_name = layer_key.replace("_scale_inv", "")

            if fp8_layer_name not in neuron_state_dict:
                continue

            fp8_layer = neuron_state_dict[fp8_layer_name]

            if hasattr(config, "quantization_config") and config.quantization_config:
                block_size = config.quantization_config.get(
                    "weight_block_size", [128, 128]
                )
            else:
                block_size = [128, 128]

            fp32_val = _dequant_block_fp8_to_fp32(fp8_layer, scales, block_size)
            neuron_state_dict[fp8_layer_name] = fp32_val.to(
                config.neuron_config.torch_dtype
            )

    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


# ---------------------------------------------------------------------------
# Top-level CausalLM
# ---------------------------------------------------------------------------


class NeuronKimiK2ForCausalLM(NeuronBaseForCausalLM):
    """Kimi-K2 for causal language modeling on Neuron."""

    _model_cls = NeuronKimiK2Model

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[KimiK2InferenceConfig]:
        return KimiK2InferenceConfig

    @staticmethod
    def _apply_ep_scale_fix():
        """Monkey-patch ExpertFusedLinear._mark_expert_parallel_weights to skip
        per-channel scale params that can't be EP-sharded (shape [1, 1, W])."""
        from neuronx_distributed.modules.moe.moe_parallel_layers import (
            ExpertFusedLinear,
        )

        _original_mark = ExpertFusedLinear._mark_expert_parallel_weights

        def _patched_mark(
            self_inner,
            iterable=None,
            expert_parallel_group_size=None,
            is_prefill=True,
            expert_distribution=None,
        ):
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_expert_model_parallel_size,
            )

            if expert_parallel_group_size is None:
                expert_parallel_group_size = get_expert_model_parallel_size()

            if expert_parallel_group_size > 1:
                if iterable is None:
                    params_to_mark = []
                    for name, p in self_inner.named_parameters():
                        if name == "scale" and p.shape[0] == 1:
                            continue
                        params_to_mark.append(p)
                    iterable = params_to_mark

                for p in iterable:
                    p.expert_model_parallel = True
                    if is_prefill:
                        p.is_prefill = True
                    p.expert_distribution = expert_distribution

        ExpertFusedLinear._mark_expert_parallel_weights = _patched_mark

    @staticmethod
    def _apply_blockwise_scale_stride_fix():
        """Monkey-patch _setup_for_scale to use stride=1 for blockwise symmetric
        scales, which avoids strided splitting failures when per-rank weight size
        is smaller than block size."""
        from neuronx_distributed.quantization.quantization_layers import (
            BaseQuantizeParallelLinear,
        )
        from neuronx_distributed.quantization.quantization_config import (
            QuantizationType,
        )

        _original_setup = BaseQuantizeParallelLinear._setup_for_scale

        def _patched_setup(self_inner, *args, **kwargs):
            _original_setup(self_inner, *args, **kwargs)
            if (
                hasattr(self_inner, "quantization_type")
                and self_inner.quantization_type == QuantizationType.BLOCKWISE_SYMMETRIC
                and hasattr(self_inner, "scale")
                and hasattr(self_inner.scale, "partition_stride")
                and self_inner.scale.partition_stride > 1
            ):
                self_inner.scale.partition_stride = 1

        BaseQuantizeParallelLinear._setup_for_scale = _patched_setup

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        """Override to apply EP scale fix before loading."""
        if getattr(self.neuron_config, "quantized", False):
            self._apply_ep_scale_fix()
            self._apply_blockwise_scale_stride_fix()
        return super().load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict,
        config: KimiK2InferenceConfig,
    ) -> dict:
        return convert_kimi_k2_hf_to_neuron_state_dict(state_dict, config)

    def checkpoint_loader_fn(self, mmap: bool = False):
        """Memory-efficient streaming checkpoint loader for 1T-parameter models.

        Loads safetensor shards one at a time, processes weights (FP8 handling,
        expert packing, router renaming), and accumulates results to avoid OOM.

        When quantized=True (FP8 path):
        - Per-channel: dequant blockwise FP8 -> BF16 -> re-quantize per-channel FP8
          with [E, 1, W] scales (expert_wise_per_channel_symmetric)
        - Blockwise: keep original FP8 bytes with [E, H/bs, W/bs] block scales
        - All other weights are dequantized to BF16

        When quantized=False (BF16 path):
        - All weights are dequantized from FP8 to BF16
        """
        import json as json_mod
        from safetensors.torch import load_file

        model_path = getattr(self.config, "_name_or_path", None)
        if model_path is None or not os.path.exists(str(model_path)):
            model_path = self.model_path

        index_path = os.path.join(model_path, "model.safetensors.index.json")

        if not os.path.exists(index_path):
            return super().checkpoint_loader_fn(mmap=mmap)

        with open(index_path, "r") as f:
            index = json_mod.load(f)

        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))

        quant_config = getattr(self.config, "quantization_config", None)
        if isinstance(quant_config, dict):
            block_size = quant_config.get("weight_block_size", [128, 128])
        else:
            block_size = [128, 128]
        n_routed_experts = getattr(self.config, "n_routed_experts", 384)
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 1)
        keep_experts_fp8 = getattr(self.config.neuron_config, "quantized", False)
        quantization_type = getattr(
            self.config.neuron_config, "quantization_type", "blockwise_symmetric"
        )
        use_per_channel = quantization_type in (
            "per_channel_symmetric",
            "expert_wise_per_channel_symmetric",
        )
        num_layers = self.config.num_hidden_layers

        # Determine which shards are needed (supports reduced-layer testing)
        needed_shards = set()
        for key, shard_file in weight_map.items():
            clean_key = key[len("model.") :] if key.startswith("model.") else key
            if "layers." in clean_key:
                parts = clean_key.split(".")
                idx = parts.index("layers") + 1
                layer_idx = int(parts[idx])
                if layer_idx < num_layers:
                    needed_shards.add(shard_file)
            else:
                needed_shards.add(shard_file)

        logger.info(
            f"Streaming loader: {len(shard_files)} shards, {len(needed_shards)} needed, "
            f"block_size={block_size}, experts={n_routed_experts}, fp8={keep_experts_fp8}, "
            f"quant_type={quantization_type}"
        )

        result_dict = {}
        for i, shard_file in enumerate(shard_files):
            if shard_file not in needed_shards:
                continue
            shard_path = os.path.join(model_path, shard_file)
            logger.info(f"Loading shard [{i + 1}/{len(shard_files)}]: {shard_file}")

            shard_data = load_file(shard_path)

            # Strip "model." prefix
            for key in list(shard_data.keys()):
                if key.startswith("model."):
                    shard_data[key[len("model.") :]] = shard_data.pop(key)

            # Filter out keys for layers beyond num_hidden_layers
            for key in list(shard_data.keys()):
                if "layers." in key:
                    parts = key.split(".")
                    idx = parts.index("layers") + 1
                    layer_idx = int(parts[idx])
                    if layer_idx >= num_layers:
                        del shard_data[key]

            # Determine layers in this shard
            layer_ids = set()
            for key in shard_data:
                if "layers." in key:
                    parts = key.split(".")
                    idx = parts.index("layers") + 1
                    layer_ids.add(int(parts[idx]))

            # Build expert weight/scale key mapping
            expert_weight_keys = set()
            expert_scale_keys = {}
            if keep_experts_fp8:
                for key in list(shard_data.keys()):
                    if ".mlp.experts." in key and ".weight" in key:
                        if ".shared_experts." not in key:
                            expert_weight_keys.add(key)

                for key in list(shard_data.keys()):
                    if "_scale_inv" in key or "weight_scale_inv" in key:
                        if key.endswith(".weight_scale_inv"):
                            wk = key.replace(".weight_scale_inv", ".weight")
                        elif "_scale_inv" in key:
                            wk = key.replace("_scale_inv", "")
                        else:
                            continue
                        if wk in expert_weight_keys:
                            expert_scale_keys[wk] = key

            # Process FP8 weights: dequant non-expert, keep expert raw
            scale_keys = [
                k for k in shard_data if "weight_scale_inv" in k or "_scale_inv" in k
            ]
            for scale_key in scale_keys:
                scales = shard_data[scale_key]
                if scale_key.endswith(".weight_scale_inv"):
                    weight_key = scale_key.replace(".weight_scale_inv", ".weight")
                elif "_scale_inv" in scale_key:
                    weight_key = scale_key.replace("_scale_inv", "")
                else:
                    del shard_data[scale_key]
                    continue

                if weight_key not in shard_data:
                    del shard_data[scale_key]
                    continue

                if shard_data[weight_key].dtype != torch.float8_e4m3fn:
                    del shard_data[scale_key]
                    continue

                if weight_key in expert_weight_keys:
                    pass  # Keep for expert packing
                else:
                    fp8_w = shard_data[weight_key]
                    fp32_w = _dequant_block_fp8_to_fp32(fp8_w, scales, block_size)
                    shard_data[weight_key] = fp32_w.to(torch.bfloat16)
                    del fp8_w, fp32_w
                    del shard_data[scale_key]

            # Remove orphan scale keys
            for k in [
                k
                for k in shard_data
                if ("_scale_inv" in k or "weight_scale_inv" in k)
                and k not in expert_scale_keys.values()
            ]:
                del shard_data[k]

            # Cast non-FP8 tensors to BF16
            for key in list(shard_data.keys()):
                t = shard_data[key]
                if torch.is_floating_point(t) and t.dtype not in (
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float32,
                ):
                    if t.dtype != torch.int64 and t.dtype != torch.int32:
                        shard_data[key] = t.to(torch.bfloat16)

            # Pack experts and rename for MoE layers
            for layer_idx in sorted(layer_ids):
                prefix = f"layers.{layer_idx}"
                if layer_idx >= first_k_dense_replace:
                    # Router rename
                    gate_key = f"{prefix}.mlp.gate.weight"
                    if gate_key in shard_data:
                        shard_data[f"{prefix}.mlp.router.linear_router.weight"] = (
                            shard_data.pop(gate_key)
                        )
                    bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
                    if bias_key in shard_data:
                        shard_data[f"{prefix}.mlp.router.linear_router.bias"] = (
                            shard_data.pop(bias_key)
                        )

                    # Pack experts
                    e0_gate = f"{prefix}.mlp.experts.0.gate_proj.weight"
                    if e0_gate in shard_data:
                        isize, hsize = shard_data[e0_gate].shape

                        if keep_experts_fp8:
                            if use_per_channel:
                                # Per-channel FP8: dequant blockwise -> BF16 -> pack -> requant per-channel
                                gate_up_bf16 = torch.zeros(
                                    n_routed_experts,
                                    hsize,
                                    2 * isize,
                                    dtype=torch.bfloat16,
                                    device="cpu",
                                )
                                down_bf16 = torch.zeros(
                                    n_routed_experts,
                                    isize,
                                    hsize,
                                    dtype=torch.bfloat16,
                                    device="cpu",
                                )

                                for e in range(n_routed_experts):
                                    gk = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                                    uk = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                                    dk = f"{prefix}.mlp.experts.{e}.down_proj.weight"
                                    gsk = expert_scale_keys.get(gk)
                                    usk = expert_scale_keys.get(uk)
                                    dsk = expert_scale_keys.get(dk)

                                    g_fp8 = (
                                        shard_data.pop(gk) if gk in shard_data else None
                                    )
                                    u_fp8 = (
                                        shard_data.pop(uk) if uk in shard_data else None
                                    )
                                    g_scale = (
                                        shard_data.pop(gsk)
                                        if gsk and gsk in shard_data
                                        else None
                                    )
                                    u_scale = (
                                        shard_data.pop(usk)
                                        if usk and usk in shard_data
                                        else None
                                    )

                                    if g_fp8 is not None and u_fp8 is not None:
                                        g_bf16 = _dequant_block_fp8_to_fp32(
                                            g_fp8, g_scale, block_size
                                        ).to(torch.bfloat16)
                                        u_bf16 = _dequant_block_fp8_to_fp32(
                                            u_fp8, u_scale, block_size
                                        ).to(torch.bfloat16)
                                        gate_up_bf16[e, :, :isize] = g_bf16.T
                                        gate_up_bf16[e, :, isize:] = u_bf16.T
                                        del (
                                            g_fp8,
                                            u_fp8,
                                            g_scale,
                                            u_scale,
                                            g_bf16,
                                            u_bf16,
                                        )

                                    d_fp8 = (
                                        shard_data.pop(dk) if dk in shard_data else None
                                    )
                                    d_scale = (
                                        shard_data.pop(dsk)
                                        if dsk and dsk in shard_data
                                        else None
                                    )
                                    if d_fp8 is not None:
                                        d_bf16 = _dequant_block_fp8_to_fp32(
                                            d_fp8, d_scale, block_size
                                        ).to(torch.bfloat16)
                                        down_bf16[e] = d_bf16.T
                                        del d_fp8, d_scale, d_bf16

                                # Re-quantize to per-channel FP8 [E, 1, W] scales
                                gu_fp8, gu_scale = _requantize_per_channel_fp8(
                                    gate_up_bf16
                                )
                                shard_data[
                                    f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                                ] = gu_fp8
                                shard_data[
                                    f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
                                ] = gu_scale
                                del gate_up_bf16, gu_fp8, gu_scale

                                dn_fp8, dn_scale = _requantize_per_channel_fp8(
                                    down_bf16
                                )
                                shard_data[
                                    f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                                ] = dn_fp8
                                shard_data[
                                    f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.scale"
                                ] = dn_scale
                                del down_bf16, dn_fp8, dn_scale

                            else:
                                # Blockwise FP8: keep original FP8 bytes with block scales
                                gate_up_weights = []
                                gate_up_scales = []
                                down_weights = []
                                down_scales = []

                                for e in range(n_routed_experts):
                                    gk = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                                    uk = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                                    dk = f"{prefix}.mlp.experts.{e}.down_proj.weight"
                                    gsk = expert_scale_keys.get(gk)
                                    usk = expert_scale_keys.get(uk)
                                    dsk = expert_scale_keys.get(dk)

                                    g_fp8 = (
                                        shard_data.pop(gk) if gk in shard_data else None
                                    )
                                    u_fp8 = (
                                        shard_data.pop(uk) if uk in shard_data else None
                                    )
                                    g_scale = (
                                        shard_data.pop(gsk)
                                        if gsk and gsk in shard_data
                                        else None
                                    )
                                    u_scale = (
                                        shard_data.pop(usk)
                                        if usk and usk in shard_data
                                        else None
                                    )

                                    if g_fp8 is not None and u_fp8 is not None:
                                        gate_up_weights.append((g_fp8, u_fp8))
                                        gate_up_scales.append((g_scale, u_scale))

                                    d_fp8 = (
                                        shard_data.pop(dk) if dk in shard_data else None
                                    )
                                    d_scale = (
                                        shard_data.pop(dsk)
                                        if dsk and dsk in shard_data
                                        else None
                                    )
                                    if d_fp8 is not None:
                                        down_weights.append(d_fp8)
                                        down_scales.append(d_scale)

                                if gate_up_weights:
                                    gu_fp8, gu_scale = _pack_experts_blockwise_fp8(
                                        gate_up_weights,
                                        gate_up_scales,
                                        block_size,
                                        tp_degree=self.config.neuron_config.tp_degree,
                                        layout="gate_up",
                                    )
                                    shard_data[
                                        f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                                    ] = gu_fp8
                                    shard_data[
                                        f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
                                    ] = gu_scale
                                    del gate_up_weights, gate_up_scales

                                if down_weights:
                                    dn_fp8, dn_scale = _pack_experts_blockwise_fp8(
                                        down_weights,
                                        down_scales,
                                        block_size,
                                        tp_degree=self.config.neuron_config.tp_degree,
                                        layout="down",
                                    )
                                    shard_data[
                                        f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                                    ] = dn_fp8
                                    shard_data[
                                        f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.scale"
                                    ] = dn_scale
                                    del down_weights, down_scales

                        else:
                            # BF16 path
                            dtype = shard_data[e0_gate].dtype
                            gate_up = torch.zeros(
                                n_routed_experts,
                                hsize,
                                2 * isize,
                                dtype=dtype,
                                device="cpu",
                            )
                            for e in range(n_routed_experts):
                                gk = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
                                uk = f"{prefix}.mlp.experts.{e}.up_proj.weight"
                                if gk in shard_data:
                                    gate_up[e, :, :isize] = shard_data.pop(gk).T
                                if uk in shard_data:
                                    gate_up[e, :, isize:] = shard_data.pop(uk).T

                            shard_data[
                                f"{prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                            ] = gate_up

                            down = torch.zeros(
                                n_routed_experts,
                                isize,
                                hsize,
                                dtype=dtype,
                                device="cpu",
                            )
                            for e in range(n_routed_experts):
                                dk = f"{prefix}.mlp.experts.{e}.down_proj.weight"
                                if dk in shard_data:
                                    down[e] = shard_data.pop(dk).T

                            shard_data[
                                f"{prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                            ] = down

                    # Clean up remaining per-expert keys
                    for e in range(n_routed_experts):
                        for proj in ["gate_proj", "up_proj", "down_proj"]:
                            for suffix in [".weight", ".weight_scale_inv"]:
                                k = f"{prefix}.mlp.experts.{e}.{proj}{suffix}"
                                if k in shard_data:
                                    del shard_data[k]

                    # Shared expert rename
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        hf_key = f"{prefix}.mlp.shared_experts.{proj}.weight"
                        nxdi_key = f"{prefix}.shared_experts.{proj}.weight"
                        if hf_key in shard_data:
                            shard_data[nxdi_key] = shard_data.pop(hf_key)

            # Cast remaining float32 non-scale tensors to BF16
            for key in list(shard_data.keys()):
                t = shard_data[key]
                if (
                    t.dtype == torch.float32
                    and not key.endswith(".scale")
                    and not key.endswith("_scale_inv")
                    and not key.endswith("linear_router.bias")
                ):
                    shard_data[key] = t.to(torch.bfloat16)

            result_dict.update(shard_data)
            del shard_data
            gc.collect()

        # Add rank_util tensors
        tp = self.config.neuron_config.tp_degree
        result_dict["rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)
        for layer_idx in range(self.config.num_hidden_layers):
            result_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
                0, tp, dtype=torch.int32
            )

        # Add fused prefix if needed
        if self._FUSED_PREFIX != "":
            for key in list(result_dict.keys()):
                result_dict[f"{self._FUSED_PREFIX}.{key}"] = result_dict.pop(key)

        logger.info(f"Streaming loader done. Total keys: {len(result_dict)}")
        return result_dict

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self) -> str:
        """Compiler args optimized for Kimi-K2 on trn2.

        Key flags:
        - -O1 for both CTE and TKG (no EP all-to-all overhead at EP=1)
        - --internal-enable-dge-levels vector_dynamic_offsets: DGE optimization
        - --enable-ccop-compute-overlap: CC overlap for MoE
        - --lnc: must match runtime NEURON_LOGICAL_NC_CONFIG
        """
        lnc = getattr(self.neuron_config, "logical_nc_config", 2)

        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            "--model-type transformer "
            "-O1"
        )
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += f" --lnc={lnc}"

        hlo2tensorizer_opts = "--verify-hlo=true"
        if (
            getattr(self.neuron_config, "quantized", False)
            and getattr(self.neuron_config, "quantization_dtype", "") == "f8e4m3"
        ):
            hlo2tensorizer_opts += " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"

        compiler_args += f" --internal-hlo2tensorizer-options='{hlo2tensorizer_opts}'"

        return compiler_args
