# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import logging
from typing import Optional, Tuple, Type

import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region
from neuronx_distributed.utils import cpu_mode
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.deepseek.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

logger = logging.getLogger(__name__)


class DeepseekV3InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return DeepseekV3RMSNorm if cpu_mode() else CustomRMSNorm


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV3Attention(NeuronAttentionBase):

    def __init__(self, config: DeepseekV3InferenceConfig, layer_idx: Optional[int] = None, tensor_model_parallel_group=None):

        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Not applicable for MLA and set as the same as attn heads
            head_dim=config.v_head_dim,
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=config.rope_scaling["factor"],
            base=config.rope_theta,
            mscale=self.config.rope_scaling["mscale"],
            mscale_all_dim=self.config.rope_scaling["mscale_all_dim"],
            beta_fast=self.config.rope_scaling["beta_fast"],
            beta_slow=self.config.rope_scaling["beta_slow"],
        )
        # TODO: manual offset qkv_proj created from base class. Refactor base so it doesnt always create this property
        self.qkv_proj = None
        self.bias = getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        assert layer_idx is not None, "Please make sure to provide a `layer_idx` when creating this class."

        self.attention_dropout = config.attention_dropout
        self.num_total_heads = config.num_attention_heads
        assert self.num_attention_heads % self.tp_degree == 0, "Number of attention heads must be a multiple of tp degree."
        if cpu_mode():
            self.num_heads = self.num_total_heads
        else:
            self.num_heads = self.num_total_heads // self.config.neuron_config.tp_degree

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.head_dim = self.v_head_dim

        self.is_causal = True
        self.init_mla_properties()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                from neuronx_distributed_inference.models.deepseek.rope_util import yarn_get_mscale
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def init_mla_properties(self):
        config = self.config
        dtype = self.torch_dtype
        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, self.num_total_heads * self.q_head_dim, bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias, dtype=dtype
            )
            self.q_a_layernorm = get_rmsnorm_cls()(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank, self.num_total_heads * self.q_head_dim, bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype
        )
        self.kv_a_layernorm = get_rmsnorm_cls()(config.kv_lora_rank)
        if self.tensor_model_parallel_group is not None:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                self.num_total_heads
                * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=self.tensor_model_parallel_group
            )
        else:
            self.kv_b_proj = nn.Linear(
                config.kv_lora_rank,
                self.num_total_heads
                * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
                bias=False,
            )

        if self.tensor_model_parallel_group is not None:
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
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
                self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias
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
        """Implements each layer's forward pass for the attention block."""
        if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        bsz, q_len, _ = hidden_states.size()

        # weight matrix absorption
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)

        out_absorb = wkv_b[:, self.v_head_dim:, :]

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        q_nope, q_pe = torch.tensor_split(
            q, (self.qk_nope_head_dim,), dim=-1
        )
        compressed_kv, k_pe = torch.tensor_split(
            compressed_kv, (self.kv_lora_rank,), dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # q_nope absorbing
        q_absorb = wkv_b[:, :self.qk_nope_head_dim]
        q_nope = torch.einsum('hdc,bhqd->bhqc', q_absorb, q_nope)

        seq_len = self.neuron_config.seq_len
        if sin_cache is None and cos_cache is None:
            cos_cache, sin_cache = self.rotary_emb(k_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos_cache, sin_cache, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos_cache, sin_cache, position_ids)

        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
        active_scores *= self.softmax_scale

        if past_key_value is None:
            active_scores = torch.where(attention_mask, active_scores, torch.finfo(active_scores.dtype).min)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                k_pe.dtype
            )

            # attention result with V absorb
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
        else:
            k_pe_prior, compressed_kv_prior = torch.tensor_split(past_key_value, [self.qk_rope_head_dim,], dim=-1)
            k_pe_prior = k_pe_prior.reshape(bsz, 1, compressed_kv_prior.shape[1], self.qk_rope_head_dim)

            # I. scores and softmax
            prior_scores = torch.matmul(q_pe, k_pe_prior.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv_prior)
            prior_scores *= self.softmax_scale
            prior_scores = torch.where(
                attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
            )
            prior_scores = prior_scores.to(torch.float32)

            softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation=False)
            softmax_prior, softmax_active = softmax_prior.to(k_pe.dtype), softmax_active.to(k_pe.dtype)

            # II. attention result with V absorb
            x = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
            attn_active = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            x = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
            attn_prior = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

            attn_output = attn_prior + attn_active

        # transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)
        past_key_value: Tuple[Tensor, Tensor] = (k_pe.squeeze(1), compressed_kv)

        return attn_output, past_key_value, cos_cache, sin_cache


def custom_compiler_args():
    """
    Over-ride function from base class for better control over compiler flags
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    # add dma optimization flag
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    return compiler_args
