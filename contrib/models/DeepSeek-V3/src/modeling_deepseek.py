# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

import gc
import logging
from typing import List, Optional, Tuple, Type

import warnings
import torch
import torch.utils.checkpoint
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region
from neuronx_distributed.utils import cpu_mode
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from src.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.modules.moe.routing import GroupLimitedRouter
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger(__name__)


def _dequantize_fp8_state_dict(state_dict: dict, block_size: int = 128) -> dict:
    """
    Dequantize FP8 block-wise weights to BF16 in-place.

    DeepSeek V3's native FP8 format stores weights as float8_e4m3fn with
    per-block scale factors in corresponding weight_scale_inv tensors.
    Block size is typically 128x128 (from config.quantization_config.weight_block_size).
    """
    scale_inv_keys = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    if not scale_inv_keys:
        return state_dict

    logger.info("Dequantizing %d FP8 weights to BF16 (block_size=%d)...", len(scale_inv_keys), block_size)

    for scale_key in scale_inv_keys:
        weight_key = scale_key.replace(".weight_scale_inv", ".weight")
        if weight_key not in state_dict:
            del state_dict[scale_key]
            continue

        weight = state_dict[weight_key]
        scale_inv = state_dict[scale_key]

        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            del state_dict[scale_key]
            continue

        M, N = weight.shape
        scales_expanded = (
            scale_inv
            .repeat_interleave(block_size, dim=0)
            .repeat_interleave(block_size, dim=1)
        )
        scaled = weight.to(torch.float32) * scales_expanded[:M, :N].to(torch.float32)
        state_dict[weight_key] = scaled.to(torch.bfloat16)
        del state_dict[scale_key]

    # Remove any remaining orphan scale_inv keys
    for key in [k for k in state_dict if k.endswith(".weight_scale_inv")]:
        del state_dict[key]

    gc.collect()
    logger.info("FP8 dequantization complete.")
    return state_dict


def convert_deepseek_v3_hf_to_neuron_state_dict(state_dict: dict, config: "DeepseekV3InferenceConfig") -> dict:
    """
    Convert HuggingFace DeepSeek V3 state dict to Neuron-compatible format.

    Transformations:
    0. Dequantize FP8 weights to BF16 (if present)
    1. Add rank utility tensors for TP sharding
    2. Rename router weights: gate.weight -> router.linear_router.weight
    3. Rename e_score_correction_bias -> router.e_score_correction_bias
    4. Fuse gate_proj + up_proj into gate_up_proj for each expert
    5. Stack down_proj weights across experts
    6. Skip dense layers (first_k_dense_replace layers)
    """
    # Dequantize FP8 weights if present (DeepSeek V3 native FP8 format)
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        # Check the underlying HF config for quantization_config
        load_config = getattr(config, "_load_config", None)
        if load_config:
            quant_config = getattr(load_config, "quantization_config", None)
    block_size = 128
    if quant_config and isinstance(quant_config, dict):
        wbs = quant_config.get("weight_block_size", [128, 128])
        block_size = wbs[0] if isinstance(wbs, (list, tuple)) else wbs
    _dequantize_fp8_state_dict(state_dict, block_size=block_size)

    num_hidden_layers = config.num_hidden_layers
    num_local_experts = config.num_local_experts
    tp_degree = getattr(config.neuron_config, "tp_degree", 1)
    first_k_dense = getattr(config, "first_k_dense_replace", 3)

    # Add rank utilities for TP
    state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

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
            state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
                state_dict[router_key].detach().clone()
            )
            del state_dict[router_key]

        # Rename e_score_correction_bias for GroupLimitedRouter
        bias_key = f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        if bias_key in state_dict:
            state_dict[f"layers.{layer_idx}.mlp.router.e_score_correction_bias"] = (
                state_dict[bias_key].detach().clone()
            )
            del state_dict[bias_key]

        # Check if expert weights exist for this layer
        expert_gate_key = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if expert_gate_key not in state_dict:
            continue

        intermediate_size, hidden_size = state_dict[expert_gate_key].shape
        device = state_dict[expert_gate_key].device
        dtype = state_dict[expert_gate_key].dtype

        # Fuse gate_proj + up_proj into gate_up_proj for all experts
        gate_up_proj = torch.empty(
            num_local_experts, hidden_size, 2 * intermediate_size,
            dtype=dtype, device=device,
        )

        for e in range(num_local_experts):
            gate_key = f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            up_key = f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"

            if gate_key in state_dict and up_key in state_dict:
                gate_proj_weights = state_dict[gate_key].T.detach().clone()
                up_proj_weights = state_dict[up_key].T.detach().clone()

                gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
                torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size).copy_(gate_proj_weights)
                torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size).copy_(up_proj_weights)

                del state_dict[gate_key]
                del state_dict[up_key]

        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Stack down_proj weights across all experts
        down_proj = torch.empty(
            num_local_experts, intermediate_size, hidden_size,
            dtype=dtype, device=device,
        )

        for e in range(num_local_experts):
            down_key = f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            if down_key in state_dict:
                down_proj_weights = state_dict[down_key].T.detach().clone()
                torch.narrow(down_proj, 0, e, 1).copy_(down_proj_weights)
                del state_dict[down_key]

        state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return state_dict


class DeepseekV3NeuronConfig(MoENeuronConfig):
    """Neuron hardware configuration for DeepSeek V3 MoE model."""
    pass


class DeepseekV3Router(GroupLimitedRouter):
    """
    GroupLimitedRouter with DeepSeek V3's routed_scaling_factor.

    After group-limited top-k selection, the selected affinities are
    L1-normalized and then scaled by routed_scaling_factor (2.5).
    This replaces the normalize_top_k_affinities step in ExpertMLPsV2,
    so the config must set normalize_top_k_affinities=False.
    """

    def __init__(self, routed_scaling_factor: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.routed_scaling_factor = routed_scaling_factor
        # e_score_correction_bias is a trained parameter loaded from checkpoint.
        # GroupLimitedRouter references it in noaux_tc_top_k but doesn't register it.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        topk_idx, _ = self.noaux_tc_top_k(expert_affinities)
        topk_idx = topk_idx.detach().to(dtype=torch.long)

        # Gather affinities for selected experts, normalize, and scale
        topk_weights = expert_affinities.gather(1, topk_idx)  # (T, top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Scatter back to dense (T, E) layout for ExpertMLPsV2
        expert_affinities_scaled = torch.zeros_like(expert_affinities)
        expert_affinities_scaled.scatter_(1, topk_idx, topk_weights)

        return router_logits, expert_affinities_scaled, topk_idx


class DeepseekV3InferenceConfig(InferenceConfig):
    """
    Inference configuration for DeepSeek V3.

    Handles MLA attention parameters, MoE routing config, dense/MoE layer
    distinction, and KV cache shape overrides for MLA's compressed cache format.

    DeepSeek V3 may use plain RoPE (rope_scaling=None in HF config) or YaRN
    for context extension. Since the attention class unconditionally reads
    rope_scaling fields, we inject a no-op YaRN config when rope_scaling is None.
    """

    _NOOP_YARN_ROPE_SCALING = {
        "type": "yarn",
        "factor": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 4096,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Standard HF config attributes expected by model_base.py
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "return_dict"):
            self.return_dict = True

        # Inject no-op Yarn config if rope_scaling is not set
        if not hasattr(self, "rope_scaling") or self.rope_scaling is None:
            self.rope_scaling = self._NOOP_YARN_ROPE_SCALING

        # Map HF config names to NXDI MoE names
        self.num_local_experts = getattr(self, "n_routed_experts", getattr(self, "num_experts", 0))
        self.n_shared_experts = getattr(self, "n_shared_experts", 0)
        self.num_experts_per_tok = getattr(self, "num_experts_per_tok", 0)

        # Store dense layer intermediate size before overriding with MoE size.
        # HF config uses "intermediate_size" for the dense FFN (18432).
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
            self.first_k_dense_replace = 3

        # MoE routing config (only when MoENeuronConfig is used)
        if hasattr(self.neuron_config, "router_config"):
            self.neuron_config.router_config.dtype = torch.float32
            self.neuron_config.router_config.act_fn = "sigmoid"
            # Normalization + scaling is handled by DeepseekV3Router, not ExpertMLPsV2
            self.neuron_config.normalize_top_k_affinities = False

        # Disable numeric CC token (workaround for all-gather/reduce-scatter)
        self.neuron_config.disable_numeric_cc_token = True

        # MLA KV cache: override head_dim and num_key_value_heads so the
        # KVCacheManager allocates (bsz, 1, max_len, rope_dim + kv_lora_rank)
        # instead of standard GQA layout.
        self.head_dim = self.qk_rope_head_dim + self.kv_lora_rank
        self.num_key_value_heads = 1

    def add_derived_config(self):
        self.num_cores_per_group = 1

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return DeepseekV3NeuronConfig

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


def custom_compiler_args():
    """
    Compiler flags for DeepSeek V3 on Neuron (standalone function for attention tests).
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
    return compiler_args


class DeepseekV3DenseMLP(nn.Module):
    """
    Dense MLP for DeepSeek V3 layers 0 through first_k_dense_replace-1.

    Uses SiLU-gated architecture: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses dense_intermediate_size (18432) instead of moe_intermediate_size (2048).
    """

    def __init__(self, config: DeepseekV3InferenceConfig):
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


class DeepseekV3Attention(nn.Module):
    """
    Multi-head Latent Attention (MLA) for DeepSeek V3.

    MLA is architecturally different from GQA, so this inherits directly from
    nn.Module instead of NeuronAttentionBase. All projections (q/kv compressed,
    kv_b, output) and the forward pass are self-contained here.
    """

    def __init__(self, config: DeepseekV3InferenceConfig, layer_idx: Optional[int] = None, tensor_model_parallel_group=None):

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
                self.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            except Exception:
                self.tensor_model_parallel_group = None
        self.rank_util = SPMDRank(world_size=self.tp_degree)

        # Data types
        self.torch_dtype = getattr(config.neuron_config, "attention_dtype", None) or config.neuron_config.torch_dtype
        self.rpl_reduce_dtype = getattr(config.neuron_config, "rpl_reduce_dtype", None)

        # Sequence parallelism
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

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
        self.bias = getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        assert layer_idx is not None, "Please make sure to provide a `layer_idx` when creating this class."

        self.attention_dropout = config.attention_dropout
        self.num_total_heads = config.num_attention_heads
        assert self.num_attention_heads % self.tp_degree == 0, "Number of attention heads must be a multiple of tp degree."
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

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
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
            combined = past_key_value[0].squeeze(1)  # (bsz, seq_len, rope_dim + kv_lora_rank)
            past_key_value = combined

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

        # Concatenate k_pe and compressed_kv into combined format for KVCacheManager.
        # KVCacheManager expects (key, value) tuple each shaped (bsz, 1, seq_len, head_dim).
        # For MLA, we store [k_pe | compressed_kv] in both slots (V is duplicate).
        combined = torch.cat([k_pe.squeeze(1), compressed_kv], dim=-1).unsqueeze(1)
        past_key_value = (combined, combined)

        return attn_output, past_key_value, cos_cache, sin_cache

class NeuronDeepseekV3DecoderLayer(nn.Module):
    """
    DeepSeek V3 decoder layer with MLA attention and Dense MLP or MoE.

    Layers 0 through first_k_dense_replace-1 use a dense MLP;
    remaining layers use Mixture-of-Experts (MoE).
    """

    def __init__(self, config: DeepseekV3InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx < getattr(config, "first_k_dense_replace", 3)

        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)
        self.moe_fused_nki_kernel_enabled = getattr(config.neuron_config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if self.is_dense_layer:
            self.mlp = DeepseekV3DenseMLP(config)
        elif self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        # Swap in DeepseekV3Router (GroupLimitedRouter + routed_scaling_factor)
        if not self.is_dense_layer:
            self.mlp.router = DeepseekV3Router(
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 2.5),
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                n_group=getattr(config, "n_group", 8),
                topk_group=getattr(config, "topk_group", 4),
                dtype=config.neuron_config.router_config.dtype,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=1,
            )

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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.FloatTensor`, *optional*):
                position ids of size `(batch_size, sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        qkv_fused_rmsnorm = None
        # We wrap input_layernorm/self_attn/post_attention_layernorm with module markers start/end
        # as a hint for compiler's modular-flow to avoid layer boundries in-between decoder layer components
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

        # MLP (Dense for first_k_dense_replace layers, MoE for rest)
        residual = hidden_states
        if self.is_dense_layer:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            if not self.moe_fused_nki_kernel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            is_speculative_decoding = self.config.neuron_config.enable_fused_speculation and (not self.config.neuron_config.is_prefill_stage)
            hidden_states = self.mlp(hidden_states, padding_mask, is_speculative_decoding=is_speculative_decoding)[0]
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronDeepseekV3Model(NeuronBaseModel):
    """
    NeuronDeepseekV3Model extends the DeepseekV3Model to be traceable.
    The forward function of this class is traced.
    """

    def setup_attr_for_model(self, config: DeepseekV3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: DeepseekV3InferenceConfig):
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
                NeuronDeepseekV3DecoderLayer(config, layer_idx)
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


class NeuronDeepseekV3ForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as DeepseekV3ForCausalLM
    """

    _model_cls = NeuronDeepseekV3Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return DeepseekV3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: DeepseekV3InferenceConfig) -> dict:
        return convert_deepseek_v3_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        """Return compiler args with --enable-mixed-precision-accumulation for FP32 matmul
        accumulation, matching Mixtral/DBRX/Qwen3 MoE/Qwen2 patterns."""
        args = custom_compiler_args()
        args += f" --lnc={self.config.neuron_config.logical_nc_config}"
        return args
