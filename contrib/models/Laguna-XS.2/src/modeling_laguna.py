# coding=utf-8
# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
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

"""
NeuronX Distributed Inference implementation of poolside/Laguna-XS.2.

Laguna-XS.2 is a 33B total / 3B active MoE model for agentic coding:
- 40 layers, 256 routed experts + 1 shared expert, top-8 routing
- Mixed attention: 10 full-attention layers (48 heads) + 30 SWA layers (64 heads)
- Softplus attention gating per head
- Dual RoPE: YaRN (full_attention) + default (sliding_attention)
- Sigmoid routing with e_score_correction_bias and L1 normalization

Novel architecture features implemented:
  - Per-layer variable attention heads (48 for full-attn, 64 for SWA)
  - Mixed SWA/full attention dispatch with per-layer mask selection
  - Softplus attention gating (F.softplus on g_proj, applied per-head)
  - Dual RoPE (YaRN for full_attention, default for sliding_attention)
  - Mixed MLP types (dense layer 0, MoE layers 1-39)
  - MoE with 256 experts, sigmoid routing, L1-normalized scores
  - MoE output scaling (routed_output * 2.5 + shared_output)
"""

import copy
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.modules.moe.routing import RouterTopK

logger = logging.getLogger(__name__)


# ====================================================================================
# Normalization
# ====================================================================================


def get_rmsnorm_cls():
    """Return appropriate RMSNorm for current execution context."""
    if cpu_mode():

        class StandardRMSNorm(nn.Module):
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
                return (self.weight * hidden_states).to(input_dtype)

        return StandardRMSNorm
    else:
        return CustomRMSNorm


# ====================================================================================
# Configuration
# ====================================================================================


class LagunaInferenceConfig(InferenceConfig):
    """Configuration for Laguna-XS.2 inference on NeuronX.

    Reads from HuggingFace config.json. Key fields:
    - num_attention_heads_per_layer: [48, 64, 64, 64, 48, ...] (40 entries)
    - layer_types: [full_attention, sliding_attention, ...] (40 entries)
    - mlp_layer_types: [dense, sparse, ...] (40 entries)
    - rope_parameters: {full_attention: {...}, sliding_attention: {...}}
    - num_experts: 256, num_experts_per_tok: 8
    - moe_intermediate_size: 512, shared_expert_intermediate_size: 512
    - moe_routed_scaling_factor: 2.5
    - gating: true (softplus attention gating)
    """

    def __init__(self, neuron_config=None, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 100352)
        self.hidden_size = kwargs.pop("hidden_size", 2048)
        self.intermediate_size = kwargs.pop("intermediate_size", 8192)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 40)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 48)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 8)
        self.head_dim = kwargs.pop("head_dim", 128)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 131072)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.hidden_act = (
            kwargs.pop("hidden_act", "silu") or "silu"
        )  # HF config has None
        self.sliding_window = kwargs.pop("sliding_window", 512)

        # Per-layer attention heads
        self.num_attention_heads_per_layer = kwargs.pop(
            "num_attention_heads_per_layer",
            [self.num_attention_heads] * self.num_hidden_layers,
        )

        # Layer types
        self.layer_types = kwargs.pop("layer_types", None)
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        self.mlp_layer_types = kwargs.pop("mlp_layer_types", None)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * self.num_hidden_layers

        # MoE config
        self.num_experts = kwargs.pop("num_experts", 256)
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", 8)
        self.moe_intermediate_size = kwargs.pop("moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = kwargs.pop(
            "shared_expert_intermediate_size", 512
        )
        self.moe_routed_scaling_factor = kwargs.pop("moe_routed_scaling_factor", 2.5)
        self.moe_apply_router_weight_on_input = kwargs.pop(
            "moe_apply_router_weight_on_input", False
        )
        self.router_aux_loss_coef = kwargs.pop("router_aux_loss_coef", 0.0)

        # Aliases required by initialize_moe_module()
        self.num_local_experts = self.num_experts
        self.n_shared_experts = 1  # Laguna always has 1 shared expert

        # RoPE
        self.rope_parameters = kwargs.pop("rope_parameters", None)
        self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.5)

        # Attention gating
        self.gating = kwargs.pop("gating", True)

        # Standard attributes
        self.pad_token_id = kwargs.pop("pad_token_id", 9)
        self.bos_token_id = kwargs.pop("bos_token_id", 2)
        self.eos_token_id = kwargs.pop("eos_token_id", [2, 24])
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)

        # Pop HF-specific keys not used by our config
        for hf_key in [
            "auto_map",
            "architectures",
            "model_type",
            "transformers_version",
            "dtype",
            "torch_dtype",
            "use_cache",
        ]:
            kwargs.pop(hf_key, None)

        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        self.num_cores_per_group = 1

        # MoE process group config (read by initialize_moe_process_group)
        tp = (
            self.neuron_config.tp_degree
            if hasattr(self, "neuron_config") and self.neuron_config
            else 4
        )
        self.moe_cte_ep_degree = 1
        self.moe_cte_tp_degree = tp
        self.moe_tkg_ep_degree = 1
        self.moe_tkg_tp_degree = tp

        if hasattr(self, "neuron_config") and self.neuron_config is not None:
            nc = self.neuron_config

            # Laguna uses sigmoid routing (not softmax) with L1 normalization
            if hasattr(nc, "router_config") and nc.router_config is not None:
                nc.router_config.act_fn = "sigmoid"

            # GLU MLP for MoE experts (SiLU gating)
            # MoENeuronConfig sets glu_mlp=True by default, but ensure glu_type
            nc.glu_type = "glu"

            # MoE TP degree should match overall TP
            nc.moe_tp_degree = tp

            # Use torch blockwise matmul (NKI shard-hidden kernel not available in SDK 2.29)
            if hasattr(nc, "blockwise_matmul_config"):
                nc.blockwise_matmul_config.use_torch_block_wise = True

            # Enable MoE fused TKG NKI kernel for token generation speedup.
            # Constraints: intermediate/TP=128 (%128==0), GLU+SiLU supported,
            # sigmoid routing natively supported in SDK 2.29.
            nc.moe_fused_nki_kernel_enabled = True

            # CTE attention NKI kernel (flash attention).
            # The CTE kernel only handles softmax(QK^T)V — the softplus gating is
            # applied AFTER it returns in NeuronLagunaAttention.forward().
            nc.attn_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LagunaInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


def get_updated_configs(config: LagunaInferenceConfig):
    """Generate per-layer configs for heterogeneous attention.

    Per-layer variables:
    - _layer_num_attention_heads: 48 (full_attention) or 64 (sliding_attention)
    - _layer_is_sliding: True for SWA layers, False for full attention layers
    - _layer_rope_theta: 500000 (full_attention/YaRN) or 10000 (SWA/default)
    - _layer_partial_rotary_factor: 0.5 (full_attention) or 1.0 (SWA)
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)

        # 008a: Per-layer variable attention heads
        layer_config._layer_num_attention_heads = config.num_attention_heads_per_layer[
            i
        ]

        # 008b: Mixed SWA/full attention dispatch
        layer_type = config.layer_types[i]
        layer_config._layer_is_sliding = layer_type == "sliding_attention"

        # 008d: Dual RoPE per layer type
        if layer_type == "sliding_attention":
            rope_params = {}
            if config.rope_parameters:
                rope_params = config.rope_parameters.get("sliding_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 10000.0)
            layer_config._layer_partial_rotary_factor = 1.0
        else:
            # full_attention: YaRN with partial rotation
            rope_params = {}
            if config.rope_parameters:
                rope_params = config.rope_parameters.get("full_attention", {})
            layer_config._layer_rope_theta = rope_params.get("rope_theta", 500000.0)
            layer_config._layer_partial_rotary_factor = rope_params.get(
                "partial_rotary_factor", config.partial_rotary_factor
            )

        updated_configs.append(layer_config)

    return updated_configs


# ====================================================================================
# Attention
# ====================================================================================


class NeuronLagunaAttention(NeuronAttentionBase):
    """Laguna attention with QK norms, per-layer variable heads, and dual RoPE.

    Features:
    - Per-layer variable Q heads (48 for full_attn, 64 for SWA)
    - QK norms (RMSNorm on head_dim)
    - Dual RoPE: YaRN (full_attn, theta=500K, partial_rotary=0.5)
                 Default (SWA, theta=10K, partial_rotary=1.0)
    - Softplus gating (008c): g_proj + F.softplus on pre-attn hidden_states
    """

    def __init__(self, config: LagunaInferenceConfig):
        num_heads = config._layer_num_attention_heads
        rope_theta = config._layer_rope_theta
        partial_rotary_factor = config._layer_partial_rotary_factor

        # 008d: Compute rotary dimension from partial_rotary_factor
        rotary_dim = int(config.head_dim * partial_rotary_factor)
        rotary_dim = rotary_dim - (rotary_dim % 2)  # Ensure even

        rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # Store for partial rotary in apply_rotary_embedding
        self._rotary_dim = rotary_dim
        self._head_dim = config.head_dim

        # QK norms
        rmsnorm_cls = get_rmsnorm_cls()
        q_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)
        k_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)

        # Pass sliding_window=None for ALL layers (Gemma4 Discovery #27).
        # SWA behavior is enforced via local_mask at the decoder layer level.
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_norm,
            k_layernorm=k_norm,
            sliding_window=None,
        )

        # 008c: Softplus attention gating.
        # g_proj produces one gate scalar per head (not per dim like Trinity).
        # ColumnParallelLinear shards output across TP ranks.
        tp_degree = config.neuron_config.tp_degree
        heads_per_rank = math.ceil(num_heads / tp_degree)
        padded_total_heads = heads_per_rank * tp_degree
        # Output size matches num_heads (padded for TP), each element gates one head
        self.attn_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            padded_total_heads,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

    def _apply_gated_o_proj(self, attn_output, gate_hidden_states, adapter_ids=None):
        """Apply softplus per-head gating then o_proj.

        Laguna gating: F.softplus(g_proj(input)) per head, applied before o_proj.
        Unlike Trinity (sigmoid, per-dim), Laguna's gate is per-HEAD (one scalar per head).
        """
        # gate_values: [B, S, num_heads_per_rank]
        gate_values = F.softplus(self.attn_gate_proj(gate_hidden_states).float())
        gate_values = gate_values.to(attn_output.dtype)

        # Expand per-head gate to per-dim: [B, S, num_heads_per_rank * head_dim]
        bsz, q_len, _ = attn_output.shape
        heads_per_rank = gate_values.shape[-1]
        # Reshape gate: [B, S, H] -> [B, S, H, 1] -> expand to [B, S, H, D] -> flatten
        gate_values = gate_values.unsqueeze(-1).expand(
            bsz, q_len, heads_per_rank, self._head_dim
        )
        gate_values = gate_values.reshape(bsz, q_len, heads_per_rank * self._head_dim)

        attn_output = attn_output * gate_values
        return self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

    def standard_causal_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        rotary_position_ids=None,
        kv_mgr=None,
        get_kv_per_layer=False,
        update_kv_per_layer=False,
        residual=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        """Override base class to insert softplus gating before o_proj.

        Based on Trinity's override pattern. The only change from base class
        is replacing `self.get_o_proj()(attn_output)` with
        `self._apply_gated_o_proj(attn_output, gate_hidden_states)`.
        """
        from neuronx_distributed_inference.modules.attention.attention_base import (
            NeuronAttentionBaseOutput,
        )

        use_polar_compatible_rope = kwargs.get("use_polar_compatible_rope", False)

        # Save original hidden_states for gate computation BEFORE dtype conversion
        gate_hidden_states = hidden_states

        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch_dtype)
        seq_ids = kwargs.get("seq_ids")
        is_context_parallel = past_key_value is None and self.cp_degree > 1
        is_data_parallel = past_key_value is not None and self.dp_degree > 1

        if is_context_parallel:
            attention_mask, hidden_states, position_ids, cos_cache, sin_cache = (
                self._split_inputs_for_context_parallel(
                    attention_mask, hidden_states, position_ids, cos_cache, sin_cache
                )
            )

        if is_data_parallel:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                get_dp_rank,
                split_along_dim,
                get_data_parallel_attention_dp_group,
                gather_from_tensor_model_parallel_region_with_dim,
            )

            dp_rank = get_dp_rank(
                self.rank_util.get_rank(),
                self.tp_degree,
                self.dp_degree,
                self.neuron_config.switch_cc,
            )
            hidden_states = split_along_dim(
                hidden_states, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            attention_mask = split_along_dim(
                attention_mask, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )
            position_ids = split_along_dim(
                position_ids, dim=0, rank=dp_rank, num_partitions=self.dp_degree
            )

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None
        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False
        if self.neuron_config.is_prefix_caching:
            is_token_gen = is_token_gen and q_len < 128

        # NKI kernel paths -- delegate to base (gating not fused in NKI kernels)
        if self.attn_block_tkg_nki_kernel_enabled and is_token_gen:
            return super().standard_causal_attention_forward(
                gate_hidden_states.to(self.torch_dtype)
                if is_context_parallel or is_data_parallel
                else gate_hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

        if (
            getattr(self.neuron_config, "attn_block_cte_nki_kernel_enabled", False)
            and not is_token_gen
            and not self.neuron_config.is_prefix_caching
        ):
            return super().standard_causal_attention_forward(
                gate_hidden_states.to(self.torch_dtype)
                if is_context_parallel or is_data_parallel
                else gate_hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                active_mask,
                adapter_ids,
                cos_cache,
                sin_cache,
                rmsnorm,
                rotary_position_ids,
                kv_mgr,
                get_kv_per_layer,
                update_kv_per_layer,
                residual,
                windowed_context_encoding_window_idx,
                **kwargs,
            )

        tkg_attn_kernel_fused_rope = is_token_gen and getattr(
            self.neuron_config, "attn_tkg_builtin_kernel_enabled", False
        )

        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=tkg_attn_kernel_fused_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

        if is_token_gen:
            if tkg_attn_kernel_fused_rope:
                attn_output, K = self.attention_tokengen_kernel_builtin(
                    Q,
                    K,
                    V,
                    position_ids,
                    past_key_value,
                    attention_mask,
                    active_mask,
                    rotary_position_ids,
                )
            else:
                # Pad attention_mask to match KV cache sequence length.
                # In vLLM continuous-batching mode, the mask may be bucket-sized
                # (e.g., 128) while the KV cache is max_length-sized (e.g., 4096).
                # The TKG NKI kernel handles this internally, but since we use
                # compute_for_token_gen (softplus gating prevents NKI kernel use),
                # we must pad manually. Padding with False masks out positions
                # beyond the current context length.
                kv_seq_len = past_key_value[1].shape[2]  # V cache: (B, H, S, D)
                if attention_mask.shape[-1] < kv_seq_len:
                    pad_len = kv_seq_len - attention_mask.shape[-1]
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=False)
                attn_output = self.attention_tokengen(
                    Q,
                    K,
                    V,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    active_mask,
                    **kwargs,
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, K, V = self.attention_context_encode(
                Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask
            )

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # *** GATED ATTENTION: apply softplus gate BEFORE o_proj ***
        attn_output = self._apply_gated_o_proj(
            attn_output, gate_hidden_states, adapter_ids=adapter_ids
        )

        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)

        kv = (K, V)
        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        if is_context_parallel and not self.sequence_parallel_enabled:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                gather_from_tensor_model_parallel_region_with_dim,
                get_context_parallel_attention_cp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=1,
                process_group=get_context_parallel_attention_cp_group(),
            )

        if is_data_parallel:
            from neuronx_distributed_inference.modules.attention.attention_base import (
                gather_from_tensor_model_parallel_region_with_dim,
                get_data_parallel_attention_dp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=0,
                process_group=get_data_parallel_attention_dp_group(),
            )

        attn_output = attn_output.to(original_dtype)
        return NeuronAttentionBaseOutput(
            attn_output, kv, cos_cache, sin_cache, residual
        )

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        """Apply rotary embedding with support for partial rotation.

        Full rotation (SWA, partial_rotary_factor=1.0): standard path.
        Partial rotation (full_attn, partial_rotary_factor=0.5):
          split Q/K at rotary_dim, rotate first part, concat.
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            if self._rotary_dim == self._head_dim:
                # Full rotation (SWA layers)
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
            else:
                # Partial rotation (full_attention layers)
                # Q, K are [batch, num_heads, seq, head_dim]
                q_rot = Q[..., : self._rotary_dim]
                q_pass = Q[..., self._rotary_dim :]
                k_rot = K[..., : self._rotary_dim]
                k_pass = K[..., self._rotary_dim :]

                q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_cache, sin_cache)

                Q = torch.cat([q_rot, q_pass], dim=-1)
                K = torch.cat([k_rot, k_pass], dim=-1)

        return Q, K, cos_cache, sin_cache


# ====================================================================================
# MLP
# ====================================================================================


class NeuronLagunaMLP(nn.Module):
    """Laguna dense MLP with SiLU gating.

    Used for layer 0 (dense). Layers 1-39 use MoE via NeuronLagunaMoE.
    """

    def __init__(self, config: LagunaInferenceConfig, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ====================================================================================
# MoE (008e + 008f)
# ====================================================================================


class RouterTopKWithBias(RouterTopK):
    """RouterTopK with expert_bias for Laguna's sigmoid routing.

    Laguna routing: sigmoid(logits) + expert_bias for top-k selection,
    then L1-normalized sigmoid scores (no bias) as routing weights.
    Based on Trinity's RouterTopKWithBias.
    """

    def __init__(self, expert_bias_size, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "expert_bias",
            torch.zeros(expert_bias_size, dtype=torch.float32),
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        # Top-k selection with expert_bias added to scores
        scores_for_selection = expert_affinities.float() + self.expert_bias.float()
        _, expert_index = torch.topk(scores_for_selection, self.top_k)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def initialize_laguna_moe(config: "LagunaInferenceConfig", rmsnorm=None):
    """Initialize MoE module for Laguna with sigmoid routing and expert bias.

    Creates an MoE module via NxDI's initialize_moe_module, then replaces the
    default RouterTopK with RouterTopKWithBias for expert_bias support.

    Args:
        config: LagunaInferenceConfig with MoE fields set for the layer.
            Must have intermediate_size=moe_intermediate_size (512) before calling.
        rmsnorm: Optional RMSNorm for fused TKG path.
    """
    try:
        moe = initialize_moe_module(
            config=config, init_tkg_module=True, rmsnorm=rmsnorm
        )
    except (TypeError, Exception) as e:
        logger.warning("Fused MoE TKG init failed: %s. Falling back.", e)
        moe = initialize_moe_module(config=config)

    # Replace router with bias-aware version (Trinity pattern)
    old_router = moe.router
    new_router = RouterTopKWithBias(
        expert_bias_size=config.num_local_experts,
        num_experts=old_router.num_experts,
        top_k=old_router.top_k,
        hidden_size=old_router.hidden_size,
        dtype=old_router.dtype,
        device=old_router.device,
        act_fn=old_router.act_fn,
        sequence_parallel_enabled=old_router.sequence_parallel_enabled,
        sequence_dimension=old_router.sequence_dimension,
        bias=old_router.bias,
        apply_act_fn_over_topk=old_router.apply_act_fn_over_topk,
        store_transposed_weights=old_router.store_transposed_weights,
    )
    new_router.linear_router = old_router.linear_router
    if hasattr(old_router, "weight_T"):
        new_router.weight_T = old_router.weight_T
    moe.router = new_router
    moe.eval()
    return moe


# ====================================================================================
# Decoder Layer
# ====================================================================================


class NeuronLagunaDecoderLayer(nn.Module):
    """Laguna decoder layer with pre/post norms for attention and MLP.

    Laguna uses 2 norms per block (input_layernorm + post_attention_layernorm).
    Structure: LN -> Attn -> residual -> LN -> MLP -> residual

    Layer 0: Dense MLP (intermediate=8192)
    Layers 1-39: MoE (256 experts, intermediate=512, + shared expert with 2.5x scaling)
    """

    def __init__(self, config: LagunaInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_layer = (
            hasattr(config, "mlp_layer_types")
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        # 008b: Mixed SWA/full attention dispatch
        self.is_sliding_window_attention = getattr(config, "_layer_is_sliding", False)
        # 008g: MoE routed output scaling factor
        self.moe_routed_scaling_factor = getattr(
            config, "moe_routed_scaling_factor", 2.5
        )

        self.self_attn = NeuronLagunaAttention(config)

        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.is_moe_layer:
            # MoE layers (1-39): Create config copy with intermediate_size=512
            moe_config = copy.deepcopy(config)
            moe_config.intermediate_size = config.moe_intermediate_size  # 512
            # Disable internal shared expert in MoE module — we handle shared expert
            # separately so we can apply 2.5x scaling to routed output in 008g:
            # result = routed_output * 2.5 + shared_output
            moe_config.n_shared_experts = 0
            self.mlp = initialize_laguna_moe(moe_config)
            # For fused TKG: provide a separate RMSNorm for the kernel's internal
            # normalization. We pass rmsnorm=None to MoE init so CTE doesn't
            # double-apply the norm (following Trinity pattern).
            # Store as a plain attribute (not nn.Module) to avoid duplicating
            # the module in state_dict. The actual module lives under self.mlp.
            self._has_moe_fused_tkg = (
                getattr(self.mlp, "moe_fused_tkg", None) is not None
            )
            if self._has_moe_fused_tkg:
                moe_rmsnorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
                self.mlp.moe_fused_tkg.post_attention_layernorm = moe_rmsnorm
            # Shared expert (standalone dense MLP with intermediate=512)
            self.shared_expert = NeuronLagunaMLP(
                config, intermediate_size=config.shared_expert_intermediate_size
            )
        else:
            # Dense layer (layer 0): standard SwiGLU MLP with intermediate=8192
            self.mlp = NeuronLagunaMLP(config)
            self._has_moe_fused_tkg = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # 008d: Force recomputation of cos/sin per layer (heterogeneous RoPE).
        kwargs.pop("cos_cache", None)
        kwargs.pop("sin_cache", None)

        # 008b: Select mask — SWA layers use local_mask, full layers use attention_mask.
        # local_mask from _create_windowed_attn_mask_tkg is (B,1,1,window_size).
        # During TKG, prior_scores is (B,H,1,cache_len) where cache_len=max_length
        # (uniform cache for all layers). Pad local_mask to cache_len with False so
        # positions beyond the sliding window are masked out.
        local_mask = kwargs.pop("local_mask", None)
        if self.is_sliding_window_attention and local_mask is not None:
            mask = local_mask
            if attention_mask is not None and mask.shape[-1] < attention_mask.shape[-1]:
                pad_len = attention_mask.shape[-1] - mask.shape[-1]
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)
        else:
            mask = attention_mask

        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 008c: Softplus gating is applied inside NeuronLagunaAttention.standard_causal_attention_forward
        # (gate_hidden_states = hidden_states before dtype conversion, gating before o_proj)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection after attention (matches HF: residual + attn_output)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states

        # Normalization strategy for fused MoE TKG (Trinity pattern):
        # - CTE (seq_len > 1): Decoder applies post_attention_layernorm.
        #   MoE's _forward_compute_bound skips norm (rmsnorm=None at init).
        # - TKG (seq_len == 1): Decoder skips post_attention_layernorm.
        #   MoEFusedTKG applies norm internally using its own RMSNorm.
        # For dense layers (layer 0) or when fused TKG is not enabled,
        # decoder always applies norm.
        is_tkg = self._has_moe_fused_tkg and hidden_states.shape[1] == 1
        if not is_tkg:
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            # MoE: routed experts + separate shared expert.
            # 008g: Laguna formula: result = routed_output * 2.5 + shared_output
            routed_output = self.mlp(hidden_states)[0]  # MoE returns (output, ...)
            # In TKG mode, hidden_states is un-normed (fused kernel handles norm
            # internally). Shared expert needs normed input.
            shared_input = (
                self.post_attention_layernorm(hidden_states)
                if is_tkg
                else hidden_states
            )
            shared_output = self.shared_expert(shared_input)
            hidden_states = (
                routed_output * self.moe_routed_scaling_factor + shared_output
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# Model
# ====================================================================================


class NeuronLagunaModel(NeuronBaseModel):
    """Laguna text model: embeddings + decoder layers + final norm + lm_head."""

    def setup_attr_for_model(self, config: LagunaInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: LagunaInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # Per-layer configs
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronLagunaDecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )

        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        # 008b: Mixed SWA/full attention — set flags for base model dual-mask flow.
        # has_mixed_attn=True tells base model to create both global and local masks.
        # sliding_window enables local windowed mask creation.
        self.has_mixed_attn = True
        self.sliding_window = config.sliding_window

        # Per-layer cache size mapping: all layers get the same cache sequence length
        # (max of sliding_window and max_length) because the KV cache sequence
        # dimension must match the attention mask dimension. SWA behavior is enforced
        # purely through the local_mask, not through smaller cache allocations.
        max_length = config.neuron_config.max_length
        sw = config.sliding_window or max_length
        uniform_cache_len = max(sw, max_length)
        self.layer_to_cache_size_mapping = [
            uniform_cache_len
        ] * config.num_hidden_layers

    def init_inference_optimization(self, config: LagunaInferenceConfig):
        """Initialize KV cache and optional on-device sampling."""
        if self.on_device_sampling:
            try:
                from neuronx_distributed_inference.modules.generation.sampling import (
                    Sampler,
                )
            except ImportError:
                from neuronx_distributed_inference.modules.sampling.utils import (
                    create_sampler as Sampler,
                )
            self.sampler = Sampler(config.neuron_config)

        from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
            KVCacheManager,
        )

        # 008b: Use layer_to_cache_size_mapping for mixed attention.
        # Laguna KV heads (8) and head_dim (128) are constant across all layers,
        # so standard KVCacheManager suffices (no custom per-layer shapes needed).
        self.kv_mgr = KVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
            layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
        )


# ====================================================================================
# Top-level Model Class
# ====================================================================================


class NeuronLagunaForCausalLM(NeuronBaseForCausalLM):
    """Laguna causal LM for NeuronX inference.

    Handles weight loading, state dict conversion, and model initialization.
    """

    _model_cls = NeuronLagunaModel

    def __init__(self, *args, **kwargs):
        # SDK 2.29: sigmoid routing is natively supported by the fused MoE TKG
        # NKI kernel (RouterActFnType.SIGMOID). No monkey-patching needed.
        super().__init__(*args, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return LagunaInferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model for weight extraction.

        Note: trust_remote_code=True fails on transformers 4.57 due to
        missing imports (RopeParameters, initialization). We load via
        safetensors directly in convert_hf_to_neuron_state_dict instead.
        """
        # Try native transformers first
        try:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        except (ImportError, Exception) as e:
            logger.warning("HF AutoModel loading failed: %s. Using safetensors.", e)
            return None

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: LagunaInferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """Convert HuggingFace Laguna state dict to NeuronX format.

        Key transformations:
        1. Strip 'model.' prefix from HF keys
        2. Remap q_norm/k_norm -> q_layernorm/k_layernorm
        3. Stack expert weights [256, H, 2*I] for gate_up, [256, I, H] for down
        4. Remap router and expert_bias keys
        5. Map shared expert weights
        6. Add rank_util tensors for TP
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}
        target_dtype = torch.bfloat16

        # Detect whether keys still have 'model.' prefix.
        # The framework's application_base.get_state_dict() strips 'model.' BEFORE
        # calling convert_hf_to_neuron_state_dict, so normally keys arrive without it.
        # But handle both cases for robustness (matches Trinity pattern).
        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

        def strip_prefix(key):
            if has_model_prefix and key.startswith("model."):
                return key[6:]
            return key

        def hf_key(layer_idx, suffix):
            """Build HF state_dict key for a layer, respecting prefix state."""
            if has_model_prefix:
                return f"model.layers.{layer_idx}.{suffix}"
            return f"layers.{layer_idx}.{suffix}"

        # Identify MoE layers
        moe_layers = set()
        if hasattr(config, "mlp_layer_types"):
            for i, t in enumerate(config.mlp_layer_types):
                if t == "sparse":
                    moe_layers.add(i)

        # --- Pass 1: Map non-MoE keys ---
        for key, weight in state_dict.items():
            new_key = strip_prefix(key)

            # Skip all MoE-related keys (handled in Pass 2)
            if any(
                x in new_key
                for x in ["mlp.experts.", "mlp.gate.", "mlp.shared_expert."]
            ):
                continue

            # Skip dense MLP keys for MoE layers (they don't exist in HF)
            skip = False
            for i in moe_layers:
                prefix = f"layers.{i}.mlp."
                if new_key.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            # Remap QK norm keys
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            # 008c: Remap g_proj -> attn_gate_proj (attention gating)
            new_key = new_key.replace(
                ".self_attn.g_proj.", ".self_attn.attn_gate_proj."
            )

            new_state_dict[new_key] = weight.detach().clone().to(target_dtype)

        # --- Pass 2: Stack MoE expert weights per layer ---
        num_experts = config.num_experts  # 256
        hidden_size = config.hidden_size  # 2048
        moe_intermediate = config.moe_intermediate_size  # 512

        for layer_idx in moe_layers:
            neuron_prefix = f"layers.{layer_idx}"

            # Router: mlp.gate.weight -> mlp.router.linear_router.weight
            router_key = hf_key(layer_idx, "mlp.gate.weight")
            if router_key in state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.linear_router.weight"] = (
                    state_dict[router_key].to(target_dtype)
                )

            # Expert bias: mlp.experts.e_score_correction_bias -> mlp.router.expert_bias
            bias_key = hf_key(layer_idx, "mlp.experts.e_score_correction_bias")
            if bias_key in state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.expert_bias"] = state_dict[
                    bias_key
                ].to(torch.float32)

            # Stack expert weights: gate+up -> [num_experts, H, 2*I], down -> [num_experts, I, H]
            gate_up_proj = torch.empty(
                num_experts, hidden_size, 2 * moe_intermediate, dtype=target_dtype
            )
            down_proj = torch.empty(
                num_experts, moe_intermediate, hidden_size, dtype=target_dtype
            )

            all_found = True
            for e in range(num_experts):
                gate_k = hf_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")
                up_k = hf_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")
                down_k = hf_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")

                if gate_k in state_dict and up_k in state_dict and down_k in state_dict:
                    gate_w = state_dict[gate_k].to(target_dtype)  # [I, H]
                    up_w = state_dict[up_k].to(target_dtype)  # [I, H]
                    down_w = state_dict[down_k].to(target_dtype)  # [H, I]

                    # Concat gate+up -> [2*I, H], transpose -> [H, 2*I]
                    gate_up_proj[e] = torch.cat([gate_w, up_w], dim=0).T
                    # Transpose down -> [I, H]
                    down_proj[e] = down_w.T
                else:
                    all_found = False
                    logger.warning(
                        "Missing expert weights for layer %d expert %d", layer_idx, e
                    )
                    break

            if all_found:
                new_state_dict[
                    f"{neuron_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                ] = gate_up_proj
                new_state_dict[
                    f"{neuron_prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                ] = down_proj

            # Shared expert: mlp.shared_expert.* -> shared_expert.*
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                se_key = hf_key(layer_idx, f"mlp.shared_expert.{proj_name}.weight")
                if se_key in state_dict:
                    new_state_dict[
                        f"{neuron_prefix}.shared_expert.{proj_name}.weight"
                    ] = state_dict[se_key].to(target_dtype)

            # Fused MoE TKG aliased weights (Trinity pattern).
            # The MoEFusedTKG module has a separate RMSNorm that needs the same
            # weights as the layer's post_attention_layernorm.
            post_attn_key = f"{neuron_prefix}.post_attention_layernorm.weight"
            if post_attn_key in new_state_dict:
                new_state_dict[
                    f"{neuron_prefix}.mlp.moe_fused_tkg.post_attention_layernorm.weight"
                ] = new_state_dict[post_attn_key].clone()

            # Router transposed weight for fused TKG kernel.
            router_w_key = f"{neuron_prefix}.mlp.router.linear_router.weight"
            if router_w_key in new_state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.weight_T"] = (
                    new_state_dict[router_w_key].detach().T.clone()
                )

        # --- rank_util tensors ---
        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        for i in range(config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Laguna has tie_word_embeddings=False, so no tying needed."""
        pass

    def get_compiler_args(self):
        """Get compiler arguments for Laguna."""
        return "--model-type=transformer"
