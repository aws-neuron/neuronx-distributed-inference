# coding=utf-8
# Copyright 2025 MiniMax and the HuggingFace Team. All rights reserved.
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
MiniMax M2 model for NXD inference - v3.
Updated implementation following latest Qwen3 MoE pattern.

Key features:
1. QK norm applied BEFORE reshape (MiniMax M2 specific) - on full projection output
   - Q norm: [num_attention_heads * head_dim] = [6144]
   - K norm: [num_key_value_heads * head_dim] = [1024]
2. Partial RoPE: rotary_dim=64, head_dim=128 (only first half gets rotation)
3. Sigmoid router with e_score_correction_bias for expert selection

v3 updates (synced from Qwen3 MoE):
- fused_qkv support
- moe_fused_nki_kernel support
- maybe_pad_intermediate for shard-on-I
- ModuleMarker wrappers for compiler optimization
- Enhanced compiler args
- qkv_kernel_enabled support
"""
import gc
import warnings
import copy
import math
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from torch_neuronx.xla_impl.ops import nki_jit
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module, initialize_moe_process_group
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig, SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP, MOE_TKG_MK_INTERMEDIATE_PER_TP
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def _helper_concat_and_delete_qkv(state_dict: Dict[str, Any], layer_num: int, attr: str):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    """
    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict: Dict[str, Any], cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, l, "scale")

    gc.collect()
    return state_dict


def maybe_dequantize_layer(neuron_state_dict, config):
    """Dequantize FP8 layers if present."""
    scale_layers = []
    for layer_key in neuron_state_dict.keys():
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)

    # delete scale layers
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def get_rmsnorm_cls():
    """Get appropriate RMSNorm class based on execution environment."""
    if cpu_mode():
        class SimpleRMSNorm(nn.Module):
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
        return SimpleRMSNorm
    return CustomRMSNorm


class MiniMaxM2QKNorm(nn.Module):
    """
    Distributed QK Norm for MiniMax M2 - applied BEFORE reshape (on full projection output).

    In MiniMax M2, qk_norm is applied on the full Q/K projection output:
    - Q: [batch, seq, num_attention_heads * head_dim]
    - K: [batch, seq, num_key_value_heads * head_dim]

    This is different from Qwen3 MoE which applies per-head norm after reshape.

    For tensor parallel:
    - Each rank computes local sum(x²)
    - All-reduce to get global sum across all TP ranks
    - Compute global RMS normalization
    - Stores FULL padded weights [tp_degree * per_rank_size]
    - Dynamically slices weights in forward() based on rank using rank_util tensor

    Parameters:
    - hidden_size: Per-rank hidden size (e.g., 128 for both Q and K per rank)
    - full_hidden_size: Original full size for RMS divisor computation
      - For Q: 6144 (original 48 heads * 128 dim, NOT padded 8192)
      - For K: 8192 (accounts for replication: 8 original KV heads replicated to 64)
    - padded_hidden_size: Total weight storage size (tp_degree * per_rank_size)
    """
    def __init__(self, hidden_size, eps=1e-6, tp_degree=1, full_hidden_size=None, padded_hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size  # Per-rank hidden size
        self.variance_epsilon = eps
        self.tp_degree = tp_degree
        # Full hidden size for RMS divisor
        # - For Q: should be original 6144 (padding zeros don't contribute to sum_sq)
        # - For K: should be 8192 (replication factor cancels out: 8x sum / 8192 = sum / 1024)
        self.full_hidden_size = full_hidden_size if full_hidden_size is not None else (hidden_size * tp_degree)
        # Padded hidden size for weight storage
        self.padded_hidden_size = padded_hidden_size if padded_hidden_size is not None else (hidden_size * tp_degree)
        # Store FULL weights - will be sliced dynamically in forward() using rank_util
        self.weight = nn.Parameter(torch.ones(self.padded_hidden_size))

    def forward(self, hidden_states, rank_util=None):
        from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute local sum of squares (not mean yet - need global normalization)
        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)

        # All-reduce to get global sum of squares across all TP ranks
        if self.tp_degree > 1:
            global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)
        else:
            global_sum_sq = local_sum_sq

        # Compute global RMS: sqrt(sum(x²) / total_elements)
        global_variance = global_sum_sq / self.full_hidden_size
        hidden_states = hidden_states * torch.rsqrt(global_variance + self.variance_epsilon)

        # Dynamically slice weight based on TP rank using rank_util tensor
        # IMPORTANT: Cannot use parallel_state.get_tensor_model_parallel_rank() here
        # because it returns a Python constant at trace time, not a tensor.
        # We need to use rank_util.rank which is a tensor for proper SPMD execution.
        if rank_util is not None and self.tp_degree > 1:
            # Reshape weight to [tp_degree, hidden_size] for rank-based indexing
            weight_reshaped = self.weight.view(self.tp_degree, self.hidden_size)
            # rank_util.rank is [0, 1, 2, ..., tp_degree-1], each SPMD device gets its rank value
            # Use index_select which is XLA-compatible for proper tracing
            # NOTE: Keep as int32 for NKI kernel compatibility (nl.load doesn't support int64)
            rank_index = rank_util.rank[:1]
            local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)
        else:
            # Single rank or no rank_util - use first slice
            local_weight = self.weight[:self.hidden_size]

        return (local_weight * hidden_states).to(input_dtype)


class RouterTopKWithBias(RouterTopK):
    """
    RouterTopK with e_score_correction_bias support for MiniMax M2.

    MiniMax M2 uses sigmoid activation with a bias term added to scores for expert selection,
    but the final weights (affinities without bias) are passed to experts.
    """

    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)

        # For expert selection, add bias to affinities (MiniMax M2 specific)
        # Cast to float32 for numerical stability
        scores_for_choice = expert_affinities.float() + self.e_score_correction_bias.unsqueeze(0)

        # Select top-k experts based on biased scores
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        expert_affinities = expert_affinities.to(dtype=original_dtype)
        # NOTE: Keep as int32 for NKI kernel compatibility (nl.load doesn't support int64)
        # torch.topk returns int64, so we need to explicitly convert
        expert_index = expert_index.detach().to(dtype=torch.int32)

        return router_logits, expert_affinities, expert_index


def convert_minimax_m2_hf_to_neuron_state_dict(neuron_state_dict: Dict[str, Any], config: "MiniMaxM2InferenceConfigV3") -> Dict[str, Any]:
    """
    Convert HuggingFace MiniMax M2 state dict to Neuron-compatible format.
    Updated for v3 following Qwen3 MoE pattern.

    Optimized for memory efficiency:
    - Uses torch.no_grad() to avoid gradient tracking overhead
    - Avoids unnecessary .clone() by using direct tensor views
    - Periodic gc.collect() during expert processing
    """
    from neuronx_distributed_inference.modules.attention.gqa import get_shardable_head_counts, _maybe_pad_interleaved, GQA

    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported for MiniMax M2"

    # Dequantize layers if needed (v3: added from Qwen3 MoE)
    maybe_dequantize_layer(neuron_state_dict, config)

    # Wrap in no_grad to avoid any gradient tracking overhead
    with torch.no_grad():
        # Add rank utility tensor for TP parallel operations
        # NOTE: Must use int32 for NKI kernel compatibility (nl.load doesn't support int64)
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Calculate sharded head counts for qk_norm weight handling
        tp_degree = config.neuron_config.tp_degree
        sharding_strategy = GQA.REPLICATE_TO_TP_DEGREE
        padded_num_attention_heads, padded_num_kv_heads = get_shardable_head_counts(
            tp_degree, config.num_attention_heads, config.num_key_value_heads, sharding_strategy
        )
        head_dim = config.head_dim
        has_qk_norm = getattr(config, 'use_qk_norm', True)

        # GC interval for expert processing (every N experts)
        gc_interval = 64

        for layer_idx in range(config.num_hidden_layers):
            # Add rank_util.rank for each attention layer
            # NOTE: Must use int32 for NKI kernel compatibility
            neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
                0, config.neuron_config.tp_degree, dtype=torch.int32
            )

            # MiniMax M2 qk_norm weights - need to shard for TP parallel
            # HF model: q_norm.weight shape = [num_attention_heads * head_dim] = [6144]
            # HF model: k_norm.weight shape = [num_key_value_heads * head_dim] = [1024]
            if has_qk_norm:
                # q_norm: apply interleaved padding [48 heads -> padded heads]
                q_norm_key = f"layers.{layer_idx}.self_attn.q_norm.weight"
                if q_norm_key in neuron_state_dict:
                    q_norm_full = neuron_state_dict[q_norm_key]  # [6144]
                    # Apply the same interleaved padding as Q projection weights
                    source_group_size = config.num_attention_heads // config.num_key_value_heads
                    q_norm_padded = _maybe_pad_interleaved(
                        q_norm_full.unsqueeze(0),  # Add batch dim: [1, 6144]
                        pad_dim=1,
                        source_heads=config.num_attention_heads,
                        target_heads=padded_num_attention_heads,
                        source_group_size=source_group_size,
                    ).squeeze(0)  # [padded_num_attention_heads * head_dim]
                    neuron_state_dict[q_norm_key] = q_norm_padded
                    if layer_idx == 0:
                        print(f"  q_norm: {q_norm_full.shape} -> {q_norm_padded.shape} (interleaved padding)")

                # k_norm: replicate from original KV heads to padded KV heads
                k_norm_key = f"layers.{layer_idx}.self_attn.k_norm.weight"
                if k_norm_key in neuron_state_dict:
                    k_norm_full = neuron_state_dict[k_norm_key]  # [1024]
                    # KV heads are replicated: each of the original heads is replicated
                    k_norm_reshaped = k_norm_full.reshape(config.num_key_value_heads, head_dim)
                    repeats = padded_num_kv_heads // config.num_key_value_heads
                    k_norm_replicated = k_norm_reshaped.repeat_interleave(repeats, dim=0)
                    k_norm_padded = k_norm_replicated.reshape(-1)  # [padded_num_kv_heads * head_dim]
                    neuron_state_dict[k_norm_key] = k_norm_padded
                    if layer_idx == 0:
                        print(f"  k_norm: {k_norm_full.shape} -> {k_norm_padded.shape} (replicated {repeats}x)")

            # Rename router weights: gate -> router.linear_router (use pop to avoid extra copy)
            gate_key = f"layers.{layer_idx}.block_sparse_moe.gate.weight"
            router_key = f"layers.{layer_idx}.block_sparse_moe.router.linear_router.weight"
            neuron_state_dict[router_key] = neuron_state_dict.pop(gate_key)

            # Remove e_score_correction_bias - not supported by generic RouterTopK
            # The bias is used for expert selection in the original model, but we use
            # standard top-k selection without bias for XLA compatibility
            bias_key = f"layers.{layer_idx}.block_sparse_moe.e_score_correction_bias"
            if bias_key in neuron_state_dict:
                del neuron_state_dict[bias_key]

            # Get expert weight dimensions
            w1_key = f"layers.{layer_idx}.block_sparse_moe.experts.0.w1.weight"
            intermediate_size, hidden_size = neuron_state_dict[w1_key].shape
            device = neuron_state_dict[w1_key].device
            dtype = neuron_state_dict[w1_key].dtype

            # Merge gate_proj (w1) and up_proj (w3) into gate_up_proj
            gate_up_proj = torch.empty(
                config.num_local_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=dtype,
                device=device,
            )

            for expert_idx in range(config.num_local_experts):
                # Optimized: use transpose view directly without clone
                w1_key = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
                w3_key = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"

                gate_up_proj_slice = torch.narrow(gate_up_proj, 0, expert_idx, 1)
                gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
                gate_proj_slice.copy_(neuron_state_dict[w1_key].T)
                up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
                up_proj_slice.copy_(neuron_state_dict[w3_key].T)

                # Delete immediately after copy
                del neuron_state_dict[w1_key]
                del neuron_state_dict[w3_key]

                # Periodic GC during expert processing
                if (expert_idx + 1) % gc_interval == 0:
                    gc.collect()

            # v3: padding gate_up_proj on intermediate size (from Qwen3 MoE)
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                gate_up_proj = gate_up_proj.reshape(config.num_local_experts, hidden_size, 2, -1)
                gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(config.num_local_experts, hidden_size, -1)
            neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

            # Handle down_proj (w2)
            down_proj = torch.empty(
                config.num_local_experts,
                intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )

            for expert_idx in range(config.num_local_experts):
                # Optimized: use transpose view directly without clone
                w2_key = f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
                down_proj_slice = torch.narrow(down_proj, 0, expert_idx, 1)
                down_proj_slice.copy_(neuron_state_dict[w2_key].T)
                del neuron_state_dict[w2_key]

                # Periodic GC during expert processing
                if (expert_idx + 1) % gc_interval == 0:
                    gc.collect()

            # v3: padding down_proj on intermediate size (from Qwen3 MoE)
            if pad_size > 0:
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
            neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

            # GC after each layer
            gc.collect()

            if (layer_idx + 1) % 10 == 0:
                print(f"  Processed layer {layer_idx + 1}/{config.num_hidden_layers}")

        # v3: fused_qkv support (from Qwen3 MoE)
        if config.neuron_config.fused_qkv:
            neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


class MiniMaxM2InferenceConfigV3(InferenceConfig):
    """Inference configuration for MiniMax M2 v3."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MiniMax M2 has num_local_experts
        # MiniMax M2 has no shared experts
        self.n_shared_experts = 0

        # v3: check whether need to pad intermediate size (from Qwen3 MoE)
        self.maybe_pad_intermediate()

        # v3: enable moe_fused_nki_kernel (from Qwen3 MoE)
        self.enable_moe_fused_nki_kernel()

        # v3: use intermediate_size for MoE (updated for consistency)
        self.moe_intermediate_size = self.intermediate_size

        # Router config for MiniMax M2: sigmoid with FP32 for accuracy
        self.neuron_config.router_config.dtype = torch.float32
        # MiniMax M2 uses sigmoid (configured via router_config in demo)
        # Keep user-specified act_fn if provided
        if not hasattr(self.neuron_config.router_config, 'act_fn') or self.neuron_config.router_config.act_fn is None:
            self.neuron_config.router_config.act_fn = "sigmoid"

        # v3: Set DISABLE_NUMERIC_CC_TOKEN=1 (from Qwen3 MoE)
        self.neuron_config.disable_numeric_cc_token = True

        # MiniMax M2 normalizes top k affinities
        self.neuron_config.normalize_top_k_affinities = True

    def maybe_pad_intermediate(self):
        """v3: Pad intermediate size for shard-on-I support (from Qwen3 MoE)."""
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.intermediate_size // moe_tp_degree
        if getattr(self.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_moe_intermediate_size = math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP) * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP * moe_tp_degree
                self.moe_intermediate_pad_size = max(padded_moe_intermediate_size - self.intermediate_size, 0)
                self.intermediate_size = padded_moe_intermediate_size

    def enable_moe_fused_nki_kernel(self):
        """v3: Enable MoE fused NKI kernel if conditions are met (from Qwen3 MoE)."""
        I_TP = self.intermediate_size // self.neuron_config.moe_tp_degree
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False) and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_local_experts",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
            "use_qk_norm",
            "rotary_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


class NeuronMiniMaxM2AttentionV3(NeuronAttentionBase):
    """
    MiniMax M2 Attention v3.

    Key features:
    1. QK norm applied BEFORE reshape (on full projection output) - MiniMax M2 specific
    2. Partial RoPE (rotary_dim=64, head_dim=128)
    """

    def __init__(self, config: MiniMaxM2InferenceConfigV3):
        # Get rotary_dim for partial RoPE
        self.rotary_dim = getattr(config, 'rotary_dim', config.head_dim)

        # Create RotaryEmbedding with rotary_dim (not head_dim) for partial RoPE
        rotary_emb = RotaryEmbedding(
            self.rotary_dim,
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
            # Disable base class qk_norm - we handle it with MiniMaxM2QKNorm
            use_qk_norm=False,
        )

        # MiniMax M2 qk_norm: applied BEFORE reshape (on full projection output)
        # This is different from Qwen3 MoE which applies per-head norm after reshape
        self.use_minimax_qk_norm = getattr(config, 'use_qk_norm', True)
        tp_degree = config.neuron_config.tp_degree
        if self.use_minimax_qk_norm:
            # Per-rank Q/K size (after tensor parallel sharding)
            # self.num_heads and self.num_key_value_heads are already per-rank values from base class
            q_size_per_rank = self.num_heads * self.head_dim  # e.g., 1 * 128 = 128
            k_size_per_rank = self.num_key_value_heads * self.head_dim  # e.g., 1 * 128 = 128

            # Calculate full_hidden_size for RMS divisor computation:
            #
            # For Q norm (interleaved padding):
            # - Original Q: 48 heads * 128 dim = 6144 elements
            # - After padding: 64 heads * 128 dim = 8192 elements, but padding is zeros
            # - GPU computes RMS over 6144 elements
            # - Neuron: global_sum_sq = all_reduce(local_sum_sq) = original sum (padding zeros contribute 0)
            # - Division should be by ORIGINAL 6144 to match GPU
            full_q_hidden_size = config.num_attention_heads * config.head_dim  # 48 * 128 = 6144

            # For K norm (replication):
            # - Original K: 8 KV heads * 128 dim = 1024 elements
            # - After replication: 64 KV heads * 128 dim = 8192 elements (each original value appears 8 times)
            # - GPU computes RMS over 1024 elements: sqrt(sum_sq / 1024)
            # - Neuron: global_sum_sq = all_reduce(local_sum_sq) = 8 * original_sum_sq (due to replication)
            # - Division by 8192: sqrt(8 * original_sum_sq / 8192) = sqrt(original_sum_sq / 1024) = GPU result
            # - So full_hidden_size should be 8192 (padded size) for K norm
            padded_num_kv_heads = self.num_key_value_heads * tp_degree  # 1 * 64 = 64
            full_k_hidden_size = padded_num_kv_heads * config.head_dim  # 64 * 128 = 8192

            # Padded hidden size for weight storage (both Q and K are padded to tp_degree heads)
            padded_q_hidden_size = self.num_heads * tp_degree * config.head_dim  # 1 * 64 * 128 = 8192
            padded_k_hidden_size = padded_num_kv_heads * config.head_dim  # 64 * 128 = 8192

            self.q_norm = MiniMaxM2QKNorm(
                q_size_per_rank,
                eps=config.rms_norm_eps,
                tp_degree=tp_degree,
                full_hidden_size=full_q_hidden_size,
                padded_hidden_size=padded_q_hidden_size
            )
            self.k_norm = MiniMaxM2QKNorm(
                k_size_per_rank,
                eps=config.rms_norm_eps,
                tp_degree=tp_degree,
                full_hidden_size=full_k_hidden_size,
                padded_hidden_size=padded_k_hidden_size
            )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM2AttentionV3 requires initialized distributed environment."
            )

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """Override to handle qk_norm BEFORE reshape and partial RoPE."""
        from neuronx_distributed_inference.modules.attention.utils import move_heads_front

        # Get Q, K, V projections
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )

        # Apply qk_norm BEFORE reshape (MiniMax M2 specific)
        # Pass rank_util for proper SPMD weight slicing
        if self.use_minimax_qk_norm:
            Q = self.q_norm(Q, self.rank_util)
            K = self.k_norm(K, self.rank_util)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        if not skip_rope:
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        return Q, K, V, cos_cache, sin_cache, residual

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Apply partial rotary embeddings (rotary_dim=64, head_dim=128)."""
        from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb

        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            # For partial rotary: only rotate first rotary_dim dimensions
            if self.rotary_dim < self.head_dim:
                Q_rot = Q[..., :self.rotary_dim]
                Q_pass = Q[..., self.rotary_dim:]
                K_rot = K[..., :self.rotary_dim]
                K_pass = K[..., self.rotary_dim:]

                Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)

                Q = torch.cat([Q_rot, Q_pass], dim=-1)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, cos_cache, sin_cache


def initialize_minimax_m2_moe_module_v3(config: MiniMaxM2InferenceConfigV3, rmsnorm=None, init_tkg_module=False):
    """
    Initialize MoE module for MiniMax M2 v3.

    NOTE: Now uses the generic initialize_moe_module to avoid XLA type mismatch issues.
    The e_score_correction_bias is handled separately via weight conversion.
    """
    # Use generic MoE initialization for XLA compatibility
    return initialize_moe_module(
        config=config,
        rmsnorm=rmsnorm,
        init_tkg_module=init_tkg_module,
        apply_act_fn_over_topk=False,
        router_bias=False,
        experts_bias=False,
    )


class NeuronMiniMaxM2DecoderLayerV3(nn.Module):
    """MiniMax M2 decoder layer v3 with attention and MoE."""

    def __init__(self, config: MiniMaxM2InferenceConfigV3, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniMaxM2AttentionV3(config=config)
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # v3: support moe_fused_nki_kernel like Qwen3 MoE
        if self.moe_fused_nki_kernel_enabled:
            self.block_sparse_moe = initialize_minimax_m2_moe_module_v3(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
            )
        else:
            self.block_sparse_moe = initialize_minimax_m2_moe_module_v3(config=config)

        # v3: store config flags for forward optimization
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
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        # v3: Module marker wrappers for compiler optimization (from Qwen3 MoE)
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

        # MoE
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronMiniMaxM2ModelV3(NeuronBaseModel):
    """NeuronMiniMaxM2Model v3 for tracing."""

    def setup_attr_for_model(self, config: MiniMaxM2InferenceConfigV3):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MiniMaxM2InferenceConfigV3):
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
            [NeuronMiniMaxM2DecoderLayerV3(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronMiniMaxM2ForCausalLMV3(NeuronBaseForCausalLM):
    """MiniMax M2 v3 for causal language modeling."""

    _model_cls = NeuronMiniMaxM2ModelV3

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """v3: Load HF model (from Qwen3 MoE pattern)."""
        # MiniMax M2 may not have a standard HF class, so we load manually
        return None

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM2InferenceConfigV3

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: MiniMaxM2InferenceConfigV3) -> dict:
        return convert_minimax_m2_hf_to_neuron_state_dict(state_dict, config)

    # v3: Wrap enable_context_encoding/enable_token_generation with compile_tag (from Qwen3 MoE)
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        """v3: Enhanced compiler args (from Qwen3 MoE)."""
        # Set compiler optimization level based on model tag
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            # Disable Modular flow for TKG graph with EP enabled as it causes perf degradation
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            compiler_args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "
            )

        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert (
                self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention
            ), "If using attn_block_tkg_nki_kernel_enabled for MiniMax M2 you must also use attn_block_tkg_nki_kernel_cascaded_attention"
            self.neuron_config.pre_rope_rmsnorm = True
            compiler_args += " --internal-max-instruction-limit=15000000"

        return compiler_args

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        """Load state dict from HuggingFace checkpoint."""
        import os
        from safetensors import safe_open
        import json

        if os.path.isdir(model_name_or_path):
            index_path = os.path.join(model_name_or_path, 'model.safetensors.index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    index = json.load(f)

                model_sd = {}
                shard_files = set(index['weight_map'].values())
                for i, shard_file in enumerate(sorted(shard_files)):
                    if i % 20 == 0:
                        print(f"  Loading shard {i+1}/{len(shard_files)}: {shard_file}")
                    shard_path = os.path.join(model_name_or_path, shard_file)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            model_sd[key] = f.get_tensor(key)

                print(f"  Loaded {len(model_sd)} parameters from {len(shard_files)} shards")

                # Remove model. prefix
                param_name_list = list(model_sd.keys())
                for param_name in param_name_list:
                    if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                        updated_param_name = param_name.replace(
                            cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                        )
                        model_sd[updated_param_name] = model_sd[param_name]
                        del model_sd[param_name]

                # Convert HF state dict to Neuron format
                model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)

                # Handle tied word embeddings if configured
                if getattr(config, "tie_word_embeddings", False):
                    cls.update_state_dict_for_tied_weights(model_sd)

                # Add _FUSED_PREFIX to keys if needed (for fused speculation)
                param_name_list = list(model_sd.keys())
                if cls._FUSED_PREFIX != "":
                    for param_name in param_name_list:
                        model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                        del model_sd[param_name]

                return model_sd
            else:
                from neuronx_distributed_inference.modules.checkpoint import load_state_dict
                return load_state_dict(model_name_or_path)
        else:
            return super().get_state_dict(model_name_or_path, config)
