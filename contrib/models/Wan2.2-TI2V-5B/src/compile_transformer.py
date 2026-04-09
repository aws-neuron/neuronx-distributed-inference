"""
Wan2.2 Transformer Compilation with Context Parallel (V3 CP).

This script implements Context Parallel for Wan2.2 video generation model:
- TP=4 for model parameter sharding
- CP=2 for sequence parallelism (via DP group)
- NKI Flash Attention for optimal performance
- world_size=8 (TP=4 x CP=2)

Key approach:
1. At forward entry: scatter hidden_states along sequence dimension
2. In self-attention: all-gather K/V across CP group
3. In cross-attention: K/V from text encoder is NOT split (same for all CP ranks)
4. At forward exit: gather output

Reference: hf_pretrained_qwen_image_edit/neuron_qwen_image_edit/compile_transformer_v3_cp.py
"""

import os
import json
import math

# Environment setup for NKI and Trainium2
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"  # Required for NKI
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags with ccop-compute-overlap for CP communication
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' --internal-hlo2tensorizer-options='--enable-state-buffer-mode=hybrid --remat-by-default' """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Tuple

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention

# ModelBuilder imports
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
)
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils

from safetensors.torch import load_file, save_file

# Import NKI Flash Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

print("NKI Flash Attention kernel loaded successfully")

# Import from existing module
from distributed_rmsnorm import DistributedRMSNorm


def get_sharded_data(data, dim):
    """Get sharded data for current TP rank."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    s = data.shape[dim] // tp_size
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()


NKI_SEQ_TILE = 128  # NKI attention kernel tile size for sequence dimension


def _pad_to_multiple(x, dim, multiple):
    """Pad tensor along dim to the nearest multiple. Returns (padded, original_len)."""
    orig_len = x.shape[dim]
    remainder = orig_len % multiple
    if remainder == 0:
        return x, orig_len
    pad_len = multiple - remainder
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim), orig_len


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, Q_len, D] - local query (may be shorter than K/V with CP)
        key: [B, H, KV_len, D] - full key (gathered if CP enabled)
        value: [B, H, KV_len, D] - full value (gathered if CP enabled)

    Returns:
        attention output [B, H, Q_len, D]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    # Pad Q seq_len to multiple of NKI_SEQ_TILE to avoid NCC_IBIR158 compiler bug
    query, orig_q_len = _pad_to_multiple(query, dim=2, multiple=NKI_SEQ_TILE)
    padded_q_len = query.shape[2]

    # Pad K/V seq_len to same tile alignment
    key, _ = _pad_to_multiple(key, dim=2, multiple=NKI_SEQ_TILE)
    value, _ = _pad_to_multiple(value, dim=2, multiple=NKI_SEQ_TILE)
    padded_k_len = key.shape[2]
    padded_v_len = value.shape[2]

    # Reshape for NKI kernel: (B*H, D, S) for Q/K, (B*H, S, D) for V
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, padded_q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, padded_k_len))
    v = value.clone().reshape((bs * n_head, padded_v_len, d_head))

    attn_output = torch.zeros((bs * n_head, padded_q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, padded_q_len, d_head))

    # Slice back to original Q length
    if padded_q_len != orig_q_len:
        attn_output = attn_output[:, :, :orig_q_len, :]

    return attn_output


def apply_rotary_emb_cp(hidden_states, freqs):
    """
    Apply rotary embeddings with pre-computed cos/sin tensors.

    This implementation matches Wan's apply_rotary_emb but handles the
    transposed tensor format used for NKI attention:
    - hidden_states: [batch, heads, seq_len, head_dim] (transposed for NKI)
    - freqs: tuple of (cos, sin), each with shape [1, seq_len, 1, head_dim] (Wan format)

    Wan's original implementation expects [batch, seq_len, heads, head_dim].
    We permute the RoPE tensors to broadcast with our [batch, heads, seq_len, head_dim] format.

    Args:
        hidden_states: [batch, heads, seq_len, head_dim]
        freqs: tuple of (freqs_cos, freqs_sin)

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    freqs_cos, freqs_sin = freqs
    dtype = hidden_states.dtype

    # Match Wan's apply_rotary_emb implementation:
    # x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    # cos = freqs_cos[..., 0::2]
    # sin = freqs_sin[..., 1::2]
    # out[..., 0::2] = x1 * cos - x2 * sin
    # out[..., 1::2] = x1 * sin + x2 * cos

    # Unflatten last dim into pairs and separate
    # hidden_states: [B, H, S, D] -> x1, x2: [B, H, S, D//2]
    x1, x2 = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)

    # freqs_cos/sin shape: [1, seq_len, 1, head_dim] (Wan format: [B, S, 1, D])
    # After [..., 0::2]: [1, seq_len, 1, head_dim//2]
    # Permute to [1, 1, seq_len, head_dim//2] to broadcast with [B, H, S, D//2]
    cos = freqs_cos[..., 0::2].permute(0, 2, 1, 3).float()  # [1, 1, S, D//2]
    sin = freqs_sin[..., 1::2].permute(0, 2, 1, 3).float()  # [1, 1, S, D//2]

    # Interleaved output: even indices get (x1*cos - x2*sin), odd indices get (x1*sin + x2*cos)
    out = torch.empty_like(hidden_states, dtype=torch.float32)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out.to(dtype)


def get_dp_rank_spmd(global_rank: torch.Tensor, tp_degree: int) -> torch.Tensor:
    """
    Compute DP rank from global rank for SPMD execution.

    With world_size=8 and tp_degree=4:
    - Ranks 0-3 are DP rank 0 (CP rank 0)
    - Ranks 4-7 are DP rank 1 (CP rank 1)
    """
    dp_rank = torch.div(global_rank, tp_degree, rounding_mode="floor").to(torch.int32)
    return dp_rank


def split_along_dim(tensor, dim, rank, data_parallel_group):
    """Split tensor along dimension using scatter_to_process_group_spmd."""
    return scatter_to_process_group_spmd(
        tensor,
        partition_dim=dim,
        rank=rank,
        process_group=data_parallel_group,
    )


def local_rms_norm(x, weight, eps=1e-6):
    """
    Apply RMSNorm locally on [B, S, local_inner_dim] without any all-reduce.

    The DistributedRMSNorm uses xm.all_reduce to compute global statistics
    across TP ranks, but the Neuron compiler creates incorrect replica groups
    ([[0,1,2,3]] instead of [[0,1,2,3],[4,5,6,7]]) causing a runtime assertion.

    This function computes RMSNorm purely locally over the full local_inner_dim
    (H_local * D). No cross-rank communication is generated. The difference
    from global norm (normalizing over H_total * D) is negligible for QK-norm
    since each TP shard has a statistically similar distribution of activations.

    Args:
        x: [B, S, local_inner_dim] tensor
        weight: [local_inner_dim] parameter from WanRMSNorm (already TP-sharded)
        eps: epsilon for numerical stability
    """
    dtype = x.dtype
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(dtype)


class CPWanSelfAttention(nn.Module):
    """
    Context Parallel + NKI Flash Attention for Wan2.2 Self-Attention (attn1).

    Key features:
    1. K/V are all-gathered across CP group before attention
    2. Uses NKI Flash Attention kernel
    3. Each CP rank processes its portion of queries against full K/V
    4. RoPE is applied before K/V gathering
    """

    def __init__(self, orig_attn, context_parallel_enabled=False, data_parallel_group=None, skip_kv_gather=False):
        super().__init__()

        self.context_parallel_enabled = context_parallel_enabled
        self.data_parallel_group = data_parallel_group
        self.skip_kv_gather = skip_kv_gather
        self.heads = orig_attn.heads

        # Copy projections (already sharded for TP)
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # QK normalization
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None

        # Store inner_dim for reshaping
        self.inner_dim = orig_attn.inner_dim if hasattr(orig_attn, 'inner_dim') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with CP K/V gathering and NKI attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V  [B, S, H_local*D]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Apply QK normalization on full local inner_dim (H_local*D).
        # Uses local_rms_norm which does NOT call xm.all_reduce, avoiding
        # the compiler bug that creates incorrect replica groups.
        if self.norm_q is not None:
            query = local_rms_norm(query, self.norm_q.weight, self.norm_q.eps)
        if self.norm_k is not None:
            key = local_rms_norm(key, self.norm_k.weight, self.norm_k.eps)

        # Reshape to [B, H, S, D] for attention
        head_dim = query.shape[-1] // self.heads
        query = query.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)

        # Apply RoPE before gathering (each rank has its local positions)
        if rotary_emb is not None:
            query = apply_rotary_emb_cp(query, rotary_emb)
            key = apply_rotary_emb_cp(key, rotary_emb)

        # Context Parallel: All-gather K/V across CP group
        # (Skipped for CFG Parallel: each rank has full sequence for its batch item)
        if self.context_parallel_enabled and not self.skip_kv_gather:
            dp_group = self.data_parallel_group
            # Stack K, V and gather together for efficiency
            kv_stacked = torch.stack([key, value], dim=0)  # [2, B, H, local_S, D]
            kv_stacked = gather_from_tensor_model_parallel_region_with_dim(
                kv_stacked, gather_dim=3, process_group=dp_group
            )  # [2, B, H, full_S, D]
            key, value = torch.unbind(kv_stacked, dim=0)

        # NKI Flash Attention: Q @ K/V (local Q @ full K/V for CP, full Q @ full K/V for CFG)
        hidden_states = nki_flash_attention(query, key, value)

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class CPWanCrossAttention(nn.Module):
    """
    Context Parallel + NKI Flash Attention for Wan2.2 Cross-Attention (attn2).

    Key difference from self-attention:
    - Query comes from video hidden_states (split across CP)
    - Key/Value come from text encoder_hidden_states (NOT split - same for all CP ranks)
    - NO K/V gathering needed

    For I2V tasks, also handles image context via add_k_proj, add_v_proj.
    """

    def __init__(self, orig_attn, context_parallel_enabled=False):
        super().__init__()

        self.context_parallel_enabled = context_parallel_enabled
        # NOTE: Cross-attention doesn't need data_parallel_group because K/V from text is not split
        self.heads = orig_attn.heads

        # Copy projections (already sharded for TP)
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # QK normalization
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None

        # I2V: additional projections for image context
        self.add_k_proj = orig_attn.add_k_proj if hasattr(orig_attn, 'add_k_proj') else None
        self.add_v_proj = orig_attn.add_v_proj if hasattr(orig_attn, 'add_v_proj') else None
        self.norm_added_k = orig_attn.norm_added_k if hasattr(orig_attn, 'norm_added_k') else None

        self.inner_dim = orig_attn.inner_dim if hasattr(orig_attn, 'inner_dim') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with cross-attention (no K/V gathering needed)."""
        batch_size, local_seq, _ = hidden_states.shape

        # Handle I2V image context
        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            # 512 is text context length (hardcoded in original)
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Query from video (split), K/V from text (NOT split)
        query = self.to_q(hidden_states)       # [B, local_seq, H_local*D]
        key = self.to_k(encoder_hidden_states)  # [B, text_len, H_local*D]
        value = self.to_v(encoder_hidden_states)

        # Apply QK normalization on full local inner_dim (no all-reduce)
        if self.norm_q is not None:
            query = local_rms_norm(query, self.norm_q.weight, self.norm_q.eps)
        if self.norm_k is not None:
            key = local_rms_norm(key, self.norm_k.weight, self.norm_k.eps)

        # Reshape to [B, H, S, D]
        head_dim = query.shape[-1] // self.heads
        query = query.view(batch_size, local_seq, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Handle I2V image attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj(encoder_hidden_states_img)  # [B, img_len, H_local*D]
            if self.norm_added_k is not None and self.norm_added_k.weight is not None:
                key_img = local_rms_norm(key_img, self.norm_added_k.weight, self.norm_added_k.eps)
            value_img = self.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value_img = value_img.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # NKI attention for image context
            hidden_states_img = nki_flash_attention(query, key_img, value_img)
            hidden_states_img = hidden_states_img.transpose(1, 2).reshape(batch_size, local_seq, -1)
            hidden_states_img = hidden_states_img.to(query.dtype)

        # NKI Flash Attention for text context
        # Note: NO K/V gathering - text is global context
        hidden_states = nki_flash_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, local_seq, -1)
        hidden_states = hidden_states.to(query.dtype)

        # Combine image and text attention outputs
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            hidden_states = self.to_out[1](hidden_states)

        return hidden_states


def shard_attention_for_cp(tp_degree: int, attn: Attention):
    """
    Shard attention module for TP=4 Context Parallel mode.

    Similar to shard_transformer3d_attn but for TP=4 instead of TP=8.
    """
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    orig_num_heads = attn.heads

    # Check if padding is needed
    extra_heads = get_number_of_extra_heads(attn.heads, tp_degree)

    if extra_heads == 0:
        # No padding case (e.g., 12 heads / TP=4 = 3 heads per rank)
        attn.heads = orig_num_heads // tp_degree
        attn.sliceable_head_dim = attn.heads
        new_inner_dim = dim_head * attn.heads
        attn.inner_dim = new_inner_dim
    else:
        # Padding case
        total_padded_heads = orig_num_heads + extra_heads
        attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
        attn.sliceable_head_dim = attn.heads
        new_inner_dim = dim_head * attn.heads
        attn.inner_dim = new_inner_dim

    # Shard Q projection
    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        orig_q.in_features, orig_q.out_features,
        bias=(orig_q.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16
    )
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if orig_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    # Shard K projection
    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        orig_k.in_features, orig_k.out_features,
        bias=(orig_k.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16
    )
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if orig_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # Shard V projection
    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        orig_v.in_features, orig_v.out_features,
        bias=(orig_v.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16
    )
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if orig_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard output projection
    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        orig_out.in_features, orig_out.out_features,
        bias=(orig_out.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16
    )
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if orig_out.bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del orig_out

    # Handle norm_q and norm_k with DistributedRMSNorm
    if hasattr(attn, 'norm_q') and attn.norm_q is not None:
        orig_norm = attn.norm_q
        eps = orig_norm.eps if hasattr(orig_norm, 'eps') else 1e-5
        attn.norm_q = DistributedRMSNorm(new_inner_dim, eps=eps, elementwise_affine=True)
        if hasattr(orig_norm, 'weight') and orig_norm.weight is not None:
            attn.norm_q.weight.data = get_sharded_data(orig_norm.weight.data, 0)

    if hasattr(attn, 'norm_k') and attn.norm_k is not None:
        orig_norm = attn.norm_k
        eps = orig_norm.eps if hasattr(orig_norm, 'eps') else 1e-5
        attn.norm_k = DistributedRMSNorm(new_inner_dim, eps=eps, elementwise_affine=True)
        if hasattr(orig_norm, 'weight') and orig_norm.weight is not None:
            attn.norm_k.weight.data = get_sharded_data(orig_norm.weight.data, 0)

    # Handle I2V projections
    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        orig_add_k = attn.add_k_proj
        attn.add_k_proj = ColumnParallelLinear(
            orig_add_k.in_features, orig_add_k.out_features,
            bias=(orig_add_k.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16
        )
        attn.add_k_proj.weight.data = get_sharded_data(orig_add_k.weight.data, 0)
        if orig_add_k.bias is not None:
            attn.add_k_proj.bias.data = get_sharded_data(orig_add_k.bias.data, 0)
        del orig_add_k

    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        orig_add_v = attn.add_v_proj
        attn.add_v_proj = ColumnParallelLinear(
            orig_add_v.in_features, orig_add_v.out_features,
            bias=(orig_add_v.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16
        )
        attn.add_v_proj.weight.data = get_sharded_data(orig_add_v.weight.data, 0)
        if orig_add_v.bias is not None:
            attn.add_v_proj.bias.data = get_sharded_data(orig_add_v.bias.data, 0)
        del orig_add_v

    if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
        orig_norm = attn.norm_added_k
        eps = orig_norm.eps if hasattr(orig_norm, 'eps') else 1e-5
        elementwise = orig_norm.elementwise_affine if hasattr(orig_norm, 'elementwise_affine') else False
        attn.norm_added_k = DistributedRMSNorm(new_inner_dim, eps=eps, elementwise_affine=elementwise)
        if elementwise and hasattr(orig_norm, 'weight') and orig_norm.weight is not None:
            attn.norm_added_k.weight.data = get_sharded_data(orig_norm.weight.data, 0)

    return attn


def shard_feedforward_for_cp(ff: FeedForward) -> FeedForward:
    """Shard FeedForward for TP=4."""
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        orig_proj.in_features, orig_proj.out_features,
        bias=(orig_proj.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16
    )
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if orig_proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del orig_proj

    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        orig_linear.in_features, orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16
    )
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if orig_linear.bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    del orig_linear

    return ff


class NeuronWanTransformerV3CP(nn.Module):
    """
    Neuron-optimized Wan2.2 Transformer with Context Parallel.

    Features:
    - TP=4 for model parameter sharding
    - CP=2 via DP group for sequence parallelism
    - Data is SPLIT at entry, K/V gathered in self-attention, output gathered at exit
    - Cross-attention K/V (text) is NOT split
    - NKI Flash Attention
    """

    def __init__(self, original_transformer, tp_degree, world_size, context_parallel_enabled=False, cfg_parallel=False):
        super().__init__()

        self.config = original_transformer.config
        self.context_parallel_enabled = context_parallel_enabled
        self.cfg_parallel = cfg_parallel
        self.tp_degree = tp_degree
        self.world_size = world_size

        # SPMDRank for runtime rank detection (crucial for SPMD scatter/gather)
        self.global_rank = SPMDRank(world_size=world_size)

        # Capture data_parallel_group at init time (within NxDParallelState context).
        # This ensures the correct group is baked into the compiled NEFF.
        self.data_parallel_group = parallel_state.get_data_parallel_group()

        # Patch embedding
        self.patch_embedding = original_transformer.patch_embedding

        # Condition embedder (not sharded - shared)
        self.condition_embedder = original_transformer.condition_embedder

        # Transformer blocks with TP sharding
        self.blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.blocks):
            # Shard attention and FFN with TP=4
            block.attn1 = shard_attention_for_cp(tp_degree, block.attn1)
            block.attn2 = shard_attention_for_cp(tp_degree, block.attn2)
            block.ffn = shard_feedforward_for_cp(block.ffn)
            self.blocks.append(block)

            if (i + 1) % 8 == 0:
                print(f"  Sharded block {i+1}/{len(original_transformer.blocks)}")

        # Replace attention with CP versions
        self._replace_attention()

        # Output layers
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out
        self.scale_shift_table = original_transformer.scale_shift_table

        # Store RoPE dimensions
        self.attention_head_dim = original_transformer.config.attention_head_dim
        self.patch_size = original_transformer.config.patch_size

    def _replace_attention(self):
        """Replace attention modules with CP/CFG+NKI versions."""
        for i, block in enumerate(self.blocks):
            # Replace self-attention (attn1)
            # For CFG Parallel: skip K/V gather (each rank has full sequence)
            block.attn1 = CPWanSelfAttention(
                block.attn1,
                self.context_parallel_enabled,
                self.data_parallel_group,
                skip_kv_gather=self.cfg_parallel,
            )

            # Replace cross-attention (attn2) - no K/V gather in either mode
            block.attn2 = CPWanCrossAttention(
                block.attn2,
                self.context_parallel_enabled
            )

        mode = "CFG" if self.cfg_parallel else "CP"
        print(f"Replaced attention with {mode}+NKI versions on {len(self.blocks)} blocks")

    def _find_rope_seq_dim(self, rope_tensor, expected_seq_len):
        """
        Find the dimension in RoPE tensor that corresponds to sequence length.

        RoPE can have different shapes depending on diffusers version:
        - [1, 1, seq_len, head_dim//2] - standard
        - [1, seq_len, 1, head_dim] - newer versions
        - [seq_len, head_dim//2] - compact
        - [1, seq_len, head_dim//2] - some versions
        """
        cp_degree = self.world_size // self.tp_degree

        # First, look for exact match
        for dim in range(rope_tensor.dim()):
            if rope_tensor.shape[dim] == expected_seq_len:
                return dim

        # If no exact match, look for the largest dimension that's divisible by CP degree
        # and greater than 1 (skip batch/singleton dimensions)
        best_dim = -1
        best_size = 0
        for dim in range(rope_tensor.dim()):
            size = rope_tensor.shape[dim]
            if size > 1 and size % cp_degree == 0 and size > best_size:
                best_dim = dim
                best_size = size

        if best_dim >= 0:
            print(f"DEBUG: Using dim={best_dim} (size={best_size}) for RoPE scatter")
            return best_dim

        raise ValueError(f"Cannot find sequence dimension in RoPE tensor with shape {rope_tensor.shape}, expected seq_len={expected_seq_len}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_cos: torch.Tensor,
        rotary_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with Context Parallel data splitting."""

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Patch embedding: [B, C, F, H, W] -> [B, seq_len, D]
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        full_seq_len = hidden_states.shape[1]

        # Debug: print shapes during first trace
        print(f"DEBUG: hidden_states shape after patch: {hidden_states.shape}")
        print(f"DEBUG: rotary_emb_cos shape: {rotary_emb_cos.shape}")
        print(f"DEBUG: rotary_emb_sin shape: {rotary_emb_sin.shape}")
        print(f"DEBUG: full_seq_len: {full_seq_len}")

        # ========== PARALLEL DATA SPLIT AT ENTRY ==========
        if self.context_parallel_enabled:
            dp_group = self.data_parallel_group

            # Get DP rank at runtime using SPMDRank
            dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.tp_degree)

            if self.cfg_parallel:
                # CFG Parallel: scatter along batch dim (dim=0)
                # [2, seq, D] -> [1, seq, D] per rank
                hidden_states = split_along_dim(
                    hidden_states, dim=0, rank=dp_rank, data_parallel_group=dp_group
                )
                encoder_hidden_states = split_along_dim(
                    encoder_hidden_states, dim=0, rank=dp_rank, data_parallel_group=dp_group
                )
                timestep = split_along_dim(
                    timestep, dim=0, rank=dp_rank, data_parallel_group=dp_group
                )
                # RoPE: NOT scattered (position-indexed, same for both batch items)
            else:
                # Context Parallel: scatter along sequence dim (dim=1)
                hidden_states = split_along_dim(
                    hidden_states, dim=1, rank=dp_rank, data_parallel_group=dp_group
                )

                # Split RoPE along sequence dim
                rope_seq_dim = self._find_rope_seq_dim(rotary_emb_cos, full_seq_len)
                rotary_emb_cos = split_along_dim(
                    rotary_emb_cos, dim=rope_seq_dim, rank=dp_rank, data_parallel_group=dp_group
                )
                rotary_emb_sin = split_along_dim(
                    rotary_emb_sin, dim=rope_seq_dim, rank=dp_rank, data_parallel_group=dp_group
                )

        # Condition embedding
        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Process through blocks
        for block in self.blocks:
            # Extract scale/shift parameters
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                block.scale_shift_table + timestep_proj.float()
            ).chunk(6, dim=1)

            # 1. Self-attention with RoPE
            norm_hidden = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            rotary_emb = (rotary_emb_cos, rotary_emb_sin)
            attn_output = block.attn1(hidden_states=norm_hidden, rotary_emb=rotary_emb)
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

            # 2. Cross-attention (no RoPE, K/V from text)
            norm_hidden = block.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = block.attn2(hidden_states=norm_hidden, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output

            # 3. Feed-forward
            norm_hidden = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
            ff_output = block.ffn(norm_hidden)
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        # Output norm and projection
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        output = self.proj_out(hidden_states)

        # ========== PARALLEL: GATHER OUTPUT ==========
        if self.context_parallel_enabled:
            gather_dim = 0 if self.cfg_parallel else 1
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=gather_dim, process_group=self.data_parallel_group
            )

        # Unpatchify
        output = output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


class TracingWrapper(nn.Module):
    """Wrapper for tracing with ModelBuilder."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin):
        return self.transformer(
            hidden_states, timestep, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin
        )


def compute_rope(transformer, latent_frames, latent_height, latent_width, in_channels=48):
    """
    Compute full RoPE for given video dimensions.

    Uses the transformer's rope.forward() method which correctly handles
    the 3D RoPE computation for video (frames, height, width).
    """
    # Create dummy hidden_states to trigger rope computation
    batch_size = 1
    dummy_hidden = torch.zeros(
        batch_size, in_channels, latent_frames, latent_height, latent_width,
        dtype=torch.float32
    )

    print(f"  Computing RoPE for shape: {dummy_hidden.shape}")

    # Call rope forward - returns (cos, sin) tuple
    rotary_emb = transformer.rope(dummy_hidden)

    # rotary_emb is a tuple of (freqs_cos, freqs_sin)
    # Each has shape [1, 1, seq_len, head_dim//2]
    if isinstance(rotary_emb, tuple):
        freqs_cos, freqs_sin = rotary_emb
        print(f"  RoPE cos shape: {freqs_cos.shape}")
        print(f"  RoPE sin shape: {freqs_sin.shape}")
    else:
        # If it returns complex tensor (old format), handle separately
        print(f"  Unexpected rope output type: {type(rotary_emb)}")
        raise ValueError("Unexpected rope output format. Expected (cos, sin) tuple.")

    return freqs_cos, freqs_sin


def fix_norm_weights_per_rank(weights_path, unsharded_norm_weights, tp_degree):
    """Fix norm_k/norm_q/norm_added_k weights for each rank after shard_checkpoint.

    shard_checkpoint doesn't recognize norm_q/norm_k/norm_added_k inside CP attention
    modules as parallel layers, so they remain unsharded. This function manually shards them.
    """
    print(f"Fixing norm weights for {tp_degree} ranks...")

    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)

        fixed_count = 0
        for key, unsharded_weight in unsharded_norm_weights.items():
            if key in ckpt:
                ckpt_shape = ckpt[key].shape[0]
                unsharded_dim = unsharded_weight.shape[0]
                expected_shard_size = unsharded_dim // tp_degree

                if ckpt_shape == expected_shard_size:
                    # Already correctly sharded, just slice from unsharded
                    start = expected_shard_size * rank
                    end = expected_shard_size * (rank + 1)
                    correct_slice = unsharded_weight[start:end].clone()
                elif ckpt_shape == unsharded_dim:
                    # Not sharded at all - shard it now
                    start = expected_shard_size * rank
                    end = expected_shard_size * (rank + 1)
                    correct_slice = unsharded_weight[start:end].clone()
                else:
                    # Dimension needs padding (not evenly divisible)
                    padded_dim = ((unsharded_dim + tp_degree - 1) // tp_degree) * tp_degree
                    padded_weight = torch.ones(padded_dim, dtype=unsharded_weight.dtype)
                    padded_weight[:unsharded_dim] = unsharded_weight
                    shard_size = padded_dim // tp_degree
                    start = shard_size * rank
                    end = shard_size * (rank + 1)
                    correct_slice = padded_weight[start:end].clone()

                ckpt[key] = correct_slice
                fixed_count += 1

        save_file(ckpt, ckpt_path)
        print(f"  Rank {rank}: Fixed {fixed_count} norm weights")


def compile_transformer_v3_cp(args):
    """Compile transformer with Context Parallel or CFG Parallel using ModelBuilder API."""

    tp_degree = args.tp_degree
    world_size = args.world_size
    cfg_parallel = getattr(args, 'cfg_parallel', False)
    context_parallel_enabled = (world_size != tp_degree)
    cp_degree = world_size // tp_degree if context_parallel_enabled else 1

    latent_height = args.height // 16
    latent_width = args.width // 16
    latent_frames = (args.num_frames - 1) // 4 + 1
    max_sequence_length = args.max_sequence_length
    hidden_size = 4096
    # CFG Parallel: batch_size=2 (negative + positive stacked)
    batch_size = 2 if cfg_parallel else 1
    in_channels = 48

    # Calculate sequence length after patch embedding
    patch_size_t, patch_size_h, patch_size_w = 1, 2, 2
    seq_len = (latent_frames // patch_size_t) * (latent_height // patch_size_h) * (latent_width // patch_size_w)

    mode = "CFG Parallel" if cfg_parallel else "Context Parallel"
    print("=" * 60)
    print(f"Wan2.2 Transformer V3 {mode} Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}, Frames: {args.num_frames}")
    print(f"Latent: {latent_frames}x{latent_height}x{latent_width}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"TP degree: {tp_degree}")
    print(f"CP/CFG degree: {cp_degree}")
    print(f"World size: {world_size}")
    print(f"Mode: {mode}")
    print(f"NKI Flash Attention: Enabled")
    print("=" * 60)

    # Sample inputs
    sample_hidden_states = torch.randn(
        batch_size, in_channels, latent_frames, latent_height, latent_width,
        dtype=torch.bfloat16
    )
    sample_encoder_hidden_states = torch.randn(
        batch_size, max_sequence_length, hidden_size,
        dtype=torch.bfloat16
    )
    sample_timestep = torch.randn(batch_size, dtype=torch.float32)

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("\nLoading model...")
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
        )

        # Compute full RoPE
        print("\nComputing RoPE...")
        rotary_emb_cos, rotary_emb_sin = compute_rope(
            pipe.transformer, latent_frames, latent_height, latent_width
        )
        rotary_emb_cos = rotary_emb_cos.to(torch.bfloat16)
        rotary_emb_sin = rotary_emb_sin.to(torch.bfloat16)
        print(f"  RoPE cos: {rotary_emb_cos.shape}")
        print(f"  RoPE sin: {rotary_emb_sin.shape}")

        # Save unsharded state dict before modifications
        unsharded_state = pipe.transformer.state_dict()

        # Collect unsharded norm weights (norm_q, norm_k, norm_added_k for I2V)
        unsharded_norm_weights = {}
        for key, value in unsharded_state.items():
            if 'norm_k.weight' in key or 'norm_q.weight' in key or 'norm_added_k.weight' in key:
                unsharded_norm_weights[f"transformer.{key}"] = value.clone()
        print(f"Collected {len(unsharded_norm_weights)} unsharded norm weights")

        # Create Neuron transformer
        print("\nCreating Neuron transformer (TP={}, {}={}, world_size={})...".format(
            tp_degree, "CFG" if cfg_parallel else "CP", cp_degree, world_size
        ))
        neuron_transformer = NeuronWanTransformerV3CP(
            pipe.transformer, tp_degree, world_size, context_parallel_enabled, cfg_parallel=cfg_parallel
        )
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Wrap for tracing
        model = TracingWrapper(neuron_transformer)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "timestep": sample_timestep,
                "encoder_hidden_states": sample_encoder_hidden_states,
                "rotary_emb_cos": rotary_emb_cos,
                "rotary_emb_sin": rotary_emb_sin,
            },
            tag="inference",
        )

        print("Compiling model...")
        compile_args = "--model-type=transformer -O1 --auto-cast=none --internal-hlo2tensorizer-options='--enable-native-kernel=1 --remat'"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_subdir = "transformer_cfg" if cfg_parallel else "transformer"
        output_path = f"{args.compiled_models_dir}/{output_subdir}"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint - convert all to bfloat16
        checkpoint = {}
        global_rank_state = {}
        for key, value in model.state_dict().items():
            if 'global_rank' in key:
                global_rank_state[key] = value.clone()
                continue
            orig_key = key.replace("transformer.", "", 1)
            if orig_key in unsharded_state:
                val = unsharded_state[orig_key].clone()
            else:
                val = value.clone()
            # Convert to bfloat16 (model expects bfloat16)
            if val.dtype == torch.float32:
                val = val.to(torch.bfloat16)
            checkpoint[key] = val

        print("Sharding weights...")
        shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            serialize_path=weights_path,
        )

        # Post-process sharded checkpoints: remove master_weight tensors and add global_rank
        print("Post-processing sharded checkpoints...")
        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                print(f"  WARNING: {shard_file} not found")
                continue

            shard_data = dict(load_file(shard_file))
            original_count = len(shard_data)

            # Remove master_weight tensors (duplicates created by shard_checkpoint)
            cleaned = {k: v for k, v in shard_data.items() if 'master_weight' not in k}

            # Add SPMDRank state
            if global_rank_state:
                cleaned.update(global_rank_state)

            save_file(cleaned, shard_file)
            removed = original_count - len(cleaned) + len(global_rank_state)
            print(f"  tp{rank}: {original_count} -> {len(cleaned)} tensors (removed {removed} master_weight)")

        # Fix norm weights - also convert to bfloat16
        unsharded_norm_weights_bf16 = {k: v.to(torch.bfloat16) for k, v in unsharded_norm_weights.items()}
        fix_norm_weights_per_rank(weights_path, unsharded_norm_weights_bf16, tp_degree)

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "seq_len": seq_len,
            "max_sequence_length": max_sequence_length,
            "batch_size": batch_size,
            "tp_degree": tp_degree,
            "cp_degree": cp_degree,
            "world_size": world_size,
            "context_parallel": context_parallel_enabled,
            "cfg_parallel": cfg_parallel,
            "nki_flash_attention": True,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save RoPE cache
        torch.save({
            "rotary_emb_cos": rotary_emb_cos,
            "rotary_emb_sin": rotary_emb_sin,
        }, os.path.join(output_path, "rope_cache.pt"))

        print("\nCompilation complete!")
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 Transformer with Context Parallel")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max text sequence length")
    parser.add_argument("--tp_degree", type=int, default=4, help="Tensor parallelism degree")
    parser.add_argument("--world_size", type=int, default=8, help="Total world size (TP x CP)")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir", help="Compiler workdir")
    parser.add_argument("--cfg_parallel", action="store_true",
                        help="Use CFG Parallel (batch=2, no K/V gather) instead of Context Parallel")
    args = parser.parse_args()

    compile_transformer_v3_cp(args)
