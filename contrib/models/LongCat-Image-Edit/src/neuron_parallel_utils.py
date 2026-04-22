"""
FLUX-specific tensor parallelism sharding functions for LongCat-Image-Edit.

LongCat uses a FLUX-style transformer with two types of blocks:
1. Dual-stream blocks (FluxTransformerBlock): separate text/image norms+FFN, joint attention
2. Single-stream blocks (FluxSingleTransformerBlock): concatenated text+image, parallel MLP+attention

This module provides sharding functions for both block types.
"""

import torch
from torch import nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils


def get_sharded_data(data, dim):
    """Shard data across tensor parallel ranks."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()


def shard_linear_column(orig_linear, gather_output=False, dtype=torch.bfloat16):
    """Replace a nn.Linear with ColumnParallelLinear."""
    new_linear = ColumnParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        gather_output=gather_output,
        dtype=dtype,
    )
    new_linear.weight.data = get_sharded_data(orig_linear.weight.data, 0)
    if orig_linear.bias is not None:
        if gather_output:
            # Bias is added after gathering, so keep full size
            new_linear.bias.data = orig_linear.bias.data.clone().to(dtype)
        else:
            new_linear.bias.data = get_sharded_data(orig_linear.bias.data, 0)
    return new_linear


def shard_linear_row(orig_linear, dtype=torch.bfloat16):
    """Replace a nn.Linear with RowParallelLinear."""
    new_linear = RowParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        input_is_parallel=True,
        dtype=dtype,
    )
    new_linear.weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if orig_linear.bias is not None:
        new_linear.bias.data = orig_linear.bias.data.detach()
    return new_linear


def shard_flux_dual_block(tp_degree, block):
    """
    Shard a FLUX dual-stream transformer block for tensor parallelism.

    Dual-stream block structure:
    - norm1.linear: [3072 -> 18432] modulation (6 * 3072), gather_output=True
    - norm1_context.linear: [3072 -> 18432] text modulation
    - attn.to_q/k/v: [3072 -> 3072] image QKV
    - attn.to_out[0]: [3072 -> 3072] image output
    - attn.add_q_proj/add_k_proj/add_v_proj: [3072 -> 3072] text QKV
    - attn.to_add_out: [3072 -> 3072] text output
    - attn.norm_q/k/added_q/added_k: RMSNorm(128) per-head, NOT sharded
    - ff.net[0].proj: [3072 -> 12288] GEGLU
    - ff.net[2]: [12288 -> 3072] output
    - ff_context (same as ff)

    LongCat: 24 heads, head_dim=128, inner_dim=3072
    With TP=4: 24/4 = 6 heads per rank (evenly divisible)
    """
    # --- Modulation layers (gather_output=True for full modulation params) ---
    if hasattr(block, 'norm1') and hasattr(block.norm1, 'linear'):
        block.norm1.linear = shard_linear_column(block.norm1.linear, gather_output=True)
    if hasattr(block, 'norm1_context') and hasattr(block.norm1_context, 'linear'):
        block.norm1_context.linear = shard_linear_column(block.norm1_context.linear, gather_output=True)

    # --- Attention: Image stream ---
    attn = block.attn

    # Update number of heads per rank
    orig_num_heads = attn.heads
    total_padded_heads = orig_num_heads + get_number_of_extra_heads(orig_num_heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)

    # Image QKV (ColumnParallel)
    attn.to_q = shard_linear_column(attn.to_q)
    attn.to_k = shard_linear_column(attn.to_k)
    attn.to_v = shard_linear_column(attn.to_v)

    # Image output (RowParallel)
    orig_out = attn.to_out[0]
    attn.to_out[0] = shard_linear_row(orig_out)
    del orig_out

    # --- Attention: Text stream ---
    if hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
        attn.add_q_proj = shard_linear_column(attn.add_q_proj)
    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        attn.add_k_proj = shard_linear_column(attn.add_k_proj)
    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        attn.add_v_proj = shard_linear_column(attn.add_v_proj)
    if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
        attn.to_add_out = shard_linear_row(attn.to_add_out)

    # Note: norm_q, norm_k, norm_added_q, norm_added_k are RMSNorm(128)
    # They operate on head_dim which doesn't change with TP, so NOT sharded.

    # --- FeedForward: Image stream ---
    if hasattr(block, 'ff'):
        shard_feedforward(block.ff)

    # --- FeedForward: Text stream ---
    if hasattr(block, 'ff_context'):
        shard_feedforward(block.ff_context)

    return block


def shard_flux_single_block(tp_degree, block):
    """
    Shard a FLUX single-stream transformer block for tensor parallelism.

    Single-stream block structure:
    - norm.linear: [3072 -> 9216] modulation (3 * 3072), gather_output=True
    - attn.to_q/k/v: [3072 -> 3072] QKV
    - proj_mlp: [3072 -> 12288] parallel MLP
    - proj_out: [15360 -> 3072] combined output (3072 attn + 12288 mlp = 15360)
      With TP=4: input is (768 attn + 3072 mlp = 3840) per rank

    CRITICAL: proj_out weight columns must be reordered to match the per-rank
    input layout [attn_shard, mlp_shard], NOT contiguous column slicing.
    The original weight layout is [attn_full(3072), mlp_full(12288)] = 15360.
    But each rank's input is [attn_shard(768), mlp_shard(3072)] = 3840.
    Standard RowParallel takes contiguous columns which MISALIGNS with this input.

    LongCat: 24 heads, head_dim=128
    """
    attn = block.attn

    # Update heads
    orig_num_heads = attn.heads
    total_padded_heads = orig_num_heads + get_number_of_extra_heads(orig_num_heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)

    # --- Modulation (gather_output=True) ---
    if hasattr(block, 'norm') and hasattr(block.norm, 'linear'):
        block.norm.linear = shard_linear_column(block.norm.linear, gather_output=True)

    # --- Attention QKV (ColumnParallel) ---
    attn.to_q = shard_linear_column(attn.to_q)
    attn.to_k = shard_linear_column(attn.to_k)
    attn.to_v = shard_linear_column(attn.to_v)

    # --- Parallel MLP (ColumnParallel) ---
    if hasattr(block, 'proj_mlp'):
        block.proj_mlp = shard_linear_column(block.proj_mlp)

    # --- Combined output projection (custom RowParallel with reordered columns) ---
    # proj_out input = [attn_output, mlp_output] concatenated per rank.
    # Original weight: [out_dim, attn_dim + mlp_dim] = [3072, 15360]
    # Per rank r, input features correspond to:
    #   attn cols: [r*attn_per_rank : (r+1)*attn_per_rank]
    #   mlp cols:  [attn_dim + r*mlp_per_rank : attn_dim + (r+1)*mlp_per_rank]
    # These are NON-CONTIGUOUS in the original weight, so we must extract them.
    if hasattr(block, 'proj_out'):
        block.proj_out = shard_proj_out_interleaved(block.proj_out, orig_num_heads * 128, block.mlp_hidden_dim, tp_degree)

    return block


def shard_proj_out_interleaved(orig_linear, attn_dim, mlp_dim, tp_degree, dtype=torch.bfloat16):
    """
    Shard proj_out for single-stream blocks with correct column reordering.

    The input to proj_out is [attn_output(per_rank), mlp_output(per_rank)]
    concatenated. But attn and mlp are sharded independently, so the per-rank
    columns are non-contiguous in the original weight.

    For rank r:
      attn_cols = [r * attn_per_rank : (r+1) * attn_per_rank]
      mlp_cols  = [attn_dim + r * mlp_per_rank : attn_dim + (r+1) * mlp_per_rank]
      weight_shard = orig_weight[:, attn_cols ++ mlp_cols]
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()

    attn_per_rank = attn_dim // tp_size
    mlp_per_rank = mlp_dim // tp_size
    input_per_rank = attn_per_rank + mlp_per_rank

    # Create RowParallelLinear with correct input size
    new_linear = RowParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        input_is_parallel=True,
        dtype=dtype,
    )

    # Extract correct non-contiguous weight columns for this rank
    attn_start = tp_rank * attn_per_rank
    attn_end = (tp_rank + 1) * attn_per_rank
    mlp_start = attn_dim + tp_rank * mlp_per_rank
    mlp_end = attn_dim + (tp_rank + 1) * mlp_per_rank

    w_attn = orig_linear.weight.data[:, attn_start:attn_end]  # [out, attn_per_rank]
    w_mlp = orig_linear.weight.data[:, mlp_start:mlp_end]     # [out, mlp_per_rank]
    w_reordered = torch.cat([w_attn, w_mlp], dim=1)           # [out, input_per_rank]

    new_linear.weight.data = w_reordered.to(dtype)

    if orig_linear.bias is not None:
        new_linear.bias.data = orig_linear.bias.data.detach().to(dtype)

    return new_linear


def shard_feedforward(ff):
    """
    Shard a FLUX FeedForward module (GEGLU variant).

    Structure: net[0].proj (GEGLU projection), net[2] (output linear)
    - net[0].proj: [3072 -> 12288] (GEGLU, may actually be [3072 -> 24576] for gated)
    - net[2]: [12288 -> 3072]
    """
    if hasattr(ff, 'net'):
        # GEGLU projection
        if hasattr(ff.net[0], 'proj'):
            ff.net[0].proj = shard_linear_column(ff.net[0].proj)
        # Output projection
        if len(ff.net) > 2:
            orig_linear = ff.net[2]
            ff.net[2] = shard_linear_row(orig_linear)
            del orig_linear
    return ff


# ============================================================================
# Vision encoder and Language model sharding (reused from Qwen reference)
# These are identical since both models use Qwen2.5-VL as text encoder.
# ============================================================================

def get_sharded_data_with_replication(data, dim, num_heads, tp_degree):
    """
    Shard data with head replication when num_heads < tp_degree.

    For GQA models where num_kv_heads < tp_degree, we replicate KV heads
    so each rank gets a copy.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()

    if num_heads >= tp_size:
        return get_sharded_data(data, dim)
    else:
        replication_factor = tp_size // num_heads
        original_head_idx = tp_rank // replication_factor
        head_dim = data.shape[dim] // num_heads
        if dim == 0:
            start = original_head_idx * head_dim
            end = (original_head_idx + 1) * head_dim
            return data[start:end].clone()
        elif dim == 1:
            start = original_head_idx * head_dim
            end = (original_head_idx + 1) * head_dim
            return data[:, start:end].clone()


def shard_qwen2_attention(tp_degree: int, self_attn):
    """
    Shard Qwen2/Qwen2.5-VL self attention module (used in language model).

    Handles GQA: num_heads=28, num_key_value_heads=4
    With TP=4: Q=28/4=7 heads/rank, KV=4/4=1 head/rank (perfect alignment)
    """
    orig_q = self_attn.q_proj
    orig_k = self_attn.k_proj
    orig_v = self_attn.v_proj
    orig_o = self_attn.o_proj

    num_kv_heads = getattr(self_attn, 'num_key_value_heads', self_attn.num_heads)
    num_q_heads = self_attn.num_heads

    kv_replicate_mode = num_kv_heads < tp_degree

    # Calculate padded Q heads
    extra_q_heads = get_number_of_extra_heads(num_q_heads, tp_degree)
    total_padded_q_heads = num_q_heads + extra_q_heads
    q_head_dim = orig_q.out_features // num_q_heads
    padded_q_out_features = total_padded_q_heads * q_head_dim

    # Update heads per rank
    self_attn.num_heads = neuronx_dist_utils.divide(total_padded_q_heads, tp_degree)
    if hasattr(self_attn, 'num_key_value_heads'):
        if kv_replicate_mode:
            self_attn.num_key_value_heads = 1
        else:
            self_attn.num_key_value_heads = self_attn.num_key_value_heads // tp_degree

    if hasattr(self_attn, 'num_key_value_groups'):
        self_attn.num_key_value_groups = self_attn.num_heads // self_attn.num_key_value_heads

    # Shard Q (with padding if needed)
    q_weight_padded = orig_q.weight.data
    q_bias_padded = orig_q.bias.data if orig_q.bias is not None else None

    if extra_q_heads > 0:
        padding_size = extra_q_heads * q_head_dim
        q_weight_padding = torch.zeros(
            (padding_size, orig_q.in_features), dtype=orig_q.weight.dtype, device=orig_q.weight.device)
        q_weight_padded = torch.cat([orig_q.weight.data, q_weight_padding], dim=0)
        if orig_q.bias is not None:
            q_bias_padding = torch.zeros(padding_size, dtype=orig_q.bias.dtype, device=orig_q.bias.device)
            q_bias_padded = torch.cat([orig_q.bias.data, q_bias_padding], dim=0)

    self_attn.q_proj = ColumnParallelLinear(
        orig_q.in_features, padded_q_out_features,
        bias=(orig_q.bias is not None), gather_output=False, dtype=torch.bfloat16)
    self_attn.q_proj.weight.data = get_sharded_data(q_weight_padded, 0)
    if orig_q.bias is not None:
        self_attn.q_proj.bias.data = get_sharded_data(q_bias_padded, 0)
    del orig_q

    # Shard K/V
    kv_head_dim = orig_k.out_features // num_kv_heads

    if kv_replicate_mode:
        kv_out = kv_head_dim
        self_attn.k_proj = nn.Linear(orig_k.in_features, kv_out, bias=(orig_k.bias is not None), dtype=torch.bfloat16)
        self_attn.k_proj.weight.data = get_sharded_data_with_replication(orig_k.weight.data, 0, num_kv_heads, tp_degree)
        if orig_k.bias is not None:
            self_attn.k_proj.bias.data = get_sharded_data_with_replication(orig_k.bias.data, 0, num_kv_heads, tp_degree)

        self_attn.v_proj = nn.Linear(orig_v.in_features, kv_out, bias=(orig_v.bias is not None), dtype=torch.bfloat16)
        self_attn.v_proj.weight.data = get_sharded_data_with_replication(orig_v.weight.data, 0, num_kv_heads, tp_degree)
        if orig_v.bias is not None:
            self_attn.v_proj.bias.data = get_sharded_data_with_replication(orig_v.bias.data, 0, num_kv_heads, tp_degree)
    else:
        self_attn.k_proj = ColumnParallelLinear(
            orig_k.in_features, orig_k.out_features,
            bias=(orig_k.bias is not None), gather_output=False, dtype=torch.bfloat16)
        self_attn.k_proj.weight.data = get_sharded_data(orig_k.weight.data, 0)
        if orig_k.bias is not None:
            self_attn.k_proj.bias.data = get_sharded_data(orig_k.bias.data, 0)

        self_attn.v_proj = ColumnParallelLinear(
            orig_v.in_features, orig_v.out_features,
            bias=(orig_v.bias is not None), gather_output=False, dtype=torch.bfloat16)
        self_attn.v_proj.weight.data = get_sharded_data(orig_v.weight.data, 0)
        if orig_v.bias is not None:
            self_attn.v_proj.bias.data = get_sharded_data(orig_v.bias.data, 0)

    del orig_k, orig_v

    # Shard O projection
    o_weight_padded = orig_o.weight.data
    if extra_q_heads > 0:
        padding_size = extra_q_heads * q_head_dim
        o_weight_padding = torch.zeros(
            (orig_o.out_features, padding_size), dtype=orig_o.weight.dtype, device=orig_o.weight.device)
        o_weight_padded = torch.cat([orig_o.weight.data, o_weight_padding], dim=1)

    self_attn.o_proj = RowParallelLinear(
        padded_q_out_features, orig_o.out_features,
        bias=(orig_o.bias is not None), input_is_parallel=True, dtype=torch.bfloat16)
    self_attn.o_proj.weight.data = get_sharded_data(o_weight_padded, 1)
    if orig_o.bias is not None:
        self_attn.o_proj.bias.data = orig_o.bias.data.detach()
    del orig_o

    return self_attn


def shard_qwen2_mlp(mlp):
    """Shard Qwen2 MLP (gate_proj, up_proj, down_proj)."""
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features, orig_gate.out_features,
        bias=(orig_gate.bias is not None), gather_output=False, dtype=torch.bfloat16)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features, orig_up.out_features,
        bias=(orig_up.bias is not None), gather_output=False, dtype=torch.bfloat16)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    mlp.down_proj = RowParallelLinear(
        orig_down.in_features, orig_down.out_features,
        bias=(orig_down.bias is not None), input_is_parallel=True, dtype=torch.bfloat16)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp


def shard_vision_attention_fp32(tp_degree: int, attn):
    """
    Shard Qwen2.5-VL Vision Encoder attention (fused QKV + proj).
    Float32 for accuracy. TP=4: 3840/4=960 (divisible).
    """
    orig_qkv = attn.qkv
    orig_proj = attn.proj

    original_num_heads = attn.num_heads
    attn.num_heads = original_num_heads // tp_degree

    attn.qkv = ColumnParallelLinear(
        orig_qkv.in_features, orig_qkv.out_features,
        bias=(orig_qkv.bias is not None), gather_output=False, dtype=torch.float32)
    attn.qkv.weight.data = get_sharded_data(orig_qkv.weight.data, 0)
    if orig_qkv.bias is not None:
        attn.qkv.bias.data = get_sharded_data(orig_qkv.bias.data, 0)
    del orig_qkv

    attn.proj = RowParallelLinear(
        orig_proj.in_features, orig_proj.out_features,
        bias=(orig_proj.bias is not None), input_is_parallel=True, dtype=torch.float32)
    attn.proj.weight.data = get_sharded_data(orig_proj.weight.data, 1)
    if orig_proj.bias is not None:
        attn.proj.bias.data = orig_proj.bias.data.detach()
    del orig_proj

    return attn


def shard_vision_mlp_fp32(mlp):
    """
    Shard Qwen2.5-VL Vision MLP (SwiGLU).
    Float32 for accuracy. intermediate_size=3420, 3420/4=855 (divisible).
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features, orig_gate.out_features,
        bias=(orig_gate.bias is not None), gather_output=False, dtype=torch.float32)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features, orig_up.out_features,
        bias=(orig_up.bias is not None), gather_output=False, dtype=torch.float32)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    mlp.down_proj = RowParallelLinear(
        orig_down.in_features, orig_down.out_features,
        bias=(orig_down.bias is not None), input_is_parallel=True, dtype=torch.float32)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp
