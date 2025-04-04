from typing import List

import torch
import torch.nn.functional as F
from neuronx_distributed.utils import cpu_mode
from torch import Tensor
from torch_neuronx.xla_impl.ops import xla_hlo_call

from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum


def fill_prefix(cache, prefix_cache):
    if cpu_mode():
        cache[[slice(0, s) for s in prefix_cache.shape]] = prefix_cache
        return cache
    else:

        @xla_hlo_call
        def xla_fill_prefix(tensor, update):
            scribe = tensor.scribe
            dtype = tensor.dtype
            shape = tensor.sizes
            start_indices = [scribe.u32.Constant(constant_value=0)] * len(shape)
            return dtype[shape].DynamicUpdateSlice(tensor, update, *start_indices)

        return xla_fill_prefix(cache, prefix_cache)


def dynamic_update_slice(
    tensor: torch.Tensor, update: torch.Tensor, start_indices: List[torch.Tensor]
):
    """
    Directly invoke DynamicUpdateSlice XLA op
    https://openxla.org/xla/operation_semantics#dynamicupdateslice
    """

    if cpu_mode():
        raise NotImplementedError("dynamic_update_slice() is not implemented for CPU mode")

    @xla_hlo_call
    def xla_dynamic_update_slice(tensor, update, *start_indices):
        dtype = tensor.dtype
        shape = tensor.sizes
        return dtype[shape].DynamicUpdateSlice(tensor, update, *start_indices)

    assert len(start_indices) == tensor.dim(), "not enough indices to index into tensor"
    return xla_dynamic_update_slice(tensor, update, *start_indices)


def get_active_block_table(
    block_table: Tensor,
    context_lens: Tensor,
    block_size: int,
):
    """
    Get a block table of active KV cache blocks, with padding only at the end.

    The original block table input param from vLLM is padded for each sequence,
    so it is not only padded at the end, but also in between. This function is
    to clean those padding, and only choose necessary KV cache blocks for
    current request. So we don't need to fetch a number of KV cache
    blocks that are not needed for attention computation.

    This function is meant to be called outside of NeuronBaseModel, so there
    is no requirement of fixed shape.

    Example:
        Inputs:
            block_tables: [[149, 143], [148,   0], [147, 146], [145,   0]]
            context_lens: [  6,  16, 170,   6]
            block_size: 128

        Expected Outputs:
            active_table:
            [149, 148, 147, 146, 145]

    Args:
        block_table: the original input param block_table from vllm.
        context_lens: the length of KV cache per sequence, excluding the KV
            states from current request.
        block_size: the size of a KV cache block, to be provided by users of
            vLLM.

    Returns:
        active_table: a block table to hold effective KV cache block id, whose
            length is the same as num_active_blocks. The active_table will be
            an empty tensor([]) if there is no active blocks needed.
    """

    assert len(block_table.shape) == 2
    assert len(context_lens.shape) == 1
    max_num_seqs, _ = block_table.shape
    assert max_num_seqs == len(context_lens)

    active_table = []
    for seq_id in range(max_num_seqs):
        context_for_seq = context_lens[seq_id]
        blocks_for_seq = block_table[seq_id, :]
        num_active_blocks_for_seq = torch.ceil(context_for_seq / block_size).int()
        active_table.append(blocks_for_seq[:num_active_blocks_for_seq])

    active_table = torch.cat(active_table)
    return active_table


def contexted_kv(
    cache,
    current,
    cache_mask,
    cache_reordered_idx,
    current_reordered_idx,
):
    """
    Combine KV cache and KV output for current posistion into one.

    We need to call contexted_kv_indexing() to get necessary input params (
    index and mask) for this function.

    This is needed for chunked prefill: in attention module, Q needs to
    attend to all K for a sequence.

    Args:
        cache: KV cache in block layout.
        current: KV computed in current step in BHSD layout.
        cache_mask: Binary array to indicate needed KV cache.
        cache_reordered_idx: index array used to retrieve KV from cache.
        current_reordered_idx: index array used to retrieve KV from current.

    Returns:
        combined_ctx: KV that will be used later for context encoding.

    """
    cache_and_current_len = cache_reordered_idx.shape[0]
    # cache is in block layout
    num_blocks, block_size, num_heads, head_dim = cache.shape
    # current output is in BHSD layout
    batch_size, _, seq_len, _ = current.shape
    size = [1, cache_and_current_len, num_heads, head_dim]

    cache = cache.reshape(num_blocks * block_size, num_heads * head_dim)
    cache = torch.index_select(cache, dim=0, index=cache_reordered_idx.int())

    current = current.permute((0, 2, 1, 3))  # BHSD -> BSHD
    current = current.reshape(batch_size * seq_len, num_heads * head_dim)
    current = torch.index_select(current, dim=0, index=current_reordered_idx.int())

    cache_mask = cache_mask.reshape(-1, 1)
    combined_ctx = torch.where(cache_mask, cache, current)
    combined_ctx = combined_ctx.reshape(size)  # BSHD
    combined_ctx = combined_ctx.permute((0, 2, 1, 3))  # BSHD -> BHSD
    return combined_ctx


def contexted_kv_v2(
    cache,
    current,
    cache_mask,
    current_reordered_idx,
):
    """
    Combine KV cache and KV output for current posistion into one.

    We need to call contexted_kv_indexing_v2() to get necessary input params (
    index and mask) for this function.

    This is needed for chunked prefill: in attention module, Q needs to
    attend to all K for a sequence.

    Args:
        cache: KV cache in BHSD layout.
        current: KV computed in current step in shape of (batch_size, num_heads, n_active_tokens, head_dim).
        cache_mask: boolean array to indicate needed KV cache in shape (batch_size, seq_len).
        current_reordered_idx: index array used to retrieve KV from current in shape (batch_size, seq_len).

    Returns:
        combined_ctx: KV that will be used later for context encoding in BHSD layout.

    """
    batch_size, num_heads, seq_len, head_dim = cache.shape
    _, _, n_active_tokens, _ = current.shape

    dtype = current_reordered_idx.dtype
    device = current_reordered_idx.device
    current_reordered_idx = torch.where(
        cache_mask,
        current_reordered_idx,
        current_reordered_idx
        + torch.arange(batch_size, dtype=dtype, device=device).unsqueeze(-1) * seq_len,
    )

    cache = cache.permute(0, 2, 1, 3)  # BHSD -> BSHD
    cache = cache.reshape(batch_size * seq_len, num_heads * head_dim)

    current = current.permute((0, 2, 1, 3))  # BHSD -> BSHD
    # For token gen, we will need to pad the current KV to have the same shape as cache
    current = F.pad(input=current, pad=(0, 0, 0, 0, 0, seq_len - n_active_tokens))
    current = current.reshape(batch_size * seq_len, num_heads * head_dim)
    current = torch.index_select(
        input=current, dim=0, index=current_reordered_idx.reshape(-1).int()
    )

    cache_mask = cache_mask.reshape(-1, 1)
    combined_ctx = torch.where(cache_mask, cache, current)
    combined_ctx = combined_ctx.reshape(batch_size, seq_len, num_heads, head_dim)  # BSHD
    combined_ctx = combined_ctx.permute((0, 2, 1, 3))  # BSHD -> BHSD
    return combined_ctx


def contexted_kv_indexing(
    new_lens,
    all_lens,
    max_total_len,
    block_size,
):
    """
    Prepare index and mask to combine KV cache and KV output for current
    posisiton into one.

    This function prepares necessary input params for the combination, and the
    combination is actually done in contexted_kv() function.

    Example:
        new_lens: [3,2,1,0]
        all_lens: [5,7,5,0]
        max_total_len: 20
        block_size: 4

        cache_mask:
            [1, 1, x, x, x, 1, 1, 1, 1, 1, x, x,  1,  1,  1,  1, x, x, x, x]
        cache_reordred_idx:
            [0, 1, x, x, x, 4, 5, 6, 7, 8, x, x, 12, 13, 14, 15, x, x, x, x]
        current_reordered_idx:
            [x, x, 0, 1, 2, x, x, x, x, x, 3, 4,  x,  x,  x,  x, 5, x, x, x]

    Args:
        new_lens: the length of new KV states (derived from current request)
            for each sequence
        all_lens: the length of KV cache and new KV for each sequence
        max_total_len: the max total length of KV which includes KV cache and
            new KV for current request
        block_size: size of a KV cache block

    Returns:
        cache_mask: a list of bool to indicate if its posistion is a KV cache
            from previous context
        cache_reordered_idx: a list of indices to re-order cache, which can
            be used to put cache in the expected positions for each sequence
        current_reordered_idx: a list of indices to re-order new KV states,
            which is used to put new KV states in the expected positions for
            each sequence
    """
    batch_size = new_lens.shape[0]
    dtype = new_lens.dtype
    device = new_lens.device
    # num of states from previous cache
    old_lens = all_lens - new_lens
    # start id in the combined KV for each seq (in combined KV, there is
    # padding only at the end)
    all_cumsum = neuron_cumsum(all_lens.reshape(1, -1).float()).flatten().int()
    all_cumsum = F.pad(all_cumsum, pad=[1, 0])
    # start id in the query for each seq
    new_cumsum = neuron_cumsum(new_lens.reshape(1, -1).float()).flatten().int()
    new_cumsum = F.pad(new_cumsum, pad=[1, 0])

    # start id in the combined KV for each seq
    all_cumsum = all_cumsum[:batch_size]
    new_steps = all_cumsum + old_lens

    num_block_per_seq = torch.ceil(old_lens / block_size).int()
    num_block_per_seq = F.pad(num_block_per_seq, pad=[1, 0])
    # block start id for each seq
    block_start_idx = neuron_cumsum(num_block_per_seq.reshape(1, -1).float()).flatten().int()
    # cache start id for each seq
    old_start = block_start_idx * block_size

    cache_mask = torch.zeros(max_total_len, dtype=torch.bool, device=device)
    cache_reordered_idx = torch.zeros(max_total_len, dtype=dtype, device=device)
    current_reordered_idx = torch.zeros(max_total_len, dtype=dtype, device=device)

    idx = torch.arange(max_total_len, dtype=dtype, device=device)
    for seq_id in range(batch_size):
        cache_reordered_idx, mask = _selective_masking(
            all_cumsum[seq_id], old_start[seq_id], old_lens[seq_id], idx, cache_reordered_idx
        )
        current_reordered_idx, _ = _selective_masking(
            new_steps[seq_id], new_cumsum[seq_id], new_lens[seq_id], idx, current_reordered_idx
        )
        cache_mask = torch.logical_or(mask, cache_mask)

    return cache_mask, cache_reordered_idx, current_reordered_idx


def _selective_masking(loc, start, length, idx, x_to_ctx):
    x = idx - (loc - start)
    upper_bound = start + length - 1
    x = torch.minimum(upper_bound, torch.maximum(start, x))

    left_bound = loc + length - 1
    left_mask = left_bound >= idx
    right_mask = loc <= idx
    mask = torch.logical_and(left_mask, right_mask)
    x_to_ctx = torch.where(mask, x, x_to_ctx)
    return x_to_ctx, mask


def contexted_kv_indexing_v2(q_lens, k_lens, max_seq_len) -> torch.Tensor:
    """
    Generate the precomputed cache mask.

    Example:
        q_lens: [3,2,0]
        k_lens: [5,7,0]
        max_seq_len: 10

        cache_mask:
            [[1, 1, x, x, x, x, x, x, x, x],
             [1, 1, 1, 1, 1, x, x, x, x, x],
             [x, x, x, x, x, x, x, x, x, x]]
        current_reordered_idx:
            [[x, x, 0, 1, 2, x, x, x, x, x],
             [x, x, x, x, x, 0, 1, x, x, x],
             [x, x, x, x, x, x, x, x, x, x]]

    Args:
        q_lens: the length of the queries in shape (batch_size, )
        k_lens: the length of the keys in shape (batch_size, )
        max_seq_len: the maximum length of the sequence

    Returns:
        cache_mask: the precomputed cache mask in shape (batch_size, max_seq_len)
        current_reordered_idx: the scatter indices mapping the newly computed KV
            to full-length KV in shape (batch_size, max_seq_len)
    """
    batch_size = q_lens.shape[0]
    dtype = q_lens.dtype
    device = q_lens.device

    cache_lens = k_lens - q_lens
    mask = torch.arange(max_seq_len, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1)
    cache_mask = mask.lt(cache_lens.unsqueeze(-1))
    current_reordered_idx = torch.where(cache_mask, 0, mask - cache_lens.unsqueeze(-1))
    current_reordered_idx = torch.where(mask.lt(k_lens.unsqueeze(-1)), current_reordered_idx, 0)

    return cache_mask, current_reordered_idx


def contexted_kv_indexing_dynamic(
    q_lens,
    k_lens,
    block_size,
):
    """
    Another impl of contexted_kv_indexing with dynamic shape.

    The length of the output indices are the sum of k_lens.
    """
    num_seqs = q_lens.shape[0]
    dtype = q_lens.dtype
    device = q_lens.device

    cache_mask = []
    for seq_id in range(num_seqs):
        num_q = q_lens[seq_id]
        num_k = k_lens[seq_id]
        num_cache = num_k - num_q

        cache_per_seq = torch.ones(num_cache, dtype=dtype, device=device)
        active_per_seq = torch.zeros(num_q, dtype=dtype, device=device)
        cache_mask_per_seq = torch.cat([cache_per_seq, active_per_seq], dim=0)
        cache_mask.append(cache_mask_per_seq)
    cache_mask = torch.cat(cache_mask, dim=0).bool()

    actual_len = cache_mask.shape[0]
    active_to_ctx = torch.zeros(actual_len, dtype=dtype, device=device)
    cnt = 0
    for i in range(actual_len):
        if not cache_mask[i]:
            active_to_ctx[i] += cnt
            cnt += 1

    cached_to_ctx = []
    cache = k_lens - q_lens
    num_blocks_per_seq = torch.ceil(cache / block_size)
    num_blocks_per_seq = F.pad(num_blocks_per_seq, [1, 0])
    num_blocks_per_seq = torch.cumsum(num_blocks_per_seq, dim=0)
    for seq_id in range(num_seqs):
        num_q = q_lens[seq_id]
        num_k = k_lens[seq_id]
        num_cache = num_k - num_q

        if num_cache != 0:
            # fill for cache id
            start_block_id = num_blocks_per_seq[seq_id]
            start_slot_id = start_block_id * block_size
            cache_per_seq = torch.arange(
                start_slot_id, start_slot_id + num_cache, dtype=dtype, device=device
            )
            cached_to_ctx.append(cache_per_seq)
        cached_to_ctx.append(torch.zeros(num_q, dtype=dtype, device=device))
    cached_to_ctx = torch.cat(cached_to_ctx, dim=0)
    return cache_mask, cached_to_ctx, active_to_ctx
