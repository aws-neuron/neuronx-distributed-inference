import torch
from torch import Tensor


def get_active_block_table(
    block_table: Tensor,
    seq_lens: Tensor,
    num_active_blocks: Tensor,
    block_size: Tensor,
):
    """
    Get a block table of active KV cache blocks, with padding at the end only.

    The original block table input param from vLLM is padded for each sequence, 
    so it is not only padded at the end, but also in between. This function is
    to clean those padding, so we don't need to fetch a number of KV cache 
    blocks that are not needed for attention computation.

    Example: 
        Inputs:
            block_tables: [[149,   0], [148,   0], [147, 146], [145,   0]]
            seq_lens: [[  6], [ 16], [170], [  6]]
            num_active_blocks: 6

        Expected Outputs:
            active_table:
            [149, 148, 147, 146, 145, 0]
    
    Args:
        block_table: the original input param block_table from vllm, where there
            could be paddings in between.
        seq_lens: the length of each sequence in a batch request.
        num_active_blocks: the max number of blocks to hold the effective KV
            cache blocks, and usually it is decided by bucket size.
        block_size: the size of a KV cache block, to be provided by users of 
            vLLM.
    
    Returns:
        active_block: a block table to hold effective KV cache block id, whose
            length is the same as num_active_blocks.
    """

    batch_size, max_num_blocks_per_seq = block_table.shape
    # assert len(seq_lens.shape) == 2, "seq_lens is expected to be a 2D tensor"
    # assert batch_size == seq_lens.shape[0]

    active_table = torch.zeros(num_active_blocks, dtype=block_table.dtype)

    block_table = block_table.reshape(batch_size * max_num_blocks_per_seq)
    seq_lens = seq_lens.reshape(batch_size)

    num_blocks_per_seq = torch.ceil(seq_lens / block_size)
    blocks_cumsum = torch.cumsum(num_blocks_per_seq, dim=0)
    start_block_id_list = torch.cat([torch.tensor([0]), blocks_cumsum])

    seq_id_list = torch.arange(batch_size)
    
    for block_id in torch.arange(num_active_blocks):
        seq_mask = blocks_cumsum <= block_id
        seq_id = torch.minimum(
            torch.max(seq_mask*(seq_id_list+1)),
            torch.tensor(batch_size-1)
        )
        seq_start_block_id = start_block_id_list[seq_id]
        offset = block_id - seq_start_block_id
        block_table_id = torch.minimum(
            seq_id * max_num_blocks_per_seq + offset, 
            torch.tensor(block_table.numel()-1)
        )
        block_table_id = block_table_id.to(torch.int)

        active_table[block_id] = block_table[block_table_id]

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
    """
    max_num_ctx = cache_reordered_idx.shape[0]
    # cache is in block layout
    num_blocks, block_size, num_heads, head_dim = cache.shape
    # current output is in BHSD layout
    batch_size, _, seq_len, _ = current.shape
    size = [1, max_num_ctx, num_heads, head_dim]

    cache = cache.reshape(num_blocks*block_size, num_heads*head_dim)
    cache = torch.index_select(cache, dim=0, index=cache_reordered_idx)
    cache = cache.reshape(size)

    current = current.permute((0,2,1,3)) # BHSD -> BSHD
    current = current.reshape(batch_size*seq_len, num_heads*head_dim)
    current = torch.index_select(current, dim=0, index=current_reordered_idx)
    current = current.reshape(size)

    cache_mask = cache_mask.reshape(size)

    combined_ctx = torch.where(cache_mask, cache, current) # BSHD 
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
        total_key_len: 20
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
        total_len: the max total length of KV which includes KV cache and new 
            KV for current request
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
    # num of states from previous cache
    old_lens = all_lens - new_lens 
    # start id in the combined KV for each seq (in combined KV, there is 
    # padding only at the end)
    all_cumsum = torch.cat([torch.tensor([0]), torch.cumsum(all_lens, dim=0)])
    # start id in the query for each seq
    new_cumsum = torch.cat([torch.tensor([0]), torch.cumsum(new_lens, dim=0)])
    
    # start id in the combined KV for each seq
    all_cumsum = all_cumsum[:batch_size]
    new_steps = all_cumsum + old_lens

    num_block_per_seq = torch.ceil(old_lens / block_size)
    num_block_per_seq = torch.cat([torch.tensor([0]), num_block_per_seq])
    # block start id for each seq
    block_start_idx = torch.cumsum(num_block_per_seq, dim=0)
    # cache start id for each seq
    old_start = block_start_idx * block_size

    cache_mask = torch.zeros(max_total_len, dtype=torch.bool)
    cache_reordered_idx = torch.zeros(max_total_len, dtype=torch.int)
    current_reordered_idx = torch.zeros(max_total_len, dtype=torch.int)

    idx = torch.arange(max_total_len)
    for seq_id in range(batch_size):
        cache_reordered_idx, mask, *_ = _selective_masking(
            all_cumsum[seq_id], old_start[seq_id], old_lens[seq_id], 
            idx, cache_reordered_idx)
        current_reordered_idx, *_ = _selective_masking(
            new_steps[seq_id], new_cumsum[seq_id], new_lens[seq_id], 
            idx, current_reordered_idx)
        cache_mask = torch.logical_or(mask, cache_mask)

    return cache_mask, cache_reordered_idx, current_reordered_idx


def _selective_masking(loc, start, length, idx, x_to_ctx): 
    x = idx - (loc - start)
    upper_bound = start + length - 1
    x = torch.minimum(upper_bound, torch.maximum(start, x))
    
    left_bound = loc + length - 1
    left_mask = (left_bound >= idx)
    right_mask = (loc <= idx)
    mask = torch.logical_and(left_mask, right_mask)
    x_to_ctx = torch.where(mask, x, x_to_ctx)
    return x_to_ctx, mask