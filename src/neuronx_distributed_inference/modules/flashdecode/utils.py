from typing import Tuple

import torch
from neuronx_distributed.parallel_layers.utils import divide

# Theoretically one should enough to avoid writing true value to garbage pos (last pos of the cache),
# pick 128 as is more compatible with compiler
EXTRA_RESERVED_SPACE = 128


def get_cache_size(seq_len, num_cores_per_group, is_ctx=False):
    """extra reserved space is only needed in decoding phase"""
    per_core_sz = divide(seq_len, num_cores_per_group)
    return per_core_sz if is_ctx else per_core_sz + EXTRA_RESERVED_SPACE


def turn_2d_mask_to_4d(attention_mask, n_positions, batch_size):
    return attention_mask[:, None, None, :].expand(batch_size, 1, 1, n_positions).to(torch.bool)


def calculate_num_cores_per_group(num_attn_heads, num_kv_heads, tp_degree):
    assert num_attn_heads % tp_degree == 0, (
        f"expect num attention heads is multiples of tp degree but got "
        f"{num_attn_heads} and {tp_degree}"
    )
    num_cores_per_group = divide(min(tp_degree, num_attn_heads), num_kv_heads)
    return num_cores_per_group


def mask_util(
    pos_ids: torch.Tensor, rank_id: torch.Tensor, num_cores_per_group: int, cache_size: int
) -> (Tuple)[torch.Tensor, torch.Tensor]:
    """
    @:param pos_ids: 2d [bsz x n_active_tokens] tensor represents position ids for all sequences in a batch
    @:param rank_id: current rank of the device
    @:return num_cores_per_group: number of cores per kv group
    @:param cache_size: size of the cache per core
    """
    assert pos_ids.dim() == 2, f"position ids have to be 2D for shape {pos_ids.shape}"
    batch_sz, n_active_tokens = pos_ids.shape

    # Core layout: 32 cores on 8 kv group (col) and 4 cores in each group
    #     0, 1, 2, 3, 4, 5, 6, 7
    # -------------------------------
    # 0 | 0, 4, 8, 12, 16, 20, 24, 28
    # 1 | 1, 5, 9, 13, 17, 21, 25, 29
    # 2 | 2, 6, 10, 14, 18, 22, 26, 30
    # 3 | 3, 7, 11, 15, 19, 23, 27, 31
    # -------------------------------
    # for rank id == 19:
    # the rank_id_in_kv_group (row index) is 3, derived by 19 % 4

    rank_id = torch.remainder(rank_id, num_cores_per_group)

    # active masks: select only one core to update active KV
    selected_core_idx = torch.remainder(pos_ids, num_cores_per_group)
    active_masks = torch.where(selected_core_idx == rank_id, 1, 0).to(dtype=pos_ids.dtype)
    if n_active_tokens > 1:  # speculation
        active_masks_causal = torch.full(
            (n_active_tokens, n_active_tokens),
            1,
            device=pos_ids.device,
        ).tril(diagonal=0)
        active_masks_causal = active_masks_causal[None, :, :].expand(
            batch_sz, n_active_tokens, n_active_tokens
        )
        active_masks = active_masks[:, None, :].expand(batch_sz, n_active_tokens, n_active_tokens)
        active_masks = torch.logical_and(active_masks, active_masks_causal).to(dtype=pos_ids.dtype)

    # prior masks: infer and update it

    # Cache layout within 1 kv group: 4 cores (row) and each has 8 positions (col), that is cache_size=8
    # Note num of positions = bucket_sz//num_cores_per_kv_group
    #     0, 1, 2, 3, 4, 5, 6, 7
    # -------------------------------
    # 0 | 0, 4, 8, 12, 16, 20, 24, 28
    # 1 | 1, 5, 9, 13, 17, 21, 25, 29
    # 2 | 2, 6, 10, 14, 18, 22, 26, 30
    # 3 | 3, 7, 11, 15, 19, 23, 27, 31
    # -------------------------------
    # for pos_id = 19:
    # the selected_pos for prior masks to be updated (col index) is 4, derived by 19 // 4

    # selected_pos = torch.div(pos_ids, num_cores_per_group, rounding_mode="floor")
    num_processed_tokens = (
        pos_ids.min(dim=-1, keepdim=True).values if n_active_tokens > 1 else pos_ids
    )
    selected_pos = torch.div(
        torch.subtract(torch.add(num_processed_tokens, num_cores_per_group - 1), rank_id),
        num_cores_per_group,
        rounding_mode="floor",
    )
    mask_shape = (
        (batch_sz, n_active_tokens, cache_size) if n_active_tokens > 1 else (batch_sz, cache_size)
    )
    # init prior mask: set True from the start to the selected_pos, and the rest False
    position_ids_to_compare = (
        selected_pos.unsqueeze(-1).expand(mask_shape)
        if n_active_tokens > 1
        else selected_pos.expand(mask_shape)
    )
    mask = torch.arange(cache_size, device=pos_ids.device).expand(mask_shape)
    prior_masks = torch.where(position_ids_to_compare > mask, 1, 0).to(dtype=pos_ids.dtype)

    return active_masks, prior_masks
