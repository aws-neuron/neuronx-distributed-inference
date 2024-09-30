# fmt: off
import torch

from neuronx_distributed_inference.modules.kvcache.utils import (
    contexted_kv,
    contexted_kv_indexing,
    get_active_block_table,
)


# Tests on get_active_block_table()
def test_get_active_block_table_case_1():
    traced_func = prepare_traced_get_active_block_table(
        block_table_shape=[4,2],
        seq_lens_shape=[4,1],
        num_active_block=6,
        block_size=128,
    )

    block_table = torch.tensor([[149, 0], [148, 0], [147, 146], [145, 0]])
    seq_lens = torch.tensor([[  6], [ 16], [170], [  6]])
    num_active_blocks = torch.tensor(6)
    block_size = torch.tensor(128)

    actual = traced_func(block_table, seq_lens, num_active_blocks, block_size)

    expected = torch.tensor([149, 148, 147, 146, 145, 0])

    assert torch.equal(actual, expected)


def test_get_active_block_table_case_2():
    traced_func = prepare_traced_get_active_block_table(
        block_table_shape=[4,2],
        seq_lens_shape=[4,1],
        num_active_block=6,
        block_size=128,
    )

    block_table = torch.tensor([[123, 128], [148, 0], [147, 146], [163, 0]])
    seq_lens = torch.tensor([[  190], [ 16], [170], [  6]])
    num_active_blocks = torch.tensor(6)
    block_size = torch.tensor(128)

    actual = traced_func(block_table, seq_lens, num_active_blocks, block_size)

    expected = torch.tensor([123, 128, 148, 147, 146, 163])

    assert torch.equal(actual, expected)


def test_get_active_block_table_case_3():
    traced_func = prepare_traced_get_active_block_table(
        block_table_shape=[3,3],
        seq_lens_shape=[3,1],
        num_active_block=8,
        block_size=24,
    )

    block_table = torch.tensor([[123, 128, 175], [148, 0, 0], [147, 146, 0]])
    seq_lens = torch.tensor([[  53], [ 16], [36]])
    num_active_blocks = torch.tensor(8)
    block_size = torch.tensor(24)

    actual = traced_func(block_table, seq_lens, num_active_blocks, block_size)

    expected = torch.tensor([123, 128, 175, 148, 147, 146, 0, 0])

    assert torch.equal(actual, expected)


def prepare_traced_get_active_block_table(
    block_table_shape,
    seq_lens_shape,
    num_active_block,
    block_size,
):
    """Need to test the function in a traced format"""
    example_inputs = (
        torch.zeros(block_table_shape),
        torch.zeros(seq_lens_shape),
        torch.tensor(num_active_block),
        torch.tensor(block_size),
    )
    traced_func = torch.jit.trace(get_active_block_table, example_inputs)
    return traced_func


# Tests on contexted_kv()
def test_contexted_kv_case_1():
    x = 0
    cache = torch.arange(start=10, end=34).reshape(6,4,1,1) # block layout
    current = torch.arange(8).reshape(1, 1, 8, 1) # BHSD
    cache_mask = torch.tensor(
        [1, 1, x, x, x, 1, 1, 1, 1, 1, x, x,  1,  1,  1,  1, x, x, x, x], dtype=torch.bool)
    cache_reordered_idx = torch.tensor(
        [0, 1, x, x, x, 5, 6, 7, 8, 9, x, x, 12, 13, 14, 15, x, x, x, x], dtype=torch.int)
    current_reordered_idx = torch.tensor(
        [x, x, 0, 1, 2, x, x, x, x, x, 3, 4,  x,  x,  x,  x, 5, x, x, x], dtype=torch.int)

    traced_func = prepare_traced_contexted_kv(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        cache_reordered_idx_shape=cache_reordered_idx.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, cache_reordered_idx, current_reordered_idx)

    expected = torch.tensor([
        10, 11,             # 2 cache for seq 0
         0,  1,  2,         # 3 current for seq 0
        15, 16, 17, 18, 19, # 5 cache for seq 1
         3,  4,             # 2 current for seq 1
        22, 23, 24, 25,     # 4 cache for seq 2
         5,                 # 1 curent for seq 2
         x,  x,  x,
    ])
    assert actual.shape == (1, 20, 1, 1)
    actual = actual.flatten()
    assert torch.equal(actual, expected)


def test_contexted_kv_case_2():
    x = 0
    cache = torch.arange(start=20, end=32).reshape(3,4,1,1) # block layout
    current = torch.arange(6).reshape(1, 1, 6, 1) # BHSD
    cache_mask =           torch.tensor([x, x, x, x, x, 1, 1, 1, 1, 1, x, x], dtype=torch.bool)
    cache_reordered_idx =   torch.tensor([x, x, x, x, x, 0, 1, 2, 3, 4, x, x], dtype=torch.int)
    current_reordered_idx = torch.tensor([0, 1, 2, 3, 4, x, x, x, x, x, 5, x], dtype=torch.int)

    traced_func = prepare_traced_contexted_kv(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        cache_reordered_idx_shape=cache_reordered_idx.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, cache_reordered_idx, current_reordered_idx)

    expected = torch.tensor([
                            # 0 cache for seq 0
         0,  1,  2,  3,  4, # 5 current for seq 0
        20, 21, 22, 23, 24, # 5 cache for seq 1
         5,                 # 1 current for seq 1
        x,                  # padding
    ])
    assert actual.shape == (1, 12, 1, 1)
    actual = actual.flatten()
    assert torch.equal(actual, expected)


def prepare_traced_contexted_kv(
    cache_shape,
    current_shape,
    cache_mask_shape,
    cache_reordered_idx_shape,
    current_reordered_idx_shape,
):
    example_inputs = (
        torch.zeros(cache_shape, dtype=torch.float),
        torch.zeros(current_shape, dtype=torch.float),
        torch.zeros(cache_mask_shape, dtype=torch.bool),
        torch.zeros(cache_reordered_idx_shape, dtype=torch.int),
        torch.zeros(current_reordered_idx_shape, dtype=torch.int),
    )

    traced_func = torch.jit.trace(contexted_kv, example_inputs)
    return traced_func


# Tests on contexted_kv_indexing()
def test_contexted_kv_indexing_case_1():
    new_lens = torch.tensor([3,2,1,0])
    all_lens = torch.tensor([5,7,5,0])
    max_total_len = torch.tensor(20)
    block_size = torch.tensor(4)

    traced_func = prepare_traced_contexted_kv_indexing(
        new_lens.shape, all_lens.shape, max_total_len, block_size
    )
    actual = traced_func(new_lens, all_lens, max_total_len, block_size)
    # actual = contexted_kv_indexing(query_lens, key_lens, max_total_key_len, block_size)
    actual_cache_mask, actual_cache_reordred_idx, actual_current_reordered_idx = actual

    x = 0
    expected_cache_mask = torch.tensor(
        [1, 1, x, x, x, 1, 1, 1, 1, 1, x, x,  1,  1,  1,  1, x, x, x, x], dtype=torch.bool)
    expected_cache_reordered_idx = torch.tensor(
        [0, 1, x, x, x, 4, 5, 6, 7, 8, x, x, 12, 13, 14, 15, x, x, x, x], dtype=torch.int)
    expected_current_reordered_idx = torch.tensor(
        [x, x, 0, 1, 2, x, x, x, x, x, 3, 4,  x,  x,  x,  x, 5, x, x, x], dtype=torch.int)

    assert torch.equal(actual_cache_mask, expected_cache_mask)
    assert torch.equal(actual_cache_reordred_idx, expected_cache_reordered_idx)
    assert torch.equal(actual_current_reordered_idx, expected_current_reordered_idx)


def test_contexted_kv_indexing_case_2():
    new_lens = torch.tensor([1,2,3,1,0])
    all_lens = torch.tensor([5,7,5,4,0])
    max_total_len = torch.tensor(21)
    block_size = torch.tensor(3)

    traced_func = prepare_traced_contexted_kv_indexing(
        new_lens.shape, all_lens.shape, max_total_len, block_size
    )
    actual = traced_func(new_lens, all_lens, max_total_len, block_size)
    # actual = contexted_kv_indexing(query_lens, key_lens, max_total_key_len, block_size)
    actual_cache_mask, actual_cache_reordred_idx, actual_current_reordered_idx = actual

    x = 0
    expected_cache_mask = torch.tensor(
        [1, 1, 1, 1, x, 1, 1, 1, 1,  1, x, x,  1,  1, x, x, x,  1,  1,  1, x], dtype=torch.bool)
    expected_cache_reordered_idx = torch.tensor(
        [0, 1, 2, 3, x, 6, 7, 8, 9, 10, x, x, 12, 13, x, x, x, 15, 16, 17, x], dtype=torch.int)
    expected_current_reordered_idx = torch.tensor(
        [x, x, x, x, 0, x, x, x, x,  x, 1, 2,  x,  x, 3, 4, 5,  x,  x,  x, 6], dtype=torch.int)

    assert torch.equal(actual_cache_mask, expected_cache_mask)
    assert torch.equal(actual_cache_reordred_idx, expected_cache_reordered_idx)
    assert torch.equal(actual_current_reordered_idx, expected_current_reordered_idx)


def prepare_traced_contexted_kv_indexing(
    new_lens_shape,
    all_lens_shape,
    max_total_len,
    block_size,
):
    example_inputs = (
        torch.zeros(new_lens_shape, dtype=torch.int),
        torch.zeros(all_lens_shape, dtype=torch.int),
        torch.tensor(max_total_len),
        torch.tensor(block_size),
    )
    traced_func = torch.jit.trace(contexted_kv_indexing, example_inputs)
    return traced_func
