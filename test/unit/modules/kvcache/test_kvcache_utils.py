# fmt: off
import unittest
from collections import OrderedDict

import torch
import torch_neuronx
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch_neuronx.xla_impl.ops import ConcatenateOp

from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    _gather_slice_into_kv_cacheline,
    _reshape_tiled_cache,
    _slice_kv_cacheline,
)
from neuronx_distributed_inference.modules.kvcache.utils import (
    contexted_kv,
    contexted_kv_indexing_dynamic,
    contexted_kv_indexing_v2,
    contexted_kv_v2,
    dynamic_update_slice,
    get_active_block_table,
)


# TODO: remove soon and only to demonstrate the equivalence
def _deprecate_slice_rhs(tensor, bucket: int, max_idx: int, dim: int):
    tensor = torch.ops.aten.slice(tensor, dim, max_idx - bucket, max_idx, 1)
    return tensor


def _deprecate_slice_lhs(tensor, bucket: int, dim: int):
    tensor = torch.ops.aten.slice(tensor, dim, 0, bucket, 1)
    return tensor


def _deprecated_slice_kv_cacheline(padding_side, seq_len: int, cache: torch.Tensor):
    if padding_side == "right":
        return _deprecate_slice_lhs(cache, seq_len, 2)  # 0-> seq_len
    max_idx = cache.shape[2]
    return _deprecate_slice_rhs(cache, seq_len, max_idx, 2)


def _deprecate_gather_slice_into_kv_cacheline(
    cache, padding_side, seq_len: int, bucket_slice: torch.Tensor
):
    max_idx = cache.shape[2]
    if padding_side == "right":
        remaining = _deprecate_slice_rhs(cache, max_idx - seq_len, max_idx, dim=2)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([bucket_slice, remaining], dim=2)
    else:
        remaining = _deprecate_slice_lhs(cache, max_idx - seq_len, dim=2)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([remaining, bucket_slice], dim=2)


def test_slice_kv_cacheline():
    cache = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    seq_len = 1

    # 1. right padding
    actual = _slice_kv_cacheline("right", seq_len, cache)
    expected = torch.tensor([[[[1, 2]], [[5, 6]]]])
    assert torch.equal(actual, expected)
    assert torch.equal(_deprecated_slice_kv_cacheline("right", seq_len, cache), expected)

    # 2. left padding
    actual = _slice_kv_cacheline("left", seq_len, cache)
    expected = torch.tensor([[[[3, 4]], [[7, 8]]]])
    assert torch.equal(actual, expected)
    assert torch.equal(_deprecated_slice_kv_cacheline("left", seq_len, cache), expected)


def test_gather_slice_into_kv_cacheline():
    cache = torch.full((1, 1, 4, 1), 50)
    bucket_slice = torch.full((1, 1, 2, 1), 100)
    seq_len = 1

    # 1. right padding
    actual = _gather_slice_into_kv_cacheline(cache, "right", seq_len, bucket_slice)
    expected = torch.tensor([[[[100], [100], [50], [50], [50]]]])
    assert torch.equal(actual, expected)
    assert torch.equal(
        _deprecate_gather_slice_into_kv_cacheline(cache, "right", seq_len, bucket_slice), expected
    )

    # 2. left padding
    actual = _gather_slice_into_kv_cacheline(cache, "left", seq_len, bucket_slice)
    expected = torch.tensor([[[[50], [50], [50], [100], [100]]]])
    assert torch.equal(actual, expected)
    assert torch.equal(
        _deprecate_gather_slice_into_kv_cacheline(cache, "left", seq_len, bucket_slice), expected
    )


def test_reshape_tiled_kv():
    tiled_cache = torch.rand(1, 2, 2, 128, 2)
    actual = _reshape_tiled_cache(tiled_cache)
    assert actual.shape == torch.Size([1, 2, 256, 2])
    assert torch.equal(actual.flatten(), tiled_cache.flatten())


# Tests on get_active_block_table()
def test_get_active_block_table_case_1():
    block_table = torch.tensor([[149, 0], [148, 0], [147, 146], [145, 0]])
    seq_lens = torch.tensor([6, 16, 170, 6])
    block_size = 128

    actual = get_active_block_table(block_table, seq_lens, block_size)

    expected = torch.tensor([149, 148, 147, 146, 145])

    assert torch.equal(actual, expected)


def test_get_active_block_table_case_2():
    block_table = torch.tensor([[123, 128], [148, 0], [147, 146], [163, 0]])
    seq_lens = torch.tensor([0, 0, 0, 0])
    block_size = 128

    actual = get_active_block_table(block_table, seq_lens, block_size)

    expected = torch.tensor([])

    assert torch.equal(actual, expected)


def test_get_active_block_table_case_3():
    block_table = torch.tensor([[123, 128, 175], [148, 0, 0], [147, 146, 0]])
    seq_lens = torch.tensor([40, 16, 0])
    block_size = 24

    actual = get_active_block_table(block_table, seq_lens, block_size)

    expected = torch.tensor([123, 128, 148])

    assert torch.equal(actual, expected)


# Tests on contexted_kv()
def test_contexted_kv_case_1():
    x = 0
    cache = torch.arange(start=10, end=34).reshape(6, 4, 1, 1)  # block layout
    current = torch.arange(8).reshape(1, 1, 8, 1)  # BHSD
    cache_mask = torch.tensor(
        [1, 1, x, x, x, 1, 1, 1, 1, 1, x, x, 1, 1, 1, 1, x, x, x, x], dtype=torch.bool
    )
    cache_reordered_idx = torch.tensor(
        [0, 1, x, x, x, 5, 6, 7, 8, 9, x, x, 12, 13, 14, 15, x, x, x, x], dtype=torch.int
    )
    current_reordered_idx = torch.tensor(
        [x, x, 0, 1, 2, x, x, x, x, x, 3, 4, x, x, x, x, 5, x, x, x], dtype=torch.int
    )

    traced_func = prepare_traced_contexted_kv(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        cache_reordered_idx_shape=cache_reordered_idx.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, cache_reordered_idx, current_reordered_idx)

    # fmt: off
    expected = torch.tensor([
        10, 11,             # 2 cache for seq 0
         0,  1,  2,         # 3 current for seq 0
        15, 16, 17, 18, 19, # 5 cache for seq 1
         3,  4,             # 2 current for seq 1
        22, 23, 24, 25,     # 4 cache for seq 2
         5,                 # 1 curent for seq 2
         x,  x,  x,         # padding
    ])
    assert actual.shape == (1, 1, 20, 1)
    actual = actual.flatten()
    assert torch.equal(actual, expected)


def test_contexted_kv_case_2():
    H = 5
    D = 6
    x = 0
    cache = torch.arange(start=20, end=32).reshape(3, 4, 1, 1).expand(-1, -1, H, D)  # block layout
    current = torch.arange(6).reshape(1, 1, 6, 1).expand(-1, H, -1, D)  # BHSD
    cache_mask = torch.tensor([x, x, x, x, x, 1, 1, 1, 1, 1, x, x], dtype=torch.bool)
    cache_reordered_idx = torch.tensor([x, x, x, x, x, 0, 1, 2, 3, 4, x, x], dtype=torch.int)
    current_reordered_idx = torch.tensor([0, 1, 2, 3, 4, x, x, x, x, x, 5, x], dtype=torch.int)

    traced_func = prepare_traced_contexted_kv(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        cache_reordered_idx_shape=cache_reordered_idx.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, cache_reordered_idx, current_reordered_idx)

    expected = torch.tensor([[
                            # 0 cache for seq 0
         0,  1,  2,  3,  4, # 5 current for seq 0
        20, 21, 22, 23, 24, # 5 cache for seq 1
         5,                 # 1 current for seq 1
         x,                 # padding
    ]])
    assert actual.shape == (1, 5, 12, 6)
    for i in range(H):
        for j in range(D):
            actual_slice = actual[:, i, :, j]
            assert torch.equal(actual_slice, expected)


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


def test_contexted_kv_v2_cte():
    B = 3
    S = 4
    H = 5
    D = 6
    x = 0
    cache_start = 0
    current_start = 20
    cache = torch.arange(cache_start, cache_start + B * S).reshape(B, 1, S, 1).expand(-1, H, -1, D) # BHSD layout
    current = torch.arange(current_start, current_start + B * S).reshape(B, 1, S, 1).expand(-1, H, -1, D) # BHSD layout
    cache_mask = torch.tensor([[x, x, x, x],
                               [1, x, x, x],
                               [1, 1, x, x]], dtype=torch.bool)
    current_reordered_idx = torch.tensor([[0, 1, 2, x],
                                          [x, 0, 1, x],
                                          [x, x, 0, 1]], dtype=torch.int)

    traced_func = prepare_traced_contexted_kv_v2(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, current_reordered_idx)

    expected = torch.tensor(
        [
            [
                current_start,
                current_start + 1,
                current_start + 2,
                current_start,
            ],  # 0 cache for seq 0
            [
                cache_start + S,
                current_start + S,
                current_start + S + 1,
                current_start + S,
            ],  # 2 cache for seq 1
            [
                cache_start + S * 2,
                cache_start + S * 2 + 1,
                current_start + S * 2,
                current_start + S * 2 + 1,
            ],  # 2 cache for seq 2
        ]
    )
    assert actual.shape == (B, H, S, D)
    for i in range(B):
        for j in range(S):
            actual_slice = actual[i, :, j, :]
            assert torch.equal(actual_slice, expected[i, j] * torch.ones(H, D, dtype=torch.int))


def test_contexted_kv_v2_tkg():
    B = 3
    S = 4
    A = 1
    H = 5
    D = 6
    x = 0
    cache_start = 0
    current_start = 20
    cache = torch.arange(cache_start, cache_start + B * S).reshape(B, 1, S, 1).expand(-1, H, -1, D) # BHSD layout
    current = torch.arange(current_start, current_start + B * A).reshape(B, 1, A, 1).expand(-1, H, -1, D) # BHSD layout
    cache_mask = torch.tensor([[1, x, x, x],
                               [1, 1, x, x],
                               [1, 1, 1, x]], dtype=torch.bool)
    current_reordered_idx = torch.tensor([[x, 0, x, x],
                                          [x, x, 0, x],
                                          [x, x, x, 0]], dtype=torch.int)

    traced_func = prepare_traced_contexted_kv_v2(
        cache_shape=cache.shape,
        current_shape=current.shape,
        cache_mask_shape=cache_mask.shape,
        current_reordered_idx_shape=current_reordered_idx.shape,
    )

    actual = traced_func(cache, current, cache_mask, current_reordered_idx)

    expected = torch.tensor(
        [
            [cache_start, current_start, current_start + x, current_start + x],  # 1 cache for seq 0
            [
                cache_start + S,
                cache_start + S + 1,
                current_start + A,
                current_start + A,
            ],  # 2 cache for seq 1
            [
                cache_start + S * 2,
                cache_start + S * 2 + 1,
                cache_start + S * 2 + 2,
                current_start + A * 2,
            ],  # 3 cache for seq 2
        ]
    )
    assert actual.shape == (B, H, S, D)
    for i in range(B):
        for j in range(S):
            actual_slice = actual[i, :, j, :]
            print(i, j)
            print(actual_slice, expected[i, j])
            assert torch.equal(actual_slice, expected[i, j] * torch.ones(H, D, dtype=torch.int))


def prepare_traced_contexted_kv_v2(
    cache_shape,
    current_shape,
    cache_mask_shape,
    current_reordered_idx_shape,
):
    example_inputs = (
        torch.zeros(cache_shape, dtype=torch.float),
        torch.zeros(current_shape, dtype=torch.float),
        torch.zeros(cache_mask_shape, dtype=torch.bool),
        torch.zeros(current_reordered_idx_shape, dtype=torch.int),
    )

    traced_func = torch.jit.trace(contexted_kv_v2, example_inputs)
    return traced_func


# Tests on contexted_kv_indexing()
def test_contexted_kv_indexing_case_1():
    new_lens = torch.tensor([3, 2, 1, 0])
    all_lens = torch.tensor([5, 7, 5, 0])
    block_size = torch.tensor(4)

    actual = contexted_kv_indexing_dynamic(new_lens, all_lens, block_size)
    actual_cache_mask, actual_cache_reordred_idx, actual_current_reordered_idx = actual

    x = 0
    expected_cache_mask = torch.tensor(
        [1, 1, x, x, x, 1, 1, 1, 1, 1, x, x, 1, 1, 1, 1, x], dtype=torch.bool
    )
    expected_cache_reordered_idx = torch.tensor(
        [0, 1, x, x, x, 4, 5, 6, 7, 8, x, x, 12, 13, 14, 15, x], dtype=torch.int
    )
    expected_current_reordered_idx = torch.tensor(
        [x, x, 0, 1, 2, x, x, x, x, x, 3, 4, x, x, x, x, 5], dtype=torch.int
    )

    assert torch.equal(actual_cache_mask, expected_cache_mask)
    assert torch.equal(actual_cache_reordred_idx, expected_cache_reordered_idx)
    assert torch.equal(actual_current_reordered_idx, expected_current_reordered_idx)


def test_contexted_kv_indexing_case_2():
    new_lens = torch.tensor([1, 2, 3, 1, 0])
    all_lens = torch.tensor([5, 7, 5, 4, 0])
    block_size = torch.tensor(3)

    actual = contexted_kv_indexing_dynamic(new_lens, all_lens, block_size)
    actual_cache_mask, actual_cache_reordred_idx, actual_current_reordered_idx = actual

    x = 0
    expected_cache_mask = torch.tensor(
        [1, 1, 1, 1, x, 1, 1, 1, 1, 1, x, x, 1, 1, x, x, x, 1, 1, 1, x], dtype=torch.bool
    )
    expected_cache_reordered_idx = torch.tensor(
        [0, 1, 2, 3, x, 6, 7, 8, 9, 10, x, x, 12, 13, x, x, x, 15, 16, 17, x], dtype=torch.int
    )
    expected_current_reordered_idx = torch.tensor(
        [x, x, x, x, 0, x, x, x, x, x, 1, 2, x, x, 3, 4, 5, x, x, x, 6], dtype=torch.int
    )

    assert torch.equal(actual_cache_mask, expected_cache_mask)
    assert torch.equal(actual_cache_reordred_idx, expected_cache_reordered_idx)
    assert torch.equal(actual_current_reordered_idx, expected_current_reordered_idx)


def test_contexted_kv_indexing_v2():
    q_lens = torch.tensor([3, 2, 0])
    k_lens = torch.tensor([5, 7, 0])
    max_seq_len = torch.tensor(10)

    traced_func = prepare_traced_contexted_kv_indexing_v2(q_lens.shape, k_lens.shape, max_seq_len)
    actual_cache_mask, actual_current_reordered_idx = traced_func(q_lens, k_lens, max_seq_len)

    x = 0
    expected_cache_mask = torch.tensor(
        [
            [1, 1, x, x, x, x, x, x, x, x],
            [1, 1, 1, 1, 1, x, x, x, x, x],
            [x, x, x, x, x, x, x, x, x, x],
        ],
        dtype=torch.bool,
    )
    expected_current_reordered_idx = torch.tensor(
        [
            [x, x, 0, 1, 2, x, x, x, x, x],
            [x, x, x, x, x, 0, 1, x, x, x],
            [x, x, x, x, x, x, x, x, x, x],
        ],
        dtype=torch.int,
    )

    assert torch.equal(actual_cache_mask, expected_cache_mask)
    assert torch.equal(actual_current_reordered_idx, expected_current_reordered_idx)


def prepare_traced_contexted_kv_indexing_v2(
    q_lens_shape,
    k_lens_shape,
    max_seq_len,
):
    example_inputs = (
        torch.zeros(q_lens_shape, dtype=torch.long),
        torch.zeros(k_lens_shape, dtype=torch.long),
        torch.tensor(max_seq_len).long(),
    )
    traced_func = torch_neuronx.trace(
        contexted_kv_indexing_v2, example_inputs, compiler_workdir="/tmp/nxd_ut/"
    )
    return traced_func


class TestDynamicUpdateSliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cache, update, seq_ids):
        assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1, "only supports single seq_id"
        # compute start_indices for update
        indices = torch.zeros(cache.dim(), dtype=seq_ids.dtype, device=seq_ids.device)
        indices = indices.scatter(dim=0, index=torch.tensor([0], dtype=torch.int64, device=cache.device), src=seq_ids)
        indices = indices.split(1)
        indices = [t.squeeze() for t in indices]
        return dynamic_update_slice(cache, update, indices)


class TestDynamicUpdateSlice(unittest.TestCase):
    def test_dynamic_update_slice(self):
        cache_tensor = torch.zeros([4, 1, 8, 12], dtype=torch.float32)
        cache_update = torch.arange(4*4, dtype=torch.float32).reshape([1, 1, 4, 4])
        seq_id_val = 1
        seq_ids = torch.tensor([seq_id_val])
        tp_degree = 1

        cpu_cache = cache_tensor.clone().detach()
        indices = torch.zeros(cpu_cache.dim(), dtype=seq_ids.dtype, device=seq_ids.device)
        indices = indices.scatter(dim=0, index=torch.tensor([0], dtype=torch.int64, device=cpu_cache.device), src=seq_ids)
        indices = indices.split(1)
        indices = [t.squeeze() for t in indices]

        # compute ref on cpu
        cpu_cache[indices[0]:indices[0]+cache_update.shape[0],
                indices[1]:indices[1]+cache_update.shape[1],
                indices[2]:indices[2]+cache_update.shape[2],
                indices[3]:indices[3]+cache_update.shape[3]] = cache_update

        builder = ModelBuilder(
            router=None,
            tp_degree=tp_degree,
            checkpoint_loader=lambda: OrderedDict(),
        )
        builder.add(
            key="main",
            model_instance=BaseModelInstance(
            lambda: TestDynamicUpdateSliceModule(),
            input_output_aliases={},
            ),
            example_inputs=[(cache_tensor, cache_update, seq_ids)],
        )

        neuron_module = builder.trace(initialize_model_weights=True)

        cache = cache_tensor.clone().detach()
        neuron_output = neuron_module(cache, cache_update, seq_ids)
        torch.testing.assert_close(cpu_cache, neuron_output)
