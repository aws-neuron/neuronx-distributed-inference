import math
import unittest
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.modules.attention.utils import (move_heads_front, 
                                                                   get_context_parallel_reordered_tp_mapping, 
                                                                   get_kv_head_indices_context_parallel_full_tp_decode, 
                                                                   validate_tp_prefill_to_dp_decode, 
                                                                   get_context_parallel_reordered_dp_mapping, 
                                                                   get_kv_head_indices_context_parallel_dp_decode,
                                                                   get_last_kv_chunk,
                                                                   get_last_kv_window,
                                                                   order_strided_tensor,
                                                                   stride_tensor,
                                                                   chunk_and_reorder_tensor,
                                                                   reshape_qkv_for_chunked_flash_attention_kernel)

class TestMoveHeadsFront(unittest.TestCase):
    def test_move_heads_front(self):
        batch_size = 2
        seq_len = 64
        num_head = 32
        head_dim = 128
        layernorm = nn.LayerNorm(head_dim)
        x = torch.randn(batch_size * seq_len * num_head * head_dim).view(
            batch_size, seq_len, num_head, head_dim
        )
        """
         Test without applying LayerNorm
        """
        output_no_layernorm = move_heads_front(x, batch_size, seq_len, num_head, head_dim)
        self.assertEqual(output_no_layernorm.shape, (batch_size, num_head, seq_len, head_dim))
        expected_output_no_layernorm = x.transpose(1, 2).contiguous()
        assert torch.allclose(output_no_layernorm, expected_output_no_layernorm)

        """
        Test with applying LayerNorm
        """
        output_with_layernorm = move_heads_front(
            x, batch_size, seq_len, num_head, head_dim, layernorm
        )
        reshaped_tensor_with_layernorm = layernorm(x.view(batch_size, seq_len, num_head, head_dim))
        expected_output_with_layernorm = reshaped_tensor_with_layernorm.transpose(1, 2).contiguous()
        self.assertEqual(output_with_layernorm.shape, (batch_size, num_head, seq_len, head_dim))
        assert torch.allclose(output_with_layernorm, expected_output_with_layernorm)

@pytest.mark.parametrize(
    "tp_degree, cp_degree, num_kv_heads, cte_rank_ordering, expected_ordering",
    # fmt: off
    [
        (8, 4, 8, None, [0, 4, 1, 5, 2, 6, 3, 7]),
        (16, 2, 8, None, [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]),
        (32, 4, 8, None, [0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31]),
        (64, 8, 8, [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 20, 21, 22, 23, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 36, 37, 38, 39, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 52, 53, 54, 55], 
                   [0, 8, 16, 24, 1, 9, 17, 25, 32, 40, 48, 56, 33, 41, 49, 57, 2, 10, 18, 26, 3, 11, 19, 27, 34, 42, 50, 58, 35, 43, 51, 59, 4, 12, 20, 28, 5, 13, 21, 29, 36, 44, 52, 60, 37, 45, 53, 61, 6, 14, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63]),
    ]
    # fmt: on
)
def test_get_context_parallel_reordered_tp_mapping(tp_degree, cp_degree, num_kv_heads, cte_rank_ordering, expected_ordering):
    ordering = get_context_parallel_reordered_tp_mapping(tp_degree, cp_degree, num_kv_heads, torch.device("cpu"), cte_rank_ordering=cte_rank_ordering)

    assert ordering == expected_ordering

@pytest.mark.parametrize(
    "num_kv_heads, tp_degree, cp_degree, expected_indices",
    # fmt: off
    [
        (16, 16, 4, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), # 16 heads, TP = 4 Attention
        (8, 32, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # 8 heads, TP = 8 Attention
        (16, 32, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), # 16 heads, TP = 8 Attention
        (4, 32, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # 4 heads, TP = 8 Attention
        (8, 64, 16, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), # 8 heads, TP = 4 Attention
    ]
    # fmt: on
)
def test_get_kv_head_indices_context_parallel_full_tp_decode(num_kv_heads, tp_degree, cp_degree, expected_indices):
    indices = get_kv_head_indices_context_parallel_full_tp_decode(num_kv_heads, tp_degree, cp_degree, torch.device("cpu"))
    assert indices.tolist() == expected_indices


@pytest.mark.parametrize(
    "num_kv_heads, world_size, dp_degree, working_case",
    # fmt: off
    [
        (8, 32, 4, True),
        (16, 32, 4, False),
        (8, 64, 16, False),
        (4, 32, 4, True),
        (8, 64, 4, True),
    ]
    # fmt: on
)
def test_validate_tp_prefill_to_dp_decode(num_kv_heads, world_size, dp_degree, working_case):
    if not working_case:
        with pytest.raises(AssertionError):
            validate_tp_prefill_to_dp_decode(num_kv_heads=num_kv_heads, world_size=world_size, dp_degree=dp_degree)
    else:
        validate_tp_prefill_to_dp_decode(num_kv_heads=num_kv_heads, world_size=world_size, dp_degree=dp_degree)


@pytest.mark.parametrize(
    "n_heads, seq_len, head_dim, chunk_size",
    # fmt: off
    [
        (8, 1024, 128, 512),
    ]
    # fmt: on
)
def test_reshape_qkv_for_chunked_flash_attention_kernel(n_heads, seq_len, head_dim, chunk_size):
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)
    n_chunks = math.ceil(seq_len / chunk_size)
    actual_q, actual_k, actual_v = reshape_qkv_for_chunked_flash_attention_kernel(q, k, v, chunk_size, torch.float32)
    assert actual_q.shape == (n_chunks * n_heads, head_dim, chunk_size)
    assert actual_k.shape == (n_chunks * n_heads, head_dim, chunk_size)
    assert actual_v.shape == (n_chunks * n_heads, chunk_size, head_dim)


@pytest.mark.parametrize(
    "test_name, position_ids, expected_position_ids",
    # fmt: off
    [    
        # Test case 1: seq 0 in chunk 0, seq 1 in chunk 1
        ('test case 1', torch.tensor([[0, 1, 2, 3, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32), torch.tensor([[0, 4], [4, 8]], dtype=torch.int32)),
        # Test case 2: both seqs in chunk 1
        ('test case 2', torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32), torch.tensor([[4, 8], [4, 8]], dtype=torch.int32)),
        # Test case 3: seq 0 in chunk 0
        ('test case 3', torch.tensor([[0, 1, 2, 3, 1, 1, 1, 1]], dtype=torch.int32), torch.tensor([[0, 4]], dtype=torch.int32)),
        # Test case 4: seq_len indivisible by chunk size (4)
        ('test case 4', torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32), torch.tensor([[4, 8]], dtype=torch.int32)),
    ]
    # fmt: on
)    
def test_get_lask_kv_chunk(test_name, position_ids, expected_position_ids):
    # Test parameters
    seq_len = position_ids.shape[1]
    batch_size = position_ids.shape[0]
    head_dim = 8
    num_kv_heads = 4
    latest_k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    latest_v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    attn_chunk_size = 4
    last_k_chunk, last_v_chunk = get_last_kv_chunk(attn_chunk_size, position_ids, latest_k, latest_v)
    if seq_len % attn_chunk_size != 0:
        latest_v = F.pad(latest_v, (0, 0, 0, attn_chunk_size - seq_len % attn_chunk_size), "constant", 0)
        latest_k = F.pad(latest_k, (0, 0, 0, attn_chunk_size - seq_len % attn_chunk_size), "constant", 0)

    # Check that the actual outputs updated_k and updated_v have the correct shape
    assert last_k_chunk.shape == (batch_size, num_kv_heads, attn_chunk_size, head_dim), f"k shape mismatch for test {test_name}"
    assert last_v_chunk.shape == (batch_size, num_kv_heads, attn_chunk_size, head_dim), f"v shape mismatch for test {test_name}"
    for b in range(batch_size):
        start = expected_position_ids[b, 0]
        end = expected_position_ids[b, 1]
        assert torch.allclose(last_k_chunk[b, :, :, :], latest_k[b, :, start:end, :]), f"k values mismatch for test {test_name}"
        assert torch.allclose(last_v_chunk[b, :, :, :], latest_v[b, :, start:end, :]), f"v values mismatch for test {test_name}"

    
@pytest.mark.parametrize(
    "tp_degree, cp_degree, dp_degree, num_kv_heads, cte_rank_ordering, expected_ordering",
    # fmt: off
    [
        (64, 16, 4, 8, None, [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31, 32, 36, 40, 44, 33, 37, 41, 45, 34, 38, 42, 46, 35, 39, 43, 47, 48, 52, 56, 60, 49, 53, 57, 61, 50, 54, 58, 62, 51, 55, 59, 63]),
        (64, 16, 8, 8, None, [0, 2, 4, 6, 8, 10, 12, 14, 9, 11, 13, 15, 1, 3, 5, 7, 16, 18, 20, 22, 24, 26, 28, 30, 25, 27, 29, 31, 17, 19, 21, 23, 32, 34, 36, 38, 40, 42, 44, 46, 41, 43, 45, 47, 33, 35, 37, 39, 48, 50, 52, 54, 56, 58, 60, 62, 57, 59, 61, 63, 49, 51, 53, 55]),
        (64, 8, 4, 8, None, [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30, 17, 19, 21, 23, 25, 27, 29, 31, 32, 34, 36, 38, 40, 42, 44, 46, 33, 35, 37, 39, 41, 43, 45, 47, 48, 50, 52, 54, 56, 58, 60, 62, 49, 51, 53, 55, 57, 59, 61, 63]),
        (64, 8, 4, 8, [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 20, 21, 22, 23, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 36, 37, 38, 39, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 52, 53, 54, 55], 
                      [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15, 16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31, 32, 34, 36, 38, 33, 35, 37, 39, 40, 42, 44, 46, 41, 43, 45, 47, 48, 50, 52, 54, 49, 51, 53, 55, 56, 58, 60, 62, 57, 59, 61, 63]),
    ]
    # fmt: on
)
def test_get_context_parallel_reordered_dp_mapping(tp_degree, cp_degree, dp_degree, num_kv_heads, cte_rank_ordering, expected_ordering):
    ordering = get_context_parallel_reordered_dp_mapping(tp_degree, cp_degree, dp_degree, num_kv_heads, torch.device("cpu"), cte_rank_ordering=cte_rank_ordering)

    assert ordering == expected_ordering


@pytest.mark.parametrize(
    "num_kv_heads, tp_degree, cp_degree, dp_degree, expected_indices",
    # fmt: off
    [
        (8, 64, 16, 4, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), # 8 heads, TP = 4 Attention
        (8, 64, 4, 16, None), # 8 heads, TP = 4 Attention. CP will have 2 * 4 KV copies, DP will need 16 copies, not a valid config.
        (16, 64, 16, 4, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), # 16 heads, TP = 4 Attention
    ]
    # fmt: on
)
def test_get_kv_head_indices_context_parallel_dp_decode(num_kv_heads, tp_degree, cp_degree, dp_degree, expected_indices):
    if expected_indices is None:
        with pytest.raises(AssertionError):
            indices = get_kv_head_indices_context_parallel_dp_decode(num_kv_heads, tp_degree, cp_degree, dp_degree, torch.device("cpu"))
    else:
        indices = get_kv_head_indices_context_parallel_dp_decode(num_kv_heads, tp_degree, cp_degree, dp_degree, torch.device("cpu"))
        assert indices.tolist() == expected_indices


@pytest.mark.parametrize(
    "test_name, position_ids, expected_position_ids",
    # fmt: off
    [    
        # Test case 1: first seq's pos ends at window size border, second seq's pos ends in middle of window
        ('test case 1', torch.tensor([[0, 1, 2, 3, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 1]], dtype=torch.int32), torch.tensor([[0,1,2,3], [4,5,6,3]], dtype=torch.int32)),
        # Test case 2: first seq's pos ends before the first window, second seq's pos ends at last window border
        ('test case 2', torch.tensor([[0, 1, 2, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32), torch.tensor([[0,1,2,3], [4,5,6,7]], dtype=torch.int32)),
    ]
    # fmt: on
)    
def test_get_last_kv_window(test_name, position_ids, expected_position_ids):
    # Test parameters
    batch_size, seq_len = position_ids.shape
    num_kv_heads, head_dim = 8, 4
    latest_k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    latest_v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    window_size = 4
    last_k_window, last_v_window = get_last_kv_window(window_size, position_ids, latest_k, latest_v)

    # Check that the actual outputs updated_k and updated_v have the correct shape
    assert last_k_window.shape == (batch_size, num_kv_heads, window_size, head_dim), f"k shape mismatch for test {test_name}"
    assert last_v_window.shape == (batch_size, num_kv_heads, window_size, head_dim), f"v shape mismatch for test {test_name}"
    for b in range(batch_size):
        for i, j in enumerate(expected_position_ids[b]):
            assert torch.allclose(last_k_window[b, :, i, :], latest_k[b, :, j, :]), f"k values mismatch for test {test_name}"
            assert torch.allclose(last_v_window[b, :, i, :], latest_v[b, :, j, :]), f"v values mismatch for test {test_name}"


@pytest.mark.parametrize(
    "tensor, dim, step_size, expected",
    # fmt: off
    [
        (torch.tensor([0, 1, 2, 3, 4, 5]), 0, 2, torch.tensor([0, 2, 4, 1, 3, 5])),
        (torch.tensor([0, 1, 2, 3, 4, 5]), 0, 3, torch.tensor([0, 3, 1, 4, 2, 5])),
        (torch.tensor([0, 1, 2, 3, 4, 5]), 0, 1, torch.tensor([0, 1, 2, 3, 4, 5])),
        (torch.tensor([0, 1, 2, 3, 4, 5]), 0, 6, torch.tensor([0, 1, 2, 3, 4, 5])),
        
        (torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), 1, 2, 
        torch.tensor([[0, 2, 1, 3], [4, 6, 5, 7]])),
        (torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), 0, 1, 
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])),
        
        (torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]], 
                    [[8, 9, 10, 11], [12, 13, 14, 15]]]), 2, 2,
        torch.tensor([[[0, 2, 1, 3], [4, 6, 5, 7]], 
                    [[8, 10, 9, 11], [12, 14, 13, 15]]])),
    ]
    # fmt: on
)
def test_stride_tensor(tensor, dim, step_size, expected):
    result = stride_tensor(tensor, dim, step_size)

    assert torch.all(result == expected)
    assert result.shape == tensor.shape

@pytest.mark.parametrize(
    "tensor, dim, step_size, expected",
    # fmt: off
    [
        (torch.tensor([0, 2, 4, 1, 3, 5]), 0, 2, torch.tensor([0, 1, 2, 3, 4, 5])),
        (torch.tensor([0, 3, 1, 4, 2, 5]), 0, 3, torch.tensor([0, 1, 2, 3, 4, 5])),
        (torch.tensor([0, 1, 2, 3, 4, 5]), 0, 1, torch.tensor([0, 1, 2, 3, 4, 5])),
        
        (torch.tensor([[0, 2, 1, 3], [4, 6, 5, 7]]), 1, 2, 
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])),
        
        (torch.tensor([[[0, 2, 1, 3], [4, 6, 5, 7]], 
                       [[8, 10, 9, 11], [12, 14, 13, 15]]]), 2, 2,
        torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]], 
                      [[8, 9, 10, 11], [12, 13, 14, 15]]])),
    ]
    # fmt: on
)
def test_order_strided_tensor(tensor, dim, step_size, expected):
    result = order_strided_tensor(tensor, dim, step_size)

    assert torch.all(result == expected)
    assert result.shape == tensor.shape

@pytest.mark.parametrize("tensor, order, dim, expected", [
    (torch.tensor([0, 1, 2, 3]), [1, 0], 0, torch.tensor([2, 3, 0, 1])),
    (torch.tensor([0, 1, 2, 3, 4, 5]), [0, 1, 2], 0, torch.tensor([0, 1, 2, 3, 4, 5])),
    (torch.tensor([0, 1, 2, 3, 4, 5]), [2, 1, 0], 0, torch.tensor([4, 5, 2, 3, 0, 1])),
    
    (torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]), [1, 0], 0, 
     torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]])),
    
    (torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), [1, 0], 1, 
     torch.tensor([[2, 3, 0, 1], [6, 7, 4, 5]])),
])
def test_chunk_and_reorder_tensor(tensor, order, dim, expected):
    result = chunk_and_reorder_tensor(tensor, order, dim)
    
    assert torch.all(torch.eq(result, expected))