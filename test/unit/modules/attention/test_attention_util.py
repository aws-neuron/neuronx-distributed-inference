import unittest
import pytest

import torch
import torch.nn as nn

from neuronx_distributed_inference.modules.attention.utils import move_heads_front, get_context_parallel_reordered_tp_mapping, get_kv_head_indices_context_parallel_full_tp_decode


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
    "tp_degree, cp_degree, expected_ordering",
    # fmt: off
    [
        (8, 4, [0, 4, 1, 5, 2, 6, 3, 7]),
        (16, 2, [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]),
        (32, 4, [0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31])
    ]
    # fmt: on
)
def test_get_context_parallel_reordered_tp_mapping(tp_degree, cp_degree, expected_ordering):
    ordering = get_context_parallel_reordered_tp_mapping(tp_degree, cp_degree)
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