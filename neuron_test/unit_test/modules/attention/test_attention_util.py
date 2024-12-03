import unittest

import torch
import torch.nn as nn

from neuronx_distributed_inference.modules.attention.utils import move_heads_front


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
