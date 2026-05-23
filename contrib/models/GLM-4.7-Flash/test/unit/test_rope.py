# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-4.7-Flash RoPE implementation.

Validates standard RoPE (no YaRN) produces correct embeddings.
"""

import unittest
import math

import torch

from src.rope_util import (
    Glm4MoeLiteRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_fn,
)


class TestRotaryEmbedding(unittest.TestCase):
    """Test Glm4MoeLiteRotaryEmbedding."""

    def test_output_shapes(self):
        """cos/sin should have shape (seq_len, dim) after forward."""
        dim = 64
        seq_len = 128
        rope = Glm4MoeLiteRotaryEmbedding(dim=dim, max_position_embeddings=200000)
        # forward expects a tensor for device/dtype reference
        x = torch.randn(1, 1, 1, dim)
        cos, sin = rope(x, seq_len=seq_len)
        self.assertEqual(cos.shape, (seq_len, dim))
        self.assertEqual(sin.shape, (seq_len, dim))

    def test_cos_sin_bounded(self):
        """cos/sin values should be in [-1, 1]."""
        dim = 64
        rope = Glm4MoeLiteRotaryEmbedding(dim=dim)
        x = torch.randn(1, 1, 1, dim)
        cos, sin = rope(x, seq_len=1024)
        self.assertTrue(cos.abs().max() <= 1.0 + 1e-6)
        self.assertTrue(sin.abs().max() <= 1.0 + 1e-6)

    def test_no_scaling(self):
        """Standard RoPE: inv_freq = base^(-2i/dim), no scaling factor."""
        dim = 8
        base = 10000
        rope = Glm4MoeLiteRotaryEmbedding(dim=dim, base=base)
        expected_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        torch.testing.assert_close(rope.inv_freq, expected_inv_freq)

    def test_position_0_is_identity(self):
        """At position 0, cos=1 and sin=0 so RoPE is identity."""
        dim = 64
        rope = Glm4MoeLiteRotaryEmbedding(dim=dim)
        x = torch.randn(1, 1, 1, dim)
        cos, sin = rope(x, seq_len=1)
        # At pos 0: cos=1, sin=0 for all dims
        torch.testing.assert_close(cos[0], torch.ones(dim))
        torch.testing.assert_close(sin[0], torch.zeros(dim), atol=1e-6, rtol=0)


class TestRotateFn(unittest.TestCase):
    """Test the interleaved rotation function."""

    def test_basic_rotation(self):
        """rotate_fn pairs (x0,x1) -> (-x1,x0)."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([-2.0, 1.0, -4.0, 3.0])
        result = rotate_fn(x.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(result, expected)

    def test_rotation_preserves_norm(self):
        """Rotation should preserve the L2 norm of each pair."""
        x = torch.randn(2, 4, 8, 64)
        rotated = rotate_fn(x)
        # L2 norm of each pair should be preserved
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        r_pairs = rotated.view(*rotated.shape[:-1], -1, 2)
        x_norms = x_pairs.norm(dim=-1)
        r_norms = r_pairs.norm(dim=-1)
        torch.testing.assert_close(x_norms, r_norms, atol=1e-5, rtol=1e-5)


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test the full apply_rotary_pos_emb function."""

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        bsz, num_heads, seq_len, head_dim = 2, 5, 16, 64
        rope = Glm4MoeLiteRotaryEmbedding(dim=head_dim)
        q = torch.randn(bsz, num_heads, seq_len, head_dim)
        cos, sin = rope(q, seq_len=seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        q_embed = apply_rotary_pos_emb(q, cos, sin, position_ids)
        self.assertEqual(q_embed.shape, q.shape)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        bsz, num_heads, seq_len, head_dim = 1, 1, 4, 64
        rope = Glm4MoeLiteRotaryEmbedding(dim=head_dim)
        q = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        cos, sin = rope(q, seq_len=seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        q_embed = apply_rotary_pos_emb(q, cos, sin, position_ids)
        self.assertEqual(q_embed.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
