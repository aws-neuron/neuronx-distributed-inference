# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fused DeltaNet log-decay bounding."""

import os
import sys
import unittest

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35 import (
    FUSED_DELTANET_DECAY_MAX,
    FUSED_DELTANET_DECAY_MIN,
    _bound_fused_deltanet_log_decay,
)


def _chunked_cumsum(g, batch_size, num_heads, total_seq_len, chunk_size):
    num_chunks = total_seq_len // chunk_size
    return g.reshape(batch_size, num_heads, num_chunks, chunk_size).cumsum(dim=-1)


class TestFusedDeltaNetDecayBounding(unittest.TestCase):
    def test_preserves_non_extreme_decay(self):
        batch_size, num_heads, total_seq_len, chunk_size = 2, 3, 16, 8
        g = torch.full(
            (batch_size, num_heads, total_seq_len),
            -0.125,
            dtype=torch.float32,
        )

        bounded = _bound_fused_deltanet_log_decay(
            g, batch_size, num_heads, total_seq_len, chunk_size
        )

        torch.testing.assert_close(bounded, g)

    def test_bounds_per_chunk_cumulative_decay(self):
        batch_size, num_heads, total_seq_len, chunk_size = 2, 3, 16, 8
        g = torch.full(
            (batch_size, num_heads, total_seq_len),
            -10.0,
            dtype=torch.float32,
        )

        bounded = _bound_fused_deltanet_log_decay(
            g, batch_size, num_heads, total_seq_len, chunk_size
        )
        bounded_cumsum = _chunked_cumsum(
            bounded, batch_size, num_heads, total_seq_len, chunk_size
        )
        expected_cumsum = _chunked_cumsum(
            g, batch_size, num_heads, total_seq_len, chunk_size
        ).clamp(min=FUSED_DELTANET_DECAY_MIN, max=FUSED_DELTANET_DECAY_MAX)

        torch.testing.assert_close(bounded_cumsum, expected_cumsum)
        self.assertGreaterEqual(float(bounded_cumsum.min()), FUSED_DELTANET_DECAY_MIN)
        self.assertLessEqual(float(bounded_cumsum.max()), FUSED_DELTANET_DECAY_MAX)
        self.assertTrue(torch.isfinite(bounded).all())


if __name__ == "__main__":
    unittest.main()
