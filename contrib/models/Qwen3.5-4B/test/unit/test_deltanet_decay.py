# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fused DeltaNet log-decay stability structure."""

import pathlib
import unittest


_CONTRIB_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_ROOT = _CONTRIB_ROOT / "src"


class TestFusedDeltaNetDecayStability(unittest.TestCase):
    def test_fused_kernel_uses_exp_of_differences(self):
        kernel_source = (
            _SRC_ROOT / "nki_kernels" / "nki_deltanet_fused.py"
        ).read_text()

        self.assertIn("decay_strict", kernel_source)
        self.assertIn("decay_diag", kernel_source)
        self.assertIn("gl_minus_gc_p", kernel_source)
        self.assertNotIn("exp_neg_gc_p", kernel_source)
        self.assertNotIn("operand0=exp_neg_gc_p", kernel_source)

    def test_modeling_does_not_clamp_fused_decay_inputs(self):
        modeling_source = (_SRC_ROOT / "modeling_qwen35.py").read_text()

        self.assertNotIn("_bound_fused_deltanet_log_decay", modeling_source)
        self.assertNotIn("FUSED_DELTANET_DECAY_MIN", modeling_source)
        self.assertIn("exp(cumsum(g)_i - cumsum(g)_j)", modeling_source)


if __name__ == "__main__":
    unittest.main()
