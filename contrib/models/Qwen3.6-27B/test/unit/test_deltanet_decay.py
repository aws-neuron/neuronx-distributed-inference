# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regressions for the fused DeltaNet decay reference math."""

import importlib.util
import os
import types
import unittest

import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VALIDATOR_PATH = os.path.join(
    _CONTRIB_ROOT,
    "scripts",
    "validate_deltanet_fused_nki.py",
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "qwen36_validate_deltanet_fused_nki",
        _VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestFusedDeltaNetDecayMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _load_validator()

    def test_stable_causal_decay_masks_before_exp(self):
        gc = torch.linspace(0.0, -300.0, 128, dtype=torch.float32).reshape(128, 1)
        lower = torch.tril(torch.ones((128, 128), dtype=torch.float32), diagonal=-1)
        lower_diag = torch.tril(torch.ones((128, 128), dtype=torch.float32))

        strict_decay = self.validator.stable_causal_decay(torch, gc, lower)
        diag_decay = self.validator.stable_causal_decay(torch, gc, lower_diag)

        self.assertTrue(torch.isfinite(strict_decay).all())
        self.assertTrue(torch.isfinite(diag_decay).all())
        self.assertTrue(torch.equal(strict_decay.triu(), torch.zeros_like(strict_decay.triu())))
        torch.testing.assert_close(torch.diagonal(diag_decay), torch.ones(128))

    def test_reference_math_is_finite_for_realistic_gate_scale(self):
        args = types.SimpleNamespace(
            seed=1234,
            seq_len=256,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        output, state = self.validator.reference_math(torch, inputs)

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(state).all())

    def test_autocp_affine_chunk_matches_current_reference(self):
        args = types.SimpleNamespace(
            seed=20260602,
            seq_len=128,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        parts = self.validator.deltanet_chunk_affine_parts(torch, inputs, 0)
        actual_output, actual_state = self.validator.apply_deltanet_chunk_affine(
            torch,
            parts,
            inputs["state_in"],
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_autocp_reference_matches_current_reference(self):
        args = types.SimpleNamespace(
            seed=20260602,
            seq_len=1024,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        for cp_chunks in (1, 2, 4, 8):
            actual_output, actual_state = self.validator.autocp_reference_math(
                torch,
                inputs,
                cp_chunks=cp_chunks,
            )

            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                actual_state,
                expected_state,
                atol=2.0e-5,
                rtol=2.0e-5,
            )

    def test_autocp_reference_matches_current_reference_multihead(self):
        args = types.SimpleNamespace(
            seed=20260602,
            seq_len=512,
            heads=3,
            multihead=True,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        actual_output, actual_state = self.validator.autocp_reference_math(
            torch,
            inputs,
            cp_chunks=2,
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_compact_autocp_reference_matches_current_reference(self):
        args = types.SimpleNamespace(
            seed=20260602,
            seq_len=1024,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        for cp_chunks in (1, 2, 4, 8):
            actual_output, actual_state = self.validator.compact_autocp_reference_math(
                torch,
                inputs,
                cp_chunks=cp_chunks,
            )

            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                actual_state,
                expected_state,
                atol=2.0e-5,
                rtol=2.0e-5,
            )

    def test_compact_autocp_reference_matches_current_reference_multihead(self):
        args = types.SimpleNamespace(
            seed=20260602,
            seq_len=512,
            heads=3,
            multihead=True,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        for cp_chunks in (1, 2, 4):
            actual_output, actual_state = self.validator.compact_autocp_reference_math(
                torch,
                inputs,
                cp_chunks=cp_chunks,
            )

            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                actual_state,
                expected_state,
                atol=2.0e-5,
                rtol=2.0e-5,
            )

    def test_reference_qk_normalization_is_zero_safe(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260601)
        query = torch.randn((4, 128), generator=generator) * 0.05
        key = torch.randn((4, 128), generator=generator) * 0.05
        query[0].zero_()
        key[1].zero_()

        query_norm, key_norm = self.validator.normalize_reference_qk(
            torch,
            query,
            key,
        )

        self.assertTrue(torch.isfinite(query_norm).all())
        self.assertTrue(torch.isfinite(key_norm).all())
        torch.testing.assert_close(query_norm[0], torch.zeros_like(query_norm[0]))
        torch.testing.assert_close(key_norm[1], torch.zeros_like(key_norm[1]))
        torch.testing.assert_close(
            torch.linalg.vector_norm(query_norm[2]),
            torch.tensor(self.validator.P_MAX ** -0.5),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(key_norm[2]),
            torch.tensor(1.0),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_multihead_launch_spec_rejects_head_group_size_above_lnc_when_spmd_disabled(self):
        previous = os.environ.get("QWEN36_DELTANET_MULTIHEAD_SPMD")
        os.environ["QWEN36_DELTANET_MULTIHEAD_SPMD"] = "0"
        try:
            with self.assertRaisesRegex(ValueError, "head-group-size exceeds --lnc"):
                self.validator.multihead_launch_spec(num_heads=2, lnc=1)
        finally:
            if previous is None:
                os.environ.pop("QWEN36_DELTANET_MULTIHEAD_SPMD", None)
            else:
                os.environ["QWEN36_DELTANET_MULTIHEAD_SPMD"] = previous

    def test_blocked_triangular_solve_matches_torch_solve(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260601)
        strict_lower = torch.tril(
            torch.randn((128, 128), generator=generator) * 0.01,
            diagonal=-1,
        )
        lhs = torch.eye(128) + strict_lower
        rhs = torch.randn((128, 128), generator=generator) * 0.05

        expected = torch.linalg.solve_triangular(lhs, rhs, upper=False)
        for block_size in (8, 16, 32):
            actual = self.validator.blocked_lower_triangular_solve(
                torch,
                lhs,
                rhs,
                block_size,
            )
            torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)

    def test_block_prefix_triangular_solve_matches_torch_solve(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260602)
        strict_lower = torch.tril(
            torch.randn((128, 128), generator=generator) * 0.05,
            diagonal=-1,
        )
        lhs = torch.eye(128) + strict_lower
        rhs = torch.randn((128, 128), generator=generator) * 0.05

        expected = torch.linalg.solve_triangular(lhs, rhs, upper=False)
        for block_size in (16, 32, 64):
            actual = self.validator.block_prefix_lower_triangular_solve(
                torch,
                lhs,
                rhs,
                block_size,
            )
            torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)

    def test_hierarchical_kkt_triangular_solve_matches_torch_solve(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260602)
        strict_lower = torch.tril(
            torch.randn((128, 128), generator=generator) * 0.01,
            diagonal=-1,
        )
        lhs = torch.eye(128) + strict_lower
        rhs = torch.randn((128, 128), generator=generator) * 0.05

        expected = torch.linalg.solve_triangular(lhs, rhs, upper=False)
        for leaf_size in (8, 16, 32):
            actual = self.validator.hierarchical_kkt_lower_triangular_solve(
                torch,
                lhs,
                rhs,
                leaf_size,
            )
            torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)

    def test_two_step_doubling_solve_matches_realistic_chunks(self):
        args = types.SimpleNamespace(
            seed=20260601,
            seq_len=512,
            heads=4,
            multihead=True,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        lower = inputs["lower_mask"]
        eye = inputs["identity"]

        max_relative_norm = 0.0
        max_absolute = 0.0
        for head_idx in range(args.heads):
            state = inputs["state_in"][head_idx].clone()
            for start in range(0, args.seq_len, self.validator.P_MAX):
                end = start + self.validator.P_MAX
                _, key = self.validator.normalize_reference_qk(
                    torch,
                    inputs["query"][head_idx, start:end],
                    inputs["key"][head_idx, start:end],
                )
                value = inputs["value"][head_idx, start:end]
                g = inputs["g_raw"][head_idx, start:end]
                beta = inputs["beta"][head_idx, start:end]

                gc = torch.cumsum(g, dim=0)
                k_beta = key * beta
                v_beta = value * beta
                decay = self.validator.stable_causal_decay(torch, gc, lower)
                a_mat = -((k_beta @ key.T) * decay) * lower
                lhs = eye - a_mat
                rhs = v_beta - ((k_beta * torch.exp(gc)) @ state)

                expected = torch.linalg.solve_triangular(lhs, rhs, upper=False)
                actual = self.validator.scan_doubling_lower_triangular_solve(
                    torch,
                    lhs,
                    rhs,
                    steps=2,
                )

                diff = actual - expected
                max_relative_norm = max(
                    max_relative_norm,
                    torch.linalg.vector_norm(diff).item()
                    / torch.linalg.vector_norm(expected).item(),
                )
                max_absolute = max(max_absolute, diff.abs().max().item())

                gl = gc[-1:]
                key_decay = key * torch.exp(gl - gc)
                state = (state * torch.exp(gl)) + (key_decay.T @ expected)

        self.assertLess(max_relative_norm, 5.0e-6)
        self.assertLess(max_absolute, 2.0e-6)

    def test_two_step_doubling_solve_truncates_weak_decay_chunks(self):
        args = types.SimpleNamespace(
            seed=1241,
            seq_len=512,
            heads=4,
            multihead=True,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=0.01,
        )

        inputs = self.validator.make_inputs(torch, args)
        lower = inputs["lower_mask"]
        eye = inputs["identity"]
        head_idx = 3
        state = inputs["state_in"][head_idx].clone()
        rel_scan2 = None
        rel_scan7 = None

        for start in range(0, args.seq_len, self.validator.P_MAX):
            end = start + self.validator.P_MAX
            _, key = self.validator.normalize_reference_qk(
                torch,
                inputs["query"][head_idx, start:end],
                inputs["key"][head_idx, start:end],
            )
            value = inputs["value"][head_idx, start:end]
            g = inputs["g_raw"][head_idx, start:end]
            beta = inputs["beta"][head_idx, start:end]

            gc = torch.cumsum(g, dim=0)
            k_beta = key * beta
            v_beta = value * beta
            decay = self.validator.stable_causal_decay(torch, gc, lower)
            a_mat = -((k_beta @ key.T) * decay) * lower
            lhs = eye - a_mat
            rhs = v_beta - ((k_beta * torch.exp(gc)) @ state)

            expected = torch.linalg.solve_triangular(lhs, rhs, upper=False)
            if start == 256:
                scan2 = self.validator.scan_doubling_lower_triangular_solve(
                    torch,
                    lhs,
                    rhs,
                    steps=2,
                )
                scan7 = self.validator.scan_doubling_lower_triangular_solve(
                    torch,
                    lhs,
                    rhs,
                    steps=7,
                )
                rel_scan2 = (
                    torch.linalg.vector_norm(scan2 - expected)
                    / torch.linalg.vector_norm(expected)
                ).item()
                rel_scan7 = (
                    torch.linalg.vector_norm(scan7 - expected)
                    / torch.linalg.vector_norm(expected)
                ).item()
                break

            gl = gc[-1:]
            key_decay = key * torch.exp(gl - gc)
            state = (state * torch.exp(gl)) + (key_decay.T @ expected)

        self.assertIsNotNone(rel_scan2)
        self.assertIsNotNone(rel_scan7)
        self.assertGreater(rel_scan2, 5.0e-3)
        self.assertLess(rel_scan7, 5.0e-6)

    def test_blocked_reference_matches_current_reference(self):
        args = types.SimpleNamespace(
            seed=1234,
            seq_len=256,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        actual_output, actual_state = self.validator.blocked_reference_math(
            torch,
            inputs,
            block_size=16,
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_blocked_reference_matches_current_reference_multihead(self):
        args = types.SimpleNamespace(
            seed=1234,
            seq_len=256,
            heads=4,
            multihead=True,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        expected_output, expected_state = self.validator.reference_math(torch, inputs)
        actual_output, actual_state = self.validator.blocked_reference_math(
            torch,
            inputs,
            block_size=16,
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=2.0e-5,
            rtol=2.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
