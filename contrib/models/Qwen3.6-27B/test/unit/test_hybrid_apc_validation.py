# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_VALIDATION_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_hybrid_apc_validation.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_hybrid_apc_validation_under_test",
    _VALIDATION_PATH,
)
_VALIDATION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VALIDATION
_SPEC.loader.exec_module(_VALIDATION)


def _args(**overrides):
    defaults = {
        "shared_prefix": "shared",
        "suffix_a": " suffix a",
        "suffix_b": " suffix b",
        "shared_prefix_2": "shared two",
        "suffix_c": " suffix c",
        "suffix_d": " suffix d",
        "max_num_seqs": 2,
        "max_tokens": 8,
        "compiled_artifacts": None,
        "model_path": "/tmp/model",
        "cte_buckets": ["256,512"],
        "align_prompts_to_cte_buckets": False,
        "require_real_tokens": True,
        "dummy_token_ids": [0],
        "output_json": None,
        "block_size": 256,
        "gdn_checkpoint_interval": 256,
        "seq_len": 2048,
        "compact_boundary_lens": None,
        "compact_suffix_tokens": 16,
        "compact_min_requests": 50,
        "compact_min_grouped_speedup": 1.5,
        "hybrid_apc_require_vllm_metadata": True,
        "hybrid_apc_disable_unbacked_prefix_reads": True,
        "hybrid_apc_enable_backed_prefix_reads": True,
        "hybrid_apc_max_backed_prefix_read_len": 0,
        "max_gdn_checkpoint_slots": 8,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_generate_batch(tokens_by_label):
    def fake_generate_batch(_args, *, enable_hybrid_apc, labeled_prompts):
        return {
            label: {
                "tokens": list(tokens_by_label[label]),
                "elapsed_seconds": 0.01,
            }
            for label, _prompt in labeled_prompts
        }

    return fake_generate_batch


def _fake_generate_grouped_batch(tokens_by_label):
    def fake_generate_grouped_batch(
        _args,
        *,
        enable_hybrid_apc,
        labeled_prompt_groups,
    ):
        results = {}
        for group in labeled_prompt_groups:
            for label, _prompt in group:
                results[label] = {
                    "tokens": list(tokens_by_label[label]),
                    "elapsed_seconds": 0.01,
                }
        return results

    return fake_generate_grouped_batch


def _compact_reference_label(label):
    if label.startswith("warm_full_"):
        return "cold_full_" + label[len("warm_full_") :]
    if label.startswith("warm_partial_"):
        return "cold_partial_" + label[len("warm_partial_") :]
    if label.startswith("mixed_warm_"):
        return "cold_partial_" + label[len("mixed_warm_") :]
    if label.startswith("eviction_probe_partial_"):
        return "cold_partial_" + label[len("eviction_probe_partial_") :]
    if label.startswith("mixed_cold__"):
        return "cold_mixed__" + label[len("mixed_cold__") :]
    if label.startswith("warmup"):
        return label
    return label


def _fake_compact_generate_batch(_args, *, enable_hybrid_apc, labeled_prompts):
    del _args, enable_hybrid_apc
    return {
        label: {
            "tokens": [sum(ord(ch) for ch in _compact_reference_label(label)) % 997 + 1],
            "elapsed_seconds": 2.0,
        }
        for label, _prompt in labeled_prompts
    }


def _fake_compact_generate_grouped_batch(
    _args,
    *,
    enable_hybrid_apc,
    labeled_prompt_groups,
):
    del _args, enable_hybrid_apc
    results = {}
    for group in labeled_prompt_groups:
        elapsed = 1.0 if len(group) > 1 else 0.5
        for label, _prompt in group:
            results[label] = {
                "tokens": [
                    sum(ord(ch) for ch in _compact_reference_label(label)) % 997 + 1
                ],
                "elapsed_seconds": elapsed,
            }
    return results


class TestHybridAPCValidationRealTokens(unittest.TestCase):
    def test_bucket_alignment_pads_prompt_token_ids(self):
        class FakeTokenizer:
            pad_token_id = 99
            eos_token_id = None

            def encode(self, prompt, add_special_tokens=False):
                del add_special_tokens
                return list(range(len(prompt.split())))

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(_model_path, trust_remote_code):
                del trust_remote_code
                return FakeTokenizer()

        fake_transformers = SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            aligned = _VALIDATION._maybe_bucket_align_labeled_prompts(
                _args(
                    cte_buckets=["4,8"],
                    align_prompts_to_cte_buckets=True,
                ),
                [("prompt", "one two three")],
            )

        self.assertEqual(
            aligned,
            [("prompt", {"prompt_token_ids": [0, 1, 2, 99]})],
        )

    def test_bucket_alignment_rejects_too_long_prompt(self):
        with self.assertRaisesRegex(ValueError, "exceeds compiled CTE buckets"):
            _VALIDATION._next_bucket(9, [4, 8])

    def test_real_token_checks_fail_all_dummy_tokens(self):
        checks = _VALIDATION._real_token_checks(
            {
                "cold_full": {"tokens": [0, 0, 0]},
                "warm_full": {"tokens": [0, 0, 0]},
            },
            {0},
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(
            checks["checks"]["cold_full"]["failure"],
            "generated tokens are empty or all configured dummy tokens",
        )

    def test_exactness_can_require_non_dummy_generated_tokens(self):
        tokens_by_label = {
            "cold_full": [0, 0],
            "cold_partial": [0, 0],
            "warmup_full": [0, 0],
            "warm_full": [0, 0],
            "warmup_partial": [0, 0],
            "warm_partial": [0, 0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "report.json"
            with patch.object(
                _VALIDATION,
                "_generate_batch",
                side_effect=_fake_generate_batch(tokens_by_label),
            ):
                rc = _VALIDATION.run_exactness(
                    _args(output_json=output_json),
                )

            self.assertEqual(rc, 1)
            report = output_json.read_text(encoding="utf-8")
            self.assertIn('"full_prefix_exact": true', report)
            self.assertIn('"partial_prefix_exact": true', report)
            self.assertIn('"real_generated_tokens_passed": false', report)

    def test_exactness_passes_when_real_tokens_are_present(self):
        tokens_by_label = {
            "cold_full": [42, 0],
            "cold_partial": [43, 0],
            "warmup_full": [42, 0],
            "warm_full": [42, 0],
            "warmup_partial": [42, 0],
            "warm_partial": [43, 0],
        }
        with patch.object(
            _VALIDATION,
            "_generate_batch",
            side_effect=_fake_generate_batch(tokens_by_label),
        ):
            rc = _VALIDATION.run_exactness(_args())

        self.assertEqual(rc, 0)

    def test_batched_exactness_checks_two_concurrent_partials(self):
        tokens_by_label = {
            "cold_partial_a": [42, 0],
            "cold_partial_b": [43, 0],
            "warmup_full_a": [44, 0],
            "warmup_full_b": [45, 0],
            "warm_partial_a": [42, 0],
            "warm_partial_b": [43, 0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "batched_report.json"
            with patch.object(
                _VALIDATION,
                "_generate_batch",
                side_effect=_fake_generate_batch(tokens_by_label),
            ):
                with patch.object(
                    _VALIDATION,
                    "_generate_grouped_batch",
                    side_effect=_fake_generate_grouped_batch(tokens_by_label),
                ):
                    rc = _VALIDATION.run_batched_exactness(
                        _args(output_json=output_json)
                    )

            self.assertEqual(rc, 0)
            report = output_json.read_text(encoding="utf-8")
            self.assertIn('"batched_partial_a_exact": true', report)
            self.assertIn('"batched_partial_b_exact": true', report)
            self.assertIn('"max_num_seqs": 2', report)

    def test_batched_exactness_requires_second_prefix(self):
        with self.assertRaisesRegex(ValueError, "--shared-prefix-2 is required"):
            _VALIDATION.run_batched_exactness(_args(shared_prefix_2=""))

    def test_batched_exactness_preflights_tkg_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "neuron_config.json"
            config_path.write_text(
                json.dumps(
                    {"neuron_config": {"tkg_batch_size": 1, "ctx_batch_size": 2}}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "tkg_batch_size=1 and max_num_seqs=2",
            ):
                _VALIDATION.run_batched_exactness(
                    _args(compiled_artifacts=tmpdir, max_num_seqs=2)
                )

    def test_batched_exactness_preflights_ctx_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "neuron_config.json"
            config_path.write_text(
                json.dumps(
                    {"neuron_config": {"tkg_batch_size": 2, "ctx_batch_size": 1}}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "ctx_batch_size=1 and max_num_seqs=2",
            ):
                _VALIDATION.run_batched_exactness(
                    _args(compiled_artifacts=tmpdir, max_num_seqs=2)
                )

    def test_runtime_additional_config_uses_compiled_max_prompt_length(self):
        additional_config = {
            "max_prompt_length": 512,
            "override_neuron_config": {
                "max_context_length": 512,
                "context_encoding_buckets": [256, 512],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "neuron_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "neuron_config": {
                            "max_context_length": 131072,
                            "context_encoding_buckets": [256, 512],
                        }
                    }
                ),
                encoding="utf-8",
            )

            aligned = _VALIDATION._align_additional_config_to_compiled_artifact(
                _args(compiled_artifacts=tmpdir),
                additional_config,
            )

        self.assertEqual(aligned["max_prompt_length"], 131072)
        self.assertEqual(
            aligned["override_neuron_config"]["max_context_length"],
            131072,
        )
        self.assertEqual(
            aligned["override_neuron_config"]["context_encoding_buckets"],
            [256, 512],
        )
        self.assertEqual(additional_config["max_prompt_length"], 512)

    def test_compact_boundary_lengths_cover_checkpoint_edges(self):
        self.assertEqual(
            _VALIDATION._compact_boundary_lengths(
                _args(block_size=4, seq_len=32, max_tokens=1, compact_suffix_tokens=2)
            ),
            [3, 4, 5, 7, 8, 9],
        )

    def test_compact_gate_requires_strict_metadata(self):
        with self.assertRaisesRegex(ValueError, "requires --hybrid-apc-require"):
            _VALIDATION.run_compact_gate(
                _args(hybrid_apc_require_vllm_metadata=False)
            )

    def test_compact_gate_reports_targeted_exactness_and_speedup(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def encode(self, prompt, add_special_tokens=False):
                del add_special_tokens
                return list(range(len(prompt.split())))

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(_model_path, trust_remote_code):
                del trust_remote_code
                return FakeTokenizer()

        fake_transformers = SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "compact.json"
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                with patch.object(
                    _VALIDATION,
                    "_generate_batch",
                    side_effect=_fake_compact_generate_batch,
                ):
                    with patch.object(
                        _VALIDATION,
                        "_generate_grouped_batch",
                        side_effect=_fake_compact_generate_grouped_batch,
                    ):
                        rc = _VALIDATION.run_compact_gate(
                            _args(
                                block_size=4,
                                gdn_checkpoint_interval=4,
                                compact_boundary_lens=["3,4"],
                                compact_suffix_tokens=2,
                                compact_min_requests=20,
                                output_json=output_json,
                            )
                        )

            self.assertEqual(rc, 0)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(report["compact_gate_passed"])
            self.assertEqual(report["boundary_lengths"], [3, 4])
            self.assertGreaterEqual(report["acceptance"]["request_count"], 20)
            self.assertTrue(report["acceptance"]["exactness_passed"])
            self.assertTrue(report["acceptance"]["speedup_passed"])


if __name__ == "__main__":
    unittest.main()
