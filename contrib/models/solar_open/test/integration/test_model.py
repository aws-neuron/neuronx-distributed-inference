"""Integration tests for NeuronSolarOpenForCausalLM.

These tests require Neuron hardware (NeuronCores). The `compiled_model` fixture
in conftest.py skips automatically when Neuron hardware is unavailable.

Solar Open is NOT in transformers — logit accuracy uses the custom
SolarOpenReferenceModel (pure PyTorch CPU) for comparison.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure contrib src is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSolarOpenSmoke:
    """Basic model sanity checks (no inference)."""

    def test_model_is_not_none(self, compiled_model):
        """Compiled model must be a non-None object."""
        assert compiled_model is not None

    def test_model_has_config(self, compiled_model):
        """Compiled model must expose a config with neuron_config."""
        assert hasattr(compiled_model, "config")
        assert hasattr(compiled_model.config, "neuron_config")

    def test_neuron_config_tp_degree(self, compiled_model):
        """tp_degree must be 2 (as set in get_neuron_config())."""
        assert compiled_model.config.neuron_config.tp_degree == 2


# ---------------------------------------------------------------------------
# Logit accuracy test
# ---------------------------------------------------------------------------


class TestSolarOpenAccuracy:
    """Logit accuracy: Neuron model vs CPU reference."""

    def test_check_accuracy_logits(
        self, compiled_model, model_dir, traced_dir, neuron_config
    ):
        """CPU reference and Neuron model logits should match within tolerance.

        Tolerance of 0.05 MAE accounts for bfloat16 rounding and Neuron's
        hardware-optimised fused operations while catching large discrepancies.
        """
        from test.integration.utils import check_logit_accuracy

        passed = check_logit_accuracy(
            model_dir=model_dir,
            traced_dir=traced_dir,
            neuron_config=neuron_config,
            tol=0.05,
        )
        assert passed, (
            "Logit MAE exceeds tolerance — check weight loading or compute graph"
        )


# ---------------------------------------------------------------------------
# Performance test (context encoding)
# ---------------------------------------------------------------------------


class TestSolarOpenPerformance:
    """Lightweight performance checks (context encoding runs without error)."""

    def test_context_encoding_runs(self, compiled_model, solar_open_config_dict):
        """Context encoding must complete without raising an exception."""
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )
        from transformers import GenerationConfig

        vocab_size = solar_open_config_dict["vocab_size"]
        seq_len = compiled_model.config.neuron_config.seq_len
        batch_size = compiled_model.config.neuron_config.max_batch_size

        torch.manual_seed(42)
        input_ids = torch.randint(
            0, min(vocab_size, 1000), (batch_size, min(seq_len // 2, 32))
        )
        attention_mask = torch.ones_like(input_ids)

        adapter = HuggingFaceGenerationAdapter(compiled_model)
        gen_config = GenerationConfig(do_sample=False, top_k=1, max_new_tokens=4)

        outputs = adapter.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=compiled_model.config.neuron_config.max_length,
        )
        assert outputs is not None
        assert outputs.shape[1] > input_ids.shape[1], (
            "Model must generate at least one new token"
        )
