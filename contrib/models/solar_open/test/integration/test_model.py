"""Integration tests for NeuronSolarOpenForCausalLM.

These tests require Neuron hardware (NeuronCores). The `compiled_model` fixture
in conftest.py skips automatically when Neuron hardware is unavailable.

Text-generation accuracy is verified by comparing the token IDs produced by
transformers.SolarOpenForCausalLM (CPU reference, transformers >= 5.0.0) and
NeuronSolarOpenForCausalLM using greedy decoding.
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
# Text-generation accuracy test
# ---------------------------------------------------------------------------


class TestSolarOpenAccuracy:
    """Text-generation accuracy: Neuron model vs transformers CPU reference."""

    def test_text_generation_matches_reference(
        self, compiled_model, model_dir, traced_dir, neuron_config
    ):
        """Last-token logits from Neuron and CPU reference (SolarOpenForCausalLM) must match.

        Compares via mean absolute error on last-token logit vectors.
        Exact token-ID matching is not used because Neuron's bfloat16 hardware
        arithmetic may produce borderline argmax differences on near-equal logits.
        Tolerance of 0.1 MAE accounts for bfloat16 rounding while catching large
        discrepancies that indicate weight loading or compute graph issues.
        """
        from test.integration.utils import check_text_accuracy

        passed = check_text_accuracy(
            model_dir=model_dir,
            traced_dir=traced_dir,
            neuron_config=neuron_config,
            tol=0.1,
        )
        assert passed, (
            "Logit MAE exceeds tolerance between CPU reference (SolarOpenForCausalLM) "
            "and Neuron model — check weight loading or compute graph"
        )


# ---------------------------------------------------------------------------
# Performance test (context encoding)
# ---------------------------------------------------------------------------


class TestSolarOpenPerformance:
    """Lightweight performance checks (context encoding runs without error)."""

    def test_context_encoding_runs(self, compiled_model):
        """Context encoding must complete without raising an exception."""
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )
        from transformers import GenerationConfig

        seq_len = compiled_model.config.neuron_config.seq_len
        batch_size = compiled_model.config.neuron_config.max_batch_size
        vocab_size = compiled_model.config.vocab_size

        torch.manual_seed(42)
        input_ids = torch.randint(
            0, min(vocab_size, 500), (batch_size, min(seq_len // 2, 32))
        )
        attention_mask = torch.ones_like(input_ids)

        adapter = HuggingFaceGenerationAdapter(compiled_model)
        # Ensure transformers_version is set to avoid TypeError in _prepare_generation_config
        if (
            hasattr(adapter, "generation_config")
            and adapter.generation_config is not None
            and adapter.generation_config.transformers_version is None
        ):
            adapter.generation_config.transformers_version = "5.0.0"

        gen_config = GenerationConfig(
            do_sample=False,
            top_k=1,
            max_new_tokens=4,
            transformers_version="5.0.0",
        )

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
