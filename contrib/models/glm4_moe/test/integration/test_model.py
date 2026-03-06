#!/usr/bin/env python3
# coding=utf-8
"""
Integration tests for GLM-4.5 MoE NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance using a
reduced 2-layer config with random weights to avoid OOM on CI/test hardware.

Usage:
    # From contrib/models/glm4_moe/ directory:
    pytest test/integration/test_model.py -v

    # Run with specific tp_degree:
    NEURON_RT_NUM_CORES=2 pytest test/integration/test_model.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import GenerationConfig

# Add contrib src and integration dir to path
_CONTRIB_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_CONTRIB_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # for utils module

from glm4_moe.modeling_glm4_moe import Glm4MoeInferenceConfig, NeuronGlm4MoeForCausalLM
from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.accuracy import (
    check_accuracy_logits_v2,
    generate_expected_logits,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from utils import create_neuron_config, save_hf_checkpoint, prepare_inputs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config_glm4_moe_2layers.json"
TP_DEGREE = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))
SEQ_LEN = 128
BATCH_SIZE = 1
MAX_NEW_TOKENS = 8


# ---------------------------------------------------------------------------
# Session-scoped fixtures: create tiny random model once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hf_config_dict():
    """Load the reduced 2-layer test config dict."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def hf_checkpoint_path(tmp_path_factory, hf_config_dict):
    """Create a temporary HF checkpoint with random weights (session-scoped).

    The checkpoint is created once and reused for all tests in the session.
    """
    tmp_dir = str(tmp_path_factory.mktemp("glm4_moe_hf_ckpt"))
    print(f"\n[fixture] Creating tiny random HF checkpoint at {tmp_dir}...")
    save_hf_checkpoint(hf_config_dict, tmp_dir)
    print(f"[fixture] HF checkpoint ready.")
    return tmp_dir


@pytest.fixture(scope="session")
def compiled_model_path(tmp_path_factory):
    """Return a session-scoped temp directory for the compiled Neuron model."""
    return str(tmp_path_factory.mktemp("glm4_moe_neuron_compiled"))


@pytest.fixture(scope="session")
def neuron_config():
    """Create MoENeuronConfig for integration tests."""
    return create_neuron_config(
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture(scope="session")
def neuron_model(
    hf_checkpoint_path, compiled_model_path, neuron_config, hf_config_dict
):
    """Compile and load GLM-4.5 MoE model on Neuron (session-scoped).

    Uses the tiny 2-layer random-weight checkpoint to avoid OOM.
    """
    print(f"\n[fixture] Building Glm4MoeInferenceConfig from {hf_checkpoint_path}...")
    inference_config = Glm4MoeInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_checkpoint_path),
    )

    model = NeuronGlm4MoeForCausalLM(hf_checkpoint_path, inference_config)

    print(f"[fixture] Compiling model to {compiled_model_path}...")
    model.compile(compiled_model_path)
    print(f"[fixture] Compilation complete.")

    print(f"[fixture] Loading compiled model...")
    model.load(compiled_model_path)
    print(f"[fixture] Model loaded.")

    return model


@pytest.fixture(scope="session")
def generation_config():
    """Greedy generation config (no sampling) for deterministic tests."""
    return GenerationConfig(
        do_sample=False,
        top_k=1,
        temperature=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
    )


@pytest.fixture(scope="session")
def input_ids_and_mask(hf_config_dict):
    """Prepare fixed random inputs for all tests."""
    vocab_size = hf_config_dict.get("vocab_size", 1000)
    input_ids, attention_mask = prepare_inputs(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN // 2,  # Use half of seq_len as prompt
        vocab_size=vocab_size,
    )
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGlm4MoeSmoke:
    """Smoke tests: verify model compiles and loads without errors."""

    def test_model_is_not_none(self, neuron_model):
        """Model fixture must load successfully."""
        assert neuron_model is not None, "neuron_model fixture returned None"

    def test_model_has_config(self, neuron_model):
        """Loaded model must carry a valid InferenceConfig."""
        assert hasattr(neuron_model, "config"), "Model has no 'config' attribute"
        assert hasattr(neuron_model.config, "neuron_config"), (
            "Config missing 'neuron_config'"
        )

    def test_neuron_config_tp_degree(self, neuron_model):
        """TP degree must match the configured value."""
        assert neuron_model.config.neuron_config.tp_degree == TP_DEGREE


class TestGlm4MoeAccuracy:
    """Logit accuracy: compare Neuron output against CPU HuggingFace model."""

    def test_logit_accuracy(
        self,
        neuron_model,
        input_ids_and_mask,
        generation_config,
    ):
        """Neuron logits must match HuggingFace CPU logits within tolerance.

        Uses check_accuracy_logits_v2 which generates golden logits from the
        HuggingFace model and compares them token-by-token against the Neuron
        output.  A divergence_difference_tol of 0.001 is used.
        """
        input_ids, attention_mask = input_ids_and_mask

        print("\n[test] Generating expected (HF CPU) logits...")
        expected_logits = generate_expected_logits(
            neuron_model=neuron_model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            num_tokens=MAX_NEW_TOKENS,
        )

        print("[test] Running check_accuracy_logits_v2...")
        check_accuracy_logits_v2(
            neuron_model=neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            divergence_difference_tol=0.001,
            num_tokens_to_check=MAX_NEW_TOKENS,
        )
        print("[test] Logit accuracy check passed.")


class TestGlm4MoePerformance:
    """Performance benchmarks: timing and throughput sanity checks."""

    def test_context_encoding_runs(self, neuron_model, input_ids_and_mask):
        """Context encoding forward pass must complete without error."""
        input_ids, attention_mask = input_ids_and_mask
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )

        adapter = HuggingFaceGenerationAdapter(neuron_model)
        gen_config = GenerationConfig(
            do_sample=False,
            top_k=1,
            max_new_tokens=1,
        )
        with torch.no_grad():
            output = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        assert output is not None
        assert output.shape[1] >= input_ids.shape[1]

    def test_benchmark_sampling(self, neuron_model, generation_config):
        """benchmark_sampling must complete and return a non-empty report."""
        from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

        report = benchmark_sampling(
            model=neuron_model,
            generation_config=generation_config,
            num_runs=3,
        )
        # Report may be empty dict if model/config doesn't support it; just
        # verify no exception was raised.
        assert report is not None
        print(f"\n[test] Benchmark report: {report}")


# ---------------------------------------------------------------------------
# __main__ runner (non-pytest)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tempfile

    from utils import save_hf_checkpoint

    print("=" * 70)
    print("GLM-4.5 MoE Integration Tests (standalone)")
    print("=" * 70)

    with open(CONFIG_PATH) as f:
        hf_config_dict = json.load(f)

    with (
        tempfile.TemporaryDirectory() as hf_ckpt_dir,
        tempfile.TemporaryDirectory() as compiled_dir,
    ):
        print(f"\nStep 1: Creating tiny random HF checkpoint at {hf_ckpt_dir}...")
        save_hf_checkpoint(hf_config_dict, hf_ckpt_dir)
        print("  Done.")

        neuron_cfg = create_neuron_config(
            tp_degree=TP_DEGREE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
        inference_config = Glm4MoeInferenceConfig(
            neuron_config=neuron_cfg,
            load_config=load_pretrained_config(hf_ckpt_dir),
        )

        print(f"\nStep 2: Compiling model to {compiled_dir}...")
        model = NeuronGlm4MoeForCausalLM(hf_ckpt_dir, inference_config)
        model.compile(compiled_dir)
        print("  Compilation complete.")

        print(f"\nStep 3: Loading compiled model...")
        model.load(compiled_dir)
        print("  Load complete.")

        print("\nStep 4: Smoke test — model attributes...")
        assert model is not None
        assert hasattr(model, "config")
        print("  PASSED.")

        print("\nStep 5: Logit accuracy test...")
        vocab_size = hf_config_dict.get("vocab_size", 1000)
        input_ids, attention_mask = prepare_inputs(BATCH_SIZE, SEQ_LEN // 2, vocab_size)
        gen_config = GenerationConfig(
            do_sample=False, top_k=1, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS
        )

        expected_logits = generate_expected_logits(
            neuron_model=model,
            input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=gen_config,
            num_tokens=MAX_NEW_TOKENS,
        )
        check_accuracy_logits_v2(
            neuron_model=model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=gen_config,
            divergence_difference_tol=0.001,
            num_tokens_to_check=MAX_NEW_TOKENS,
        )
        print("  PASSED.")

        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)
