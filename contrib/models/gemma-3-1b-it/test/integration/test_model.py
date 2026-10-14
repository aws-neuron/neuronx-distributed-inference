#!/usr/bin/env python3
"""
Integration test for Gemma 3 1B IT on NeuronX.

This test compiles and runs the model using the contrib's subclassed
implementation, then verifies that it generates coherent text.

Usage (on a Neuron instance):

    cd neuronx-distributed-inference
    PYTHONPATH="contrib/models/gemma-3-1b-it/src:src:$PYTHONPATH" \
        python contrib/models/gemma-3-1b-it/test/integration/test_model.py

Or with pytest:

    PYTHONPATH="contrib/models/gemma-3-1b-it/src:src:$PYTHONPATH" \
        pytest contrib/models/gemma-3-1b-it/test/integration/test_model.py -v --capture=tee-sys
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Ensure contrib src is on the path.
_CONTRIB_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")
if _CONTRIB_SRC not in sys.path:
    sys.path.insert(0, _CONTRIB_SRC)

from modeling_gemma3 import Gemma3_1B_InferenceConfig, NeuronGemma3_1B_ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("GEMMA3_1B_MODEL_PATH", "google/gemma-3-1b-it")
COMPILED_MODEL_PATH = os.environ.get(
    "GEMMA3_1B_COMPILED_PATH", "/tmp/gemma3-1b-it-compiled"
)

# Neuron config matching the validated working configuration.
NEURON_CONFIG_KWARGS = dict(
    tp_degree=1,
    batch_size=1,
    seq_len=512,
    max_context_length=512,
    torch_dtype=torch.bfloat16,
    attn_kernel_enabled=False,
    k_cache_transposed=True,
    on_device_sampling_config=None,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (if needed) and load the model."""
    neuron_config = NeuronConfig(**NEURON_CONFIG_KWARGS)
    config = Gemma3_1B_InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    model = NeuronGemma3_1B_ForCausalLM(MODEL_PATH, config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"\nCompiling model to {COMPILED_MODEL_PATH} ...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    print(f"Loading model from {COMPILED_MODEL_PATH} ...")
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate(model, input_ids, max_new_tokens: int = 20) -> torch.Tensor:
    """Autoregressive generation via forward-loop."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        seq_len = generated.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated.shape[0], -1)
        with torch.no_grad():
            outputs = model(generated, position_ids=position_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke(compiled_model):
    """Model loads without errors."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    print("PASS: smoke test")


def test_generates_text(compiled_model, tokenizer):
    """Model generates non-empty, non-trivial output."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = generate(compiled_model, inputs.input_ids, max_new_tokens=20)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    assert len(text) > len(prompt), f"Output not longer than prompt: {text!r}"
    print(f"PASS: generates text\n  Output: {text}")


def test_coherence(compiled_model, tokenizer):
    """Output is coherent (not gibberish or degenerate repetition)."""
    prompt = "Explain what a neural network is in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = generate(compiled_model, inputs.input_ids, max_new_tokens=40)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    words = text.split()
    assert len(words) > 8, f"Too few words: {text!r}"

    # Check for degenerate repetition (same word 6+ times in a row).
    for i in range(len(words) - 5):
        assert not all(words[i + j] == words[i] for j in range(6)), (
            f"Degenerate repetition: {text!r}"
        )

    print(f"PASS: coherence\n  Output: {text[:120]}...")


def test_vocab_size(compiled_model):
    """Config has the correct 1B vocab_size (262144, not 262208)."""
    assert compiled_model.config.vocab_size == 262144, (
        f"Expected 262144, got {compiled_model.config.vocab_size}"
    )
    print(f"PASS: vocab_size = {compiled_model.config.vocab_size}")


def test_head_dim(compiled_model):
    """Config has head_dim=256."""
    head_dim = getattr(compiled_model.config, "head_dim", None)
    assert head_dim == 256, f"Expected head_dim=256, got {head_dim}"
    print(f"PASS: head_dim = {head_dim}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Gemma 3 1B IT -- Integration Test")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    neuron_config = NeuronConfig(**NEURON_CONFIG_KWARGS)
    config = Gemma3_1B_InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    model = NeuronGemma3_1B_ForCausalLM(MODEL_PATH, config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"\nCompiling to {COMPILED_MODEL_PATH} ...")
        t0 = time.time()
        model.compile(COMPILED_MODEL_PATH)
        print(f"Compiled in {time.time() - t0:.1f}s")

    print(f"\nLoading from {COMPILED_MODEL_PATH} ...")
    model.load(COMPILED_MODEL_PATH)
    print("Loaded.\n")

    print("1. Smoke test ...")
    test_smoke(model)

    print("\n2. Vocab size ...")
    test_vocab_size(model)

    print("\n3. Head dim ...")
    test_head_dim(model)

    print("\n4. Generation ...")
    test_generates_text(model, tok)

    print("\n5. Coherence ...")
    test_coherence(model, tok)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
