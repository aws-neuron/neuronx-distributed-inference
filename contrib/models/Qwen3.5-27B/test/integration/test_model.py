# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Qwen3.5-27B on Neuron.

Tests compilation, loading, inference accuracy, and performance using
the full 27B model with pre-downloaded HuggingFace weights on a trn2 instance.

Note: A mini model option is not provided because DeltaNet layers require NKI
kernels that only execute on Neuron devices, and the hybrid DeltaNet + GQA
architecture needs at least TP=4 for the full model to fit in HBM.

Environment variables:
    QWEN35_MODEL_PATH       Path to HF model weights (required)
    QWEN35_COMPILED_PATH    Path to compiled artifacts (default: /tmp/qwen35_27b_traced)
    QWEN35_TP_DEGREE        Tensor parallelism degree (default: 4)
    QWEN35_SEQ_LEN          Max sequence length (default: 128)
    TTFT_THRESHOLD_MS       Max TTFT in ms (default: 5000)
    THROUGHPUT_THRESHOLD     Min throughput in tok/s (default: 5.0)

Prerequisites:
    - trn2.3xlarge or larger with TP >= 4 NeuronCores available
    - NXDI installed (neuronx_distributed_inference)
    - HuggingFace weights downloaded to QWEN35_MODEL_PATH
    - SDK 2.29+ (NKI 0.3.0 required for DeltaNet kernels)

Usage:
    # Full model (trn2.3xlarge, TP=4):
    QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-27B \\
    QWEN35_COMPILED_PATH=/mnt/models/qwen35_traced \\
    pytest test/integration/test_model.py --capture=tee-sys
"""

import gc
import os
import sys
import time

import pytest
import torch

# Ensure the contrib root (Qwen3.5-27B/) is on sys.path
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

# ── Configuration from environment ──────────────────────────────────────

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "")
COMPILED_PATH = os.environ.get("QWEN35_COMPILED_PATH", "/tmp/qwen35_27b_traced")
TP_DEGREE = int(os.environ.get("QWEN35_TP_DEGREE", "4"))
SEQ_LEN = int(os.environ.get("QWEN35_SEQ_LEN", "128"))
TTFT_THRESHOLD_MS = float(os.environ.get("TTFT_THRESHOLD_MS", "5000"))
THROUGHPUT_THRESHOLD = float(os.environ.get("THROUGHPUT_THRESHOLD", "5.0"))

requires_model_path = pytest.mark.skipif(
    not MODEL_PATH,
    reason=(
        "QWEN35_MODEL_PATH not set. Integration tests require the full 27B model "
        "weights. Set QWEN35_MODEL_PATH=/path/to/Qwen3.5-27B to run these tests."
    ),
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_path():
    """Return path to model weights."""
    return MODEL_PATH


@pytest.fixture(scope="module")
def compiled_model(model_path):
    """Compile and load the model on Neuron."""
    import json

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )

    # Read config.json directly (model_type 'qwen3_5' may not be in
    # AutoConfig registry for all transformers versions)
    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", False)

    inf_config = Qwen35InferenceConfig(
        neuron_config=neuron_config,
        **config_dict,
    )

    # Compile if no existing artifacts
    compiled_path = COMPILED_PATH
    neff_path = os.path.join(compiled_path, "model.pt")
    if not os.path.exists(neff_path):
        print(f"Compiling to {compiled_path}...")
        model = NeuronQwen35ForCausalLM(model_path, inf_config)
        model.compile(compiled_path)
        del model
        gc.collect()

    # Load
    print(f"Loading from {compiled_path}...")
    model = NeuronQwen35ForCausalLM(compiled_path)
    model.load(compiled_path)
    return model


@pytest.fixture(scope="module")
def tokenizer(model_path):
    """Load tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def generation_config(tokenizer):
    """Create generation config."""
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def _generate(model, tokenizer, generation_config, prompt, max_new_tokens=20):
    """Generate text using the NXDI model."""
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    gen_model = HuggingFaceGenerationAdapter(model)
    outputs = gen_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    return outputs[0].tolist(), tokenizer.decode(outputs[0], skip_special_tokens=True)


def _is_repetitive(text, max_repeat=5):
    """Check for excessive word repetition."""
    words = text.split()
    if len(words) < max_repeat:
        return False
    for i in range(len(words) - max_repeat + 1):
        if len(set(words[i : i + max_repeat])) == 1:
            return True
    return False


# ── Smoke Tests ─────────────────────────────────────────────────────────


@requires_model_path
def test_model_loads(compiled_model):
    """Model compiles and loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "neuron_config")
    print("  Model loaded successfully")


@requires_model_path
def test_model_generates(compiled_model, tokenizer, generation_config):
    """Model generates at least 5 tokens."""
    tokens, text = _generate(
        compiled_model,
        tokenizer,
        generation_config,
        "Hello, I am a language model",
        max_new_tokens=20,
    )
    input_len = len(tokenizer.encode("Hello, I am a language model"))
    new_tokens = len(tokens) - input_len
    assert new_tokens >= 5, f"Expected >= 5 new tokens, got {new_tokens}"
    print(f"  Generated {new_tokens} tokens: {text[:100]}...")


# ── Accuracy Tests ──────────────────────────────────────────────────────


@requires_model_path
def test_output_coherence(compiled_model, tokenizer, generation_config):
    """Output should contain multiple words and not be excessively repetitive."""
    _, text = _generate(
        compiled_model,
        tokenizer,
        generation_config,
        "The capital of France is",
        max_new_tokens=30,
    )
    generated = text[len("The capital of France is") :].strip()
    words = generated.split()
    assert len(words) >= 3, f"Expected >= 3 words, got {len(words)}: '{generated}'"
    assert not _is_repetitive(generated), (
        f"Output is excessively repetitive: '{generated}'"
    )
    print(f"  Output coherent ({len(words)} words): {generated[:80]}...")


@requires_model_path
def test_top_token_valid(compiled_model, tokenizer, generation_config):
    """First generated token should be a valid decodable token."""
    tokens, _ = _generate(
        compiled_model,
        tokenizer,
        generation_config,
        "Hello!",
        max_new_tokens=1,
    )
    input_len = len(tokenizer.encode("Hello!"))
    first_new = tokens[input_len]
    assert 0 <= first_new < tokenizer.vocab_size, (
        f"Token {first_new} out of vocab range"
    )
    decoded = tokenizer.decode([first_new])
    assert len(decoded) > 0, f"Token {first_new} decoded to empty string"
    print(f"  First token: {first_new} -> '{decoded}'")


@requires_model_path
def test_capital_of_france(compiled_model, tokenizer, generation_config):
    """'The capital of France is' should produce 'Paris' as first token."""
    tokens, text = _generate(
        compiled_model,
        tokenizer,
        generation_config,
        "The capital of France is",
        max_new_tokens=5,
    )
    generated = text[len("The capital of France is") :].strip()
    assert "paris" in generated.lower(), (
        f"Expected 'Paris' in output, got: '{generated}'"
    )
    print(f"  Capital of France: {generated}")


# ── Performance Tests ───────────────────────────────────────────────────


@requires_model_path
def test_performance_ttft(compiled_model, tokenizer, generation_config):
    """Time to first token should be within threshold."""
    prompt = "Hello, I am a language model"

    # Warmup
    _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=1)

    # Measure
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _generate(
            compiled_model, tokenizer, generation_config, prompt, max_new_tokens=1
        )
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"  TTFT: {avg_ms:.1f} ms (threshold: {TTFT_THRESHOLD_MS} ms)")
    assert avg_ms < TTFT_THRESHOLD_MS, (
        f"TTFT {avg_ms:.1f}ms > threshold {TTFT_THRESHOLD_MS}ms"
    )


@requires_model_path
def test_performance_throughput(compiled_model, tokenizer, generation_config):
    """Throughput should meet minimum threshold."""
    prompt = "Once upon a time"
    num_new_tokens = 20

    # Warmup
    _generate(compiled_model, tokenizer, generation_config, prompt, max_new_tokens=5)

    # Measure
    t0 = time.perf_counter()
    tokens, _ = _generate(
        compiled_model,
        tokenizer,
        generation_config,
        prompt,
        max_new_tokens=num_new_tokens,
    )
    elapsed = time.perf_counter() - t0

    input_len = len(tokenizer.encode(prompt))
    actual_new = len(tokens) - input_len
    throughput = actual_new / elapsed if elapsed > 0 else 0

    print(
        f"  Throughput: {throughput:.1f} tok/s ({actual_new} tokens in {elapsed:.2f}s)"
    )
    print(f"  Threshold: {THROUGHPUT_THRESHOLD} tok/s")
    assert throughput > THROUGHPUT_THRESHOLD, (
        f"Throughput {throughput:.1f} tok/s < threshold {THROUGHPUT_THRESHOLD}"
    )


# ── Multi-Prompt Quality Test ──────────────────────────────────────────


@requires_model_path
def test_multi_prompt_generation(compiled_model, tokenizer, generation_config):
    """Multiple prompts should produce coherent outputs."""
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "The largest ocean on Earth is",
        "To make a chocolate cake, you need",
    ]

    for prompt in prompts:
        _, text = _generate(
            compiled_model,
            tokenizer,
            generation_config,
            prompt,
            max_new_tokens=30,
        )
        generated = text[len(prompt) :].strip()
        words = generated.split()
        assert len(words) >= 2, (
            f"Prompt '{prompt}' generated too few words: '{generated}'"
        )
        assert not _is_repetitive(generated), (
            f"Prompt '{prompt}' produced repetitive output: '{generated}'"
        )
        print(f"  '{prompt[:30]}...' -> {generated[:60]}...")


# ── Standalone runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3.5-27B Integration Tests")
    print("=" * 60)

    if not MODEL_PATH:
        print("\nQWEN35_MODEL_PATH not set. Provide the model path to run tests:")
        print("  QWEN35_MODEL_PATH=/path/to/Qwen3.5-27B \\")
        print("  QWEN35_COMPILED_PATH=/mnt/models/qwen35_traced \\")
        print("  python -m pytest test/integration/test_model.py --capture=tee-sys")
        sys.exit(0)

    # Setup
    from transformers import AutoTokenizer, GenerationConfig as GenConfig

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_cfg = GenConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # Build model
    import json

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

    nc = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", False)
    ic = Qwen35InferenceConfig(neuron_config=nc, **config_dict)

    cp = COMPILED_PATH
    if not os.path.exists(os.path.join(cp, "model.pt")):
        print(f"Compiling to {cp}...")
        m = NeuronQwen35ForCausalLM(MODEL_PATH, ic)
        m.compile(cp)
        del m
        gc.collect()

    print(f"Loading from {cp}...")
    model = NeuronQwen35ForCausalLM(cp)
    model.load(cp)

    tests = [
        ("model_loads", lambda: test_model_loads(model)),
        ("model_generates", lambda: test_model_generates(model, tok, gen_cfg)),
        ("output_coherence", lambda: test_output_coherence(model, tok, gen_cfg)),
        ("top_token_valid", lambda: test_top_token_valid(model, tok, gen_cfg)),
        ("capital_of_france", lambda: test_capital_of_france(model, tok, gen_cfg)),
        ("performance_ttft", lambda: test_performance_ttft(model, tok, gen_cfg)),
        (
            "performance_throughput",
            lambda: test_performance_throughput(model, tok, gen_cfg),
        ),
        (
            "multi_prompt_generation",
            lambda: test_multi_prompt_generation(model, tok, gen_cfg),
        ),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'=' * 60}")
