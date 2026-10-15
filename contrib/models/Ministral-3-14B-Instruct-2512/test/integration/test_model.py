#!/usr/bin/env python3
"""
Integration tests for Ministral-3-14B-Instruct-2512 (Leanstral) on NeuronX.

Tests require:
  - trn2.3xlarge instance with SDK 2.28
  - NEURON_PLATFORM_TARGET_OVERRIDE=trn2
  - Model checkpoint at MODEL_PATH
  - Pre-compiled model at COMPILED_MODEL_PATH (or will compile on first run)

Run:
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    export NEURON_COMPILE_CACHE_URL=""
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    pytest test/integration/test_model.py -v --capture=tee-sys
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_leanstral import build_inference_config, get_model_cls

# ---- Configuration ----
# Override via environment variables if needed
MODEL_PATH = os.environ.get(
    "LEANSTRAL_MODEL_PATH", "/mnt/models/Ministral-3-14B-Instruct-2512"
)
COMPILED_MODEL_PATH = os.environ.get(
    "LEANSTRAL_COMPILED_PATH", "/mnt/models/compiled_leanstral_contrib"
)
TP_DEGREE = int(os.environ.get("LEANSTRAL_TP_DEGREE", "4"))
SEQ_LEN = 2048
N_POSITIONS = 4096
VISION_SEQ_LEN = 4096

TEXT_PROMPT = (
    "The theory of general relativity, proposed by Albert Einstein in 1915, "
    "fundamentally changed"
)
NUM_DECODE_STEPS = 10


# ---- Fixtures ----


@pytest.fixture(scope="module")
def compiled_model():
    """Build, compile (if needed), and load the Leanstral model."""
    config = build_inference_config(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        vision_seq_len=VISION_SEQ_LEN,
        enable_tkg_kernel=True,
    )
    ModelCls = get_model_cls()
    model = ModelCls(MODEL_PATH, config)

    # Compile if not already compiled
    model.compile(COMPILED_MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)

    # Enable vision encoder
    model.enable_vision_encoder()

    # Warmup: run one short generation to populate caches
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    adapter = HuggingFaceGenerationAdapter(model)
    warmup_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    _ = adapter.generate(
        input_ids=warmup_ids,
        attention_mask=torch.ones_like(warmup_ids),
        max_new_tokens=2,
        do_sample=False,
    )

    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(MODEL_PATH, "tokenizer.json"))


# ---- Helper functions ----


def extract_logits(outputs):
    """Extract logits from model output (handles various output formats)."""
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    elif isinstance(outputs, torch.Tensor):
        return outputs
    elif isinstance(outputs, (tuple, list)):
        return outputs[0]
    else:
        return outputs.logits


def greedy_decode(model, tokenizer, prompt, num_steps):
    """Run prefill + greedy decode for num_steps tokens.

    Returns (logits_list, token_list) where each logits entry is the
    last-position logits tensor for that decode step.
    """
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)
    prompt_len = input_ids.shape[1]
    all_logits = []
    all_tokens = []

    # Prefill
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(prompt_len, dtype=torch.int32).unsqueeze(0),
        seq_ids=torch.zeros(1, dtype=torch.int32),
        sampling_params=torch.zeros(1, 3, dtype=torch.float32),
    )
    logits = extract_logits(out)
    step_logits = (logits[:, -1, :] if logits.dim() == 3 else logits).float().cpu()
    all_logits.append(step_logits)
    next_token = step_logits.argmax(dim=-1).squeeze().item()
    all_tokens.append(next_token)

    # Decode
    for step in range(num_steps - 1):
        total_len = prompt_len + len(all_tokens)
        out = model(
            input_ids=torch.tensor([[all_tokens[-1]]], dtype=torch.long),
            attention_mask=torch.ones(1, total_len, dtype=torch.int32),
            position_ids=torch.tensor([[total_len - 1]], dtype=torch.int32),
            seq_ids=torch.zeros(1, dtype=torch.int32),
            sampling_params=torch.zeros(1, 3, dtype=torch.float32),
        )
        logits = extract_logits(out)
        step_logits = (
            (logits[:, -1, :] if logits.dim() == 3 else logits[:1]).float().cpu()
        )
        all_logits.append(step_logits)
        next_token = step_logits.argmax(dim=-1).squeeze().item()
        all_tokens.append(next_token)

    return all_logits, all_tokens


# ---- Tests ----


def test_smoke(compiled_model):
    """Smoke test: model loads and has expected attributes."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model, "cpu_projector")
    assert hasattr(compiled_model, "vision_encoder_model")
    assert compiled_model.config.text_config.num_hidden_layers == 40
    assert compiled_model.config.text_config.hidden_size == 5120
    print("Smoke test passed: model loaded with correct config")


def test_text_generation(compiled_model, tokenizer):
    """Test text-only generation produces coherent output."""
    logits_list, tokens = greedy_decode(
        compiled_model, tokenizer, TEXT_PROMPT, NUM_DECODE_STEPS
    )
    text = tokenizer.decode(tokens)
    print(f"Generated ({NUM_DECODE_STEPS} tokens): {text}")

    # Basic sanity checks
    assert len(tokens) == NUM_DECODE_STEPS, (
        f"Expected {NUM_DECODE_STEPS} tokens, got {len(tokens)}"
    )
    assert all(isinstance(t, int) for t in tokens), "All tokens must be integers"
    assert all(0 <= t < 131072 for t in tokens), "Token IDs must be in vocab range"

    # Logits shape check
    for i, logits in enumerate(logits_list):
        assert logits.shape[-1] == 131072, (
            f"Step {i}: logits vocab dim = {logits.shape[-1]}, expected 131072"
        )

    print(f"Text generation test passed: {NUM_DECODE_STEPS} valid tokens generated")


def test_output_coherence(compiled_model, tokenizer):
    """Test that generated text is not repetitive gibberish."""
    _, tokens = greedy_decode(compiled_model, tokenizer, TEXT_PROMPT, NUM_DECODE_STEPS)
    text = tokenizer.decode(tokens)

    # Check for excessive repetition
    words = text.split()
    if len(words) >= 5:
        max_repeat = 5
        for i in range(len(words) - max_repeat):
            repeated = all(words[i + j] == words[i] for j in range(max_repeat))
            assert not repeated, (
                f"Excessive word repetition detected at position {i}: "
                f"'{words[i]}' repeated {max_repeat}+ times"
            )

    # Check that we're not just producing the same token repeatedly
    unique_tokens = set(tokens)
    assert len(unique_tokens) >= min(3, NUM_DECODE_STEPS), (
        f"Only {len(unique_tokens)} unique tokens in {NUM_DECODE_STEPS} steps -- "
        f"possible degenerate generation"
    )

    print(f"Coherence test passed: {len(unique_tokens)} unique tokens, text: {text}")


def test_logit_validity(compiled_model, tokenizer):
    """Test that logits are finite and have reasonable distribution."""
    logits_list, _ = greedy_decode(
        compiled_model, tokenizer, TEXT_PROMPT, NUM_DECODE_STEPS
    )

    for step, logits in enumerate(logits_list):
        # Must be finite
        assert torch.isfinite(logits).all(), f"Step {step}: non-finite logits detected"

        # Must not be all zeros
        assert logits.abs().sum() > 0, f"Step {step}: all-zero logits"

        # Softmax should produce a valid probability distribution
        probs = torch.softmax(logits.squeeze(), dim=-1)
        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) < 0.01, (
            f"Step {step}: softmax sum = {prob_sum}, expected ~1.0"
        )

    print(f"Logit validity test passed: all {len(logits_list)} steps have valid logits")


def test_throughput(compiled_model, tokenizer):
    """Measure and report decode throughput."""
    num_tokens = 20

    # Warmup
    greedy_decode(compiled_model, tokenizer, "Hello", 3)

    # Timed run
    start = time.perf_counter()
    _, tokens = greedy_decode(compiled_model, tokenizer, TEXT_PROMPT, num_tokens)
    elapsed = time.perf_counter() - start

    throughput = num_tokens / elapsed
    text = tokenizer.decode(tokens)
    print(f"Throughput: {throughput:.1f} tok/s ({num_tokens} tokens in {elapsed:.2f}s)")
    print(f"Generated: {text}")

    # Minimum throughput sanity check (very conservative)
    assert throughput > 5.0, (
        f"Throughput {throughput:.1f} tok/s is below minimum threshold of 5 tok/s"
    )


# ---- Main ----

if __name__ == "__main__":
    print("=" * 70)
    print("Ministral-3-14B-Instruct-2512 (Leanstral) Integration Tests")
    print("=" * 70)
    print(f"Model path:    {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_MODEL_PATH}")
    print(f"TP degree:     {TP_DEGREE}")
    print()
    pytest.main([__file__, "-v", "--capture=tee-sys"])
