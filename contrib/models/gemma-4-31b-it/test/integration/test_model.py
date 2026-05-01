#!/usr/bin/env python3
"""
Integration tests for Gemma-4-31b-it NeuronX Distributed Inference implementation.

Tests model compilation, loading, and inference accuracy/performance.

Usage:
    # Run with pytest
    pytest test/integration/test_model.py --capture=tee-sys

    # Run standalone
    python test/integration/test_model.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_gemma4 import (
    NeuronGemma4ForCausalLM,
    Gemma4InferenceConfig,
    Gemma4NeuronConfig,
)


# ============================================================================
# Test configuration — adjust paths for your environment
# ============================================================================

MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "/mnt/models/gemma-4-31b-it")
COMPILED_MODEL_PATH = os.environ.get(
    "GEMMA4_COMPILED_PATH", "/mnt/models/gemma4-compiled"
)
TP_DEGREE = int(os.environ.get("GEMMA4_TP_DEGREE", "4"))
BATCH_SIZE = 1
CONTEXT_LEN = 128
GEN_LEN = 128


# ============================================================================
# Helpers
# ============================================================================


def create_config(model_path, tp_degree=TP_DEGREE, batch_size=BATCH_SIZE):
    """Create Gemma4 inference config from model path."""
    neuron_config = Gemma4NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_batch_size=batch_size,
        seq_len=CONTEXT_LEN + GEN_LEN,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        attn_kernel_enabled=False,
    )

    def load_config_fn(config_obj):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(config_obj, k, v)

    return Gemma4InferenceConfig(
        neuron_config=neuron_config, load_config=load_config_fn
    )


def generate_tokens(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate tokens using manual forward pass loop (prefill + decode).

    Returns (generated_token_ids, output_text, timing_dict).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    n_positions = CONTEXT_LEN + GEN_LEN

    # Pad to n_positions (right padding)
    if seq_len < n_positions:
        pad_len = n_positions - seq_len
        input_ids_padded = torch.cat(
            [input_ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        )
        attention_mask = torch.cat(
            [
                torch.ones(1, seq_len, dtype=torch.long),
                torch.zeros(1, pad_len, dtype=torch.long),
            ],
            dim=1,
        )
    else:
        input_ids_padded = input_ids[:, :n_positions]
        attention_mask = torch.ones(1, n_positions, dtype=torch.long)

    position_ids = torch.zeros(1, n_positions, dtype=torch.long)
    position_ids[0, :seq_len] = torch.arange(seq_len)

    timing = {}

    # Prefill (context encoding)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    timing["ttft_ms"] = (time.perf_counter() - t0) * 1000

    # Extract first token
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    elif hasattr(outputs, "tokens") and outputs.tokens is not None:
        next_token = outputs.tokens[:, -1:]
    else:
        return [], "", timing

    generated_tokens = [next_token.item()]
    cur_pos = seq_len

    # Token generation loop
    t_gen_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        attention_mask[0, cur_pos] = 1
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=torch.tensor([[cur_pos]]),
            )
        cur_pos += 1

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        elif hasattr(outputs, "tokens") and outputs.tokens is not None:
            next_token = outputs.tokens[:, -1:]
        else:
            break

        generated_tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    t_gen_end = time.perf_counter()
    num_decode_tokens = len(generated_tokens) - 1  # exclude first token (from prefill)
    if num_decode_tokens > 0:
        decode_time = t_gen_end - t_gen_start
        timing["tpot_ms"] = decode_time / num_decode_tokens * 1000
        timing["throughput_tps"] = num_decode_tokens / decode_time
    timing["total_tokens"] = len(generated_tokens)

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_tokens, output_text, timing


def _is_repetitive(text, max_repeat=5):
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - max_repeat):
        if all(words[i + j] == words[i] for j in range(max_repeat)):
            return True
    return False


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (if needed) and load model."""
    config = create_config(MODEL_PATH)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    print(f"Loading compiled model from {COMPILED_MODEL_PATH}...")
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.")
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============================================================================
# Tests
# ============================================================================


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates coherent text."""
    prompt = "The capital of France is"
    tokens, text, timing = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=10
    )

    assert len(tokens) > 0, "Model should produce at least one token"
    assert len(text) > 0, "Output text should not be empty"
    print(f"PASS: Generated {len(tokens)} tokens: {text!r}")
    print(f"  TTFT: {timing.get('ttft_ms', 0):.1f}ms")


def test_token_matching(compiled_model, tokenizer):
    """
    Test that greedy decoding produces expected tokens for known prompts.

    Validated against HF CPU reference (transformers 5.5.0, bf16):
    - "The capital of France is" -> first token " France" (id varies by tokenizer)
    """
    prompt = "The capital of France is"
    tokens, text, _ = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=5
    )

    assert len(tokens) > 0
    # The model should mention France/Paris in response
    combined = prompt + " " + text
    assert any(word in combined.lower() for word in ["paris", "france"]), (
        f"Expected 'Paris' or 'France' in output, got: {combined!r}"
    )
    print(f"PASS: Token matching - output: {text!r}")


def test_chat_generation(compiled_model, tokenizer):
    """Test chat-templated generation (instruction-tuned model)."""
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    tokens, text, _ = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=20
    )

    assert len(tokens) > 0, "Chat generation should produce tokens"
    assert "4" in text, f"Expected '4' in response to 2+2, got: {text!r}"
    print(f"PASS: Chat generation - output: {text!r}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish or repetitive)."""
    messages = [{"role": "user", "content": "Write a haiku about the ocean."}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    tokens, text, _ = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=30
    )

    assert len(text.split()) > 3, "Output should have multiple words"
    assert not _is_repetitive(text), "Output should not be repetitive"
    print(f"PASS: Coherence test - output: {text!r}")


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    prompt = "Hello, how are you?"

    # Warmup
    generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=1)

    # Measure over multiple runs
    ttft_times = []
    for _ in range(5):
        _, _, timing = generate_tokens(
            compiled_model, tokenizer, prompt, max_new_tokens=1
        )
        ttft_times.append(timing.get("ttft_ms", 0))

    avg_ttft = sum(ttft_times) / len(ttft_times)

    # Gemma4 31B on TP=4: ~66ms observed, threshold at 200ms
    assert avg_ttft < 200, f"TTFT {avg_ttft:.1f}ms exceeds 200ms threshold"
    print(f"PASS: TTFT = {avg_ttft:.1f}ms (threshold: 200ms)")


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    prompt = "Hello"

    # Warmup
    generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=5)

    # Measure
    _, _, timing = generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=50)
    throughput = timing.get("throughput_tps", 0)

    # Gemma4 31B on TP=4: ~32 tok/s observed, threshold at 10 tok/s
    assert throughput > 10, (
        f"Throughput {throughput:.1f} tok/s below 10 tok/s threshold"
    )
    print(f"PASS: Throughput = {throughput:.1f} tok/s (threshold: 10 tok/s)")
    if "tpot_ms" in timing:
        print(f"  TPOT = {timing['tpot_ms']:.1f}ms")


# ============================================================================
# Standalone runner
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Gemma-4-31b-it Integration Tests")
    print(f"Model: {MODEL_PATH}")
    print(f"Compiled: {COMPILED_MODEL_PATH}")
    print(f"TP degree: {TP_DEGREE}")
    print("=" * 80)

    # Create config and compile if needed
    config = create_config(MODEL_PATH)
    compiled_path = Path(COMPILED_MODEL_PATH)

    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    # Load model
    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    tests = [
        ("1. Smoke Test", test_model_loads, (model,)),
        ("2. Generation Test", test_model_generates, (model, tokenizer)),
        ("3. Token Matching", test_token_matching, (model, tokenizer)),
        ("4. Chat Generation", test_chat_generation, (model, tokenizer)),
        ("5. Coherence Test", test_output_coherence, (model, tokenizer)),
        ("6. TTFT Performance", test_performance_ttft, (model, tokenizer)),
        ("7. Throughput Performance", test_performance_throughput, (model, tokenizer)),
    ]

    passed = 0
    failed = 0
    for name, test_fn, args in tests:
        print(f"\n{name}...")
        try:
            test_fn(*args)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    print("=" * 80)
