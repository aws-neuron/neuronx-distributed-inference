#!/usr/bin/env python3
"""
Integration tests for ZAYA1-base NeuronX Distributed Inference contrib model.

Tests compilation, loading, prefill accuracy, token generation, and performance.

Requirements:
  - AWS Neuron instance (trn2.3xlarge or larger)
  - SDK 2.27+ with neuronx-distributed-inference 0.7+
  - Zyphra's custom transformers fork:
    pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
  - KV cache patch for batch > 1 (see README.md)
  - NxD spmd.py patches (see README.md)

Environment:
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2  (required for NKI kernels on trn2)

Usage:
  # Run with pytest (model must be pre-compiled)
  ZAYA_HF_MODEL_DIR=/path/to/ZAYA1-base \\
  ZAYA_COMPILED_MODEL_DIR=/path/to/compiled \\
  pytest test_model.py -v --capture=tee-sys

  # Or run standalone (compiles if needed)
  python3 test_model.py --compile --tp-degree 2
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Monkey-patch torch.jit.script to prevent @jit_fuser crashes from Zyphra's
# modeling_zaya.py. Must be done before importing modeling_zaya.
_real_jit_script = torch.jit.script
torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)

# Add src/ to path
SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modeling_zaya import NeuronZayaForCausalLM, ZayaInferenceConfig, ZayaNeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# CRITICAL: Restore torch.jit.script before compilation/loading
torch.jit.script = _real_jit_script


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "ZAYA_HF_MODEL_DIR", os.path.expanduser("~/models/ZAYA1-base")
)
COMPILED_PATH = os.environ.get(
    "ZAYA_COMPILED_MODEL_DIR", os.path.expanduser("~/neuron_models/ZAYA1-base-tp2")
)

# Default compilation settings
TP_DEGREE = 2
BATCH_SIZE = 1
MAX_CONTEXT_LENGTH = 128
SEQ_LEN = 256
BUCKETS = [256]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_model(model_path, compiled_path, tp_degree=TP_DEGREE, batch_size=BATCH_SIZE):
    """Build and optionally compile the NxDI model."""
    neuron_config = ZayaNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=MAX_CONTEXT_LENGTH,
        seq_len=SEQ_LEN,
        on_device_generation=None,
        is_continuous_batching=True,
        buckets=BUCKETS,
    )
    config = ZayaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    model = NeuronZayaForCausalLM(model_path, config=config)
    return model


def prefill(model, input_ids):
    """Run prefill (context encoding) and return logits."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    seq_ids = torch.arange(batch_size, dtype=torch.int32)
    position_ids = (
        torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    model.reset()
    with torch.no_grad():
        output = model.forward(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
        )
    return output[0]  # logits


def generate_tokens(model, input_ids, max_new_tokens=20):
    """Autoregressive generation using NxDI forward API."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Prefill
    logits = prefill(model, input_ids)
    if logits.dim() == 3:
        next_logits = logits[:, -1, :]
    else:
        next_logits = logits
    next_token_id = torch.argmax(next_logits.float(), dim=-1)  # [batch]
    generated = [
        input_ids[b].tolist() + [next_token_id[b].item()] for b in range(batch_size)
    ]
    current_pos = seq_len

    # Token generation loop
    for _ in range(max_new_tokens - 1):
        tkg_input = next_token_id.unsqueeze(1)  # [batch, 1]
        tkg_pos = torch.full((batch_size, 1), current_pos, dtype=torch.long)
        tkg_seq = torch.arange(batch_size, dtype=torch.int32)

        with torch.no_grad():
            tkg_out = model.forward(
                input_ids=tkg_input,
                seq_ids=tkg_seq,
                position_ids=tkg_pos,
            )

        tkg_logits = tkg_out[0]
        if tkg_logits.dim() == 3:
            next_logits = tkg_logits[:, -1, :]
        else:
            next_logits = tkg_logits
        next_token_id = torch.argmax(next_logits.float(), dim=-1)

        for b in range(batch_size):
            generated[b].append(next_token_id[b].item())
        current_pos += 1

    return generated


def is_repetitive(text, window=5):
    """Check for degenerate repetition."""
    words = text.split()
    if len(words) < window:
        return False
    for i in range(len(words) - window + 1):
        if len(set(words[i : i + window])) == 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    """Load compiled ZAYA1-base model."""
    compiled = Path(COMPILED_PATH)
    if not compiled.exists():
        pytest.skip(
            f"Compiled model not found at {COMPILED_PATH}. "
            f"Set ZAYA_COMPILED_MODEL_DIR or run: python3 test_model.py --compile"
        )
    m = build_model(MODEL_PATH, COMPILED_PATH)
    m.load(COMPILED_PATH)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoads:
    """Smoke tests for model loading."""

    def test_model_loads(self, model):
        assert model is not None
        assert hasattr(model, "config")
        assert hasattr(model.config, "neuron_config")

    def test_config_fields(self, model):
        config = model.config
        assert config.hidden_size == 2048
        assert config.num_attention_heads == 16
        assert config.vocab_size == 262272
        assert hasattr(config, "zaya_layers")
        assert len(config.zaya_layers) == 80
        assert config.zaya_layers[0] == "a"  # first layer is attention
        assert config.zaya_layers[1] == 16  # second layer is MoE with 16 experts

    def test_tied_weights(self, model):
        assert model.config.tie_word_embeddings is True


class TestPrefill:
    """Tests for context encoding (prefill) accuracy."""

    def test_paris(self, model, tokenizer):
        """The capital of France is -> Paris"""
        prompt = "The capital of France is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        logits = prefill(model, input_ids)
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[0, :]
        top_token = torch.argmax(last_logits.float()).item()
        decoded = tokenizer.decode([top_token]).strip().lower()
        assert decoded == "paris", f"Expected 'paris', got '{decoded}'"

    def test_logit_distribution(self, model, tokenizer):
        """Logits should have non-trivial variance (not corrupted weights)."""
        input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
        logits = prefill(model, input_ids)
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[0, :]
        std = last_logits.float().std().item()
        assert std > 0.1, f"Logit std too low ({std:.4f}), possible weight corruption"


class TestGeneration:
    """Tests for token generation quality."""

    def test_generates_text(self, model, tokenizer):
        """Model generates coherent text beyond the prompt."""
        input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
        generated = generate_tokens(model, input_ids, max_new_tokens=20)
        text = tokenizer.decode(generated[0])
        assert len(text) > len("The capital of France is")
        assert "Paris" in text or "paris" in text.lower()

    def test_not_repetitive(self, model, tokenizer):
        """Output should not be degenerately repetitive."""
        input_ids = tokenizer(
            "Albert Einstein was born in", return_tensors="pt"
        ).input_ids
        generated = generate_tokens(model, input_ids, max_new_tokens=30)
        text = tokenizer.decode(generated[0])
        assert not is_repetitive(text), f"Output is degenerate: {text[:100]}"


class TestPerformance:
    """Performance benchmarks. These report metrics but use generous thresholds."""

    def test_ttft(self, model, tokenizer):
        """Time to first token should be under 200ms."""
        input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids
        seq_ids = torch.arange(1, dtype=torch.int32)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

        # Warmup
        for _ in range(3):
            model.reset()
            with torch.no_grad():
                model.forward(
                    input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids
                )

        # Measure
        times = []
        for _ in range(10):
            model.reset()
            t0 = time.time()
            with torch.no_grad():
                model.forward(
                    input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids
                )
            times.append((time.time() - t0) * 1000)

        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        print(f"TTFT: avg={avg:.1f}ms, P50={p50:.1f}ms")
        assert avg < 200, f"TTFT {avg:.1f}ms exceeds 200ms threshold"

    def test_throughput(self, model, tokenizer):
        """TKG throughput should exceed 10 tok/s."""
        input_ids = tokenizer("Hello", return_tensors="pt").input_ids

        # Prefill
        logits = prefill(model, input_ids)
        if logits.dim() == 3:
            nl = logits[0, -1, :]
        else:
            nl = logits[0, :]
        next_id = torch.argmax(nl.float()).item()
        pos = input_ids.shape[1]

        # Warmup
        for _ in range(5):
            tkg_in = torch.tensor([[next_id]], dtype=torch.long)
            tkg_pos = torch.tensor([[pos]], dtype=torch.long)
            tkg_seq = torch.arange(1, dtype=torch.int32)
            with torch.no_grad():
                model.forward(input_ids=tkg_in, seq_ids=tkg_seq, position_ids=tkg_pos)

        # Benchmark
        latencies = []
        for _ in range(50):
            t0 = time.time()
            with torch.no_grad():
                model.forward(input_ids=tkg_in, seq_ids=tkg_seq, position_ids=tkg_pos)
            latencies.append((time.time() - t0) * 1000)

        avg_ms = sum(latencies) / len(latencies)
        throughput = 1000.0 / avg_ms
        print(f"TKG: avg={avg_ms:.1f}ms/token, throughput={throughput:.1f} tok/s")
        assert throughput > 10, (
            f"Throughput {throughput:.1f} tok/s below 10 tok/s threshold"
        )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------


def main():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Test ZAYA1-base NxDI model")
    parser.add_argument("--compile", action="store_true", help="Compile before testing")
    parser.add_argument("--tp-degree", type=int, default=TP_DEGREE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--compiled-path", default=COMPILED_PATH)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("=" * 60)
    print("ZAYA1-base NxDI Integration Test")
    print("=" * 60)

    model = build_model(
        args.model_path,
        args.compiled_path,
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
    )

    if args.compile:
        compiled = Path(args.compiled_path)
        if compiled.exists():
            shutil.rmtree(compiled)
        compiled.mkdir(parents=True, exist_ok=True)

        print(f"\nCompiling (TP={args.tp_degree}, batch={args.batch_size})...")
        t0 = time.time()
        model.compile(compiled_model_path=args.compiled_path)
        print(f"Compilation done in {time.time() - t0:.1f}s")

    print(f"\nLoading from {args.compiled_path}...")
    t0 = time.time()
    model.load(args.compiled_path)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # --- Tests ---
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            failed += 1

    print("\n--- Smoke Tests ---")
    check("model loaded", model is not None)
    check("has config", hasattr(model, "config"))
    check("hidden_size=2048", model.config.hidden_size == 2048)
    check("80 layers", len(model.config.zaya_layers) == 80)
    check("tied embeddings", model.config.tie_word_embeddings is True)

    print("\n--- Prefill Test ---")
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    logits = prefill(model, input_ids)
    if logits.dim() == 3:
        last = logits[0, -1, :]
    else:
        last = logits[0, :]
    top = tokenizer.decode([torch.argmax(last.float()).item()]).strip().lower()
    check('"Paris" prediction', top == "paris", f"got '{top}'")
    check("logit std > 0.1", last.float().std().item() > 0.1)

    print("\n--- Generation Test ---")
    generated = generate_tokens(model, input_ids, max_new_tokens=20)
    text = tokenizer.decode(generated[0])
    print(f"  Generated: {text[:120]}")
    check("Paris in output", "paris" in text.lower())
    check("not repetitive", not is_repetitive(text))

    print("\n--- Performance ---")
    # TTFT
    seq_ids = torch.arange(1, dtype=torch.int32)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    for _ in range(3):
        model.reset()
        with torch.no_grad():
            model.forward(
                input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids
            )
    ttft_times = []
    for _ in range(10):
        model.reset()
        t0 = time.time()
        with torch.no_grad():
            model.forward(
                input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids
            )
        ttft_times.append((time.time() - t0) * 1000)
    avg_ttft = sum(ttft_times) / len(ttft_times)
    print(f"  TTFT: avg={avg_ttft:.1f}ms")
    check("TTFT < 200ms", avg_ttft < 200, f"got {avg_ttft:.1f}ms")

    # TKG throughput
    nl = logits[0, -1, :] if logits.dim() == 3 else logits[0, :]
    next_id = torch.argmax(nl.float()).item()
    tkg_in = torch.tensor([[next_id]], dtype=torch.long)
    tkg_pos = torch.tensor([[input_ids.shape[1]]], dtype=torch.long)
    tkg_seq = torch.arange(1, dtype=torch.int32)
    for _ in range(5):
        with torch.no_grad():
            model.forward(input_ids=tkg_in, seq_ids=tkg_seq, position_ids=tkg_pos)
    lats = []
    for _ in range(50):
        t0 = time.time()
        with torch.no_grad():
            model.forward(input_ids=tkg_in, seq_ids=tkg_seq, position_ids=tkg_pos)
        lats.append((time.time() - t0) * 1000)
    avg_lat = sum(lats) / len(lats)
    tput = 1000.0 / avg_lat
    print(f"  TKG: avg={avg_lat:.1f}ms/token, throughput={tput:.1f} tok/s")
    check("throughput > 10 tok/s", tput > 10, f"got {tput:.1f}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
