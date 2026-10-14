#!/usr/bin/env python3
"""
Integration tests for Kimi-K2.5/K2.6 multimodal NeuronX implementation.

Tests compilation, loading, and multimodal inference on trn2.48xlarge.
Supports both K2.5 and K2.6 checkpoints — override paths via environment variables.

Requirements:
    - trn2.48xlarge with NEURON_LOGICAL_NC_CONFIG=2 (64 logical cores)
    - LOCAL_WORLD_SIZE=64
    - Model weights at KIMI_MODEL_PATH (default: /mnt/nvme/models/Kimi-K2.5)
    - Pre-computed MoonViT embeddings at KIMI_VISION_EMB_PATH
    - Neuron SDK 2.29 (Deep Learning AMI Neuron Ubuntu 24.04 20260410)
    - tiktoken package installed (pip install tiktoken)

Environment variables:
    KIMI_MODEL_PATH       - Path to K2.5 or K2.6 checkpoint (default: /mnt/nvme/models/Kimi-K2.5)
    KIMI_TEXT_MODEL_DIR   - Path for text-only model dir (default: /home/ubuntu/models/Kimi-K2.5-text)
    KIMI_COMPILED_PATH    - Path for compiled NEFFs (default: derived from TEXT_MODEL_DIR)
    KIMI_VISION_EMB_PATH  - Path to pre-computed vision embeddings (default: /mnt/nvme/models/moonvit_448_real_embeddings.pt)

Usage:
    # Full test with K2.5 (default):
    NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
        pytest test_model.py -v --capture=tee-sys

    # Full test with K2.6:
    NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
        KIMI_MODEL_PATH=/mnt/nvme/models/Kimi-K2.6 \
        KIMI_TEXT_MODEL_DIR=/home/ubuntu/models/Kimi-K2.6-text \
        KIMI_COMPILED_PATH=/mnt/nvme/models/Kimi-K2.6-text/neuron-compiled \
        pytest test_model.py -v --capture=tee-sys

    # Load-only (skip compile, use existing NEFFs):
    NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
        pytest test_model.py -v --capture=tee-sys -k "not compile"

    # Standalone (use PYTHONUNBUFFERED for real-time log output):
    NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
        PYTHONUNBUFFERED=1 python test_model.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_kimi_k2 import (
    NeuronKimiK2ForCausalLM,
    NeuronKimiK2Model,
    KimiK2InferenceConfig,
)
from modeling_kimi_k25 import (
    apply_k25_patches,
    apply_k25_checkpoint_patch,
    build_k25_config,
    create_text_only_model_dir,
    BOS_TOKEN_ID,
    IM_USER_TOKEN_ID,
    IM_END_TOKEN_ID,
    IM_ASSISTANT_TOKEN_ID,
    MEDIA_PLACEHOLDER_TOKEN_ID,
)


# ---------------------------------------------------------------------------
# Configuration — override via environment variables for K2.6 or custom paths
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("KIMI_MODEL_PATH", "/mnt/nvme/models/Kimi-K2.5")
TEXT_MODEL_DIR = os.environ.get(
    "KIMI_TEXT_MODEL_DIR", "/home/ubuntu/models/Kimi-K2.5-text"
)
COMPILED_MODEL_PATH = os.environ.get(
    "KIMI_COMPILED_PATH",
    "/mnt/nvme/models/Kimi-K2.5-text/neuron-compiled-k25-vl-s512-v5",
)
VISION_EMBEDDINGS_PATH = os.environ.get(
    "KIMI_VISION_EMB_PATH",
    "/mnt/nvme/models/moonvit_448_real_embeddings.pt",
)

# Model configuration (TP=64, EP=1, LNC=2)
TP_DEGREE = 64
EP_DEGREE = 1
LNC = 2
BATCH_SIZE = 1
SEQ_LEN = 512
N_ACTIVE_TOKENS = 128
N_VISION_TOKENS = 256  # 448x448 image → 32x32 patches → 16x16 merged


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load K2.5 tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def vision_embeddings():
    """Load pre-computed MoonViT embeddings."""
    if not os.path.exists(VISION_EMBEDDINGS_PATH):
        pytest.skip(f"Vision embeddings not found at {VISION_EMBEDDINGS_PATH}")
    emb = torch.load(VISION_EMBEDDINGS_PATH, map_location="cpu")
    assert emb.shape == (N_VISION_TOKENS, 7168), f"Unexpected shape: {emb.shape}"
    return emb.to(torch.bfloat16)


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (if needed) and load K2.5 multimodal model."""
    # 1. Create text-only model dir
    text_model_dir = create_text_only_model_dir(MODEL_PATH, TEXT_MODEL_DIR)

    # 2. Apply vision patches BEFORE model init
    apply_k25_patches(NeuronKimiK2ForCausalLM, NeuronKimiK2Model, ep_degree=EP_DEGREE)

    # 3. Build config
    config = build_k25_config(
        text_model_dir,
        tp_degree=TP_DEGREE,
        ep_degree=EP_DEGREE,
        lnc=LNC,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        n_active_tokens=N_ACTIVE_TOKENS,
        quantized=True,
    )

    # 4. Initialize model
    model = NeuronKimiK2ForCausalLM(text_model_dir, config=config)

    # 5. Apply checkpoint patches
    apply_k25_checkpoint_patch(model)

    # 6. Compile if needed
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        t0 = time.time()
        model.compile(COMPILED_MODEL_PATH)
        print(f"Compilation done in {(time.time() - t0) / 60:.1f} min")

    # 7. Load
    print(f"\nLoading model from {COMPILED_MODEL_PATH}...")
    t0 = time.time()
    model.load(COMPILED_MODEL_PATH)
    print(f"Model loaded in {(time.time() - t0) / 60:.1f} min")

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_multimodal_prompt(
    tokenizer, vision_embeddings, seq_len, user_text="Describe this image in detail."
):
    """Build multimodal prompt with vision placeholder tokens.

    Returns:
        input_ids_list: List of token IDs
        ve: [1, seq_len, 7168] vision embedding tensor
        vm: [1, seq_len, 1] vision mask tensor
        n_prompt: Number of prompt tokens
    """
    n_vision = vision_embeddings.shape[0]
    hidden_size = vision_embeddings.shape[1]

    text_ids = tokenizer.encode(user_text)
    placeholder_ids = [MEDIA_PLACEHOLDER_TOKEN_ID] * n_vision

    input_ids_list = (
        [BOS_TOKEN_ID, IM_USER_TOKEN_ID]
        + placeholder_ids
        + text_ids
        + [IM_END_TOKEN_ID, IM_ASSISTANT_TOKEN_ID]
    )
    n_prompt = len(input_ids_list)

    vision_start = 2  # After BOS + im_user

    ve = torch.zeros(1, seq_len, hidden_size, dtype=torch.bfloat16)
    vm = torch.full((1, seq_len, 1), fill_value=seq_len - 1, dtype=torch.int32)

    for i in range(n_vision):
        ve[0, i] = vision_embeddings[i]
        vm[0, i, 0] = vision_start + i

    return input_ids_list, ve, vm, n_prompt


def generate_multimodal(
    model, tokenizer, vision_embeddings, max_new_tokens=32, min_tokens_before_eos=3
):
    """Generate tokens from a multimodal prompt.

    Returns:
        output_text: Generated text
        generated_ids: List of generated token IDs
        cte_time: CTE latency in seconds
        tkg_time: TKG latency in seconds
    """
    input_ids_list, ve, vm, n_prompt = build_multimodal_prompt(
        tokenizer, vision_embeddings, SEQ_LEN
    )

    model.reset()
    seq_ids = torch.arange(BATCH_SIZE, dtype=torch.long)

    # Context encoding
    cte_input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    cte_mask = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    cte_mask[0, :n_prompt] = 1
    cte_pos = torch.arange(n_prompt, dtype=torch.long).unsqueeze(0)

    t0 = time.perf_counter()
    output = model(
        input_ids=cte_input_ids,
        attention_mask=cte_mask,
        position_ids=cte_pos,
        seq_ids=seq_ids,
        vision_embeddings=ve,
        vision_mask=vm,
    )
    cte_time = time.perf_counter() - t0

    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.dim() == 3:
        first_token = logits[0, -1].argmax(dim=-1).item()
    else:
        first_token = logits[0].argmax(dim=-1).item()

    generated_ids = [first_token]
    current_pos = n_prompt

    # Token generation
    t_gen_start = time.perf_counter()
    for step in range(max_new_tokens - 1):
        next_input_ids = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
        next_position_ids = torch.tensor([[current_pos]], dtype=torch.long)
        next_attention_mask = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        next_attention_mask[0, : current_pos + 1] = 1

        output = model(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            position_ids=next_position_ids,
            seq_ids=seq_ids,
            vision_embeddings=torch.zeros(0, dtype=torch.bfloat16),
            vision_mask=torch.zeros(0, dtype=torch.bool),
        )

        logits_tkg = output.logits if hasattr(output, "logits") else output[0]
        if logits_tkg.dim() == 3:
            next_token = logits_tkg[0, -1].argmax(dim=-1).item()
        elif logits_tkg.dim() == 2:
            next_token = logits_tkg[0].argmax(dim=-1).item()
        else:
            next_token = logits_tkg.argmax(dim=-1).item()

        generated_ids.append(next_token)
        current_pos += 1

        if next_token in (IM_END_TOKEN_ID, tokenizer.eos_token_id):
            break
        if current_pos >= SEQ_LEN:
            break

    tkg_time = time.perf_counter() - t_gen_start

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, generated_ids, cte_time, tkg_time


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: Model loaded successfully")


def test_multimodal_generates(compiled_model, tokenizer, vision_embeddings):
    """Test multimodal generation produces coherent output."""
    output, gen_ids, cte_time, tkg_time = generate_multimodal(
        compiled_model, tokenizer, vision_embeddings, max_new_tokens=32
    )
    assert len(output) > 0, "Output should not be empty"
    words = output.split()
    assert len(words) >= 3, f"Output too short: {output}"
    print(f"PASS: Multimodal generation - Output: {output[:200]}")
    print(f"  CTE: {cte_time:.3f}s, TKG: {tkg_time:.3f}s")


def test_vision_affects_output(compiled_model, tokenizer, vision_embeddings):
    """Test that vision embeddings actually affect model output.

    Compare output with real vision vs zero vision — they should differ.
    """
    seq_ids = torch.arange(BATCH_SIZE, dtype=torch.long)
    hidden_size = 7168
    sl = SEQ_LEN

    n_test = 128
    test_ids_list = [BOS_TOKEN_ID, IM_USER_TOKEN_ID] + [MEDIA_PLACEHOLDER_TOKEN_ID] * (
        n_test - 2
    )
    test_input_ids = torch.tensor([test_ids_list], dtype=torch.long)
    test_mask = torch.zeros(1, sl, dtype=torch.long)
    test_mask[0, :n_test] = 1
    test_pos = torch.arange(n_test, dtype=torch.long).unsqueeze(0)

    # Test A: Real vision
    ve_real = torch.zeros(1, sl, hidden_size, dtype=torch.bfloat16)
    vm_real = torch.full((1, sl, 1), fill_value=sl - 1, dtype=torch.int32)
    n_vision = min(vision_embeddings.shape[0], n_test - 2)
    for i in range(n_vision):
        ve_real[0, i] = vision_embeddings[i]
        vm_real[0, i, 0] = i + 2

    compiled_model.reset()
    out_a = compiled_model(
        input_ids=test_input_ids,
        attention_mask=test_mask,
        position_ids=test_pos,
        seq_ids=seq_ids,
        vision_embeddings=ve_real,
        vision_mask=vm_real,
    )
    logits_a = out_a.logits if hasattr(out_a, "logits") else out_a[0]

    # Test B: Zero vision
    ve_zero = torch.zeros(1, sl, hidden_size, dtype=torch.bfloat16)
    vm_zero = torch.full((1, sl, 1), fill_value=sl - 1, dtype=torch.int32)

    compiled_model.reset()
    out_b = compiled_model(
        input_ids=test_input_ids,
        attention_mask=test_mask,
        position_ids=test_pos,
        seq_ids=seq_ids,
        vision_embeddings=ve_zero,
        vision_mask=vm_zero,
    )
    logits_b = out_b.logits if hasattr(out_b, "logits") else out_b[0]

    # Compare
    if logits_a.dim() == 3:
        token_a = logits_a[0, -1].argmax(dim=-1).item()
        token_b = logits_b[0, -1].argmax(dim=-1).item()
    else:
        token_a = logits_a[0].argmax(dim=-1).item()
        token_b = logits_b[0].argmax(dim=-1).item()

    max_diff = (logits_a.float() - logits_b.float()).abs().max().item()

    assert token_a != token_b, (
        f"Vision did not affect output: token_a={token_a}, token_b={token_b}, "
        f"max_logit_diff={max_diff:.6f}"
    )
    print(
        f"PASS: Vision affects output (token_a={token_a}, token_b={token_b}, "
        f"max_logit_diff={max_diff:.3f})"
    )


def test_output_coherence(compiled_model, tokenizer, vision_embeddings):
    """Test that output is not gibberish or repetitive."""
    output, gen_ids, _, _ = generate_multimodal(
        compiled_model, tokenizer, vision_embeddings, max_new_tokens=64
    )
    words = output.split()
    assert len(words) >= 3, f"Output too short: {output}"

    # Check for excessive repetition
    if len(words) >= 10:
        for i in range(len(words) - 5):
            repeated = all(words[i + j] == words[i] for j in range(5))
            assert not repeated, f"Excessive repetition in output: {output}"

    print(f"PASS: Coherence test - Output: {output[:200]}")


def test_performance_tpot(compiled_model, tokenizer, vision_embeddings):
    """Measure per-token output latency (TPOT)."""
    # Warmup
    generate_multimodal(compiled_model, tokenizer, vision_embeddings, max_new_tokens=10)

    # Measure
    n_runs = 3
    tpots = []
    for _ in range(n_runs):
        _, gen_ids, cte_time, tkg_time = generate_multimodal(
            compiled_model, tokenizer, vision_embeddings, max_new_tokens=32
        )
        n_tkg_tokens = len(gen_ids) - 1  # First token comes from CTE
        if n_tkg_tokens > 0:
            tpot = (tkg_time * 1000) / n_tkg_tokens
            tpots.append(tpot)

    if tpots:
        median_tpot = sorted(tpots)[len(tpots) // 2]
        tok_per_sec = 1000.0 / median_tpot
        print(f"PASS: TPOT = {median_tpot:.1f} ms ({tok_per_sec:.1f} tok/s)")
        # K2.5 at TP=64 EP=1 LNC=2: expected ~21 ms/token (45.9 tok/s)
        assert median_tpot < 100, f"TPOT {median_tpot:.1f}ms exceeds 100ms threshold"
    else:
        pytest.skip("Could not measure TPOT")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Force line-buffered stdout for real-time log output when redirected to file
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Configure logging to show checkpoint loader progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    print("=" * 80)
    print("Kimi-K2.5/K2.6 Multimodal Integration Tests")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Text model dir: {TEXT_MODEL_DIR}")
    print(f"  Compiled path: {COMPILED_MODEL_PATH}")
    print(f"  Vision embeddings: {VISION_EMBEDDINGS_PATH}")
    print("=" * 80)

    # Load vision embeddings
    if not os.path.exists(VISION_EMBEDDINGS_PATH):
        print(f"ERROR: Vision embeddings not found at {VISION_EMBEDDINGS_PATH}")
        sys.exit(1)
    vision_emb = torch.load(VISION_EMBEDDINGS_PATH, map_location="cpu").to(
        torch.bfloat16
    )
    print(f"Vision embeddings: {vision_emb.shape}")

    # Setup model
    text_model_dir = create_text_only_model_dir(MODEL_PATH, TEXT_MODEL_DIR)
    apply_k25_patches(NeuronKimiK2ForCausalLM, NeuronKimiK2Model, ep_degree=EP_DEGREE)

    config = build_k25_config(
        text_model_dir,
        tp_degree=TP_DEGREE,
        ep_degree=EP_DEGREE,
        lnc=LNC,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        n_active_tokens=N_ACTIVE_TOKENS,
        quantized=True,
    )

    model = NeuronKimiK2ForCausalLM(text_model_dir, config=config)
    apply_k25_checkpoint_patch(model)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        t0 = time.time()
        model.compile(COMPILED_MODEL_PATH)
        print(f"Compilation done in {(time.time() - t0) / 60:.1f} min")

    print(f"\nLoading model from {COMPILED_MODEL_PATH}...")
    t0 = time.time()
    model.load(COMPILED_MODEL_PATH)
    print(f"Model loaded in {(time.time() - t0) / 60:.1f} min")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Multimodal Generation Test...")
    test_multimodal_generates(model, tok, vision_emb)

    print("\n3. Vision A/B Test...")
    test_vision_affects_output(model, tok, vision_emb)

    print("\n4. Coherence Test...")
    test_output_coherence(model, tok, vision_emb)

    print("\n5. TPOT Performance Test...")
    test_performance_tpot(model, tok, vision_emb)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
