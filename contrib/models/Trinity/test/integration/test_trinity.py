#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integration tests for Trinity (AfmoeForCausalLM) NeuronX implementation.

Supports all three Trinity model sizes (Nano, Mini, Large) via environment variables.

Usage:
    # Set paths for your model size
    export TRINITY_MODEL_PATH="/path/to/model"
    export TRINITY_COMPILED_PATH="/path/to/compiled"

    # Run tests
    pytest test/integration/test_trinity.py --capture=tee-sys

Prerequisites:
    - Pre-compiled model at TRINITY_COMPILED_PATH
    - HuggingFace model weights at TRINITY_MODEL_PATH (downloaded with trust_remote_code=True)
    - Appropriate instance for model size (see README.md)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_trinity import NeuronTrinityForCausalLM, TrinityInferenceConfig

logger = logging.getLogger(__name__)

# Configuration via environment variables
MODEL_PATH = os.environ.get("TRINITY_MODEL_PATH")
COMPILED_MODEL_PATH = os.environ.get("TRINITY_COMPILED_PATH")

# Performance thresholds (conservative upper bounds -- fail if exceeded)
# These are generous limits to catch regressions, not tight benchmarks.
# TTFT: max acceptable latency in ms per forward pass
MAX_TTFT_MS = float(os.environ.get("TRINITY_MAX_TTFT_MS", "5000"))
# Throughput: min acceptable tokens/second (naive loop, not CTE+TKG pipeline)
MIN_THROUGHPUT_TOK_S = float(os.environ.get("TRINITY_MIN_THROUGHPUT_TOK_S", "0.5"))

_MISSING_ENV = []
if not MODEL_PATH:
    _MISSING_ENV.append("TRINITY_MODEL_PATH")
if not COMPILED_MODEL_PATH:
    _MISSING_ENV.append("TRINITY_COMPILED_PATH")

if _MISSING_ENV:
    pytestmark = pytest.mark.skip(
        reason=f"Required environment variables not set: {', '.join(_MISSING_ENV)}"
    )


def load_neuron_config_from_compiled(compiled_path: str):
    """Load neuron configuration from compiled model's neuron_config.json."""
    config_path = Path(compiled_path) / "neuron_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)

    if "neuron_config" in config_data:
        return config_data["neuron_config"]
    else:
        return config_data


def create_model_for_inference(compiled_path: str, model_path: str):
    """Create model for inference using compiled neuron_config."""
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)

    dtype_str = neuron_config_dict.get("torch_dtype", "torch.bfloat16")
    if isinstance(dtype_str, str):
        dtype = (
            getattr(torch, dtype_str.split(".")[1])
            if dtype_str.startswith("torch.")
            else torch.bfloat16
        )
    else:
        dtype = dtype_str

    neuron_config_kwargs = {
        "tp_degree": neuron_config_dict.get("tp_degree", 4),
        "batch_size": neuron_config_dict.get("batch_size", 1),
        "seq_len": neuron_config_dict.get("seq_len", 2048),
        "torch_dtype": dtype,
    }

    neuron_config = MoENeuronConfig(**neuron_config_kwargs)

    try:
        model_config = TrinityInferenceConfig.from_pretrained(
            model_path, neuron_config=neuron_config
        )
    except (TypeError, AttributeError):
        model_config = TrinityInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

    model = NeuronTrinityForCausalLM(model_path, model_config)
    return model, neuron_config


def generate_with_neuron_model(model, input_ids, max_new_tokens: int):
    """Generate tokens using manual forward pass loop."""
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = (
            torch.arange(seq_len).unsqueeze(0).expand(generated_ids.shape[0], -1)
        )

        with torch.no_grad():
            outputs = model(generated_ids, position_ids=position_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    return generated_ids


@pytest.fixture(scope="module")
def compiled_model():
    """Load pre-compiled model."""
    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_model_loads(compiled_model):
    """Test that model loads successfully (smoke test)."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    logger.info("Smoke test passed - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(
        compiled_model, inputs.input_ids, max_new_tokens=20
    )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    logger.info("Generation test passed")
    logger.info("  Output: %s", output_text)


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish or repetitive)."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(
        compiled_model, inputs.input_ids, max_new_tokens=30
    )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text.split()) > 3, "Output should have multiple words"
    assert not _is_repetitive(output_text), "Output should not be repetitive"

    logger.info("Coherence test passed")
    logger.info("  Output: %s...", output_text[:100])


def test_top_token_valid(compiled_model, tokenizer):
    """Test that the top predicted token is a valid, decodable token.

    Unlike model-specific tests, this does not check for a specific expected token
    since different Trinity sizes produce different outputs.
    """
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    seq_len = inputs.input_ids.shape[1]
    position_ids = (
        torch.arange(seq_len).unsqueeze(0).expand(inputs.input_ids.shape[0], -1)
    )

    with torch.no_grad():
        outputs = compiled_model(inputs.input_ids, position_ids=position_ids)

    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    next_token_logits = logits[:, -1, :]
    top_token_id = torch.argmax(next_token_logits, dim=-1).item()
    top_token = tokenizer.decode([top_token_id]).strip()

    logger.info("Top token: '%s' (id=%d)", top_token, top_token_id)
    logger.info("Top logit: %.2f", next_token_logits[0, top_token_id].item())

    # The top token should be a non-empty, printable string
    assert len(top_token) > 0, f"Top token should be non-empty, got '{top_token}'"
    assert top_token_id < tokenizer.vocab_size, "Token ID should be within vocab range"
    logger.info("Top token validation passed")


# Deterministic prompts for 64-token generation comparison
TOKEN_MATCH_PROMPTS = [
    "Hello, how are you?",
    "Explain quantum computing in simple terms.",
    "Write a Python function that calculates the Fibonacci sequence.",
    "The capital of France is",
    "def fibonacci(n):",
    "What is the meaning of life?",
    "Once upon a time in a land far away,",
    "The quick brown fox jumps over the lazy",
]

# Number of tokens to generate for match rate testing
NUM_MATCH_TOKENS = 64


def test_token_match_rate(compiled_model, tokenizer):
    """Test Neuron vs CPU token match rate over 64 generated tokens.

    Generates tokens using greedy decoding (argmax) on both Neuron and CPU,
    then compares token-by-token. At least one prompt must achieve 100% match.
    """
    # Load CPU reference model
    logger.info("Loading CPU reference model for token match comparison...")
    cpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    cpu_model = cpu_model.to("cpu")
    cpu_model.eval()
    logger.info("CPU reference model loaded")

    best_rate = 0.0
    best_prompt = ""
    results = []

    for prompt in TOKEN_MATCH_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        # CPU reference: greedy generation
        with torch.no_grad():
            cpu_output = cpu_model.generate(
                input_ids,
                max_new_tokens=NUM_MATCH_TOKENS,
                do_sample=False,
            )
        cpu_tokens = cpu_output[0, prompt_len : prompt_len + NUM_MATCH_TOKENS]

        # Neuron: greedy generation via forward loop
        neuron_tokens_full = generate_with_neuron_model(
            compiled_model, input_ids, max_new_tokens=NUM_MATCH_TOKENS
        )
        neuron_tokens = neuron_tokens_full[0, prompt_len:]

        # Compare
        min_len = min(len(cpu_tokens), len(neuron_tokens))
        if min_len > 0:
            matched = (cpu_tokens[:min_len] == neuron_tokens[:min_len]).sum().item()
            rate = matched / min_len
        else:
            matched = 0
            rate = 0.0

        results.append((prompt, matched, min_len, rate))
        logger.info(
            "  '%s' -> %d/%d (%.1f%%)",
            prompt[:50],
            matched,
            min_len,
            rate * 100,
        )

        if rate > best_rate:
            best_rate = rate
            best_prompt = prompt

    del cpu_model

    # Log summary
    logger.info("Token match summary:")
    for prompt, matched, total, rate in results:
        logger.info("  %s: %d/%d (%.1f%%)", prompt[:50], matched, total, rate * 100)
    logger.info("Best: %.1f%% ('%s')", best_rate * 100, best_prompt[:50])

    assert best_rate >= 1.0, (
        f"No prompt achieved 100% token match. "
        f"Best: {best_rate * 100:.1f}% ('{best_prompt[:50]}')"
    )


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False

    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i + j] == word for j in range(max_repeat)):
            return True

    new_text = text[-100:] if len(text) > 100 else text
    if len(new_text) > 20:
        char_counts = {}
        for c in new_text:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(new_text)
        if max_char_ratio > 0.5:
            return True

    return False


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance.

    Pass criteria: avg TTFT must be below MAX_TTFT_MS (default 5000ms).
    Override with TRINITY_MAX_TTFT_MS environment variable.
    """
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids

    # Warmup
    for _ in range(3):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=position_ids)

    # Measure TTFT
    times = []
    for _ in range(10):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)

        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=position_ids)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    avg_ttft = sum(times) / len(times)
    min_ttft = min(times)
    max_ttft = max(times)
    logger.info(
        "TTFT: avg=%.2fms, min=%.2fms, max=%.2fms (threshold: %.0fms)",
        avg_ttft,
        min_ttft,
        max_ttft,
        MAX_TTFT_MS,
    )
    assert avg_ttft < MAX_TTFT_MS, (
        f"TTFT regression: {avg_ttft:.1f}ms exceeds threshold {MAX_TTFT_MS:.0f}ms"
    )


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput using naive forward loop.

    Pass criteria: throughput must exceed MIN_THROUGHPUT_TOK_S (default 0.5 tok/s).
    Override with TRINITY_MIN_THROUGHPUT_TOK_S environment variable.

    NOTE: This uses a naive loop (re-encodes full context each token), so throughput
    is much lower than proper CTE+TKG pipeline. The threshold is intentionally low.
    """
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    num_tokens = 50

    # Warmup
    _ = generate_with_neuron_model(compiled_model, input_ids, max_new_tokens=5)

    # Measure throughput
    start = time.perf_counter()
    _ = generate_with_neuron_model(compiled_model, input_ids, max_new_tokens=num_tokens)
    end = time.perf_counter()

    total_time = end - start
    throughput = num_tokens / total_time
    logger.info(
        "Throughput: %.2f tok/s (%d tokens in %.1fs, threshold: %.1f tok/s)",
        throughput,
        num_tokens,
        total_time,
        MIN_THROUGHPUT_TOK_S,
    )
    assert throughput > MIN_THROUGHPUT_TOK_S, (
        f"Throughput regression: {throughput:.2f} tok/s below threshold "
        f"{MIN_THROUGHPUT_TOK_S:.1f} tok/s"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    logger.info("=" * 80)
    logger.info("Trinity (AfmoeForCausalLM) Integration Tests")
    logger.info("=" * 80)
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Compiled path: %s", COMPILED_MODEL_PATH)

    logger.info("Loading compiled model from %s...", COMPILED_MODEL_PATH)
    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    logger.info("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("")
    logger.info("=" * 80)
    logger.info("Running Tests")
    logger.info("=" * 80)

    logger.info("")
    logger.info("1. Smoke Test (Model Loading)...")
    test_model_loads(model)

    logger.info("")
    logger.info("2. Generation Test...")
    test_model_generates(model, tokenizer)

    logger.info("")
    logger.info("3. Coherence Test...")
    test_output_coherence(model, tokenizer)

    logger.info("")
    logger.info("4. Top Token Validation...")
    test_top_token_valid(model, tokenizer)

    logger.info("")
    logger.info("5. Token Match Rate (64 tokens, Neuron vs CPU)...")
    test_token_match_rate(model, tokenizer)

    logger.info("")
    logger.info("=" * 80)
    logger.info("All tests passed!")
    logger.info("=" * 80)
