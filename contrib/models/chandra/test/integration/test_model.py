#!/usr/bin/env python3
"""
Integration tests for Chandra OCR VLM on NeuronX Distributed Inference.

Chandra (datalab-to/chandra) is a fine-tuned Qwen3-VL-8B for OCR. Unlike most
contrib models that validate text-backbone-only, this test validates the **full
multimodal VLM pipeline** with actual image input via NxDI's built-in Qwen3-VL
support (NeuronQwen3VLForCausalLM + ImageToTextInferenceConfig).

Requirements:
  - trn2.3xlarge with LNC=2 (4 logical NeuronCores)
  - Neuron SDK 2.28 (DLAMI 20260227)
  - venv: /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  - Model downloaded to MODEL_PATH (default: ~/models/chandra)

Usage:
  # On a trn2.3xlarge instance:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference \
      pytest test/integration/test_model.py -v --capture=tee-sys

  # Or run directly:
  python test/integration/test_model.py
"""

import os
import sys
import time

import pytest
import torch

# Add src to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from modeling_chandra import (
    get_chandra_neuron_configs,
    load_chandra_vllm,
    resize_image,
    run_chandra_ocr,
)

# --- Configuration ---
MODEL_PATH = os.environ.get(
    "CHANDRA_MODEL_PATH", os.path.expanduser("~/models/chandra")
)
BATCH_SIZE = 1  # Use 1 for accuracy test; batch tests are separate
TP_DEGREE = 4
SEQ_LEN = 8192

# Throughput thresholds (measured on trn2.3xlarge, LNC=2, SDK 2.28)
MIN_THROUGHPUT_TOKS = 50.0  # tok/s minimum for single-request steady state

# --- Fixtures ---


@pytest.fixture(scope="module")
def llm():
    """Load Chandra via vLLM-neuron. Shared across all tests in this module."""
    model = load_chandra_vllm(
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        tp_degree=TP_DEGREE,
    )
    yield model


@pytest.fixture(scope="module")
def processor():
    """Load the Chandra/Qwen3-VL processor for prompt formatting."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def test_image():
    """Create a synthetic test image with known text content."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Use default font -- available on all systems
    draw.text((50, 50), "Hello World", fill=(0, 0, 0))
    draw.text((50, 100), "Neuron OCR Test 2026", fill=(0, 0, 0))
    return img


# --- Tests ---


def test_smoke_load(llm):
    """Smoke test: model loads successfully."""
    assert llm is not None
    print("PASS: Model loaded successfully")


def test_config_valid():
    """Verify default configs produce valid dictionaries."""
    configs = get_chandra_neuron_configs()
    assert "text_neuron_config" in configs
    assert "vision_neuron_config" in configs
    assert configs["text_neuron_config"]["tp_degree"] == 4
    assert configs["text_neuron_config"]["torch_dtype"] == "bfloat16"
    assert configs["vision_neuron_config"]["attn_kernel_enabled"] is False
    assert configs["vision_neuron_config"]["mlp_kernel_enabled"] is False
    print("PASS: Config validation")


def test_image_resize():
    """Verify image resize keeps images within patch budget."""
    from PIL import Image

    # Large image should be resized
    large = Image.new("RGB", (3000, 2000), color=(128, 128, 128))
    resized = resize_image(large, max_long_side=1024)
    assert max(resized.size) <= 1024
    assert resized.size == (1024, 682)  # proportional

    # Small image should not be resized
    small = Image.new("RGB", (512, 384), color=(128, 128, 128))
    not_resized = resize_image(small, max_long_side=1024)
    assert not_resized.size == (512, 384)
    print("PASS: Image resize")


def test_ocr_generates_output(llm, processor, test_image):
    """Test that Chandra generates OCR output from an image input."""
    result = run_chandra_ocr(llm, test_image, processor, max_tokens=256)

    assert result["num_tokens"] > 0, "Model should generate at least one token"
    assert len(result["text"].strip()) > 0, "Output text should not be empty"
    assert result["latency_s"] > 0, "Latency should be positive"
    assert result["tokens_per_sec"] > 0, "Throughput should be positive"

    print(
        f"PASS: Generated {result['num_tokens']} tokens in {result['latency_s']:.2f}s"
    )
    print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
    print(f"  Output preview: {result['text'][:200]}")


def test_ocr_accuracy(llm, processor, test_image):
    """
    Accuracy test: verify the model recognizes text in the synthetic image.

    The test image contains "Hello World" and "Neuron OCR Test 2026".
    We check that at least one of these phrases appears in the output.
    """
    result = run_chandra_ocr(llm, test_image, processor, max_tokens=512)
    text = result["text"].lower()

    # Check for key phrases from the synthetic image
    found_hello = "hello" in text
    found_world = "world" in text
    found_neuron = "neuron" in text
    found_ocr = "ocr" in text
    found_2026 = "2026" in text

    matches = sum([found_hello, found_world, found_neuron, found_ocr, found_2026])

    print(f"  Output: {result['text'][:300]}")
    print(
        f"  Matches: hello={found_hello}, world={found_world}, "
        f"neuron={found_neuron}, ocr={found_ocr}, 2026={found_2026}"
    )
    print(f"  Score: {matches}/5 keywords found")

    # Check for complete phrases (stricter than individual keywords)
    found_hello_world = "hello world" in text
    found_neuron_ocr = "neuron ocr" in text

    # Require at least 4 of 5 keywords AND at least one complete phrase
    assert matches >= 4, (
        f"OCR accuracy too low: only {matches}/5 keywords found in output. "
        f"Expected at least 4. Output: {result['text'][:200]}"
    )
    assert found_hello_world or found_neuron_ocr, (
        f"No complete phrase found. Expected 'hello world' or 'neuron ocr' in output. "
        f"Output: {result['text'][:200]}"
    )
    print(f"PASS: OCR accuracy ({matches}/5 keywords, phrase match OK)")


def test_throughput(llm, processor, test_image):
    """
    Throughput test: verify token generation speed meets minimum threshold.

    Runs warmup + 2 steady-state passes and checks the faster one.
    """
    # Warmup
    _ = run_chandra_ocr(llm, test_image, processor, max_tokens=128)

    # Steady-state runs
    results = []
    for i in range(2):
        r = run_chandra_ocr(llm, test_image, processor, max_tokens=256)
        results.append(r)
        print(
            f"  Run {i + 1}: {r['num_tokens']} tokens, "
            f"{r['latency_s']:.2f}s, {r['tokens_per_sec']:.1f} tok/s"
        )

    best_throughput = max(r["tokens_per_sec"] for r in results)
    print(
        f"  Best throughput: {best_throughput:.1f} tok/s (threshold: {MIN_THROUGHPUT_TOKS})"
    )

    assert best_throughput >= MIN_THROUGHPUT_TOKS, (
        f"Throughput {best_throughput:.1f} tok/s below threshold {MIN_THROUGHPUT_TOKS} tok/s"
    )
    print(f"PASS: Throughput {best_throughput:.1f} tok/s")


def test_output_not_repetitive(llm, processor, test_image):
    """Verify output is not degenerate (not all same token or pure repetition)."""
    result = run_chandra_ocr(llm, test_image, processor, max_tokens=256)
    text = result["text"]

    # Check for excessive single-character repetition
    if len(text) > 20:
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_ratio = max(char_counts.values()) / len(text)
        assert max_ratio < 0.5, f"Output is repetitive (max char ratio {max_ratio:.2f})"

    # Check for word-level repetition
    words = text.split()
    if len(words) >= 10:
        for i in range(len(words) - 5):
            streak = all(words[i + j] == words[i] for j in range(5))
            assert not streak, f"Output has 5+ repeated word: '{words[i]}'"

    print(f"PASS: Output not repetitive ({len(text)} chars, {len(words)} words)")


# --- Main ---

if __name__ == "__main__":
    print("=" * 70)
    print("Chandra OCR VLM - Integration Tests")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: tp={TP_DEGREE}, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print()

    # Run tests manually (outside pytest)
    from PIL import Image, ImageDraw

    print("Loading model...")
    t0 = time.time()
    llm = load_chandra_vllm(
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        tp_degree=TP_DEGREE,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Create test image
    img = Image.new("RGB", (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Hello World", fill=(0, 0, 0))
    draw.text((50, 100), "Neuron OCR Test 2026", fill=(0, 0, 0))

    print("--- test_smoke_load ---")
    test_smoke_load(llm)

    print("\n--- test_config_valid ---")
    test_config_valid()

    print("\n--- test_image_resize ---")
    test_image_resize()

    print("\n--- test_ocr_generates_output ---")
    test_ocr_generates_output(llm, processor, img)

    print("\n--- test_ocr_accuracy ---")
    test_ocr_accuracy(llm, processor, img)

    print("\n--- test_throughput ---")
    test_throughput(llm, processor, img)

    print("\n--- test_output_not_repetitive ---")
    test_output_not_repetitive(llm, processor, img)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
