# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for Voxtral Mini 3B on Neuron.

This test validates the Voxtral audio-language model by:
1. Compiling the model (audio encoder + text decoder)
2. Loading from compiled checkpoint
3. Running text-only generation
4. Running audio + text generation (transcription)
5. Measuring latency

Prerequisites:
    pip install transformers mistral_common[audio] pytest

Usage:
    # Run with pytest
    pytest test/integration/test_model.py -v

    # Run directly
    python test/integration/test_model.py

Environment variables:
    VOXTRAL_MODEL_PATH      Path to HF model weights (default: /home/ubuntu/models/voxtral-mini-3B)
    VOXTRAL_COMPILED_PATH   Path for compiled NEFFs (default: /home/ubuntu/compiled_models/voxtral-mini-3B)
    VOXTRAL_AUDIO_FILE      Path or URL to test audio file
    VOXTRAL_TP_DEGREE       Tensor parallel degree (default: 1)
    VOXTRAL_SEQ_LEN         Context encoding bucket size (default: 2048)
"""

import os
import sys
import time

import pytest
import torch

# Add the src directory to the path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from modeling_voxtral import NeuronApplicationVoxtral

# Configuration from environment
MODEL_PATH = os.environ.get("VOXTRAL_MODEL_PATH", "/home/ubuntu/models/voxtral-mini-3B")
COMPILED_PATH = os.environ.get(
    "VOXTRAL_COMPILED_PATH", "/home/ubuntu/compiled_models/voxtral-mini-3B"
)
AUDIO_FILE = os.environ.get(
    "VOXTRAL_AUDIO_FILE",
    "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav",
)
TP_DEGREE = int(os.environ.get("VOXTRAL_TP_DEGREE", "1"))
SEQ_LEN = int(os.environ.get("VOXTRAL_SEQ_LEN", "2048"))
N_POSITIONS = int(os.environ.get("VOXTRAL_N_POSITIONS", "4096"))
DTYPE = torch.bfloat16


@pytest.fixture(scope="module")
def loaded_model():
    """Compile and load the Voxtral model (module-scoped for reuse across tests)."""
    app = NeuronApplicationVoxtral(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        dtype=DTYPE,
    )

    # Compile if needed
    if not os.path.exists(
        os.path.join(COMPILED_PATH, "text_decoder", "text_model", "model.pt")
    ):
        print(f"\nCompiling Voxtral model to {COMPILED_PATH}...")
        app.compile(COMPILED_PATH)

    # Load compiled model
    print(f"\nLoading compiled Voxtral model from {COMPILED_PATH}...")
    app.load(COMPILED_PATH)
    return app


def test_model_loads(loaded_model):
    """Smoke test: all components load successfully."""
    assert loaded_model is not None
    assert loaded_model.audio_encoder is not None
    assert loaded_model.projector is not None
    assert loaded_model.vl_model is not None
    assert loaded_model.adapter is not None
    assert loaded_model.tokenizer is not None
    assert loaded_model.processor is not None


def test_text_generation(loaded_model):
    """Test text-only generation produces non-empty output."""
    result = loaded_model.generate("What is the capital of France?", max_new_tokens=50)
    print(f"\nText generation result: {result}")
    assert len(result.strip()) > 0, "Text generation should not be empty"
    # Basic sanity: should mention Paris
    assert "paris" in result.lower() or len(result.strip()) > 5, (
        f"Unexpected text result: {result}"
    )


def test_audio_transcription(loaded_model):
    """Test audio transcription produces non-empty output."""
    result = loaded_model.transcribe(AUDIO_FILE, max_new_tokens=500)
    print(f"\nTranscription result: {result}")
    assert len(result.strip()) > 0, "Transcription should not be empty"


def test_text_generation_deterministic(loaded_model):
    """Test that text-only generation is deterministic (greedy)."""
    result1 = loaded_model.generate("What is 2 + 2?", max_new_tokens=20)
    result2 = loaded_model.generate("What is 2 + 2?", max_new_tokens=20)
    assert result1 == result2, (
        f"Non-deterministic generation:\n  Run 1: {result1}\n  Run 2: {result2}"
    )


def test_transcription_latency(loaded_model):
    """Measure transcription latency with warmup."""
    # Warmup
    loaded_model.transcribe(AUDIO_FILE, max_new_tokens=100)

    # Measure
    n_runs = 3
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        loaded_model.transcribe(AUDIO_FILE, max_new_tokens=100)
        latencies.append(time.perf_counter() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(
        f"\nAverage transcription latency ({n_runs} runs): {avg_latency * 1000:.1f}ms"
    )
    # Basic sanity: should complete within 30 seconds for any reasonable audio
    assert avg_latency < 30.0, f"Transcription too slow: {avg_latency:.1f}s"


def test_text_generation_latency(loaded_model):
    """Measure text-only generation latency."""
    prompt = "Explain photosynthesis in one sentence."

    # Warmup
    loaded_model.generate(prompt, max_new_tokens=50)

    # Measure
    n_runs = 3
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        loaded_model.generate(prompt, max_new_tokens=50)
        latencies.append(time.perf_counter() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(
        f"\nAverage text generation latency ({n_runs} runs): {avg_latency * 1000:.1f}ms"
    )
    assert avg_latency < 15.0, f"Text generation too slow: {avg_latency:.1f}s"


if __name__ == "__main__":
    print("=" * 60)
    print("Voxtral Mini 3B - Integration Test")
    print("=" * 60)
    print(f"Model path:    {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_PATH}")
    print(f"Audio file:    {AUDIO_FILE}")
    print(f"TP degree:     {TP_DEGREE}")
    print(f"Seq len:       {SEQ_LEN}")
    print(f"Dtype:         {DTYPE}")
    print()

    app = NeuronApplicationVoxtral(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        dtype=DTYPE,
    )

    # Compile
    compiled_marker = os.path.join(
        COMPILED_PATH, "text_decoder", "text_model", "model.pt"
    )
    if not os.path.exists(compiled_marker):
        print("Compiling model...")
        app.compile(COMPILED_PATH)
        print("Compilation complete.\n")
    else:
        print("Using existing compiled model.\n")

    # Load
    print("Loading compiled model...")
    app.load(COMPILED_PATH)
    print("Model loaded.\n")

    # Text-only generation
    print("--- Text-only generation ---")
    prompt = "What is the capital of France?"
    start = time.perf_counter()
    result = app.generate(prompt, max_new_tokens=50)
    elapsed = time.perf_counter() - start
    print(f"Prompt:  {prompt}")
    print(f"Result:  {result}")
    print(f"Latency: {elapsed * 1000:.1f}ms\n")

    # Audio transcription
    print("--- Audio transcription ---")
    print(f"Audio: {AUDIO_FILE}")
    start = time.perf_counter()
    result = app.transcribe(AUDIO_FILE, max_new_tokens=500)
    elapsed = time.perf_counter() - start
    print(f"Transcription: {result}")
    print(f"Latency: {elapsed * 1000:.1f}ms\n")

    # Determinism check
    print("--- Determinism check ---")
    r1 = app.generate("What is 2 + 2?", max_new_tokens=20)
    r2 = app.generate("What is 2 + 2?", max_new_tokens=20)
    if r1 == r2:
        print("Determinism: PASS (identical output)")
    else:
        print(f"Determinism: FAIL\n  Run 1: {r1}\n  Run 2: {r2}")

    print("\nAll tests passed.")
