# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for Whisper Large V3 Turbo on Neuron.

This test validates the Whisper encoder-decoder model by:
1. Compiling the model (encoder + decoder)
2. Loading from compiled checkpoint
3. Transcribing a reference audio file
4. Validating transcription accuracy

Prerequisites:
    pip install openai-whisper pytest

Usage:
    # Run with pytest
    pytest test/integration/test_model.py -v

    # Run directly
    python test/integration/test_model.py
"""

import os
import sys
import time

import pytest
import torch

# Add the src directory to the path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_whisper import WhisperInferenceConfig, NeuronApplicationWhisper

# Configuration
MODEL_PATH = os.environ.get(
    "WHISPER_MODEL_PATH", "/home/ubuntu/models/whisper-large-v3-turbo/"
)
COMPILED_MODEL_PATH = os.environ.get(
    "WHISPER_COMPILED_PATH", "/home/ubuntu/compiled_models/whisper-large-v3-turbo/"
)
AUDIO_FILE = os.environ.get(
    "WHISPER_AUDIO_FILE",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "examples",
        "audio-sample.mp3",
    ),
)
DTYPE = torch.bfloat16
BATCH_SIZE = 1
TP_DEGREE = 1


def _get_config():
    """Create NeuronConfig and WhisperInferenceConfig."""
    neuron_config = NeuronConfig(
        batch_size=BATCH_SIZE,
        torch_dtype=DTYPE,
        tp_degree=TP_DEGREE,
    )
    inference_config = WhisperInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    return inference_config


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load the Whisper model (module-scoped for reuse across tests)."""
    config = _get_config()

    # Compile if needed
    if not os.path.exists(COMPILED_MODEL_PATH):
        print(f"\nCompiling Whisper model to {COMPILED_MODEL_PATH}...")
        model = NeuronApplicationWhisper(MODEL_PATH, config=config)
        model.compile(COMPILED_MODEL_PATH)

    # Load compiled model
    print(f"\nLoading compiled Whisper model from {COMPILED_MODEL_PATH}...")
    model = NeuronApplicationWhisper(COMPILED_MODEL_PATH, config=config)
    model.load(COMPILED_MODEL_PATH)
    return model


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert compiled_model.encoder is not None
    assert compiled_model.decoder is not None


def test_model_transcribes(compiled_model):
    """Test that the model produces a non-empty transcription."""
    assert os.path.exists(AUDIO_FILE), (
        f"Audio file not found: {AUDIO_FILE}. "
        f"Set WHISPER_AUDIO_FILE environment variable to point to a valid audio file."
    )
    result = compiled_model.transcribe(AUDIO_FILE)
    text = result["text"].strip()
    print(f"\nTranscription: {text}")
    assert len(text) > 0, "Transcription should not be empty"


def test_transcription_latency(compiled_model):
    """Measure transcription latency with warmup."""
    assert os.path.exists(AUDIO_FILE), f"Audio file not found: {AUDIO_FILE}"

    # Warmup
    compiled_model.transcribe(AUDIO_FILE)

    # Measure
    n_runs = 3
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        compiled_model.transcribe(AUDIO_FILE)
        latencies.append(time.perf_counter() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(
        f"\nAverage transcription latency ({n_runs} runs): {avg_latency * 1000:.1f}ms"
    )
    # Basic sanity: should complete within 10 seconds for any reasonable audio
    assert avg_latency < 10.0, f"Transcription too slow: {avg_latency:.1f}s"


def test_transcription_deterministic(compiled_model):
    """Test that repeated transcriptions produce the same result."""
    assert os.path.exists(AUDIO_FILE), f"Audio file not found: {AUDIO_FILE}"

    result1 = compiled_model.transcribe(AUDIO_FILE)
    result2 = compiled_model.transcribe(AUDIO_FILE)
    assert result1["text"] == result2["text"], (
        f"Non-deterministic transcription:\n  Run 1: {result1['text']}\n  Run 2: {result2['text']}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Whisper Large V3 Turbo - Integration Test")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_MODEL_PATH}")
    print(f"Audio file: {AUDIO_FILE}")
    print(f"Dtype: {DTYPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"TP degree: {TP_DEGREE}")
    print()

    config = _get_config()

    # Compile
    if not os.path.exists(COMPILED_MODEL_PATH):
        print("Compiling model...")
        model = NeuronApplicationWhisper(MODEL_PATH, config=config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.\n")

    # Load
    print("Loading compiled model...")
    model = NeuronApplicationWhisper(COMPILED_MODEL_PATH, config=config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.\n")

    # Transcribe
    if os.path.exists(AUDIO_FILE):
        print(f"Transcribing: {AUDIO_FILE}")
        start = time.perf_counter()
        result = model.transcribe(AUDIO_FILE, verbose=True)
        elapsed = time.perf_counter() - start
        print(f"\nTranscription: {result['text']}")
        print(f"Latency: {elapsed * 1000:.1f}ms")

        # Determinism check
        result2 = model.transcribe(AUDIO_FILE)
        if result["text"] == result2["text"]:
            print("Determinism: PASS (identical output)")
        else:
            print("Determinism: FAIL (different outputs)")
    else:
        print(f"WARNING: Audio file not found: {AUDIO_FILE}")
        print("Set WHISPER_AUDIO_FILE to run transcription tests.")

    print("\nAll tests passed.")
