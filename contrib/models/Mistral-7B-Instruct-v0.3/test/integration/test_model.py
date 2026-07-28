"""
Integration test for Mistral-7B-Instruct-v0.3 on NeuronX.

Tests basic model loading and inference via vLLM-neuron.
Requires: trn2.3xlarge instance with SDK 2.29.
"""

import json
import os
import subprocess
import sys
import time

import requests


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TP_DEGREE = 4
MAX_MODEL_LEN = 8192
PORT = 8000


def test_smoke():
    """Test that the model loads and serves via vLLM."""
    base_url = f"http://localhost:{PORT}"

    # Check if server is already running
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            assert len(models["data"]) > 0, "No models loaded"
            print(f"PASS: Server running with model {models['data'][0]['id']}")
            return
    except requests.ConnectionError:
        pass

    print("Server not running. Start vLLM server first:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {MODEL_ID} \\")
    print(f"    --tensor-parallel-size {TP_DEGREE} \\")
    print(f"    --max-model-len {MAX_MODEL_LEN} \\")
    print(f"    --max-num-seqs 1 \\")
    print(f"    --no-enable-prefix-caching \\")
    print(f"    --port {PORT}")
    sys.exit(1)


def test_generation():
    """Test basic text generation."""
    base_url = f"http://localhost:{PORT}"
    payload = {
        "model": MODEL_ID,
        "prompt": "The capital of France is",
        "max_tokens": 32,
        "temperature": 0,
    }

    resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=60)
    assert resp.status_code == 200, f"Generation failed: {resp.text}"

    result = resp.json()
    text = result["choices"][0]["text"]
    assert len(text) > 0, "Empty generation"
    assert "Paris" in text, f"Expected 'Paris' in output, got: {text}"
    print(f"PASS: Generated text contains 'Paris': {text.strip()[:80]}")


def test_throughput():
    """Test that throughput meets minimum threshold."""
    base_url = f"http://localhost:{PORT}"

    # Warm up
    for _ in range(2):
        requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": MODEL_ID,
                "prompt": "Hello",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=60,
        )

    # Measure
    prompt = "Explain the theory of relativity in simple terms. " * 5
    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={"model": MODEL_ID, "prompt": prompt, "max_tokens": 128, "temperature": 0},
        timeout=120,
    )
    elapsed = time.time() - start

    result = resp.json()
    tokens = result["usage"]["completion_tokens"]
    tok_s = tokens / elapsed

    assert tok_s > 50, f"Throughput too low: {tok_s:.1f} tok/s (expected >50)"
    print(f"PASS: Throughput {tok_s:.1f} tok/s ({tokens} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    test_smoke()
    test_generation()
    test_throughput()
    print("\nAll tests passed.")
