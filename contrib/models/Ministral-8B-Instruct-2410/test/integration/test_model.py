"""
Integration test for Ministral-8B-Instruct-2410 on NeuronX.

Tests basic model loading and inference via vLLM-neuron.
Requires: trn2.3xlarge instance with SDK 2.29.
Note: Config must be patched first (sliding_window=null, layer_types removed).
"""

import sys
import time

import requests


MODEL_PATH = "/home/ubuntu/models/Ministral-8B-Instruct-2410"
TP_DEGREE = 4
MAX_MODEL_LEN = 8192
PORT = 8000


def test_smoke():
    """Test that the model loads and serves via vLLM."""
    base_url = f"http://localhost:{PORT}"
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            assert len(models["data"]) > 0, "No models loaded"
            print(f"PASS: Server running with model {models['data'][0]['id']}")
            return
    except requests.ConnectionError:
        pass

    print("Server not running. Apply config patch and start vLLM server first.")
    sys.exit(1)


def test_generation():
    """Test basic text generation."""
    base_url = f"http://localhost:{PORT}"
    model_name = requests.get(f"{base_url}/v1/models").json()["data"][0]["id"]
    payload = {
        "model": model_name,
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
    model_name = requests.get(f"{base_url}/v1/models").json()["data"][0]["id"]

    for _ in range(2):
        requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "Hello",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=60,
        )

    prompt = "Explain the theory of relativity in simple terms. " * 5
    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0,
        },
        timeout=120,
    )
    elapsed = time.time() - start

    result = resp.json()
    tokens = result["usage"]["completion_tokens"]
    tok_s = tokens / elapsed

    assert tok_s > 40, f"Throughput too low: {tok_s:.1f} tok/s (expected >40)"
    print(f"PASS: Throughput {tok_s:.1f} tok/s ({tokens} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    test_smoke()
    test_generation()
    test_throughput()
    print("\nAll tests passed.")
