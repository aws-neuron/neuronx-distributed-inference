#!/usr/bin/env python3
"""
Integration tests for Qwen3-Coder-480B-A35B-Instruct on vLLM/Neuron.

Tests generation quality, TTFT, throughput, and concurrent serving against
a running vLLM server. This model uses the existing qwen3_moe architecture
in NxD Inference, so we test via the vLLM OpenAI-compatible API.

Prerequisites:
    - vLLM server running with Qwen3-Coder-480B (see src/qwen3_coder_vllm.sh)

Usage:
    # With pytest:
    pytest test/integration/test_model.py -v --capture=tee-sys

    # Standalone:
    python test/integration/test_model.py
"""

import json
import os
import sys
import threading
import time

import pytest
import requests

API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000")
MODEL = os.environ.get("VLLM_MODEL", "/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/")

# Thresholds (conservative for 480B MoE on trn2.48xlarge)
TTFT_THRESHOLD_S = 5.0  # Max TTFT for short prompts (s)
THROUGHPUT_THRESHOLD = 5.0  # Min single-request throughput (tok/s)
CONCURRENT_TPS_THRESHOLD = 20  # Min aggregate TPS at 4 concurrent


def _check_server():
    """Check if vLLM server is running."""
    try:
        resp = requests.get(f"{API_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def check_server():
    """Skip all tests if server is not running."""
    if not _check_server():
        pytest.skip(f"vLLM server not running at {API_URL}")


@pytest.fixture(scope="module")
def warmup():
    """Warmup request to ensure model is ready."""
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={"model": MODEL, "prompt": "Hello", "max_tokens": 16, "temperature": 0.1},
        timeout=120,
    )
    assert resp.status_code == 200, f"Warmup failed: {resp.text}"
    return resp.json()


def test_smoke(warmup):
    """Test that server responds and model generates output."""
    data = warmup
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["text"]) > 0
    print(f"  Smoke test: {data['usage']['completion_tokens']} tokens generated")


def test_generation_quality_fibonacci(warmup):
    """Test that model generates correct Fibonacci code."""
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Write a Python function to compute the nth Fibonacci number using memoization:\n```python\n",
            "max_tokens": 256,
            "temperature": 0.0,
        },
        timeout=120,
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["text"]

    # Should contain function definition and memoization-related keywords
    assert "def " in text or "fibonacci" in text.lower(), (
        f"No function definition found: {text[:200]}"
    )
    # Should not be garbage (repetitive single chars)
    unique_chars = len(set(text.strip()))
    assert unique_chars > 5, (
        f"Output appears to be garbage (only {unique_chars} unique chars)"
    )
    print(f"  Fibonacci: {len(text)} chars, coherent")


def test_generation_quality_math(warmup):
    """Test basic math reasoning."""
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "What is 2 + 2? Answer with just the number:",
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=120,
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["text"].strip()
    assert "4" in text, f"Expected '4' in response, got: '{text}'"
    print(f"  Math: '{text}'")


def test_generation_quality_translation(warmup):
    """Test translation capability."""
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Translate to French: 'The cat sat on the mat.'",
            "max_tokens": 32,
            "temperature": 0.0,
        },
        timeout=120,
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["text"].strip().lower()
    # Should contain French words
    french_indicators = ["le", "la", "chat", "sur", "tapis", "assis", "mat"]
    has_french = any(word in text for word in french_indicators)
    assert has_french, f"No French detected in: '{text}'"
    print(f"  Translation: '{text[:100]}'")


def test_ttft_short_prompt(warmup):
    """Test TTFT for a short prompt (should use small bucket with auto-bucketing)."""
    payload = {
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 32,
        "temperature": 0.1,
        "stream": True,
    }
    start = time.time()
    ttft = None
    resp = requests.post(
        f"{API_URL}/v1/completions", json=payload, timeout=120, stream=True
    )
    for line in resp.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                if ttft is None:
                    ttft = time.time() - start
                    break

    assert ttft is not None, "No streaming tokens received"
    assert ttft < TTFT_THRESHOLD_S, (
        f"TTFT {ttft:.2f}s exceeds {TTFT_THRESHOLD_S}s threshold"
    )
    print(f"  TTFT (short prompt): {ttft:.3f}s (threshold: {TTFT_THRESHOLD_S}s)")


def test_throughput_single_request(warmup):
    """Test single-request decode throughput."""
    start = time.time()
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Write a Python hello world program",
            "max_tokens": 128,
            "temperature": 0.1,
        },
        timeout=120,
    )
    elapsed = time.time() - start

    assert resp.status_code == 200
    tokens = resp.json()["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0

    assert tps > THROUGHPUT_THRESHOLD, (
        f"Throughput {tps:.2f} tok/s below {THROUGHPUT_THRESHOLD} tok/s threshold"
    )
    print(f"  Throughput: {tps:.2f} tok/s ({tokens} tokens in {elapsed:.2f}s)")


def test_concurrent_throughput(warmup):
    """Test aggregate throughput with 4 concurrent requests."""
    prompts = [
        "Write a Python function to compute Fibonacci numbers.",
        "Explain what tensor parallelism is in 3 sentences.",
        "Write a hello world program in Rust.",
        "What is 127 * 389? Show your work.",
    ]
    results = [None] * len(prompts)

    def send(idx, prompt):
        try:
            data = json.dumps(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.6,
                }
            ).encode()
            import urllib.request

            req = urllib.request.Request(
                f"{API_URL}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                result = json.loads(r.read().decode())
            results[idx] = {
                "completion_tokens": result["usage"]["completion_tokens"],
                "ok": len(result["choices"][0]["message"]["content"].strip()) > 10,
            }
        except Exception as e:
            results[idx] = {"error": str(e)}

    wall_start = time.time()
    threads = [
        threading.Thread(target=send, args=(i, p)) for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    wall_elapsed = time.time() - wall_start

    valid = [r for r in results if r and "error" not in r]
    total_tokens = sum(r["completion_tokens"] for r in valid)
    aggregate_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0
    all_ok = all(r.get("ok", False) for r in valid)

    assert len(valid) == len(prompts), (
        f"Only {len(valid)}/{len(prompts)} requests succeeded"
    )
    assert all_ok, "Some responses had quality issues"
    assert aggregate_tps > CONCURRENT_TPS_THRESHOLD, (
        f"Aggregate TPS {aggregate_tps:.2f} below {CONCURRENT_TPS_THRESHOLD} threshold"
    )
    print(
        f"  Concurrent (4): {aggregate_tps:.2f} agg tok/s, {total_tokens} tokens, "
        f"{wall_elapsed:.2f}s wall time"
    )


def test_chat_completions_api(warmup):
    """Test the chat completions API endpoint."""
    resp = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 64,
            "temperature": 0.6,
        },
        timeout=120,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert len(content.strip()) > 10, f"Response too short: '{content}'"
    print(f"  Chat API: {data['usage']['completion_tokens']} tokens")


if __name__ == "__main__":
    print("=" * 70)
    print("Qwen3-Coder-480B-A35B-Instruct Integration Tests")
    print(f"Server: {API_URL}")
    print(f"Model: {MODEL}")
    print("=" * 70)

    if not _check_server():
        print(f"\nERROR: vLLM server not running at {API_URL}")
        print("Start the server first: ./src/qwen3_coder_vllm.sh")
        sys.exit(1)

    # Warmup
    print("\n--- Warmup ---")
    resp = requests.post(
        f"{API_URL}/v1/completions",
        json={"model": MODEL, "prompt": "Hello", "max_tokens": 16, "temperature": 0.1},
        timeout=120,
    )
    assert resp.status_code == 200, f"Warmup failed: {resp.text}"
    warmup_data = resp.json()
    print(f"  OK: {warmup_data['usage']['completion_tokens']} tokens")

    # Run tests
    tests = [
        ("Smoke Test", lambda: test_smoke(warmup_data)),
        ("Fibonacci Quality", lambda: test_generation_quality_fibonacci(warmup_data)),
        ("Math Quality", lambda: test_generation_quality_math(warmup_data)),
        (
            "Translation Quality",
            lambda: test_generation_quality_translation(warmup_data),
        ),
        ("TTFT (short prompt)", lambda: test_ttft_short_prompt(warmup_data)),
        (
            "Single-Request Throughput",
            lambda: test_throughput_single_request(warmup_data),
        ),
        ("Concurrent Throughput (4x)", lambda: test_concurrent_throughput(warmup_data)),
        ("Chat Completions API", lambda: test_chat_completions_api(warmup_data)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    sys.exit(0 if failed == 0 else 1)
