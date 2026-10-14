#!/usr/bin/env python3
"""
Test client for ZAYA1-base vLLM server.

Tests the OpenAI-compatible API endpoint with various prompts
to verify the model is serving correctly.

Usage:
    # Basic test (server running on localhost:8000):
    python test_vllm_client.py

    # Custom server:
    python test_vllm_client.py --host localhost --port 8000

    # With concurrent requests:
    python test_vllm_client.py --concurrent 4
"""

import argparse
import asyncio
import json
import time
import urllib.request
import urllib.error


# Test prompts with expected content
TEST_PROMPTS = [
    {
        "prompt": "The capital of France is",
        "expected_contains": "Paris",
        "max_tokens": 20,
    },
    {
        "prompt": "The largest ocean on Earth is the",
        "expected_contains": "Pacific",
        "max_tokens": 20,
    },
    {
        "prompt": "Albert Einstein was born in",
        "expected_contains": "Ulm",
        "max_tokens": 30,
    },
    {
        "prompt": "The speed of light in meters per second is approximately",
        "expected_contains": "299",
        "max_tokens": 30,
    },
]


def get_model_name(base_url):
    """Auto-detect the model name from the server."""
    try:
        url = f"{base_url}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            models = result.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "ZAYA1-base"  # fallback


def send_completion_request(
    base_url, model_name, prompt, max_tokens=30, temperature=0.0
):
    """Send a completion request to the vLLM server."""
    url = f"{base_url}/v1/completions"
    data = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
            elapsed = time.time() - start
            return result, elapsed
    except urllib.error.URLError as e:
        return {"error": str(e)}, time.time() - start


def test_health(base_url):
    """Check if the server is healthy."""
    try:
        url = f"{base_url}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def test_models(base_url):
    """List available models."""
    try:
        url = f"{base_url}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            return result
    except Exception as e:
        return {"error": str(e)}


def run_tests(base_url):
    """Run all test prompts sequentially."""
    print(f"\n{'=' * 60}")
    print(f"Testing ZAYA1-base vLLM Server at {base_url}")
    print(f"{'=' * 60}\n")

    # Health check
    print("[1/3] Health check...", end=" ")
    if test_health(base_url):
        print("OK")
    else:
        print("FAILED - Server not responding")
        return False

    # Model list + auto-detect name
    print("[2/3] Model list...", end=" ")
    models = test_models(base_url)
    if "error" not in models:
        model_ids = [m["id"] for m in models.get("data", [])]
        print(f"OK - Models: {model_ids}")
    else:
        print(f"FAILED - {models['error']}")
        return False

    model_name = get_model_name(base_url)
    print(f"  Using model: {model_name}")

    # Completion tests
    print(f"[3/3] Running {len(TEST_PROMPTS)} completion tests...\n")

    passed = 0
    total_tokens = 0
    total_time = 0

    for i, test in enumerate(TEST_PROMPTS):
        prompt = test["prompt"]
        expected = test["expected_contains"]
        max_tokens = test["max_tokens"]

        print(f'  Test {i + 1}: "{prompt}"')
        result, elapsed = send_completion_request(
            base_url, model_name, prompt, max_tokens
        )

        if "error" in result:
            print(f"    ERROR: {result['error']}")
            continue

        text = result["choices"][0]["text"]
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        total_tokens += completion_tokens
        total_time += elapsed

        contains_expected = expected.lower() in text.lower()
        status = "PASS" if contains_expected else "FAIL"
        passed += 1 if contains_expected else 0

        print(f'    Output: "{text.strip()}"')
        print(f"    Expected '{expected}': {status}")
        print(f"    Tokens: {completion_tokens}, Time: {elapsed:.2f}s")
        print()

    # Summary
    print(f"{'=' * 60}")
    print(f"Results: {passed}/{len(TEST_PROMPTS)} passed")
    if total_time > 0:
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {total_tokens / total_time:.1f} tok/s")
    print(f"{'=' * 60}")

    return passed == len(TEST_PROMPTS)


def main():
    parser = argparse.ArgumentParser(description="Test ZAYA1-base vLLM server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    success = run_tests(base_url)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
