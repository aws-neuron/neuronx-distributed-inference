#!/usr/bin/env python3
"""Test client for Qwen3-Coder-Next vLLM server.

Usage:
    python test_vllm_client.py [--port PORT] [--prompt PROMPT]
"""

import argparse
import json
import time
import requests


def chat_completion(
    base_url: str, messages: list[dict], max_tokens: int = 128, temperature: float = 0.0
):
    """Send a chat completion request."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "/mnt/Qwen3-Coder-Next",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload)
    elapsed = time.time() - t0
    resp.raise_for_status()
    result = resp.json()
    return result, elapsed


def completion(
    base_url: str, prompt: str, max_tokens: int = 128, temperature: float = 0.0
):
    """Send a text completion request."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": "/mnt/Qwen3-Coder-Next",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload)
    elapsed = time.time() - t0
    resp.raise_for_status()
    result = resp.json()
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-Coder-Next vLLM server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"Testing vLLM server at {base_url}")
    print("=" * 60)

    # Test 1: Health check
    print("\n--- Test 1: Health Check ---")
    try:
        resp = requests.get(f"{base_url}/health")
        print(f"Status: {resp.status_code}")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # Test 2: Model list
    print("\n--- Test 2: List Models ---")
    resp = requests.get(f"{base_url}/v1/models")
    models = resp.json()
    print(
        f"Models: {json.dumps(models['data'][0]['id'] if models.get('data') else 'none', indent=2)}"
    )

    # Test 3: Simple completion
    print("\n--- Test 3: Completion (Fibonacci) ---")
    prompt = 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n'
    result, elapsed = completion(base_url, prompt, max_tokens=64)
    text = result["choices"][0]["text"]
    tokens = result["usage"]["completion_tokens"]
    print(f"Prompt: {repr(prompt[:50])}")
    print(f"Output ({tokens} tokens, {elapsed:.2f}s, {tokens / elapsed:.1f} tok/s):")
    print(f"  {text[:200]}")

    # Test 4: Chat completion
    print("\n--- Test 4: Chat Completion ---")
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result, elapsed = chat_completion(base_url, messages, max_tokens=32)
    content = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["completion_tokens"]
    print(f"Response ({tokens} tokens, {elapsed:.2f}s):")
    print(f"  {content}")

    # Test 5: Code generation
    print("\n--- Test 5: Code Generation ---")
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to check if a number is prime. Be concise.",
        },
    ]
    result, elapsed = chat_completion(base_url, messages, max_tokens=128)
    content = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["completion_tokens"]
    print(f"Response ({tokens} tokens, {elapsed:.2f}s, {tokens / elapsed:.1f} tok/s):")
    print(f"  {content[:300]}")

    print("\n" + "=" * 60)
    print("All tests complete!")


if __name__ == "__main__":
    main()
