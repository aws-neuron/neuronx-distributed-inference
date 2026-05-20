#!/usr/bin/env python3
# Copyright 2025 (c) Amazon.com and Affiliates
"""Online inference client for Isaac vLLM server.

Sends requests to a running vLLM OpenAI-compatible API server.

Usage:
    # Start server first (see start-vllm-server.sh)
    python run_online_inference.py [--base-url http://localhost:8080]
"""

import argparse
import json
import time

import requests


def chat_completion(base_url, messages, max_tokens=100, temperature=0):
    """Send a chat completion request to the vLLM server."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "Isaac-0.2-2B-Preview",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    response = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    response.raise_for_status()
    result = response.json()
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    args = parser.parse_args()

    # Test 1: Text-only
    print("=" * 60)
    print("Test 1: Text-only")
    print("=" * 60)
    messages = [
        {"role": "user", "content": "What is the capital of France? Explain briefly."}
    ]
    result, elapsed = chat_completion(args.base_url, messages)
    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    print(f"Response: {text[:200]}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Usage: {usage}")

    # Test 2: Text-only (longer)
    print("\n" + "=" * 60)
    print("Test 2: Text-only (longer)")
    print("=" * 60)
    messages = [
        {
            "role": "user",
            "content": "Explain quantum entanglement in simple terms.",
        }
    ]
    result, elapsed = chat_completion(args.base_url, messages)
    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    print(f"Response: {text[:200]}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Usage: {usage}")

    # Test 3: Image+text
    print("\n" + "=" * 60)
    print("Test 3: Image+text")
    print("=" * 60)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                    },
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    try:
        result, elapsed = chat_completion(args.base_url, messages)
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"Response: {text[:200]}")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Usage: {usage}")
    except Exception as e:
        print(f"Image+text failed: {e}")

    print("\nAll online tests completed.")


if __name__ == "__main__":
    main()
