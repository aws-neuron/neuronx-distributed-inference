#!/usr/bin/env python3
"""Online inference example for InternVL3-8B-Instruct via OpenAI-compatible API."""

import requests
import json


def text_completion(prompt: str, base_url: str = "http://localhost:8000"):
    """Send a text-only completion request."""
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "/mnt/models/InternVL3-8B-Instruct/",
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    result = response.json()
    return result["choices"][0]["text"]


def multimodal_chat(
    prompt: str, image_url: str, base_url: str = "http://localhost:8000"
):
    """Send a multimodal chat request with image."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "/mnt/models/InternVL3-8B-Instruct/",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    print("=== Text Completion ===")
    text = text_completion("The capital of France is")
    print(f"Response: {text}")
    print()

    print("=== Multimodal Chat ===")
    try:
        # Use a public test image
        response = multimodal_chat(
            "What do you see in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Multimodal test failed (may require additional setup): {e}")


if __name__ == "__main__":
    main()
