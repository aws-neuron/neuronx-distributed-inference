#!/usr/bin/env python3
"""
Test Qwen3-ASR-1.7B transcription via vLLM OpenAI-compatible API.

Usage:
    # Start the vLLM server first:
    bash start-vllm-server.sh

    # Then run this test:
    python3 test_transcription.py [audio_file.wav]
"""

import base64
import json
import sys
import time
import urllib.request

VLLM_URL = "http://localhost:8000"


def get_model_id():
    """Get the model ID from the running server."""
    req = urllib.request.Request(f"{VLLM_URL}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data["data"][0]["id"]


def transcribe(audio_path: str, model_id: str) -> dict:
    """Send audio to the vLLM server and get transcription."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    headers = {"Content-Type": "application/json"}
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=req_data,
        headers=headers,
        method="POST",
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    elapsed = time.time() - t0

    return {
        "text": result["choices"][0]["message"]["content"],
        "elapsed": elapsed,
        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
    }


def parse_asr_output(raw_text: str) -> str:
    """Parse Qwen3-ASR output format: 'language {lang}<asr_text>{text}'"""
    marker = "<asr_text>"
    if marker in raw_text:
        return raw_text.split(marker, 1)[1].strip()
    return raw_text.strip()


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_speech_real.wav"

    # Check server health
    try:
        req = urllib.request.Request(f"{VLLM_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print("ERROR: Server not healthy")
                return 1
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        print("Start the server with: bash start-vllm-server.sh")
        return 1

    model_id = get_model_id()
    print(f"Model: {model_id}")
    print(f"Audio: {audio_path}")
    print()

    # Run transcription
    result = transcribe(audio_path, model_id)

    raw_text = result["text"]
    clean_text = parse_asr_output(raw_text)

    print(f"Raw output: {raw_text}")
    print(f"Transcription: {clean_text}")
    print(f"Latency: {result['elapsed']:.3f}s")
    print(
        f"Tokens: prompt={result['prompt_tokens']}, completion={result['completion_tokens']}"
    )

    # Performance run (5 iterations)
    print("\nPerformance (5 runs):")
    latencies = []
    for i in range(5):
        r = transcribe(audio_path, model_id)
        latencies.append(r["elapsed"])
    avg = sum(latencies) / len(latencies)
    print(f"  Avg: {avg:.3f}s, Min: {min(latencies):.3f}s, Max: {max(latencies):.3f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
