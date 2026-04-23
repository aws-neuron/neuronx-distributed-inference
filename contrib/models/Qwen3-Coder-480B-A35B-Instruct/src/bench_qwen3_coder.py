#!/usr/bin/env python3
"""
Benchmark script for Qwen3-Coder-480B-A35B-Instruct on vLLM/Neuron.

Measures TTFT, decode throughput, concurrent throughput, and generation quality.
Requires a running vLLM server (see qwen3_coder_vllm.sh).

Usage:
    python bench_qwen3_coder.py                    # Full benchmark
    python bench_qwen3_coder.py --quick             # Quick smoke test
    python bench_qwen3_coder.py --max-concurrency 4 # Limit concurrency
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

import requests

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_MODEL = "/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/"


def completions_request(api_url, model, prompt, max_tokens, temperature=0.6):
    """Send a non-streaming completions request."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    start = time.time()
    resp = requests.post(f"{api_url}/v1/completions", json=payload, timeout=600)
    elapsed = time.time() - start

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    data = resp.json()
    usage = data.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_sec": usage.get("completion_tokens", 0) / elapsed
        if elapsed > 0
        else 0,
        "text": data["choices"][0]["text"] if data.get("choices") else "",
    }


def streaming_request(api_url, model, prompt, max_tokens, temperature=0.6):
    """Send a streaming request to measure TTFT."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
    }
    start = time.time()
    ttft = None
    token_count = 0
    try:
        resp = requests.post(
            f"{api_url}/v1/completions", json=payload, timeout=600, stream=True
        )
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    if ttft is None:
                        ttft = time.time() - start
                    try:
                        chunk = json.loads(decoded[6:])
                        if chunk.get("choices", [{}])[0].get("text"):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return {"error": str(e)}

    elapsed = time.time() - start
    return {
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "token_count": token_count,
        "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
        "decode_tps": token_count / (elapsed - ttft)
        if (ttft and elapsed > ttft)
        else 0,
    }


def chat_request(api_url, model, prompt, max_tokens, temperature=0.6):
    """Send a chat completions request (for concurrent benchmark)."""
    data = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
        }
    ).encode()

    start = time.time()
    try:
        req = urllib.request.Request(
            f"{api_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.time() - start
        usage = result["usage"]
        content = result["choices"][0]["message"]["content"]
        return {
            "elapsed": elapsed,
            "completion_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "tok_s": usage["completion_tokens"] / elapsed if elapsed > 0 else 0,
            "ok": len(content.strip()) > 10,
        }
    except Exception as e:
        return {"elapsed": time.time() - start, "error": str(e)}


CONCURRENT_PROMPTS = [
    "Write a Python function to compute the first N Fibonacci numbers.",
    "Explain what tensor parallelism is in 3 sentences.",
    "Write a hello world program in Rust.",
    "What is 127 * 389? Show your work.",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Write a Python function for binary search on a sorted list.",
    "Explain the difference between TCP and UDP in simple terms.",
    "Write a Python class that implements a stack with push, pop, and peek.",
]


def benchmark_concurrency(api_url, model, n_concurrent, max_tokens=256):
    """Send n_concurrent requests simultaneously."""
    results = [None] * n_concurrent
    threads = []

    wall_start = time.time()
    for i in range(n_concurrent):
        prompt = CONCURRENT_PROMPTS[i % len(CONCURRENT_PROMPTS)]
        t = threading.Thread(
            target=lambda idx, p: results.__setitem__(
                idx, chat_request(api_url, model, p, max_tokens)
            ),
            args=(i, prompt),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=600)
    wall_elapsed = time.time() - wall_start

    valid = [r for r in results if r and "error" not in r]
    errors = [r for r in results if r and "error" in r]

    total_tokens = sum(r["completion_tokens"] for r in valid)
    aggregate_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0
    avg_per_req = sum(r["tok_s"] for r in valid) / len(valid) if valid else 0
    avg_latency = sum(r["elapsed"] for r in valid) / len(valid) if valid else 0
    all_ok = all(r.get("ok", False) for r in valid)

    return {
        "concurrency": n_concurrent,
        "wall_time": wall_elapsed,
        "total_tokens": total_tokens,
        "aggregate_tps": aggregate_tps,
        "avg_per_req_tps": avg_per_req,
        "avg_latency": avg_latency,
        "success": len(valid),
        "errors": len(errors),
        "quality_ok": all_ok,
    }


def run_benchmark(api_url, model, quick=False, max_concurrency=8):
    print("=" * 70)
    print("Qwen3-Coder-480B-A35B-Instruct Benchmark")
    print(f"Server: {api_url}")
    print("=" * 70)

    # Warmup
    print("\n--- Warmup ---")
    w = completions_request(api_url, model, "Hello", 16, temperature=0.1)
    if "error" in w:
        print(f"FAILED: {w['error']}")
        sys.exit(1)
    print(f"Warmup: {w['elapsed_s']:.2f}s, {w['completion_tokens']} tokens")

    # TTFT (streaming)
    print("\n--- TTFT Measurement (streaming) ---")
    ttft_tests = [
        ("Hello world", 32),
        ("Write a Python function to sort a list", 64),
    ]
    if not quick:
        ttft_tests.append(
            (
                "Explain the architecture of a transformer model in detail, covering "
                "attention mechanisms, feed-forward layers, and positional encoding",
                64,
            )
        )

    for prompt, max_tok in ttft_tests:
        r = streaming_request(api_url, model, prompt, max_tok, temperature=0.1)
        if "error" not in r:
            print(
                f"  ~{len(prompt.split())} words | TTFT: {r['ttft_s']:.3f}s | "
                f"Decode: {r['decode_tps']:.1f} tok/s | Tokens: {r['token_count']}"
            )
        else:
            print(f"  ERROR: {r['error']}")

    # Single-request throughput
    print("\n--- Single-Request Throughput ---")
    test_cases = [
        ("Write a Python hello world program", 128),
        ("Write a binary search function in Python with type hints", 256),
    ]
    if not quick:
        test_cases.append(
            (
                "Write a Python function that implements quicksort. Include docstring and type hints.",
                512,
            )
        )

    header = (
        f"{'PromptTok':<11} {'MaxTok':<8} {'OutTok':<8} {'Time(s)':<10} {'Tok/s':<8}"
    )
    print(header)
    print("-" * len(header))

    for prompt, max_tok in test_cases:
        r = completions_request(api_url, model, prompt, max_tok, temperature=0.1)
        if "error" not in r:
            print(
                f"{r['prompt_tokens']:<11} {max_tok:<8} {r['completion_tokens']:<8} "
                f"{r['elapsed_s']:<10.2f} {r['tokens_per_sec']:<8.2f}"
            )

    # Concurrent throughput
    print("\n--- Concurrent Throughput ---")
    header = f"{'Conc':<6} {'Wall(s)':<10} {'Tokens':<10} {'AggTPS':<10} {'AvgReqTPS':<12} {'Quality'}"
    print(header)
    print("-" * len(header))

    for c in [1, 2, 4, 8]:
        if c > max_concurrency:
            break
        r = benchmark_concurrency(api_url, model, c)
        print(
            f"{r['concurrency']:<6} {r['wall_time']:<10.2f} {r['total_tokens']:<10} "
            f"{r['aggregate_tps']:<10.2f} {r['avg_per_req_tps']:<12.2f} "
            f"{'PASS' if r['quality_ok'] else 'FAIL'}"
        )

    # Quality check
    print("\n--- Generation Quality ---")
    quality_prompts = [
        (
            "Write a Python function to compute the nth Fibonacci number using memoization:\n```python\n",
            256,
        ),
        ("What is 2 + 2? Answer with just the number:", 8),
    ]
    if not quick:
        quality_prompts.append(("Translate to French: 'The cat sat on the mat.'", 32))

    for prompt, max_tok in quality_prompts:
        r = completions_request(api_url, model, prompt, max_tok, temperature=0.0)
        if "error" not in r:
            text = r["text"][:200].replace("\n", " ")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Output ({r['completion_tokens']} tok): {text}")
        else:
            print(f"  ERROR: {r['error']}")

    print("\n" + "=" * 70)
    print("Benchmark complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-Coder-480B on vLLM/Neuron"
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="vLLM server URL")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model path (as registered in vLLM)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick smoke test (fewer prompts)"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=8, help="Max concurrency level"
    )
    args = parser.parse_args()

    run_benchmark(
        args.api_url, args.model, quick=args.quick, max_concurrency=args.max_concurrency
    )
