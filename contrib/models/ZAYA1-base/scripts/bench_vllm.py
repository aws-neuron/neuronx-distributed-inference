#!/usr/bin/env python3
"""Benchmark vLLM serving throughput at various concurrency levels."""

import json
import time
import urllib.request
import urllib.error
import concurrent.futures

BASE_URL = "http://localhost:8000"
SEP = "=" * 60


def get_model_name():
    url = f"{BASE_URL}/v1/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode())
        return result["data"][0]["id"]


MODEL_NAME = get_model_name()
print(f"Model: {MODEL_NAME}")

PROMPTS = [
    "The capital of France is",
    "The largest ocean on Earth is the",
    "Albert Einstein was born in",
    "The speed of light in meters per second is approximately",
    "Water boils at a temperature of",
    "The chemical formula for water is",
    "The largest planet in our solar system is",
    "The Great Wall of China was built during the",
]


def send_request(prompt, max_tokens=30):
    url = f"{BASE_URL}/v1/completions"
    data = json.dumps(
        {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            elapsed = time.time() - start
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return {"tokens": tokens, "time": elapsed, "error": None}
    except Exception as e:
        return {"tokens": 0, "time": time.time() - start, "error": str(e)}


def benchmark(concurrency, num_requests=8):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    print(f"\n--- Concurrency={concurrency}, Requests={num_requests} ---")

    wall_start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, p) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    wall_time = time.time() - wall_start

    errors = sum(1 for r in results if r["error"])
    total_tokens = sum(r["tokens"] for r in results)
    latencies = [r["time"] for r in results if not r["error"]]

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
    else:
        avg_lat = min_lat = max_lat = p50 = 0

    reqs_per_sec = (num_requests - errors) / wall_time if wall_time > 0 else 0
    tok_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    print(f"  Wall time:  {wall_time:.2f}s")
    print(f"  Errors:     {errors}/{num_requests}")
    print(f"  Total toks: {total_tokens}")
    print(f"  Throughput: {reqs_per_sec:.2f} req/s, {tok_per_sec:.1f} tok/s")
    print(
        f"  Latency:    avg={avg_lat:.2f}s, p50={p50:.2f}s, min={min_lat:.2f}s, max={max_lat:.2f}s"
    )

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time": wall_time,
        "errors": errors,
        "total_tokens": total_tokens,
        "reqs_per_sec": reqs_per_sec,
        "tok_per_sec": tok_per_sec,
        "avg_latency": avg_lat,
        "p50_latency": p50,
        "min_latency": min_lat,
        "max_latency": max_lat,
    }


print(SEP)
print("ZAYA1-base vLLM Throughput Benchmark")
print(SEP)

# Warmup
print("\nWarmup (1 request)...")
send_request("Hello world", 10)

all_results = []
for conc in [1, 2, 4]:
    r = benchmark(conc, num_requests=8)
    all_results.append(r)

print(f"\n{SEP}")
print("Summary:")
header = f"{'Concurrency':<12} {'Req/s':<8} {'Tok/s':<10} {'Avg Lat':<10} {'Errors':<8}"
print(header)
for r in all_results:
    line = f"{r['concurrency']:<12} {r['reqs_per_sec']:<8.2f} {r['tok_per_sec']:<10.1f} {r['avg_latency']:<10.2f}s {r['errors']:<8}"
    print(line)
print(SEP)
