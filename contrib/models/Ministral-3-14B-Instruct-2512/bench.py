#!/usr/bin/env python3
"""Benchmark vLLM server with streaming for TTFT, TPOT, output tok/s measurements.

Uses the OpenAI-compatible chat/completions API with streaming to measure:
- TTFT (time to first token)
- Output tok/s (decode throughput)
- TPOT (time per output token)
- E2E latency

Workloads: short-short(128/128), short-long(128/512), long-short(2048/128), long-long(2048/512)
Concurrency: 1, 4
"""

import argparse
import asyncio
import json
import time
import statistics
import aiohttp

CONFIG = {
    "base_url": "http://localhost:8000",
    "model": "/home/ubuntu/models/Ministral-3-14B-text-bf16",
}

WORKLOADS = {
    "short-short": (128, 128),
    "short-long": (128, 512),
    "long-short": (2048, 128),
    "long-long": (2048, 512),
}


def make_prompt(n_tokens):
    """Generate a prompt of approximately n_tokens tokens."""
    base = "Explain the following topic in great detail with examples and analysis. "
    filler = "The quick brown fox jumps over the lazy dog. "
    n_repeats = max(1, int(n_tokens / 13))
    prompt = base + filler * n_repeats
    return prompt


async def stream_request(session, model, prompt, max_tokens, request_id=0):
    """Send a streaming chat completion request and measure timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {"ignore_eos": True},
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    total_tokens = 0

    try:
        async with session.post(
            f"{CONFIG['base_url']}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:200]}"}

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = now
                        token_times.append(now)
                        total_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()

    if t_first_token is None:
        return {"error": "No tokens received"}

    ttft = (t_first_token - t_start) * 1000  # ms
    e2e = (t_end - t_start) * 1000  # ms

    # Calculate inter-token times (TPOT) for decode phase
    if len(token_times) > 1:
        inter_token = [
            (token_times[i] - token_times[i - 1]) * 1000
            for i in range(1, len(token_times))
        ]
    else:
        inter_token = [0]

    decode_time = (t_end - t_first_token) if total_tokens > 1 else 0
    output_toks = max(1, total_tokens - 1)
    tok_per_sec = output_toks / decode_time if decode_time > 0 else 0

    return {
        "request_id": request_id,
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "total_tokens": total_tokens,
        "output_tok_s": tok_per_sec,
        "tpot_ms": statistics.median(inter_token) if inter_token else 0,
        "inter_token_times": inter_token,
    }


async def run_workload(
    model, workload_name, input_tokens, output_tokens, concurrency, n_requests=5
):
    """Run a workload at given concurrency."""
    prompt = make_prompt(input_tokens)

    print(
        f"\n  Workload: {workload_name} (in={input_tokens}, out={output_tokens}), "
        f"concurrency={concurrency}, requests={n_requests}"
    )

    # Warmup
    print(f"  Warming up...")
    async with aiohttp.ClientSession() as session:
        result = await stream_request(session, model, prompt, output_tokens, 0)
        if "error" in result:
            print(f"  ERROR in warmup: {result['error']}")
            return None

    # Benchmark
    results = []
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, n_requests, concurrency):
            batch_size = min(concurrency, n_requests - batch_start)
            tasks = [
                stream_request(session, model, prompt, output_tokens, batch_start + i)
                for i in range(batch_size)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    # Filter errors
    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]

    if not good:
        print(f"  ALL REQUESTS FAILED: {errors}")
        return None

    if errors:
        print(f"  {len(errors)} errors out of {len(results)} requests")

    # Aggregate stats
    ttfts = [r["ttft_ms"] for r in good]
    e2es = [r["e2e_ms"] for r in good]
    toks = [r["output_tok_s"] for r in good]
    all_inter = []
    for r in good:
        all_inter.extend(r["inter_token_times"])

    def pcts(data):
        data = sorted(data)
        n = len(data)
        return {
            "median": statistics.median(data),
            "p95": data[int(n * 0.95)] if n > 1 else data[0],
            "p99": data[int(n * 0.99)] if n > 1 else data[0],
        }

    stats = {
        "workload": workload_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "n_requests": len(good),
        "n_errors": len(errors),
        "ttft": pcts(ttfts),
        "e2e": pcts(e2es),
        "output_tok_s": pcts(toks),
        "tpot": pcts(all_inter) if all_inter else {"median": 0, "p95": 0, "p99": 0},
        "avg_tokens": statistics.mean([r["total_tokens"] for r in good]),
    }

    print(
        f"  TTFT (ms):      median={stats['ttft']['median']:.1f}  P95={stats['ttft']['p95']:.1f}  P99={stats['ttft']['p99']:.1f}"
    )
    print(
        f"  Output tok/s:   median={stats['output_tok_s']['median']:.1f}  P95={stats['output_tok_s']['p95']:.1f}  P99={stats['output_tok_s']['p99']:.1f}"
    )
    print(
        f"  TPOT (ms):      median={stats['tpot']['median']:.1f}  P95={stats['tpot']['p95']:.1f}  P99={stats['tpot']['p99']:.1f}"
    )
    print(
        f"  E2E (ms):       median={stats['e2e']['median']:.1f}  P95={stats['e2e']['p95']:.1f}  P99={stats['e2e']['p99']:.1f}"
    )
    print(f"  Avg tokens:     {stats['avg_tokens']:.0f}")

    return stats


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CONFIG["model"])
    parser.add_argument("--base-url", default=CONFIG["base_url"])
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS.keys()))
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--requests", type=int, default=5, help="Requests per workload")
    args = parser.parse_args()

    CONFIG["base_url"] = args.base_url
    CONFIG["model"] = args.model

    print(f"Benchmarking {CONFIG['model']}")
    print(f"Server: {CONFIG['base_url']}")
    print(f"Workloads: {args.workloads}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Requests per workload: {args.requests}")

    all_results = []
    for wl_name in args.workloads:
        if wl_name not in WORKLOADS:
            print(f"Unknown workload: {wl_name}")
            continue
        in_toks, out_toks = WORKLOADS[wl_name]
        for conc in args.concurrency:
            stats = await run_workload(
                CONFIG["model"], wl_name, in_toks, out_toks, conc, args.requests
            )
            if stats:
                all_results.append(stats)

    # Print summary table
    print("\n" + "=" * 100)
    print(
        f"{'Workload':<15} {'Conc':>4} {'TTFT-P50':>10} {'TTFT-P95':>10} {'tok/s-P50':>10} {'TPOT-P50':>10} {'E2E-P50':>10}"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['workload']:<15} {r['concurrency']:>4} "
            f"{r['ttft']['median']:>9.1f} {r['ttft']['p95']:>9.1f} "
            f"{r['output_tok_s']['median']:>9.1f} "
            f"{r['tpot']['median']:>9.1f} "
            f"{r['e2e']['median']:>9.1f}"
        )
    print("=" * 100)

    # Save raw results
    with open("/home/ubuntu/bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to /home/ubuntu/bench_results.json")


if __name__ == "__main__":
    asyncio.run(main())
