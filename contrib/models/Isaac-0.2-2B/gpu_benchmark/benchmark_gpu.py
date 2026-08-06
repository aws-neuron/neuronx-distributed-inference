#!/usr/bin/env python3
# Copyright 2025 (c) Amazon.com and Affiliates
"""GPU benchmark for Isaac-0.2-2B-Preview using vLLM.

Measures TTFT, TPOT, tok/s across multiple workloads to match Neuron benchmark.
Follows GPU Benchmark Standard (steering/gpu-benchmark-standard.md).

Usage:
    pip install vllm transformers torch pillow
    python benchmark_gpu.py [--model PerceptronAI/Isaac-0.2-2B-Preview] [--warmup 5] [--iterations 10]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


# ── Workload definitions matching Neuron benchmark ──────────────────────

WORKLOADS = {
    "short-short": {"input_tokens": 128, "output_tokens": 128},
    "short-long": {"input_tokens": 128, "output_tokens": 512},
    "long-short": {"input_tokens": 2048, "output_tokens": 128},
    "long-long": {"input_tokens": 2048, "output_tokens": 512},
}

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a synthetic prompt of approximately target_tokens length."""
    repeated = FILLER_TEXT * (target_tokens // 10 + 10)
    token_ids = tokenizer.encode(repeated)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def percentiles(values, pcts=(50, 95, 99)):
    """Calculate percentiles."""
    if not values:
        return {f"p{p}": None for p in pcts}
    s = sorted(values)
    n = len(s)
    return {f"p{p}": s[min(int(p / 100 * n), n - 1)] for p in pcts}


def benchmark_vllm_offline(model_path, workloads, warmup, iterations, dtype):
    """Run benchmark using vLLM offline (Python API)."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {model_path}")
    print(f"dtype: {dtype}")

    llm = LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    results = {}

    for wl_name, wl_config in workloads.items():
        input_tokens = wl_config["input_tokens"]
        output_tokens = wl_config["output_tokens"]
        print(f"\n{'=' * 60}")
        print(f"Workload: {wl_name} (input={input_tokens}, output={output_tokens})")
        print(f"{'=' * 60}")

        prompt = build_prompt(tokenizer, input_tokens)
        actual_input = len(tokenizer.encode(prompt))
        print(f"  Actual input tokens: {actual_input}")

        sampling_params = SamplingParams(
            temperature=0,  # Greedy for reproducibility
            max_tokens=output_tokens,
        )

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            llm.generate([prompt], sampling_params)

        # Timed iterations
        print(f"  Benchmarking ({iterations} runs)...")
        ttfts = []
        tpots = []
        throughputs = []
        e2e_latencies = []
        output_lengths = []

        for i in range(iterations):
            t_start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            t_end = time.perf_counter()

            output = outputs[0]
            n_output_tokens = len(output.outputs[0].token_ids)
            e2e = t_end - t_start

            # Extract TTFT from metrics if available
            metrics = output.metrics
            if (
                metrics
                and hasattr(metrics, "first_token_time")
                and metrics.first_token_time
            ):
                ttft = metrics.first_token_time - metrics.arrival_time
            else:
                # Approximate: E2E - decode time
                ttft = e2e / (n_output_tokens + 1) if n_output_tokens > 0 else e2e

            # TPOT = decode time / (output tokens - 1)
            decode_time = e2e - ttft
            tpot = decode_time / max(n_output_tokens - 1, 1)
            tps = n_output_tokens / e2e if e2e > 0 else 0

            ttfts.append(ttft * 1000)  # to ms
            tpots.append(tpot * 1000)  # to ms
            throughputs.append(tps)
            e2e_latencies.append(e2e * 1000)  # to ms
            output_lengths.append(n_output_tokens)

        results[wl_name] = {
            "input_tokens": actual_input,
            "target_output_tokens": output_tokens,
            "avg_output_tokens": statistics.mean(output_lengths),
            "ttft_ms": percentiles(ttfts),
            "tpot_ms": percentiles(tpots),
            "throughput_tok_s": percentiles(throughputs),
            "e2e_latency_ms": percentiles(e2e_latencies),
            "raw_ttfts": ttfts,
            "raw_tpots": tpots,
            "raw_throughputs": throughputs,
            "raw_e2e": e2e_latencies,
        }

        print(f"  TTFT (P50): {percentiles(ttfts)['p50']:.1f} ms")
        print(f"  TPOT (P50): {percentiles(tpots)['p50']:.2f} ms")
        print(f"  Throughput (P50): {percentiles(throughputs)['p50']:.1f} tok/s")
        print(f"  E2E (P50): {percentiles(e2e_latencies)['p50']:.1f} ms")
        print(f"  Avg output tokens: {statistics.mean(output_lengths):.0f}")

    return results


def benchmark_image_text(model_path, warmup, iterations, dtype):
    """Benchmark image+text workload."""
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}")
    print("Image+Text Benchmark")
    print(f"{'=' * 60}")

    llm = LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Use a simple test prompt with image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        try:
            llm.chat(messages, sampling_params)
        except Exception as e:
            print(f"  Warmup error (may be expected): {e}")
            return None

    # Timed iterations
    print(f"  Benchmarking ({iterations} runs)...")
    e2e_latencies = []
    output_lengths = []

    for i in range(iterations):
        t_start = time.perf_counter()
        outputs = list(llm.chat(messages, sampling_params))
        t_end = time.perf_counter()

        output = outputs[0]
        n_tokens = len(output.outputs[0].token_ids)
        e2e = (t_end - t_start) * 1000

        e2e_latencies.append(e2e)
        output_lengths.append(n_tokens)

    avg_tokens = statistics.mean(output_lengths)
    avg_e2e = statistics.mean(e2e_latencies)
    avg_tps = avg_tokens / (avg_e2e / 1000) if avg_e2e > 0 else 0

    result = {
        "avg_output_tokens": avg_tokens,
        "e2e_latency_ms": percentiles(e2e_latencies),
        "throughput_tok_s": avg_tps,
        "text_preview": outputs[0].outputs[0].text[:150] if outputs else "",
    }

    print(f"  Output tokens: {avg_tokens:.0f}")
    print(f"  E2E (P50): {percentiles(e2e_latencies)['p50']:.1f} ms")
    print(f"  Throughput: {avg_tps:.1f} tok/s")

    return result


def get_gpu_info():
    """Get GPU information."""
    info = {}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = (
            getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        )
    return info


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark for Isaac-0.2-2B")
    parser.add_argument(
        "--model",
        default="PerceptronAI/Isaac-0.2-2B-Preview",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short-short", "short-long", "long-short", "long-long"],
        choices=list(WORKLOADS.keys()),
    )
    parser.add_argument(
        "--skip-image", action="store_true", help="Skip image+text benchmark"
    )
    parser.add_argument("--output", default="gpu_benchmark_results.json")
    args = parser.parse_args()

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('gpu_name', 'unknown')}")
    print(f"GPU Memory: {gpu_info.get('gpu_memory_gb', 0):.1f} GB")
    print(f"Model: {args.model}")
    print(f"dtype: {args.dtype}")
    print(f"Workloads: {args.workloads}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")

    # Select workloads
    selected = {k: WORKLOADS[k] for k in args.workloads}

    # Run text benchmarks
    text_results = benchmark_vllm_offline(
        args.model, selected, args.warmup, args.iterations, args.dtype
    )

    # Run image+text benchmark
    image_result = None
    if not args.skip_image:
        image_result = benchmark_image_text(
            args.model, args.warmup, args.iterations, args.dtype
        )

    # Compile all results
    all_results = {
        "metadata": {
            "model": args.model,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "gpu": gpu_info,
            "framework": "vLLM",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "text_benchmarks": text_results,
        "image_text_benchmark": image_result,
    }

    # Summary table
    print(f"\n{'=' * 80}")
    print("GPU BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Workload':<15} {'In':>5} {'Out':>5} {'TTFT P50':>10} {'TPOT P50':>10} "
        f"{'tok/s P50':>10} {'E2E P50':>10}"
    )
    print("-" * 70)
    for wl_name, r in text_results.items():
        print(
            f"{wl_name:<15} {r['input_tokens']:>5} {r['avg_output_tokens']:>5.0f} "
            f"{r['ttft_ms']['p50']:>10.1f} {r['tpot_ms']['p50']:>10.2f} "
            f"{r['throughput_tok_s']['p50']:>10.1f} {r['e2e_latency_ms']['p50']:>10.1f}"
        )
    if image_result:
        print(
            f"{'image+text':<15} {'N/A':>5} {image_result['avg_output_tokens']:>5.0f} "
            f"{'N/A':>10} {'N/A':>10} "
            f"{image_result['throughput_tok_s']:>10.1f} "
            f"{image_result['e2e_latency_ms']['p50']:>10.1f}"
        )

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
