#!/usr/bin/env python3
"""
Test multi-bucket stream compilation and benchmarking for FlashVSR DiT.

This script:
1. Compiles DiT stream model with multiple frame counts (f=8, f=4, f=2) co-resident
2. Loads all bucket NEFFs simultaneously (zero swap overhead)
3. Benchmarks each bucket size individually
4. Tests greedy chunk scheduler with a simulated long video

Usage:
    export FLASHVSR_STREAM_BUCKETS=8,4,2
    python test_multi_bucket.py --weights-dir ~/FlashVSR-v1.1 --compile-dir ~/compiled/multi_bucket

Requirements:
    - trn2.3xlarge (LNC=2, 4 logical NeuronCores)
    - Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    - FlashVSR-v1.1 weights downloaded
"""

import os
import sys
import time
import argparse
import concurrent.futures

# Patch ThreadPoolExecutor before NxDI imports
original_tpe_init = concurrent.futures.ThreadPoolExecutor.__init__


def patched_tpe_init(self, *args, **kwargs):
    kwargs["max_workers"] = 1
    original_tpe_init(self, *args, **kwargs)


concurrent.futures.ThreadPoolExecutor.__init__ = patched_tpe_init

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ.setdefault("FLASHVSR_STREAM_BUCKETS", "8,4,2")

import torch
import torch_neuronx
import numpy as np

# Add source path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASHVSR_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(FLASHVSR_ROOT))

from src.modeling_flashvsr import (
    FlashVSRApplication,
    FlashVSRInferenceConfig,
    precompute_freqs_cis_3d,
    build_rope_for_grid,
    HEAD_DIM,
    DIM,
    NUM_HEADS,
    PATCH_T,
    PATCH_H,
    PATCH_W,
    STREAM_FRAME_COUNTS,
)
from src.pipeline import neuron_dit_forward, build_greedy_chunk_schedule
from neuronx_distributed_inference.models.config import NeuronConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-bucket stream benchmark")
    parser.add_argument(
        "--weights-dir", required=True, help="Path to FlashVSR-v1.1 weights"
    )
    parser.add_argument(
        "--compile-dir",
        default=os.path.expanduser("~/compiled/multi_bucket_stream"),
        help="Directory to save/load compiled NEFFs",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument(
        "--skip-compile", action="store_true", help="Skip compilation, load existing"
    )
    return parser.parse_args()


def compile_multi_bucket(args):
    """Compile stream DiT with multiple frame count buckets."""
    print(f"\n{'=' * 60}")
    print(f"Compiling multi-bucket stream DiT")
    print(f"  Buckets: {STREAM_FRAME_COUNTS}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  TP degree: {args.tp_degree}")
    print(f"  Output dir: {args.compile_dir}")
    print(f"{'=' * 60}\n")

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        torch_dtype=torch.bfloat16,
        batch_size=1,
        save_sharded_checkpoint=True,
    )
    stream_config = FlashVSRInferenceConfig(
        neuron_config=neuron_config,
        attn_mode="stream",
        height=args.height,
        width=args.width,
    )

    stream_app = FlashVSRApplication(model_path=args.weights_dir, config=stream_config)

    t0 = time.time()
    stream_app.compile(args.compile_dir)
    compile_time = time.time() - t0
    print(f"\nCompilation complete in {compile_time:.1f}s")
    return compile_time


def load_and_benchmark(args):
    """Load multi-bucket model and benchmark each bucket."""
    print(f"\n{'=' * 60}")
    print(f"Loading multi-bucket stream DiT")
    print(f"  Buckets: {STREAM_FRAME_COUNTS}")
    print(f"  Compile dir: {args.compile_dir}")
    print(f"{'=' * 60}\n")

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        torch_dtype=torch.bfloat16,
        batch_size=1,
        save_sharded_checkpoint=True,
    )
    stream_config = FlashVSRInferenceConfig(
        neuron_config=neuron_config,
        attn_mode="stream",
        height=args.height,
        width=args.width,
    )

    stream_app = FlashVSRApplication(model_path=args.weights_dir, config=stream_config)

    t0 = time.time()
    stream_app.load(args.compile_dir)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Precompute RoPE
    base_freqs = precompute_freqs_cis_3d(HEAD_DIM)

    # Load prompt embedding
    prompt_path = os.path.join(args.weights_dir, "posi_prompt.pth")
    if os.path.exists(prompt_path):
        prompt_emb = torch.load(prompt_path, map_location="cpu")
        if prompt_emb.dim() == 2:
            prompt_emb = prompt_emb.unsqueeze(0)
        prompt_emb = prompt_emb.to(dtype=torch.bfloat16)
    else:
        # Use zeros if prompt not available
        prompt_emb = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

    lat_h = args.height // 8
    lat_w = args.width // 8

    # Benchmark each bucket size
    results = {}
    print(f"\n{'=' * 60}")
    print(f"Benchmarking individual bucket sizes")
    print(f"{'=' * 60}\n")

    for frame_count in STREAM_FRAME_COUNTS:
        print(f"\n--- Bucket f={frame_count} ---")
        # Create input at this bucket's shape
        latent_input = torch.randn(
            1, 16, frame_count, lat_h, lat_w, dtype=torch.bfloat16
        )

        tokens_per_frame = (args.height // 16) * (args.width // 16)
        seq_len = frame_count * tokens_per_frame
        lq_residual = torch.zeros(1, seq_len, DIM, dtype=torch.bfloat16)

        # Warmup
        print(f"  Warming up ({args.warmup_runs} runs)...")
        with torch.no_grad():
            for _ in range(args.warmup_runs):
                _ = neuron_dit_forward(
                    stream_app,
                    base_freqs,
                    latent_input,
                    prompt_emb,
                    args.height,
                    args.width,
                    1,
                    lq_residual,
                )

        # Timed runs
        times = []
        with torch.no_grad():
            for i in range(args.benchmark_runs):
                t0 = time.time()
                _ = neuron_dit_forward(
                    stream_app,
                    base_freqs,
                    latent_input,
                    prompt_emb,
                    args.height,
                    args.width,
                    1,
                    lq_residual,
                )
                elapsed = time.time() - t0
                times.append(elapsed)
                print(f"  Run {i + 1}: {elapsed * 1000:.1f} ms")

        avg = np.mean(times)
        std = np.std(times)
        results[frame_count] = {
            "avg_ms": avg * 1000,
            "std_ms": std * 1000,
            "times": times,
        }
        print(f"  Average: {avg * 1000:.1f} ± {std * 1000:.1f} ms")
        print(f"  Per-latent-frame: {avg * 1000 / frame_count:.1f} ms")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"MULTI-BUCKET BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Instance: trn2.3xlarge (LNC=2, TP={args.tp_degree})")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Buckets compiled co-resident: {STREAM_FRAME_COUNTS}")
    print()
    print(
        f"  {'Bucket':<10} {'Avg (ms)':<12} {'Std (ms)':<12} {'Per-frame (ms)':<15} {'Speedup vs f=2'}"
    )
    print(f"  {'-' * 60}")

    f2_time = results.get(2, {}).get("avg_ms", None)
    for fc in sorted(results.keys()):
        r = results[fc]
        per_frame = r["avg_ms"] / fc
        speedup = ""
        if f2_time and fc != 2:
            # Effective speedup: how much faster to process the same amount of latent frames
            # fc frames at avg_ms vs fc/2 calls of f=2 at f2_time each
            equivalent_f2_calls = fc / 2
            equivalent_f2_time = equivalent_f2_calls * f2_time
            speedup = f"{equivalent_f2_time / r['avg_ms']:.2f}x"
        print(
            f"  f={fc:<7} {r['avg_ms']:<12.1f} {r['std_ms']:<12.1f} {per_frame:<15.1f} {speedup}"
        )

    # Simulate long video (1-min at 30fps)
    print(f"\n{'=' * 60}")
    print(f"SIMULATED 1-MIN VIDEO (1793 frames → 448 latent frames)")
    print(f"{'=' * 60}")

    num_latent_frames = 448
    schedule = build_greedy_chunk_schedule(num_latent_frames, STREAM_FRAME_COUNTS)

    # Estimate total time
    total_estimated = 0
    chunk_counts = {}
    for fc in schedule:
        chunk_counts[fc] = chunk_counts.get(fc, 0) + 1
        if fc == 6:
            # First chunk — estimate based on f=6 (not benchmarked here, use ~1700ms)
            total_estimated += 1700
        elif fc in results:
            total_estimated += results[fc]["avg_ms"]
        else:
            # Fallback: linear interpolation
            total_estimated += fc * (results.get(2, {}).get("avg_ms", 416) / 2)

    print(f"  Greedy schedule: {len(schedule)} total chunks")
    for fc in sorted(chunk_counts.keys(), reverse=True):
        print(f"    f={fc}: {chunk_counts[fc]} chunks")
    print(f"  Estimated total DiT time: {total_estimated / 1000:.1f}s")

    # Compare with f=2 only
    schedule_f2 = build_greedy_chunk_schedule(num_latent_frames, [2])
    total_f2 = 1700 + (len(schedule_f2) - 1) * (f2_time or 416)
    print(f"  Baseline (f=2 only): {len(schedule_f2)} chunks, {total_f2 / 1000:.1f}s")
    if total_f2 > 0:
        print(f"  Speedup: {total_f2 / total_estimated:.2f}x")

    # Restore ThreadPoolExecutor
    concurrent.futures.ThreadPoolExecutor.__init__ = original_tpe_init

    return results


def main():
    args = parse_args()

    print(
        f"FLASHVSR_STREAM_BUCKETS = {os.environ.get('FLASHVSR_STREAM_BUCKETS', 'not set')}"
    )
    print(f"Stream frame counts: {STREAM_FRAME_COUNTS}")

    if not args.skip_compile:
        if not os.path.exists(args.compile_dir) or not os.listdir(args.compile_dir):
            compile_multi_bucket(args)
        else:
            print(f"Compile dir exists: {args.compile_dir} — skipping compilation")
            print(f"  (Use a different --compile-dir or delete to recompile)")

    load_and_benchmark(args)


if __name__ == "__main__":
    main()
