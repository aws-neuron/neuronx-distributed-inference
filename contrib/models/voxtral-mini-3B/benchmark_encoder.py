#!/usr/bin/env python3
"""
Voxtral Audio Encoder Optimization Benchmark (Task 013)

Measures detailed component latencies before/after compiler flag optimization.
Run this TWICE: once with baseline compiler args, once with optimized args.

Usage:
    # Baseline (compile with minimal flags first)
    python benchmark_encoder.py --label baseline

    # Optimized (recompile with optimized flags)
    python benchmark_encoder.py --label optimized

Environment variables:
    VOXTRAL_MODEL_PATH      Path to HF model weights (default: /mnt/models/voxtral-mini-3B)
    VOXTRAL_COMPILED_PATH   Path for compiled NEFFs (default: /mnt/models/compiled_voxtral)
    VOXTRAL_TP_DEGREE       Tensor parallel degree (default: 1)
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from modeling_voxtral import NeuronApplicationVoxtral

MODEL_PATH = os.environ.get("VOXTRAL_MODEL_PATH", "/mnt/models/voxtral-mini-3B")
COMPILED_PATH = os.environ.get("VOXTRAL_COMPILED_PATH", "/mnt/models/compiled_voxtral")
TP_DEGREE = int(os.environ.get("VOXTRAL_TP_DEGREE", "1"))
SEQ_LEN = 2048
N_POSITIONS = 4096
DTYPE = torch.bfloat16

AUDIO_URL = (
    "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav"
)


def measure_encoder_latency(app, input_features, n_warmup=3, n_runs=10):
    """Measure audio encoder latency (Neuron NEFF)."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            app.audio_encoder(input_features.to(app.dtype))

    latencies = []
    for _ in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            enc_output = app.audio_encoder(input_features.to(app.dtype))
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return latencies, enc_output


def measure_projector_latency(app, audio_hidden_flat, n_warmup=3, n_runs=10):
    """Measure projector latency (CPU or Neuron)."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            app.projector(audio_hidden_flat)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            proj_output = app.projector(audio_hidden_flat)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return latencies, proj_output


def measure_full_pipeline_latency(app, input_features, n_warmup=2, n_runs=5):
    """Measure full audio pipeline: encoder -> reshape -> projector."""
    # Warmup
    for _ in range(n_warmup):
        app.run_audio_pipeline(input_features)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        app.run_audio_pipeline(input_features)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return latencies


def measure_e2e_transcription(app, audio_url, n_warmup=1, n_runs=3):
    """Measure end-to-end transcription (TTFT proxy)."""
    # Warmup
    app.transcribe(audio_url, max_new_tokens=50)

    latencies = []
    token_counts = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = app.transcribe(audio_url, max_new_tokens=100)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        # Rough token count
        token_counts.append(len(app.tokenizer.encode(result)))

    return latencies, token_counts


def measure_text_generation(
    app, prompt="What is the capital of France?", n_warmup=2, n_runs=5
):
    """Measure text-only decode throughput (tok/s)."""
    max_tokens = 100

    # Warmup
    for _ in range(n_warmup):
        app.generate(prompt, max_new_tokens=max_tokens)

    latencies = []
    token_counts = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = app.generate(prompt, max_new_tokens=max_tokens)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        token_counts.append(len(app.tokenizer.encode(result)))

    return latencies, token_counts


def stats(latencies):
    """Return min, median, mean, max of a list."""
    s = sorted(latencies)
    n = len(s)
    return {
        "min": s[0],
        "median": s[n // 2],
        "mean": sum(s) / n,
        "max": s[-1],
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Voxtral Encoder Benchmark")
    parser.add_argument(
        "--label", default="baseline", help="Label for this run (baseline/optimized)"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Only compile, don't benchmark"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Voxtral Encoder Benchmark — {args.label}")
    print("=" * 70)
    print(f"Model path:    {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_PATH}")
    print(f"TP degree:     {TP_DEGREE}")
    print(f"Seq len:       {SEQ_LEN}")
    print()

    # Build app
    app = NeuronApplicationVoxtral(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        dtype=DTYPE,
    )

    # Compile
    compiled_marker = os.path.join(
        COMPILED_PATH, "text_decoder", "text_model", "model.pt"
    )
    if not os.path.exists(compiled_marker):
        print("Compiling model...")
        t0 = time.perf_counter()
        app.compile(COMPILED_PATH)
        compile_time = time.perf_counter() - t0
        print(f"Compilation complete in {compile_time:.1f}s\n")
    else:
        print("Using existing compiled model.\n")
        compile_time = 0

    if args.compile_only:
        print("Compile-only mode. Done.")
        return

    # Load
    print("Loading compiled model...")
    t0 = time.perf_counter()
    app.load(COMPILED_PATH)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # Prepare audio input
    print("Preparing audio input...")
    input_features = torch.randn(1, 128, 3000, dtype=DTYPE)

    # --- Encoder benchmark ---
    print("\n--- Audio Encoder (Neuron NEFF) ---")
    enc_latencies, enc_output = measure_encoder_latency(app, input_features)
    enc_stats = stats(enc_latencies)
    print(
        f"  Encoder output shape: {enc_output.shape if hasattr(enc_output, 'shape') else type(enc_output)}"
    )
    print(
        f"  Latency (ms): min={enc_stats['min']:.1f}  median={enc_stats['median']:.1f}  mean={enc_stats['mean']:.1f}  max={enc_stats['max']:.1f}  (n={enc_stats['n']})"
    )

    # Extract hidden states for projector benchmark
    if isinstance(enc_output, dict):
        audio_hidden = enc_output.get("last_hidden_state", list(enc_output.values())[0])
    elif isinstance(enc_output, tuple):
        audio_hidden = enc_output[0]
    else:
        audio_hidden = enc_output
    audio_hidden_flat = audio_hidden.reshape(-1, 5120)  # [375, 5120]
    print(f"  Reshaped audio hidden: {audio_hidden_flat.shape}")

    # --- Projector benchmark ---
    print("\n--- Projector ---")
    proj_latencies, proj_output = measure_projector_latency(app, audio_hidden_flat)
    proj_stats = stats(proj_latencies)
    print(f"  Projector output shape: {proj_output.shape}")
    print(
        f"  Latency (ms): min={proj_stats['min']:.1f}  median={proj_stats['median']:.1f}  mean={proj_stats['mean']:.1f}  max={proj_stats['max']:.1f}  (n={proj_stats['n']})"
    )

    # --- Full audio pipeline ---
    print("\n--- Full Audio Pipeline (encoder + reshape + projector) ---")
    pipe_latencies = measure_full_pipeline_latency(app, input_features)
    pipe_stats = stats(pipe_latencies)
    print(
        f"  Latency (ms): min={pipe_stats['min']:.1f}  median={pipe_stats['median']:.1f}  mean={pipe_stats['mean']:.1f}  max={pipe_stats['max']:.1f}  (n={pipe_stats['n']})"
    )

    # --- E2E transcription ---
    print("\n--- E2E Transcription (with real audio) ---")
    trans_latencies, trans_tokens = measure_e2e_transcription(app, AUDIO_URL)
    trans_stats = stats(trans_latencies)
    avg_tokens = sum(trans_tokens) / len(trans_tokens)
    print(
        f"  Latency (ms): min={trans_stats['min']:.1f}  median={trans_stats['median']:.1f}  mean={trans_stats['mean']:.1f}  max={trans_stats['max']:.1f}  (n={trans_stats['n']})"
    )
    print(f"  Avg tokens generated: {avg_tokens:.0f}")

    # --- Text-only generation ---
    print("\n--- Text-only Generation ---")
    text_latencies, text_tokens = measure_text_generation(app)
    text_stats = stats(text_latencies)
    avg_text_tokens = sum(text_tokens) / len(text_tokens)
    avg_text_latency_s = text_stats["mean"] / 1000
    toks_per_s = avg_text_tokens / avg_text_latency_s if avg_text_latency_s > 0 else 0
    print(
        f"  Latency (ms): min={text_stats['min']:.1f}  median={text_stats['median']:.1f}  mean={text_stats['mean']:.1f}  max={text_stats['max']:.1f}  (n={text_stats['n']})"
    )
    print(f"  Avg tokens: {avg_text_tokens:.0f}  Throughput: {toks_per_s:.1f} tok/s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"SUMMARY — {args.label}")
    print("=" * 70)
    results = {
        "label": args.label,
        "encoder_median_ms": round(enc_stats["median"], 1),
        "projector_median_ms": round(proj_stats["median"], 1),
        "pipeline_median_ms": round(pipe_stats["median"], 1),
        "transcription_median_ms": round(trans_stats["median"], 1),
        "text_throughput_toks": round(toks_per_s, 1),
        "compile_time_s": round(compile_time, 1),
    }
    print(json.dumps(results, indent=2))

    # Save results
    results_file = f"benchmark_{args.label}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "label": args.label,
                "config": {
                    "model_path": MODEL_PATH,
                    "compiled_path": COMPILED_PATH,
                    "tp_degree": TP_DEGREE,
                    "seq_len": SEQ_LEN,
                    "dtype": str(DTYPE),
                },
                "encoder": {"stats_ms": enc_stats, "raw_ms": enc_latencies},
                "projector": {"stats_ms": proj_stats, "raw_ms": proj_latencies},
                "pipeline": {"stats_ms": pipe_stats, "raw_ms": pipe_latencies},
                "transcription": {
                    "stats_ms": trans_stats,
                    "raw_ms": trans_latencies,
                    "token_counts": trans_tokens,
                },
                "text_generation": {
                    "stats_ms": text_stats,
                    "raw_ms": text_latencies,
                    "token_counts": text_tokens,
                    "throughput_toks": toks_per_s,
                },
                "compile_time_s": compile_time,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
