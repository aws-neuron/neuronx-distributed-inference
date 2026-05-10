#!/usr/bin/env python3
"""
ZAYA1-base GPU Benchmark -- Head-to-head comparison with Neuron trn2.3xlarge

Benchmarks ZAYA1-base (Zyphra/ZAYA1-base, 8.84B MoE, 17.7 GB BF16) on a single
NVIDIA GPU using HuggingFace transformers.generate() with:
  - Eager mode (baseline)
  - torch.compile (reduce-overhead + max-autotune)
  - Batch size sweep (1, 2, 4)
  - BF16 precision (matching Neuron config)

Neuron baseline (trn2.3xlarge, TP=2, NxDI contrib):
  - batch=1: 22.3 tok/s (44.8ms/token)
  - batch=4: 86.3 tok/s (11.59ms/token aggregate)
  - TTFT: 75.6ms
  - vLLM concurrency=4: 72.3 tok/s

Requires Zyphra's custom transformers fork:
  pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"

Usage:
  python benchmark_gpu.py                          # Full sweep (eager + compile + batch)
  python benchmark_gpu.py --mode eager              # Eager only
  python benchmark_gpu.py --mode compile            # torch.compile only
  python benchmark_gpu.py --batch-sizes 1           # Single batch size
  python benchmark_gpu.py --max-new-tokens 100      # More tokens per run
  python benchmark_gpu.py --model-path /data/models/ZAYA1-base  # Local model path
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "Zyphra/ZAYA1-base"
MAX_NEW_TOKENS = 50
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20

PROMPTS = [
    "The capital of France is",
    "The largest ocean on Earth is the",
    "Albert Einstein was born in",
    "The speed of light in meters per second is approximately",
    "In Python, you can read a file with",
    "The Fibonacci sequence begins with",
]

# Neuron reference numbers (trn2.3xlarge, TP=2, SDK 2.28, NxDI contrib)
NEURON_REFERENCE = {
    "instance": "trn2.3xlarge",
    "tp": 2,
    "batch_1": {"tok_s": 22.3, "ms_per_tok": 44.8, "ttft_ms": 75.6},
    "batch_4": {"tok_s": 86.3, "ms_per_tok": 11.59},
    "vllm_conc4": {"tok_s": 72.3},
}


def parse_args():
    p = argparse.ArgumentParser(description="ZAYA1-base GPU Benchmark")
    p.add_argument(
        "--model-path",
        default=None,
        help="Local model path (downloads from HF if not set)",
    )
    p.add_argument(
        "--mode",
        choices=["eager", "compile", "all"],
        default="all",
        help="Benchmark mode",
    )
    p.add_argument(
        "--compile-modes",
        nargs="+",
        default=["reduce-overhead", "max-autotune"],
        help="torch.compile modes to test",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to sweep",
    )
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    p.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    p.add_argument("--output", type=str, default="benchmark_gpu_results.json")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument(
        "--attn-impl",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention implementation (default: sdpa for best GPU perf)",
    )
    return p.parse_args()


def get_gpu_info():
    """Get GPU name, memory, and instance type."""
    import torch

    info = {"gpu_name": "none", "gpu_memory_gb": 0, "instance_type": "unknown"}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        info["gpu_count"] = torch.cuda.device_count()
    # Instance type from EC2 metadata
    try:
        r = subprocess.run(
            [
                "curl",
                "-s",
                "--connect-timeout",
                "1",
                "http://169.254.169.254/latest/meta-data/instance-type",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout:
            info["instance_type"] = r.stdout.strip()
    except Exception:
        pass
    return info


def measure_generation(
    model, input_ids, attention_mask, max_new_tokens, use_compiled=None
):
    """Measure a single generation call. Returns (latencies_dict, output_ids).

    Measures:
      - ttft: time to first token (generate 1 token)
      - total: end-to-end time for max_new_tokens
      - tkg_per_token: (total - ttft) / (new_tokens - 1)
    """
    import torch

    gen_model = use_compiled if use_compiled is not None else model
    input_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    # TTFT: generate exactly 1 new token
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_1 = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
        )
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    # Full generation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_full = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    new_tokens = out_full.shape[1] - input_len
    tkg_ms = ((total - ttft) / max(new_tokens - 1, 1)) * 1000
    per_sample_tps = (1000.0 / tkg_ms) if tkg_ms > 0 else 0
    batch_tps = per_sample_tps * batch_size

    return {
        "ttft_ms": ttft * 1000,
        "total_ms": total * 1000,
        "new_tokens": int(new_tokens),
        "tkg_ms_per_tok": tkg_ms,
        "per_sample_tok_s": per_sample_tps,
        "batch_tok_s": batch_tps,
        "batch_size": batch_size,
    }, out_full


def run_sweep(
    model, tokenizer, args, gpu_info, mode_label="eager", compiled_model=None
):
    """Run batch-size sweep for a given mode (eager or compiled)."""
    import torch

    results = {}
    for bs in args.batch_sizes:
        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"  {mode_label} | batch_size={bs} | max_new_tokens={args.max_new_tokens}"
        )
        logger.info(f"{'=' * 70}")

        all_runs = []
        for prompt in PROMPTS:
            prompt_batch = [prompt] * bs
            inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(
                model.device
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Warmup
            for _ in range(args.warmup):
                measure_generation(
                    model,
                    input_ids,
                    attention_mask,
                    args.max_new_tokens,
                    use_compiled=compiled_model,
                )

            # Benchmark runs
            prompt_runs = []
            for _ in range(args.runs):
                m, out = measure_generation(
                    model,
                    input_ids,
                    attention_mask,
                    args.max_new_tokens,
                    use_compiled=compiled_model,
                )
                prompt_runs.append(m)

            # Aggregate this prompt
            avg_ttft = sum(r["ttft_ms"] for r in prompt_runs) / len(prompt_runs)
            avg_tkg = sum(r["tkg_ms_per_tok"] for r in prompt_runs) / len(prompt_runs)
            avg_ps_tps = sum(r["per_sample_tok_s"] for r in prompt_runs) / len(
                prompt_runs
            )
            avg_batch_tps = sum(r["batch_tok_s"] for r in prompt_runs) / len(
                prompt_runs
            )

            # Show output for first run
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            logger.info(
                f"  '{prompt[:45]:45s}' | TTFT={avg_ttft:.1f}ms TKG={avg_tkg:.1f}ms/tok "
                f"per_sample={avg_ps_tps:.1f} batch={avg_batch_tps:.1f} tok/s"
            )
            logger.info(f"    -> '{decoded[:100]}'")

            all_runs.extend(prompt_runs)

        # Aggregate all prompts for this batch size
        ttfts = [r["ttft_ms"] for r in all_runs]
        tkgs = [r["tkg_ms_per_tok"] for r in all_runs]
        ps_tps = [r["per_sample_tok_s"] for r in all_runs]
        batch_tps = [r["batch_tok_s"] for r in all_runs]

        ttfts.sort()
        tkgs.sort()
        ps_tps.sort()
        batch_tps.sort()
        n = len(all_runs)

        summary = {
            "batch_size": bs,
            "mode": mode_label,
            "runs": n,
            "ttft_avg_ms": sum(ttfts) / n,
            "ttft_p50_ms": ttfts[n // 2],
            "ttft_p99_ms": ttfts[int(n * 0.99)],
            "tkg_avg_ms": sum(tkgs) / n,
            "tkg_p50_ms": tkgs[n // 2],
            "tkg_p99_ms": tkgs[int(n * 0.99)],
            "per_sample_tok_s_avg": sum(ps_tps) / n,
            "per_sample_tok_s_p50": ps_tps[n // 2],
            "batch_tok_s_avg": sum(batch_tps) / n,
            "batch_tok_s_p50": batch_tps[n // 2],
            "gpu_mem_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

        # Neuron comparison
        neuron_key = f"batch_{bs}"
        if neuron_key in NEURON_REFERENCE:
            nr = NEURON_REFERENCE[neuron_key]
            summary["neuron_batch_tok_s"] = nr["tok_s"]
            summary["gpu_vs_neuron_ratio"] = round(
                summary["batch_tok_s_avg"] / nr["tok_s"], 2
            )

        results[f"bs{bs}"] = summary

        logger.info(f"\n  SUMMARY ({mode_label}, BS={bs}):")
        logger.info(
            f"    TTFT:       {summary['ttft_avg_ms']:.1f} ms (avg), {summary['ttft_p50_ms']:.1f} ms (p50)"
        )
        logger.info(
            f"    TKG:        {summary['tkg_avg_ms']:.1f} ms/tok (avg), {summary['tkg_p50_ms']:.1f} ms (p50)"
        )
        logger.info(f"    Per-sample: {summary['per_sample_tok_s_avg']:.1f} tok/s")
        logger.info(f"    Batch:      {summary['batch_tok_s_avg']:.1f} tok/s")
        logger.info(
            f"    GPU mem:    {summary['gpu_mem_allocated_gb']:.1f} GB alloc, {summary['gpu_mem_peak_gb']:.1f} GB peak"
        )
        if "gpu_vs_neuron_ratio" in summary:
            logger.info(
                f"    vs Neuron:  {summary['gpu_vs_neuron_ratio']:.2f}x (GPU/Neuron)"
            )

    return results


def main():
    args = parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    gpu_info = get_gpu_info()
    logger.info(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB)")
    logger.info(f"Instance: {gpu_info['instance_type']}")
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Enable TF32 for tensor cores (better perf on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_path = args.model_path or MODEL_ID
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    logger.info(f"Loading model: {model_path} ({args.dtype}, attn={args.attn_impl})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).eval()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    logger.info(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    logger.info(f"Attention implementation: {args.attn_impl}")

    # Verify model is on GPU
    first_param = next(model.parameters())
    logger.info(f"Model device: {first_param.device} (dtype={first_param.dtype})")
    assert first_param.is_cuda, "MODEL IS NOT ON GPU!"

    # Quick sanity check
    inp = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False)
    logger.info(f"Sanity: '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

    all_results = {
        "model": "Zyphra/ZAYA1-base",
        "model_params": "8.84B total (800M active MoE)",
        "dtype": args.dtype,
        "attn_implementation": args.attn_impl,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "load_time_s": round(load_time, 1),
        "gpu_info": gpu_info,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "neuron_reference": NEURON_REFERENCE,
        "benchmarks": {},
    }

    # --- Eager Mode ---
    if args.mode in ("eager", "all"):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"  EAGER MODE")
        logger.info(f"{'#' * 70}")
        eager_results = run_sweep(model, tokenizer, args, gpu_info, mode_label="eager")
        all_results["benchmarks"]["eager"] = eager_results

    # --- torch.compile Modes ---
    if args.mode in ("compile", "all"):
        for compile_mode in args.compile_modes:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  torch.compile (mode={compile_mode})")
            logger.info(f"{'#' * 70}")

            try:
                import torch._dynamo

                torch._dynamo.reset()

                compiled_model = torch.compile(model, mode=compile_mode)

                # Extra warmup for JIT compilation (first calls trigger tracing)
                logger.info(f"  Compile warmup (JIT tracing)...")
                inp = tokenizer("warmup", return_tensors="pt").to(model.device)
                for i in range(10):
                    with torch.no_grad():
                        compiled_model.generate(
                            **inp, max_new_tokens=5, do_sample=False
                        )
                    if i == 0:
                        logger.info(f"    First compile call done")
                logger.info(f"  Compile warmup complete")

                compile_results = run_sweep(
                    model,
                    tokenizer,
                    args,
                    gpu_info,
                    mode_label=f"compile_{compile_mode}",
                    compiled_model=compiled_model,
                )
                all_results["benchmarks"][f"compile_{compile_mode}"] = compile_results

            except Exception as e:
                logger.error(f"  torch.compile ({compile_mode}) FAILED: {e}")
                all_results["benchmarks"][f"compile_{compile_mode}"] = {"error": str(e)}

            # Reset for next compile mode
            try:
                import torch._dynamo

                torch._dynamo.reset()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    # --- Final Summary Table ---
    logger.info(f"\n{'=' * 90}")
    logger.info(
        f"FINAL SUMMARY: ZAYA1-base on {gpu_info['gpu_name']} ({gpu_info['instance_type']})"
    )
    logger.info(f"{'=' * 90}")
    logger.info(
        f"  Model load: {load_time:.1f}s | dtype: {args.dtype} | max_new_tokens: {args.max_new_tokens}"
    )
    logger.info("")
    logger.info(
        f"  {'Mode':<28} {'BS':<4} {'TTFT(ms)':<10} {'TKG(ms)':<10} {'Sample':<10} {'Batch':<10} {'vs Neuron':<10}"
    )
    logger.info(
        f"  {'-' * 28} {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}"
    )

    for mode_key, mode_results in all_results["benchmarks"].items():
        if isinstance(mode_results, dict) and "error" in mode_results:
            logger.info(f"  {mode_key:<28} FAILED: {mode_results['error'][:50]}")
            continue
        for bs_key, summary in sorted(mode_results.items()):
            vs = summary.get("gpu_vs_neuron_ratio", "N/A")
            vs_str = f"{vs:.2f}x" if isinstance(vs, (int, float)) else vs
            logger.info(
                f"  {mode_key:<28} {summary['batch_size']:<4} "
                f"{summary['ttft_avg_ms']:<10.1f} {summary['tkg_avg_ms']:<10.1f} "
                f"{summary['per_sample_tok_s_avg']:<10.1f} {summary['batch_tok_s_avg']:<10.1f} "
                f"{vs_str:<10}"
            )

    logger.info("")
    logger.info(f"  Neuron reference (trn2.3xlarge, TP=2):")
    logger.info(
        f"    batch=1: {NEURON_REFERENCE['batch_1']['tok_s']} tok/s, TTFT={NEURON_REFERENCE['batch_1']['ttft_ms']}ms"
    )
    logger.info(f"    batch=4: {NEURON_REFERENCE['batch_4']['tok_s']} tok/s")
    logger.info(f"    vLLM c=4: {NEURON_REFERENCE['vllm_conc4']['tok_s']} tok/s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
