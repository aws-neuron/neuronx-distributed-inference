#!/usr/bin/env python3
"""GPU benchmark for InternVL3-8B-Instruct.

Measures TTFT, output tok/s, E2E latency across batch sizes and optimizations.
Follows GPU Benchmark Standard from OpencodeDocs/steering/gpu-benchmark-standard.md.
"""

import argparse
import json
import time
import gc
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

# Monkey-patch for transformers 5.7.0 compatibility with InternVL3 custom model.
# InternVLChatModel.__init__ doesn't call super().__init__ in a way that sets
# the new all_tied_weights_keys attribute required by transformers >= 5.7.0.
_orig_ptm_init = PreTrainedModel.__init__


def _patched_ptm_init(self, *args, **kwargs):
    _orig_ptm_init(self, *args, **kwargs)
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}


PreTrainedModel.__init__ = _patched_ptm_init


def get_gpu_memory_mb():
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_peak_mb():
    """Return peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def percentiles(data, pcts=[50, 95, 99]):
    """Compute percentiles from list of values."""
    arr = np.array(data)
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def build_prompt(tokenizer, target_tokens):
    """Build a prompt of approximately target_tokens length."""
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "we find ourselves at the intersection of technology and creativity. "
        "The possibilities are endless when we consider the potential applications. "
    )
    # Repeat to exceed target, then truncate
    repeated = base_text * (target_tokens // 10 + 10)
    ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


def benchmark_generate(model, tokenizer, prompt, max_new_tokens, device, batch_size=1):
    """Run a single generation and measure E2E time.

    Uses model.language_model.generate() to bypass the VLM wrapper's
    img_context_token_id assertion, since we're benchmarking text-only
    generation (matching the Neuron TKG benchmark methodology).

    Returns dict with e2e_s, num_output_tokens, num_input_tokens, batch_size.
    """
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(
        device
    )
    input_len = inputs["input_ids"].shape[1]

    # Use language_model directly to avoid VLM wrapper assertion
    gen_model = getattr(model, "language_model", model)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    num_output_tokens = outputs.shape[1] - input_len

    return {
        "e2e_s": total_time,
        "num_output_tokens": int(num_output_tokens),
        "num_input_tokens": int(input_len),
        "batch_size": batch_size,
    }


def measure_ttft(model, tokenizer, prompt, device, batch_size=1):
    """Measure TTFT by generating exactly 1 token.

    Uses model.language_model.generate() for text-only benchmarking.
    """
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(
        device
    )

    # Use language_model directly to avoid VLM wrapper assertion
    gen_model = getattr(model, "language_model", model)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        gen_model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    return (t_end - t_start) * 1000  # ms


def run_workload(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device,
    batch_size,
    num_warmup=5,
    num_runs=10,
):
    """Run a complete workload with warmup and measurement."""
    print(f"    Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        benchmark_generate(model, tokenizer, prompt, max_new_tokens, device, batch_size)

    # Measure TTFT separately
    print(f"    Measuring TTFT ({num_runs} runs)...")
    ttft_values = []
    for _ in range(num_runs):
        ttft = measure_ttft(model, tokenizer, prompt, device, batch_size)
        ttft_values.append(ttft)

    # Measure full generation
    print(f"    Measuring generation ({num_runs} runs)...")
    results = []
    for _ in range(num_runs):
        r = benchmark_generate(
            model, tokenizer, prompt, max_new_tokens, device, batch_size
        )
        results.append(r)

    # Compute derived metrics
    ttft_stats = percentiles(ttft_values)

    e2e_values = [r["e2e_s"] * 1000 for r in results]  # ms
    e2e_stats = percentiles(e2e_values)

    # output tok/s = (num_output_tokens - 1) / (e2e - ttft)
    # Using median TTFT for TPOT calculation
    median_ttft_s = ttft_stats["p50"] / 1000
    output_tps_values = []
    for r in results:
        decode_time = r["e2e_s"] - median_ttft_s
        if decode_time > 0 and r["num_output_tokens"] > 1:
            tps = (r["num_output_tokens"] - 1) / decode_time
            output_tps_values.append(tps)
    output_tps_stats = (
        percentiles(output_tps_values)
        if output_tps_values
        else {"p50": 0, "p95": 0, "p99": 0}
    )

    # Total throughput (batch * per-request tok/s)
    total_tps = output_tps_stats["p50"] * batch_size

    peak_mem = get_gpu_memory_peak_mb()

    return {
        "ttft_ms": ttft_stats,
        "output_tps": output_tps_stats,
        "total_tps": total_tps,
        "e2e_ms": e2e_stats,
        "num_output_tokens": int(np.median([r["num_output_tokens"] for r in results])),
        "num_input_tokens": results[0]["num_input_tokens"],
        "batch_size": batch_size,
        "peak_gpu_mem_mb": peak_mem,
        "ttft_raw": ttft_values,
        "e2e_raw": e2e_values,
        "output_tps_raw": output_tps_values,
    }


WORKLOADS = {
    "short-short": (128, 128),
    "short-long": (128, 512),
    "long-short": (2048, 128),
    "long-long": (2048, 512),
}


def main():
    parser = argparse.ArgumentParser(
        description="GPU benchmark for InternVL3-8B-Instruct"
    )
    parser.add_argument(
        "--model",
        default="/home/ubuntu/models/InternVL3-8B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short-short", "short-long", "long-short", "long-long"],
        help="Workload IDs to test",
    )
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=10, help="Measurement runs")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--flash-attn", action="store_true", help="Use flash_attention_2"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Load Qwen2ForCausalLM directly with SDPA (bypasses VLM wrapper eager attention)",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--output", default="gpu_benchmark_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"=== GPU Benchmark: InternVL3-8B-Instruct ===")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print(f"dtype: {args.dtype}")
    print(f"torch.compile: {args.compile}")
    print(f"flash_attention_2: {args.flash_attn}")
    print(f"LLM-only (SDPA): {args.llm_only}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Workloads: {args.workloads}")
    print()

    # Load model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    t_load_start = time.perf_counter()
    attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"

    if args.llm_only:
        # Load Qwen2ForCausalLM directly with specified attn_implementation.
        # This avoids the InternVLChatModel wrapper which forces eager attention
        # and doesn't support attn_implementation parameter.
        from transformers import Qwen2ForCausalLM, AutoConfig
        from safetensors.torch import load_file
        import glob

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        llm_config = config.llm_config
        llm_config._attn_implementation = attn_impl

        # Create model on meta device (zero memory), load weights to GPU,
        # then materialize buffers (rotary_emb.inv_freq) on GPU.
        with torch.device("meta"):
            model = Qwen2ForCausalLM(llm_config)

        shard_files = sorted(glob.glob(os.path.join(args.model, "model*.safetensors")))
        prefix = "language_model."
        for sf in shard_files:
            shard = load_file(sf, device="cuda:0")
            for k, v in shard.items():
                if k.startswith(prefix):
                    param_name = k[len(prefix) :]
                    parts = param_name.split(".")
                    obj = model
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    setattr(
                        obj,
                        parts[-1],
                        torch.nn.Parameter(v.to(dtype), requires_grad=False),
                    )
            del shard
            gc.collect()
            torch.cuda.empty_cache()

        # Materialize rotary embedding buffers (computed from config, not weights)
        rope = model.model.rotary_emb
        dim = llm_config.hidden_size // llm_config.num_attention_heads
        base = getattr(llm_config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim)
        )
        rope.inv_freq = inv_freq
        if hasattr(rope, "original_inv_freq"):
            rope.original_inv_freq = inv_freq.clone()
    else:
        # Load full VLM (uses eager attention by default)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s")
    print(f"GPU memory after load: {get_gpu_memory_mb():.0f} MB")

    if args.compile:
        print("Compiling model with torch.compile...")
        t_compile_start = time.perf_counter()
        model = torch.compile(model, mode="reduce-overhead")
        # Trigger compilation with a small forward pass
        dummy = tokenizer("Hello", return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**dummy, max_new_tokens=2, do_sample=False)
        t_compile = time.perf_counter() - t_compile_start
        print(f"Compilation done in {t_compile:.1f}s")

    # Build prompts for each workload
    prompts = {}
    for wl_id in args.workloads:
        input_tokens, _ = WORKLOADS[wl_id]
        prompts[wl_id] = build_prompt(tokenizer, input_tokens)

    # Run benchmarks
    all_results = {
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            if torch.cuda.is_available()
            else 0,
            "dtype": args.dtype,
            "torch_compile": args.compile,
            "flash_attention_2": args.flash_attn,
            "llm_only": args.llm_only,
            "attn_implementation": attn_impl,
            "load_time_s": t_load,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        },
        "results": {},
    }

    for wl_id in args.workloads:
        input_tokens, output_tokens = WORKLOADS[wl_id]
        print(f"\n--- Workload: {wl_id} (in={input_tokens}, out={output_tokens}) ---")

        for bs in args.batch_sizes:
            print(f"  Batch size: {bs}")
            torch.cuda.reset_peak_memory_stats()

            try:
                result = run_workload(
                    model,
                    tokenizer,
                    prompts[wl_id],
                    output_tokens,
                    device,
                    bs,
                    args.num_warmup,
                    args.num_runs,
                )

                key = f"{wl_id}_bs{bs}"
                all_results["results"][key] = result

                print(f"    TTFT P50: {result['ttft_ms']['p50']:.1f} ms")
                print(f"    Output tok/s P50: {result['output_tps']['p50']:.1f}")
                print(f"    Total tok/s: {result['total_tps']:.1f}")
                print(f"    E2E P50: {result['e2e_ms']['p50']:.1f} ms")
                print(f"    Peak GPU: {result['peak_gpu_mem_mb']:.0f} MB")

            except torch.cuda.OutOfMemoryError:
                print(
                    f"    OOM at batch_size={bs}, skipping larger batches for this workload"
                )
                key = f"{wl_id}_bs{bs}"
                all_results["results"][key] = {"error": "OOM"}
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"    Error: {e}")
                key = f"{wl_id}_bs{bs}"
                all_results["results"][key] = {"error": str(e)}

    # Save results
    # Remove raw arrays for clean JSON (keep percentiles)
    for key, val in all_results["results"].items():
        if isinstance(val, dict):
            val.pop("ttft_raw", None)
            val.pop("e2e_raw", None)
            val.pop("output_tps_raw", None)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(
        f"{'Workload':<15} {'BS':>3} {'TTFT P50':>10} {'tok/s P50':>10} {'Total tok/s':>12} {'E2E P50':>10} {'Peak MB':>8}"
    )
    print("-" * 75)
    for key, val in all_results["results"].items():
        if "error" in val:
            print(f"{key:<15} {'':>3} {'ERROR':>10}")
            continue
        wl = key.rsplit("_bs", 1)[0]
        bs = val["batch_size"]
        print(
            f"{wl:<15} {bs:>3} {val['ttft_ms']['p50']:>9.1f}ms {val['output_tps']['p50']:>9.1f} {val['total_tps']:>11.1f} {val['e2e_ms']['p50']:>9.1f}ms {val['peak_gpu_mem_mb']:>7.0f}"
        )


if __name__ == "__main__":
    main()
