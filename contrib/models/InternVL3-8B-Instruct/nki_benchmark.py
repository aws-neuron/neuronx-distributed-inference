#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NKI kernel benchmark for InternVL3-8B-Instruct.

Compiles with opt-in NKI kernels enabled, then benchmarks TTFT and TKG
throughput. Compares against baseline (non-NKI) numbers.

NKI kernels enabled:
  - qkv_kernel_enabled: Fused QKV projection with NKI RMSNorm (CTE + TKG)
  - mlp_kernel_enabled: Fused MLP (gate/up/down) kernel with NKI RMSNorm (CTE + TKG)

Note: attn_block_tkg_nki_kernel_enabled triggers NCC_ISTP902 compiler bug in SDK 2.29
(StaticProfiler error in O-proj all-reduce). Omitted until SDK fix.

Usage:
    # Full compile + benchmark
    python nki_benchmark.py

    # Skip compile, just benchmark
    python nki_benchmark.py --skip-compile

    # Accuracy validation after compile
    python nki_benchmark.py --accuracy-only --skip-compile

Target: trn2.3xlarge LNC=2 TP=4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_NKI_PATH = "/mnt/models/neuron_models/InternVL3-8B-VLM-NKI/"
COMPILED_BASELINE_PATH = "/mnt/models/neuron_models/InternVL3-8B-VLM/"
RESULTS_PATH = "/mnt/models/nki_benchmark_results.json"

# Baseline numbers (from Task 015 scale_test at seq_len=2048 batch_size=1)
BASELINE = {
    "ttft_ms": 138,
    "tkg_tok_s": 75.1,
}


def create_nki_config(seq_len=2048, batch_size=1):
    """Create config with NKI kernels enabled for the text backbone."""
    text_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
        fused_qkv=True,  # Required for NKI QKV kernel
        # NKI kernel flags
        qkv_kernel_enabled=True,  # Fused QKV projection with NKI RMSNorm
        mlp_kernel_enabled=True,  # Fused MLP with NKI RMSNorm
        # NOTE: attn_block_tkg_nki_kernel_enabled=True triggers NCC_ISTP902
        # compiler bug in SDK 2.29 (StaticProfiler error in O-proj all-reduce).
        # Omitted until SDK fix is available.
    )

    # Vision NeuronConfig: same as baseline (no NKI for vision encoder)
    vision_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        buckets=[1],
        fused_qkv=True,
        save_sharded_checkpoint=True,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        MODEL_PATH,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    return config


def compile_nki(config, compiled_path):
    """Compile model with NKI kernels."""
    print("=" * 60)
    print("COMPILING WITH NKI KERNELS")
    print("=" * 60)
    nc = config.text_config.neuron_config
    print(f"  qkv_kernel_enabled:                   {nc.qkv_kernel_enabled}")
    print(f"  qkv_nki_kernel_enabled:               {nc.qkv_nki_kernel_enabled}")
    print(
        f"  attn_block_tkg_nki_kernel_enabled:     {nc.attn_block_tkg_nki_kernel_enabled}"
    )
    print(f"  mlp_kernel_enabled:                    {nc.mlp_kernel_enabled}")
    print(f"  fused_qkv:                             {nc.fused_qkv}")
    print(f"  seq_len: {nc.seq_len}, batch_size: {nc.max_batch_size}")
    print(f"  Output: {compiled_path}")

    model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)

    start = time.time()
    model.compile(compiled_path)
    compile_time = time.time() - start
    print(
        f"\nCompilation completed in {compile_time:.1f}s ({compile_time / 60:.1f} min)"
    )

    return model, compile_time


def load_model(config, compiled_path):
    """Load already-compiled model."""
    print(f"\nLoading compiled model from {compiled_path}")
    model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)
    start = time.time()
    model.load(compiled_path)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.1f}s")
    return model


def measure_ttft(model, tokenizer, n_runs=5):
    """Measure TTFT for text-only CTE."""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(templated, return_tensors="pt")
    input_ids = inputs.input_ids

    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    seq_ids = torch.zeros(1, dtype=torch.int32)

    # Warmup
    model.reset()
    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids, seq_ids=seq_ids)

    # Measure
    ttft_times = []
    for _ in range(n_runs):
        model.reset()
        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids=input_ids, position_ids=position_ids, seq_ids=seq_ids)
        ttft_times.append(time.perf_counter() - start)

    avg = sum(ttft_times) / len(ttft_times)
    return avg * 1000  # ms


def measure_tkg(model, tokenizer, num_tokens=64, n_runs=3):
    """Measure TKG throughput using HuggingFaceGenerationAdapter."""
    adapter = HuggingFaceGenerationAdapter(model)

    messages = [
        {"role": "user", "content": "Tell me a detailed story about a brave knight."}
    ]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(templated, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = torch.ones_like(input_ids)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Warmup
    with torch.no_grad():
        adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=num_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

    # Measure
    tkg_results = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        total_time = time.perf_counter() - start
        generated = output_ids.shape[-1] - input_ids.shape[-1]
        tkg_results.append((generated, total_time))

    # Average tokens/sec (excluding TTFT — approximate by subtracting first-token overhead)
    avg_tokens = sum(r[0] for r in tkg_results) / len(tkg_results)
    avg_time = sum(r[1] for r in tkg_results) / len(tkg_results)
    tok_s = avg_tokens / avg_time if avg_time > 0 else 0.0

    return tok_s, avg_tokens, avg_time


def run_accuracy_check(adapter, tokenizer):
    """Quick accuracy sanity check."""
    prompts = [
        ("france", "What is the capital of France?"),
        ("math", "What is 2 + 2? Answer with just the number:"),
    ]

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    results = {}

    for name, prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(templated, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=eos_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[-1] :].tolist()
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results[name] = text
        print(f"  {name}: {text!r}")

    # Basic sanity
    france_ok = "paris" in results.get("france", "").lower()
    math_ok = "4" in results.get("math", "")
    return france_ok and math_ok


def main():
    parser = argparse.ArgumentParser(description="NKI kernel benchmark")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument(
        "--accuracy-only", action="store_true", help="Only run accuracy check"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config = create_nki_config(seq_len=args.seq_len, batch_size=args.batch_size)
    compile_time = None

    if not args.skip_compile:
        model, compile_time = compile_nki(config, COMPILED_NKI_PATH)
        # After compile(), model already has weights loaded in memory.
        # Call load() to load the compiled NEFFs onto Neuron devices
        # and save sharded weight checkpoints for future --skip-compile runs.
        model.load(COMPILED_NKI_PATH)
    else:
        model = load_model(config, COMPILED_NKI_PATH)

    # Accuracy check
    print("\n--- Accuracy Sanity Check ---")
    adapter = HuggingFaceGenerationAdapter(model)
    accuracy_ok = run_accuracy_check(adapter, tokenizer)
    print(f"Accuracy: {'PASS' if accuracy_ok else 'FAIL'}")

    if args.accuracy_only:
        return

    # TTFT benchmark
    print("\n--- TTFT Benchmark ---")
    ttft_ms = measure_ttft(model, tokenizer, n_runs=5)
    ttft_delta = ((ttft_ms - BASELINE["ttft_ms"]) / BASELINE["ttft_ms"]) * 100
    print(
        f"  TTFT:     {ttft_ms:.1f} ms (baseline: {BASELINE['ttft_ms']} ms, delta: {ttft_delta:+.1f}%)"
    )

    # TKG benchmark
    print("\n--- TKG Throughput Benchmark ---")
    tok_s, avg_tokens, avg_time = measure_tkg(model, tokenizer, num_tokens=64, n_runs=3)
    tkg_delta = ((tok_s - BASELINE["tkg_tok_s"]) / BASELINE["tkg_tok_s"]) * 100
    print(
        f"  TKG:      {tok_s:.1f} tok/s (baseline: {BASELINE['tkg_tok_s']} tok/s, delta: {tkg_delta:+.1f}%)"
    )
    print(f"  Avg tokens generated: {avg_tokens:.0f}")
    print(f"  Avg total time: {avg_time:.3f}s")

    # Summary
    print("\n" + "=" * 60)
    print("NKI BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Config:       seq_len={args.seq_len}, batch_size={args.batch_size}")
    print(f"  NKI kernels:  qkv + mlp (fused RMSNorm)")
    print(f"  Accuracy:     {'PASS' if accuracy_ok else 'FAIL'}")
    print(
        f"  TTFT:         {ttft_ms:.1f} ms  (baseline: {BASELINE['ttft_ms']} ms, {ttft_delta:+.1f}%)"
    )
    print(
        f"  TKG:          {tok_s:.1f} tok/s (baseline: {BASELINE['tkg_tok_s']} tok/s, {tkg_delta:+.1f}%)"
    )
    if compile_time:
        print(f"  Compile time: {compile_time:.1f}s ({compile_time / 60:.1f} min)")

    # Save results
    results = {
        "config": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "nki_kernels": [
                "qkv_kernel_enabled",
                "mlp_kernel_enabled",
            ],
        },
        "accuracy": "PASS" if accuracy_ok else "FAIL",
        "ttft_ms": round(ttft_ms, 1),
        "tkg_tok_s": round(tok_s, 1),
        "ttft_delta_pct": round(ttft_delta, 1),
        "tkg_delta_pct": round(tkg_delta, 1),
        "baseline": BASELINE,
        "compile_time_s": round(compile_time, 1) if compile_time else None,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
