#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task 015-016: Sequence length and batch size scaling tests for InternVL3-8B-Instruct.

Tests compilation and inference at various seq_len and batch_size configurations.
Records compilation time, TTFT, TKG throughput, and HBM usage.

Usage:
    # Run all scaling tests (compiles each configuration)
    python scale_test.py

    # Test specific seq_len
    python scale_test.py --seq-len 4096

    # Test specific batch size at given seq_len
    python scale_test.py --seq-len 4096 --batch-size 4

    # Skip compilation (use existing compiled models)
    python scale_test.py --skip-compile --seq-len 2048

Target: trn2.3xlarge LNC=2 TP=4
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_BASE = "/mnt/models/neuron_models/"
RESULTS_PATH = "/mnt/models/scale_test_results.json"

# Special token IDs
IMG_CONTEXT_ID = 151667
IMG_START_ID = 151665
IMG_END_ID = 151666


def get_compiled_path(seq_len, batch_size):
    """Get compiled model path for a specific configuration."""
    return os.path.join(COMPILED_BASE, f"InternVL3-8B-s{seq_len}-b{batch_size}")


def create_config(seq_len, batch_size):
    """Create InternVL3 config for given seq_len and batch_size."""
    text_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
    )

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


def get_hbm_usage():
    """Get current HBM usage from neuron-monitor."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "neuron-monitor",
                "-c",
                '{"period":"1s","neuron_runtimes":[{"tag_filter":".*","metrics":[{"type":"memory_info"}]}]}',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Simple parsing - just get overall usage
        if result.returncode == 0:
            import re

            matches = re.findall(r'"hbm_total_bytes":\s*(\d+)', result.stdout)
            if matches:
                return int(matches[0]) / (1024**3)  # GB
    except Exception:
        pass
    return None


def measure_ttft(model, tokenizer, prompt, seq_len, batch_size, pixel_values=None):
    """Measure time to first token (TTFT) for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    if pixel_values is not None:
        # Build multimodal input
        text_ids = input_ids[0]
        img_tokens = torch.full((256,), IMG_CONTEXT_ID, dtype=torch.long)
        input_ids = torch.cat(
            [
                text_ids,
                torch.tensor([IMG_START_ID]),
                img_tokens,
                torch.tensor([IMG_END_ID]),
            ]
        ).unsqueeze(0)

    # Repeat for batch
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    actual_seq_len = input_ids.shape[-1]
    position_ids = torch.arange(actual_seq_len, dtype=torch.int32).unsqueeze(0)
    if batch_size > 1:
        position_ids = position_ids.repeat(batch_size, 1)
    seq_ids = torch.arange(batch_size, dtype=torch.int32)

    # Warmup
    model.reset()
    with torch.no_grad():
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
            pixel_values=pixel_values,
        )

    # Measure TTFT (average of 3 runs)
    ttft_times = []
    for _ in range(3):
        model.reset()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                seq_ids=seq_ids,
                pixel_values=pixel_values,
            )
        ttft = time.perf_counter() - start
        ttft_times.append(ttft)

    avg_ttft = sum(ttft_times) / len(ttft_times)
    return avg_ttft, actual_seq_len


def measure_tkg_throughput(model, tokenizer, prompt, num_tokens=32, batch_size=1):
    """Measure TKG throughput (tokens/sec) for autoregressive generation."""
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    adapter = HuggingFaceGenerationAdapter(model)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    attention_mask = torch.ones_like(input_ids)

    # Warmup
    with torch.no_grad():
        adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
        )

    # Measure (average of 3 runs)
    tkg_times = []
    for _ in range(3):
        start = time.perf_counter()
        with torch.no_grad():
            out = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start
        tkg_times.append(elapsed)

    avg_time = sum(tkg_times) / len(tkg_times)
    # Total tokens generated = num_tokens * batch_size
    total_tokens = num_tokens * batch_size
    throughput = total_tokens / avg_time

    return throughput, avg_time, num_tokens


def run_test(seq_len, batch_size, skip_compile=False):
    """Run a single scaling test configuration."""
    compiled_path = get_compiled_path(seq_len, batch_size)
    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "compiled_path": compiled_path,
        "status": "pending",
    }

    print(f"\n{'=' * 60}")
    print(f"Testing: seq_len={seq_len}, batch_size={batch_size}")
    print(f"{'=' * 60}")

    try:
        # Create config
        config = create_config(seq_len, batch_size)
        model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)

        # Compile
        if not skip_compile:
            print(f"\nCompiling to {compiled_path}...")
            start = time.time()
            model.compile(compiled_path)
            compile_time = time.time() - start
            result["compile_time_s"] = round(compile_time, 1)
            print(f"Compilation: {compile_time:.1f}s ({compile_time / 60:.1f} min)")
        else:
            print(f"\nSkipping compile, loading from {compiled_path}")
            result["compile_time_s"] = "skipped"

        # Load
        print("Loading compiled model...")
        start = time.time()
        model.load(compiled_path)
        load_time = time.time() - start
        result["load_time_s"] = round(load_time, 1)
        print(f"Load: {load_time:.1f}s")

        # HBM usage
        hbm = get_hbm_usage()
        if hbm:
            result["hbm_usage_gb"] = round(hbm, 2)
            print(f"HBM usage: {hbm:.2f} GB")

        # Tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # Test 1: Text-only TTFT
        print("\n--- Text-only TTFT ---")
        prompt = "What is the capital of France?"
        ttft, actual_seq = measure_ttft(model, tokenizer, prompt, seq_len, batch_size)
        result["text_ttft_ms"] = round(ttft * 1000, 1)
        result["text_ttft_input_len"] = actual_seq
        print(
            f"TTFT: {ttft * 1000:.1f} ms (input_len={actual_seq}, batch={batch_size})"
        )

        # Test 2: TKG throughput
        print("\n--- TKG throughput ---")
        throughput, gen_time, num_tok = measure_tkg_throughput(
            model, tokenizer, prompt, num_tokens=32, batch_size=batch_size
        )
        result["tkg_throughput_tok_s"] = round(throughput, 1)
        result["tkg_gen_time_s"] = round(gen_time, 3)
        result["tkg_num_tokens"] = num_tok
        print(
            f"Throughput: {throughput:.1f} tok/s ({gen_time:.3f}s for {num_tok} tokens, batch={batch_size})"
        )

        # Test 3: Multimodal TTFT (only for batch_size=1)
        if batch_size == 1:
            print("\n--- Multimodal TTFT ---")
            pixel_values = torch.randn(1, 3, 448, 448)
            mm_ttft, mm_seq = measure_ttft(
                model,
                tokenizer,
                "Describe this image:",
                seq_len,
                batch_size,
                pixel_values=pixel_values,
            )
            result["multimodal_ttft_ms"] = round(mm_ttft * 1000, 1)
            result["multimodal_input_len"] = mm_seq
            print(f"Multimodal TTFT: {mm_ttft * 1000:.1f} ms (input_len={mm_seq})")

        # Test 4: Long prompt TTFT (fill most of seq_len)
        if seq_len >= 4096:
            print(f"\n--- Long prompt TTFT (target ~{seq_len // 2} tokens) ---")
            # Generate a long prompt by repeating text
            base_text = "The quick brown fox jumps over the lazy dog. " * 100
            long_inputs = tokenizer(
                base_text, return_tensors="pt", max_length=seq_len // 2, truncation=True
            )
            long_prompt_len = long_inputs.input_ids.shape[-1]
            long_ids = long_inputs.input_ids
            if batch_size > 1:
                long_ids = long_ids.repeat(batch_size, 1)
            long_pos = torch.arange(long_prompt_len, dtype=torch.int32).unsqueeze(0)
            if batch_size > 1:
                long_pos = long_pos.repeat(batch_size, 1)
            long_seq_ids = torch.arange(batch_size, dtype=torch.int32)

            model.reset()
            long_ttft_times = []
            for _ in range(3):
                model.reset()
                start = time.perf_counter()
                with torch.no_grad():
                    model(
                        input_ids=long_ids, position_ids=long_pos, seq_ids=long_seq_ids
                    )
                long_ttft_times.append(time.perf_counter() - start)

            long_ttft = sum(long_ttft_times) / len(long_ttft_times)
            result["long_ttft_ms"] = round(long_ttft * 1000, 1)
            result["long_ttft_input_len"] = long_prompt_len
            print(
                f"Long TTFT: {long_ttft * 1000:.1f} ms (input_len={long_prompt_len}, batch={batch_size})"
            )

        result["status"] = "PASS"
        print(f"\nResult: PASS")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"\nResult: FAIL - {e}")

    finally:
        # Cleanup to free HBM
        if "model" in locals():
            del model
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="InternVL3 scaling tests")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Test specific seq_len (default: test 2048, 4096, 8192)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Test specific batch_size (default: test 1, 2, 4)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation, use existing compiled models",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("InternVL3-8B-Instruct: Scaling Tests (Tasks 015-016)")
    print("=" * 60)
    print(f"Instance: trn2.3xlarge LNC=2 TP=4")
    print(f"Model: {MODEL_PATH}")

    all_results = []

    if args.seq_len and args.batch_size:
        # Single specific test
        result = run_test(args.seq_len, args.batch_size, args.skip_compile)
        all_results.append(result)
    elif args.seq_len:
        # Test specific seq_len with batch_size sweep
        for bs in [1, 2, 4]:
            result = run_test(args.seq_len, bs, args.skip_compile)
            all_results.append(result)
            if result["status"] == "FAIL":
                print(f"\nStopping batch sweep at batch_size={bs} (failed)")
                break
    elif args.batch_size:
        # Test specific batch_size with seq_len sweep
        for sl in [2048, 4096, 8192]:
            result = run_test(sl, args.batch_size, args.skip_compile)
            all_results.append(result)
            if result["status"] == "FAIL":
                print(f"\nStopping seq_len sweep at seq_len={sl} (failed)")
                break
    else:
        # Full scaling matrix
        # Phase 1: seq_len sweep at batch_size=1
        print("\n\n*** PHASE 1: Sequence Length Sweep (batch_size=1) ***")
        max_passing_seq_len = 2048
        for sl in [2048, 4096, 8192]:
            result = run_test(sl, 1, args.skip_compile)
            all_results.append(result)
            if result["status"] == "PASS":
                max_passing_seq_len = sl
            else:
                print(
                    f"\nSeq_len ceiling: {sl} fails. Max passing: {max_passing_seq_len}"
                )
                break

        # Phase 2: batch_size sweep at recommended seq_len
        recommended_seq_len = min(max_passing_seq_len, 4096)
        print(f"\n\n*** PHASE 2: Batch Size Sweep (seq_len={recommended_seq_len}) ***")
        for bs in [2, 4]:
            result = run_test(recommended_seq_len, bs, args.skip_compile)
            all_results.append(result)
            if result["status"] == "FAIL":
                print(
                    f"\nBatch size ceiling at seq_len={recommended_seq_len}: bs={bs} fails"
                )
                break

    # Summary
    print("\n\n" + "=" * 60)
    print("SCALING TEST SUMMARY")
    print("=" * 60)

    print(
        f"\n{'seq_len':<10} {'batch':<8} {'status':<8} {'compile':<12} {'TTFT(ms)':<12} {'tok/s':<10} {'mm_TTFT':<12}"
    )
    print("-" * 72)
    for r in all_results:
        compile_t = r.get("compile_time_s", "?")
        if isinstance(compile_t, (int, float)):
            compile_str = f"{compile_t:.0f}s"
        else:
            compile_str = str(compile_t)

        ttft = r.get("text_ttft_ms", "?")
        ttft_str = f"{ttft}" if isinstance(ttft, str) else f"{ttft:.1f}"

        tps = r.get("tkg_throughput_tok_s", "?")
        tps_str = f"{tps}" if isinstance(tps, str) else f"{tps:.1f}"

        mm = r.get("multimodal_ttft_ms", "-")
        mm_str = f"{mm}" if isinstance(mm, str) else f"{mm:.1f}"

        print(
            f"{r['seq_len']:<10} {r['batch_size']:<8} {r['status']:<8} {compile_str:<12} {ttft_str:<12} {tps_str:<10} {mm_str:<12}"
        )

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
