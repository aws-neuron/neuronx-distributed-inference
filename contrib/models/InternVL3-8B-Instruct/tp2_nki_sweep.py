#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TP=2 NKI kernel sweep for InternVL3-8B-Instruct.

Tests all NKI kernel combinations at TP=2 on trn2.3xlarge LNC=2:
  A) baseline        — no NKI kernels
  B) qkv+mlp         — qkv_kernel_enabled + mlp_kernel_enabled
  C) attn_block_tkg   — attn_block_tkg_nki_kernel_enabled (was NCC_ISTP902 at TP=4)
  D) all_kernels      — qkv + mlp + attn_block_tkg

Each config: compile → load → accuracy check → TTFT (5 runs) → TKG (3 runs, 64 tok).

Usage:
    python tp2_nki_sweep.py                   # Run all configs
    python tp2_nki_sweep.py --configs baseline qkv_mlp   # Run specific configs
    python tp2_nki_sweep.py --skip-compile    # Skip compilation (load from disk)

Output: /mnt/models/tp2_nki_sweep_results.json
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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
BASE_COMPILED_DIR = "/mnt/models/neuron_models"
RESULTS_PATH = "/mnt/models/tp2_nki_sweep_results.json"

TP_DEGREE = 2
SEQ_LEN = 2048
BATCH_SIZE = 1

# NKI kernel configurations to test
CONFIGS = {
    "baseline": {
        "label": "Baseline (no NKI)",
        "fused_qkv": False,
        "qkv_kernel_enabled": False,
        "mlp_kernel_enabled": False,
        "attn_block_tkg_nki_kernel_enabled": False,
    },
    "qkv_mlp": {
        "label": "QKV + MLP kernels",
        "fused_qkv": True,
        "qkv_kernel_enabled": True,
        "mlp_kernel_enabled": True,
        "attn_block_tkg_nki_kernel_enabled": False,
    },
    "attn_block": {
        "label": "Attn Block TKG kernel",
        "fused_qkv": True,
        "qkv_kernel_enabled": True,  # Required by attn_block_tkg
        "mlp_kernel_enabled": False,
        "attn_block_tkg_nki_kernel_enabled": True,
    },
    "all_kernels": {
        "label": "All NKI kernels",
        "fused_qkv": True,
        "qkv_kernel_enabled": True,
        "mlp_kernel_enabled": True,
        "attn_block_tkg_nki_kernel_enabled": True,
    },
}


def compiled_path_for(config_name):
    return f"{BASE_COMPILED_DIR}/InternVL3-8B-tp2-{config_name}"


def create_config(config_name):
    """Create inference config for a given NKI kernel combination."""
    cfg = CONFIGS[config_name]

    text_neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        max_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
        fused_qkv=cfg["fused_qkv"],
        qkv_kernel_enabled=cfg["qkv_kernel_enabled"],
        mlp_kernel_enabled=cfg["mlp_kernel_enabled"],
        attn_block_tkg_nki_kernel_enabled=cfg["attn_block_tkg_nki_kernel_enabled"],
    )

    vision_neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        max_batch_size=1,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        buckets=[1],
        fused_qkv=True,  # Vision encoder always has fused QKV
        save_sharded_checkpoint=True,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        MODEL_PATH,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    return config


def compile_and_load(config_name, config, skip_compile=False):
    """Compile (or skip) and load model. Returns (model, compile_time) or raises."""
    cpath = compiled_path_for(config_name)
    compile_time = None

    if not skip_compile:
        print(f"\n--- Compiling: {CONFIGS[config_name]['label']} ---")
        nc = config.text_config.neuron_config
        print(f"  tp_degree={nc.tp_degree}, fused_qkv={nc.fused_qkv}")
        print(
            f"  qkv_kernel={nc.qkv_kernel_enabled}, mlp_kernel={nc.mlp_kernel_enabled}"
        )
        print(f"  attn_block_tkg={nc.attn_block_tkg_nki_kernel_enabled}")
        print(f"  Output: {cpath}")

        model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)
        start = time.time()
        model.compile(cpath)
        compile_time = time.time() - start
        print(f"  Compilation: {compile_time:.1f}s ({compile_time / 60:.1f} min)")

        # Load compiled NEFFs and save sharded weights
        model.load(cpath)
    else:
        print(f"\n--- Loading (skip-compile): {CONFIGS[config_name]['label']} ---")
        model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)
        model.load(cpath)

    return model, compile_time


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
    times = []
    for _ in range(n_runs):
        model.reset()
        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids=input_ids, position_ids=position_ids, seq_ids=seq_ids)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def measure_tkg(model, tokenizer, num_tokens=64, n_runs=3):
    """Measure TKG throughput via HuggingFaceGenerationAdapter."""
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
    results = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            out = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        elapsed = time.perf_counter() - start
        generated = out.shape[-1] - input_ids.shape[-1]
        results.append((generated, elapsed))

    avg_tok = sum(r[0] for r in results) / len(results)
    avg_time = sum(r[1] for r in results) / len(results)
    tok_s = avg_tok / avg_time if avg_time > 0 else 0.0
    return tok_s, avg_tok, avg_time


def run_accuracy_check(model, tokenizer):
    """Quick accuracy sanity check — returns (pass, details)."""
    adapter = HuggingFaceGenerationAdapter(model)
    prompts = [
        ("france", "What is the capital of France?"),
        ("math", "What is 2 + 2? Answer with just the number:"),
    ]
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    details = {}
    for name, text in prompts:
        messages = [{"role": "user", "content": text}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(templated, return_tensors="pt")
        input_ids = ids.input_ids
        with torch.no_grad():
            out = adapter.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        gen = tokenizer.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True)
        details[name] = gen
        print(f"    {name}: {gen!r}")

    ok = "paris" in details.get("france", "").lower() and "4" in details.get("math", "")
    return ok, details


def unload_model(model):
    """Best-effort cleanup of a loaded Neuron model."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    # Short sleep to let the runtime release NC resources
    time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="TP=2 NKI kernel sweep")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Which configs to test (default: all)",
    )
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    all_results = {}

    for config_name in args.configs:
        cfg_info = CONFIGS[config_name]
        print("\n" + "=" * 70)
        print(f"CONFIG: {config_name} — {cfg_info['label']}")
        print(f"  TP={TP_DEGREE}, seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
        print("=" * 70)

        result = {
            "label": cfg_info["label"],
            "tp_degree": TP_DEGREE,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "fused_qkv": cfg_info["fused_qkv"],
            "qkv_kernel_enabled": cfg_info["qkv_kernel_enabled"],
            "mlp_kernel_enabled": cfg_info["mlp_kernel_enabled"],
            "attn_block_tkg_nki_kernel_enabled": cfg_info[
                "attn_block_tkg_nki_kernel_enabled"
            ],
        }

        try:
            config = create_config(config_name)
            model, compile_time = compile_and_load(
                config_name, config, args.skip_compile
            )
            result["compile_time_s"] = round(compile_time, 1) if compile_time else None
            result["compile_status"] = "PASS"

            # Accuracy
            print(f"\n  Accuracy check:")
            acc_ok, acc_details = run_accuracy_check(model, tokenizer)
            result["accuracy"] = "PASS" if acc_ok else "FAIL"
            result["accuracy_details"] = acc_details
            print(f"  Accuracy: {result['accuracy']}")

            # TTFT
            print(f"\n  TTFT benchmark (5 runs):")
            ttft = measure_ttft(model, tokenizer, n_runs=5)
            result["ttft_ms"] = round(ttft, 1)
            print(f"  TTFT: {ttft:.1f} ms")

            # TKG
            print(f"\n  TKG benchmark (3 runs, 64 tok):")
            tok_s, avg_tok, avg_time = measure_tkg(
                model, tokenizer, num_tokens=64, n_runs=3
            )
            result["tkg_tok_s"] = round(tok_s, 1)
            result["tkg_avg_tokens"] = round(avg_tok, 1)
            result["tkg_avg_time_s"] = round(avg_time, 3)
            print(f"  TKG: {tok_s:.1f} tok/s ({avg_tok:.0f} tokens in {avg_time:.3f}s)")

            result["status"] = "PASS"

        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n  FAILED: {e}")
            # Print full traceback for debugging
            traceback.print_exc()

        all_results[config_name] = result

        # Attempt to unload before next config
        try:
            unload_model(model)
        except Exception:
            pass

        # Save intermediate results after each config
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Intermediate results saved to {RESULTS_PATH}")

    # Final summary
    print("\n\n" + "=" * 70)
    print(f"TP={TP_DEGREE} NKI KERNEL SWEEP — FINAL SUMMARY")
    print("=" * 70)

    # Find baseline for delta computation
    baseline_ttft = all_results.get("baseline", {}).get("ttft_ms")
    baseline_tkg = all_results.get("baseline", {}).get("tkg_tok_s")

    header = f"{'Config':<20} {'Status':<8} {'Accuracy':<8} {'TTFT(ms)':<10} {'TKG(tok/s)':<12} {'TTFT Δ':<10} {'TKG Δ':<10}"
    print(header)
    print("-" * len(header))

    for name in args.configs:
        r = all_results.get(name, {})
        status = r.get("status", "N/A")
        acc = r.get("accuracy", "N/A")
        ttft = r.get("ttft_ms", None)
        tkg = r.get("tkg_tok_s", None)

        ttft_str = f"{ttft:.1f}" if ttft else "N/A"
        tkg_str = f"{tkg:.1f}" if tkg else "N/A"

        if baseline_ttft and ttft and name != "baseline":
            ttft_d = ((ttft - baseline_ttft) / baseline_ttft) * 100
            ttft_delta = f"{ttft_d:+.1f}%"
        else:
            ttft_delta = "—"

        if baseline_tkg and tkg and name != "baseline":
            tkg_d = ((tkg - baseline_tkg) / baseline_tkg) * 100
            tkg_delta = f"{tkg_d:+.1f}%"
        else:
            tkg_delta = "—"

        print(
            f"{name:<20} {status:<8} {acc:<8} {ttft_str:<10} {tkg_str:<12} {ttft_delta:<10} {tkg_delta:<10}"
        )

    print(f"\nFull results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
