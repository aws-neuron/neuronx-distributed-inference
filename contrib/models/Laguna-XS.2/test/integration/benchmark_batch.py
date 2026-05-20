#!/usr/bin/env python3
"""Batch size optimization benchmark for Laguna-XS.2.

Measures TKG throughput (tok/s), TTFT (CTE latency), and HBM usage
at different batch sizes.

Usage:
    export LAGUNA_MODEL_PATH=/mnt/models/Laguna-XS.2
    export LAGUNA_TP_DEGREE=4
    export LAGUNA_SEQ_LEN=8192

    # Test a single batch size
    python benchmark_batch.py --batch-size 1

    # Test multiple batch sizes
    python benchmark_batch.py --batch-size 1 2 4 8
"""

import argparse
import json
import os
import sys
import time

import torch

# Add src to path
test_dir = os.path.dirname(os.path.abspath(__file__))
contrib_dir = os.path.dirname(os.path.dirname(test_dir))
sys.path.insert(0, contrib_dir)

from src.modeling_laguna import (
    NeuronLagunaForCausalLM,
    LagunaInferenceConfig,
)

MODEL_PATH = os.environ.get("LAGUNA_MODEL_PATH", "/mnt/models/Laguna-XS.2")
TP_DEGREE = int(os.environ.get("LAGUNA_TP_DEGREE", "4"))
SEQ_LEN = int(os.environ.get("LAGUNA_SEQ_LEN", "8192"))
COMPILED_BASE = os.environ.get("LAGUNA_COMPILED_BASE", "/mnt/models/laguna-bench")

# Benchmark params
NUM_WARMUP_TOKENS = 5
NUM_MEASURE_TOKENS = 50
PROMPT = "The capital of France is"


def get_hbm_usage():
    """Get total HBM usage across all neuron cores in GB."""
    try:
        import subprocess

        result = subprocess.run(
            ["neuron-monitor", "--once"], capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        total_hbm = 0
        for nc in data.get("neuron_runtime_data", []):
            for report in (
                nc.get("report", {})
                .get("neuroncore_counters", {})
                .get("neuroncores_in_use", {})
                .values()
            ):
                hbm = report.get("mem_used_hbm", 0)
                total_hbm += hbm
        return total_hbm / (1024**3)  # Convert to GB
    except Exception:
        return -1.0


def build_config(batch_size):
    from neuronx_distributed_inference.models.config import MoENeuronConfig

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=batch_size,
        max_batch_size=batch_size,
        seq_len=SEQ_LEN,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
    )

    config = LagunaInferenceConfig.from_pretrained(
        MODEL_PATH,
        neuron_config=neuron_config,
    )
    return config


def benchmark_batch_size(batch_size, recompile=False):
    """Benchmark a single batch size. Returns dict of metrics."""
    compiled_path = f"{COMPILED_BASE}-bs{batch_size}"

    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING: batch_size={batch_size}, seq_len={SEQ_LEN}, TP={TP_DEGREE}")
    print(f"{'=' * 60}")

    config = build_config(batch_size)

    # Check if already compiled
    needs_compile = not os.path.exists(compiled_path) or recompile
    if needs_compile:
        print(f"  Compiling to {compiled_path}...")
        compile_start = time.time()
        model = NeuronLagunaForCausalLM(MODEL_PATH, config)
        model.compile(compiled_path)
        compile_time = time.time() - compile_start
        print(f"  Compilation: {compile_time:.1f}s")
        del model
    else:
        print(f"  Using cached compilation at {compiled_path}")
        compile_time = 0.0

    # Load model
    print(f"  Loading model...")
    load_start = time.time()
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.load(compiled_path)
    load_time = time.time() - load_start
    print(f"  Load time: {load_time:.1f}s")

    # HBM usage after load
    hbm_gb = get_hbm_usage()
    print(f"  HBM usage: {hbm_gb:.2f} GB")

    # Tokenize prompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    prompt_len = len(ids)

    # Prepare batched inputs for CTE
    input_ids = torch.zeros(batch_size, SEQ_LEN, dtype=torch.int32)
    attention_mask = torch.zeros(batch_size, SEQ_LEN, dtype=torch.int32)
    position_ids = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)

    for b in range(batch_size):
        input_ids[b, :prompt_len] = torch.tensor(ids, dtype=torch.int32)
        attention_mask[b, :prompt_len] = 1
        position_ids[b, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)

    # === TTFT (CTE) Measurement ===
    # Warmup CTE
    print(f"  Warming up CTE...")
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # Measure TTFT (average of 3 runs)
    ttft_times = []
    for _ in range(3):
        # Reset KV cache by running CTE again
        t0 = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        ttft_times.append(time.time() - t0)

    ttft_avg = sum(ttft_times) / len(ttft_times)
    print(f"  TTFT (avg of 3): {ttft_avg * 1000:.1f} ms")

    # Get first token from CTE
    logits = outputs.logits if hasattr(outputs, "logits") else outputs.tokens
    if logits.dim() == 3:
        token_ids = logits[:, -1, :].argmax(dim=-1)  # [batch_size]
    elif logits.dim() == 2:
        token_ids = logits.argmax(dim=-1)
    else:
        token_ids = logits.argmax().unsqueeze(0).expand(batch_size)

    # === TKG Throughput Measurement ===
    cur_pos = prompt_len
    total_tokens = NUM_WARMUP_TOKENS + NUM_MEASURE_TOKENS

    print(
        f"  Running TKG: {NUM_WARMUP_TOKENS} warmup + {NUM_MEASURE_TOKENS} measured tokens..."
    )

    # Warmup TKG
    for step in range(NUM_WARMUP_TOKENS):
        tkg_in = token_ids.unsqueeze(1)  # [batch_size, 1]
        am_len = cur_pos + 1
        tkg_mask = torch.cat(
            [
                torch.ones(batch_size, am_len, dtype=torch.long),
                torch.zeros(batch_size, SEQ_LEN - am_len, dtype=torch.long),
            ],
            dim=1,
        )
        with torch.no_grad():
            out = model(
                input_ids=tkg_in,
                attention_mask=tkg_mask,
                position_ids=torch.full((batch_size, 1), cur_pos, dtype=torch.long),
            )
        cur_pos += 1
        out_logits = out.logits if hasattr(out, "logits") else out.tokens
        if out_logits.dim() == 3:
            token_ids = out_logits[:, -1, :].argmax(dim=-1)
        elif out_logits.dim() == 2:
            token_ids = out_logits.argmax(dim=-1)
        else:
            token_ids = out_logits.argmax().unsqueeze(0).expand(batch_size)

    # Measure TKG
    tkg_start = time.time()
    for step in range(NUM_MEASURE_TOKENS):
        tkg_in = token_ids.unsqueeze(1)
        am_len = cur_pos + 1
        tkg_mask = torch.cat(
            [
                torch.ones(batch_size, am_len, dtype=torch.long),
                torch.zeros(batch_size, SEQ_LEN - am_len, dtype=torch.long),
            ],
            dim=1,
        )
        with torch.no_grad():
            out = model(
                input_ids=tkg_in,
                attention_mask=tkg_mask,
                position_ids=torch.full((batch_size, 1), cur_pos, dtype=torch.long),
            )
        cur_pos += 1
        out_logits = out.logits if hasattr(out, "logits") else out.tokens
        if out_logits.dim() == 3:
            token_ids = out_logits[:, -1, :].argmax(dim=-1)
        elif out_logits.dim() == 2:
            token_ids = out_logits.argmax(dim=-1)
        else:
            token_ids = out_logits.argmax().unsqueeze(0).expand(batch_size)

    tkg_elapsed = time.time() - tkg_start
    total_tkg_tokens = NUM_MEASURE_TOKENS * batch_size
    tkg_tok_per_sec = total_tkg_tokens / tkg_elapsed
    tkg_latency_ms = (tkg_elapsed / NUM_MEASURE_TOKENS) * 1000  # per-step latency

    print(
        f"  TKG throughput: {tkg_tok_per_sec:.1f} tok/s ({batch_size} * {NUM_MEASURE_TOKENS} tokens in {tkg_elapsed:.2f}s)"
    )
    print(f"  TKG latency: {tkg_latency_ms:.1f} ms/step")
    print(f"  TPOT (per token): {tkg_latency_ms / batch_size:.1f} ms")

    # Decode generated text for sanity check
    generated_text = tokenizer.decode(token_ids[0:1].tolist(), skip_special_tokens=True)
    print(f"  Last token decoded (batch 0): '{generated_text}'")

    results = {
        "batch_size": batch_size,
        "seq_len": SEQ_LEN,
        "tp_degree": TP_DEGREE,
        "compile_time_s": compile_time,
        "load_time_s": load_time,
        "hbm_gb": hbm_gb,
        "ttft_ms": ttft_avg * 1000,
        "tkg_tok_per_sec": tkg_tok_per_sec,
        "tkg_latency_ms": tkg_latency_ms,
        "tpot_ms": tkg_latency_ms / batch_size,
        "num_measure_tokens": NUM_MEASURE_TOKENS,
    }

    # Cleanup
    del model
    import gc

    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--recompile", action="store_true", help="Force recompilation even if cached"
    )
    args = parser.parse_args()

    all_results = []
    for bs in args.batch_size:
        try:
            result = benchmark_batch_size(bs, recompile=args.recompile)
            all_results.append(result)
        except Exception as e:
            print(f"\n  FAILED at batch_size={bs}: {e}")
            all_results.append({"batch_size": bs, "error": str(e)})

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: seq_len={SEQ_LEN}, TP={TP_DEGREE}")
    print(f"{'=' * 80}")
    print(
        f"{'BS':>4} | {'TKG tok/s':>10} | {'TTFT (ms)':>10} | {'Latency ms':>11} | {'TPOT ms':>8} | {'HBM (GB)':>9} | {'Status'}"
    )
    print(
        f"{'-' * 4}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 11}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 10}"
    )
    for r in all_results:
        if "error" in r:
            print(
                f"{r['batch_size']:>4} | {'—':>10} | {'—':>10} | {'—':>11} | {'—':>8} | {'—':>9} | FAIL: {r['error'][:40]}"
            )
        else:
            print(
                f"{r['batch_size']:>4} | {r['tkg_tok_per_sec']:>10.1f} | {r['ttft_ms']:>10.1f} | {r['tkg_latency_ms']:>11.1f} | {r['tpot_ms']:>8.1f} | {r['hbm_gb']:>9.2f} | PASS"
            )
    print()

    # Save results
    results_path = f"{COMPILED_BASE}-results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
