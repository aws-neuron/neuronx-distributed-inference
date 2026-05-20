#!/usr/bin/env python3
"""Benchmark specific workload shapes for Laguna-XS.2.

Runs:
1. Coding workload: realistic prompt, 256 output tokens, BS=1, seq_len=8192
2. Short prompt high throughput: 128 tokens in, 512 out, BS=4, seq_len=4096

Usage:
    python benchmark_workloads.py
"""

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
from neuronx_distributed_inference.models.config import MoENeuronConfig
from transformers import AutoTokenizer

MODEL_PATH = "/mnt/models/Laguna-XS.2"


def build_config(batch_size, seq_len):
    neuron_config = MoENeuronConfig(
        tp_degree=4,
        batch_size=batch_size,
        max_batch_size=batch_size,
        seq_len=seq_len,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
    )
    return LagunaInferenceConfig.from_pretrained(
        MODEL_PATH, neuron_config=neuron_config
    )


def run_benchmark(
    model, tokenizer, prompt, batch_size, seq_len, num_output_tokens, label
):
    """Run a benchmark for a given workload."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(ids)

    # Prepare CTE input
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.int32)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int32)
    position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for b in range(batch_size):
        input_ids[b, :prompt_len] = torch.tensor(ids, dtype=torch.int32)
        attention_mask[b, :prompt_len] = 1
        position_ids[b, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)

    # Warmup CTE
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # === Measure TTFT (5 runs) ===
    ttft_runs = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        ttft_runs.append(time.time() - t0)

    ttft_median = sorted(ttft_runs)[len(ttft_runs) // 2]
    ttft_p95 = sorted(ttft_runs)[-1]  # with 5 runs, max is ~p95

    # Get first token
    logits = outputs.logits if hasattr(outputs, "logits") else outputs.tokens
    if logits.dim() == 3:
        token_ids = logits[:, -1, :].argmax(dim=-1)
    else:
        token_ids = logits.argmax(dim=-1)

    # === Measure TKG ===
    NUM_WARMUP = 5
    cur_pos = prompt_len
    generated = [token_ids[0].item()]

    # Warmup TKG
    for step in range(NUM_WARMUP):
        tkg_in = token_ids.unsqueeze(1)
        am_len = cur_pos + 1
        tkg_mask = torch.cat(
            [
                torch.ones(batch_size, am_len, dtype=torch.long),
                torch.zeros(batch_size, seq_len - am_len, dtype=torch.long),
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
        else:
            token_ids = out_logits.argmax(dim=-1)
        generated.append(token_ids[0].item())

    # Measure TKG
    per_token_times = []
    for step in range(num_output_tokens):
        tkg_in = token_ids.unsqueeze(1)
        am_len = cur_pos + 1
        tkg_mask = torch.cat(
            [
                torch.ones(batch_size, am_len, dtype=torch.long),
                torch.zeros(batch_size, seq_len - am_len, dtype=torch.long),
            ],
            dim=1,
        )
        t0 = time.time()
        with torch.no_grad():
            out = model(
                input_ids=tkg_in,
                attention_mask=tkg_mask,
                position_ids=torch.full((batch_size, 1), cur_pos, dtype=torch.long),
            )
        per_token_times.append(time.time() - t0)
        cur_pos += 1
        out_logits = out.logits if hasattr(out, "logits") else out.tokens
        if out_logits.dim() == 3:
            token_ids = out_logits[:, -1, :].argmax(dim=-1)
        else:
            token_ids = out_logits.argmax(dim=-1)
        generated.append(token_ids[0].item())

        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if token_ids[0].item() in eos_ids:
            break

    actual_tokens = len(per_token_times)
    total_time = sum(per_token_times)
    tpot_median = sorted(per_token_times)[actual_tokens // 2] * 1000
    tpot_p95 = (
        sorted(per_token_times)[min(int(actual_tokens * 0.95), actual_tokens - 1)]
        * 1000
    )
    tok_per_sec = (actual_tokens * batch_size) / total_time
    e2e_latency = ttft_median + total_time

    text = tokenizer.decode(generated[:60], skip_special_tokens=True)

    print(f"\n{'=' * 70}")
    print(f"WORKLOAD: {label}")
    print(f"{'=' * 70}")
    print(f"  Config: BS={batch_size}, seq_len={seq_len}, TP=4")
    print(f"  Input: {prompt_len} tokens")
    print(f"  Output: {actual_tokens} tokens (requested {num_output_tokens})")
    print(f"")
    print(
        f"  TTFT:          {ttft_median * 1000:.1f} ms (median), {ttft_p95 * 1000:.1f} ms (p95)"
    )
    print(f"  TKG tok/s:     {tok_per_sec:.1f} (aggregate: {batch_size} streams)")
    print(f"  TPOT:          {tpot_median:.2f} ms (median), {tpot_p95:.2f} ms (p95)")
    print(f"  E2E latency:   {e2e_latency * 1000:.0f} ms")
    print(f"  Step latency:  {(total_time / actual_tokens) * 1000:.2f} ms/step")
    print(f"")
    print(f"  Output (first 60 tokens): {repr(text[:250])}")

    return {
        "label": label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prompt_tokens": prompt_len,
        "output_tokens": actual_tokens,
        "ttft_median_ms": ttft_median * 1000,
        "ttft_p95_ms": ttft_p95 * 1000,
        "tkg_tok_per_sec": tok_per_sec,
        "tpot_median_ms": tpot_median,
        "tpot_p95_ms": tpot_p95,
        "e2e_latency_ms": e2e_latency * 1000,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ============================================================
    # Workload 1: Coding workload (BS=1, seq_len=8192)
    # ============================================================
    coding_prompt = """# Task: Implement a binary search tree with insert, search, and delete operations
# The tree should support in-order traversal and finding the minimum/maximum values.

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        else:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)

    def search(self, val):
"""

    short_prompt = (
        "Write a Python function that sorts a list of numbers using quicksort."
    )

    results = []

    # --- Config 1: Coding workload (BS=1, 8K) ---
    print("\n>>> Loading model: BS=1, seq_len=8192")
    config = build_config(batch_size=1, seq_len=8192)
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.load("/mnt/models/laguna-bench-bs1")
    print("    Model loaded.")

    r = run_benchmark(
        model,
        tokenizer,
        coding_prompt,
        batch_size=1,
        seq_len=8192,
        num_output_tokens=256,
        label="Coding (long input, long output)",
    )
    results.append(r)

    # Also test short-prompt on same model (short-long workload at BS=1)
    r = run_benchmark(
        model,
        tokenizer,
        short_prompt,
        batch_size=1,
        seq_len=8192,
        num_output_tokens=512,
        label="Short prompt, long output (BS=1, 8K)",
    )
    results.append(r)

    del model
    import gc

    gc.collect()

    # --- Config 2: Throughput (BS=4, 4K) ---
    print("\n>>> Loading model: BS=4, seq_len=4096")
    config = build_config(batch_size=4, seq_len=4096)
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.load("/mnt/models/laguna-bench-4k-bs4")
    print("    Model loaded.")

    r = run_benchmark(
        model,
        tokenizer,
        short_prompt,
        batch_size=4,
        seq_len=4096,
        num_output_tokens=512,
        label="Short prompt, long output (BS=4, 4K)",
    )
    results.append(r)

    r = run_benchmark(
        model,
        tokenizer,
        coding_prompt,
        batch_size=4,
        seq_len=4096,
        num_output_tokens=256,
        label="Coding throughput (BS=4, 4K)",
    )
    results.append(r)

    del model
    gc.collect()

    # --- Config 3: Max throughput (BS=8, 2K) ---
    print("\n>>> Loading model: BS=8, seq_len=2048")
    config = build_config(batch_size=8, seq_len=2048)
    model = NeuronLagunaForCausalLM(MODEL_PATH, config)
    model.load("/mnt/models/laguna-bench-2k-bs8")
    print("    Model loaded.")

    r = run_benchmark(
        model,
        tokenizer,
        short_prompt,
        batch_size=8,
        seq_len=2048,
        num_output_tokens=512,
        label="Short prompt, max throughput (BS=8, 2K)",
    )
    results.append(r)

    del model
    gc.collect()

    # === SUMMARY ===
    print(f"\n{'=' * 80}")
    print(f"PERFORMANCE SUMMARY: Laguna-XS.2 on trn2.3xlarge TP=4")
    print(f"{'=' * 80}")
    print(
        f"{'Workload':<45} | {'TTFT ms':>8} | {'tok/s':>7} | {'TPOT ms':>8} | {'E2E ms':>8}"
    )
    print(f"{'-' * 45}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 8}")
    for r in results:
        print(
            f"{r['label']:<45} | {r['ttft_median_ms']:>8.0f} | {r['tkg_tok_per_sec']:>7.1f} | {r['tpot_median_ms']:>8.1f} | {r['e2e_latency_ms']:>8.0f}"
        )
    print()


if __name__ == "__main__":
    main()
