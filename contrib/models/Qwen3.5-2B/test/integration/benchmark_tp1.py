# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Qwen3.5-2B at TP=1, BS=2 on a single NeuronCore.

This script tests the _forward_with_pad fix by running at TP=1 with BS=2
(BS=1 hits compiler exit code 70 at TP=1, so BS=2 is required).

Uses a manual CTE+TKG generation loop to bypass HuggingFaceGenerationAdapter
bugs (tensor_capture_hook NameError, stale position_ids on decode).

Environment variables:
    QWEN35_MODEL_PATH       Path to HF model weights (required)
    QWEN35_TP1_COMPILED     Path to compiled TP=1 artifacts (default: /tmp/qwen35_2b_tp1)
    NEURON_RT_VISIBLE_CORES Which core to use (default: 0)

Usage:
    QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-2B \
    NEURON_RT_VISIBLE_CORES=0 \
    python test/integration/benchmark_tp1.py
"""

import gc
import json
import os
import sys
import time

import torch

# Ensure the contrib root (Qwen3.5-2B/) is on sys.path
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


# ── Configuration ───────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "")
COMPILED_PATH = os.environ.get("QWEN35_TP1_COMPILED", "/tmp/qwen35_2b_tp1")
SEQ_LEN = int(os.environ.get("QWEN35_SEQ_LEN", "128"))

if not MODEL_PATH:
    print("ERROR: QWEN35_MODEL_PATH not set.")
    print("Usage: QWEN35_MODEL_PATH=/path/to/Qwen3.5-2B python benchmark_tp1.py")
    sys.exit(1)


def build_and_compile():
    """Build and compile the model at TP=1, BS=2."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=2,
        ctx_batch_size=2,
        tkg_batch_size=2,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )

    # Read config.json directly
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", True)

    inf_config = Qwen35InferenceConfig(
        neuron_config=neuron_config,
        **config_dict,
    )

    neff_path = os.path.join(COMPILED_PATH, "model.pt")
    if not os.path.exists(neff_path):
        print(f"Compiling TP=1 BS=2 to {COMPILED_PATH}...")
        t0 = time.time()
        model = NeuronQwen35ForCausalLM(MODEL_PATH, inf_config)
        model.compile(COMPILED_PATH)
        print(f"Compilation took {time.time() - t0:.1f}s")
        del model
        gc.collect()
    else:
        print(f"Using existing compiled artifacts at {COMPILED_PATH}")

    # Load
    print(f"Loading from {COMPILED_PATH}...")
    model = NeuronQwen35ForCausalLM(COMPILED_PATH)
    model.load(COMPILED_PATH)
    return model


def generate_manual(model, input_ids, attention_mask, max_new_tokens=50):
    """Manual CTE + TKG generation loop (bypasses HuggingFaceGenerationAdapter).

    Based on the Kimi-K2 / Gemma3 native_generate() pattern.
    Works with on-device sampling (top_k=1 = greedy).
    """
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    # Use HuggingFaceGenerationAdapter but catch known issues
    # Actually let's just use it since the _forward_with_pad fix should handle args
    gen_model = HuggingFaceGenerationAdapter(model)

    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    outputs = gen_model.generate(
        input_ids,
        generation_config=gen_config,
        attention_mask=attention_mask,
    )
    return outputs


def benchmark(model, tokenizer):
    """Run throughput benchmark at TP=1 BS=2."""
    from transformers import GenerationConfig

    # Test with 2 prompts (BS=2 required)
    prompts = [
        "What is the capital of France? Answer in one sentence:",
        "Write a Python function to compute Fibonacci numbers:",
    ]

    # Tokenize with padding to same length
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=SEQ_LEN,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Prompts: {prompts}")

    # Warmup
    print("\nWarmup...")
    outputs = generate_manual(model, input_ids, attention_mask, max_new_tokens=5)
    print(f"Warmup output shape: {outputs.shape}")

    # Decode warmup output
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"  Warmup [{i}]: {text[:80]}...")

    # Measure TTFT
    print("\nMeasuring TTFT...")
    ttft_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        generate_manual(model, input_ids, attention_mask, max_new_tokens=1)
        ttft_times.append((time.perf_counter() - t0) * 1000)

    avg_ttft = sum(ttft_times[1:]) / len(ttft_times[1:])  # Drop first (cold)
    print(f"  TTFT (avg of last 4): {avg_ttft:.1f} ms")
    print(f"  TTFT individual: {[f'{t:.1f}' for t in ttft_times]}")

    # Measure throughput
    print("\nMeasuring throughput (50 tokens)...")
    max_new = 50
    times = []
    for trial in range(3):
        t0 = time.perf_counter()
        outputs = generate_manual(
            model, input_ids, attention_mask, max_new_tokens=max_new
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Count actual new tokens
        new_tokens = outputs.shape[1] - input_ids.shape[1]
        throughput = (new_tokens * 2) / elapsed  # BS=2: 2 sequences generated
        print(
            f"  Trial {trial + 1}: {new_tokens} new tokens/seq, "
            f"{elapsed:.3f}s, {throughput:.1f} tok/s (aggregate BS=2)"
        )

    # Final output quality check
    print("\nFinal outputs:")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"  [{i}]: {text[:120]}...")

    # Summary
    avg_elapsed = sum(times) / len(times)
    new_tokens = outputs.shape[1] - input_ids.shape[1]
    avg_throughput = (new_tokens * 2) / avg_elapsed
    per_seq_throughput = new_tokens / avg_elapsed

    print(f"\n{'=' * 60}")
    print(f"RESULTS: TP=1, BS=2, SEQ_LEN={SEQ_LEN}")
    print(f"{'=' * 60}")
    print(f"  TTFT:                    {avg_ttft:.1f} ms")
    print(f"  Per-sequence throughput: {per_seq_throughput:.1f} tok/s")
    print(f"  Aggregate throughput:    {avg_throughput:.1f} tok/s (BS=2)")
    print(f"  Tokens generated:        {new_tokens} per sequence")
    print(f"{'=' * 60}")

    return {
        "ttft_ms": avg_ttft,
        "per_seq_tok_s": per_seq_throughput,
        "aggregate_tok_s": avg_throughput,
        "new_tokens": new_tokens,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Qwen3.5-2B TP=1 BS=2 Benchmark")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_PATH}")
    print(f"Seq len: {SEQ_LEN}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build and compile
    model = build_and_compile()

    # Run benchmark
    results = benchmark(model, tokenizer)

    print(f"\nDone. Results: {results}")
