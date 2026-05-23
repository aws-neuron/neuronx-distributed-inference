#!/usr/bin/env python3
"""Compile and test GLM-4.7-Flash with FP8 quantized MoE expert weights.

This script:
  1. Quantizes expert weights from BF16 to FP8 E4M3 (if not already done)
  2. Compiles the model with quantized=True (MoEFusedTKG path)
  3. Loads and runs inference to validate correctness
  4. Benchmarks TPOT for FP8 vs BF16 comparison

Usage:
  # Full pipeline (quantize + compile + test):
  python compile_fp8.py --quantize --compile --test

  # Just compile (quantized checkpoint already exists):
  python compile_fp8.py --compile

  # Just test (compiled model already exists):
  python compile_fp8.py --test

  # Quick benchmark (existing compiled model):
  python compile_fp8.py --benchmark
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "/mnt/models/GLM-4.7-Flash-contrib")
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"
os.environ["UNSAFE_FP8FNCAST"] = "1"

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from src.modeling_glm4_moe_lite import (
    Glm4MoeLiteGenerationAdapter,
    Glm4MoeLiteInferenceConfig,
    NeuronGlm4MoeLiteForCausalLM,
)

# Register glm4_moe_lite config (not in transformers registry)
try:
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    class Glm4MoeLiteConfig(Glm4MoeConfig):
        model_type = "glm4_moe_lite"

    AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig)
except Exception:
    pass  # Already registered or glm4_moe not available

# Paths
MODEL_PATH = "/mnt/models/GLM-4.7-Flash"
QUANTIZED_PATH = "/mnt/models/GLM-4.7-Flash-FP8"
COMPILED_FP8_PATH = "/mnt/models/compiled_glm4_fp8_sob"

# Config
BATCH_SIZE = 4
CTX_BATCH_SIZE = 1  # CTE processes 1 prompt at a time (eliminates left-padding issues)
SEQ_LEN = 16384
TP_DEGREE = 4

# Bucketing config: CTE bucket sizes for short-prompt TTFT optimization
# Each bucket compiles a separate NEFF, so more buckets = longer compile time
# With 4 CTE buckets: compile time ~60-80 min (vs ~20 min unbucketed)
ENABLE_BUCKETING = True
CTE_BUCKETS = [128, 512, 2048, 4096, 8192, 16384]
# TKG buckets: single bucket for maximum compiler optimization of the TKG NEFF.
# Multiple TKG buckets cause massive TPOT regression (6.8x) due to bucket switching overhead.
TKG_BUCKETS = [16384]


def step_quantize():
    """Step 1: Quantize expert weights to FP8."""
    print("\n" + "=" * 70)
    print("STEP 1: Quantize Expert Weights (BF16 -> FP8 E4M3)")
    print("=" * 70)

    if os.path.exists(os.path.join(QUANTIZED_PATH, "model.safetensors.index.json")):
        print(f"  Quantized checkpoint already exists at {QUANTIZED_PATH}")
        print("  Skipping quantization. Use --force-quantize to redo.")
        return

    # Run the quantization script
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "/mnt/models/GLM-4.7-Flash-contrib/scripts/quantize_experts_fp8.py",
            "--model-path",
            MODEL_PATH,
            "--output-path",
            QUANTIZED_PATH,
            "--tp-degree",
            str(TP_DEGREE),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed with return code {result.returncode}")


def step_compile():
    """Step 2: Compile model with FP8 quantized weights."""
    print("\n" + "=" * 70)
    print("STEP 2: Compile Model (FP8 Quantized, MoEFusedTKG)")
    print("=" * 70)

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=ENABLE_BUCKETING,
        context_encoding_buckets=CTE_BUCKETS if ENABLE_BUCKETING else None,
        token_generation_buckets=TKG_BUCKETS if ENABLE_BUCKETING else None,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        # Enable continuous batching for proper KV cache indexing with ctx_batch_size=1
        is_continuous_batching=True,
        # FP8 quantization config
        quantized=True,
        quantization_type="expert_wise_per_channel_symmetric",
        quantization_dtype="f8e4m3",
        quantized_checkpoints_path=QUANTIZED_PATH,
        modules_to_not_convert=[
            "lm_head",
            "embed_tokens",
            "self_attn",
            "norm",
            "layers.0.mlp",
            "shared_experts",
            "router",
        ],
        # Use MoEFusedTKG for FP8 routed experts (shared experts handled separately)
        moe_fused_nki_kernel_enabled=True,
    )

    inf_config = Glm4MoeLiteInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    print(f"\nConfig:")
    print(f"  TP degree: {TP_DEGREE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  CTE batch size: {CTX_BATCH_SIZE}")
    print(f"  TKG batch size: {BATCH_SIZE}")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Dtype: bfloat16 (FP8 experts)")
    print(f"  LNC: 2")
    print(f"  Bucketing: {ENABLE_BUCKETING}")
    if ENABLE_BUCKETING:
        print(f"  CTE buckets: {CTE_BUCKETS}")
        print(f"  TKG buckets: {TKG_BUCKETS}")
    print(f"  Continuous batching: True")
    print(f"  Quantized: True")
    print(f"  Quantization type: expert_wise_per_channel_symmetric")
    print(f"  Quantized checkpoint: {QUANTIZED_PATH}")
    print(f"  MoE kernel: MoEFusedTKG (FP8 routed) + separate shared expert (BF16)")
    print(f"\nCompiling to: {COMPILED_FP8_PATH}")

    os.makedirs(COMPILED_FP8_PATH, exist_ok=True)

    t0 = time.time()
    model = NeuronGlm4MoeLiteForCausalLM(MODEL_PATH, inf_config)
    model.compile(COMPILED_FP8_PATH)
    compile_time = time.time() - t0

    print(
        f"\nCompilation complete in {compile_time:.1f}s ({compile_time / 60:.1f} min)"
    )
    print(f"Artifacts saved to: {COMPILED_FP8_PATH}")
    return compile_time


def step_test():
    """Step 3: Load and run inference to validate correctness."""
    print("\n" + "=" * 70)
    print("STEP 3: Test Inference (FP8 Quantized)")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Must use right-padding with ctx_batch_size=1

    # Load model
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=ENABLE_BUCKETING,
        context_encoding_buckets=CTE_BUCKETS if ENABLE_BUCKETING else None,
        token_generation_buckets=TKG_BUCKETS if ENABLE_BUCKETING else None,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        is_continuous_batching=True,
        quantized=True,
        quantization_type="expert_wise_per_channel_symmetric",
        quantization_dtype="f8e4m3",
        quantized_checkpoints_path=QUANTIZED_PATH,
        modules_to_not_convert=[
            "lm_head",
            "embed_tokens",
            "self_attn",
            "norm",
            "layers.0.mlp",
            "shared_experts",
            "router",
        ],
        moe_fused_nki_kernel_enabled=True,
    )

    inf_config = Glm4MoeLiteInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    print("  Loading compiled model...")
    t0 = time.time()
    model = NeuronGlm4MoeLiteForCausalLM(COMPILED_FP8_PATH, inf_config)
    model.load(COMPILED_FP8_PATH)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    gen_model = Glm4MoeLiteGenerationAdapter(model)
    gen_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Test prompts — with ctx_batch_size=1, CTE processes one sequence at a time
    # so left-padding between sequences is no longer an issue. We can now pass
    # mixed-length prompts in a single batch.
    test_prompts = [
        "The capital of France is",
        "In machine learning, a transformer model works by",
        "The square root of 144 is",
        "Python is a programming language known for",
    ]

    # Test 1: Mixed-length batch (the main test for ctx_batch_size=1 fixing left-padding)
    print(f"\n  TEST 1: Mixed-length batch (BS={BATCH_SIZE}, ctx_bs={CTX_BATCH_SIZE})")
    print("  This tests whether ctx_batch_size=1 fixes left-padding for mixed lengths.")

    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True)
    print(f"  Input shape: {inputs.input_ids.shape}")
    print(f"  Pad token positions per sequence:")
    for i, mask in enumerate(inputs.attention_mask):
        n_pad = (mask == 0).sum().item()
        print(f"    [{i}] '{test_prompts[i][:40]}...' -> {n_pad} pad tokens")

    t0 = time.time()
    outputs = gen_model.generate(
        inputs.input_ids,
        generation_config=gen_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
    )
    gen_time = time.time() - t0

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated = decoded

    print(f"\n  Generation complete in {gen_time:.2f}s")
    print("\n  --- Outputs ---")
    for i, (prompt, output) in enumerate(zip(test_prompts, generated)):
        generated_part = output[len(prompt) :]
        print(f"  [{i}] {prompt}")
        print(f"      -> {generated_part[:100]}...")
        print()

    # Validate outputs are non-empty and coherent (basic sanity)
    all_valid = True
    for i, output in enumerate(generated):
        if len(output) <= len(test_prompts[i]):
            print(f"  WARNING: Output {i} has no generated tokens!")
            all_valid = False
        # Check for repetitive garbage patterns
        gen_part = output[len(test_prompts[i]) :]
        if len(gen_part) > 10:
            # Check for excessive repetition (same char/word repeated)
            chars = set(gen_part[:20])
            if len(chars) <= 3:
                print(f"  WARNING: Output {i} appears to be repetitive garbage!")
                all_valid = False

    if all_valid:
        print("  PASS: All outputs generated successfully")
    else:
        print("  FAIL: Some outputs are empty or garbage")

    return all_valid


def step_benchmark():
    """Step 4: Benchmark TPOT with FP8."""
    print("\n" + "=" * 70)
    print("STEP 4: TPOT Benchmark (FP8 Quantized)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Must use right-padding with ctx_batch_size=1

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=ENABLE_BUCKETING,
        context_encoding_buckets=CTE_BUCKETS if ENABLE_BUCKETING else None,
        token_generation_buckets=TKG_BUCKETS if ENABLE_BUCKETING else None,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        is_continuous_batching=True,
        quantized=True,
        quantization_type="expert_wise_per_channel_symmetric",
        quantization_dtype="f8e4m3",
        quantized_checkpoints_path=QUANTIZED_PATH,
        modules_to_not_convert=[
            "lm_head",
            "embed_tokens",
            "self_attn",
            "norm",
            "layers.0.mlp",
            "shared_experts",
            "router",
        ],
        moe_fused_nki_kernel_enabled=True,
    )

    inf_config = Glm4MoeLiteInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    print("  Loading compiled model...")
    model = NeuronGlm4MoeLiteForCausalLM(COMPILED_FP8_PATH, inf_config)
    model.load(COMPILED_FP8_PATH)

    gen_model = Glm4MoeLiteGenerationAdapter(model)
    gen_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Warmup
    print("  Warming up (5 iterations)...")
    warmup_text = "The quick brown fox " * 10
    warmup_inputs = tokenizer(
        [warmup_text] * BATCH_SIZE, return_tensors="pt", padding=True
    )
    for _ in range(5):
        gen_model.generate(
            warmup_inputs.input_ids,
            generation_config=gen_config,
            attention_mask=warmup_inputs.attention_mask,
            max_new_tokens=10,
        )

    # Measure TPOT: generate 128 tokens, measure E2E time
    # TPOT ≈ (E2E - TTFT) / (n_tokens - 1)
    print("\n  Measuring TPOT (128 in, 128 out)...")
    prompt = "In the field of quantum computing, " * 8  # ~128 tokens
    inputs = tokenizer(
        [prompt] * BATCH_SIZE,
        return_tensors="pt",
        padding=True,
        max_length=128,
        truncation=True,
    )

    # Measure TTFT (1 token)
    ttft_times = []
    for _ in range(10):
        t0 = time.time()
        gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1,
        )
        ttft_times.append(time.time() - t0)

    # Measure E2E (128 tokens)
    e2e_times = []
    n_tokens_list = []
    for _ in range(10):
        t0 = time.time()
        outputs = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,
        )
        e2e_times.append(time.time() - t0)
        n_tokens_list.append(outputs.shape[1] - inputs.input_ids.shape[1])

    # Calculate metrics
    avg_ttft = sum(ttft_times) / len(ttft_times)
    avg_e2e = sum(e2e_times) / len(e2e_times)
    avg_n_tokens = sum(n_tokens_list) / len(n_tokens_list)
    avg_tpot = (avg_e2e - avg_ttft) / max(avg_n_tokens - 1, 1)
    throughput = BATCH_SIZE / avg_tpot  # Total tok/s

    print(f"\n  Results (FP8, BS={BATCH_SIZE}, 128in/128out):")
    print(f"    TTFT:       {avg_ttft * 1000:.1f} ms")
    print(f"    TPOT:       {avg_tpot * 1000:.1f} ms")
    print(f"    Throughput: {throughput:.1f} tok/s (batch)")
    print(f"    E2E:        {avg_e2e * 1000:.0f} ms")
    print(f"    Tokens:     {avg_n_tokens:.0f}")

    # Compare with BF16 baseline (from previous benchmarks)
    bf16_tpot_ms = 419.0  # BS=16 from formal benchmark (adjusted for BS=4)
    # BS=4 BF16 TPOT from formal_benchmark_bs4.json
    print(f"\n  Comparison with BF16 baseline:")
    print(f"    BF16 TPOT (BS=16): 419.0 ms → {BATCH_SIZE * 1000 / 419.0:.1f} tok/s")
    print(
        f"    FP8  TPOT (BS={BATCH_SIZE}): {avg_tpot * 1000:.1f} ms → {throughput:.1f} tok/s"
    )

    return {
        "ttft_ms": avg_ttft * 1000,
        "tpot_ms": avg_tpot * 1000,
        "throughput_tok_s": throughput,
        "e2e_ms": avg_e2e * 1000,
        "batch_size": BATCH_SIZE,
        "dtype": "fp8_e4m3_experts",
    }


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4.7-Flash FP8 compilation and testing"
    )
    parser.add_argument("--quantize", action="store_true", help="Run quantization step")
    parser.add_argument("--compile", action="store_true", help="Run compilation step")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--benchmark", action="store_true", help="Run TPOT benchmark")
    parser.add_argument(
        "--force-quantize",
        action="store_true",
        help="Force re-quantization even if checkpoint exists",
    )
    parser.add_argument(
        "--no-bucketing",
        action="store_true",
        help="Disable bucketing (single 4096 CTE bucket, faster compile)",
    )
    parser.add_argument(
        "--ctx-batch-size",
        type=int,
        default=None,
        help="Override CTE batch size (default: 1). Use 4 to match legacy behavior.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override TKG batch size (default: 4). Adjusts compiled model path.",
    )
    parser.add_argument(
        "--max-cte-bucket",
        type=int,
        default=None,
        help="Maximum CTE bucket size (default: 16384). Reduce for larger BS to avoid CTE OOM.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override SEQ_LEN (default: 16384). Reduces KV cache size for larger BS.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps (quantize + compile + test + benchmark)",
    )
    args = parser.parse_args()

    # Apply --no-bucketing override
    global \
        ENABLE_BUCKETING, \
        CTX_BATCH_SIZE, \
        COMPILED_FP8_PATH, \
        BATCH_SIZE, \
        CTE_BUCKETS, \
        SEQ_LEN, \
        TKG_BUCKETS
    if args.no_bucketing:
        ENABLE_BUCKETING = False
    if args.ctx_batch_size is not None:
        CTX_BATCH_SIZE = args.ctx_batch_size
        if CTX_BATCH_SIZE != 1:
            # Use different output path for non-default ctx_batch_size
            COMPILED_FP8_PATH = (
                f"/mnt/models/compiled_glm4_fp8_bucketed_ctx{CTX_BATCH_SIZE}"
            )
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
        COMPILED_FP8_PATH = f"/mnt/models/compiled_glm4_fp8_bs{BATCH_SIZE}"
    if args.max_cte_bucket is not None:
        CTE_BUCKETS = [b for b in CTE_BUCKETS if b <= args.max_cte_bucket]
    if args.seq_len is not None:
        SEQ_LEN = args.seq_len
        CTE_BUCKETS = [b for b in CTE_BUCKETS if b <= SEQ_LEN]
        TKG_BUCKETS = [SEQ_LEN]
        if args.batch_size is not None:
            COMPILED_FP8_PATH = (
                f"/mnt/models/compiled_glm4_fp8_bs{BATCH_SIZE}_seq{SEQ_LEN}"
            )
        else:
            COMPILED_FP8_PATH = f"/mnt/models/compiled_glm4_fp8_seq{SEQ_LEN}"

    if args.all:
        args.quantize = args.compile = args.test = args.benchmark = True

    if not any([args.quantize, args.compile, args.test, args.benchmark]):
        parser.print_help()
        print(
            "\nPlease specify at least one step: --quantize, --compile, --test, --benchmark, or --all"
        )
        sys.exit(1)

    print("=" * 70)
    print("GLM-4.7-Flash FP8 Expert Quantization Pipeline")
    print(f"  Model:     {MODEL_PATH}")
    print(f"  Quantized: {QUANTIZED_PATH}")
    print(f"  Compiled:  {COMPILED_FP8_PATH}")
    print(
        f"  Config:    BS={BATCH_SIZE}, CTX_BS={CTX_BATCH_SIZE}, SEQ={SEQ_LEN}, TP={TP_DEGREE}, LNC=2"
    )
    print(f"  Bucketing: {ENABLE_BUCKETING}")
    if ENABLE_BUCKETING:
        print(f"  CTE buckets: {CTE_BUCKETS}")
        print(f"  TKG buckets: {TKG_BUCKETS}")
    print("=" * 70)

    if args.quantize:
        step_quantize()

    if args.compile:
        step_compile()

    if args.test:
        step_test()

    if args.benchmark:
        step_benchmark()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
