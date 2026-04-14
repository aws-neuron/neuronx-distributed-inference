"""
Benchmark latency for Granite 4.0-H-Small (V15 quadratic vs V16 NKI).

Measures:
- Prefill latency (first token)
- Decode latency (per-token, steady state)
- End-to-end generation throughput

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python benchmark_latency.py --model-dir /path/to/traced_model --version v16-nki
"""

import sys
import os
import time
import argparse
import torch
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/ubuntu/Granite4/neuronx-distributed-inference/src")

MODEL_PATH = "/home/ubuntu/Granite4/granite-4.0-h-small/"


def load_model(compiled_path):
    """Load a compiled Granite model."""
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from neuronx_distributed_inference.models.granite.modeling_granite import (
        NeuronGraniteForCausalLM,
        GraniteInferenceConfig,
    )

    neuron_config = MoENeuronConfig(
        tp_degree=4,
        batch_size=1,
        max_context_length=128,
        seq_len=2048,
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
    )

    config = GraniteInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    model = NeuronGraniteForCausalLM(MODEL_PATH, config)
    logger.info(f"Loading compiled model from {compiled_path}...")
    model.load(compiled_path)
    logger.info("Model loaded successfully.")
    return model


def benchmark_generation(
    model, tokenizer, prompt, max_new_tokens, num_warmup=3, num_runs=10
):
    """Benchmark prefill + decode latency for a single prompt."""
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    gen_model = HuggingFaceGenerationAdapter(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    # Warmup
    logger.info(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Timed runs
    logger.info(f"Running {num_runs} timed generations...")
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        outputs = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        end = time.perf_counter()

        total_time = end - start
        generated_tokens = outputs.shape[1] - prompt_len
        latencies.append(
            {
                "total_time_s": total_time,
                "generated_tokens": generated_tokens,
                "prompt_len": prompt_len,
            }
        )
        logger.info(f"  Run {i + 1}: {total_time:.3f}s, {generated_tokens} tokens")

    return latencies


def benchmark_prefill_only(model, tokenizer, prompts, num_warmup=3, num_runs=10):
    """Benchmark prefill latency (1 token generated = prefill + 1 decode step)."""
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    gen_model = HuggingFaceGenerationAdapter(model)
    results = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        # Warmup
        for _ in range(num_warmup):
            _ = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
            )
            end = time.perf_counter()
            times.append(end - start)

        results[prompt] = {
            "prompt_len": prompt_len,
            "mean_s": np.mean(times),
            "std_s": np.std(times),
            "min_s": np.min(times),
            "max_s": np.max(times),
            "times": times,
        }
        logger.info(
            f"  Prefill '{prompt[:30]}...' (len={prompt_len}): "
            f"{np.mean(times) * 1000:.1f} +/- {np.std(times) * 1000:.1f} ms"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Granite 4.0-H-Small latency"
    )
    parser.add_argument(
        "--model-dir", required=True, help="Path to compiled model directory"
    )
    parser.add_argument(
        "--version", default="unknown", help="Version label (e.g., v15, v16-nki)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50, help="Tokens to generate"
    )
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=10, help="Timed iterations")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = load_model(args.model_dir)

    print(f"\n{'=' * 70}")
    print(f" BENCHMARK: Granite 4.0-H-Small — {args.version}")
    print(f" Model dir: {args.model_dir}")
    print(f" Max new tokens: {args.max_new_tokens}")
    print(f" Warmup: {args.num_warmup}, Runs: {args.num_runs}")
    print(f"{'=' * 70}\n")

    # Test prompts of varying lengths
    prompts = [
        "The",  # ~1 token
        "The capital of France is",  # ~6 tokens
        "Explain the concept of artificial intelligence in simple terms.",  # ~10 tokens
        "Write a short Python function that calculates the Fibonacci sequence up to n numbers. Include type hints and a docstring.",  # ~20 tokens
    ]

    # ---- Prefill Benchmark ----
    print(f"\n--- Prefill Latency (1 new token) ---")
    prefill_results = benchmark_prefill_only(
        model,
        tokenizer,
        prompts,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )
    print(
        f"\n{'Prompt':<50} {'Len':>4} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10}"
    )
    print("-" * 90)
    for prompt, res in prefill_results.items():
        print(
            f"{prompt[:48]:<50} {res['prompt_len']:>4} {res['mean_s'] * 1000:>10.1f} "
            f"{res['std_s'] * 1000:>10.1f} {res['min_s'] * 1000:>10.1f}"
        )

    # ---- Generation Benchmark ----
    gen_prompt = "The capital of France is"
    print(f"\n--- Generation Latency ({args.max_new_tokens} new tokens) ---")
    gen_results = benchmark_generation(
        model,
        tokenizer,
        gen_prompt,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    total_times = [r["total_time_s"] for r in gen_results]
    gen_tokens = [r["generated_tokens"] for r in gen_results]
    per_token = [r["total_time_s"] / r["generated_tokens"] for r in gen_results]
    throughput = [r["generated_tokens"] / r["total_time_s"] for r in gen_results]

    print(f"\nPrompt: '{gen_prompt}' -> {gen_tokens[0]} tokens generated")
    print(
        f"  Total time:     {np.mean(total_times) * 1000:.1f} +/- {np.std(total_times) * 1000:.1f} ms"
    )
    print(
        f"  Per-token:      {np.mean(per_token) * 1000:.1f} +/- {np.std(per_token) * 1000:.1f} ms"
    )
    print(
        f"  Throughput:     {np.mean(throughput):.1f} +/- {np.std(throughput):.1f} tokens/s"
    )
    print(f"  Min total:      {np.min(total_times) * 1000:.1f} ms")
    print(f"  Max total:      {np.max(total_times) * 1000:.1f} ms")

    # ---- Longer generation benchmark ----
    gen_prompt_long = "Explain the concept of artificial intelligence in simple terms."
    max_long = 100
    print(f"\n--- Long Generation ({max_long} new tokens) ---")
    long_results = benchmark_generation(
        model,
        tokenizer,
        gen_prompt_long,
        max_new_tokens=max_long,
        num_warmup=args.num_warmup,
        num_runs=5,  # fewer runs for longer generation
    )

    total_times_l = [r["total_time_s"] for r in long_results]
    gen_tokens_l = [r["generated_tokens"] for r in long_results]
    per_token_l = [r["total_time_s"] / r["generated_tokens"] for r in long_results]
    throughput_l = [r["generated_tokens"] / r["total_time_s"] for r in long_results]

    print(
        f"\nPrompt: '{gen_prompt_long[:40]}...' -> {gen_tokens_l[0]} tokens generated"
    )
    print(
        f"  Total time:     {np.mean(total_times_l) * 1000:.1f} +/- {np.std(total_times_l) * 1000:.1f} ms"
    )
    print(
        f"  Per-token:      {np.mean(per_token_l) * 1000:.1f} +/- {np.std(per_token_l) * 1000:.1f} ms"
    )
    print(
        f"  Throughput:     {np.mean(throughput_l):.1f} +/- {np.std(throughput_l):.1f} tokens/s"
    )

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f" SUMMARY: {args.version}")
    print(f"{'=' * 70}")
    print(
        f"  Prefill (6 tokens):    {prefill_results[prompts[1]]['mean_s'] * 1000:.1f} ms"
    )
    print(
        f"  Prefill (20 tokens):   {prefill_results[prompts[3]]['mean_s'] * 1000:.1f} ms"
    )
    print(f"  Decode per-token:      {np.mean(per_token) * 1000:.1f} ms")
    print(f"  Decode throughput:     {np.mean(throughput):.1f} tok/s")
    print(f"  Long gen per-token:    {np.mean(per_token_l) * 1000:.1f} ms")
    print(f"  Long gen throughput:   {np.mean(throughput_l):.1f} tok/s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
