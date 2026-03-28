#!/usr/bin/env python3
"""
Validation script for Qwen2.5-Omni-7B NeuronX contrib (text thinker only).

Compiles the Neuron model, runs greedy decoding on test prompts,
then compares token-by-token against the HuggingFace reference model (CPU).

Usage:
    python validate.py --model-path /home/ubuntu/models/Qwen2.5-Omni-7B

    # Skip compilation if already compiled:
    python validate.py --model-path /home/ubuntu/models/Qwen2.5-Omni-7B --skip-compile
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from modeling_qwen2_5_omni import (
    Qwen2_5OmniInferenceConfig,
    NeuronQwen2_5OmniForCausalLM,
)

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


TEST_PROMPTS = [
    "The capital of France is",
    "In the year 2024, artificial intelligence",
    "The Pythagorean theorem states that",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
]

MAX_NEW_TOKENS = 30


def compile_neuron_model(model_path: str, compiled_path: str, tp_degree: int = 2):
    """Compile the Neuron model."""
    print(f"\n{'=' * 60}")
    print(f"COMPILING Neuron model")
    print(f"  Model: {model_path}")
    print(f"  Output: {compiled_path}")
    print(f"  TP: {tp_degree}, dtype: bfloat16")
    print(f"{'=' * 60}")

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=128,
        max_context_length=128,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
    )

    # Use custom from_pretrained which reads thinker_config.text_config
    config = Qwen2_5OmniInferenceConfig.from_pretrained(
        model_path,
        neuron_config=neuron_config,
    )

    model = NeuronQwen2_5OmniForCausalLM(model_path, config)

    start = time.time()
    model.compile(compiled_path)
    elapsed = time.time() - start
    print(f"\nCompilation completed in {elapsed:.1f}s")

    del model
    gc.collect()
    return elapsed


def load_neuron_model(model_path: str, compiled_path: str, tp_degree: int = 2):
    """Load a compiled Neuron model."""
    print(f"\nLoading Neuron model from {compiled_path}...")

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=128,
        max_context_length=128,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
    )

    # Use custom from_pretrained which reads thinker_config.text_config
    config = Qwen2_5OmniInferenceConfig.from_pretrained(
        model_path,
        neuron_config=neuron_config,
    )

    model = NeuronQwen2_5OmniForCausalLM(model_path, config)
    model.load(compiled_path)
    print("Neuron model loaded.")
    return model


def generate_neuron(model, input_ids, max_new_tokens: int):
    """Generate tokens using the Neuron model (greedy)."""
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        seq_len = generated.shape[1]
        # Standard 2D position_ids: [batch, seq_len]
        # The rotary embedding internally expands to 3D for M-RoPE
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated.shape[0], -1)

        with torch.no_grad():
            outputs = model(generated, position_ids=position_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() in (151643, 151645):
            break

    return generated


def generate_hf_reference(
    model_path: str, tokenizer, prompts: list, max_new_tokens: int
):
    """Generate reference outputs using HuggingFace thinker model (bfloat16, CPU).

    The Omni model is not in AutoModelForCausalLM, so we load
    Qwen2_5OmniThinkerForConditionalGeneration directly, which is the
    text-only thinker backbone that our Neuron contrib implements.
    """
    print(f"\n{'=' * 60}")
    print("Generating HuggingFace reference outputs (thinker, bfloat16)...")
    print(f"{'=' * 60}")

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    print("Loading HF thinker model (this may take a minute for 7B bfloat16)...")
    hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    hf_model.eval()
    print("HF thinker model loaded.")

    results = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        with torch.no_grad():
            output = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        gen_tokens = output[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        results[prompt] = {
            "tokens": gen_tokens,
            "text": gen_text,
        }
        print(f"  [{prompt[:40]}...] -> {gen_text[:60]}...")

    del hf_model
    gc.collect()
    return results


def validate_token_match(neuron_results: dict, hf_results: dict):
    """Compare Neuron vs HF token outputs."""
    print(f"\n{'=' * 60}")
    print("TOKEN MATCH COMPARISON")
    print(f"{'=' * 60}")

    total_tokens = 0
    matching_tokens = 0

    for prompt in neuron_results:
        if prompt not in hf_results:
            continue

        n_tokens = neuron_results[prompt]["tokens"]
        h_tokens = hf_results[prompt]["tokens"]

        compare_len = min(len(n_tokens), len(h_tokens))
        matches = sum(1 for i in range(compare_len) if n_tokens[i] == h_tokens[i])

        total_tokens += compare_len
        matching_tokens += matches

        pct = (matches / compare_len * 100) if compare_len > 0 else 0
        status = "PASS" if pct >= 95 else "WARN" if pct >= 80 else "FAIL"

        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Neuron: {neuron_results[prompt]['text'][:80]}...")
        print(f"  HF:     {hf_results[prompt]['text'][:80]}...")
        print(f"  Match:  {matches}/{compare_len} tokens ({pct:.1f}%) [{status}]")

        if pct < 100:
            for i in range(compare_len):
                if n_tokens[i] != h_tokens[i]:
                    print(
                        f"  First mismatch at position {i}: neuron={n_tokens[i]} vs hf={h_tokens[i]}"
                    )
                    break

    overall_pct = (matching_tokens / total_tokens * 100) if total_tokens > 0 else 0
    overall_status = (
        "PASS" if overall_pct >= 95 else "WARN" if overall_pct >= 80 else "FAIL"
    )

    print(f"\n{'=' * 60}")
    print(
        f"OVERALL: {matching_tokens}/{total_tokens} tokens ({overall_pct:.1f}%) [{overall_status}]"
    )
    print(f"{'=' * 60}")

    return overall_pct


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen2.5-Omni-7B on Neuron")
    parser.add_argument("--model-path", required=True, help="Path to HF model weights")
    parser.add_argument(
        "--compiled-path",
        default="/home/ubuntu/neuron_models/Qwen2.5-Omni-7B/",
        help="Path for compiled Neuron model",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=2, help="Tensor parallelism degree"
    )
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument(
        "--skip-reference", action="store_true", help="Skip HF reference comparison"
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"Qwen2.5-Omni-7B Validation")
    print(f"{'=' * 60}")
    print(f"Model path:    {args.model_path}")
    print(f"Compiled path: {args.compiled_path}")
    print(f"TP degree:     {args.tp_degree}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_compile:
        compile_time = compile_neuron_model(
            args.model_path, args.compiled_path, args.tp_degree
        )
    else:
        print("\nSkipping compilation (--skip-compile)")

    model = load_neuron_model(args.model_path, args.compiled_path, args.tp_degree)

    print(f"\n{'=' * 60}")
    print("Running Neuron inference...")
    print(f"{'=' * 60}")

    neuron_results = {}
    neuron_times = []

    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        start = time.time()
        output_ids = generate_neuron(model, input_ids, args.max_new_tokens)
        elapsed = time.time() - start

        gen_tokens = output_ids[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        neuron_results[prompt] = {
            "tokens": gen_tokens,
            "text": gen_text,
        }
        neuron_times.append(elapsed)

        n_gen = len(gen_tokens)
        tok_s = n_gen / elapsed if elapsed > 0 else 0
        print(
            f"  [{prompt[:40]}...] {n_gen} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)"
        )
        print(f"    -> {gen_text[:80]}...")

    del model
    gc.collect()

    if not args.skip_reference:
        hf_results = generate_hf_reference(
            args.model_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
        )
        overall_pct = validate_token_match(neuron_results, hf_results)
    else:
        print("\nSkipping HF reference comparison (--skip-reference)")
        overall_pct = None

    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    if not args.skip_compile:
        print(f"  Compile time:     {compile_time:.1f}s")
    avg_time = sum(neuron_times) / len(neuron_times) if neuron_times else 0
    print(f"  Avg gen time:     {avg_time:.2f}s ({args.max_new_tokens} tokens)")
    avg_toks = args.max_new_tokens / avg_time if avg_time > 0 else 0
    print(f"  Avg throughput:   {avg_toks:.1f} tok/s")
    if overall_pct is not None:
        print(f"  Token match:      {overall_pct:.1f}%")
        status = (
            "PASS" if overall_pct >= 95 else "WARN" if overall_pct >= 80 else "FAIL"
        )
        print(f"  Status:           {status}")
    print(f"{'=' * 60}")

    results_path = Path(args.compiled_path) / "validation_results.json"
    os.makedirs(args.compiled_path, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": "Qwen2.5-Omni-7B",
                "tp_degree": args.tp_degree,
                "max_new_tokens": args.max_new_tokens,
                "token_match_pct": overall_pct,
                "avg_throughput_toks": avg_toks,
                "neuron_results": {k: v["text"] for k, v in neuron_results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
