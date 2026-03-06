#!/usr/bin/env python3
# coding=utf-8
"""Offline inference example for GLM-4.5 MoE using vLLM + NxDI backend.

Usage:
    python run_offline_inference.py \
        --model /path/to/GLM-4.5-Air \
        --tp-degree 32 \
        --seq-len 4096

Requires:
    - VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference
    - transformers>=4.56.0
    - vllm-neuron with NxDI backend support
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GLM-4.5 MoE offline inference via vLLM + NxDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the GLM-4.5 MoE HuggingFace checkpoint",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=32,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model dtype",
    )
    return parser.parse_args()


def build_neuron_config(args: argparse.Namespace) -> dict:
    """Build the NxDI neuron config dict for vLLM override."""
    return {
        "tp_degree": args.tp_degree,
        "moe_tp_degree": args.tp_degree,
        "moe_ep_degree": 1,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_context_length": args.seq_len,
        "torch_dtype": args.dtype,
        "fused_qkv": True,
        "flash_decoding_enabled": True,
        "on_device_sampling_config": {
            "dynamic": True,
            "global_topk": 64,
            "top_p": 1.0,
            "temperature": 1.0,
        },
    }


def main() -> None:
    """Run offline inference with GLM-4.5 MoE via vLLM."""
    args = parse_args()

    # Set backend
    os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

    try:
        import json

        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm is not installed. Please install the vllm-neuron fork.")
        sys.exit(1)

    neuron_config = build_neuron_config(args)

    print(f"\nLoading GLM-4.5 MoE model from: {args.model}")
    print(f"  tp_degree={args.tp_degree}, seq_len={args.seq_len}, dtype={args.dtype}")

    llm = LLM(
        model=args.model,
        max_model_len=args.seq_len,
        tensor_parallel_size=args.tp_degree,
        max_num_seqs=args.batch_size,
        override_neuron_config=neuron_config,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,  # greedy
    )

    prompts = [
        "Explain the mixture-of-experts architecture in one paragraph.",
        "What is the capital of France?",
        "Write a Python function to compute the Fibonacci sequence.",
    ]

    print(f"\nRunning inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 70)
    for i, output in enumerate(outputs):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[{i + 1}] Prompt: {prompt_text[:80]}...")
        print(f"    Output: {generated_text[:200]}")
    print("=" * 70)
    print(f"\nDone. Generated {len(outputs)} responses.")


if __name__ == "__main__":
    main()
