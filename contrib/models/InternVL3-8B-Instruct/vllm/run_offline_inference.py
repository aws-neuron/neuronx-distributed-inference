#!/usr/bin/env python3
"""Offline inference example for InternVL3-8B-Instruct with vLLM on Neuron."""

from vllm import LLM, SamplingParams


def main():
    model_path = "/mnt/models/InternVL3-8B-Instruct/"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        max_model_len=4096,
        max_num_seqs=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        additional_config={
            "override_neuron_config": {
                "vision_neuron_config": {"fused_qkv": True, "buckets": [1]},
            }
        },
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    # Text-only
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Output: {generated!r}")
        print()


if __name__ == "__main__":
    main()
