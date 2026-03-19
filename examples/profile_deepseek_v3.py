"""
Minimal profiling script for DeepSeek V3 671B.
Designed to be wrapped by `neuron-profile inspect` for Phase 10.

Loads pre-compiled model from sharded checkpoints, runs a few
inference iterations (prefill + decode), then exits.

Usage:
  neuron-profile inspect -o /scratch/profiles/inspect_output -- \
      python examples/profile_deepseek_v3.py \
          --traced-model-path /scratch/deepseek_v3_traced
"""

import argparse
import time

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced-model-path", type=str, default="/scratch/deepseek_v3_traced")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of generation iterations for profiling")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Tokens to generate per iteration (keep small for profiling)")
    args = parser.parse_args()

    # Load compiled model
    print(f"Loading compiled model from {args.traced_model_path}...")
    t0 = time.time()
    model = NeuronDeepseekV3ForCausalLM(args.traced_model_path)
    model.load(args.traced_model_path)
    print(f"  Load time: {time.time() - t0:.1f}s")

    nc = model.neuron_config
    print(f"  tp={nc.tp_degree}, bs={nc.batch_size}, seq_len={nc.seq_len}")

    # Tokenizer + generation config
    tokenizer = AutoTokenizer.from_pretrained(args.traced_model_path)
    gen_config = GenerationConfig(
        do_sample=True, top_k=1, top_p=1.0, temperature=1.0,
        eos_token_id=1, pad_token_id=1,
    )

    prompt = "The capital of France is"
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    gen_model = HuggingFaceGenerationAdapter(model)

    # Warmup (1 iteration to prime everything)
    print("Warmup...")
    _ = gen_model.generate(
        inputs.input_ids, generation_config=gen_config,
        attention_mask=inputs.attention_mask, max_new_tokens=4,
    )
    model.reset()

    # Profiled iterations
    print(f"Running {args.num_iterations} profiled iterations ({args.max_new_tokens} tokens each)...")
    for i in range(args.num_iterations):
        t0 = time.time()
        outputs = gen_model.generate(
            inputs.input_ids, generation_config=gen_config,
            attention_mask=inputs.attention_mask, max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        print(f"  Iteration {i+1}: {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
        model.reset()

    print("Profiling script complete.")


if __name__ == "__main__":
    main()
