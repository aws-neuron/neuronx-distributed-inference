"""
Run NXDI native benchmark_sampling on DeepSeek V3 671B.

Requires: pre-compiled model at --traced-model-path (from generation_deepseek_v3.py).

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python examples/benchmark_deepseek_v3.py \
      --traced-model-path /scratch/deepseek_v3_traced \
      --num-runs 20
"""

import argparse
import time

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter


def main():
    parser = argparse.ArgumentParser(description="NXDI native benchmark for DeepSeek V3")
    parser.add_argument("--traced-model-path", type=str, default="/scratch/deepseek_v3_traced")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--benchmark-report-path", type=str, default="/tmp/deepseek_v3_benchmark_report.json")
    args = parser.parse_args()

    # --- Load compiled model ---
    print(f"\nLoading compiled model from {args.traced_model_path}...")
    t0 = time.time()
    model = NeuronDeepseekV3ForCausalLM(args.traced_model_path)
    model.load(args.traced_model_path)
    print(f"  Load time: {time.time() - t0:.1f}s")

    neuron_config = model.neuron_config
    print(f"  tp_degree={neuron_config.tp_degree}, batch_size={neuron_config.batch_size}, "
          f"seq_len={neuron_config.seq_len}, max_context_length={neuron_config.max_context_length}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.traced_model_path)

    # --- Generation config ---
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        eos_token_id=1,
        pad_token_id=1,
    )

    # --- Sanity generation ---
    print("\n--- Sanity generation ---")
    prompts = ["The capital of France is"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)

    print("  Warmup...")
    _ = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4,
    )

    print("  Generating (32 tokens)...")
    t0 = time.time()
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
    )
    gen_time = time.time() - t0
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    print(f"  Generated {new_tokens} tokens in {gen_time:.2f}s ({new_tokens/gen_time:.1f} tok/s)")
    print(f"  Output: {text}")

    # Reset model state before benchmark
    model.reset()

    # --- NXDI native benchmark ---
    print(f"\n--- NXDI benchmark_sampling (num_runs={args.num_runs}) ---")
    print(f"  Input length: {neuron_config.max_context_length // 2} (random tokens)")
    print(f"  Max length: {neuron_config.max_length}")
    print(f"  Report path: {args.benchmark_report_path}")

    t0 = time.time()
    report = benchmark_sampling(
        model,
        generation_config=generation_config,
        num_runs=args.num_runs,
        benchmark_report_path=args.benchmark_report_path,
    )
    bench_time = time.time() - t0
    print(f"\n  Total benchmark time: {bench_time:.1f}s")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  NXDI BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for component, metrics in report.items():
        if metrics is None:
            continue
        print(f"\n  {component}:")
        print(f"    p50:  {metrics['latency_ms_p50']:.2f} ms")
        print(f"    p90:  {metrics['latency_ms_p90']:.2f} ms")
        print(f"    p99:  {metrics['latency_ms_p99']:.2f} ms")
        print(f"    p100: {metrics['latency_ms_p100']:.2f} ms")
        print(f"    avg:  {metrics['latency_ms_avg']:.2f} ms")
        print(f"    throughput: {metrics['throughput']:.2f} tok/s")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
