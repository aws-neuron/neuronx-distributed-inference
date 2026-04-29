#!/usr/bin/env python3
"""
Quick demo: Run Qwen3-Omni-30B-A3B-Instruct thinker text model on Neuron.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    NEURON_RT_VISIBLE_CORES=0-31 python run_demo.py \
        --model-path /home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct \
        --compiled-model-path /home/ubuntu/traced_model/Qwen3-Omni-30B-A3B-Instruct \
        --tp-degree 32 \
        --prompt "Hello, who are you?"
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs

sys.path.insert(0, str(Path(__file__).parent / "src"))
from modeling_qwen3_omni_moe import (
    NeuronQwen3OmniMoeForCausalLM,
    Qwen3OmniMoeInferenceConfig,
    load_qwen3_omni_thinker_text_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--tp-degree", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Hello, who are you?")
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"TP degree: {args.tp_degree}")

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_context_length=args.max_context_length,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={"top_k": args.top_k, "do_sample": False},
    )

    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(args.model_path),
    )

    model = NeuronQwen3OmniMoeForCausalLM(args.model_path, config)

    compiled_path = Path(args.compiled_model_path)
    if not compiled_path.exists():
        print("Compiling model (this may take several minutes)...")
        t0 = time.perf_counter()
        model.compile(args.compiled_model_path)
        print(f"Compilation took {time.perf_counter() - t0:.1f}s")

    print("Loading model to Neuron...")
    t0 = time.perf_counter()
    model.load(args.compiled_model_path)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [args.prompt] * args.batch_size

    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    t0 = time.perf_counter()
    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        do_sample=False,
        max_length=model.neuron_config.max_length,
    )
    elapsed = time.perf_counter() - t0

    for i, text in enumerate(output_tokens):
        print(f"\nOutput[{i}]: {text}")
    print(f"\nGeneration took {elapsed:.2f}s")


if __name__ == "__main__":
    main()
