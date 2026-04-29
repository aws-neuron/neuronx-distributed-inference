#!/usr/bin/env python3
"""
Multimodal demo: Run Qwen3-Omni-30B-A3B-Instruct with vision+text on Neuron.

Usage (vision+text):
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    NEURON_RT_VISIBLE_CORES=0-31 python run_multimodal_demo.py \
        --model-path /home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct \
        --compiled-model-path /home/ubuntu/traced_model/Qwen3-Omni-multimodal \
        --image-path /path/to/image.jpg \
        --prompt "Describe this image."

Usage (text-only):
    NEURON_RT_VISIBLE_CORES=0-31 python run_multimodal_demo.py \
        --model-path /home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct \
        --compiled-model-path /home/ubuntu/traced_model/Qwen3-Omni-multimodal \
        --prompt "The capital of France is"
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from modeling_qwen3_omni import (
    NeuronQwen3OmniForCausalLM,
    Qwen3OmniInferenceConfig,
    Qwen3OmniMoeNeuronConfig,
    load_qwen3_omni_multimodal_config,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig, NeuronConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--vision-tp-degree", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-context-length", type=int, default=2048)
    parser.add_argument("--vision-seq-len", type=int, default=4096,
                        help="Max vision sequence length (patches)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to input image (optional, text-only if not provided)")
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Text TP degree: {args.tp_degree}")
    print(f"Vision TP degree: {args.vision_tp_degree}")
    print(f"Mode: {'vision+text' if args.image_path else 'text-only'}")

    # Text model neuron config (MoE)
    text_neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_context_length=args.max_context_length,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={"top_k": args.top_k, "do_sample": False},
        blockwise_matmul_config={"use_torch_block_wise": True},
    )

    # Vision model neuron config
    vision_neuron_config = NeuronConfig(
        tp_degree=args.vision_tp_degree,
        batch_size=1,
        seq_len=args.vision_seq_len,
        torch_dtype=torch.bfloat16,
    )

    config = Qwen3OmniInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_qwen3_omni_multimodal_config(args.model_path),
    )

    model = NeuronQwen3OmniForCausalLM(args.model_path, config)

    compiled_path = Path(args.compiled_model_path)
    if not compiled_path.exists():
        print("Compiling model (this may take a while)...")
        t0 = time.perf_counter()
        model.compile(args.compiled_model_path)
        print(f"Compilation took {time.perf_counter() - t0:.1f}s")

    print("Loading model to Neuron...")
    t0 = time.perf_counter()
    model.load(args.compiled_model_path)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Prepare inputs — limit max_pixels so raw patches fit vision_seq_len
    patch_size = 16
    max_pixels = args.vision_seq_len * (patch_size ** 2)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
        max_pixels=max_pixels, use_fast=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    images = None
    if args.image_path:
        from PIL import Image
        img = Image.open(args.image_path).convert("RGB")
        images = [[img]]

    input_ids, attention_mask, vision_inputs = NeuronQwen3OmniForCausalLM.prepare_input_args(
        prompts=[args.prompt],
        images=images,
        processor=processor,
        config=config,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Input shape: {input_ids.shape}")
    if vision_inputs:
        print(f"Vision inputs: {list(vision_inputs.keys())}")

    print("Generating...")
    t0 = time.perf_counter()

    # Use the HuggingFaceGenerationAdapter for generation
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    generation_model = HuggingFaceGenerationAdapter(model)

    outputs = generation_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        **vision_inputs,
    )

    elapsed = time.perf_counter() - t0

    for i in range(outputs.shape[0]):
        text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f"\nOutput[{i}]: {text}")
    print(f"\nGeneration took {elapsed:.2f}s")


if __name__ == "__main__":
    main()
