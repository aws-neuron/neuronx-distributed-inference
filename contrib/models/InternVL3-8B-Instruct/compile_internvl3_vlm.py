#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compile InternVL3-8B-Instruct full VLM on Neuron.

This compiles three NEFFs:
1. Vision encoder: InternViT-300M + pixel shuffle + MLP projector
2. Text CTE: Qwen2.5-7B context encoding (with vision embedding injection)
3. Text TKG: Qwen2.5-7B token generation

Usage:
    python compile_internvl3_vlm.py [--text-only] [--vision-only]

Target: trn2.3xlarge LNC=2 TP=4
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_PATH = "/mnt/models/neuron_models/InternVL3-8B-VLM/"


def create_config():
    """Create InternVL3 VLM inference config with text + vision NeuronConfigs."""

    # Text NeuronConfig: TP=4 on trn2.3xlarge LNC=2
    text_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        seq_len=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
    )

    # Vision NeuronConfig: Must match text TP degree (NxDI requirement)
    # Vision encoder is small; weights are replicated across TP ranks.
    # Bucket = [1] (one image at a time)
    vision_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        seq_len=256,  # 256 vision tokens after pixel shuffle
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        buckets=[1],  # number of images
        fused_qkv=True,  # vision encoder has fused QKV weights
        save_sharded_checkpoint=True,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        MODEL_PATH,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    return config


def compile_and_test():
    """Compile full VLM and run smoke tests."""
    print("=" * 60)
    print("InternVL3-8B-Instruct: Full VLM Compilation")
    print("=" * 60)

    config = create_config()

    print(f"\nModel path: {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_PATH}")
    print(f"\n--- Text Config ---")
    print(f"  TP degree: {config.text_config.neuron_config.tp_degree}")
    print(f"  Seq len: {config.text_config.neuron_config.seq_len}")
    print(f"  Batch size: {config.text_config.neuron_config.max_batch_size}")
    print(f"  hidden_size: {config.text_config.hidden_size}")
    print(f"  num_hidden_layers: {config.text_config.num_hidden_layers}")
    print(f"  vocab_size: {config.text_config.vocab_size}")
    print(f"\n--- Vision Config ---")
    print(f"  TP degree: {config.vision_config.neuron_config.tp_degree}")
    print(f"  Buckets: {config.vision_config.neuron_config.buckets}")

    # Create model
    print("\n--- Creating model ---")
    model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)

    # Compile
    print("\n--- Compiling (text + vision) ---")
    start = time.time()
    model.compile(COMPILED_PATH)
    elapsed = time.time() - start
    print(f"\nCompilation completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Load
    print("\n--- Loading compiled model ---")
    start = time.time()
    model.load(COMPILED_PATH)
    elapsed = time.time() - start
    print(f"Load completed in {elapsed:.1f}s")

    # Smoke test 1: Text-only
    print("\n--- Smoke test: text-only ---")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    seq_ids = torch.zeros(1, dtype=torch.int32)

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )
        logits = outputs.logits
        top5 = torch.topk(logits[0, -1].float(), 5)
        print("Top-5 next tokens:")
        for t, v in zip(top5.indices, top5.values):
            print(
                f"  {tokenizer.decode([t.item()])!r} (id={t.item()}, logit={v.item():.4f})"
            )

    # Smoke test 2: Multimodal (synthetic image)
    print("\n--- Smoke test: multimodal ---")
    # Build input with 256 <IMG_CONTEXT> tokens
    IMG_CONTEXT_ID = 151667
    IMG_START_ID = 151665
    IMG_END_ID = 151666

    text_before = "Describe this image:"
    text_before_ids = tokenizer(text_before, return_tensors="pt").input_ids[0]

    # Construct: <text> <img> <IMG_CONTEXT>*256 </img>
    img_tokens = torch.full((256,), IMG_CONTEXT_ID, dtype=torch.long)
    full_ids = torch.cat(
        [
            text_before_ids,
            torch.tensor([IMG_START_ID]),
            img_tokens,
            torch.tensor([IMG_END_ID]),
        ]
    ).unsqueeze(0)

    full_seq_len = full_ids.shape[-1]
    full_position_ids = torch.arange(full_seq_len, dtype=torch.int32).unsqueeze(0)
    full_seq_ids = torch.zeros(1, dtype=torch.int32)

    # Synthetic pixel values (random noise as placeholder)
    pixel_values = torch.randn(1, 3, 448, 448)

    print(f"Input IDs shape: {full_ids.shape}")
    print(f"Pixel values shape: {pixel_values.shape}")
    print(
        f"Number of <IMG_CONTEXT> tokens: {(full_ids == IMG_CONTEXT_ID).sum().item()}"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            position_ids=full_position_ids,
            seq_ids=full_seq_ids,
            pixel_values=pixel_values,
        )
        logits = outputs.logits
        top5 = torch.topk(logits[0, -1].float(), 5)
        print("Top-5 next tokens (multimodal):")
        for t, v in zip(top5.indices, top5.values):
            print(
                f"  {tokenizer.decode([t.item()])!r} (id={t.item()}, logit={v.item():.4f})"
            )

    print("\n" + "=" * 60)
    print("SUCCESS: InternVL3 full VLM compiled and running on Neuron")
    print("=" * 60)


if __name__ == "__main__":
    compile_and_test()
