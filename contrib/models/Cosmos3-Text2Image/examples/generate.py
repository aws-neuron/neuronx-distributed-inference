#!/usr/bin/env python3
"""
Generate images with Cosmos3 on Neuron.

Supports both Cosmos3-Nano (16B) and Cosmos3-Super-Text2Image (65B).

Usage:
    # Generate with Nano:
    python generate.py \
        --model-path /path/to/Cosmos3-Nano \
        --compiled-path /path/to/compiled \
        --vae-path /path/to/vae_decoder.pt \
        --prompt "A cat sitting on a windowsill" \
        --output generated.png

    # Generate with Super:
    python generate.py \
        --model-path /path/to/Cosmos3-Super-Text2Image \
        --compiled-path /path/to/compiled_super \
        --vae-path /path/to/vae_decoder.pt \
        --tp 8 \
        --prompt "A majestic mountain at sunrise" \
        --output generated.png

Environment:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
"""

import argparse
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/tmp/neuron_cache")

import torch
import torch_neuronx
from modeling_cosmos3 import (
    Cosmos3BackboneInferenceConfig,
    NeuronCosmos3BackboneApplication,
)
from pipeline import (
    build_position_ids,
    denoise,
    denormalize_latents,
    patchify,
    tokenize_prompt,
)
from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoTokenizer
from diffusers import UniPCMultistepScheduler
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Cosmos3 on Neuron"
    )
    parser.add_argument("--model-path", required=True, help="Path to HF model weights")
    parser.add_argument(
        "--compiled-path", required=True, help="Path to compiled backbone"
    )
    parser.add_argument(
        "--vae-path", required=True, help="Path to compiled VAE decoder (.pt)"
    )
    parser.add_argument(
        "--tp", type=int, default=4, help="TP degree (4 for Nano, 8 for Super)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean",
        help="Text prompt",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="", help="Negative prompt"
    )
    parser.add_argument("--steps", type=int, default=35, help="Denoising steps")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default="generated.png", help="Output image path"
    )
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--max-text-len", type=int, default=256, help="Max text tokens")
    parser.add_argument(
        "--cfg-parallel",
        action="store_true",
        help="Use CFG-parallel mode (backbone compiled with --cfg-parallel)",
    )
    args = parser.parse_args()

    MAX_TEXT = args.max_text_len
    NUM_VIS = (args.height // 32) * (args.width // 32)  # patches for 512x512 = 256

    # Detect model variant from hidden_size
    config_path = os.path.join(args.model_path, "transformer", "config.json")
    with open(config_path) as f:
        model_cfg = json.load(f)

    hidden_size = model_cfg.get("hidden_size", 4096)
    is_super = hidden_size >= 5120

    print("=" * 60)
    print(
        f"Cosmos3 {'Super' if is_super else 'Nano'} Image Generation on Neuron (TP={args.tp})"
    )
    print("=" * 60)

    # --- Load backbone ---
    print(
        f"\n[1/4] Loading backbone ({model_cfg.get('num_hidden_layers', 36)} layers, TP={args.tp})..."
    )
    t0 = time.time()
    neuron_config = NeuronConfig(
        tp_degree=args.tp, world_size=args.tp, torch_dtype=torch.bfloat16
    )
    config = Cosmos3BackboneInferenceConfig(
        neuron_config=neuron_config,
        cfg_parallel_enabled=args.cfg_parallel,
        hidden_size=hidden_size,
        intermediate_size=model_cfg.get("intermediate_size", 12288),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 36),
        num_attention_heads=model_cfg.get("num_attention_heads", 32),
        num_key_value_heads=model_cfg.get("num_key_value_heads", 8),
        head_dim=128,
        vocab_size=model_cfg.get("vocab_size", 151936),
        patch_channels=192,
        latent_channels=48,
        rope_theta=model_cfg.get("rope_theta", 5000000.0),
        mrope_section=model_cfg.get("mrope_section", [24, 20, 20]),
    )
    config.max_text_len = MAX_TEXT
    config.num_vision_patches = NUM_VIS

    transformer_path = os.path.join(args.model_path, "transformer")
    app = NeuronCosmos3BackboneApplication(model_path=transformer_path, config=config)
    app.load(args.compiled_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Load VAE ---
    print("\n[2/4] Loading VAE...")
    t0 = time.time()
    vae = torch.jit.load(args.vae_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Load tokenizer ---
    print("\n[3/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # --- Tokenize ---
    print(f"\nPrompt: '{args.prompt}'")
    cond_ids, cond_len = tokenize_prompt(
        tokenizer, args.prompt, height=args.height, width=args.width, max_len=MAX_TEXT
    )
    uncond_ids, uncond_len = tokenize_prompt(
        tokenizer,
        "",
        height=args.height,
        width=args.width,
        max_len=MAX_TEXT,
        negative=True,
    )
    print(f"  Cond tokens: {cond_len}, Uncond tokens: {uncond_len}")

    # --- Position IDs ---
    T = 1  # single frame for images
    pH = args.height // 32  # 512 -> 16
    pW = args.width // 32

    cond_pos = build_position_ids(MAX_TEXT, cond_len, T, pH, pW)
    uncond_pos = build_position_ids(MAX_TEXT, uncond_len, T, pH, pW)

    # --- Warmup ---
    print("\n[4/4] Warming up backbone (both CFG paths)...")
    t0 = time.time()
    if args.cfg_parallel:
        dummy_patches = torch.randn(2, NUM_VIS, 192, dtype=torch.bfloat16)
        dummy_ts = torch.tensor([0.5, 0.5], dtype=torch.bfloat16)
        for _ in range(2):
            _ = app(
                torch.cat([cond_ids, uncond_ids], dim=0),
                dummy_patches,
                dummy_ts,
                cond_pos,
            )
    else:
        dummy_patches = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
        dummy_ts = torch.tensor([0.5], dtype=torch.bfloat16)
        for _ in range(2):
            _ = app(cond_ids, dummy_patches, dummy_ts, cond_pos)
            _ = app(uncond_ids, dummy_patches, dummy_ts, uncond_pos)
    print(f"  Warmup done in {time.time() - t0:.1f}s")

    # --- Generate ---
    H_latent = args.height // 16
    W_latent = args.width // 16

    gen = torch.manual_seed(args.seed)
    latents = torch.randn(
        1, 48, T, H_latent, W_latent, generator=gen, dtype=torch.float32
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )

    print(f"\n  Denoising: {args.steps} steps, CFG={args.cfg_scale}")
    import logging

    logging.basicConfig(level=logging.INFO)

    latents = denoise(
        backbone=app,
        cond_ids=cond_ids,
        uncond_ids=uncond_ids,
        cond_pos=cond_pos,
        uncond_pos=uncond_pos,
        scheduler=scheduler,
        latents=latents,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        cfg_parallel=args.cfg_parallel,
    )

    # --- Denormalize + VAE decode ---
    vae_config_path = os.path.join(args.model_path, "vae", "config.json")
    latents = denormalize_latents(latents, vae_config_path)

    print("  VAE decoding...")
    t0 = time.time()
    with torch.no_grad():
        pixels = vae(latents.float())
    vae_time = time.time() - t0
    print(f"  VAE: {vae_time * 1000:.0f}ms")

    # --- Save ---
    pixels = pixels.squeeze(2).squeeze(0)
    pixels = ((pixels + 1.0) / 2.0).clamp(0, 1)
    pixels = (pixels * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    img = Image.fromarray(pixels, mode="RGB")
    img.save(args.output)

    print(f"\nSaved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
