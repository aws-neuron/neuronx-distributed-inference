#!/usr/bin/env python3
"""
Compile Cosmos3 backbone for Neuron.

Supports both Cosmos3-Nano (16B, TP=4) and Cosmos3-Super-Text2Image (65B, TP=8).

Usage:
    # Nano at 512x512 on trn2.3xlarge (TP=4):
    python compile.py --model-path /path/to/Cosmos3-Nano --tp 4 --output /path/to/compiled

    # Nano at 1024x1024:
    python compile.py --model-path /path/to/Cosmos3-Nano --tp 4 --height 1024 --width 1024 --output /path/to/compiled_1024p

    # Super at 512x512 on trn2.48xlarge (TP=8):
    python compile.py --model-path /path/to/Cosmos3-Super-Text2Image --tp 8 --output /path/to/compiled

    # Super at 1024x1024:
    python compile.py --model-path /path/to/Cosmos3-Super-Text2Image --tp 8 --height 1024 --width 1024 --output /path/to/compiled_1024p

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
from neuronx_distributed_inference.models.config import NeuronConfig


# Model configurations
MODEL_CONFIGS = {
    "Cosmos3-Nano": {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    "Cosmos3-Super-Text2Image": {
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
}

# Resolution presets (height, width) -> num_vision_patches
# Formula: num_patches = (height // 32) * (width // 32)
# Where 32 = scale_factor_spatial(16) * patch_size(2)
RESOLUTION_PRESETS = {
    "512x512": (512, 512, 256),  # 16×16 = 256 patches
    "768x768": (768, 768, 576),  # 24×24 = 576 patches
    "1024x1024": (1024, 1024, 1024),  # 32×32 = 1024 patches
}


def detect_model_variant(model_path: str) -> str:
    """Detect model variant from config.json."""
    config_path = os.path.join(model_path, "transformer", "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    hidden_size = cfg.get("hidden_size", cfg.get("d_model", 4096))
    if hidden_size >= 5120:
        return "Cosmos3-Super-Text2Image"
    return "Cosmos3-Nano"


def main():
    parser = argparse.ArgumentParser(description="Compile Cosmos3 backbone for Neuron")
    parser.add_argument("--model-path", required=True, help="Path to HF model weights")
    parser.add_argument(
        "--output", required=True, help="Output path for compiled model"
    )
    parser.add_argument(
        "--tp", type=int, default=None, help="TP degree (auto-detected if not set)"
    )
    parser.add_argument(
        "--max-text-len", type=int, default=256, help="Max text token length"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Target image height in pixels"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Target image width in pixels"
    )
    parser.add_argument(
        "--num-vision-patches",
        type=int,
        default=None,
        help="Override: number of vision patches (auto-calculated from height/width if not set)",
    )
    parser.add_argument(
        "--cfg-parallel",
        action="store_true",
        help="Enable CFG-parallel (batch=2): pack cond+uncond in a single call for ~20%% speedup",
    )
    args = parser.parse_args()

    # Calculate vision patches from resolution
    if args.num_vision_patches is not None:
        num_vision_patches = args.num_vision_patches
    else:
        # Formula: (height / scale_factor_spatial / patch_size) * (width / scale_factor_spatial / patch_size)
        # = (height / 32) * (width / 32)
        pH = args.height // 32
        pW = args.width // 32
        num_vision_patches = pH * pW
        if args.height % 32 != 0 or args.width % 32 != 0:
            raise ValueError(
                f"Height ({args.height}) and width ({args.width}) must be divisible by 32 "
                f"(scale_factor_spatial=16 × patch_size=2)"
            )

    total_seq = args.max_text_len + num_vision_patches

    # Detect model variant
    variant = detect_model_variant(args.model_path)
    model_cfg = MODEL_CONFIGS[variant]
    print(f"Detected model: {variant}")
    print(
        f"  hidden_size={model_cfg['hidden_size']}, layers={model_cfg['num_hidden_layers']}"
    )

    # Auto-select TP
    tp = args.tp
    if tp is None:
        tp = 4 if variant == "Cosmos3-Nano" else 8
    print(f"  TP degree: {tp}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(
        f"  Vision patches: {num_vision_patches} (patch grid: {args.height // 32}x{args.width // 32})"
    )
    print(
        f"  Total sequence length: {total_seq} (text={args.max_text_len} + vision={num_vision_patches})"
    )
    if args.cfg_parallel:
        print(f"  CFG-parallel: ENABLED (batch=2, cond+uncond in single call)")

    # Create config
    neuron_config = NeuronConfig(
        tp_degree=tp, world_size=tp, torch_dtype=torch.bfloat16
    )
    config = Cosmos3BackboneInferenceConfig(
        neuron_config=neuron_config,
        cfg_parallel_enabled=args.cfg_parallel,
        head_dim=128,
        vocab_size=151936,
        patch_channels=192,
        latent_channels=48,
        rope_theta=5000000.0,
        mrope_section=[24, 20, 20],
        **model_cfg,
    )
    print(
        f"  Total sequence length: {total_seq} (text={args.max_text_len} + vision={num_vision_patches})"
    )

    # Create config
    neuron_config = NeuronConfig(
        tp_degree=tp, world_size=tp, torch_dtype=torch.bfloat16
    )
    config = Cosmos3BackboneInferenceConfig(
        neuron_config=neuron_config,
        head_dim=128,
        vocab_size=151936,
        patch_channels=192,
        latent_channels=48,
        rope_theta=5000000.0,
        mrope_section=[24, 20, 20],
        **model_cfg,
    )
    config.max_text_len = args.max_text_len
    config.num_vision_patches = num_vision_patches

    # Compile
    transformer_path = os.path.join(args.model_path, "transformer")
    print(f"\nCompiling {variant} backbone...")
    print(f"  Weights: {transformer_path}")
    print(f"  Output:  {args.output}")

    t0 = time.time()
    app = NeuronCosmos3BackboneApplication(model_path=transformer_path, config=config)
    app.compile(args.output)
    elapsed = time.time() - t0

    print(f"\nCompilation complete in {elapsed:.1f}s")
    print(f"Compiled model saved to: {args.output}")
    print(f"\nTo generate images at {args.height}x{args.width}, run:")
    print(f"  python generate.py --model-path {args.model_path} \\")
    print(f"    --compiled-path {args.output} \\")
    print(f"    --vae-path <path_to_vae>/vae_decoder.pt \\")
    print(f"    --height {args.height} --width {args.width} \\")
    print(f'    --prompt "your prompt"')


if __name__ == "__main__":
    main()
