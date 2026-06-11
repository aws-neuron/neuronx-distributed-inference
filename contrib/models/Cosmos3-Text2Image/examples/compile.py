#!/usr/bin/env python3
"""
Compile Cosmos3 backbone for Neuron.

Supports both Cosmos3-Nano (16B, TP=4) and Cosmos3-Super-Text2Image (65B, TP=8).

Usage:
    # Nano on trn2.3xlarge (TP=4):
    python compile.py --model-path /path/to/Cosmos3-Nano --tp 4 --output /path/to/compiled

    # Super on trn2.48xlarge (TP=8):
    python compile.py --model-path /path/to/Cosmos3-Super-Text2Image --tp 8 --output /path/to/compiled

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
        "--num-vision-patches",
        type=int,
        default=256,
        help="Number of vision patches (256 for 512x512)",
    )
    args = parser.parse_args()

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
    config.num_vision_patches = args.num_vision_patches

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


if __name__ == "__main__":
    main()
