#!/usr/bin/env python3
"""
LTX-2 VAE Decoder — Standalone Compilation Script
==================================================
Compiles the tensor-parallel VAE decoder for Neuron.

Default tile: 4x16 latent (128x512 pixels) — optimal for 1024x1536 output.

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  pip install diffusers

  # Compile 4x16 tile (recommended for 1024x1536 output)
  NEURON_RT_VISIBLE_CORES=0-3 python compile_vae.py \\
    --tp-degree 4 --height 128 --width 512

  # Compile 8x8 tile (for 512x768 output)
  NEURON_RT_VISIBLE_CORES=0-3 python compile_vae.py \\
    --tp-degree 4 --height 256 --width 256

Notes:
  - Tile area must satisfy H_latent * W_latent <= 64 (SRAM limit)
  - 4x16 tiles are 12.5% faster per-tile than 8x8 tiles
  - Compilation requires ~150GB RAM (use trn2.48xlarge or larger)
  - Compilation takes ~30 minutes on trn2.48xlarge
"""

import argparse
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modeling_vae import compile_vae_decoder


def main():
    parser = argparse.ArgumentParser(description="Compile LTX-2 VAE decoder with TP")
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Tile pixel height (default 128 for 4-latent)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Tile pixel width (default 512 for 16-latent)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=121, help="Number of video frames"
    )
    parser.add_argument(
        "--tp-degree", type=int, default=4, help="Tensor parallel degree"
    )
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/ltx2_vae_tp4")
    parser.add_argument(
        "--compiler-workdir", type=str, default="/home/ubuntu/compiler_workdir_vae"
    )
    parser.add_argument("--vae-path", type=str, default="Lightricks/LTX-2")
    args = parser.parse_args()

    # Verify tile area constraint
    latent_h = args.height // 32
    latent_w = args.width // 32
    area = latent_h * latent_w
    if area > 64:
        print(
            f"ERROR: Tile area {latent_h}x{latent_w} = {area} exceeds 64-element SRAM limit"
        )
        print("Reduce tile dimensions. Maximum compilable tiles:")
        print("  8x8 (256x256 px), 4x16 (128x512 px), 4x14 (128x448 px)")
        sys.exit(1)

    compile_vae_decoder(
        tp_degree=args.tp_degree,
        tile_height=args.height,
        tile_width=args.width,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        compiler_workdir=args.compiler_workdir,
        vae_path=args.vae_path,
    )


if __name__ == "__main__":
    main()
