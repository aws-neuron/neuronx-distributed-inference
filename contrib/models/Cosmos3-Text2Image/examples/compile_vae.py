#!/usr/bin/env python3
"""
Compile the Cosmos3 VAE decoder for Neuron using torch_neuronx.trace().

The VAE decoder is a standard AutoencoderKLWan (48 latent channels).
It runs on a single NeuronCore (no TP needed).

Usage:
    python compile_vae.py --model-path /path/to/Cosmos3-Nano --output /path/to/vae_decoder.pt

Environment:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

    IMPORTANT: Requires diffusers >= 0.39.0.dev0 for Cosmos3 VAE support.
    Install with: pip install git+https://github.com/huggingface/diffusers.git
"""

import argparse
import os
import time

import torch
import torch_neuronx
from diffusers import AutoencoderKLWan


def main():
    parser = argparse.ArgumentParser(description="Compile Cosmos3 VAE decoder")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to HF model (parent dir with vae/ subfolder)",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for compiled VAE (.pt)"
    )
    parser.add_argument("--height", type=int, default=512, help="Target image height")
    parser.add_argument("--width", type=int, default=512, help="Target image width")
    args = parser.parse_args()

    vae_path = os.path.join(args.model_path, "vae")
    print(f"Loading VAE from: {vae_path}")

    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32)
    vae.eval()

    # Latent dimensions for target resolution
    # spatial_compression = 16, latent_channels = 48, temporal = 1 for images
    H_latent = args.height // 16
    W_latent = args.width // 16
    T = 1

    print(f"Target resolution: {args.height}x{args.width}")
    print(f"Latent shape: [1, 48, {T}, {H_latent}, {W_latent}]")

    # Example latent input
    example_input = torch.randn(1, 48, T, H_latent, W_latent, dtype=torch.float32)

    # Wrap decoder for tracing
    class VAEDecodeWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent).sample

    wrapper = VAEDecodeWrapper(vae)

    print(f"\nCompiling VAE decoder...")
    t0 = time.time()
    compiled = torch_neuronx.trace(
        wrapper,
        example_input,
        compiler_args=["--model-type=unet", "--auto-cast=matmult", "-O1"],
    )
    elapsed = time.time() - t0
    print(f"Compilation complete in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    compiled.save(args.output)
    print(f"Saved compiled VAE to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
