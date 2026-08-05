#!/usr/bin/env python3
"""
Compile the Cosmos3 VAE decoder for Neuron using torch_neuronx.trace().

The VAE decoder is a standard AutoencoderKLWan (48 latent channels).
It runs on a single NeuronCore (no TP needed).

Key challenge: F.interpolate (nearest-exact) is not supported in XLA tracing.
Solution: Monkey-patch all nn.Upsample modules with repeat_interleave equivalents.

Usage:
    # For 512x512 images:
    python compile_vae.py --model-path /path/to/Cosmos3-Nano --output /path/to/vae_512.pt

    # For 1024x1024 images:
    python compile_vae.py --model-path /path/to/Cosmos3-Nano --height 1024 --width 1024 \
        --output /path/to/vae_1024.pt

Environment:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

    IMPORTANT: Requires diffusers >= 0.39.0.dev0 for Cosmos3 VAE support.
    Install with: pip install git+https://github.com/huggingface/diffusers.git
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch_neuronx
from diffusers import AutoencoderKLWan


def patch_upsample_modules(model: nn.Module) -> int:
    """Replace all nn.Upsample modules with XLA-compatible repeat_interleave versions.

    nn.Upsample uses F.interpolate which isn't supported in XLA tracing.
    We replace with repeat_interleave which achieves the same nearest-neighbor upsampling.

    Returns:
        Number of modules patched.
    """

    class NeuronUpsample(nn.Module):
        """XLA-compatible nearest-neighbor upsampling via repeat_interleave."""

        def __init__(self, scale_factor):
            super().__init__()
            if isinstance(scale_factor, (tuple, list)):
                self.scale_h = int(scale_factor[0])
                self.scale_w = int(scale_factor[1])
            else:
                self.scale_h = int(scale_factor)
                self.scale_w = int(scale_factor)

        def forward(self, x):
            if x.dim() == 5:
                # [B, C, T, H, W] — 3D video latents
                b, c, t, h, w = x.shape
                x = x.view(b * t, c, h, w)
                x = x.repeat_interleave(self.scale_h, dim=2)
                x = x.repeat_interleave(self.scale_w, dim=3)
                x = x.view(b, c, t, h * self.scale_h, w * self.scale_w)
            else:
                # [B, C, H, W] — 2D
                x = x.repeat_interleave(self.scale_h, dim=2)
                x = x.repeat_interleave(self.scale_w, dim=3)
            return x

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Upsample):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            attr_name = parts[-1]
            replacement = NeuronUpsample(module.scale_factor)
            if attr_name.isdigit():
                parent[int(attr_name)] = replacement
            else:
                setattr(parent, attr_name, replacement)
            count += 1
    return count


class VAEDecodeWrapper(nn.Module):
    """Wrapper that bypasses temporal caching for single-frame decode.

    Pipeline: post_quant_conv -> decoder(no cache, first_chunk=True) -> unpatchify -> clamp
    """

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.patch_size = vae.config.patch_size

    def forward(self, z):
        # z: [1, 48, 1, H_latent, W_latent]
        x = self.post_quant_conv(z)
        out = self.decoder(x, feat_cache=None, feat_idx=[0], first_chunk=True)
        out = self._unpatchify(out)
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out

    def _unpatchify(self, x):
        """Inline unpatchify matching diffusers implementation."""
        p = self.patch_size
        b, c_pp, t, h, w = x.shape
        c = c_pp // (p * p)
        x = x.view(b, c, p, p, t, h, w)
        x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
        x = x.view(b, c, t, h * p, w * p)
        return x


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
    print(f"  Parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")

    # Patch upsample modules for XLA compatibility
    num_patched = patch_upsample_modules(vae)
    print(f"  Patched {num_patched} nn.Upsample modules for Neuron compatibility")

    # Latent dimensions for target resolution
    # spatial_compression = 16, latent_channels = 48, temporal = 1 for images
    H_latent = args.height // 16
    W_latent = args.width // 16
    T = 1

    print(f"\nTarget resolution: {args.height}x{args.width}")
    print(f"Latent shape: [1, 48, {T}, {H_latent}, {W_latent}]")
    print(f"Output shape: [1, 3, {T}, {args.height}, {args.width}]")

    # Create wrapper and example input
    wrapper = VAEDecodeWrapper(vae)
    wrapper.eval()
    example_input = torch.randn(1, 48, T, H_latent, W_latent, dtype=torch.float32)

    # Verify CPU output
    print("\nVerifying CPU decode...")
    with torch.no_grad():
        cpu_output = wrapper(example_input)
    print(f"  CPU output shape: {cpu_output.shape}")
    print(
        f"  CPU output range: [{cpu_output.min().item():.3f}, {cpu_output.max().item():.3f}]"
    )

    # Compile for Neuron
    print(f"\nCompiling VAE decoder for {args.height}x{args.width}...")
    print(f"  (3D convolutions, ~700M params, may take 5-15 minutes)")
    t0 = time.time()

    compiled = torch_neuronx.trace(
        wrapper,
        example_input,
        compiler_args=[
            "--auto-cast",
            "matmult",
            "--model-type=unet-inference",
        ],
    )
    elapsed = time.time() - t0
    print(f"  Compilation complete in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.jit.save(compiled, args.output)
    neff_size = os.path.getsize(args.output) / (1024**2)
    print(f"  Saved to: {args.output} ({neff_size:.1f} MB)")

    # Quick benchmark
    print("\nBenchmarking...")
    with torch.no_grad():
        neuron_output = compiled(example_input)
    diff = (cpu_output - neuron_output).abs().max().item()
    print(f"  Max diff vs CPU: {diff:.6f}")

    # Warmup + measure
    for _ in range(3):
        with torch.no_grad():
            _ = compiled(example_input)
    times = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            _ = compiled(example_input)
        times.append((time.time() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"  Average latency: {avg_ms:.1f}ms")
    print(
        f"\nDone! VAE decoder for {args.height}x{args.width}: {avg_ms:.1f}ms on 1 NeuronCore"
    )


if __name__ == "__main__":
    main()
