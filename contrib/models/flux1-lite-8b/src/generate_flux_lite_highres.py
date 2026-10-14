# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FLUX.1-lite-8B high-resolution generation (2048x2048, 4096x4096) on AWS Neuron.

This script extends the base FLUX.1-lite generation to high resolutions that are
NOT supported by the original model weights or standard inference pipelines:

  - 2048x2048 (16,384 tokens): Backbone compiles natively at TP=4 on trn2.3xlarge.
    VAE decoder exceeds instruction limit at 2K, so tiled VAE decode is used.

  - 4096x4096 (65,536 tokens): Requires context parallelism (TP=4, CP=4) on
    trn2.48xlarge. The backbone processes all 65,536 tokens natively with the
    sequence split across 4 CP shards (16,384 tokens each). VAE decoder uses
    tiled decode with 25 overlapping tiles.

IMPORTANT: The original FLUX.1-lite-8B model was trained at 1024x1024. Generating
at 2K/4K is an extrapolation — image quality may vary compared to the native
resolution. This capability is useful for customers who have fine-tuned their own
models at higher resolutions or want to evaluate the architecture's scaling behavior.

Instance requirements:
  - 2048x2048: trn2.3xlarge (LNC=2, TP=4) — same as 1K
  - 4096x4096: trn2.48xlarge (LNC=2, 16 of 64 cores)
    MUST set: NEURON_RT_VISIBLE_CORES=0-15

Usage:
    # 2K on trn2.3xlarge:
    python generate_flux_lite_highres.py \\
        --checkpoint_dir /shared/flux1-lite-8b \\
        --height 2048 --width 2048

    # 4K on trn2.48xlarge (requires NEURON_RT_VISIBLE_CORES=0-15):
    NEURON_RT_VISIBLE_CORES=0-15 python generate_flux_lite_highres.py \\
        --checkpoint_dir /shared/flux1-lite-8b \\
        --height 4096 --width 4096
"""

import argparse
import json
import os
import time

import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)

# VAE decoder is always compiled at 1024x1024 output (128x128 latent).
# Higher resolutions use tiled decode over the compiled 1K decoder.
VAE_COMPILE_SIZE = 1024
LATENT_TILE_SIZE = VAE_COMPILE_SIZE // 8  # 128 latent pixels
LATENT_OVERLAP = 16  # overlap in latent space (128 pixels in image space)
LATENT_STRIDE = LATENT_TILE_SIZE - LATENT_OVERLAP  # 112


def get_parallelism_config(height, width, backbone_tp_degree):
    """Determine world_size and whether context parallelism is needed.

    Returns:
        (world_size, context_parallel_enabled)
    """
    tokens = (height * width) // 256  # patch_size=2 in 8x downsampled latent

    if tokens <= 16384:
        # 1K (4096 tokens) or 2K (16384 tokens): standard TP, no CP needed
        world_size = get_flux_parallelism_config(backbone_tp_degree)
        return world_size, False
    else:
        # 4K (65536 tokens): need context parallelism to split the sequence
        # CP = world_size / tp_degree. For 4K we want each shard to have
        # 16384 tokens (same as working 2K), so CP = 65536 / 16384 = 4.
        cp_degree = tokens // 16384
        world_size = backbone_tp_degree * cp_degree
        return world_size, True


def setup_tiled_vae_decode(flux_app):
    """Monkey-patch the VAE decoder with a tiled decode implementation.

    The FLUX VAE decoder is compiled at 1024x1024 output (128x128 latent).
    For higher resolutions, the full latent is decoded in overlapping tiles
    with Gaussian-weighted blending for seamless stitching.
    """
    original_vae_decode = flux_app.pipe.vae.decode

    def tiled_vae_decode(latent, **kwargs):
        B, C, H, W = latent.shape

        # Extract return_dict from kwargs (pipeline passes return_dict=False)
        return_dict = kwargs.pop("return_dict", True)

        if H <= LATENT_TILE_SIZE and W <= LATENT_TILE_SIZE:
            return original_vae_decode(latent, return_dict=return_dict, **kwargs)

        # Compute tile grid positions
        row_starts = list(range(0, H - LATENT_TILE_SIZE + 1, LATENT_STRIDE))
        if row_starts[-1] + LATENT_TILE_SIZE < H:
            row_starts.append(H - LATENT_TILE_SIZE)
        col_starts = list(range(0, W - LATENT_TILE_SIZE + 1, LATENT_STRIDE))
        if col_starts[-1] + LATENT_TILE_SIZE < W:
            col_starts.append(W - LATENT_TILE_SIZE)

        n_tiles = len(row_starts) * len(col_starts)
        print(
            f"  Tiled VAE decode: {H}x{W} latent -> "
            f"{len(row_starts)}x{len(col_starts)} = {n_tiles} tiles"
        )

        # Output image dimensions
        out_h = H * 8
        out_w = W * 8
        output = torch.zeros(B, 3, out_h, out_w)
        weight = torch.zeros(1, 1, out_h, out_w)

        # Gaussian blend weight for seamless tile stitching
        tile_out_size = LATENT_TILE_SIZE * 8
        y = torch.linspace(-1, 1, tile_out_size)
        x = torch.linspace(-1, 1, tile_out_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        gauss = torch.exp(-(xx**2 + yy**2) * 3.0).unsqueeze(0).unsqueeze(0)

        tile_idx = 0
        vae_start = time.time()
        for r in row_starts:
            for c in col_starts:
                tile_idx += 1
                tile_latent = latent[
                    :, :, r : r + LATENT_TILE_SIZE, c : c + LATENT_TILE_SIZE
                ].contiguous()

                # Decode tile
                tile_result = original_vae_decode(
                    tile_latent, return_dict=False, **kwargs
                )
                if isinstance(tile_result, tuple):
                    tile_pixels = tile_result[0]
                elif hasattr(tile_result, "sample"):
                    tile_pixels = tile_result.sample
                else:
                    tile_pixels = tile_result

                tile_pixels = tile_pixels.detach().cpu().float()

                # Accumulate with Gaussian weighting
                out_r = r * 8
                out_c = c * 8
                output[
                    :, :, out_r : out_r + tile_out_size, out_c : out_c + tile_out_size
                ] += tile_pixels * gauss
                weight[
                    :, :, out_r : out_r + tile_out_size, out_c : out_c + tile_out_size
                ] += gauss

                if tile_idx % 5 == 0 or tile_idx == n_tiles:
                    elapsed = time.time() - vae_start
                    print(
                        f"    Tile {tile_idx}/{n_tiles} done "
                        f"({elapsed:.1f}s, {elapsed / tile_idx:.2f}s/tile)"
                    )

        output = output / weight.clamp(min=1e-8)

        vae_total = time.time() - vae_start
        print(
            f"  Tiled VAE decode complete: {vae_total:.1f}s "
            f"({vae_total / n_tiles:.2f}s/tile)"
        )

        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=output)
        else:
            return (output,)

    flux_app.pipe.vae.decode = tiled_vae_decode
    return n_tiles_needed(flux_app.height, flux_app.width)


def n_tiles_needed(height, width):
    """Compute the number of VAE decode tiles needed for a given resolution."""
    latent_h = height // 8
    latent_w = width // 8
    if latent_h <= LATENT_TILE_SIZE and latent_w <= LATENT_TILE_SIZE:
        return 1
    row_starts = list(range(0, latent_h - LATENT_TILE_SIZE + 1, LATENT_STRIDE))
    if row_starts[-1] + LATENT_TILE_SIZE < latent_h:
        row_starts.append(latent_h - LATENT_TILE_SIZE)
    col_starts = list(range(0, latent_w - LATENT_TILE_SIZE + 1, LATENT_STRIDE))
    if col_starts[-1] + LATENT_TILE_SIZE < latent_w:
        col_starts.append(latent_w - LATENT_TILE_SIZE)
    return len(row_starts) * len(col_starts)


def run_generate(args):
    height = args.height
    width = args.width
    backbone_tp_degree = args.backbone_tp_degree or 4
    tokens = (height * width) // 256

    # Determine parallelism
    world_size, context_parallel = get_parallelism_config(
        height, width, backbone_tp_degree
    )
    cp_degree = world_size // backbone_tp_degree

    print(f"FLUX.1-lite-8B High-Resolution Generation")
    print(f"  Resolution: {height}x{width} ({tokens} tokens)")
    print(f"  TP={backbone_tp_degree}, CP={cp_degree}, world_size={world_size}")
    if context_parallel:
        print(f"  Tokens per CP shard: {tokens // cp_degree}")
    print(f"  VAE decoder compiled at {VAE_COMPILE_SIZE}x{VAE_COMPILE_SIZE}")

    # Check NEURON_RT_VISIBLE_CORES for 4K
    if world_size > 8 and "NEURON_RT_VISIBLE_CORES" not in os.environ:
        print(
            "\nWARNING: NEURON_RT_VISIBLE_CORES is not set. For world_size > 8 on "
            "trn2.48xlarge, you MUST set NEURON_RT_VISIBLE_CORES=0-15 to prevent "
            "the runtime from creating 64-rank collectives that cause deadlock."
        )
        print("Set: export NEURON_RT_VISIBLE_CORES=0-15")
        return

    dtype = torch.bfloat16

    # Create configs
    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        backbone_tp_degree,
        dtype,
        height,
        width,
        context_parallel_enabled=context_parallel,
    )

    # Override decoder to compile at 1K (will tile for higher resolutions)
    needs_tiled_vae = (height > VAE_COMPILE_SIZE) or (width > VAE_COMPILE_SIZE)
    if needs_tiled_vae:
        decoder_config.height = VAE_COMPILE_SIZE
        decoder_config.width = VAE_COMPILE_SIZE
        vae_tiles = n_tiles_needed(height, width)
        print(f"  VAE tiling: {vae_tiles} tiles with {LATENT_OVERLAP}px overlap")

    # Create application
    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=height,
        width=width,
    )

    # Compile
    print("\nCompiling model...")
    compile_start = time.time()
    flux_app.compile(args.compile_workdir)
    compile_time = time.time() - compile_start
    print(f"Compilation: {compile_time:.1f}s ({compile_time / 60:.1f} min)")

    # Load
    print("Loading model...")
    load_start = time.time()
    flux_app.load(args.compile_workdir)
    load_time = time.time() - load_start
    print(f"Model loaded: {load_time:.1f}s ({load_time / 60:.1f} min)")

    # Setup tiled VAE decode if needed
    if needs_tiled_vae:
        setup_tiled_vae_decode(flux_app)
        print("Tiled VAE decode enabled")

    # Warmup
    print(f"\nGenerating warmup image...")
    t0 = time.time()
    image = flux_app(
        args.prompt,
        height=height,
        width=width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    ).images[0]
    warmup_time = time.time() - t0
    print(f"Warmup: {warmup_time:.2f}s")

    # Benchmark
    print(f"\nBenchmarking ({args.num_images} rounds)...")
    times = []
    for i in range(args.num_images):
        t0 = time.time()
        image = flux_app(
            args.prompt,
            height=height,
            width=width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Image {i + 1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    print(f"\nResults:")
    print(f"  Resolution: {height}x{width}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Average: {avg_time:.2f}s/image")
    print(f"  Compilation: {compile_time:.1f}s")
    print(f"  Model load: {load_time:.1f}s")

    if args.save_image:
        filename = f"flux_lite_{height}x{width}_output.png"
        image.save(filename)
        print(f"  Saved: {filename} ({image.size})")

    if args.save_results:
        results = {
            "resolution": f"{height}x{width}",
            "tokens": tokens,
            "backbone_tp": backbone_tp_degree,
            "cp_degree": cp_degree,
            "world_size": world_size,
            "context_parallel": context_parallel,
            "num_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "warmup_time_s": round(warmup_time, 2),
            "benchmark_times_s": [round(t, 2) for t in times],
            "average_time_s": round(avg_time, 2),
            "compile_time_s": round(compile_time, 1),
            "model_load_time_s": round(load_time, 1),
            "vae_tiles": vae_tiles if needs_tiled_vae else 1,
        }
        results_file = f"flux_lite_{height}x{width}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLUX.1-lite-8B high-resolution generation on AWS Neuron"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
    )
    parser.add_argument("-hh", "--height", type=int, default=2048)
    parser.add_argument("-w", "--width", type=int, default=2048)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=25)
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5)
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, default="/shared/flux1-lite-8b/"
    )
    parser.add_argument(
        "--compile_workdir", type=str, default="/tmp/flux-lite-highres/compiled/"
    )
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument(
        "--backbone_tp_degree",
        type=int,
        default=None,
        help="Tensor parallelism degree (default: 4)",
    )
    args = parser.parse_args()
    run_generate(args)
