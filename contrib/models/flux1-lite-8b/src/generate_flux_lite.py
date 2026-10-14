# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FLUX.1-lite-8B-alpha generation script for AWS Neuron.

FLUX.1-lite-8B-alpha (Freepik) is architecturally identical to FLUX.1-dev with a
reduced backbone: 8 double-stream MMDiT blocks instead of 19. It uses the same
CLIP + T5-XXL text encoders, FluxPipeline, VAE, scheduler, and RoPE configuration.

Because of this architectural compatibility, FLUX.1-lite runs natively on NxDI's
first-party FLUX.1 implementation with no code modifications. The NxDI FLUX.1
application reads `num_layers` and `num_single_layers` from the model's config.json
at runtime, so it automatically adapts to FLUX.1-lite's configuration:
  - num_layers: 8 (vs 19 in FLUX.1-dev)
  - num_single_layers: 38 (same as FLUX.1-dev)

Usage:
    # Download the model (requires HuggingFace access):
    huggingface-cli download Freepik/flux.1-lite-8B-alpha --local-dir /shared/flux1-lite-8b

    # Generate an image:
    python generate_flux_lite.py \\
        --checkpoint_dir /shared/flux1-lite-8b \\
        --compile_workdir /tmp/flux-lite/compiled/ \\
        --prompt "A cat holding a sign that says hello world" \\
        --height 1024 --width 1024 \\
        --num_inference_steps 25 \\
        --save_image

Requirements:
    pip install diffusers transformers accelerate sentencepiece protobuf
"""

import argparse
import time

import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)

DEFAULT_CKPT_DIR = "/shared/flux1-lite-8b/"
DEFAULT_COMPILE_DIR = "/tmp/flux-lite/compiled/"


def run_generate(args):
    print(f"FLUX.1-lite-8B generation with args: {args}")

    backbone_tp_degree = args.backbone_tp_degree if args.backbone_tp_degree else 4
    world_size = get_flux_parallelism_config(backbone_tp_degree)
    dtype = torch.bfloat16

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        backbone_tp_degree,
        dtype,
        args.height,
        args.width,
    )

    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=args.height,
        width=args.width,
    )

    print("Compiling model...")
    compile_start = time.time()
    flux_app.compile(args.compile_workdir)
    compile_time = time.time() - compile_start
    print(f"Compilation completed in {compile_time:.1f}s")

    flux_app.load(args.compile_workdir)

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup_rounds):
        flux_app(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ).images[0]

    # Generate
    total_time = 0
    for i in range(args.num_images):
        start = time.time()
        image = flux_app(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
        gen_time = time.time() - start
        total_time += gen_time

        if args.save_image:
            filename = f"flux_lite_output_{i + 1}.png"
            image.save(filename)
            print(f"Image {i + 1} saved to {filename} in {gen_time:.2f}s")
        else:
            print(f"Image {i + 1} generated in {gen_time:.2f}s")

    avg_time = total_time / args.num_images
    steps_per_sec = args.num_inference_steps / avg_time
    print(f"\nResults:")
    print(f"  Average generation time: {avg_time:.2f}s")
    print(f"  Pipeline steps/sec: {steps_per_sec:.2f}")
    print(f"  Compilation time: {compile_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX.1-lite-8B on AWS Neuron (NxDI)")
    parser.add_argument(
        "-p", "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=25)
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5)
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--compile_workdir", type=str, default=DEFAULT_COMPILE_DIR)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--warmup_rounds", type=int, default=5)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--backbone_tp_degree",
        type=int,
        default=None,
        help="Tensor parallelism degree (default: 4 for trn2.3xlarge)",
    )
    args = parser.parse_args()
    run_generate(args)
