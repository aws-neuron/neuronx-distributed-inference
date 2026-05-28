#!/usr/bin/env python3
"""
FLUX.2-klein-base-9B inference on Neuron using NxD Inference.

Usage:
    # Compile and generate
    python generate_flux2_klein.py --checkpoint_dir /path/to/model --compile_workdir /tmp/flux2_klein/

    # With custom prompt
    python generate_flux2_klein.py -p "A futuristic cityscape at sunset" --save_image

    # Benchmark mode
    python generate_flux2_klein.py --num_images 5 --save_image

Requirements:
    - trn2.3xlarge instance with LNC=2 (4 logical cores, TP=4)
    - Neuron SDK 2.29+
    - diffusers >= 0.37.1 (for Flux2KleinPipeline)
    - HuggingFace access to black-forest-labs/FLUX.2-klein-base-9B
"""

import argparse
import os
import sys
import time

import torch

# Add src dir to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from application import (
    NeuronFlux2KleinApplication,
    create_flux2_klein_config,
)

DEFAULT_MODEL_PATH = "/shared/flux2-klein/"
DEFAULT_COMPILE_DIR = "/tmp/flux2_klein/compiled/"


def main(args):
    print(f"FLUX.2-klein-base-9B Neuron Inference")
    print(f"  Model: {args.checkpoint_dir}")
    print(f"  TP degree: {args.tp_degree}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print()

    # Create config
    backbone_config = create_flux2_klein_config(
        model_path=args.checkpoint_dir,
        backbone_tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )

    # Create application
    app = NeuronFlux2KleinApplication(
        model_path=args.checkpoint_dir,
        backbone_config=backbone_config,
        height=args.height,
        width=args.width,
    )

    # Compile
    print("Compiling transformer backbone...")
    t0 = time.time()
    app.compile(args.compile_workdir)
    compile_time = time.time() - t0
    print(f"Compilation: {compile_time:.1f}s")

    # Load
    print("Loading compiled model...")
    t0 = time.time()
    app.load(args.compile_workdir)
    load_time = time.time() - t0
    print(f"Load: {load_time:.1f}s")

    # Warmup
    print("Warming up...")
    for _ in range(2):
        app(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )

    # Generate
    print(f"\nGenerating {args.num_images} images...")
    latencies = []
    for i in range(args.num_images):
        t0 = time.time()
        result = app(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)
        print(f"  Image {i + 1}: {elapsed:.2f}s")

        if args.save_image:
            image = result.images[0]
            filename = os.path.join(args.compile_workdir, f"output_{i + 1}.png")
            image.save(filename)
            print(f"    Saved: {filename}")

    # Summary
    import numpy as np

    latencies = np.array(latencies)
    print(f"\nResults:")
    print(f"  Mean latency: {latencies.mean():.2f}s")
    print(f"  Std: {latencies.std():.2f}s")
    print(f"  Steps/sec: {args.num_inference_steps / latencies.mean():.2f}")
    print(f"  img/s: {1.0 / latencies.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein-base-9B Neuron inference"
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=50)
    parser.add_argument("-g", "--guidance_scale", type=float, default=4.0)
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=4,
        help="Tensor parallelism degree (4 for trn2.3xlarge LNC=2)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--compile_workdir",
        type=str,
        default=DEFAULT_COMPILE_DIR,
        help="Path for compiled model artifacts",
    )
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    main(args)
