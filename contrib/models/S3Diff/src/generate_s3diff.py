# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
S3Diff one-step 4x super-resolution on AWS Neuron.

Downloads required weights (SD-Turbo, S3Diff LoRA, DEResNet), compiles all
components, and runs super-resolution inference. Supports arbitrary input
resolutions via tiling (images whose 4x upscaled size exceeds 512x512 are
automatically processed with overlapping tiles).

Usage:
    python generate_s3diff.py \
        --input_image /path/to/lr_image.png \
        --output_image /path/to/sr_output.png \
        --compile_dir /tmp/s3diff/compiled/

    # Multi-resolution examples:
    # 128x128 input -> 512x512 output (single tile, ~0.5s)
    # 256x256 input -> 1024x1024 output (9 tiles, ~4.8s)
    # 512x512 input -> 2048x2048 output (25 tiles, ~13.3s)

Requirements:
    pip install diffusers transformers peft accelerate torchvision
"""

import argparse
import os
import time

import torch
from PIL import Image

try:
    from .modeling_s3diff import S3DiffNeuronPipeline
except ImportError:
    from modeling_s3diff import S3DiffNeuronPipeline


DEFAULT_SD_TURBO_PATH = "/shared/sd-turbo/"
DEFAULT_S3DIFF_WEIGHTS = "/shared/s3diff/s3diff.pkl"
DEFAULT_DE_NET_WEIGHTS = "/shared/s3diff/de_net.pth"
DEFAULT_COMPILE_DIR = "/tmp/s3diff/compiled/"


def download_weights(sd_turbo_path, s3diff_weights_path, de_net_path):
    """Download model weights if not already present."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if not os.path.exists(sd_turbo_path):
        print("Downloading SD-Turbo...")
        snapshot_download("stabilityai/sd-turbo", local_dir=sd_turbo_path)

    if not os.path.exists(s3diff_weights_path):
        print("Downloading S3Diff weights...")
        os.makedirs(os.path.dirname(s3diff_weights_path), exist_ok=True)
        hf_hub_download(
            "zhangap/S3Diff",
            filename="s3diff.pkl",
            local_dir=os.path.dirname(s3diff_weights_path),
        )

    if not os.path.exists(de_net_path):
        print("Downloading DEResNet weights...")
        os.makedirs(os.path.dirname(de_net_path), exist_ok=True)
        # DEResNet weights are in the S3Diff GitHub repo
        import subprocess

        repo_dir = "/tmp/s3diff_repo"
        if not os.path.exists(repo_dir):
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/ArcticHare105/S3Diff.git",
                    repo_dir,
                ],
                check=True,
            )
        import shutil

        shutil.copy2(
            os.path.join(repo_dir, "assets", "mm-realsr", "de_net.pth"),
            de_net_path,
        )


def main():
    parser = argparse.ArgumentParser(description="S3Diff 4x SR on AWS Neuron")
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input low-resolution image (any size; will be 4x upscaled)",
    )
    parser.add_argument(
        "--output_image", type=str, default="sr_output.png", help="Output path"
    )
    parser.add_argument("--sd_turbo_path", type=str, default=DEFAULT_SD_TURBO_PATH)
    parser.add_argument("--s3diff_weights", type=str, default=DEFAULT_S3DIFF_WEIGHTS)
    parser.add_argument("--de_net_weights", type=str, default=DEFAULT_DE_NET_WEIGHTS)
    parser.add_argument("--compile_dir", type=str, default=DEFAULT_COMPILE_DIR)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--warmup_rounds", type=int, default=5)
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Pixel-space tile size for VAE/UNet (default: 512). Must be divisible by 8.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=128,
        help="Pixel-space overlap between tiles (default: 128). Must be divisible by 8.",
    )
    parser.add_argument("--download", action="store_true", help="Download weights")
    args = parser.parse_args()

    if args.download:
        download_weights(args.sd_turbo_path, args.s3diff_weights, args.de_net_weights)

    # Create a test image if none provided
    if args.input_image is None:
        print("No input image provided, creating a 128x128 test pattern...")
        import numpy as np

        test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        lr_image = Image.fromarray(test_img)
    else:
        lr_image = Image.open(args.input_image).convert("RGB")

    lr_w, lr_h = lr_image.size
    hr_w, hr_h = lr_w * 4, lr_h * 4
    print(f"Input image: {lr_w}x{lr_h} -> Output: {hr_w}x{hr_h}")
    if hr_h > args.tile_size or hr_w > args.tile_size:
        print(
            f"Tiling enabled (tile_size={args.tile_size}, overlap={args.tile_overlap})"
        )

    # Build pipeline (lr_size is always 128 for DEResNet)
    pipeline = S3DiffNeuronPipeline(
        sd_turbo_path=args.sd_turbo_path,
        s3diff_weights_path=args.s3diff_weights,
        de_net_path=args.de_net_weights,
        compile_dir=args.compile_dir,
        lr_size=128,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    print("\nLoading model...")
    pipeline.load()

    print("\nCompiling...")
    t0 = time.time()
    pipeline.compile()
    compile_time = time.time() - t0
    print(f"Total compilation: {compile_time:.1f}s")

    # Warmup
    print(f"\nWarming up ({args.warmup_rounds} rounds)...")
    for _ in range(args.warmup_rounds):
        pipeline(lr_image)

    # Benchmark
    print(f"\nGenerating {args.num_images} images...")
    total_time = 0
    for i in range(args.num_images):
        t0 = time.time()
        sr_image = pipeline(lr_image)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Image {i + 1}: {elapsed:.3f}s")

    avg_time = total_time / args.num_images
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Throughput: {1.0 / avg_time:.2f} img/s")
    print(f"  Compilation: {compile_time:.1f}s")

    sr_image.save(args.output_image)
    print(f"  Saved: {args.output_image}")


if __name__ == "__main__":
    main()
