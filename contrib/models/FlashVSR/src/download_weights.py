#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download and prepare FlashVSR-v1.1 weights for Neuron inference.

Downloads the model from HuggingFace and organizes weights into the expected
directory structure for the FlashVSR Neuron pipeline.

Usage:
    python -m src.download_weights --output-dir /path/to/FlashVSR-v1.1

Required files from HuggingFace:
    - JunhaoZhuang/FlashVSR-v1.1:
        - diffusion_pytorch_model_streaming_dmd.safetensors (DiT weights)
        - LQ_proj_in.ckpt (LQ projection weights)
        - TCDecoder.ckpt (TCDecoder weights)
        - posi_prompt.pth (pre-computed text embedding)
"""

import argparse
import os
import sys


def download_weights(output_dir: str, token: str = None):
    """Download FlashVSR-v1.1 weights from HuggingFace.

    Args:
        output_dir: Directory to save weights
        token: HuggingFace access token (if model is gated)
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    repo_id = "JunhaoZhuang/FlashVSR-v1.1"
    print(f"Downloading FlashVSR-v1.1 weights from {repo_id}...")

    # Required files
    required_files = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
        "posi_prompt.pth",
    ]

    for filename in required_files:
        target = os.path.join(output_dir, filename)
        if os.path.exists(target):
            print(f"  [SKIP] {filename} already exists")
            continue

        print(f"  Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                token=token,
            )
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [WARN] Failed to download {filename}: {e}")

    # Create symlink for NxDI checkpoint loader compatibility
    actual = os.path.join(
        output_dir, "diffusion_pytorch_model_streaming_dmd.safetensors"
    )
    symlink = os.path.join(output_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(actual) and not os.path.exists(symlink):
        os.symlink(os.path.basename(actual), symlink)
        print(f"  Created symlink: diffusion_pytorch_model.safetensors")

    print(f"\nWeights saved to: {output_dir}")
    print(f"Contents:")
    for f in sorted(os.listdir(output_dir)):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download FlashVSR-v1.1 weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save weights",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (if model is gated)",
    )
    args = parser.parse_args()
    download_weights(args.output_dir, args.token)


if __name__ == "__main__":
    main()
