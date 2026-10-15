"""
HunyuanVideo-1.5 Video Generation on Neuron.

Simplified entry point for generating videos from text prompts.
Requires pre-compiled models (see README for compilation steps).

Usage:
    export HUNYUAN_REPO_DIR=./HunyuanVideo-1.5
    export HUNYUAN_MODELS_DIR=./models
    export HUNYUAN_COMPILED_DIR=./compiled
    export NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2
    export NEURON_FUSE_SOFTMAX=1 XLA_DISABLE_FUNCTIONALIZATION=1

    python generate_video.py --prompt "A golden retriever running in a sunny meadow"
"""

import os
import sys
import argparse

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set Neuron environment
os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "1")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("HUNYUAN_ATTN_MODE", "torch")


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-1.5 Text-to-Video on Neuron"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A golden retriever running in a sunny meadow",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="CFG guidance scale (0 = no CFG)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output_video", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    from e2e_pipeline import run_pipeline

    no_cfg = args.guidance_scale == 0
    result = run_pipeline(
        prompt=args.prompt,
        steps=args.steps,
        output_dir=args.output_dir,
        no_cfg=no_cfg,
        seed=args.seed,
    )

    if result and "frames" in result:
        print(
            f"\nGeneration complete: {len(result['frames'])} frames saved to {args.output_dir}"
        )
    else:
        print("\nGeneration failed")


if __name__ == "__main__":
    main()
