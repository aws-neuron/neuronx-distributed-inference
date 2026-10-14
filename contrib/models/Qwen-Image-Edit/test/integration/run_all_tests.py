#!/usr/bin/env python3
"""
Run All Unit Tests: Compare Neuron vs CPU/GPU inference for all components

This script runs all unit tests to identify which component is causing
output differences between Neuron and CPU/GPU inference.

Components tested:
1. VAE (Encoder + Decoder)
2. Transformer
3. Text Encoder (Vision Encoder + Language Model)

Usage:
    python tests/run_all_tests.py --compiled_models_dir /path/to/compiled_models

    # Run specific tests
    python tests/run_all_tests.py --test vae
    python tests/run_all_tests.py --test transformer
    python tests/run_all_tests.py --test text_encoder
"""

import os
import sys
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def run_test(test_script, args):
    """Run a test script in a subprocess to avoid environment conflicts."""
    cmd = [
        sys.executable, test_script,
        "--compiled_models_dir", args.compiled_models_dir,
    ]

    # VAE test supports --height and --width
    if "test_vae" in test_script:
        cmd.extend(["--height", str(args.height)])
        cmd.extend(["--width", str(args.width)])

    # Text encoder only supports --image_size and --max_sequence_length
    if "text_encoder" in test_script:
        cmd.extend(["--image_size", str(args.image_size)])
        cmd.extend(["--max_sequence_length", str(args.max_sequence_length)])

    # Transformer supports multiple options
    if "transformer" in test_script:
        cmd.extend(["--height", str(args.height)])
        cmd.extend(["--width", str(args.width)])
        cmd.extend(["--max_sequence_length", str(args.max_sequence_length)])
        cmd.extend(["--batch_size", str(args.batch_size)])
        cmd.extend(["--patch_multiplier", str(args.patch_multiplier)])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all unit tests for Qwen-Image-Edit Neuron inference"
    )
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Vision encoder image size")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for transformer test")
    parser.add_argument("--patch_multiplier", type=int, default=2,
                        help="Patch multiplier for transformer")
    parser.add_argument("--test", type=str, default="all",
                        choices=["vae", "transformer", "text_encoder", "all"],
                        help="Which test(s) to run")
    args = parser.parse_args()

    # Get test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*80)
    print("QWEN-IMAGE-EDIT NEURON UNIT TESTS")
    print("="*80)
    print(f"\nCompiled models directory: {args.compiled_models_dir}")
    print(f"Image size: {args.height}x{args.width}")
    print(f"Vision encoder image size: {args.image_size}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Tests to run: {args.test}")

    results = {}

    # Run VAE test
    if args.test in ["vae", "all"]:
        print("\n" + "="*80)
        print("VAE TESTS")
        print("="*80)
        vae_test = os.path.join(test_dir, "test_vae.py")
        if os.path.exists(vae_test):
            results["vae"] = run_test(vae_test, args)
        else:
            print(f"Test script not found: {vae_test}")
            results["vae"] = None

    # Run Transformer test
    if args.test in ["transformer", "all"]:
        print("\n" + "="*80)
        print("TRANSFORMER TESTS")
        print("="*80)
        transformer_test = os.path.join(test_dir, "test_transformer.py")
        if os.path.exists(transformer_test):
            results["transformer"] = run_test(transformer_test, args)
        else:
            print(f"Test script not found: {transformer_test}")
            results["transformer"] = None

    # Run Text Encoder test
    if args.test in ["text_encoder", "all"]:
        print("\n" + "="*80)
        print("TEXT ENCODER TESTS")
        print("="*80)
        text_encoder_test = os.path.join(test_dir, "test_text_encoder.py")
        if os.path.exists(text_encoder_test):
            results["text_encoder"] = run_test(text_encoder_test, args)
        else:
            print(f"Test script not found: {text_encoder_test}")
            results["text_encoder"] = None

    # Final Summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    for name, passed in results.items():
        if passed is True:
            status = "PASSED"
        elif passed is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"  {name:20s}: {status}")

    # Recommendations
    print("\n" + "="*80)
    print("DEBUGGING RECOMMENDATIONS")
    print("="*80)
    print("""
If you see blurry output images, the issue is likely in one of these areas:

1. VAE Decoder (Most Common)
   - Check if cosine similarity is < 0.99 for the decoder
   - VAE decoder numerical errors can cause blurry images
   - Try: Increase normalization precision or check interpolation mode

2. Transformer (Diffusion)
   - Check if output differs significantly across timesteps
   - Large errors accumulate across denoising steps
   - Try: Check attention implementation and RoPE encoding

3. Text Encoder
   - Vision encoder errors affect conditioning
   - Language model errors affect prompt understanding
   - Try: Check embedding and attention layers

4. Scaling/Normalization
   - Check if latent_mean/latent_std are applied correctly
   - Verify dtype conversions (bfloat16 <-> float32)

To debug further:
   - Run individual component tests with --save_images
   - Compare intermediate outputs at each step
   - Check for NaN/Inf values in outputs
""")


if __name__ == "__main__":
    main()
