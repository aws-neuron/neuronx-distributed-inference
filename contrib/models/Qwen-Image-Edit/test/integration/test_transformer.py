#!/usr/bin/env python3
"""
Transformer Unit Test: Compare Neuron vs CPU/GPU inference results

This test compares the QwenImageTransformer2DModel outputs between:
1. Original model running on CPU
2. Compiled model running on Neuron (trn2)

Key metrics:
- Max Absolute Error (MAE)
- Mean Absolute Error (Mean AE)
- Cosine Similarity
- Output statistics (mean, std, min, max)
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Neuron environment BEFORE imports
TP_DEGREE = 8
os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import torch
import torch.nn.functional as F
import numpy as np

from diffusers import QwenImageEditPlusPipeline
from neuron_qwen_image_edit.neuron_rope import patch_qwenimage_rope


CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def compute_metrics(cpu_output, neuron_output, name="output"):
    """Compute comparison metrics between CPU and Neuron outputs."""
    # Ensure same dtype for comparison
    cpu_out = cpu_output.float().detach().cpu()
    neuron_out = neuron_output.float().detach().cpu()

    # Handle shape mismatch (Neuron output may be padded)
    if cpu_out.shape != neuron_out.shape:
        print(f"  Shape mismatch: CPU {cpu_out.shape} vs Neuron {neuron_out.shape}")
        # Truncate to smaller shape
        min_shape = [min(c, n) for c, n in zip(cpu_out.shape, neuron_out.shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        cpu_out = cpu_out[slices]
        neuron_out = neuron_out[slices]
        print(f"  Comparing truncated shape: {cpu_out.shape}")

    # Absolute error
    abs_error = torch.abs(cpu_out - neuron_out)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error
    rel_error = abs_error / (torch.abs(cpu_out) + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Cosine similarity
    cpu_flat = cpu_out.flatten()
    neuron_flat = neuron_out.flatten()
    cosine_sim = F.cosine_similarity(cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)).item()

    # Statistics
    cpu_stats = {
        "mean": cpu_out.mean().item(),
        "std": cpu_out.std().item(),
        "min": cpu_out.min().item(),
        "max": cpu_out.max().item(),
    }
    neuron_stats = {
        "mean": neuron_out.mean().item(),
        "std": neuron_out.std().item(),
        "min": neuron_out.min().item(),
        "max": neuron_out.max().item(),
    }

    print(f"\n{'='*60}")
    print(f"Metrics for {name}")
    print(f"{'='*60}")
    print(f"Shape: {cpu_out.shape}")
    print(f"\nError Metrics:")
    print(f"  Max Absolute Error:  {max_abs_error:.6e}")
    print(f"  Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"  Max Relative Error:  {max_rel_error:.6e}")
    print(f"  Mean Relative Error: {mean_rel_error:.6e}")
    print(f"  Cosine Similarity:   {cosine_sim:.6f}")
    print(f"\nCPU Output Statistics:")
    print(f"  Mean: {cpu_stats['mean']:.6f}, Std: {cpu_stats['std']:.6f}")
    print(f"  Min:  {cpu_stats['min']:.6f}, Max: {cpu_stats['max']:.6f}")
    print(f"\nNeuron Output Statistics:")
    print(f"  Mean: {neuron_stats['mean']:.6f}, Std: {neuron_stats['std']:.6f}")
    print(f"  Min:  {neuron_stats['min']:.6f}, Max: {neuron_stats['max']:.6f}")

    # Check for NaN/Inf
    if torch.isnan(neuron_out).any():
        print(f"\n  WARNING: Neuron output contains NaN values!")
    if torch.isinf(neuron_out).any():
        print(f"\n  WARNING: Neuron output contains Inf values!")

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_sim": cosine_sim,
        "cpu_stats": cpu_stats,
        "neuron_stats": neuron_stats,
    }


def test_transformer_single_step(args):
    """Test transformer for a single denoising step."""
    print("\n" + "="*60)
    print("Testing Transformer (Single Step)")
    print("="*60)

    dtype = torch.bfloat16
    batch_size = args.batch_size
    height, width = args.height, args.width

    # Calculate dimensions
    latent_height = height // 8
    latent_width = width // 8
    patch_size = 2
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    temporal_frames = args.patch_multiplier  # 2 for image editing
    num_patches = temporal_frames * patch_h * patch_w

    in_channels = 64
    text_hidden_size = 3584
    max_seq_len = args.max_sequence_length

    print(f"\nConfiguration:")
    print(f"  Image size: {height}x{width}")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Patch size: {patch_size}")
    print(f"  Temporal frames: {temporal_frames}")
    print(f"  Num patches: {num_patches}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Batch size: {batch_size}")

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Patch RoPE for Neuron compatibility
    print("Patching RoPE for Neuron compatibility...")
    pipe.transformer = patch_qwenimage_rope(pipe.transformer)
    pipe.transformer.eval()

    # Create test inputs
    print("\nCreating test inputs...")
    # hidden_states: (batch, num_patches, in_channels)
    hidden_states = torch.randn(batch_size, num_patches, in_channels, dtype=dtype)
    # encoder_hidden_states: (batch, seq_len, text_hidden_size)
    encoder_hidden_states = torch.randn(batch_size, max_seq_len, text_hidden_size, dtype=dtype)
    # timestep: (batch,) - use a typical timestep value
    timestep = torch.tensor([500.0] * batch_size, dtype=torch.float32)
    # img_shapes for CPU model
    img_shapes = [(temporal_frames, patch_h, patch_w)] * batch_size

    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  img_shapes: {img_shapes}")

    # CPU inference
    print("\nRunning CPU inference...")
    with torch.no_grad():
        cpu_output = pipe.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False
        )
    cpu_output = cpu_output[0]
    print(f"  CPU output shape: {cpu_output.shape}")

    # Check compiled model
    transformer_path = f"{args.compiled_models_dir}/transformer"
    if not os.path.exists(transformer_path):
        print(f"\nERROR: Compiled transformer not found at {transformer_path}")
        print("Please run compile_transformer.py first.")
        return None

    # Load Neuron compiled model
    print(f"\nLoading compiled transformer from {transformer_path}...")
    import neuronx_distributed
    compiled_transformer = neuronx_distributed.trace.parallel_model_load(transformer_path)

    # Prepare inputs for Neuron (timestep must be float32)
    timestep_f32 = timestep.to(torch.float32)

    # Neuron inference
    print("Running Neuron inference...")
    with torch.no_grad():
        neuron_output = compiled_transformer(
            hidden_states,
            encoder_hidden_states,
            timestep_f32
        )
    neuron_output = neuron_output[0]
    print(f"  Neuron output shape: {neuron_output.shape}")

    # Compare results
    metrics = compute_metrics(cpu_output, neuron_output, "Transformer Output")

    return metrics


def test_transformer_multiple_timesteps(args):
    """Test transformer across multiple timesteps to check consistency."""
    print("\n" + "="*60)
    print("Testing Transformer (Multiple Timesteps)")
    print("="*60)

    dtype = torch.bfloat16
    batch_size = args.batch_size
    height, width = args.height, args.width

    # Calculate dimensions
    latent_height = height // 8
    latent_width = width // 8
    patch_size = 2
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    temporal_frames = args.patch_multiplier
    num_patches = temporal_frames * patch_h * patch_w

    in_channels = 64
    text_hidden_size = 3584
    max_seq_len = args.max_sequence_length

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    pipe.transformer = patch_qwenimage_rope(pipe.transformer)
    pipe.transformer.eval()

    # Check compiled model
    transformer_path = f"{args.compiled_models_dir}/transformer"
    if not os.path.exists(transformer_path):
        print(f"\nERROR: Compiled transformer not found at {transformer_path}")
        return None

    print(f"Loading compiled transformer from {transformer_path}...")
    import neuronx_distributed
    compiled_transformer = neuronx_distributed.trace.parallel_model_load(transformer_path)

    # Test at different timesteps
    timesteps_to_test = [999.0, 750.0, 500.0, 250.0, 1.0]
    results = []

    # Use same random inputs for all timesteps
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, num_patches, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, max_seq_len, text_hidden_size, dtype=dtype)
    img_shapes = [(temporal_frames, patch_h, patch_w)] * batch_size

    for t in timesteps_to_test:
        timestep = torch.tensor([t] * batch_size, dtype=torch.float32)

        print(f"\n--- Timestep {t} ---")

        with torch.no_grad():
            # CPU
            cpu_output = pipe.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                return_dict=False
            )[0]

            # Neuron
            neuron_output = compiled_transformer(
                hidden_states,
                encoder_hidden_states,
                timestep
            )[0]

        # Quick metrics
        abs_error = torch.abs(cpu_output.float() - neuron_output.float())
        max_ae = abs_error.max().item()
        mean_ae = abs_error.mean().item()
        cosine_sim = F.cosine_similarity(
            cpu_output.flatten().unsqueeze(0).float(),
            neuron_output.flatten().unsqueeze(0).float()
        ).item()

        print(f"  Max AE: {max_ae:.6e}, Mean AE: {mean_ae:.6e}, Cosine: {cosine_sim:.6f}")
        results.append({
            "timestep": t,
            "max_abs_error": max_ae,
            "mean_abs_error": mean_ae,
            "cosine_sim": cosine_sim,
        })

    # Summary
    print("\n--- Timestep Summary ---")
    avg_cosine = np.mean([r["cosine_sim"] for r in results])
    max_error = max([r["max_abs_error"] for r in results])
    print(f"Average Cosine Similarity: {avg_cosine:.6f}")
    print(f"Max Absolute Error (all timesteps): {max_error:.6e}")

    return results


def test_transformer_block_by_block(args):
    """Test individual transformer blocks to identify problematic layers."""
    print("\n" + "="*60)
    print("Testing Transformer Block-by-Block")
    print("="*60)
    print("NOTE: This test requires manual inspection of intermediate outputs.")
    print("The compiled model doesn't expose individual blocks.")
    print("This test compares the CPU model's block outputs for debugging.")

    dtype = torch.bfloat16
    batch_size = 1
    height, width = args.height, args.width

    # Calculate dimensions
    latent_height = height // 8
    latent_width = width // 8
    patch_size = 2
    patch_h = latent_height // patch_size
    patch_w = latent_width // patch_size
    temporal_frames = args.patch_multiplier
    num_patches = temporal_frames * patch_h * patch_w

    in_channels = 64
    text_hidden_size = 3584

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    pipe.transformer = patch_qwenimage_rope(pipe.transformer)
    pipe.transformer.eval()

    transformer = pipe.transformer
    num_blocks = len(transformer.transformer_blocks)
    print(f"Transformer has {num_blocks} blocks")

    # Create test inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, num_patches, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, args.max_sequence_length, text_hidden_size, dtype=dtype)
    timestep = torch.tensor([500.0], dtype=torch.float32)
    img_shapes = [(temporal_frames, patch_h, patch_w)]

    # Check output statistics at each block
    print("\n--- Block Output Statistics (CPU) ---")
    print("This helps identify where numerical issues might occur.")

    # We need to hook into the model to get intermediate outputs
    # For now, just run the full model and check final output
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False
        )[0]

    print(f"\nFinal output statistics:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    print(f"  Has NaN: {torch.isnan(output).any()}")
    print(f"  Has Inf: {torch.isinf(output).any()}")

    return {"num_blocks": num_blocks}


def main():
    parser = argparse.ArgumentParser(description="Transformer Unit Test: CPU vs Neuron")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max text sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 or 2)")
    parser.add_argument("--patch_multiplier", type=int, default=2,
                        help="Patch multiplier (2 for image editing)")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--test", type=str, default="single",
                        choices=["single", "timesteps", "blocks", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    print("="*60)
    print("Transformer Unit Test: Comparing Neuron vs CPU Inference")
    print("="*60)
    print(f"Image size: {args.height}x{args.width}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch multiplier: {args.patch_multiplier}")
    print(f"Compiled models: {args.compiled_models_dir}")

    results = {}

    if args.test in ["single", "all"]:
        results["single_step"] = test_transformer_single_step(args)

    if args.test in ["timesteps", "all"]:
        results["timesteps"] = test_transformer_multiple_timesteps(args)

    if args.test in ["blocks", "all"]:
        results["blocks"] = test_transformer_block_by_block(args)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if "single_step" in results and results["single_step"]:
        m = results["single_step"]
        status = "PASS" if m["cosine_sim"] > 0.99 else "WARN" if m["cosine_sim"] > 0.95 else "FAIL"
        print(f"Single Step:  Cosine Sim = {m['cosine_sim']:.6f}  Max AE = {m['max_abs_error']:.2e}  [{status}]")

    if "timesteps" in results and results["timesteps"]:
        avg_cos = np.mean([r["cosine_sim"] for r in results["timesteps"]])
        status = "PASS" if avg_cos > 0.99 else "WARN" if avg_cos > 0.95 else "FAIL"
        print(f"Multi-Timestep: Avg Cosine = {avg_cos:.6f}  [{status}]")


if __name__ == "__main__":
    main()
