#!/usr/bin/env python3
"""
VAE Unit Test: Compare Neuron vs CPU/GPU inference results

This test compares the VAE encoder and decoder outputs between:
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

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Set Neuron environment before importing neuron libraries
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

from diffusers import QwenImageEditPlusPipeline
from neuron_qwen_image_edit.autoencoder_kl_qwenimage_neuron import AutoencoderKLQwenImage as NeuronAutoencoder
from neuron_qwen_image_edit.neuron_commons import f32Wrapper


CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def compute_metrics(cpu_output, neuron_output, name="output"):
    """Compute comparison metrics between CPU and Neuron outputs."""
    # Ensure same dtype for comparison
    cpu_out = cpu_output.float().detach().cpu()
    neuron_out = neuron_output.float().detach().cpu()

    # Absolute error
    abs_error = torch.abs(cpu_out - neuron_out)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error (avoid division by zero)
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

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_sim": cosine_sim,
        "cpu_stats": cpu_stats,
        "neuron_stats": neuron_stats,
    }


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def test_vae_encoder(args):
    """Test VAE encoder: CPU vs Neuron."""
    print("\n" + "="*60)
    print("Testing VAE Encoder")
    print("="*60)

    dtype = torch.bfloat16
    height, width = args.height, args.width
    temporal_frames = 1

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Create Neuron-compatible VAE with same weights
    print("Creating Neuron-compatible VAE...")
    original_vae_config = pipe.vae.config
    neuron_vae = NeuronAutoencoder(
        base_dim=original_vae_config.base_dim,
        z_dim=original_vae_config.z_dim,
        dim_mult=original_vae_config.dim_mult,
        num_res_blocks=original_vae_config.num_res_blocks,
        attn_scales=original_vae_config.attn_scales,
        temperal_downsample=original_vae_config.temperal_downsample,
        dropout=original_vae_config.dropout,
        input_channels=original_vae_config.input_channels,
        latents_mean=original_vae_config.latents_mean,
        latents_std=original_vae_config.latents_std,
    )
    neuron_vae.load_state_dict(pipe.vae.state_dict())
    neuron_vae = neuron_vae.to(dtype)

    # Get encoder
    cpu_encoder = neuron_vae.encoder
    cpu_encoder.eval()

    # Create test input
    print(f"\nCreating test input: (1, 3, {temporal_frames}, {height}, {width})")
    test_input = torch.randn(1, 3, temporal_frames, height, width, dtype=dtype)

    # CPU inference
    print("Running CPU inference...")
    with torch.no_grad():
        cpu_output = cpu_encoder(test_input)

    # Load and run Neuron model
    vae_encoder_path = f"{args.compiled_models_dir}/vae_encoder/model.pt"
    if not os.path.exists(vae_encoder_path):
        print(f"\nERROR: Compiled VAE encoder not found at {vae_encoder_path}")
        print("Please run compile_vae.py first.")
        return None

    print(f"Loading compiled encoder from {vae_encoder_path}...")
    import torch_neuronx
    compiled_encoder = torch.jit.load(vae_encoder_path)

    print("Running Neuron inference...")
    with torch.no_grad():
        neuron_output = compiled_encoder(test_input)

    # Compare results
    metrics = compute_metrics(cpu_output, neuron_output, "VAE Encoder")

    return metrics


def test_vae_decoder(args):
    """Test VAE decoder: CPU vs Neuron."""
    print("\n" + "="*60)
    print("Testing VAE Decoder")
    print("="*60)

    dtype = torch.bfloat16
    height, width = args.height, args.width
    latent_height = height // 8
    latent_width = width // 8
    temporal_frames = 1
    z_dim = 16  # QwenImage VAE z_dim

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Create Neuron-compatible VAE with same weights
    print("Creating Neuron-compatible VAE...")
    original_vae_config = pipe.vae.config
    neuron_vae = NeuronAutoencoder(
        base_dim=original_vae_config.base_dim,
        z_dim=original_vae_config.z_dim,
        dim_mult=original_vae_config.dim_mult,
        num_res_blocks=original_vae_config.num_res_blocks,
        attn_scales=original_vae_config.attn_scales,
        temperal_downsample=original_vae_config.temperal_downsample,
        dropout=original_vae_config.dropout,
        input_channels=original_vae_config.input_channels,
        latents_mean=original_vae_config.latents_mean,
        latents_std=original_vae_config.latents_std,
    )
    neuron_vae.load_state_dict(pipe.vae.state_dict())
    neuron_vae = neuron_vae.to(dtype)

    # Get decoder
    cpu_decoder = neuron_vae.decoder
    cpu_decoder.eval()

    # Create test input (latent space)
    print(f"\nCreating test input: (1, {z_dim}, {temporal_frames}, {latent_height}, {latent_width})")
    test_input = torch.randn(1, z_dim, temporal_frames, latent_height, latent_width, dtype=dtype)

    # CPU inference
    print("Running CPU inference...")
    with torch.no_grad():
        cpu_output = cpu_decoder(test_input)

    # Load and run Neuron model
    vae_decoder_path = f"{args.compiled_models_dir}/vae_decoder/model.pt"
    if not os.path.exists(vae_decoder_path):
        print(f"\nERROR: Compiled VAE decoder not found at {vae_decoder_path}")
        print("Please run compile_vae.py first.")
        return None

    print(f"Loading compiled decoder from {vae_decoder_path}...")
    import torch_neuronx
    compiled_decoder = torch.jit.load(vae_decoder_path)

    print("Running Neuron inference...")
    with torch.no_grad():
        neuron_output = compiled_decoder(test_input)

    # Compare results
    metrics = compute_metrics(cpu_output, neuron_output, "VAE Decoder")

    # Additional: visualize difference if output is image-like
    if args.save_images:
        save_comparison_images(cpu_output, neuron_output, "vae_decoder", args)

    return metrics


def test_vae_roundtrip(args):
    """Test full VAE roundtrip: encode -> decode."""
    print("\n" + "="*60)
    print("Testing VAE Roundtrip (Encode -> Decode)")
    print("="*60)

    dtype = torch.bfloat16
    height, width = args.height, args.width
    temporal_frames = 1

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Create Neuron-compatible VAE with same weights
    original_vae_config = pipe.vae.config
    neuron_vae = NeuronAutoencoder(
        base_dim=original_vae_config.base_dim,
        z_dim=original_vae_config.z_dim,
        dim_mult=original_vae_config.dim_mult,
        num_res_blocks=original_vae_config.num_res_blocks,
        attn_scales=original_vae_config.attn_scales,
        temperal_downsample=original_vae_config.temperal_downsample,
        dropout=original_vae_config.dropout,
        input_channels=original_vae_config.input_channels,
        latents_mean=original_vae_config.latents_mean,
        latents_std=original_vae_config.latents_std,
    )
    neuron_vae.load_state_dict(pipe.vae.state_dict())
    neuron_vae = neuron_vae.to(dtype)
    neuron_vae.eval()

    # Create test image input
    print(f"\nCreating test input: (1, 3, {temporal_frames}, {height}, {width})")
    test_input = torch.randn(1, 3, temporal_frames, height, width, dtype=dtype)

    # CPU roundtrip
    print("Running CPU roundtrip...")
    with torch.no_grad():
        cpu_encoded = neuron_vae.encoder(test_input)
        cpu_quant = neuron_vae.quant_conv(cpu_encoded)
        # Take mean (first half of channels)
        cpu_latent = cpu_quant[:, :16, :, :, :]
        cpu_post_quant = neuron_vae.post_quant_conv(cpu_latent)
        cpu_decoded = neuron_vae.decoder(cpu_post_quant)

    # Check compiled models exist
    encoder_path = f"{args.compiled_models_dir}/vae_encoder/model.pt"
    decoder_path = f"{args.compiled_models_dir}/vae_decoder/model.pt"
    quant_conv_path = f"{args.compiled_models_dir}/quant_conv/model.pt"
    post_quant_conv_path = f"{args.compiled_models_dir}/post_quant_conv/model.pt"

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"\nERROR: Compiled VAE models not found")
        return None

    # Load compiled models
    print("Loading compiled models...")
    import torch_neuronx
    compiled_encoder = torch.jit.load(encoder_path)
    compiled_decoder = torch.jit.load(decoder_path)

    # Load quant_conv and post_quant_conv if available
    compiled_quant_conv = None
    compiled_post_quant_conv = None
    if os.path.exists(quant_conv_path):
        print(f"  Loading quant_conv from {quant_conv_path}")
        compiled_quant_conv = torch.jit.load(quant_conv_path)
    else:
        print(f"  WARNING: quant_conv not compiled, using CPU version")

    if os.path.exists(post_quant_conv_path):
        print(f"  Loading post_quant_conv from {post_quant_conv_path}")
        compiled_post_quant_conv = torch.jit.load(post_quant_conv_path)
    else:
        print(f"  WARNING: post_quant_conv not compiled, using CPU version")

    # Neuron roundtrip
    print("Running Neuron roundtrip...")
    with torch.no_grad():
        neuron_encoded = compiled_encoder(test_input)

        # Use compiled quant_conv if available
        if compiled_quant_conv is not None:
            neuron_quant = compiled_quant_conv(neuron_encoded)
        else:
            neuron_quant = neuron_vae.quant_conv(neuron_encoded)

        neuron_latent = neuron_quant[:, :16, :, :, :]

        # Use compiled post_quant_conv if available
        if compiled_post_quant_conv is not None:
            neuron_post_quant = compiled_post_quant_conv(neuron_latent)
        else:
            neuron_post_quant = neuron_vae.post_quant_conv(neuron_latent)

        neuron_decoded = compiled_decoder(neuron_post_quant)

    # Compare intermediate results
    print("\n--- Intermediate Comparisons ---")
    compute_metrics(cpu_encoded, neuron_encoded, "Encoder Output")
    compute_metrics(cpu_quant, neuron_quant, "After quant_conv (full 32 channels)")
    compute_metrics(cpu_latent, neuron_latent, "Latent (first 16 channels)")
    compute_metrics(cpu_post_quant, neuron_post_quant, "After post_quant_conv")
    metrics = compute_metrics(cpu_decoded, neuron_decoded, "Final Decoded Output")

    # Additional test: Decoder with SAME input to isolate decoder error
    print("\n--- Decoder Isolation Test (same input) ---")
    with torch.no_grad():
        # Use CPU post_quant output as input to both decoders
        cpu_decoder_from_cpu_input = neuron_vae.decoder(cpu_post_quant)
        neuron_decoder_from_cpu_input = compiled_decoder(cpu_post_quant)
    compute_metrics(cpu_decoder_from_cpu_input, neuron_decoder_from_cpu_input,
                    "Decoder (same CPU input)")

    # Save comparison images
    if args.save_images:
        save_comparison_images(cpu_decoded, neuron_decoded, "vae_roundtrip", args)

    return metrics


def save_comparison_images(cpu_output, neuron_output, prefix, args):
    """Save CPU vs Neuron output as images for visual comparison."""
    import os

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy images (assume output is [-1, 1])
    cpu_img = ((cpu_output[0, :, 0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    neuron_img = ((neuron_output[0, :, 0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Compute difference (amplified for visibility)
    diff = np.abs(cpu_img.astype(float) - neuron_img.astype(float))
    diff_amplified = (diff * 10).clip(0, 255).astype(np.uint8)

    # Save images
    Image.fromarray(cpu_img).save(os.path.join(output_dir, f"{prefix}_cpu.png"))
    Image.fromarray(neuron_img).save(os.path.join(output_dir, f"{prefix}_neuron.png"))
    Image.fromarray(diff_amplified).save(os.path.join(output_dir, f"{prefix}_diff_10x.png"))

    print(f"\nSaved comparison images to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="VAE Unit Test: CPU vs Neuron")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--test", type=str, default="all",
                        choices=["encoder", "decoder", "roundtrip", "all"],
                        help="Which test to run")
    parser.add_argument("--save_images", action="store_true",
                        help="Save comparison images")
    args = parser.parse_args()

    print("="*60)
    print("VAE Unit Test: Comparing Neuron vs CPU Inference")
    print("="*60)
    print(f"Image size: {args.height}x{args.width}")
    print(f"Compiled models: {args.compiled_models_dir}")

    results = {}

    if args.test in ["encoder", "all"]:
        results["encoder"] = test_vae_encoder(args)

    if args.test in ["decoder", "all"]:
        results["decoder"] = test_vae_decoder(args)

    if args.test in ["roundtrip", "all"]:
        results["roundtrip"] = test_vae_roundtrip(args)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        if metrics:
            status = "PASS" if metrics["cosine_sim"] > 0.99 else "WARN" if metrics["cosine_sim"] > 0.95 else "FAIL"
            print(f"{name:15s}: Cosine Sim = {metrics['cosine_sim']:.6f}  Max AE = {metrics['max_abs_error']:.2e}  [{status}]")
        else:
            print(f"{name:15s}: SKIPPED (compiled model not found)")


if __name__ == "__main__":
    main()
