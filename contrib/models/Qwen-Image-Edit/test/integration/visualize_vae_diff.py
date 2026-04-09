#!/usr/bin/env python3
"""
VAE Visual Comparison: Generate side-by-side images to visualize differences

This script:
1. Takes a real image as input
2. Encodes it using both CPU and Neuron VAE
3. Decodes it using both CPU and Neuron VAE
4. Generates comparison images showing the differences

This is useful for visually identifying the source of blurry outputs.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Neuron environment before importing neuron libraries
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diffusers import QwenImageEditPlusPipeline
from neuron_qwen_image_edit.autoencoder_kl_qwenimage_neuron import AutoencoderKLQwenImage as NeuronAutoencoder


CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"


def tensor_to_pil(tensor, name=""):
    """Convert tensor to PIL image."""
    # Assume tensor is (B, C, T, H, W) or (B, C, H, W)
    if len(tensor.shape) == 5:
        tensor = tensor[0, :, 0]  # Take first batch, first frame
    elif len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first batch

    # Clamp to valid range
    tensor = torch.clamp(tensor, -1, 1)

    # Convert to numpy
    img = ((tensor.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5)
    img = img.clip(0, 255).astype(np.uint8)

    return Image.fromarray(img)


def create_diff_image(img1, img2, amplification=10):
    """Create a difference image between two PIL images."""
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)

    diff = np.abs(arr1 - arr2)
    diff_amplified = (diff * amplification).clip(0, 255).astype(np.uint8)

    return Image.fromarray(diff_amplified)


def add_label(img, label):
    """Add a text label to an image."""
    # Create a copy
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Add black background for text
    bbox = draw.textbbox((10, 10), label, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill="black")
    draw.text((10, 10), label, fill="white", font=font)

    return img


def test_vae_visual(args):
    """Generate visual comparison of VAE outputs."""
    print("\n" + "="*60)
    print("VAE Visual Comparison Test")
    print("="*60)

    dtype = torch.bfloat16
    height, width = args.height, args.width

    # Load original pipeline
    print("\nLoading original pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    # Create Neuron-compatible VAE
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
    neuron_vae.eval()

    # Load or create test image
    if args.input_image and os.path.exists(args.input_image):
        print(f"\nLoading input image: {args.input_image}")
        input_img = Image.open(args.input_image).convert('RGB')
        input_img = input_img.resize((width, height))
    else:
        print(f"\nCreating test image ({width}x{height})...")
        # Create a gradient image for testing
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        r = xv
        g = yv
        b = ((xv.astype(float) + yv.astype(float)) / 2).astype(np.uint8)
        input_img = Image.fromarray(np.stack([r, g, b], axis=-1))

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(input_img)).float() / 127.5 - 1
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
    img_tensor = img_tensor.to(dtype)

    print(f"  Input tensor shape: {img_tensor.shape}")

    # Load compiled models
    encoder_path = f"{args.compiled_models_dir}/vae_encoder/model.pt"
    decoder_path = f"{args.compiled_models_dir}/vae_decoder/model.pt"

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"\nERROR: Compiled VAE models not found")
        print(f"  Encoder: {encoder_path}")
        print(f"  Decoder: {decoder_path}")
        return

    print(f"\nLoading compiled models...")
    import torch_neuronx
    compiled_encoder = torch.jit.load(encoder_path)
    compiled_decoder = torch.jit.load(decoder_path)

    # === Test 1: Encoder comparison ===
    print("\n--- Encoder Comparison ---")
    with torch.no_grad():
        cpu_encoded = neuron_vae.encoder(img_tensor)
        neuron_encoded = compiled_encoder(img_tensor)

    enc_diff = torch.abs(cpu_encoded - neuron_encoded)
    print(f"  CPU encoded shape: {cpu_encoded.shape}")
    print(f"  Neuron encoded shape: {neuron_encoded.shape}")
    print(f"  Encoder Max Diff: {enc_diff.max().item():.6e}")
    print(f"  Encoder Mean Diff: {enc_diff.mean().item():.6e}")

    # === Test 2: Decoder comparison with SAME latent ===
    print("\n--- Decoder Comparison (Same Latent) ---")

    # Use CPU encoded latent
    cpu_quant = neuron_vae.quant_conv(cpu_encoded)
    cpu_latent = cpu_quant[:, :16, :, :, :]

    # Decode with both
    with torch.no_grad():
        cpu_post_quant = neuron_vae.post_quant_conv(cpu_latent)
        cpu_decoded = neuron_vae.decoder(cpu_post_quant)

        neuron_post_quant = neuron_vae.post_quant_conv(cpu_latent)
        neuron_decoded = compiled_decoder(neuron_post_quant)

    dec_diff = torch.abs(cpu_decoded - neuron_decoded)
    print(f"  CPU decoded shape: {cpu_decoded.shape}")
    print(f"  Neuron decoded shape: {neuron_decoded.shape}")
    print(f"  Decoder Max Diff: {dec_diff.max().item():.6e}")
    print(f"  Decoder Mean Diff: {dec_diff.mean().item():.6e}")

    # === Test 3: Full roundtrip comparison ===
    print("\n--- Full Roundtrip Comparison ---")

    # CPU roundtrip
    with torch.no_grad():
        cpu_full_encoded = neuron_vae.encoder(img_tensor)
        cpu_full_quant = neuron_vae.quant_conv(cpu_full_encoded)
        cpu_full_latent = cpu_full_quant[:, :16, :, :, :]
        cpu_full_post_quant = neuron_vae.post_quant_conv(cpu_full_latent)
        cpu_full_decoded = neuron_vae.decoder(cpu_full_post_quant)

    # Neuron roundtrip
    with torch.no_grad():
        neuron_full_encoded = compiled_encoder(img_tensor)
        neuron_full_quant = neuron_vae.quant_conv(neuron_full_encoded)
        neuron_full_latent = neuron_full_quant[:, :16, :, :, :]
        neuron_full_post_quant = neuron_vae.post_quant_conv(neuron_full_latent)
        neuron_full_decoded = compiled_decoder(neuron_full_post_quant)

    full_diff = torch.abs(cpu_full_decoded - neuron_full_decoded)
    print(f"  Full Roundtrip Max Diff: {full_diff.max().item():.6e}")
    print(f"  Full Roundtrip Mean Diff: {full_diff.mean().item():.6e}")

    # Cosine similarity
    cpu_flat = cpu_full_decoded.float().flatten()
    neuron_flat = neuron_full_decoded.float().flatten()
    cosine_sim = F.cosine_similarity(cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)).item()
    print(f"  Cosine Similarity: {cosine_sim:.6f}")

    # === Generate comparison images ===
    print("\n--- Generating Comparison Images ---")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to images
    input_pil = input_img
    cpu_decoded_pil = tensor_to_pil(cpu_full_decoded, "CPU")
    neuron_decoded_pil = tensor_to_pil(neuron_full_decoded, "Neuron")
    diff_pil = create_diff_image(cpu_decoded_pil, neuron_decoded_pil, amplification=20)

    # Add labels
    input_labeled = add_label(input_pil, "Original")
    cpu_labeled = add_label(cpu_decoded_pil, "CPU Decoded")
    neuron_labeled = add_label(neuron_decoded_pil, "Neuron Decoded")
    diff_labeled = add_label(diff_pil, "Diff (20x)")

    # Create comparison grid
    grid_width = width * 2
    grid_height = height * 2
    comparison = Image.new('RGB', (grid_width, grid_height))
    comparison.paste(input_labeled, (0, 0))
    comparison.paste(cpu_labeled, (width, 0))
    comparison.paste(neuron_labeled, (0, height))
    comparison.paste(diff_labeled, (width, height))

    # Save images
    output_path = os.path.join(output_dir, "vae_comparison.png")
    comparison.save(output_path)
    print(f"\nComparison image saved to: {output_path}")

    # Save individual images
    input_pil.save(os.path.join(output_dir, "vae_input.png"))
    cpu_decoded_pil.save(os.path.join(output_dir, "vae_cpu_decoded.png"))
    neuron_decoded_pil.save(os.path.join(output_dir, "vae_neuron_decoded.png"))
    diff_pil.save(os.path.join(output_dir, "vae_diff_20x.png"))

    print("\nIndividual images saved:")
    print(f"  - {output_dir}/vae_input.png")
    print(f"  - {output_dir}/vae_cpu_decoded.png")
    print(f"  - {output_dir}/vae_neuron_decoded.png")
    print(f"  - {output_dir}/vae_diff_20x.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Encoder Max Diff:     {enc_diff.max().item():.6e}")
    print(f"Decoder Max Diff:     {dec_diff.max().item():.6e}")
    print(f"Roundtrip Max Diff:   {full_diff.max().item():.6e}")
    print(f"Cosine Similarity:    {cosine_sim:.6f}")

    if cosine_sim > 0.999:
        print("\nConclusion: VAE output is nearly identical. Issue likely elsewhere.")
    elif cosine_sim > 0.99:
        print("\nConclusion: Minor VAE differences. May contribute to slight blur.")
    elif cosine_sim > 0.95:
        print("\nConclusion: Significant VAE differences. Likely causing blur.")
    else:
        print("\nConclusion: Major VAE differences. Primary cause of blur!")


def main():
    parser = argparse.ArgumentParser(description="VAE Visual Comparison")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--compiled_models_dir", type=str,
                        default=COMPILED_MODELS_DIR,
                        help="Directory containing compiled models")
    parser.add_argument("--input_image", type=str, default=None,
                        help="Optional input image path")
    args = parser.parse_args()

    test_vae_visual(args)


if __name__ == "__main__":
    main()
