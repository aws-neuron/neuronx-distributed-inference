"""
Wan2.2 VAE Encoder Compilation - V3 (torch_neuronx.trace).

For Image-to-Video (I2V): encodes a single input image into latent space.

Key design (aligned with hf_pretrained_qwen_image_edit/compile_vae.py):
1. torch_neuronx.trace() — same API as Qwen VAE encoder
2. bfloat16 with upcast_norms_to_f32 for GroupNorm/LayerNorm
3. attention_wrapper for SDPA override
4. Input: post-patchify (1, 12, 1, 256, 256)
5. --model-type=unet-inference in NEURON_CC_FLAGS
6. encoder_frames=1 (I2V encodes 1 image)
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference -O1 --auto-cast=none --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import torch_neuronx
import argparse

from neuron_commons import attention_wrapper, f32Wrapper

# Override SDPA (must be done before tracing)
torch.nn.functional.scaled_dot_product_attention = attention_wrapper


class EncoderWrapper(nn.Module):
    """Simple wrapper for VAE encoder."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class QuantConvWrapper(nn.Module):
    """Wrapper for quant_conv."""
    def __init__(self, quant_conv):
        super().__init__()
        self.conv = quant_conv

    def forward(self, x):
        return self.conv(x)


def upcast_norms_to_f32(module):
    """Upcast GroupNorm/LayerNorm to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def save_model_config(output_path, config):
    """Save model configuration."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_encoder_v3(args):
    """Compile VAE encoder V3 with torch_neuronx.trace() (like Qwen VAE)."""
    compiled_models_dir = args.compiled_models_dir
    height = args.height
    width = args.width

    batch_size = 1
    encoder_frames = 1
    patch_size = 2

    # Post-patchify dimensions
    in_channels = 3 * patch_size * patch_size  # 12
    patchified_height = height // patch_size    # 256
    patchified_width = width // patch_size      # 256

    # Encoder output spatial dims (8x spatial downsampling within encoder)
    latent_height = patchified_height // 8  # 32
    latent_width = patchified_width // 8    # 32

    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 VAE Encoder V3 Compilation (torch_neuronx.trace)")
    print("=" * 60)
    print(f"Resolution: {height}x{width}")
    print(f"Encoder input (post-patchify): ({batch_size}, {in_channels}, {encoder_frames}, {patchified_height}, {patchified_width})")
    print(f"Encoder output spatial: {latent_height}x{latent_width}")
    print(f"Encoder dtype: {dtype}")
    print(f"attention_wrapper: enabled")
    print(f"upcast_norms_to_f32: enabled")
    print(f"Compiler flags: {compiler_flags.strip()}")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
    )

    # ========== Compile Encoder (bfloat16) ==========
    print("\nPreparing encoder (bfloat16 + upcast norms)...")
    encoder = vae.encoder
    encoder = encoder.to(dtype)
    encoder.eval()
    upcast_norms_to_f32(encoder)

    encoder_wrapper = EncoderWrapper(encoder)

    # Input: post-patchify shape, 1 frame
    encoder_input = torch.rand(
        (batch_size, in_channels, encoder_frames, patchified_height, patchified_width),
        dtype=dtype
    )
    print(f"Encoder input shape: {encoder_input.shape}")

    print("\nTracing encoder...")
    with torch.no_grad():
        compiled_encoder = torch_neuronx.trace(
            encoder_wrapper,
            encoder_input,
            compiler_workdir=f"{args.compiler_workdir}/vae_encoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

    # Save encoder
    encoder_output_path = f"{compiled_models_dir}/encoder"
    os.makedirs(encoder_output_path, exist_ok=True)
    print(f"Saving encoder to {encoder_output_path}...")
    torch.jit.save(compiled_encoder, os.path.join(encoder_output_path, "model.pt"))

    # Save config
    encoder_config = {
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "patchified_height": patchified_height,
        "patchified_width": patchified_width,
        "encoder_frames": encoder_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "dtype": "bfloat16",
        "includes_patchify": False,
    }
    save_model_config(encoder_output_path, encoder_config)

    # ========== Compile quant_conv (bfloat16) ==========
    print("\nCompiling quant_conv (bfloat16)...")
    quant_conv = vae.quant_conv.to(dtype)
    quant_conv.eval()

    z_channels = vae.config.z_dim * 2  # 32
    quant_conv_input = torch.rand(
        (batch_size, z_channels, encoder_frames, latent_height, latent_width),
        dtype=dtype
    )
    print(f"quant_conv input shape: {quant_conv_input.shape}")

    with torch.no_grad():
        compiled_qc = torch_neuronx.trace(
            quant_conv,
            quant_conv_input,
            compiler_workdir=f"{args.compiler_workdir}/quant_conv",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

    # Save quant_conv
    qc_output_path = f"{compiled_models_dir}/quant_conv"
    os.makedirs(qc_output_path, exist_ok=True)
    print(f"Saving quant_conv to {qc_output_path}...")
    torch.jit.save(compiled_qc, os.path.join(qc_output_path, "model.pt"))

    qc_config = {
        "batch_size": batch_size,
        "z_channels": z_channels,
        "encoder_frames": encoder_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "dtype": "bfloat16",
    }
    save_model_config(qc_output_path, qc_config)

    print("\n" + "=" * 60)
    print("Compilation Complete!")
    print("=" * 60)
    print(f"Encoder saved to: {encoder_output_path}")
    print(f"quant_conv saved to: {qc_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 VAE Encoder V3")
    parser.add_argument("--height", type=int, default=512, help="Height of generated video")
    parser.add_argument("--width", type=int, default=512, help="Width of generated video")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir", help="Compiler workdir")
    parser.add_argument("--tp_degree", type=int, default=8, help="(unused, for script compatibility)")
    parser.add_argument("--world_size", type=int, default=8, help="(unused, for script compatibility)")
    args = parser.parse_args()

    compile_encoder_v3(args)
