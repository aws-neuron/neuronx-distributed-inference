"""
VAE Compilation for LongCat-Image-Edit (Standard 2D AutoencoderKL / FLUX VAE).

LongCat uses a standard 2D AutoencoderKL (FLUX-style) -- much simpler than
the reference's 3D causal VAE.

Config: latent_channels=16, block_out_channels=[128,256,512,512], no quant_conv
Input: [B, 3, H, W] (standard 2D images)
Latent: [B, 16, H//8, W//8]

Compilation: torch_neuronx.trace() on single device
"""

import os

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = " --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import json
import torch_neuronx
from torch import nn

from diffusers import LongCatImageEditPipeline
from neuron_commons import f32Wrapper, upcast_norms_to_f32

# Override SDPA for VAE tracing
from neuron_commons import attention_wrapper
torch.nn.functional.scaled_dot_product_attention = attention_wrapper

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"


def fix_nearest_exact(module):
    """
    Fix 'nearest-exact' interpolation mode to 'nearest' for Neuron compatibility.

    Neuron doesn't support 'nearest-exact' mode. We monkey-patch the upsample
    modules to use 'nearest' instead.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Upsample):
            if child.mode == 'nearest-exact':
                child.mode = 'nearest'
                print(f"  Fixed {name}: nearest-exact -> nearest")
        elif hasattr(child, 'mode') and getattr(child, 'mode', None) == 'nearest-exact':
            child.mode = 'nearest'
            print(f"  Fixed {name}: nearest-exact -> nearest")
        else:
            fix_nearest_exact(child)


class VAEEncoderWrapper(nn.Module):
    """Wrapper for VAE encoder to trace with torch_neuronx."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        # Encode and return the latent distribution mean (for deterministic encoding)
        h = self.vae.encoder(x)
        if hasattr(self.vae, 'quant_conv') and self.vae.quant_conv is not None:
            h = self.vae.quant_conv(h)
        # Return moments (mean and logvar concatenated)
        return h


class VAEDecoderWrapper(nn.Module):
    """Wrapper for VAE decoder to trace with torch_neuronx."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        if hasattr(self.vae, 'post_quant_conv') and self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)
        return self.vae.decoder(z)


def compile_vae(args):
    """
    Compile 2D AutoencoderKL for LongCat-Image-Edit.

    This is a standard 2D VAE (not 3D like Qwen reference).
    Input: [B, 3, H, W] for encoder
    Input: [B, latent_channels, H//8, W//8] for decoder
    """
    latent_height = args.height // 8
    latent_width = args.width // 8
    batch_size = args.batch_size
    dtype = torch.bfloat16

    load_kwargs = {"local_files_only": True, "torch_dtype": dtype}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)

    vae = pipe.vae
    vae.eval()

    # Get latent channels from config
    latent_channels = vae.config.latent_channels  # 16 for FLUX VAE
    print(f"  VAE config: latent_channels={latent_channels}")
    print(f"  VAE config: block_out_channels={vae.config.block_out_channels}")

    # Fix nearest-exact interpolation
    print("Fixing nearest-exact interpolation...")
    fix_nearest_exact(vae)

    # Upcast norms to float32
    print("Upcasting normalization layers to float32...")
    upcast_norms_to_f32(vae)

    # Compile VAE Encoder
    print("\nCompiling VAE encoder...")
    print(f"  Input shape: ({batch_size}, 3, {args.height}, {args.width})")

    encoder_wrapper = VAEEncoderWrapper(vae)
    encoder_wrapper.eval()

    with torch.no_grad():
        encoder_input = torch.rand((batch_size, 3, args.height, args.width), dtype=dtype)
        compiled_encoder = torch_neuronx.trace(
            encoder_wrapper,
            encoder_input,
            compiler_workdir=f"{args.compiler_workdir}/vae_encoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

        encoder_dir = f"{args.compiled_models_dir}/vae_encoder"
        os.makedirs(encoder_dir, exist_ok=True)
        torch.jit.save(compiled_encoder, f"{encoder_dir}/model.pt")
        print(f"VAE encoder compiled and saved to {encoder_dir}")

    # Compile VAE Decoder
    print("\nCompiling VAE decoder...")
    print(f"  Input shape: ({batch_size}, {latent_channels}, {latent_height}, {latent_width})")

    decoder_wrapper = VAEDecoderWrapper(vae)
    decoder_wrapper.eval()

    with torch.no_grad():
        decoder_input = torch.rand((batch_size, latent_channels, latent_height, latent_width), dtype=dtype)
        compiled_decoder = torch_neuronx.trace(
            decoder_wrapper,
            decoder_input,
            compiler_workdir=f"{args.compiler_workdir}/vae_decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

        decoder_dir = f"{args.compiled_models_dir}/vae_decoder"
        os.makedirs(decoder_dir, exist_ok=True)
        torch.jit.save(compiled_decoder, f"{decoder_dir}/model.pt")
        print(f"VAE decoder compiled and saved to {decoder_dir}")

    # Save VAE config
    vae_config = {
        "height": args.height,
        "width": args.width,
        "batch_size": batch_size,
        "latent_channels": latent_channels,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "vae_type": "2d_autoencoder_kl",
        "scaling_factor": getattr(vae.config, 'scaling_factor', 0.3611),
        "shift_factor": getattr(vae.config, 'shift_factor', 0.1159),
    }
    config_path = f"{args.compiled_models_dir}/vae_config.json"
    with open(config_path, "w") as f:
        json.dump(vae_config, f, indent=2)
    print(f"\nVAE config saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--height", type=int, default=512, help="VAE tile height")
    parser.add_argument("--width", type=int, default=512, help="VAE tile width")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    args = parser.parse_args()

    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    print("=" * 60)
    print("VAE Compilation for LongCat-Image-Edit (2D AutoencoderKL)")
    print("=" * 60)
    print(f"Compile tile size: {args.height}x{args.width}")
    print(f"Batch size: {args.batch_size}")
    print()
    print("NOTE: For inference at larger resolutions (e.g., 1024x1024),")
    print("      tiled VAE processing will be used automatically.")
    print()

    compile_vae(args)
