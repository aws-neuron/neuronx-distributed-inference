import os

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # For trn2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # For trn2

compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """  #  --verbose=INFO
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import argparse
import torch_neuronx
from torch import nn

from diffusers import QwenImageEditPlusPipeline
from neuron_commons import attention_wrapper, f32Wrapper

# Import modified VAE that uses 'nearest' instead of 'nearest-exact'
# (Neuron doesn't support 'nearest-exact' interpolation mode)
from autoencoder_kl_qwenimage_neuron import AutoencoderKLQwenImage as NeuronAutoencoder

# Override SDPA
torch.nn.functional.scaled_dot_product_attention = attention_wrapper

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


class VAEEncoderWrapper(nn.Module):
    """Wrapper for VAE encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class VAEDecoderWrapper(nn.Module):
    """Wrapper for VAE decoder."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def compile_vae(args):
    """
    Compile VAE for QwenImage.

    Note: QwenImage VAE uses 3D convolutions (for video/multi-frame support).
    Input shape: (batch, channels, temporal_frames, height, width)
    For single image inference, temporal_frames=1.
    """
    latent_height = args.height // 8
    latent_width = args.width // 8
    temporal_frames = args.temporal_frames  # Number of temporal frames
    latent_temporal = temporal_frames  # Temporal dimension in latent space

    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    batch_size = args.batch_size
    dtype = torch.bfloat16

    load_kwargs = {"local_files_only": True, "torch_dtype": dtype}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **load_kwargs)

    # Replace VAE with Neuron-compatible version (uses 'nearest' instead of 'nearest-exact')
    print("Replacing VAE with Neuron-compatible version...")
    original_vae_config = pipe.vae.config
    neuron_vae = NeuronAutoencoder(
        base_dim=original_vae_config.base_dim,
        z_dim=original_vae_config.z_dim,
        dim_mult=original_vae_config.dim_mult,
        num_res_blocks=original_vae_config.num_res_blocks,
        attn_scales=original_vae_config.attn_scales,
        temperal_downsample=original_vae_config.temperal_downsample,
        dropout=original_vae_config.dropout,
        input_channels=getattr(original_vae_config, "input_channels", 3),
        latents_mean=original_vae_config.latents_mean,
        latents_std=original_vae_config.latents_std,
    )
    # Load weights from original VAE
    neuron_vae.load_state_dict(pipe.vae.state_dict())
    neuron_vae = neuron_vae.to(dtype)
    pipe.vae = neuron_vae

    z_dim = pipe.vae.config.z_dim  # 16 for QwenImage VAE

    # Compile VAE Encoder
    print("Compiling VAE encoder...")
    print(
        f"  Input shape: ({batch_size}, 3, {temporal_frames}, {args.height}, {args.width})"
    )
    encoder = pipe.vae.encoder
    encoder.eval()
    upcast_norms_to_f32(encoder)

    with torch.no_grad():
        # Encoder input: (batch, channels, temporal_frames, height, width) - 5D for Conv3d
        encoder_input = torch.rand(
            (batch_size, 3, temporal_frames, args.height, args.width), dtype=dtype
        )
        compiled_encoder = torch_neuronx.trace(
            encoder,
            encoder_input,
            compiler_workdir=f"{compiler_workdir}/vae_encoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

        encoder_dir = f"{compiled_models_dir}/vae_encoder"
        if not os.path.exists(encoder_dir):
            os.makedirs(encoder_dir)
        torch.jit.save(compiled_encoder, f"{encoder_dir}/model.pt")
        print(f"VAE encoder compiled and saved to {encoder_dir}")

    # Compile VAE Decoder
    # NOTE: At LNC=2 (trn2.3xlarge default), NEURON_CUSTOM_SILU=1 and
    # NEURON_FUSE_SOFTMAX=1 cause an internal compiler error (NCC_IBIR182)
    # for the VAE decoder. The encoder compiles fine with these flags.
    # We disable them for decoder compilation and restore afterward.
    saved_silu = os.environ.get("NEURON_CUSTOM_SILU")
    saved_softmax = os.environ.get("NEURON_FUSE_SOFTMAX")
    os.environ["NEURON_CUSTOM_SILU"] = "0"
    os.environ["NEURON_FUSE_SOFTMAX"] = "0"

    print("Compiling VAE decoder...")
    print(
        f"  Input shape: ({batch_size}, {z_dim}, {latent_temporal}, {latent_height}, {latent_width})"
    )
    print(
        f"  NOTE: NEURON_CUSTOM_SILU and NEURON_FUSE_SOFTMAX disabled for decoder (LNC=2 compatibility)"
    )
    decoder = pipe.vae.decoder
    decoder.eval()
    upcast_norms_to_f32(decoder)

    with torch.no_grad():
        # Decoder input: (batch, z_dim, temporal_frames, latent_height, latent_width) - 5D
        decoder_input = torch.rand(
            (batch_size, z_dim, latent_temporal, latent_height, latent_width),
            dtype=dtype,
        )
        compiled_decoder = torch_neuronx.trace(
            decoder,
            decoder_input,
            compiler_workdir=f"{compiler_workdir}/vae_decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False,
        )

        decoder_dir = f"{compiled_models_dir}/vae_decoder"
        if not os.path.exists(decoder_dir):
            os.makedirs(decoder_dir)
        torch.jit.save(compiled_decoder, f"{decoder_dir}/model.pt")
        print(f"VAE decoder compiled and saved to {decoder_dir}")

    # Restore NEURON_CUSTOM_SILU and NEURON_FUSE_SOFTMAX after decoder compilation
    if saved_silu is not None:
        os.environ["NEURON_CUSTOM_SILU"] = saved_silu
    if saved_softmax is not None:
        os.environ["NEURON_FUSE_SOFTMAX"] = saved_softmax

    # Compile quant_conv and post_quant_conv if they exist
    if hasattr(pipe.vae, "quant_conv") and pipe.vae.quant_conv is not None:
        print("Compiling quant_conv...")
        with torch.no_grad():
            quant_input = torch.rand(
                (batch_size, z_dim * 2, latent_temporal, latent_height, latent_width),
                dtype=dtype,
            )
            compiled_quant = torch_neuronx.trace(
                pipe.vae.quant_conv,
                quant_input,
                compiler_workdir=f"{compiler_workdir}/quant_conv",
                compiler_args=compiler_flags,
                inline_weights_to_neff=False,
            )
            quant_dir = f"{compiled_models_dir}/quant_conv"
            if not os.path.exists(quant_dir):
                os.makedirs(quant_dir)
            torch.jit.save(compiled_quant, f"{quant_dir}/model.pt")
            print(f"quant_conv compiled and saved to {quant_dir}")

    if hasattr(pipe.vae, "post_quant_conv") and pipe.vae.post_quant_conv is not None:
        print("Compiling post_quant_conv...")
        with torch.no_grad():
            post_quant_input = torch.rand(
                (batch_size, z_dim, latent_temporal, latent_height, latent_width),
                dtype=dtype,
            )
            compiled_post_quant = torch_neuronx.trace(
                pipe.vae.post_quant_conv,
                post_quant_input,
                compiler_workdir=f"{compiler_workdir}/post_quant_conv",
                compiler_args=compiler_flags,
                inline_weights_to_neff=False,
            )
            post_quant_dir = f"{compiled_models_dir}/post_quant_conv"
            if not os.path.exists(post_quant_dir):
                os.makedirs(post_quant_dir)
            torch.jit.save(compiled_post_quant, f"{post_quant_dir}/model.pt")
            print(f"post_quant_conv compiled and saved to {post_quant_dir}")

    # Save VAE config
    import json

    vae_config = {
        "height": args.height,
        "width": args.width,
        "temporal_frames": temporal_frames,
        "batch_size": batch_size,
        "z_dim": z_dim,
        "latent_height": latent_height,
        "latent_width": latent_width,
    }
    config_path = f"{compiled_models_dir}/vae_config.json"
    with open(config_path, "w") as f:
        json.dump(vae_config, f, indent=2)
    print(f"VAE config saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model (local dir or HuggingFace ID). If not set, uses MODEL_ID with CACHE_DIR",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated image (compile tile size)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated image (compile tile size)",
    )
    parser.add_argument(
        "--temporal_frames",
        type=int,
        default=1,
        help="Number of temporal frames (1 for single image)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for VAE (default: 1)"
    )
    parser.add_argument(
        "--compiler_workdir",
        type=str,
        default="compiler_workdir",
        help="Directory for compiler artifacts",
    )
    parser.add_argument(
        "--compiled_models_dir",
        type=str,
        default="compiled_models",
        help="Directory for compiled models",
    )
    args = parser.parse_args()

    # Override MODEL_ID and CACHE_DIR if model_path is provided
    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    print("=" * 60)
    print("VAE Compilation for Neuron")
    print("=" * 60)
    print(f"Compile tile size: {args.height}x{args.width}")
    print(f"Batch size: {args.batch_size}")
    print("")
    print("NOTE: For inference at larger resolutions (e.g., 1024x1024),")
    print("      tiled VAE processing will be used automatically.")
    print("      The VAE is compiled at this tile size for memory efficiency.")
    print("      With batch_size > 1, multiple tiles can be processed in parallel.")
    print("")

    compile_vae(args)
