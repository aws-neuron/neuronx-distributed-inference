"""
Wan2.2 VAE Decoder Compilation - V3 (Optimized ModelBuilder).

Key optimizations over V2:
1. Explicit compiler_args with --model-type=unet-inference passed to builder.compile()
   (V2 relied on env vars, but ModelBuilder defaults to --model-type=transformer
   which optimizes for attention patterns instead of Conv3D)
2. bfloat16 for decoder - halves memory bandwidth for all Conv3D operations
3. post_quant_conv kept in float32 (cheap, runs once, needs precision)

Note: world_size must match the transformer's NxDParallelState context.
The decoder weights are duplicated (not sharded) across all ranks.
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Environment compiler flags (applies to all compilations)
compiler_flags = """ --target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import argparse

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file


class DecoderWrapper(nn.Module):
    """
    Wrapper for VAE decoder to handle feat_cache as individual tensor arguments.
    ModelBuilder requires all inputs to be tensors (no lists).
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x,
                feat_cache_0, feat_cache_1, feat_cache_2, feat_cache_3, feat_cache_4,
                feat_cache_5, feat_cache_6, feat_cache_7, feat_cache_8, feat_cache_9,
                feat_cache_10, feat_cache_11, feat_cache_12, feat_cache_13, feat_cache_14,
                feat_cache_15, feat_cache_16, feat_cache_17, feat_cache_18, feat_cache_19,
                feat_cache_20, feat_cache_21, feat_cache_22, feat_cache_23, feat_cache_24,
                feat_cache_25, feat_cache_26, feat_cache_27, feat_cache_28, feat_cache_29,
                feat_cache_30, feat_cache_31, feat_cache_32, feat_cache_33):
        feat_cache = [
            feat_cache_0, feat_cache_1, feat_cache_2, feat_cache_3, feat_cache_4,
            feat_cache_5, feat_cache_6, feat_cache_7, feat_cache_8, feat_cache_9,
            feat_cache_10, feat_cache_11, feat_cache_12, feat_cache_13, feat_cache_14,
            feat_cache_15, feat_cache_16, feat_cache_17, feat_cache_18, feat_cache_19,
            feat_cache_20, feat_cache_21, feat_cache_22, feat_cache_23, feat_cache_24,
            feat_cache_25, feat_cache_26, feat_cache_27, feat_cache_28, feat_cache_29,
            feat_cache_30, feat_cache_31, feat_cache_32, feat_cache_33
        ]
        return self.decoder(x, feat_cache)


class PostQuantConvWrapper(nn.Module):
    """Wrapper for post_quant_conv."""
    def __init__(self, post_quant_conv):
        super().__init__()
        self.conv = post_quant_conv

    def forward(self, x):
        return self.conv(x)


def save_model_config(output_path, config):
    """Save model configuration."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_v3(args):
    """Compile VAE decoder V3 with optimized ModelBuilder settings."""
    latent_height = args.height // 16
    latent_width = args.width // 16
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = 2  # CACHE_T=2
    latent_frames = (args.num_frames - 1) // 4 + 1
    in_channels = 48
    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 VAE Decoder V3 Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"num_frames={args.num_frames} -> latent_frames={latent_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Decoder dtype: {dtype}")
    print(f"Compiler args: --model-type=unet-inference -O1 --auto-cast=none")
    print("=" * 60)

    # Load VAE in float32 first, then convert decoder to bfloat16
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
    )

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Decoder (bfloat16) ==========
        print("\nPreparing decoder (bfloat16)...")
        decoder = vae.decoder
        decoder = decoder.to(dtype)
        decoder.eval()

        # Prepare inputs in bfloat16
        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=dtype
        )

        # Create feat_cache in bfloat16
        feat_cache = [
            torch.rand((batch_size, 48, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=dtype),
            torch.rand((batch_size, 1024, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=dtype),
            torch.rand((batch_size, 512, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=dtype),
            torch.rand((batch_size, 12, 2, latent_height*8, latent_width*8), dtype=dtype),
        ]

        # Wrap decoder
        decoder_wrapper = DecoderWrapper(decoder)

        # Build trace kwargs
        trace_kwargs = {"x": decoder_input}
        for i, fc in enumerate(feat_cache):
            trace_kwargs[f"feat_cache_{i}"] = fc

        # Initialize ModelBuilder
        print("\nInitializing ModelBuilder for decoder...")
        decoder_builder = ModelBuilder(model=decoder_wrapper)

        print("Tracing decoder...")
        decoder_builder.trace(
            kwargs=trace_kwargs,
            tag="decode",
        )

        # KEY FIX: Pass explicit compiler_args to override ModelBuilder's default
        # --model-type=transformer. Without this, the compiler optimizes for
        # attention patterns instead of Conv3D operations.
        print("Compiling decoder...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        traced_decoder = decoder_builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save decoder
        decoder_output_path = f"{compiled_models_dir}/decoder"
        os.makedirs(decoder_output_path, exist_ok=True)
        print(f"Saving decoder to {decoder_output_path}...")
        traced_decoder.save(os.path.join(decoder_output_path, "nxd_model.pt"))

        # Save weights (single checkpoint, will be duplicated at runtime)
        print("Saving decoder weights...")
        decoder_weights_path = os.path.join(decoder_output_path, "weights")
        os.makedirs(decoder_weights_path, exist_ok=True)
        decoder_checkpoint = decoder_wrapper.state_dict()
        save_file(decoder_checkpoint, os.path.join(decoder_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        decoder_config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "latent_frames": latent_frames,
            "decoder_frames": decoder_frames,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "bfloat16",
        }
        save_model_config(decoder_output_path, decoder_config)

        # ========== Compile post_quant_conv (float32) ==========
        # post_quant_conv is cheap (runs once) and benefits from float32 precision
        print("\nCompiling post_quant_conv (float32)...")
        post_quant_conv_wrapper = PostQuantConvWrapper(vae.post_quant_conv)

        post_quant_conv_input = torch.rand(
            (batch_size, in_channels, latent_frames, latent_height, latent_width),
            dtype=torch.float32
        )

        pqc_builder = ModelBuilder(model=post_quant_conv_wrapper)

        print("Tracing post_quant_conv...")
        pqc_builder.trace(
            kwargs={"x": post_quant_conv_input},
            tag="conv",
        )

        print("Compiling post_quant_conv...")
        pqc_compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        traced_pqc = pqc_builder.compile(
            compiler_args=pqc_compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save post_quant_conv
        pqc_output_path = f"{compiled_models_dir}/post_quant_conv"
        os.makedirs(pqc_output_path, exist_ok=True)
        print(f"Saving post_quant_conv to {pqc_output_path}...")
        traced_pqc.save(os.path.join(pqc_output_path, "nxd_model.pt"))

        # Save weights
        print("Saving post_quant_conv weights...")
        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        pqc_checkpoint = post_quant_conv_wrapper.state_dict()
        save_file(pqc_checkpoint, os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        pqc_config = {
            "batch_size": batch_size,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "float32",
        }
        save_model_config(pqc_output_path, pqc_config)

        print("\n" + "=" * 60)
        print("Compilation Complete!")
        print("=" * 60)
        print(f"Decoder saved to: {decoder_output_path}")
        print(f"post_quant_conv saved to: {pqc_output_path}")
        print(f"\nKey optimizations:")
        print(f"  - compiler_args: --model-type=unet-inference (NOT transformer)")
        print(f"  - Decoder dtype: bfloat16 (2x less memory bandwidth)")
        print(f"  - post_quant_conv dtype: float32 (precision)")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 VAE Decoder V3")
    parser.add_argument("--height", type=int, default=512, help="Height of generated video")
    parser.add_argument("--width", type=int, default=512, help="Width of generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--tp_degree", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--world_size", type=int, default=8, help="World size (must match transformer)")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir", help="Compiler workdir")
    args = parser.parse_args()

    compile_decoder_v3(args)
