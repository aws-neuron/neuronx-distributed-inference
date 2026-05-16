"""
Wan2.2 VAE Decoder Compilation - V3 NoCache (Zero-argument feat_cache).

Key insight: In the NxDModel-based decoder, feat_cache is passed as 35 separate
input arguments (~960MB per call). But NxDModel doesn't modify input tensors
in-place, so feat_cache is effectively ALWAYS zeros between calls.

This version internalizes feat_cache as registered buffers (loaded once to device),
reducing NxDModel arguments from 35 to 1. Only x (~300KB) is transferred per call,
eliminating ~960MB data transfer overhead.

Optionally supports decoder_frames=3 to reduce total decoder calls from 11 to 8
for 81-frame video (21 latent frames). Default is decoder_frames=2 (same as V3).
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import argparse
from functools import reduce
import operator

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file


# feat_cache shapes for 512x512 / latent 32x32
def get_feat_cache_shapes(batch_size, latent_height, latent_width, dtype=torch.bfloat16):
    """Return the 34 feat_cache tensor shapes for the Wan decoder."""
    lh, lw = latent_height, latent_width
    return [
        (batch_size, 48, 2, lh, lw),           # 0: conv_in
        (batch_size, 1024, 2, lh, lw),          # 1-4: mid_block
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),          # 5-11: up_block_0
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh, lw),
        (batch_size, 1024, 2, lh*2, lw*2),      # 12-18: up_block_1
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*2, lw*2),
        (batch_size, 1024, 2, lh*4, lw*4),      # 19-24: up_block_2
        (batch_size, 512, 2, lh*4, lw*4),
        (batch_size, 512, 2, lh*4, lw*4),
        (batch_size, 512, 2, lh*4, lw*4),
        (batch_size, 512, 2, lh*4, lw*4),
        (batch_size, 512, 2, lh*4, lw*4),
        (batch_size, 512, 2, lh*8, lw*8),       # 25-33: up_block_3 + conv_out
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 256, 2, lh*8, lw*8),
        (batch_size, 12, 2, lh*8, lw*8),
    ]


class DecoderWrapperNoCache(nn.Module):
    """
    Decoder wrapper with feat_cache as registered buffers (not input arguments).

    Eliminates ~960MB per-call data transfer by keeping feat_cache on device.
    Only x (~300KB) is transferred per call.
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, decoder, feat_cache_shapes, dtype=torch.bfloat16):
        super().__init__()
        self.decoder = decoder

        # Register feat_cache as persistent buffers (loaded with weights, stay on device)
        for i, shape in enumerate(feat_cache_shapes):
            self.register_buffer(f'feat_cache_{i}', torch.zeros(shape, dtype=dtype))

    def forward(self, x):
        # Build feat_cache list from registered buffers (already on device)
        feat_cache = [
            getattr(self, f'feat_cache_{i}')
            for i in range(self.NUM_FEAT_CACHE)
        ]
        return self.decoder(x, feat_cache)


class PostQuantConvWrapper(nn.Module):
    def __init__(self, post_quant_conv):
        super().__init__()
        self.conv = post_quant_conv

    def forward(self, x):
        return self.conv(x)


def save_model_config(output_path, config):
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_v3_nocache(args):
    latent_height = args.height // 16
    latent_width = args.width // 16
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = args.decoder_frames
    in_channels = 48
    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 VAE Decoder V3 NoCache Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"Decoder frames: {decoder_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Key: feat_cache as buffers -> only 1 input argument")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Decoder (bfloat16, 1 input arg) ==========
        print("\nPreparing decoder (bfloat16, no external feat_cache)...")
        decoder = vae.decoder.to(dtype).eval()

        feat_cache_shapes = get_feat_cache_shapes(batch_size, latent_height, latent_width, dtype)
        wrapper = DecoderWrapperNoCache(decoder, feat_cache_shapes, dtype)

        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=dtype,
        )

        print(f"  Input: {decoder_input.shape} ({decoder_input.nelement()*2/1024:.0f}KB)")
        print(f"  Buffers: {sum(reduce(operator.mul, s) for s in feat_cache_shapes)*2/1024/1024:.0f}MB (on device, no transfer)")

        builder = ModelBuilder(model=wrapper)
        print("Tracing...")
        builder.trace(kwargs={"x": decoder_input}, tag="decode")

        print("Compiling...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        traced = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{compiled_models_dir}/decoder_nocache"
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving to {output_path}...")
        traced.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights (includes decoder weights + feat_cache buffers)
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)
        checkpoint = wrapper.state_dict()
        save_file(checkpoint, os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "decoder_frames": decoder_frames,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "bfloat16",
            "nocache": True,
        }
        save_model_config(output_path, config)

        # ========== Compile post_quant_conv (float32, same as v3) ==========
        latent_frames = (args.num_frames - 1) // 4 + 1
        print("\nCompiling post_quant_conv (float32)...")
        pqc_wrapper = PostQuantConvWrapper(vae.post_quant_conv)
        pqc_input = torch.rand(
            (batch_size, in_channels, latent_frames, latent_height, latent_width),
            dtype=torch.float32,
        )

        pqc_builder = ModelBuilder(model=pqc_wrapper)
        pqc_builder.trace(kwargs={"x": pqc_input}, tag="conv")
        traced_pqc = pqc_builder.compile(
            compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
            compiler_workdir=args.compiler_workdir,
        )

        pqc_output_path = f"{compiled_models_dir}/post_quant_conv"
        os.makedirs(pqc_output_path, exist_ok=True)
        traced_pqc.save(os.path.join(pqc_output_path, "nxd_model.pt"))

        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        pqc_checkpoint = pqc_wrapper.state_dict()
        save_file(pqc_checkpoint, os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

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
        print(f"Decoder: {output_path} (1 input arg, ~300KB per call)")
        print(f"post_quant_conv: {pqc_output_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--decoder_frames", type=int, default=2)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    parser.add_argument("--cache_dir", type=str, default="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir")
    args = parser.parse_args()

    compile_decoder_v3_nocache(args)
