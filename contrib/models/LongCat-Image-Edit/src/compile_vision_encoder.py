"""
Vision Encoder Compilation using ModelBuilder API for TP=4 Acceleration.

Compiles the Qwen2.5-VL Vision Encoder (shared between Qwen-Image-Edit and
LongCat-Image-Edit) using ModelBuilder API with tp_degree=4 and world_size=8.

Key features:
- Float32 precision for accuracy (required for vision encoder)
- Vision encoder hidden_size=1280, QKV=3840, MLP intermediate=3420
- TP=4 works: 3840/4=960, 3420/4=855 (both divisible)
- Uses native F.scaled_dot_product_attention (no monkey-patch needed)

Usage:
    python compile_vision_encoder.py --image_size 448
"""

import os
import json
import gc

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse

from diffusers import LongCatImageEditPipeline

from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state

from neuron_parallel_utils import shard_vision_attention_fp32, shard_vision_mlp_fp32, get_sharded_data

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"


def load_pipeline(dtype=torch.float32):
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x, *args, **kwargs):
        t = x.dtype
        output = self.original(x.to(torch.float32), *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


class NeuronVisionEncoder(nn.Module):
    """Neuron-optimized Qwen2.5-VL Vision Encoder with TP=4, float32."""

    def __init__(self, original_visual, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.visual = original_visual
        self.embed_dim = original_visual.config.hidden_size
        self.num_heads = original_visual.config.num_heads

        print(f"  Vision encoder: embed_dim={self.embed_dim}, num_heads={self.num_heads}")

        for i, block in enumerate(self.visual.blocks):
            if hasattr(block, 'attn'):
                block.attn = shard_vision_attention_fp32(tp_degree, block.attn)
            if hasattr(block, 'mlp'):
                block.mlp = shard_vision_mlp_fp32(block.mlp)
            if i == 0:
                print(f"  Sharded block 0")
        print(f"  Sharded all {len(self.visual.blocks)} blocks")

        upcast_norms_to_f32(self.visual)

    def forward(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw)


class TracingWrapper(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
    def forward(self, pixel_values, grid_thw):
        return self.vision_encoder(pixel_values, grid_thw)


def compile_vision_encoder(args):
    tp_degree = 4
    world_size = 8
    image_size = args.image_size
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2

    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w
    channels_per_patch = 3 * temporal_patch_size * patch_size * patch_size  # 1176

    print("=" * 60)
    print("Compiling Vision Encoder (TP=4, float32)")
    print("=" * 60)
    print(f"  Image: {image_size}x{image_size}, Patches: {num_patches}")

    sample_pixel_values = torch.randn(num_patches, channels_per_patch, dtype=torch.float32)
    sample_grid_thw = torch.tensor([[1, num_patches_h, num_patches_w]], dtype=torch.int64)

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("Loading model...")
        pipe = load_pipeline(torch.float32)

        original_visual = pipe.text_encoder.model.visual
        unsharded_state = original_visual.state_dict()

        print(f"\nCreating Neuron vision encoder (TP={tp_degree})...")
        neuron_ve = NeuronVisionEncoder(original_visual, tp_degree)
        neuron_ve = neuron_ve.to(torch.float32)
        neuron_ve.eval()

        del pipe
        gc.collect()

        model = TracingWrapper(neuron_ve)

        builder = ModelBuilder(model=model)
        print("Tracing...")
        builder.trace(
            kwargs={"pixel_values": sample_pixel_values, "grid_thw": sample_grid_thw},
            tag="inference",
        )

        print("Compiling...")
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=args.compiler_workdir,
        )

        output_path = f"{args.compiled_models_dir}/vision_encoder"
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        checkpoint = {}
        for key, value in model.state_dict().items():
            orig_key = key.replace("vision_encoder.visual.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        shard_checkpoint(checkpoint=checkpoint, model=model, serialize_path=weights_path)

        # Post-process: add inv_freq buffers and clean up master_weight keys
        from safetensors.torch import load_file, save_file
        inv_freq_buffers = {}
        for name, buf in neuron_ve.visual.named_buffers():
            if 'inv_freq' in name:
                inv_freq_buffers[f"vision_encoder.visual.{name}"] = buf.to(torch.float32).clone()

        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                continue
            data = dict(load_file(shard_file))
            cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}
            cleaned.update(inv_freq_buffers)
            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {len(data)} -> {len(cleaned)} tensors")

        config = {
            "tp_degree": tp_degree,
            "world_size": world_size,
            "image_size": image_size,
            "patch_size": patch_size,
            "num_patches": num_patches,
            "channels_per_patch": channels_per_patch,
            "embed_dim": neuron_ve.embed_dim,
            "num_heads": neuron_ve.num_heads,
            "dtype": "float32",
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nVision Encoder compiled: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_vision_encoder(args)
