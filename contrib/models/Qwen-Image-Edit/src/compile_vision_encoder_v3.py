"""
Vision Encoder Compilation using ModelBuilder API (V3) for TP=4 Acceleration.

This script compiles the Qwen2.5-VL Vision Encoder using ModelBuilder API with
tp_degree=4 and world_size=8 for faster inference while maintaining float32 precision.

Key features:
- Uses ModelBuilder API (NxDModel) for compilation
- Configuration: tp_degree=4, world_size=8 (matching V3 CP transformer)
- Float32 precision for accuracy (required for vision encoder)
- Vision encoder hidden_size=1280, QKV=3840, MLP intermediate=3420
- TP=4 works: 3840/4=960, 3420/4=855 (both divisible)

Usage:
    python compile_vision_encoder_v3.py --image_size 448
"""

import os
import json
import gc

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse

from diffusers import QwenImageEditPlusPipeline

# ModelBuilder imports
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def load_pipeline(dtype=torch.float32):
    """Load pipeline with appropriate kwargs."""
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class f32Wrapper(nn.Module):
    """Wrapper to run normalization layers in float32 for numerical stability."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, *args, **kwargs):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y, *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.LayerNorm,)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif "RMSNorm" in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def get_sharded_data(data, dim):
    """Get this rank's portion of sharded data."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_degree = parallel_state.get_tensor_model_parallel_size()

    total_size = data.shape[dim]
    shard_size = total_size // tp_degree

    start = tp_rank * shard_size
    end = start + shard_size

    if dim == 0:
        return data[start:end].clone()
    elif dim == 1:
        return data[:, start:end].clone()
    else:
        raise ValueError(f"Unsupported shard dimension: {dim}")


def shard_vision_attention_fp32(tp_degree: int, attn):
    """
    Shard Qwen2.5-VL Vision Encoder attention module with float32 precision.

    Vision attention uses fused QKV projection:
    - qkv: (in_features, 3 * in_features) -> splits into Q, K, V
    - proj: output projection

    Qwen2.5-VL vision encoder:
    - hidden_size (embed_dim) = 1280
    - num_heads = 16, head_dim = 80
    - QKV dim = 3840 = 1280 * 3
    - 3840 / 4 = 960 (divisible, TP=4 works)

    IMPORTANT: Must also update num_heads after sharding!
    - With TP=4: num_heads becomes 16/4 = 4 per rank
    """
    orig_qkv = attn.qkv
    orig_proj = attn.proj

    # Update num_heads for this rank (critical for correct attention computation)
    original_num_heads = attn.num_heads
    attn.num_heads = original_num_heads // tp_degree

    # Shard fused QKV projection
    attn.qkv = ColumnParallelLinear(
        orig_qkv.in_features,
        orig_qkv.out_features,
        bias=(orig_qkv.bias is not None),
        gather_output=False,
        dtype=torch.float32,
    )
    attn.qkv.weight.data = get_sharded_data(orig_qkv.weight.data, 0)
    if orig_qkv.bias is not None:
        attn.qkv.bias.data = get_sharded_data(orig_qkv.bias.data, 0)
    del orig_qkv

    # Shard output projection
    attn.proj = RowParallelLinear(
        orig_proj.in_features,
        orig_proj.out_features,
        bias=(orig_proj.bias is not None),
        input_is_parallel=True,
        dtype=torch.float32,
    )
    attn.proj.weight.data = get_sharded_data(orig_proj.weight.data, 1)
    if orig_proj.bias is not None:
        attn.proj.bias.data = orig_proj.bias.data.detach()
    del orig_proj

    return attn


def shard_vision_mlp_fp32(mlp):
    """
    Shard Qwen2.5-VL Vision Encoder MLP module with float32 precision.

    Vision MLP uses SwiGLU-style architecture:
    - gate_proj: (hidden_size, intermediate_size)
    - up_proj: (hidden_size, intermediate_size)
    - down_proj: (intermediate_size, hidden_size)

    Qwen2.5-VL vision encoder:
    - hidden_size = 1280
    - intermediate_size = 3420
    - 3420 / 4 = 855 (divisible)
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    # Shard gate projection
    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features,
        orig_gate.out_features,
        bias=(orig_gate.bias is not None),
        gather_output=False,
        dtype=torch.float32,
    )
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    # Shard up projection
    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features,
        orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False,
        dtype=torch.float32,
    )
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Shard down projection
    mlp.down_proj = RowParallelLinear(
        orig_down.in_features,
        orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True,
        dtype=torch.float32,
    )
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp


class NeuronVisionEncoderV3(nn.Module):
    """
    Neuron-optimized Qwen2.5-VL Vision Encoder with TP=4, float32 precision.

    Uses ModelBuilder API with tp_degree=4, world_size=8.

    Key features:
    - TP=4 for parallel computation (3420 QKV dim / 4 = 855, divisible)
    - Float32 precision for accuracy (required for vision encoder)
    - World_size=8 for compatibility with V3 CP transformer
    """

    def __init__(self, original_visual, tp_degree):
        super().__init__()

        self.tp_degree = tp_degree

        # Keep the full visual encoder (we'll modify its layers in-place)
        self.visual = original_visual

        # Get model structure info from config
        self.embed_dim = original_visual.config.hidden_size  # 1280
        self.num_heads = original_visual.config.num_heads  # 16

        print(f"  Vision encoder config:")
        print(f"    embed_dim (hidden_size): {self.embed_dim}")
        print(f"    num_heads: {self.num_heads}")
        print(f"    QKV dim: {self.embed_dim * 3} = {self.embed_dim} * 3")
        print(f"    QKV per rank: {self.embed_dim * 3 // tp_degree}")

        # Shard the transformer blocks
        for i, block in enumerate(self.visual.blocks):
            if hasattr(block, "attn"):
                block.attn = shard_vision_attention_fp32(tp_degree, block.attn)
            if hasattr(block, "mlp"):
                block.mlp = shard_vision_mlp_fp32(block.mlp)
            if i == 0:
                print(f"  Sharded block 0 attention and MLP")

        print(f"  Sharded all {len(self.visual.blocks)} blocks")

        # Upcast norms to float32 (already float32, but ensure wrapper)
        upcast_norms_to_f32(self.visual)

    def forward(self, pixel_values, grid_thw):
        """
        Forward pass for vision encoder.

        Args:
            pixel_values: (num_patches, channels_per_patch) - flattened image patches
            grid_thw: (num_images, 3) - temporal, height, width grid dimensions

        Returns:
            image_embeds: (num_output_tokens, hidden_size) - vision embeddings after merger
        """
        return self.visual(pixel_values, grid_thw)


class TracingWrapper(nn.Module):
    """Wrapper for ModelBuilder tracing."""

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values, grid_thw):
        return self.vision_encoder(pixel_values, grid_thw)


def compile_vision_encoder_v3(args):
    """
    Compile Vision Encoder using ModelBuilder API.

    Configuration:
    - tp_degree=4: Works with vision encoder dimensions (3420 / 4 = 855)
    - world_size=8: Matches V3 CP transformer
    - dtype=float32: Required for accuracy
    """
    tp_degree = 4  # Fixed: vision encoder dimensions require TP=4
    world_size = 8  # Fixed: match V3 CP transformer

    image_size = args.image_size
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2

    # Validate image_size
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size}). "
            f"Valid sizes: 224, 336, 448, 560, etc."
        )

    num_patches_per_side = image_size // patch_size
    if num_patches_per_side % spatial_merge_size != 0:
        raise ValueError(
            f"image_size / patch_size ({num_patches_per_side}) must be divisible by "
            f"spatial_merge_size ({spatial_merge_size}). "
            f"Valid image sizes: 224, 336, 448, 560, etc."
        )

    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    num_patches = num_patches_h * num_patches_w

    # pixel_values shape: (num_patches, channels_per_patch)
    channels_per_patch = 3 * temporal_patch_size * patch_size * patch_size  # 1176

    print("=" * 60)
    print("Compiling Vision Encoder V3 (ModelBuilder API, TP=4, float32)")
    print("=" * 60)
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches: {num_patches}")
    print(f"  Channels per patch: {channels_per_patch}")
    print(f"  TP degree: {tp_degree}")
    print(f"  World size: {world_size}")
    print(f"  Dtype: float32 (required for accuracy)")
    print("")

    # Sample inputs
    sample_pixel_values = torch.randn(
        num_patches, channels_per_patch, dtype=torch.float32
    )
    sample_grid_thw = torch.tensor(
        [[1, num_patches_h, num_patches_w]], dtype=torch.int64
    )

    print(f"Sample input shapes:")
    print(f"  pixel_values: {sample_pixel_values.shape}")
    print(f"  grid_thw: {sample_grid_thw.shape}")
    print("")

    # Use NxDParallelState context for compilation
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # On trn2.3xlarge (or instances with <96GB RAM), loading the full pipeline
        # in fp32 (~95 GB) will OOM. Load in bf16 to save memory, then extract
        # the vision encoder and explicitly convert its weights to fp32.
        # On trn2.48xlarge, fp32 loading works fine.
        load_dtype = torch.bfloat16 if args.load_bf16 else torch.float32
        print(f"Loading model in {load_dtype}...")
        pipe = load_pipeline(load_dtype)

        # Extract vision encoder
        original_visual = pipe.text_encoder.model.visual

        # Save unsharded state dict before modifications.
        # CRITICAL: If pipeline was loaded in bf16, the state dict will be bf16.
        # Vision encoder requires fp32 for accuracy, so we must explicitly cast.
        print("Saving unsharded state dict...")
        unsharded_state = {
            k: v.to(torch.float32) for k, v in original_visual.state_dict().items()
        }

        # Convert vision encoder to fp32 before sharding
        if load_dtype != torch.float32:
            original_visual = original_visual.to(torch.float32)

        # Create Neuron vision encoder with sharding
        print(
            f"\nCreating Neuron vision encoder (sharding layers with TP={tp_degree})..."
        )
        neuron_vision_encoder = NeuronVisionEncoderV3(original_visual, tp_degree)
        neuron_vision_encoder = neuron_vision_encoder.to(torch.float32)
        neuron_vision_encoder.eval()

        # Clear pipeline to save memory (important on trn2.3xlarge)
        del pipe
        gc.collect()

        # Wrap for tracing
        model = TracingWrapper(neuron_vision_encoder)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "pixel_values": sample_pixel_values,
                "grid_thw": sample_grid_thw,
            },
            tag="inference",
        )

        print("Compiling model...")
        # Use --auto-cast=none to preserve float32 precision
        # NOTE: Using -O1 instead of -O2 because -O2 can cause numerical issues in some cases
        compile_args = "--model-type=transformer -O1 --auto-cast=none"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/vision_encoder_v3"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint for sharding
        print("Preparing checkpoint...")
        checkpoint = {}
        for key, value in model.state_dict().items():
            # Use unsharded weights where available
            # Key format: vision_encoder.visual.blocks.X... -> blocks.X...
            orig_key = key.replace("vision_encoder.visual.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        # Shard checkpoint
        print("Sharding weights...")
        shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            serialize_path=weights_path,
        )

        # Post-process checkpoints: remove master_weight and add inv_freq
        print("\nPost-processing checkpoints...")
        from safetensors.torch import load_file, save_file

        # Collect inv_freq buffers from original model (they are not in state_dict)
        inv_freq_buffers = {}
        for name, buf in neuron_vision_encoder.visual.named_buffers():
            if "inv_freq" in name:
                full_key = f"vision_encoder.visual.{name}"
                inv_freq_buffers[full_key] = buf.to(torch.float32).clone()
        print(f"  Collected {len(inv_freq_buffers)} inv_freq buffers")

        for rank in range(tp_degree):
            shard_file = os.path.join(
                weights_path, f"tp{rank}_sharded_checkpoint.safetensors"
            )
            if not os.path.exists(shard_file):
                print(f"  WARNING: {shard_file} not found!")
                continue

            # Load checkpoint
            data = dict(load_file(shard_file))
            original_count = len(data)
            original_size = sum(v.numel() * v.element_size() for v in data.values())

            # Remove master_weight tensors (they duplicate the sharded weights)
            cleaned = {k: v for k, v in data.items() if "master_weight" not in k}

            # Add inv_freq buffers
            cleaned.update(inv_freq_buffers)

            cleaned_size = sum(v.numel() * v.element_size() for v in cleaned.values())

            # Save optimized checkpoint
            save_file(cleaned, shard_file)
            print(
                f"  tp{rank}: {original_count} -> {len(cleaned)} tensors, "
                f"{original_size / 1e9:.2f}GB -> {cleaned_size / 1e9:.2f}GB"
            )

        # Save config
        config = {
            "tp_degree": tp_degree,
            "world_size": world_size,
            "image_size": image_size,
            "patch_size": patch_size,
            "num_patches": num_patches,
            "channels_per_patch": channels_per_patch,
            "embed_dim": neuron_vision_encoder.embed_dim,
            "num_heads": neuron_vision_encoder.num_heads,
            "dtype": "float32",
        }
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nVision Encoder V3 compiled successfully!")
        print(f"  Output: {output_path}")
        print(f"  Config: {config_path}")
        print(f"  Weights: {weights_path}")

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Vision Encoder V3 using ModelBuilder API"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Vision encoder input image size (default: 448)",
    )
    parser.add_argument(
        "--compiled_models_dir",
        type=str,
        default="/opt/dlami/nvme/compiled_models",
        help="Output directory for compiled models",
    )
    parser.add_argument(
        "--compiler_workdir",
        type=str,
        default="/opt/dlami/nvme/compiler_workdir",
        help="Compiler working directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model (local dir or HuggingFace ID)",
    )
    parser.add_argument(
        "--load_bf16",
        action="store_true",
        default=False,
        help="Load pipeline in bf16 to save memory (for trn2.3xlarge). "
        "Weights are automatically cast to fp32 for compilation.",
    )

    args = parser.parse_args()

    # Override MODEL_ID if model_path is provided
    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_vision_encoder_v3(args)
