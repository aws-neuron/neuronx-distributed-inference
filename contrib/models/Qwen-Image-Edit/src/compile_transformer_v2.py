"""
Transformer compilation using ModelBuilder (V2 API).

Key approach:
1. RoPE frequencies computed OUTSIDE the model and passed as INPUT tensors
2. Model does NOT compute RoPE internally - avoids XLA constant-folding
3. Uses ModelBuilder for compilation

This avoids the RoPE buffer constant-folding issue that broke previous V2 attempts.
Achieves ~2x speedup over V1 (parallel_model_trace) API.
"""

import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags optimized for transformer
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Tuple, List

from diffusers import QwenImageEditPlusPipeline

# ModelBuilder imports
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import save_file

from neuron_parallel_utils import (
    shard_qwen_attention,
    shard_feedforward,
    shard_modulation,
    get_sharded_data,
)
from neuron_commons import neuron_scaled_dot_product_attention

# Override SDPA for Neuron compatibility
print("Overriding SDPA for Neuron compatibility")
torch.nn.functional.scaled_dot_product_attention = neuron_scaled_dot_product_attention

# NOTE: We'll patch apply_rotary_emb_qwen AFTER defining apply_rotary_emb_precomputed
# This is done below after the function definition

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def apply_rotary_emb_precomputed(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings using PRE-COMPUTED cos/sin tensors.

    Handles BOTH use_real=True and use_real=False cases:
    - use_real=False (QwenImage default): Complex multiplication simulation
    - use_real=True: Standard cos/sin rotation

    Args:
        x: [B, S, H, D] - input tensor, D = head_dim = 128
        freqs_cis: Tuple of (cos, sin), each [S, D/2] - NOT interleaved (D/2 = 64)

    Returns:
        Rotated tensor [B, S, H, D]
    """
    cos, sin = freqs_cis  # Each [S, 64]

    # Move to same device as x
    cos = cos.to(x.device)
    sin = sin.to(x.device)

    if not use_real:
        # QwenImage uses use_real=False (complex multiplication)
        # Original code:
        #   x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        #   freqs_cis = freqs_cis.unsqueeze(1)  # [S, 1, D/2] for broadcasting with [B, S, H, D/2]
        #   x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        #
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # where x = a + bi, freqs = c + di = cos + i*sin

        # Reshape x to [B, S, H, D/2, 2] then split into real/imag
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, 64, 2]
        x_real = x_reshaped[..., 0]  # [B, S, H, 64]
        x_imag = x_reshaped[..., 1]  # [B, S, H, 64]

        # Expand cos/sin for broadcasting: [S, 64] -> [1, S, 1, 64]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]

        # Complex multiplication: (x_real + i*x_imag) * (cos + i*sin)
        # real part: x_real * cos - x_imag * sin
        # imag part: x_real * sin + x_imag * cos
        out_real = x_real * cos - x_imag * sin  # [B, S, H, 64]
        out_imag = x_real * sin + x_imag * cos  # [B, S, H, 64]

        # Stack and flatten back to [B, S, H, 128]
        out = torch.stack([out_real, out_imag], dim=-1)  # [B, S, H, 64, 2]
        out = out.flatten(-2)  # [B, S, H, 128]

        return out.to(x.dtype)
    else:
        # use_real=True path (standard rotation)
        # Expand for broadcasting: [S, D/2] -> [1, S, 1, D/2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]

        # Interleave: [c0, c1, ...] -> [c0, c0, c1, c1, ...]
        cos = cos.repeat_interleave(2, dim=-1)  # [1, S, 1, 128]
        sin = sin.repeat_interleave(2, dim=-1)  # [1, S, 1, 128]

        # Create rotated version
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        else:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)

        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


# Patch apply_rotary_emb_qwen to use our pre-computed version
import diffusers.models.transformers.transformer_qwenimage as qwen_module
qwen_module.apply_rotary_emb_qwen = apply_rotary_emb_precomputed
print("Patched apply_rotary_emb_qwen for pre-computed RoPE")


class NeuronQwenTransformerV2(nn.Module):
    """
    Neuron-optimized QwenImage Transformer for V2 API.

    Key difference: RoPE frequencies are passed as INPUT, not computed internally.
    This avoids XLA constant-folding issues.
    """

    def __init__(self, original_transformer, tp_degree):
        super().__init__()

        self.config = original_transformer.config
        self.in_channels = original_transformer.config.in_channels
        self.out_channels = original_transformer.config.out_channels
        self.patch_size = original_transformer.config.patch_size

        # Input projections (keep original)
        self.img_in = original_transformer.img_in  # Linear for image patches
        self.txt_in = original_transformer.txt_in  # Linear for text

        # Time/text embedding (keep original)
        self.time_text_embed = original_transformer.time_text_embed

        # Text norm (keep original)
        self.txt_norm = original_transformer.txt_norm

        # NOTE: We do NOT copy pos_embed (RoPE) - it will be passed as input!

        # Transformer blocks (need to shard)
        self.transformer_blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.transformer_blocks):
            # Shard attention
            block.attn = shard_qwen_attention(tp_degree, block.attn)
            # Shard MLPs
            block.img_mlp = shard_feedforward(block.img_mlp)
            block.txt_mlp = shard_feedforward(block.txt_mlp)
            # Shard modulation
            block.img_mod = shard_modulation(block.img_mod)
            block.txt_mod = shard_modulation(block.txt_mod)
            self.transformer_blocks.append(block)

            if (i + 1) % 10 == 0:
                print(f"  Sharded block {i+1}/{len(original_transformer.transformer_blocks)}")

        # Final layers (keep original)
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out

        # Store head_dim for RoPE
        self.head_dim = 128  # QwenImage uses 128
        self.num_heads = original_transformer.transformer_blocks[0].attn.heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_rotary_emb: torch.Tensor,  # [num_patches, 64, 2] for (cos, sin), NOT interleaved
        txt_rotary_emb: torch.Tensor,  # [text_seq, 64, 2] for (cos, sin), NOT interleaved
    ) -> torch.Tensor:
        """
        Forward pass with RoPE as INPUT.

        Args:
            hidden_states: [B, num_patches, in_channels]
            encoder_hidden_states: [B, text_seq, text_dim]
            timestep: [B]
            img_rotary_emb: [num_patches, 64, 2] - pre-computed RoPE (NOT interleaved)
            txt_rotary_emb: [text_seq, 64, 2] - pre-computed RoPE (NOT interleaved)
        """
        # Split RoPE into cos/sin
        # Shape: [S, 64] - NOT interleaved, apply_rotary_emb_precomputed will do repeat_interleave
        img_freqs_cos = img_rotary_emb[..., 0]  # [num_patches, 64]
        img_freqs_sin = img_rotary_emb[..., 1]
        txt_freqs_cos = txt_rotary_emb[..., 0]  # [text_seq, 64]
        txt_freqs_sin = txt_rotary_emb[..., 1]

        # Image input projection
        hidden_states = self.img_in(hidden_states)  # [B, num_patches, inner_dim]

        # Text processing: norm first, then projection
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)  # [B, text_seq, inner_dim]

        # Time embedding (takes timestep and hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)

        # Create rotary_emb tuple in format expected by diffusers
        # Using (cos, sin) tuple format for Neuron compatibility
        image_rotary_emb = ((img_freqs_cos, img_freqs_sin), (txt_freqs_cos, txt_freqs_sin))

        # Process through transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Final norm and projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output


class TracingWrapperV2(nn.Module):
    """Wrapper for ModelBuilder tracing."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_rotary_emb, txt_rotary_emb):
        return self.transformer(
            hidden_states, encoder_hidden_states, timestep,
            img_rotary_emb, txt_rotary_emb
        )


def get_rope_from_original_model(
    pipe,
    frame: int,
    height: int,
    width: int,
    text_seq_len: int,
    dtype=torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get RoPE directly from the original QwenEmbedRope model.

    This ensures the RoPE values are EXACTLY the same as what V1 uses.

    Returns:
        img_rotary_emb: [num_patches, 64, 2] - stacked (cos, sin) from complex freqs
        txt_rotary_emb: [text_seq_len, 64, 2] - stacked (cos, sin) from complex freqs
    """
    print(f"  Getting RoPE from original model...")
    print(f"  video_fhw: ({frame}, {height}, {width}), text_seq_len: {text_seq_len}")

    # Call original pos_embed to get complex freqs
    video_fhw = (frame, height, width)
    vid_freqs, txt_freqs = pipe.transformer.pos_embed(
        video_fhw, txt_seq_lens=[text_seq_len], device=torch.device('cpu')
    )

    print(f"  vid_freqs from model: {vid_freqs.shape}, dtype: {vid_freqs.dtype}")
    print(f"  txt_freqs from model: {txt_freqs.shape}, dtype: {txt_freqs.dtype}")

    # Convert complex to (cos, sin)
    # Complex freqs are e^(i*angle) = cos(angle) + i*sin(angle)
    img_cos = vid_freqs.real.float()  # [num_patches, 64]
    img_sin = vid_freqs.imag.float()  # [num_patches, 64]
    txt_cos = txt_freqs.real.float()  # [text_seq_len, 64]
    txt_sin = txt_freqs.imag.float()  # [text_seq_len, 64]

    # Stack to [S, 64, 2]
    img_rotary_emb = torch.stack([img_cos, img_sin], dim=-1).to(dtype)
    txt_rotary_emb = torch.stack([txt_cos, txt_sin], dim=-1).to(dtype)

    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")
    print(f"  img_cos stats: min={img_cos.min():.4f}, max={img_cos.max():.4f}")
    print(f"  img_sin stats: min={img_sin.min():.4f}, max={img_sin.max():.4f}")

    return img_rotary_emb, txt_rotary_emb


def compile_transformer_v2(args):
    """Compile transformer using ModelBuilder V2 API with RoPE as input."""

    tp_degree = args.tp_degree

    # Calculate dimensions
    latent_h = args.height // 8
    latent_w = args.width // 8
    patch_size = 2
    patch_h = latent_h // patch_size
    patch_w = latent_w // patch_size
    temporal_frames = args.patch_multiplier
    num_patches = temporal_frames * patch_h * patch_w

    text_seq_len = args.max_sequence_length
    text_hidden_size = 3584
    in_channels = 64
    head_dim = 128

    print("=" * 60)
    print("Transformer V2 Compilation")
    print("=" * 60)
    print(f"Image: {args.height}x{args.width}")
    print(f"Patches: {num_patches} ({temporal_frames}x{patch_h}x{patch_w})")
    print(f"Text seq: {text_seq_len}")
    print(f"TP degree: {tp_degree}")

    # Sample inputs
    sample_hidden_states = torch.randn(1, num_patches, in_channels, dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.randn(1, text_seq_len, text_hidden_size, dtype=torch.bfloat16)
    sample_timestep = torch.randn(1, dtype=torch.float32)

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        print("\nLoading model...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            cache_dir=CACHE_DIR
        )

        # Get RoPE directly from original model (ensures exact match with V1)
        print("\nGetting RoPE from original model...")
        img_rotary_emb, txt_rotary_emb = get_rope_from_original_model(
            pipe=pipe,
            frame=temporal_frames,
            height=patch_h,
            width=patch_w,
            text_seq_len=text_seq_len,
        )

        # Verify shapes are correct (64 = head_dim // 2)
        rope_dim = head_dim // 2  # 64
        assert img_rotary_emb.shape[-2] == rope_dim, f"img_rotary_emb shape wrong: {img_rotary_emb.shape}, expected dim -2 = {rope_dim}"
        assert txt_rotary_emb.shape[-2] == rope_dim, f"txt_rotary_emb shape wrong: {txt_rotary_emb.shape}, expected dim -2 = {rope_dim}"

        # Save unsharded state dict before modifications
        unsharded_state = pipe.transformer.state_dict()

        print("Creating Neuron transformer (sharding layers)...")
        neuron_transformer = NeuronQwenTransformerV2(pipe.transformer, tp_degree)
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Wrap for tracing
        model = TracingWrapperV2(neuron_transformer)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "encoder_hidden_states": sample_encoder_hidden_states,
                "timestep": sample_timestep,
                "img_rotary_emb": img_rotary_emb,
                "txt_rotary_emb": txt_rotary_emb,
            },
            tag="inference",
        )

        print("Compiling model...")
        compile_args = "--model-type=transformer -O1 --auto-cast=none --lnc=2 --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4' --internal-hlo2tensorizer-options='--enable-native-kernel=1 --remat'"
        traced_model = builder.compile(
            compiler_args=compile_args,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/transformer_v2"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint for sharding
        checkpoint = {}
        for key, value in model.state_dict().items():
            # Use unsharded weights where available
            orig_key = key.replace("transformer.", "", 1)
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

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_patches": num_patches,
            "text_seq_len": text_seq_len,
            "patch_multiplier": args.patch_multiplier,
            "tp_degree": tp_degree,
            "head_dim": head_dim,
            "frame": temporal_frames,
            "patch_h": patch_h,
            "patch_w": patch_w,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save pre-computed RoPE
        torch.save({
            "img_rotary_emb": img_rotary_emb,
            "txt_rotary_emb": txt_rotary_emb,
        }, os.path.join(output_path, "rope_cache.pt"))

        print("\nCompilation complete!")
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--patch_multiplier", type=int, default=3)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    args = parser.parse_args()

    compile_transformer_v2(args)
