"""
Transformer compilation using ModelBuilder (V2 API) with NKI Flash Attention.

Key approach:
1. Uses ModelBuilder API for compilation (like V2)
2. Uses NKI Flash Attention kernel for hardware-optimized attention (like V1 Flash)
3. RoPE frequencies computed OUTSIDE the model and passed as INPUT tensors
4. Disables XLA functionalization to allow NKI in-place operations

This combines the best of both:
- ModelBuilder's XLA optimization
- NKI's hardware-optimized attention kernel
"""

import os
import json
import math

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
# CRITICAL: Disable XLA functionalization to allow NKI kernel in-place operations
# Without this, NKI kernels will fail with "Cannot update immutable parameter"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags optimized for transformer
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

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

# Import NKI Flash Attention - use EXACTLY the same imports as Flux
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

# Create NKI callable - EXACTLY like Flux does
_flash_fwd_call = nki_jit()(attention_isa_kernel)

NKI_AVAILABLE = True
print("NKI Flash Attention kernel loaded successfully")

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, S, D] - query tensor
        key: [B, H, S, D] - key tensor
        value: [B, H, S, D] - value tensor

    Returns:
        attention output [B, H, S, D]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    # Reshape for NKI kernel: [B*H, D, S] for Q/K, [B*H, S, D] for V
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    # Pre-allocate output
    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    scale = 1 / math.sqrt(d_head)

    # Use sharded kernel for VC_SIZE=2
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](
            q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    # Reshape back to [B, H, S, D]
    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))

    return attn_output


def apply_rotary_emb_precomputed(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings using PRE-COMPUTED cos/sin tensors.

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
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, 64, 2]
        x_real = x_reshaped[..., 0]  # [B, S, H, 64]
        x_imag = x_reshaped[..., 1]  # [B, S, H, 64]

        # Expand cos/sin for broadcasting: [S, 64] -> [1, S, 1, 64]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, 64]

        # Complex multiplication: (x_real + i*x_imag) * (cos + i*sin)
        out_real = x_real * cos - x_imag * sin  # [B, S, H, 64]
        out_imag = x_real * sin + x_imag * cos  # [B, S, H, 64]

        # Stack and flatten back to [B, S, H, 128]
        out = torch.stack([out_real, out_imag], dim=-1)  # [B, S, H, 64, 2]
        out = out.flatten(-2)  # [B, S, H, 128]

        return out.to(x.dtype)
    else:
        # use_real=True path (standard rotation)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

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


class NKIQwenAttention(nn.Module):
    """
    Custom attention module for QwenImage that uses NKI Flash Attention directly.

    This completely replaces diffusers' Attention class, similar to how Flux
    uses NeuronFluxAttention.
    """

    def __init__(self, orig_attn):
        """Initialize from an existing sharded attention module."""
        super().__init__()

        # Copy all the layers from the original attention
        self.heads = orig_attn.heads
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # Text projections
        self.add_q_proj = orig_attn.add_q_proj if hasattr(orig_attn, 'add_q_proj') else None
        self.add_k_proj = orig_attn.add_k_proj if hasattr(orig_attn, 'add_k_proj') else None
        self.add_v_proj = orig_attn.add_v_proj if hasattr(orig_attn, 'add_v_proj') else None
        self.to_add_out = orig_attn.to_add_out if hasattr(orig_attn, 'to_add_out') else None

        # Norms
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None
        self.norm_added_q = orig_attn.norm_added_q if hasattr(orig_attn, 'norm_added_q') else None
        self.norm_added_k = orig_attn.norm_added_k if hasattr(orig_attn, 'norm_added_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,  # Image stream [B, S_img, C]
        encoder_hidden_states: torch.Tensor = None,  # Text stream [B, S_txt, C]
        encoder_hidden_states_mask: torch.Tensor = None,
        image_rotary_emb: Tuple = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with NKI Flash Attention."""
        if encoder_hidden_states is None:
            raise ValueError("NKIQwenAttention requires encoder_hidden_states")

        batch_size = hidden_states.shape[0]
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = self.to_q(hidden_states)
        img_key = self.to_k(hidden_states)
        img_value = self.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = self.add_q_proj(encoder_hidden_states)
        txt_key = self.add_k_proj(encoder_hidden_states)
        txt_value = self.add_v_proj(encoder_hidden_states)

        # Get head dimension
        inner_dim = img_query.shape[-1]
        head_dim = inner_dim // self.heads

        # Reshape to [B, S, H, D] then transpose to [B, H, S, D] - exactly like Flux
        img_query = img_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        img_key = img_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        img_value = img_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        txt_query = txt_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        txt_key = txt_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        txt_value = txt_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)

        # Apply RoPE - note: input is now [B, H, S, D]
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # Transpose to [B, S, H, D] for RoPE, then back to [B, H, S, D]
            img_query = apply_rotary_emb_precomputed(img_query.transpose(1, 2), img_freqs, use_real=False).transpose(1, 2)
            img_key = apply_rotary_emb_precomputed(img_key.transpose(1, 2), img_freqs, use_real=False).transpose(1, 2)
            txt_query = apply_rotary_emb_precomputed(txt_query.transpose(1, 2), txt_freqs, use_real=False).transpose(1, 2)
            txt_key = apply_rotary_emb_precomputed(txt_key.transpose(1, 2), txt_freqs, use_real=False).transpose(1, 2)

        # Concatenate for joint attention along sequence dim: [B, H, S_txt + S_img, D]
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        # Use NKI Flash Attention - input is [B, H, S, D] exactly like Flux
        joint_hidden_states = nki_flash_attention(joint_query, joint_key, joint_value)

        # Transpose back and reshape: [B, H, S, D] -> [B, S, H*D]
        joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = self.to_out[0](img_attn_output)
        if len(self.to_out) > 1:
            img_attn_output = self.to_out[1](img_attn_output)  # dropout

        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


def replace_attention_with_nki(transformer):
    """Replace all attention modules with NKI versions."""
    for i, block in enumerate(transformer.transformer_blocks):
        block.attn = NKIQwenAttention(block.attn)
    print(f"Replaced attention modules with NKI versions on {len(transformer.transformer_blocks)} blocks")


class NeuronQwenTransformerV2Flash(nn.Module):
    """
    Neuron-optimized QwenImage Transformer for V2 Flash.

    Combines:
    - ModelBuilder API for compilation (V2)
    - NKI Flash Attention for hardware-optimized attention (V1 Flash)
    - Pre-computed RoPE as input tensors
    """

    def __init__(self, original_transformer, tp_degree):
        super().__init__()

        self.config = original_transformer.config
        self.in_channels = original_transformer.config.in_channels
        self.out_channels = original_transformer.config.out_channels
        self.patch_size = original_transformer.config.patch_size

        # Input projections (keep original)
        self.img_in = original_transformer.img_in
        self.txt_in = original_transformer.txt_in

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
        self.head_dim = 128
        self.num_heads = original_transformer.transformer_blocks[0].attn.heads

        # Replace attention modules with NKI versions AFTER sharding
        replace_attention_with_nki(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_rotary_emb: torch.Tensor,  # [num_patches, 64, 2]
        txt_rotary_emb: torch.Tensor,  # [text_seq, 64, 2]
    ) -> torch.Tensor:
        """Forward pass with RoPE as INPUT and NKI Flash Attention."""
        # Split RoPE into cos/sin
        img_freqs_cos = img_rotary_emb[..., 0]  # [num_patches, 64]
        img_freqs_sin = img_rotary_emb[..., 1]
        txt_freqs_cos = txt_rotary_emb[..., 0]  # [text_seq, 64]
        txt_freqs_sin = txt_rotary_emb[..., 1]

        # Image input projection
        hidden_states = self.img_in(hidden_states)

        # Text processing
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Time embedding
        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)

        # Create rotary_emb tuple
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


class TracingWrapperV2Flash(nn.Module):
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
    """Get RoPE directly from the original QwenEmbedRope model."""
    print(f"  Getting RoPE from original model...")
    print(f"  video_fhw: ({frame}, {height}, {width}), text_seq_len: {text_seq_len}")

    video_fhw = (frame, height, width)
    vid_freqs, txt_freqs = pipe.transformer.pos_embed(
        video_fhw, txt_seq_lens=[text_seq_len], device=torch.device('cpu')
    )

    print(f"  vid_freqs from model: {vid_freqs.shape}, dtype: {vid_freqs.dtype}")
    print(f"  txt_freqs from model: {txt_freqs.shape}, dtype: {txt_freqs.dtype}")

    # Convert complex to (cos, sin)
    img_cos = vid_freqs.real.float()
    img_sin = vid_freqs.imag.float()
    txt_cos = txt_freqs.real.float()
    txt_sin = txt_freqs.imag.float()

    # Stack to [S, 64, 2]
    img_rotary_emb = torch.stack([img_cos, img_sin], dim=-1).to(dtype)
    txt_rotary_emb = torch.stack([txt_cos, txt_sin], dim=-1).to(dtype)

    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    return img_rotary_emb, txt_rotary_emb


def compile_transformer_v2_flash(args):
    """Compile transformer using ModelBuilder V2 API with NKI Flash Attention."""

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
    print("Transformer V2 Flash Compilation")
    print("=" * 60)
    print(f"Image: {args.height}x{args.width}")
    print(f"Patches: {num_patches} ({temporal_frames}x{patch_h}x{patch_w})")
    print(f"Text seq: {text_seq_len}")
    print(f"TP degree: {tp_degree}")
    print(f"NKI Flash Attention: Enabled")
    print(f"XLA_DISABLE_FUNCTIONALIZATION: {os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', 'not set')}")

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

        # Get RoPE from original model
        print("\nGetting RoPE from original model...")
        img_rotary_emb, txt_rotary_emb = get_rope_from_original_model(
            pipe=pipe,
            frame=temporal_frames,
            height=patch_h,
            width=patch_w,
            text_seq_len=text_seq_len,
        )

        # Save unsharded state dict before modifications
        unsharded_state = pipe.transformer.state_dict()

        print("Creating Neuron transformer (sharding layers + NKI attention)...")
        neuron_transformer = NeuronQwenTransformerV2Flash(pipe.transformer, tp_degree)
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Wrap for tracing
        model = TracingWrapperV2Flash(neuron_transformer)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model with NKI Flash Attention...")
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
        output_path = f"{args.compiled_models_dir}/transformer_v2_flash"
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
            "nki_flash_attention": True,
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
        print("\nTo run inference:")
        print(f"  python run_qwen_image_edit.py --images img1.png img2.png --prompt '...' --use_v2_flash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--patch_multiplier", type=int, default=3)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    args = parser.parse_args()

    compile_transformer_v2_flash(args)
