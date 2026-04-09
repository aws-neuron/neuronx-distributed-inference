"""
Transformer compilation with Context Parallel (V3 CP) using ModelBuilder API.

Key approach:
1. Uses ModelBuilder API (like V2) for compilation
2. Configures world_size=8, tp_degree=4 (implicit CP=2)
3. K/V are all-gathered across DP group before attention
4. Uses NKI Flash Attention for optimal performance

This is inspired by Flux's context parallel implementation which achieves
near-H100 performance on TRN2.

Context Parallel works by:
- Model parameters are sharded with TP=4
- DP group (2 ranks) is used for sequence parallelism
- Each DP rank processes half the sequence (queries)
- K/V are all-gathered so each rank sees full K/V
"""

import os
import json
import math

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags - same as Flux for CP mode
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' --internal-hlo2tensorizer-options='--enable-state-buffer-mode=hybrid --remat-by-default' """
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
    SPMDRank,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
)

from neuron_parallel_utils import (
    shard_qwen_attention,
    shard_feedforward,
    shard_modulation,
    get_sharded_data,
)

# Import NKI Flash Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

print("NKI Flash Attention kernel loaded successfully")

CACHE_DIR = "/opt/dlami/nvme/qwen_image_edit_hf_cache_dir"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, S, D]
        key: [B, H, S, D]
        value: [B, H, S, D]

    Returns:
        attention output [B, H, S, D]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    return attn_output.reshape((bs, n_head, q_len, d_head))


class CPNKIQwenAttention(nn.Module):
    """
    Context Parallel + NKI Flash Attention for QwenImage.

    Key features:
    1. K/V are all-gathered across CP group before attention
    2. Uses NKI Flash Attention kernel
    3. Each CP rank processes its portion of queries against full K/V
    """

    def __init__(self, orig_attn, context_parallel_enabled=False, data_parallel_group=None):
        super().__init__()

        self.context_parallel_enabled = context_parallel_enabled
        self.data_parallel_group = data_parallel_group
        self.heads = orig_attn.heads
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        self.add_q_proj = orig_attn.add_q_proj if hasattr(orig_attn, 'add_q_proj') else None
        self.add_k_proj = orig_attn.add_k_proj if hasattr(orig_attn, 'add_k_proj') else None
        self.add_v_proj = orig_attn.add_v_proj if hasattr(orig_attn, 'add_v_proj') else None
        self.to_add_out = orig_attn.to_add_out if hasattr(orig_attn, 'to_add_out') else None

        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None
        self.norm_added_q = orig_attn.norm_added_q if hasattr(orig_attn, 'norm_added_q') else None
        self.norm_added_k = orig_attn.norm_added_k if hasattr(orig_attn, 'norm_added_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        image_rotary_emb: Tuple = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with Context Parallel K/V gathering and NKI attention.
        """
        if encoder_hidden_states is None:
            raise ValueError("CPNKIQwenAttention requires encoder_hidden_states")

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

        inner_dim = img_query.shape[-1]
        head_dim = inner_dim // self.heads

        # Reshape to [B, H, S, D]
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

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_precomputed(img_query.transpose(1, 2), img_freqs, use_real=False).transpose(1, 2)
            img_key = apply_rotary_emb_precomputed(img_key.transpose(1, 2), img_freqs, use_real=False).transpose(1, 2)
            txt_query = apply_rotary_emb_precomputed(txt_query.transpose(1, 2), txt_freqs, use_real=False).transpose(1, 2)
            txt_key = apply_rotary_emb_precomputed(txt_key.transpose(1, 2), txt_freqs, use_real=False).transpose(1, 2)

        # Context Parallel: All-gather K/V across DP group
        if self.context_parallel_enabled:
            # Gather image K/V
            img_stacked_kv = torch.stack([img_key, img_value], dim=0)
            img_stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                img_stacked_kv, gather_dim=3, process_group=self.data_parallel_group
            )
            img_key, img_value = torch.unbind(img_stacked_kv, dim=0)

            # Gather text K/V
            txt_stacked_kv = torch.stack([txt_key, txt_value], dim=0)
            txt_stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                txt_stacked_kv, gather_dim=3, process_group=self.data_parallel_group
            )
            txt_key, txt_value = torch.unbind(txt_stacked_kv, dim=0)

        # Concatenate for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        # NKI Flash Attention
        joint_hidden_states = nki_flash_attention(joint_query, joint_key, joint_value)

        # Transpose and reshape
        joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split back (use original local seq_txt for splitting)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = self.to_out[0](img_attn_output)
        if len(self.to_out) > 1:
            img_attn_output = self.to_out[1](img_attn_output)

        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


def apply_rotary_emb_precomputed(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Apply rotary embeddings using pre-computed cos/sin tensors."""
    cos, sin = freqs_cis
    cos = cos.to(x.device)
    sin = sin.to(x.device)

    if not use_real:
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        out = torch.stack([out_real, out_imag], dim=-1)
        out = out.flatten(-2)

        return out.to(x.dtype)
    else:
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


# Patch apply_rotary_emb_qwen
import diffusers.models.transformers.transformer_qwenimage as qwen_module
qwen_module.apply_rotary_emb_qwen = apply_rotary_emb_precomputed
print("Patched apply_rotary_emb_qwen for pre-computed RoPE")


def split_along_dim(tensor, dim, rank, data_parallel_group):
    """Split tensor along dimension using scatter_to_process_group_spmd."""
    tensor = scatter_to_process_group_spmd(
        tensor,
        partition_dim=dim,
        rank=rank,
        process_group=data_parallel_group,
    )
    return tensor


def get_dp_rank_spmd(global_rank: torch.Tensor, tp_degree: int) -> torch.Tensor:
    """
    Compute DP rank from global rank for SPMD execution.

    With world_size=8 and tp_degree=4:
    - Ranks 0-3 are DP rank 0
    - Ranks 4-7 are DP rank 1
    """
    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor",
    ).to(torch.int32)
    return dp_rank


class NeuronQwenTransformerV3CP(nn.Module):
    """
    Neuron-optimized QwenImage Transformer with Context Parallel.

    Features:
    - TP=4 for model parameter sharding
    - CP enabled (via DP group) for sequence parallelism
    - Data is SPLIT at entry, K/V gathered in attention, output gathered at exit
    - NKI Flash Attention
    """

    def __init__(self, original_transformer, tp_degree, world_size, context_parallel_enabled=False):
        super().__init__()

        self.config = original_transformer.config
        self.in_channels = original_transformer.config.in_channels
        self.out_channels = original_transformer.config.out_channels
        self.patch_size = original_transformer.config.patch_size
        self.context_parallel_enabled = context_parallel_enabled
        self.tp_degree = tp_degree
        self.world_size = world_size

        # SPMDRank for getting global rank at runtime (crucial for SPMD scatter/gather)
        self.global_rank = SPMDRank(world_size=world_size)

        # DP group for CP communication
        self.data_parallel_group = parallel_state.get_data_parallel_group()

        # Input projections
        self.img_in = original_transformer.img_in
        self.txt_in = original_transformer.txt_in

        # Time/text embedding
        self.time_text_embed = original_transformer.time_text_embed

        # Text norm
        self.txt_norm = original_transformer.txt_norm

        # Transformer blocks with TP sharding
        self.transformer_blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.transformer_blocks):
            # Shard with TP degree
            block.attn = shard_qwen_attention(tp_degree, block.attn)
            block.img_mlp = shard_feedforward(block.img_mlp)
            block.txt_mlp = shard_feedforward(block.txt_mlp)
            block.img_mod = shard_modulation(block.img_mod)
            block.txt_mod = shard_modulation(block.txt_mod)
            self.transformer_blocks.append(block)

            if (i + 1) % 10 == 0:
                print(f"  Sharded block {i+1}/{len(original_transformer.transformer_blocks)}")

        # Replace attention with CP+NKI version
        self._replace_attention()

        # Final layers
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out

        self.head_dim = 128
        self.num_heads = original_transformer.transformer_blocks[0].attn.heads

    def _replace_attention(self):
        """Replace attention modules with CP+NKI versions."""
        for i, block in enumerate(self.transformer_blocks):
            block.attn = CPNKIQwenAttention(
                block.attn, self.context_parallel_enabled, self.data_parallel_group
            )
        print(f"Replaced attention with CP+NKI versions on {len(self.transformer_blocks)} blocks")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_rotary_emb: torch.Tensor,
        txt_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with Context Parallel data splitting."""

        # Store original shapes for verification
        orig_hidden_shape = hidden_states.shape
        orig_enc_shape = encoder_hidden_states.shape

        # ========== CONTEXT PARALLEL: SPLIT DATA AT ENTRY ==========
        if self.context_parallel_enabled:
            # Compute DP rank at runtime using SPMDRank (returns different values per rank)
            dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.tp_degree)

            # Split hidden_states along sequence dim (dim=1)
            hidden_states = split_along_dim(
                hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )

            # Split encoder_hidden_states along sequence dim (dim=1)
            encoder_hidden_states = split_along_dim(
                encoder_hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )

            # Split RoPE along position dim (dim=0)
            img_rotary_emb = split_along_dim(
                img_rotary_emb, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )
            txt_rotary_emb = split_along_dim(
                txt_rotary_emb, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )

        # Split RoPE into cos/sin
        img_freqs_cos = img_rotary_emb[..., 0]
        img_freqs_sin = img_rotary_emb[..., 1]
        txt_freqs_cos = txt_rotary_emb[..., 0]
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

        # Process through blocks
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

        # ========== CONTEXT PARALLEL: GATHER OUTPUT ==========
        if self.context_parallel_enabled:
            # Before gather: output has shape [B, local_patches, C]
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=1, process_group=self.data_parallel_group
            )
            # After gather: output should have shape [B, full_patches, C]
            # Verify that we recovered the original sequence length
            # orig_hidden_shape[1] is the original num_patches

        return output


class TracingWrapper(nn.Module):
    """Wrapper for tracing."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_rotary_emb, txt_rotary_emb):
        return self.transformer(
            hidden_states, encoder_hidden_states, timestep,
            img_rotary_emb, txt_rotary_emb
        )


def get_rope_from_original_model(pipe, frame, height, width, text_seq_len, dtype=torch.bfloat16):
    """Get RoPE from original model."""
    print(f"  Getting RoPE: video_fhw=({frame}, {height}, {width}), text_seq_len={text_seq_len}")

    video_fhw = (frame, height, width)
    vid_freqs, txt_freqs = pipe.transformer.pos_embed(
        video_fhw, txt_seq_lens=[text_seq_len], device=torch.device('cpu')
    )

    img_cos = vid_freqs.real.float()
    img_sin = vid_freqs.imag.float()
    txt_cos = txt_freqs.real.float()
    txt_sin = txt_freqs.imag.float()

    img_rotary_emb = torch.stack([img_cos, img_sin], dim=-1).to(dtype)
    txt_rotary_emb = torch.stack([txt_cos, txt_sin], dim=-1).to(dtype)

    print(f"  img_rotary_emb: {img_rotary_emb.shape}")
    print(f"  txt_rotary_emb: {txt_rotary_emb.shape}")

    return img_rotary_emb, txt_rotary_emb


def compile_transformer_v3_cp(args):
    """Compile transformer with Context Parallel using ModelBuilder API."""

    tp_degree = args.tp_degree
    world_size = args.world_size
    context_parallel_enabled = (world_size != tp_degree)

    if context_parallel_enabled:
        cp_degree = world_size // tp_degree
        print(f"Context Parallel enabled: CP={cp_degree}")
    else:
        cp_degree = 1

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

    # Calculate CP alignment padding (padding goes to patches, not text)
    # This keeps text_seq_len unchanged, avoiding RoPE position issues
    if context_parallel_enabled:
        local_patches = num_patches // cp_degree
        local_text = text_seq_len // cp_degree
        local_total = local_patches + local_text

        # NKI Flash Attention requires sequence length to be multiple of 128
        alignment = 128
        need_padding = (alignment - local_total % alignment) % alignment
        patches_padding = need_padding * cp_degree  # Total padding for patches
        num_patches_padded = num_patches + patches_padding
    else:
        patches_padding = 0
        num_patches_padded = num_patches

    print("=" * 60)
    print("Transformer V3 Context Parallel Compilation")
    print("=" * 60)
    print(f"Image: {args.height}x{args.width}")
    print(f"Original patches: {num_patches}")
    if patches_padding > 0:
        print(f"Padded patches: {num_patches_padded} (+{patches_padding} for CP alignment)")
    print(f"Total text seq: {text_seq_len}")
    print(f"TP degree: {tp_degree}")
    print(f"World size: {world_size}")
    print(f"Context Parallel: {context_parallel_enabled} (CP={cp_degree})")
    print(f"NKI Flash Attention: Enabled")
    print(f"Batch size: {args.batch_size}")

    # Sample inputs (use padded num_patches for compilation)
    batch_size = args.batch_size
    sample_hidden_states = torch.randn(batch_size, num_patches_padded, in_channels, dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_hidden_size, dtype=torch.bfloat16)
    sample_timestep = torch.randn(batch_size, dtype=torch.float32)

    # Use NxDParallelState context for compilation
    # world_size=8, tensor_model_parallel_size=4 means DP=2 (used for CP)
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("\nLoading model...")
        load_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
        if CACHE_DIR:
            load_kwargs["cache_dir"] = CACHE_DIR
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **load_kwargs)

        # Get full RoPE
        print("\nGetting RoPE...")
        img_rotary_emb, txt_rotary_emb = get_rope_from_original_model(
            pipe=pipe,
            frame=temporal_frames,
            height=patch_h,
            width=patch_w,
            text_seq_len=text_seq_len,
        )

        print(f"  img RoPE (original): {img_rotary_emb.shape}")
        print(f"  txt RoPE: {txt_rotary_emb.shape}")

        # Pad img_rotary_emb if needed for CP alignment
        if patches_padding > 0:
            # Repeat last position's RoPE for padding (position doesn't matter for padding tokens)
            rope_padding = img_rotary_emb[-1:].repeat(patches_padding, 1, 1)
            img_rotary_emb = torch.cat([img_rotary_emb, rope_padding], dim=0)
            print(f"  img RoPE (padded): {img_rotary_emb.shape} (+{patches_padding})")

        # Save unsharded state dict before modifications
        unsharded_state = pipe.transformer.state_dict()

        # Create Neuron transformer
        print("\nCreating Neuron transformer (sharding layers with TP={}, world_size={})...".format(tp_degree, world_size))
        neuron_transformer = NeuronQwenTransformerV3CP(
            pipe.transformer, tp_degree, world_size, context_parallel_enabled
        )
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Wrap for tracing
        model = TracingWrapper(neuron_transformer)

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
        # Pass compiler args directly to compile() for State Buffer optimization
        # --enable-native-kernel=1: enables native kernel mode
        # --remat: enables rematerialization to save memory
        # NOTE: Using -O1 instead of -O2 because -O2 can cause numerical issues in some cases
        compile_args = "--model-type=transformer -O1 --auto-cast=none --internal-hlo2tensorizer-options='--enable-native-kernel=1 --remat'"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/transformer_v3_cp"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint for sharding
        checkpoint = {}
        global_rank_state = {}  # Save SPMDRank state separately (not sharded)
        for key, value in model.state_dict().items():
            # Save SPMDRank module state separately - it's not sharded, same on all ranks
            if 'global_rank' in key:
                print(f"  Saving SPMDRank key separately: {key}")
                global_rank_state[key] = value.clone()
                continue
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

        # Post-process sharded checkpoints:
        # 1. Remove master_weight tensors (they duplicate sharded weights, wastes ~50% space)
        # 2. Add global_rank state (SPMDRank) to each checkpoint
        print("\nPost-processing sharded checkpoints...")
        from safetensors.torch import load_file, save_file
        for rank in range(tp_degree):  # Only TP checkpoints are created, CP duplicates them at load time
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                print(f"  WARNING: {shard_file} not found")
                continue

            shard_data = dict(load_file(shard_file))
            original_count = len(shard_data)
            original_size = sum(v.numel() * v.element_size() for v in shard_data.values())

            # Remove master_weight tensors (they duplicate the sharded weights)
            cleaned = {k: v for k, v in shard_data.items() if 'master_weight' not in k}

            # Add SPMDRank state (same value for all ranks)
            if global_rank_state:
                cleaned.update(global_rank_state)

            cleaned_size = sum(v.numel() * v.element_size() for v in cleaned.values())
            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {original_count} -> {len(cleaned)} tensors, "
                  f"{original_size/1e9:.2f}GB -> {cleaned_size/1e9:.2f}GB")

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_patches": num_patches,
            "num_patches_padded": num_patches_padded,
            "patches_padding": patches_padding,
            "text_seq_len": text_seq_len,
            "patch_multiplier": args.patch_multiplier,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "context_parallel": context_parallel_enabled,
            "cp_degree": cp_degree,
            "head_dim": head_dim,
            "frame": temporal_frames,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "nki_flash_attention": True,
            "batch_size": batch_size,
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
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (local dir or HuggingFace ID). If not set, uses MODEL_ID with CACHE_DIR")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--patch_multiplier", type=int, default=3)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for compiled model (default: 1)")
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    args = parser.parse_args()

    # Override MODEL_ID and CACHE_DIR if model_path is provided
    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_transformer_v3_cp(args)
