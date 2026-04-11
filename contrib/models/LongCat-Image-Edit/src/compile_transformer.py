"""
LongCat FLUX-style Transformer compilation with Context Parallel (Compiled).

Key approach:
1. Uses ModelBuilder API for compilation
2. Configures world_size=8, tp_degree=4 (implicit CP=2)
3. K/V are all-gathered across DP group before attention
4. Uses NKI Flash Attention for optimal performance

LongCat Transformer Architecture (FLUX-style):
- 10 dual-stream blocks (FluxTransformerBlock): separate text/image norms+FFN, joint attention
- 20 single-stream blocks (FluxSingleTransformerBlock): concatenated text+image, parallel MLP+attention
- 24 attention heads, head_dim=128, inner_dim=3072
- joint_attention_dim=3584, in_channels=64 (packed latents)
- ~6.2B parameters
"""

import os
import json
import math

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --auto-cast=none --enable-fast-loading-neuron-binaries --tensorizer-options='--enable-ccop-compute-overlap' --internal-hlo2tensorizer-options='--enable-state-buffer-mode=hybrid --remat-by-default' """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Tuple, List

from diffusers import LongCatImageEditPipeline

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
    shard_flux_dual_block,
    shard_flux_single_block,
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

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"


def nki_flash_attention(query, key, value):
    """NKI Flash Attention wrapper. Args all [B, H, S, D]."""
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


def apply_rotary_emb_precomputed(x, freqs_cos, freqs_sin):
    """
    Apply FLUX-style real-valued rotary embeddings using pre-computed cos/sin.

    LongCat's pos_embed outputs (cos, sin) each [S, D] where D = head_dim (128),
    already repeat_interleaved. So we do NOT expand them here.

    The rotation uses use_real_unbind_dim=-1 convention (same as FLUX):
        x is stored as [x0_real, x0_imag, x1_real, x1_imag, ...]
        rotated = [-x0_imag, x0_real, -x1_imag, x1_real, ...]

    Args:
        x: [B, S, H, D] input tensor (sequence_dim=1)
        freqs_cos: [S, D] cosine values (full head_dim, already repeat_interleaved)
        freqs_sin: [S, D] sine values (full head_dim, already repeat_interleaved)

    Returns:
        Tensor with RoPE applied, same shape as x
    """
    # cos/sin are [S, D] -- expand to [1, S, 1, D] for broadcasting with [B, S, H, D]
    cos = freqs_cos.unsqueeze(0).unsqueeze(2).to(x.device)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2).to(x.device)

    # Create rotated: [-x_imag, x_real, -x_imag, x_real, ...] (use_real_unbind_dim=-1)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)  # [B, S, H, D]

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class CPNKIFluxDualAttention(nn.Module):
    """
    Context Parallel + NKI Flash Attention for FLUX dual-stream blocks.

    In dual-stream blocks, text and image have separate QKV projections
    but attend jointly (concatenated K/V).
    """

    def __init__(self, orig_attn, context_parallel_enabled=False, data_parallel_group=None):
        super().__init__()
        self.context_parallel_enabled = context_parallel_enabled
        self.data_parallel_group = data_parallel_group
        self.heads = orig_attn.heads

        # Image stream projections
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # Text stream projections
        self.add_q_proj = orig_attn.add_q_proj if hasattr(orig_attn, 'add_q_proj') else None
        self.add_k_proj = orig_attn.add_k_proj if hasattr(orig_attn, 'add_k_proj') else None
        self.add_v_proj = orig_attn.add_v_proj if hasattr(orig_attn, 'add_v_proj') else None
        self.to_add_out = orig_attn.to_add_out if hasattr(orig_attn, 'to_add_out') else None

        # QK normalization (per-head, NOT sharded)
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None
        self.norm_added_q = orig_attn.norm_added_q if hasattr(orig_attn, 'norm_added_q') else None
        self.norm_added_k = orig_attn.norm_added_k if hasattr(orig_attn, 'norm_added_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: Tuple = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with CP K/V gathering, RoPE, and NKI attention."""
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

        # Apply RoPE (FLUX-style, already real-valued)
        if image_rotary_emb is not None:
            img_cos, img_sin, txt_cos, txt_sin = image_rotary_emb
            # RoPE expects [B, S, H, D], transpose back
            img_query = apply_rotary_emb_precomputed(
                img_query.transpose(1, 2), img_cos, img_sin).transpose(1, 2)
            img_key = apply_rotary_emb_precomputed(
                img_key.transpose(1, 2), img_cos, img_sin).transpose(1, 2)
            txt_query = apply_rotary_emb_precomputed(
                txt_query.transpose(1, 2), txt_cos, txt_sin).transpose(1, 2)
            txt_key = apply_rotary_emb_precomputed(
                txt_key.transpose(1, 2), txt_cos, txt_sin).transpose(1, 2)

        # Context Parallel: All-gather K/V across DP group
        if self.context_parallel_enabled:
            img_stacked_kv = torch.stack([img_key, img_value], dim=0)
            img_stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                img_stacked_kv, gather_dim=3, process_group=self.data_parallel_group)
            img_key, img_value = torch.unbind(img_stacked_kv, dim=0)

            txt_stacked_kv = torch.stack([txt_key, txt_value], dim=0)
            txt_stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                txt_stacked_kv, gather_dim=3, process_group=self.data_parallel_group)
            txt_key, txt_value = torch.unbind(txt_stacked_kv, dim=0)

        # Concatenate for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        # NKI Flash Attention
        joint_hidden_states = nki_flash_attention(joint_query, joint_key, joint_value)

        # Reshape and split
        joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = self.to_out[0](img_attn_output)
        if len(self.to_out) > 1:
            img_attn_output = self.to_out[1](img_attn_output)

        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class CPNKIFluxSingleAttention(nn.Module):
    """
    Context Parallel + NKI Flash Attention for FLUX single-stream blocks.

    In single-stream blocks, text and image are already concatenated.
    Self-attention is performed on the concatenated sequence.
    """

    def __init__(self, orig_attn, context_parallel_enabled=False, data_parallel_group=None):
        super().__init__()
        self.context_parallel_enabled = context_parallel_enabled
        self.data_parallel_group = data_parallel_group
        self.heads = orig_attn.heads

        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v

        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: Tuple = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward: self-attention on concatenated text+image sequence."""
        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = query.shape[-1]
        head_dim = inner_dim // self.heads

        # Reshape to [B, H, S, D]
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE (single stream operates on concatenated sequence)
        if image_rotary_emb is not None:
            full_cos, full_sin = image_rotary_emb
            query = apply_rotary_emb_precomputed(
                query.transpose(1, 2), full_cos, full_sin).transpose(1, 2)
            key = apply_rotary_emb_precomputed(
                key.transpose(1, 2), full_cos, full_sin).transpose(1, 2)

        # Context Parallel: All-gather K/V
        if self.context_parallel_enabled:
            stacked_kv = torch.stack([key, value], dim=0)
            stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                stacked_kv, gather_dim=3, process_group=self.data_parallel_group)
            key, value = torch.unbind(stacked_kv, dim=0)

        # NKI Flash Attention
        attn_output = nki_flash_attention(query, key, value)

        # Reshape
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim)
        attn_output = attn_output.to(query.dtype)

        return attn_output


def split_along_dim(tensor, dim, rank, data_parallel_group):
    """Split tensor along dimension using scatter_to_process_group_spmd."""
    return scatter_to_process_group_spmd(
        tensor, partition_dim=dim, rank=rank, process_group=data_parallel_group)


def get_dp_rank_spmd(global_rank, tp_degree):
    """Compute DP rank from global rank. Ranks 0-3 -> DP 0, Ranks 4-7 -> DP 1."""
    return torch.div(global_rank, tp_degree, rounding_mode="floor").to(torch.int32)


class NeuronLongCatTransformer(nn.Module):
    """
    Neuron-optimized LongCat FLUX-style Transformer with Context Parallel.

    Forward flow:
    1. x_embedder(hidden_states) -> [B, img_seq, 3072]
    2. context_embedder(encoder_hidden_states) -> [B, txt_seq, 3072]
    3. time_embed(timestep) -> [B, 3072]
    4. CP scatter: split img/txt sequences across CP ranks
    5. 10x dual-stream blocks (joint attention with CP all-gather K/V)
    6. 20x single-stream blocks (self-attention with CP all-gather K/V)
    7. CP gather: reconstruct full sequence
    8. norm_out + proj_out -> [B, img_seq, 64]
    """

    def __init__(self, original_transformer, tp_degree, world_size, context_parallel_enabled=False):
        super().__init__()

        self.config = original_transformer.config
        self.context_parallel_enabled = context_parallel_enabled
        self.tp_degree = tp_degree
        self.world_size = world_size

        self.global_rank = SPMDRank(world_size=world_size)
        self.data_parallel_group = parallel_state.get_data_parallel_group()

        # Input projections (FLUX-style)
        self.x_embedder = original_transformer.x_embedder  # Linear(64, 3072)
        self.context_embedder = original_transformer.context_embedder  # Linear(3584, 3072)

        # Time embedding (LongCat uses 'time_embed', not 'time_text_embed')
        self.time_embed = original_transformer.time_embed

        # Dual-stream blocks (10 blocks)
        self.transformer_blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.transformer_blocks):
            block = shard_flux_dual_block(tp_degree, block)
            self.transformer_blocks.append(block)
            if (i + 1) % 5 == 0:
                print(f"  Sharded dual-stream block {i+1}/{len(original_transformer.transformer_blocks)}")

        # Single-stream blocks (20 blocks)
        self.single_transformer_blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.single_transformer_blocks):
            block = shard_flux_single_block(tp_degree, block)
            self.single_transformer_blocks.append(block)
            if (i + 1) % 10 == 0:
                print(f"  Sharded single-stream block {i+1}/{len(original_transformer.single_transformer_blocks)}")

        # Replace attention with CP+NKI versions
        self._replace_attention()

        # Final layers
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out

        self.head_dim = 128
        self.num_heads = self.transformer_blocks[0].attn.heads if hasattr(self.transformer_blocks[0], 'attn') else 6

    def _replace_attention(self):
        """Replace attention modules with CP+NKI versions."""
        for i, block in enumerate(self.transformer_blocks):
            block.attn = CPNKIFluxDualAttention(
                block.attn, self.context_parallel_enabled, self.data_parallel_group)
        print(f"  Replaced {len(self.transformer_blocks)} dual-stream attention modules")

        for i, block in enumerate(self.single_transformer_blocks):
            block.attn = CPNKIFluxSingleAttention(
                block.attn, self.context_parallel_enabled, self.data_parallel_group)
        print(f"  Replaced {len(self.single_transformer_blocks)} single-stream attention modules")

    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, img_seq, 64] packed latents
        encoder_hidden_states: torch.Tensor, # [B, txt_seq, 3584]
        timestep: torch.Tensor,              # [B] (raw, will be * 1000 internally)
        img_rotary_cos: torch.Tensor,        # [img_seq, 128] (head_dim, repeat_interleaved)
        img_rotary_sin: torch.Tensor,        # [img_seq, 128]
        txt_rotary_cos: torch.Tensor,        # [txt_seq, 128]
        txt_rotary_sin: torch.Tensor,        # [txt_seq, 128]
    ) -> torch.Tensor:
        """Forward pass with Context Parallel data splitting."""

        # Input projections
        hidden_states = self.x_embedder(hidden_states)  # [B, img_seq, 3072]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # [B, txt_seq, 3072]

        # Time embedding (original multiplies by 1000, time_embed needs dtype)
        timestep = timestep.to(hidden_states.dtype) * 1000
        temb = self.time_embed(timestep, hidden_states.dtype)  # [B, 3072]

        # ========== CONTEXT PARALLEL: SPLIT DATA AT ENTRY ==========
        if self.context_parallel_enabled:
            dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.tp_degree)

            hidden_states = split_along_dim(
                hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group)
            encoder_hidden_states = split_along_dim(
                encoder_hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group)

            # Split RoPE
            img_rotary_cos = split_along_dim(
                img_rotary_cos, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group)
            img_rotary_sin = split_along_dim(
                img_rotary_sin, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group)
            txt_rotary_cos = split_along_dim(
                txt_rotary_cos, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group)
            txt_rotary_sin = split_along_dim(
                txt_rotary_sin, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group)

        # Dual-stream blocks
        dual_rope = (img_rotary_cos, img_rotary_sin, txt_rotary_cos, txt_rotary_sin)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=dual_rope,
            )

        # Single-stream blocks
        # Each block takes separate (hidden_states, encoder_hidden_states),
        # concatenates internally, processes, and splits back.
        single_cos = torch.cat([txt_rotary_cos, img_rotary_cos], dim=0)
        single_sin = torch.cat([txt_rotary_sin, img_rotary_sin], dim=0)
        single_rope = (single_cos, single_sin)

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=single_rope,
            )

        # Final norm and projection (only on image hidden states)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # ========== CONTEXT PARALLEL: GATHER OUTPUT ==========
        if self.context_parallel_enabled:
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=1, process_group=self.data_parallel_group)

        return output


class TracingWrapper(nn.Module):
    """Wrapper for tracing."""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_rotary_cos, img_rotary_sin, txt_rotary_cos, txt_rotary_sin):
        return self.transformer(
            hidden_states, encoder_hidden_states, timestep,
            img_rotary_cos, img_rotary_sin, txt_rotary_cos, txt_rotary_sin)


def compile_transformer(args):
    """Compile FLUX-style transformer with Context Parallel using ModelBuilder API."""

    tp_degree = args.tp_degree
    world_size = args.world_size
    context_parallel_enabled = (world_size != tp_degree)
    cp_degree = world_size // tp_degree if context_parallel_enabled else 1

    # Calculate dimensions
    # Pipeline does 2x2 packing (_pack_latents): [B,16,H,W] -> [B,(H/2)*(W/2), 64]
    # Model config says patch_size=1 (no additional patchification on top of packing)
    vae_scale_factor = 8
    latent_h = 2 * (args.height // (vae_scale_factor * 2))  # Match pipeline calc
    latent_w = 2 * (args.width // (vae_scale_factor * 2))
    patch_h = latent_h // 2  # After 2x2 FLUX packing
    patch_w = latent_w // 2

    # For image editing: target + source image patches
    num_img_patches = 2 * patch_h * patch_w
    text_seq_len = args.max_sequence_length

    text_hidden_size = 3584  # Qwen2.5-VL hidden size
    in_channels = 64  # packed latent channels
    head_dim = 128

    # CP alignment padding
    if context_parallel_enabled:
        local_img = num_img_patches // cp_degree
        local_txt = text_seq_len // cp_degree
        local_total = local_img + local_txt

        alignment = 128
        need_padding = (alignment - local_total % alignment) % alignment
        img_padding = need_padding * cp_degree
        num_img_patches_padded = num_img_patches + img_padding
    else:
        img_padding = 0
        num_img_patches_padded = num_img_patches

    print("=" * 60)
    print("LongCat FLUX Transformer Compilation")
    print("=" * 60)
    print(f"Image: {args.height}x{args.width}")
    print(f"Image patches (target+source): {num_img_patches}")
    if img_padding > 0:
        print(f"Padded image patches: {num_img_patches_padded} (+{img_padding})")
    print(f"Text seq len: {text_seq_len}")
    print(f"TP={tp_degree}, World={world_size}, CP={cp_degree}")
    print(f"Batch size: {args.batch_size}")

    batch_size = args.batch_size

    # Load pipeline first (need it for RoPE computation)
    print("\nLoading model...")
    load_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)

    # Pre-compute RoPE using model's own pos_embed (exact match with inference)
    from neuron_rope import compute_rope_from_model
    txt_cos, txt_sin, img_cos, img_sin = compute_rope_from_model(
        pipe, height=args.height, width=args.width,
        text_seq_len=text_seq_len, dtype=torch.bfloat16,
    )

    # Pad img RoPE if needed for CP alignment
    if img_padding > 0:
        rope_padding_cos = img_cos[-1:].repeat(img_padding, 1)
        rope_padding_sin = img_sin[-1:].repeat(img_padding, 1)
        img_cos = torch.cat([img_cos, rope_padding_cos], dim=0)
        img_sin = torch.cat([img_sin, rope_padding_sin], dim=0)

    print(f"RoPE: img_cos={img_cos.shape}, txt_cos={txt_cos.shape}")

    sample_hidden_states = torch.randn(batch_size, num_img_patches_padded, in_channels, dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_hidden_size, dtype=torch.bfloat16)
    sample_timestep = torch.randn(batch_size, dtype=torch.float32)

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # Save unsharded state dict
        unsharded_state = pipe.transformer.state_dict()

        # Create Neuron transformer
        print(f"\nCreating Neuron transformer (TP={tp_degree}, world_size={world_size})...")
        neuron_transformer = NeuronLongCatTransformer(
            pipe.transformer, tp_degree, world_size, context_parallel_enabled)
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        model = TracingWrapper(neuron_transformer)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "encoder_hidden_states": sample_encoder_hidden_states,
                "timestep": sample_timestep,
                "img_rotary_cos": img_cos,
                "img_rotary_sin": img_sin,
                "txt_rotary_cos": txt_cos,
                "txt_rotary_sin": txt_sin,
            },
            tag="inference",
        )

        print("Compiling model...")
        compile_args = "--model-type=transformer -O1 --auto-cast=none --lnc=2 --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4' --internal-hlo2tensorizer-options='--enable-native-kernel=1 --remat'"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/transformer"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        checkpoint = {}
        global_rank_state = {}
        for key, value in model.state_dict().items():
            if 'global_rank' in key:
                global_rank_state[key] = value.clone()
                continue
            orig_key = key.replace("transformer.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        print("Sharding weights...")
        shard_checkpoint(checkpoint=checkpoint, model=model, serialize_path=weights_path)

        # Post-process: clean up + fix proj_out interleaved weight sharding
        # shard_checkpoint() uses standard contiguous column sharding for RowParallel,
        # but proj_out in single-stream blocks needs non-contiguous interleaved columns
        # because the per-rank input is [attn_shard, mlp_shard] not contiguous columns.
        print("\nPost-processing sharded checkpoints...")
        from safetensors.torch import load_file, save_file

        # Get proj_out dimensions from original model
        attn_dim = pipe.transformer.config.num_attention_heads * head_dim  # 24 * 128 = 3072
        num_single_blocks = len(neuron_transformer.single_transformer_blocks)
        mlp_dim = pipe.transformer.single_transformer_blocks[0].mlp_hidden_dim  # 12288

        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                continue
            shard_data = dict(load_file(shard_file))
            original_count = len(shard_data)
            cleaned = {k: v for k, v in shard_data.items() if 'master_weight' not in k}
            # Fix global_rank.rank: SPMDRank needs each TP rank to have its own
            # rank value (torch.tensor([rank])). Without this, all ranks think
            # they are rank 0, breaking CP scatter/gather operations.
            for gk, gv in global_rank_state.items():
                cleaned[gk] = torch.tensor([rank], dtype=torch.int32)

            # Fix proj_out weights for all single-stream blocks
            attn_per_rank = attn_dim // tp_degree
            mlp_per_rank = mlp_dim // tp_degree
            for block_idx in range(num_single_blocks):
                w_key = f"transformer.single_transformer_blocks.{block_idx}.proj_out.weight"
                if w_key in cleaned:
                    # Get original unsharded weight
                    orig_key = f"single_transformer_blocks.{block_idx}.proj_out.weight"
                    orig_w = unsharded_state[orig_key]
                    # Extract correct non-contiguous columns for this rank
                    attn_start = rank * attn_per_rank
                    attn_end = (rank + 1) * attn_per_rank
                    mlp_start = attn_dim + rank * mlp_per_rank
                    mlp_end = attn_dim + (rank + 1) * mlp_per_rank
                    w_attn = orig_w[:, attn_start:attn_end]
                    w_mlp = orig_w[:, mlp_start:mlp_end]
                    cleaned[w_key] = torch.cat([w_attn, w_mlp], dim=1).to(torch.bfloat16)

            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {original_count} -> {len(cleaned)} tensors")

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_img_patches": num_img_patches,
            "num_img_patches_padded": num_img_patches_padded,
            "img_padding": img_padding,
            "text_seq_len": text_seq_len,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "context_parallel": context_parallel_enabled,
            "cp_degree": cp_degree,
            "head_dim": head_dim,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "pack_size": 2,  # FLUX 2x2 packing
            "nki_flash_attention": True,
            "batch_size": batch_size,
            "model_type": "flux",
            "num_dual_blocks": len(neuron_transformer.transformer_blocks),
            "num_single_blocks": len(neuron_transformer.single_transformer_blocks),
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save pre-computed RoPE
        torch.save({
            "img_rotary_cos": img_cos,
            "img_rotary_sin": img_sin,
            "txt_rotary_cos": txt_cos,
            "txt_rotary_sin": txt_sin,
        }, os.path.join(output_path, "rope_cache.pt"))

        print("\nCompilation complete!")
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    args = parser.parse_args()

    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_transformer(args)
