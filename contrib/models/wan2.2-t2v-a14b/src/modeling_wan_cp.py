# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# Copyright 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# Neuron context-parallel (CP) port of diffusers.models.transformers.transformer_wan.
# Every rank holds full weights; the sequence dimension is split across CP ranks.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Deviation from HF: Neuron CP infrastructure imports
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_size,
    get_world_group,
)
from neuronx_distributed_inference.utils.distributed import get_dp_rank_spmd


def split_along_dim(tensor: torch.Tensor, dim: int, rank, process_group) -> torch.Tensor:
    """Scatter (split) a tensor along the given dimension across CP ranks."""
    return scatter_to_process_group_spmd(tensor, partition_dim=dim, rank=rank, process_group=process_group)


# ---------------------------------------------------------------------------
# Inlined HF helpers (Deviation from HF: inlined to avoid import dependency)
# ---------------------------------------------------------------------------

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        exp = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / (half - self.downscale_freq_shift)
        emb = timesteps[:, None].float() * torch.exp(exp)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
        return emb


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inputs.float(), self.normalized_shape,
                            self.weight.float() if self.weight is not None else None,
                            self.bias.float() if self.bias is not None else None, self.eps).to(inputs.dtype)


def apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """Wan-style RoPE. Input: [B, heads, S, head_dim], freqs: [1, S, 1, D] transposed to [1, 1, S, D]."""
    freqs_cos = freqs_cos.transpose(1, 2)
    freqs_sin = freqs_sin.transpose(1, 2)
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).type_as(hidden_states)


# ---------------------------------------------------------------------------
# NeuronWanAttention — matches WanAttention forward signature and state dict
# ---------------------------------------------------------------------------

class NeuronWanAttention(nn.Module):
    """Neuron CP port of WanAttention.

    Deviation from HF: In self-attention, K/V are all-gathered across CP ranks
    so Q (local) can attend to the full sequence. Cross-attention is fully local.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=True)

        self.norm_q = torch.nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        cp_group=None,  # Deviation from HF: CP process group for sequence-parallel all-gather
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Deviation from HF: CP all-gather K/V to full sequence for self-attention
        if not self.is_cross_attention and cp_group is not None:
            key = gather_from_tensor_model_parallel_region_with_dim(key, gather_dim=2, process_group=cp_group)
            value = gather_from_tensor_model_parallel_region_with_dim(value, gather_dim=2, process_group=cp_group)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.head_dim).type_as(query)

        hidden_states = self.to_out(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# NeuronWanFeedForward — matches FeedForward state dict (net.0.proj / net.2)
# ---------------------------------------------------------------------------

class NeuronWanFeedForward(nn.Module):
    """Fully local FFN. State dict: up_proj.weight, down_proj.weight."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, inner_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


# ---------------------------------------------------------------------------
# NeuronWanTransformerBlock — matches WanTransformerBlock
# ---------------------------------------------------------------------------

class NeuronWanTransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False, eps: float = 1e-6, added_kv_proj_dim: int | None = None):
        super().__init__()
        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = NeuronWanAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps)
        # 2. Cross-attention
        self.attn2 = NeuronWanAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
                                        cross_attention_dim_head=dim // num_heads)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # 3. Feed-forward
        self.ffn = NeuronWanFeedForward(dim, inner_dim=ffn_dim)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor, rotary_emb: torch.Tensor,
                cp_group=None,  # Deviation from HF: CP group for context parallelism
                ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, cp_group=cp_group)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None, cp_group=None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# NeuronWanTimeTextImageEmbedding — matches WanTimeTextImageEmbedding
# ---------------------------------------------------------------------------

class NeuronWanTimeTextImageEmbedding(nn.Module):
    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int,
                 image_embed_dim: int | None = None, pos_embed_seq_len: int | None = None):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder_linear_1 = nn.Linear(time_freq_dim, dim, bias=True)
        self.time_embedder_act = nn.SiLU()
        self.time_embedder_linear_2 = nn.Linear(dim, dim, bias=True)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim, bias=True)
        self.text_embedder_linear_1 = nn.Linear(text_embed_dim, dim, bias=True)
        self.text_embedder_act = nn.GELU(approximate="tanh")
        self.text_embedder_linear_2 = nn.Linear(dim, dim, bias=True)
        self.image_embedder = None  # Deviation from HF: I2V not ported

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor,
                encoder_hidden_states_image: torch.Tensor | None = None,
                timestep_seq_len: int | None = None) -> tuple:
        t_emb = self.timesteps_proj(timestep).to(encoder_hidden_states.dtype)
        temb = self.time_embedder_linear_2(self.time_embedder_act(self.time_embedder_linear_1(t_emb)))
        temb = temb.type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb)).unflatten(1, (6, -1))
        encoder_hidden_states = self.text_embedder_linear_2(
            self.text_embedder_act(self.text_embedder_linear_1(encoder_hidden_states)))
        return temb, timestep_proj, encoder_hidden_states


# ---------------------------------------------------------------------------
# NeuronWanTransformer3DModel — matches WanTransformer3DModel
# ---------------------------------------------------------------------------

class NeuronWanTransformer3DModel(nn.Module):
    """Neuron CP port of WanTransformer3DModel. Full model (not split)."""

    def __init__(self, patch_size=(1, 2, 2), num_attention_heads=40, attention_head_dim=128,  # config defaults
                 in_channels=16, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=13824,  # config defaults
                 num_layers=40, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,  # config defaults
                 image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024, pos_embed_seq_len=None, **kwargs):  # config
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers

        # 1. Patch & position embedding
        self.rope = None  # Deviation from HF: RoPE computed externally, passed as args
        self.register_buffer("freqs_cos", torch.zeros(1), persistent=False)
        self.register_buffer("freqs_sin", torch.zeros(1), persistent=False)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        # 2. Condition embeddings
        self.condition_embedder = NeuronWanTimeTextImageEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim, image_embed_dim=image_dim, pos_embed_seq_len=pos_embed_seq_len)
        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            NeuronWanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
            for _ in range(num_layers)])
        # 4. Output
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, self.out_channels * math.prod(patch_size), bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        # Deviation from HF: CP infrastructure
        self._dp_group = None
        self.global_rank = SPMDRank(world_size=get_world_group().size())

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor,
                encoder_hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                encoder_hidden_states_image: torch.Tensor | None = None, return_dict: bool = True,
                attention_kwargs: dict[str, Any] | None = None, **kwargs) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)

        # Deviation from HF: CP scatter sequence and RoPE
        if self._dp_group is None:
            self._dp_group = get_data_parallel_group()
        cp_group = self._dp_group
        cp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), get_tensor_model_parallel_size())
        hidden_states = split_along_dim(hidden_states, 1, cp_rank, cp_group)
        freqs_cos = split_along_dim(freqs_cos, 1, cp_rank, cp_group)
        freqs_sin = split_along_dim(freqs_sin, 1, cp_rank, cp_group)
        rotary_emb = (freqs_cos, freqs_sin)

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, cp_group=cp_group)

        # Output
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Deviation from HF: CP all-gather before unpatchify
        hidden_states = gather_from_tensor_model_parallel_region_with_dim(hidden_states, gather_dim=1, process_group=cp_group)

        # Deviation from HF: decomposed unpatchify (XLA-safe, avoids 8D reshape)
        output = self._unpatchify(hidden_states, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
        if not return_dict:
            return (output,)
        return output

    def _unpatchify(self, hidden_states, batch_size, ppf, pph, ppw, p_t, p_h, p_w):
        D = self.out_channels * p_t * p_h * p_w
        hidden_states = hidden_states.reshape(batch_size, ppf, pph * ppw, D)
        hidden_states = hidden_states.reshape(batch_size * ppf, pph * ppw, p_t, p_h, p_w, self.out_channels)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 3, 4)
        hidden_states = hidden_states.reshape(batch_size * ppf, pph * ppw, D).permute(0, 2, 1)
        hidden_states = hidden_states.reshape(batch_size * ppf, D, pph, ppw)
        hidden_states = F.pixel_shuffle(hidden_states, p_h)
        hidden_states = hidden_states.reshape(batch_size, ppf, self.out_channels * p_t, pph * p_h, ppw * p_w)
        return hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size, self.out_channels, p_t * ppf, pph * p_h, ppw * p_w)


# ---------------------------------------------------------------------------
# Split-model halves (port-specific, not in HF reference)
# ---------------------------------------------------------------------------

class CPWanFirstHalf(nn.Module):
    """First half: patch_embedding + condition_embedder + blocks[0:20] + CP scatter."""

    def __init__(self, *, patch_size, num_attention_heads, attention_head_dim,
                 in_channels, out_channels, text_dim, freq_dim, ffn_dim,
                 num_layers, cross_attn_norm, eps, num_first_half_layers=20, **kwargs):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = NeuronWanTimeTextImageEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6, text_embed_dim=text_dim)
        self.blocks = nn.ModuleList([
            NeuronWanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, "rms_norm_across_heads", cross_attn_norm, eps)
            for _ in range(num_first_half_layers)])
        self._dp_group = None
        self.global_rank = SPMDRank(world_size=get_world_group().size())

    def forward(self, hidden_states, timestep, encoder_hidden_states, freqs_cos, freqs_sin):
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)

        # Deviation from HF: CP scatter
        if self._dp_group is None:
            self._dp_group = get_data_parallel_group()
        cp_group = self._dp_group
        cp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), get_tensor_model_parallel_size())
        hidden_states = split_along_dim(hidden_states, 1, cp_rank, cp_group)
        freqs_cos = split_along_dim(freqs_cos, 1, cp_rank, cp_group)
        freqs_sin = split_along_dim(freqs_sin, 1, cp_rank, cp_group)
        rotary_emb = (freqs_cos, freqs_sin)

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, cp_group=cp_group)

        # Deviation from HF: gather back to full sequence for inter-NEFF transfer
        hidden_states = gather_from_tensor_model_parallel_region_with_dim(hidden_states, gather_dim=1, process_group=cp_group)
        return hidden_states, temb, timestep_proj, encoder_hidden_states


class CPWanSecondHalf(nn.Module):
    """Second half: blocks[20:40] + norm_out + proj_out + gather + unpatchify."""

    def __init__(self, *, patch_size, num_attention_heads, attention_head_dim,
                 in_channels, out_channels, text_dim, freq_dim, ffn_dim,
                 num_layers, cross_attn_norm, eps, num_second_half_layers=20,
                 num_frames=13, height=480, width=832, **kwargs):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        p_t, p_h, p_w = patch_size
        latent_f = (num_frames - 1) // 4 + 1
        self.post_patch_num_frames = latent_f // p_t
        self.post_patch_height = (height // 8) // p_h
        self.post_patch_width = (width // 8) // p_w

        self.blocks = nn.ModuleList([
            NeuronWanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, "rms_norm_across_heads", cross_attn_norm, eps)
            for _ in range(num_second_half_layers)])
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size), bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self._dp_group = None
        self.global_rank = SPMDRank(world_size=get_world_group().size())

    def forward(self, hidden_states, temb, timestep_proj, encoder_hidden_states_proj, freqs_cos, freqs_sin):
        # Deviation from HF: CP scatter for second half
        if self._dp_group is None:
            self._dp_group = get_data_parallel_group()
        cp_group = self._dp_group
        cp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), get_tensor_model_parallel_size())
        hidden_states = split_along_dim(hidden_states, 1, cp_rank, cp_group)
        freqs_cos = split_along_dim(freqs_cos, 1, cp_rank, cp_group)
        freqs_sin = split_along_dim(freqs_sin, 1, cp_rank, cp_group)
        rotary_emb = (freqs_cos, freqs_sin)

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states_proj, timestep_proj, rotary_emb, cp_group=cp_group)

        # Output
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Deviation from HF: CP gather before unpatchify
        hidden_states = gather_from_tensor_model_parallel_region_with_dim(hidden_states, gather_dim=1, process_group=cp_group)

        # Unpatchify
        p_t, p_h, p_w = self.patch_size
        batch_size = hidden_states.shape[0]
        ppf, pph, ppw = self.post_patch_num_frames, self.post_patch_height, self.post_patch_width
        D = self.out_channels * p_t * p_h * p_w
        hidden_states = hidden_states.reshape(batch_size, ppf, pph * ppw, D)
        hidden_states = hidden_states.reshape(batch_size * ppf, pph * ppw, p_t, p_h, p_w, self.out_channels)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 3, 4)
        hidden_states = hidden_states.reshape(batch_size * ppf, pph * ppw, D).permute(0, 2, 1)
        hidden_states = hidden_states.reshape(batch_size * ppf, D, pph, ppw)
        hidden_states = F.pixel_shuffle(hidden_states, p_h)
        hidden_states = hidden_states.reshape(batch_size, ppf, self.out_channels * p_t, pph * p_h, ppw * p_w)
        return hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size, self.out_channels, p_t * ppf, pph * p_h, ppw * p_w)


# --- Backward-compatible aliases ---
CPWanAttention = NeuronWanAttention
CPWanFeedForward = NeuronWanFeedForward
CPWanTransformerBlock = NeuronWanTransformerBlock
CPWanTimeTextImageEmbedding = NeuronWanTimeTextImageEmbedding
CPWanTransformer3DModel = NeuronWanTransformer3DModel


def convert_hf_to_neuron_state_dict(state_dict: dict, config=None) -> dict:
    """Convert HF diffusers state dict to Neuron CP port state dict."""
    new_sd = {}
    for key, value in state_dict.items():
        if key.startswith("rope."):
            continue
        new_key = key
        new_key = new_key.replace(".attn1.to_out.0.", ".attn1.to_out.0.")  # preserved
        new_key = new_key.replace(".attn2.to_out.0.", ".attn2.to_out.0.")  # preserved
        new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.up_proj.")
        new_key = new_key.replace(".ffn.net.2.", ".ffn.down_proj.")
        new_key = new_key.replace("condition_embedder.text_embedder.linear_1.", "condition_embedder.text_embedder_linear_1.")
        new_key = new_key.replace("condition_embedder.text_embedder.linear_2.", "condition_embedder.text_embedder_linear_2.")
        new_key = new_key.replace("condition_embedder.time_embedder.linear_1.", "condition_embedder.time_embedder_linear_1.")
        new_key = new_key.replace("condition_embedder.time_embedder.linear_2.", "condition_embedder.time_embedder_linear_2.")
        new_sd[new_key] = value
    return new_sd
