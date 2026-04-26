# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MoonViT vision encoder for Kimi-K2.5 on Neuron.
#
# Architecture: 27-layer ViT (hidden=1152, heads=16, mlp=4304)
#   - 400M parameters (466M with projector)
#   - 2D RoPE (real-number decomposition, no complex ops)
#   - Eager attention (no flash_attn)
#   - PatchMergerMLP projector: 2x2 merge → 7168 text hidden dim
#
# Input: pixel_values [N_patches, 3, 14, 14] (patchified image)
# Output: projected embeddings [N_merged, 7168]
#
# For 448x448 image: 1024 patches → 256 merged vision tokens
#
# Usage:
#   1. Create wrapper: NeuronMoonViTWrapper(patch_h=32, patch_w=32)
#   2. Load weights: load_vision_weights(model, model_path, 32, 32)
#   3. Trace on Neuron: torch_neuronx.trace(model, (pixels, cos, sin))
#   4. Run inference: model(pixel_values, rope_cos, rope_sin) → [256, 7168]
#
# Note: All 64 Neuron cores are consumed by the text decoder (TP=64).
# MoonViT must be traced and run BEFORE loading the text decoder, or
# pre-computed embeddings must be used.

import json
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_SIZE = 14
MERGE_KERNEL = (2, 2)


# ============================================================================
# Neuron-compatible attention
# ============================================================================


def eager_attention(q, k, v):
    """Simple eager attention for a single sequence.

    Args:
        q, k, v: [seq_len, num_heads, head_dim]
    Returns:
        output: [seq_len, hidden_dim]
    """
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scale = math.sqrt(q.shape[-1])
    attn_weight = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(0, 1).reshape(q.shape[1], -1)
    return attn_output


# ============================================================================
# Real-number RoPE (no complex ops)
# ============================================================================


def precompute_rope_real(dim, max_height, max_width, theta_base=10000):
    """Precompute 2D RoPE cos/sin tables using real-number decomposition.

    Returns:
        cos_table: [max_height, max_width, dim//2]
        sin_table: [max_height, max_width, dim//2]
    """
    N = max_height * max_width
    flat_pos = torch.arange(0, N).float()
    x_pos = flat_pos % max_width
    y_pos = flat_pos // max_width

    dim_range = torch.arange(0, dim, 4)[: dim // 4].float()
    freqs = 1.0 / (theta_base ** (dim_range / dim))

    x_freqs = torch.outer(x_pos, freqs)
    y_freqs = torch.outer(y_pos, freqs)

    cos_x, sin_x = torch.cos(x_freqs), torch.sin(x_freqs)
    cos_y, sin_y = torch.cos(y_freqs), torch.sin(y_freqs)

    cos_table = torch.stack([cos_x, cos_y], dim=-1).reshape(N, -1)
    sin_table = torch.stack([sin_x, sin_y], dim=-1).reshape(N, -1)

    return cos_table.reshape(max_height, max_width, -1), sin_table.reshape(
        max_height, max_width, -1
    )


def apply_rope_real(xq, xk, cos_table, sin_table):
    """Apply 2D RoPE using real-number cos/sin decomposition.

    Args:
        xq, xk: [..., num_heads, head_dim]
        cos_table, sin_table: [..., head_dim/2]
    Returns:
        xq_out, xk_out: [..., num_heads, head_dim]
    """
    cos = cos_table.unsqueeze(-2)
    sin = sin_table.unsqueeze(-2)

    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_real, xq_imag = xq_pairs[..., 0], xq_pairs[..., 1]
    xk_real, xk_imag = xk_pairs[..., 0], xk_pairs[..., 1]

    xq_out_real = xq_real * cos - xq_imag * sin
    xq_out_imag = xq_real * sin + xq_imag * cos
    xk_out_real = xk_real * cos - xk_imag * sin
    xk_out_imag = xk_real * sin + xk_imag * cos

    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1).flatten(-2)

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


# ============================================================================
# Encoder Layer
# ============================================================================


class NeuronMoonViTEncoderLayer(nn.Module):
    """Single MoonViT encoder layer with real-number RoPE and eager attention."""

    def __init__(self, num_heads, hidden_dim, mlp_dim):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=True)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=True)

    def forward(self, hidden_states, rope_cos, rope_sin):
        """
        Args:
            hidden_states: [seq_len, hidden_dim]
            rope_cos, rope_sin: [seq_len, head_dim/2]
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        xqkv = self.wqkv(hidden_states)
        xqkv = xqkv.view(-1, 3, self.num_heads, self.head_dim)
        xq, xk, xv = xqkv[:, 0], xqkv[:, 1], xqkv[:, 2]

        xq, xk = apply_rope_real(xq, xk, rope_cos, rope_sin)
        attn_out = eager_attention(xq, xk, xv)
        attn_out = self.wo(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.fc0(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc1(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# MoonViT Wrapper (Neuron-traceable)
# ============================================================================


class NeuronMoonViTWrapper(nn.Module):
    """Neuron-traceable MoonViT + PatchMergerMLP wrapper.

    For a fixed image resolution (default 448x448):
    - Input: pixel_values [N_patches, 3, 14, 14] (patchified)
    - Output: projected embeddings [N_merged, text_hidden_size]

    Pipeline:
    1. Conv2d patch embedding
    2. Learnable positional embedding (+ temporal at t=0)
    3. 27 encoder layers with 2D RoPE + eager attention
    4. Final LayerNorm
    5. PatchMergerMLP: 2x2 merge → Linear → GELU → Linear → 7168
    """

    def __init__(
        self,
        num_layers=27,
        hidden_dim=1152,
        num_heads=16,
        mlp_dim=4304,
        text_hidden_size=7168,
        patch_h=32,
        patch_w=32,
        merge_kernel=(2, 2),
        projector_ln_eps=1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.merge_kernel = merge_kernel
        self.seq_len = patch_h * patch_w

        merged_h = patch_h // merge_kernel[0]
        merged_w = patch_w // merge_kernel[1]
        self.n_merged = merged_h * merged_w
        self.merge_dim = hidden_dim * merge_kernel[0] * merge_kernel[1]

        self.patch_conv = nn.Conv2d(3, hidden_dim, kernel_size=14, stride=14)

        self.layers = nn.ModuleList(
            [
                NeuronMoonViTEncoderLayer(num_heads, hidden_dim, mlp_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # PatchMergerMLP projector
        self.proj_norm = nn.LayerNorm(hidden_dim, eps=projector_ln_eps)
        self.proj_fc0 = nn.Linear(self.merge_dim, self.merge_dim)
        self.proj_fc1 = nn.Linear(self.merge_dim, text_hidden_size)

    def forward(self, pixel_values, rope_cos, rope_sin):
        """
        Args:
            pixel_values: [N_patches, 3, 14, 14]
            rope_cos: [seq_len, head_dim/2]
            rope_sin: [seq_len, head_dim/2]
        Returns:
            projected: [N_merged, text_hidden_size]
        """
        x = self.patch_conv(pixel_values).view(pixel_values.shape[0], -1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        x = self.final_norm(x)

        # Patch merging (2x2)
        kh, kw = self.merge_kernel
        new_h = self.patch_h // kh
        new_w = self.patch_w // kw
        x = x.view(1, new_h, kh, new_w, kw, self.hidden_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.squeeze(0)
        x = x.reshape(new_h * new_w, kh * kw, self.hidden_dim)

        # Projector
        x = self.proj_norm(x)
        x = x.reshape(self.n_merged, -1)
        x = self.proj_fc0(x)
        x = F.gelu(x)
        x = self.proj_fc1(x)

        return x


# ============================================================================
# Weight Loading
# ============================================================================


def load_vision_weights(model, model_path, patch_h, patch_w, device="cpu"):
    """Load MoonViT + PatchMergerMLP weights from K2.5 safetensors.

    Args:
        model: NeuronMoonViTWrapper instance
        model_path: Path to K2.5 HF model directory
        patch_h: Patch grid height
        patch_w: Patch grid width
        device: Device for weight loading

    Returns:
        model with loaded weights
    """
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    vision_keys = {
        k: v
        for k, v in weight_map.items()
        if k.startswith("vision_tower.") or k.startswith("mm_projector.")
    }

    shard_to_keys = {}
    for key, shard in vision_keys.items():
        shard_to_keys.setdefault(shard, []).append(key)

    all_weights = {}
    for shard, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(model_path, shard)
        with safe_open(shard_path, framework="pt", device=str(device)) as f:
            for key in keys:
                all_weights[key] = f.get_tensor(key)

    state_dict = {}

    # Patch conv
    state_dict["patch_conv.weight"] = all_weights[
        "vision_tower.patch_embed.proj.weight"
    ]
    state_dict["patch_conv.bias"] = all_weights["vision_tower.patch_embed.proj.bias"]

    # Positional embedding
    pos_weight = all_weights["vision_tower.patch_embed.pos_emb.weight"]
    init_h, init_w, dim = pos_weight.shape
    if (init_h, init_w) == (patch_h, patch_w):
        pos_embed = pos_weight.flatten(end_dim=1)
    else:
        pos_embed = (
            F.interpolate(
                pos_weight.permute(2, 0, 1).unsqueeze(0),
                size=(patch_h, patch_w),
                mode="bicubic",
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .flatten(end_dim=1)
        )

    # Temporal embedding (t=0 for single image)
    time_weight = all_weights.get("vision_tower.patch_embed.pos_emb.time_weight")
    if time_weight is not None:
        pos_embed = pos_embed + time_weight[0]

    state_dict["pos_embed"] = pos_embed

    # Encoder layers
    for i in range(27):
        prefix = f"vision_tower.encoder.blocks.{i}"
        layer_prefix = f"layers.{i}"
        state_dict[f"{layer_prefix}.norm0.weight"] = all_weights[
            f"{prefix}.norm0.weight"
        ]
        state_dict[f"{layer_prefix}.norm0.bias"] = all_weights[f"{prefix}.norm0.bias"]
        state_dict[f"{layer_prefix}.norm1.weight"] = all_weights[
            f"{prefix}.norm1.weight"
        ]
        state_dict[f"{layer_prefix}.norm1.bias"] = all_weights[f"{prefix}.norm1.bias"]
        state_dict[f"{layer_prefix}.wqkv.weight"] = all_weights[f"{prefix}.wqkv.weight"]
        state_dict[f"{layer_prefix}.wqkv.bias"] = all_weights[f"{prefix}.wqkv.bias"]
        state_dict[f"{layer_prefix}.wo.weight"] = all_weights[f"{prefix}.wo.weight"]
        state_dict[f"{layer_prefix}.wo.bias"] = all_weights[f"{prefix}.wo.bias"]
        state_dict[f"{layer_prefix}.fc0.weight"] = all_weights[
            f"{prefix}.mlp.fc0.weight"
        ]
        state_dict[f"{layer_prefix}.fc0.bias"] = all_weights[f"{prefix}.mlp.fc0.bias"]
        state_dict[f"{layer_prefix}.fc1.weight"] = all_weights[
            f"{prefix}.mlp.fc1.weight"
        ]
        state_dict[f"{layer_prefix}.fc1.bias"] = all_weights[f"{prefix}.mlp.fc1.bias"]

    # Final LayerNorm
    state_dict["final_norm.weight"] = all_weights[
        "vision_tower.encoder.final_layernorm.weight"
    ]
    state_dict["final_norm.bias"] = all_weights[
        "vision_tower.encoder.final_layernorm.bias"
    ]

    # PatchMergerMLP projector
    state_dict["proj_norm.weight"] = all_weights["mm_projector.pre_norm.weight"]
    state_dict["proj_norm.bias"] = all_weights["mm_projector.pre_norm.bias"]
    state_dict["proj_fc0.weight"] = all_weights["mm_projector.proj.0.weight"]
    state_dict["proj_fc0.bias"] = all_weights["mm_projector.proj.0.bias"]
    state_dict["proj_fc1.weight"] = all_weights["mm_projector.proj.2.weight"]
    state_dict["proj_fc1.bias"] = all_weights["mm_projector.proj.2.bias"]

    # Register pos_embed as buffer
    pos_embed = state_dict.pop("pos_embed")
    model.register_buffer("pos_embed", pos_embed)

    model.load_state_dict(state_dict, strict=False)
    return model
