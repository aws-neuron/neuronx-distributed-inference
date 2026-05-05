# coding=utf-8
# Copyright 2025 Google LLC. Ported to standalone PyTorch.
#
# Licensed under the Apache License, Version 2.0

"""
Gemma-4 Vision Encoder (ViT) + Image Processing + Embedder.

Runs on CPU. Produces image embeddings that can be injected into
the Neuron-compiled text decoder via the vision_embeddings/vision_mask mechanism.

Architecture:
    Image (PIL/numpy)
    → Resize (aspect-ratio-preserving, divisible by patch_size * pooling_kernel_size)
    → Patchify (16×16 patches → flattened 768-dim vectors)
    → Position Embedding (2D one-hot @ learned table)
    → 16 ViT layers (RoPE, QK/V-norm, Gated MLP)
    → 2D Average Pooling (kernel=3 → 9x reduction)
    → Scale by sqrt(hidden_size)
    → Multimodal Embedder (RMSNorm + Linear → text_hidden_size)
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Gemma4VisionConfig:
    """Vision encoder configuration from config.json['vision_config']."""
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    intermediate_size: int = 3072
    head_dim: int = 64
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    default_output_length: int = 280
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_clipped_linears: bool = True
    standardize: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Gemma4VisionConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {}
        for k, v in d.items():
            if k in fields:
                kw[k] = v
            elif k == "rope_parameters" and isinstance(v, dict):
                if "rope_theta" in v:
                    kw["rope_theta"] = v["rope_theta"]
        return cls(**kw)


# ===========================================================================
# Image Preprocessing
# ===========================================================================

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int, width: int,
    patch_size: int, max_patches: int, pooling_kernel_size: int,
) -> tuple:
    """Compute target size preserving aspect ratio within patch budget."""
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    max_side_length = (max_patches // pooling_kernel_size ** 2) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("Image too small for given patch budget")
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    return target_height, target_width


def preprocess_image(
    image,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
):
    """
    Preprocess a PIL image for the Gemma-4 vision encoder.

    Returns:
        pixel_values: [1, max_patches, patch_size*patch_size*3] float32 tensor
        position_ids: [1, max_patches, 2] long tensor
        num_soft_tokens: int
    """
    from PIL import Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not hasattr(image, 'size'):
        raise ValueError("Expected PIL Image or path string")

    max_patches = max_soft_tokens * pooling_kernel_size ** 2
    height, width = image.size[1], image.size[0]

    # Aspect-ratio-preserving resize
    target_h, target_w = get_aspect_ratio_preserving_size(
        height, width, patch_size, max_patches, pooling_kernel_size,
    )

    if target_h != height or target_w != width:
        image = image.resize((target_w, target_h), Image.BILINEAR)

    # Convert to tensor [C, H, W], rescale to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]

    C, H, W = img_tensor.shape
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Patchify: [3, H, W] → [num_patches, patch_size*patch_size*3]
    patches = img_tensor.reshape(C, patch_h, patch_size, patch_w, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0)  # [ph, pw, ps, ps, 3]
    patches = patches.reshape(patch_h * patch_w, -1)  # [num_patches, 768]
    num_patches = patches.shape[0]
    num_soft_tokens = num_patches // (pooling_kernel_size ** 2)

    # Position IDs: (x, y) for each patch
    grid_x, grid_y = torch.meshgrid(
        torch.arange(patch_w), torch.arange(patch_h), indexing="xy",
    )
    positions = torch.stack([grid_x, grid_y], dim=-1).reshape(num_patches, 2)

    # Pad to max_patches
    pad_len = max_patches - num_patches
    if pad_len > 0:
        patches = F.pad(patches, (0, 0, 0, pad_len), value=0.0)
        positions = F.pad(positions, (0, 0, 0, pad_len), value=-1)

    return patches.unsqueeze(0), positions.unsqueeze(0), num_soft_tokens


# ===========================================================================
# Vision Encoder Components
# ===========================================================================


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class ClippableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_clipping: bool = True):
        super().__init__()
        self.use_clipping = use_clipping
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if use_clipping:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipping:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):
    """Apply 2D RoPE: split head_dim into x and y halves, rotate each independently."""
    ndim = position_ids.shape[-1]  # 2 for images
    num_rotated_per_dim = 2 * (x.shape[-1] // (2 * ndim))
    split_sizes = [num_rotated_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class VisionRotaryEmbedding(nn.Module):
    """2D RoPE for vision patches."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        base = config.rope_theta
        dim = config.head_dim
        spatial_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2).float() / spatial_dim))
        self.register_buffer("inv_freq", inv_freq[None, :, None], persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: [B, S, H] hidden states (for dtype/device)
            position_ids: [B, S, 2] patch (x, y) positions
        Returns:
            (cos, sin) each [B, S, head_dim]
        """
        inv_freq = self.inv_freq.float().expand(position_ids.shape[0], -1, 1).to(x.device)
        all_cos, all_sin = [], []
        for i in range(2):
            dim_pos = position_ids[:, :, i].float()[:, None, :]  # [B, 1, S]
            freqs = (inv_freq @ dim_pos).transpose(1, 2)  # [B, S, dim//4]
            emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, dim//2]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)  # [B, S, dim]
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(3 * self.patch_size ** 2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        pos_emb = one_hot @ self.position_embedding_table
        pos_emb = pos_emb.sum(dim=1)
        pos_emb = torch.where(padding_positions.unsqueeze(-1), 0.0, pos_emb)
        return pos_emb

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        pos_emb = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + pos_emb


class VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        clip = config.use_clipped_linears
        self.q_proj = ClippableLinear(config.hidden_size, self.num_heads * self.head_dim, use_clipping=clip)
        self.k_proj = ClippableLinear(config.hidden_size, self.num_kv_heads * self.head_dim, use_clipping=clip)
        self.v_proj = ClippableLinear(config.hidden_size, self.num_kv_heads * self.head_dim, use_clipping=clip)
        self.o_proj = ClippableLinear(self.num_heads * self.head_dim, config.hidden_size, use_clipping=clip)

        self.q_norm = Gemma4RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(config.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None):
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_multidimensional_rope(q, cos, sin, position_ids)
        q = q.transpose(1, 2)  # [B, H, S, D]

        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = apply_multidimensional_rope(k, cos, sin, position_ids)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        # Expand KV for GQA (here num_heads == num_kv_heads, so no-op)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (bf16 like HF reference)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=1.0,
        )

        attn = attn.transpose(1, 2).reshape(B, S, -1).to(hidden_states.dtype)
        return self.o_proj(attn)


class VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        clip = config.use_clipped_linears
        self.gate_proj = ClippableLinear(config.hidden_size, config.intermediate_size, use_clipping=clip)
        self.up_proj = ClippableLinear(config.hidden_size, config.intermediate_size, use_clipping=clip)
        self.down_proj = ClippableLinear(config.intermediate_size, config.hidden_size, use_clipping=clip)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = VisionAttention(config, layer_idx)
        self.mlp = VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionPooler(nn.Module):
    """2D spatial average pooling with sqrt(hidden_size) scaling."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.root_hidden_size = config.hidden_size ** 0.5

    def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k ** 2

        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length,
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


# ===========================================================================
# Top-level Vision Encoder
# ===========================================================================


class Gemma4VisionEncoder(nn.Module):
    """
    Complete Gemma4 vision encoder (ViT).

    Input: patchified pixel values [B, num_patches, 768] + position_ids [B, num_patches, 2]
    Output: pooled vision features [num_valid_tokens, hidden_size]
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = VisionPatchEmbedder(config)
        self.rotary_emb = VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.pooler = VisionPooler(config)

    @torch.no_grad()
    def forward(self, pixel_values, pixel_position_ids):
        """
        Args:
            pixel_values: [B, max_patches, patch_dim] float tensor
            pixel_position_ids: [B, max_patches, 2] long tensor, -1 for padding
        Returns:
            hidden_states: [num_valid_tokens, hidden_size] (padding stripped)
        """
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pooling_kernel_size ** 2)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)

        # Build bidirectional attention mask for padding
        # ~padding = True for valid tokens
        valid_mask = ~padding_positions  # [B, S]
        # For SDPA: [B, 1, 1, S] float mask (0 = attend, -inf = mask)
        attn_mask = valid_mask[:, None, None, :].float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                position_ids=pixel_position_ids,
            )

        hidden_states, pooler_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length,
        )

        # Strip padding
        hidden_states = hidden_states[pooler_mask]

        return hidden_states


class Gemma4VisionEmbedder(nn.Module):
    """Projects vision encoder output into text embedding space."""

    def __init__(self, vision_hidden_size: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4RMSNorm(vision_hidden_size, eps=eps, with_scale=False)
        self.embedding_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


# ===========================================================================
# Weight Loading
# ===========================================================================


def load_vision_encoder(model_path: str, config: Gemma4VisionConfig,
                        text_hidden_size: int, dtype=torch.bfloat16):
    """
    Load vision encoder + embedder from HF checkpoint.

    Returns:
        (encoder, embedder) tuple
    """
    import glob
    from safetensors import safe_open

    encoder = Gemma4VisionEncoder(config)
    embedder = Gemma4VisionEmbedder(config.hidden_size, text_hidden_size, config.rms_norm_eps)

    vision_weights = {}
    embedder_weights = {}
    files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    for f in files:
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if "vision_tower." in key:
                    local_key = key.replace("model.vision_tower.", "")
                    # HF uses encoder.layers.X, we use layers.X directly
                    local_key = local_key.replace("encoder.layers.", "layers.")
                    # HF uses encoder.rotary_emb, we use rotary_emb
                    local_key = local_key.replace("encoder.rotary_emb.", "rotary_emb.")
                    vision_weights[local_key] = st.get_tensor(key)
                elif "embed_vision." in key:
                    local_key = key.replace("model.embed_vision.", "")
                    embedder_weights[local_key] = st.get_tensor(key)

    missing, unexpected = encoder.load_state_dict(vision_weights, strict=False)
    if missing:
        print(f"  Vision encoder missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Vision encoder unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    missing_e, unexpected_e = embedder.load_state_dict(embedder_weights, strict=False)
    if missing_e:
        print(f"  Vision embedder missing keys ({len(missing_e)}): {missing_e[:5]}...")
    if unexpected_e:
        print(f"  Vision embedder unexpected keys ({len(unexpected_e)}): {unexpected_e[:5]}...")

    encoder = encoder.to(dtype=dtype).eval()
    embedder = embedder.to(dtype=dtype).eval()

    return encoder, embedder
