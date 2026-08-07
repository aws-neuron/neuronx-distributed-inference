# coding=utf-8
# Copyright 2026 Google Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemma4 Vision Encoder for NeuronX Distributed Inference

Architecture: gemma4_vision (NOT SigLIP)
- 27 transformer layers with 4-norm sandwich pattern
- Learned 2D position embedding table + 2D RoPE inside encoder layers
- Spatial average pooler (3x3 kernel, output 280 tokens)
- Multi-modal projector: RMSNorm(no_scale) + Linear(1152 -> text_hidden_size)
- MHA (16 heads, head_dim=72), scaling=1.0 with QK norms + V norm
- GELU(tanh) SwiGLU MLP
"""

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

logger = logging.getLogger(__name__)


# ====================================================================================
# Normalization (shared with text decoder)
# ====================================================================================


class Gemma4RMSNorm(nn.Module):
    """Gemma4 RMSNorm: weight * (x / rms(x))."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)


class Gemma4VNorm(nn.Module):
    """RMSNorm WITHOUT learnable scale (with_scale=False)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.type_as(x)


# ====================================================================================
# 2D Rotary Position Embedding for Vision
# ====================================================================================


class Gemma4VisionRotaryEmbedding(nn.Module):
    """
    2D rotary position embedding for Gemma4 vision encoder.

    Head dim is split in half: first half encodes x-position, second half encodes y-position.
    Each spatial dimension gets its own frequency ladder.

    Config: rope_theta=100.0, head_dim=72 -> spatial_dim=36, 18 freq pairs per axis
    """

    def __init__(self, head_dim: int, rope_theta: float = 100.0):
        super().__init__()
        self.head_dim = head_dim
        self.spatial_dim = head_dim // 2  # 36

        # Compute frequencies for one spatial dimension
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.spatial_dim, 2, dtype=torch.float32)
                / self.spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """
        Args:
            x: hidden states [B, seq, hidden] (only used for dtype/device)
            position_ids: [B, num_patches, 2] -- (x, y) spatial coordinates

        Returns:
            cos: [B, num_patches, head_dim]
            sin: [B, num_patches, head_dim]
        """
        all_cos = []
        all_sin = []

        for dim_idx in range(2):  # x and y
            dim_pos = position_ids[:, :, dim_idx].float()  # [B, num_patches]
            # [B, num_patches, 1] @ [1, 1, spatial_dim//2] -> [B, num_patches, spatial_dim//2]
            freqs = dim_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
            # Duplicate for rotate_half: [B, num_patches, spatial_dim]
            emb = torch.cat([freqs, freqs], dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())

        # Concatenate x and y dimensions: [B, num_patches, head_dim]
        cos = torch.cat(all_cos, dim=-1)
        sin = torch.cat(all_sin, dim=-1)
        return cos, sin


def apply_multidimensional_rope(x, cos, sin, unsqueeze_dim=2):
    """
    Apply 2D RoPE: first half of head_dim gets x-rotation, second half gets y-rotation.

    Compiler-friendly implementation that avoids split/cat on small dimensions.
    Instead, we build a full-head_dim rotate_half index and apply the standard
    rotary formula in one pass: x_rotated = x * cos + rotate_half(x) * sin.

    The key insight: for 2D RoPE with head_dim=72 (spatial_dim=36), we treat
    each spatial half independently. Within each half of size 36, rotate_half
    swaps the two sub-halves of size 18. We construct a single rotate_half
    over the full head_dim that applies independently to each spatial part.

    Args:
        x: [B, num_heads, seq, head_dim] (BHSD layout)
        cos, sin: [B, seq, head_dim] (will be unsqueezed for heads dim)
    """
    input_dtype = x.dtype

    # Cast cos/sin to input dtype to avoid float32 promotion
    cos = cos.to(input_dtype).unsqueeze(unsqueeze_dim - 1)  # [B, 1, seq, head_dim]
    sin = sin.to(input_dtype).unsqueeze(unsqueeze_dim - 1)

    # Build rotate_half for full head_dim: within each spatial half,
    # swap sub-halves and negate the first. For head_dim=72:
    #   x_part1 = x[..., :36], x_part2 = x[..., 36:]
    #   rotate_half(x_part1) = cat(-x[...,18:36], x[...,0:18])  -> indices 18..35, 0..17 with negation mask
    #   rotate_half(x_part2) = cat(-x[...,54:72], x[...,36:54])  -> indices 54..71, 36..53 with negation mask
    # Combined: indices [18..35, 0..17, 54..71, 36..53]
    head_dim = x.shape[-1]
    spatial_dim = head_dim // 2
    quarter = spatial_dim // 2

    # rotate_half over full head_dim in one shot
    x_rotated = torch.cat(
        [
            -x[..., quarter:spatial_dim].contiguous(),  # -x[18:36]
            x[..., :quarter].contiguous(),  # x[0:18]
            -x[..., spatial_dim + quarter :].contiguous(),  # -x[54:72]
            x[..., spatial_dim : spatial_dim + quarter].contiguous(),  # x[36:54]
        ],
        dim=-1,
    )

    return x * cos + x_rotated * sin


# ====================================================================================
# Vision Patch Embedder
# ====================================================================================


class Gemma4VisionPatchEmbedder(nn.Module):
    """
    Patch embedding for Gemma4 vision encoder.

    Flattened patches (3 * patch_size^2 = 768) projected to hidden_size via Linear.
    Learned 2D position embedding table: [2, position_embedding_size, hidden_size].
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 1152
        self.patch_size = config.patch_size  # 16
        self.position_embedding_size = config.position_embedding_size  # 10240

        # Linear projection: 3*16*16=768 -> 1152
        input_dim = 3 * self.patch_size * self.patch_size
        self.input_proj = nn.Linear(input_dim, self.hidden_size, bias=False)

        # Learned 2D position table: [2, 10240, 1152]
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        """
        Args:
            pixel_values: [B, num_patches, 3*patch_size^2]
            pixel_position_ids: [B, num_patches, 2] -- (x, y) coords, -1 for padding
            padding_positions: [B, num_patches] -- True where padded
        """
        # Linear projection
        hidden_states = self.input_proj(pixel_values)  # [B, num_patches, 1152]

        # Learned 2D position embeddings via one-hot lookup
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped_positions.long(), num_classes=self.position_embedding_size
        ).to(self.position_embedding_table.dtype)
        # one_hot: [B, num_patches, 2, 10240]
        one_hot = one_hot.permute(0, 2, 1, 3)  # [B, 2, num_patches, 10240]
        position_embeddings = torch.matmul(
            one_hot, self.position_embedding_table
        )  # [B, 2, num_patches, 1152]
        position_embeddings = position_embeddings.sum(dim=1)  # [B, num_patches, 1152]

        # Zero out padding positions
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1),
            torch.zeros_like(position_embeddings),
            position_embeddings,
        )

        return hidden_states + position_embeddings


# ====================================================================================
# Vision Attention
# ====================================================================================


class Gemma4VisionAttention(nn.Module):
    """
    Gemma4 vision attention: MHA with QK norms, V norm, 2D RoPE, scaling=1.0.

    - 16 heads, head_dim=72
    - bias=False on all projections
    - Bidirectional (no causal mask)
    - scaling=1.0 (norms handle magnitude)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size  # 1152
        self.num_heads = config.num_attention_heads  # 16
        self.head_dim = config.head_dim  # 72
        self.layer_idx = layer_idx

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=True,
            dtype=torch.bfloat16,
            pad=True,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=True,
            dtype=torch.bfloat16,
            pad=True,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=True,
            dtype=torch.bfloat16,
            pad=True,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=False,
            dtype=torch.bfloat16,
        )

        # QK norms with learnable scale
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # V norm WITHOUT learnable scale
        self.v_norm = Gemma4VNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        position_ids=None,
    ):
        """
        Args:
            hidden_states: [B, seq, hidden_size]
            attention_mask: optional [B, 1, seq, seq]
            position_embeddings: (cos, sin) from VisionRotaryEmbedding
            position_ids: [B, num_patches, 2] for 2D RoPE
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )

        # Apply norms (on last dim = head_dim)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        # Apply 2D RoPE to Q and K
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Transpose to BHSD for RoPE application
            query_states = query_states.transpose(1, 2)  # [B, H, S, D]
            key_states = key_states.transpose(1, 2)
            query_states = apply_multidimensional_rope(query_states, cos, sin)
            key_states = apply_multidimensional_rope(key_states, cos, sin)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        value_states = value_states.transpose(1, 2)  # [B, H, S, D]

        # Scaled dot-product attention with scaling=1.0
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        # No 1/sqrt(head_dim) scaling -- norms handle magnitude

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        # [B, H, S, D] -> [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


# ====================================================================================
# Vision MLP
# ====================================================================================


class Gemma4VisionMLP(nn.Module):
    """SwiGLU MLP: gate * up with GELU(tanh) on gate."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ====================================================================================
# Vision Encoder Layer (4-norm sandwich)
# ====================================================================================


class Gemma4VisionEncoderLayer(nn.Module):
    """
    Gemma4 vision encoder layer with 4-norm sandwich pattern:
    input_layernorm -> attn -> post_attention_layernorm -> residual
    pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.mlp = Gemma4VisionMLP(config)

        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        position_ids=None,
    ):
        # Attention block (sandwich norm)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block (sandwich norm)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


# ====================================================================================
# Vision Encoder
# ====================================================================================


class Gemma4VisionEncoder(nn.Module):
    """27-layer Gemma4 vision transformer encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(
            head_dim=config.head_dim,
            rope_theta=config.rope_parameters.get("rope_theta", 100.0)
            if isinstance(config.rope_parameters, dict)
            else getattr(config.rope_parameters, "rope_theta", 100.0),
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, inputs_embeds, attention_mask=None, pixel_position_ids=None):
        """
        Args:
            inputs_embeds: [B, num_patches, hidden_size]
            attention_mask: optional [B, 1, seq, seq]
            pixel_position_ids: [B, num_patches, 2] for 2D RoPE
        """
        hidden_states = inputs_embeds

        # Compute 2D RoPE embeddings once for all layers
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
            hidden_states = layer_outputs[0]

        return BaseModelOutput(last_hidden_state=hidden_states)


# ====================================================================================
# Vision Pooler
# ====================================================================================


class Gemma4VisionPooler(nn.Module):
    """
    Spatial average pooling: pools 3x3 patches into 1 token,
    then scales by sqrt(hidden_size).
    No learnable parameters.
    """

    def __init__(self, config):
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5  # sqrt(1152) ≈ 33.94
        self.pooling_kernel_size = config.pooling_kernel_size  # 3

    def forward(
        self, hidden_states, pixel_position_ids, padding_positions, output_length
    ):
        """
        Args:
            hidden_states: [B, num_patches, hidden_size]
            pixel_position_ids: [B, num_patches, 2]
            padding_positions: [B, num_patches] -- True where padded
            output_length: target number of output tokens (e.g., 280)

        Returns:
            pooled: [B, output_length, hidden_size]
            pooler_mask: [B, output_length] -- True where valid
        """
        # Zero out padding
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, pooler_mask = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, padding_positions, output_length
            )
        else:
            pooler_mask = ~padding_positions

        # Scale by sqrt(hidden_size)
        hidden_states = hidden_states * self.root_hidden_size

        return hidden_states, pooler_mask

    def _avg_pool_by_positions(
        self, hidden_states, pixel_position_ids, padding_positions, output_length
    ):
        """
        Position-aware 2D spatial average pooling.

        Groups patches by their (x//k, y//k) bin, averages within each bin.
        k = pooling_kernel_size (3).
        """
        k = self.pooling_kernel_size
        k_squared = k * k
        bsz = hidden_states.shape[0]

        clamped_positions = pixel_position_ids.clamp(min=0)
        # Compute bin indices from 2D positions
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        # Flatten 2D bin to 1D: bin_x + (max_x // k) * bin_y
        bin_1d = (
            kernel_idxs[..., 0]
            + (torch.div(max_x, k, rounding_mode="floor")) * kernel_idxs[..., 1]
        )

        # Zero out padding bin indices
        bin_1d = torch.where(padding_positions, torch.zeros_like(bin_1d), bin_1d)

        # Build weight matrix via one-hot and average
        weights = F.one_hot(bin_1d.long(), output_length).float()
        # Zero out padding contributions
        weights = weights * (~padding_positions).unsqueeze(-1).float()
        # Normalize by number of valid patches per bin
        bin_counts = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights = weights / bin_counts

        # Pool: [B, output_length, num_patches] @ [B, num_patches, hidden] -> [B, output_length, hidden]
        pooled = torch.bmm(weights.transpose(1, 2), hidden_states.float()).to(
            hidden_states.dtype
        )

        # Pooler mask: bins with at least one valid patch
        pooler_mask = weights.sum(dim=1) > 0

        return pooled, pooler_mask


# ====================================================================================
# Vision Model (full pipeline)
# ====================================================================================


class Gemma4VisionModel(nn.Module):
    """
    Complete Gemma4 vision model:
    pixel_values -> patch_embedder -> encoder (27 layers) -> pooler -> standardize
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        # Standardization buffers (config.standardize=True for 31B)
        if getattr(config, "standardize", False):
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))
            self.register_buffer("std_scale", torch.ones(config.hidden_size))
            self._standardize = True
        else:
            self._standardize = False

    def forward(self, pixel_values, pixel_position_ids):
        """
        Args:
            pixel_values: [B, num_patches, 3*patch_size^2] -- flattened patches
            pixel_position_ids: [B, num_patches, 2] -- (x, y) spatial coords, -1 for padding

        Returns:
            hidden_states: [total_valid_tokens, hidden_size] -- flattened, padding stripped
        """
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[1] // (
            pooling_kernel_size * pooling_kernel_size
        )

        # Detect padding patches
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [B, num_patches]

        # Pixel normalization: [0,1] -> [-1,1]
        pixel_values = 2.0 * (pixel_values - 0.5)

        # Patch embedding + learned 2D position embedding
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        # Build bidirectional attention mask (None = full attention, or mask padding)
        attention_mask = None
        if padding_positions.any():
            # Create mask: [B, 1, 1, num_patches] -- True=attend, False=mask
            valid_mask = (~padding_positions).unsqueeze(1).unsqueeze(1).float()
            # Convert to additive mask (-inf for masked positions)
            attention_mask = (1.0 - valid_mask) * torch.finfo(inputs_embeds.dtype).min

        # Encoder (27 transformer layers with 2D RoPE)
        encoder_output = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states = encoder_output.last_hidden_state

        # Spatial pooling (3x3 avg pool, scale by sqrt(hidden_size))
        hidden_states, pooler_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length
        )

        # Strip padding tokens
        hidden_states = hidden_states[pooler_mask]  # [total_valid_tokens, hidden_size]

        # Standardize (if enabled)
        if self._standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states


# ====================================================================================
# Multi-Modal Projector
# ====================================================================================


class Gemma4MultimodalEmbedder(nn.Module):
    """
    Projects vision encoder output to text hidden space.
    RMSNorm(no_scale) + Linear(vision_hidden -> text_hidden)

    Weight name in safetensors: model.embed_vision.embedding_projection.weight
    """

    def __init__(
        self, vision_hidden_size: int, text_hidden_size: int, eps: float = 1e-6
    ):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4VNorm(vision_hidden_size, eps=eps)
        self.embedding_projection = nn.Linear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def forward(self, inputs_embeds):
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


# ====================================================================================
# Full Vision + Projector (compiled as one NEFF)
# ====================================================================================


class NeuronGemma4VisionModel(nn.Module):
    """
    Combined vision encoder + multi-modal projector.
    This gets compiled as a single Neuron NEFF.

    pixel_values -> VisionModel -> MultimodalEmbedder -> text-space embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_config = (
            config.vision_config if hasattr(config, "vision_config") else config
        )

        vision_cfg = self.vision_config
        self.vision_model = Gemma4VisionModel(vision_cfg)

        text_hidden_size = (
            config.text_config.hidden_size
            if hasattr(config, "text_config")
            else config.hidden_size
        )
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_hidden_size,
            eps=vision_cfg.rms_norm_eps,
        )

    def forward(self, pixel_values, pixel_position_ids):
        """
        Args:
            pixel_values: [B, num_patches, 3*patch_size^2]
            pixel_position_ids: [B, num_patches, 2]

        Returns:
            projected_embeddings: [total_valid_tokens, text_hidden_size]
        """
        vision_output = self.vision_model(pixel_values, pixel_position_ids)
        projected = self.embed_vision(vision_output)
        return projected
