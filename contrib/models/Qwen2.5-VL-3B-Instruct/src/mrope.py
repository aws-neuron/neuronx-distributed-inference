# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
Multimodal Rotary Position Embeddings (MRoPE) for Qwen2.5-VL

MRoPE extends 1D RoPE to 3D for multimodal inputs:
- For vision tokens: Applies separate rotary embeddings on temporal, height, and width dimensions
- For text tokens: All three position indices are the same, reducing to standard 1D RoPE
"""

import torch
import torch.nn as nn


class Qwen2VLRotaryEmbedding(nn.Module):
    """
    Multimodal Rotary Position Embedding for Qwen2.5-VL
    
    This implements MRoPE which applies 3D rotary position embeddings for vision
    tokens (temporal, height, width) and standard 1D rotary embeddings for text tokens.
    """
    
    def __init__(self, dim, max_position_embeddings=128000, base=1000000.0, device=None):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
        
        # For compatibility with inference
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Forward pass for MRoPE
        
        Args:
            x: Input tensor (for device/dtype reference)
            position_ids: Position indices with shape (3, batch_size, seq_len)
                         [temporal_positions, height_positions, width_positions]
        
        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        # Expand inv_freq to match position_ids shape
        # inv_freq shape: (dim/2,)
        # Need shape: (3, batch_size, dim/2, 1) for broadcasting
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        inv_freq_expanded = inv_freq_expanded.expand(3, position_ids.shape[1], -1, 1)
        
        # position_ids shape: (3, batch_size, seq_len)
        # Reshape to (3, batch_size, 1, seq_len) for matmul
        position_ids_expanded = position_ids[:, :, None, :].float()
        
        # Compute frequencies
        # Result shape: (3, batch_size, dim/2, seq_len)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Apply attention scaling (Qwen2-VL uses scaling factor of 1.0)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    
    This is a helper function for applying rotary embeddings.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding with Multimodal Sections to query and key tensors.
    
    This implements the MRoPE mechanism from the Qwen2-VL paper, which applies separate
    rotary embeddings to different parts of the hidden dimension corresponding to
    temporal, height, and width positional information.
    
    Args:
        q: Query tensor with shape (batch, heads, seq_len, head_dim)
        k: Key tensor with shape (batch, heads, seq_len, head_dim)
        cos: Cosine part of rotary embedding with shape (3, batch, seq_len, head_dim)
        sin: Sine part of rotary embedding with shape (3, batch, seq_len, head_dim)
        mrope_section: List of 3 integers [temporal_dim, height_dim, width_dim] 
                      defining how to split head_dim
        unsqueeze_dim: Dimension to unsqueeze for broadcasting (default=1 for heads dim)
    
    Returns:
        Tuple of (q_embed, k_embed) with rotary position embeddings applied
    """
    # mrope_section defines how to split the head dimension
    # For example, [16, 24, 24] means:
    # - First 16 dims get temporal rotary embedding
    # - Next 24 dims get height rotary embedding  
    # - Last 24 dims get width rotary embedding
    
    # Double the sections since we have both cos and sin
    mrope_section = [s * 2 for s in mrope_section]
    
    # Split cos and sin along head_dim according to mrope_section
    # Then interleave them according to the 3D position indices
    cos_parts = cos.split(mrope_section, dim=-1)
    sin_parts = sin.split(mrope_section, dim=-1)
    
    # Reconstruct cos and sin by taking the appropriate part for each section
    # cos has shape (3, batch, seq_len, head_dim) where first dim is [temporal, height, width]
    cos = torch.cat([cos_parts[i % 3][i % 3] for i in range(len(mrope_section))], dim=-1)
    sin = torch.cat([sin_parts[i % 3][i % 3] for i in range(len(mrope_section))], dim=-1)
    
    # Unsqueeze to add heads dimension for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """
    Apply rotary position embeddings for vision tokens.
    
    This is used in the vision encoder for 2D spatial rotary embeddings.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding
    
    Returns:
        Tuple of (q_embed, k_embed) with rotary position embeddings applied
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed
