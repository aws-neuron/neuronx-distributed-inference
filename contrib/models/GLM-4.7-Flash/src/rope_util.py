# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standard Rotary Position Embedding for GLM-4.7-Flash.

GLM-4.7-Flash uses standard RoPE without YaRN scaling extension.
This is a simplified version of the DeepSeek-V3 rope_util.py.
"""

import torch
import torch.utils.checkpoint
from torch import nn


class Glm4MoeLiteRotaryEmbedding(nn.Module):
    """Standard RoPE with no scaling (factor=1.0, no YaRN)."""

    def __init__(self, dim, max_position_embeddings=200000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

    def get_freqs_table(self, device, seq_len):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        return freqs

    def forward(self, x, seq_len=None, freqs=None):
        device = x.device
        dtype = x.dtype
        if freqs is None:
            freqs = self.get_freqs_table(device, seq_len)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def rotate_fn(x: torch.Tensor):
    """Interleaved rotation: pairs (x0,x1) -> (-x1,x0), (x2,x3) -> (-x3,x2), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids):
    """Apply rotary position embedding with interleaved layout."""
    cos_sglang = cos.chunk(2, dim=-1)[0][position_ids]
    sin_sglang = sin.chunk(2, dim=-1)[0][position_ids]

    sin = sin_sglang.repeat_interleave(2, dim=-1)[0]
    cos = cos_sglang.repeat_interleave(2, dim=-1)[0]

    q_embed = (q * cos) + rotate_fn(q) * sin
    return q_embed.to(q.dtype)
