import math

import torch
import torch.utils.checkpoint
from torch import nn


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
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
        cos = (emb.cos()).to(dtype)
        sin = (emb.sin()).to(dtype)
        return cos, sin


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):

    def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1,
            mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self._mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        super().__init__(dim, max_position_embeddings, base, device)

    def get_freqs_table(self, device, seq_len):
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)
        return freqs

    def forward(self, x, seq_len=None, freqs=None):
        device = x.device
        dtype = x.dtype
        if freqs is None:
            freqs = self.get_freqs_table(device, seq_len)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self._mscale).to(dtype)
        sin = (emb.sin() * self._mscale).to(dtype)
        return cos, sin


def rotate_fn(x: torch.Tensor):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids):
    cos_sglang = cos.chunk(2, dim=-1)[0][position_ids]
    sin_sglang = sin.chunk(2, dim=-1)[0][position_ids]

    sin = sin_sglang.repeat_interleave(2, dim=-1)[0]
    cos = cos_sglang.repeat_interleave(2, dim=-1)[0]

    q_embed = (q * cos) + rotate_fn(q) * sin
    return q_embed.to(q.dtype)
