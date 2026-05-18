# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FlashVSR DiT model for NxD Inference on AWS Trainium.

Implements the FlashVSR video super-resolution DiT (Denoising Diffusion Transformer)
optimized for AWS Neuron hardware using NxD Inference (NxDI) compilation patterns.

Architecture (Wan 2.1 1.3B variant):
  - 30 DiT blocks with self-attention (LCSA) + cross-attention (text conditioning)
  - Factored 3D RoPE (temporal + height + width)
  - QK-norm with DistributedRMSNorm for TP accuracy
  - NKI tiled flash attention (attention_cte) -- never materializes full S*S matrix
  - TP sharding: ColumnParallel Q/K/V, RowParallel O (both self and cross-attn)

Two compilation modes:
  "first": f=6 latent frames (first chunk). No KV cache input.
  "stream": f=2 latent frames (streaming chunks). No KV cache (Phase 1).

Production config: trn2.3xlarge, TP=4, SDK 2.29, 8.27 FPS.

Model: JunhaoZhuang/FlashVSR-v1.1
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------------------------------------------------
# TP sharding helpers -- graceful fallback when NxDI is not installed
# -------------------------------------------------------------------

try:
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import (
        set_tensor_model_parallel_attributes,
    )
    import neuronx_distributed.trace.trace as _nxd_trace

    from nkilib.core.attention.attention_cte import (
        attention_cte as _nkilib_attention_cte,
    )

    HAS_NXDI = True
    HAS_NKI_FLASH = True
except ImportError:
    ColumnParallelLinear = None
    RowParallelLinear = None
    HAS_NXDI = False
    HAS_NKI_FLASH = False


# -------------------------------------------------------------------
# Block-sparse LCSA toggle (Phase 2 -- requires trn2.48xlarge TP=16)
# -------------------------------------------------------------------
# When True, self-attention uses per-layer block-sparse LCSA masking
# generated INSIDE the traced model via topk + index_select + gather.
# When False (default/production), uses dense attention_cte on full sequence.
USE_BLOCK_SPARSE_LCSA = False

# LCSA hyperparameters (Phase 2 only)
LCSA_TOPK_RATIO = 2.0
LCSA_LOCAL_RANGE = 11
LCSA_MAX_ACTIVE = 130
LCSA_CTE_CHUNK_SIZE = 30

# -------------------------------------------------------------------
# Model constants (Wan 2.1 1.3B / FlashVSR)
# -------------------------------------------------------------------

DIM = 1536
FFN_DIM = 8960
NUM_HEADS = 12
HEAD_DIM = 128
NUM_LAYERS = 30
PATCH_T, PATCH_H, PATCH_W = 1, 2, 2
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
EPS = 1e-6

# LCSA window
LCSA_WIN = (2, 8, 8)
LCSA_WINDOW_TOKENS = LCSA_WIN[0] * LCSA_WIN[1] * LCSA_WIN[2]  # 128


# -------------------------------------------------------------------
# Window partitioning utilities
# -------------------------------------------------------------------


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B, F, H, W, C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]) -> torch.Tensor:
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(
        windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]
    ) -> torch.Tensor:
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


# -------------------------------------------------------------------
# LCSA mask generation (CPU-side, not compiled)
# -------------------------------------------------------------------


@torch.no_grad()
def build_local_block_mask(
    block_h: int,
    block_w: int,
    win_h: int = 11,
    win_w: int = 11,
    include_self: bool = True,
    device=None,
) -> torch.Tensor:
    """Build a local spatial block mask with sliding window.

    Returns a (block_h*block_w, block_h*block_w) boolean mask where
    mask[i, j] = True iff block j is within the (win_h x win_w) window
    centered on block i.
    """
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


# -------------------------------------------------------------------
# RoPE (real-valued cos/sin -- no complex numbers for Neuron)
# -------------------------------------------------------------------


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 3D factored RoPE frequencies as real-valued cos/sin pairs.

    Returns (f_cos, f_sin, h_cos, h_sin, w_cos, w_sin) -- each (end, dim_component/2).
    """
    f_dim = dim - 2 * (dim // 3)
    h_dim = dim // 3
    w_dim = dim // 3

    def _freqs(d):
        freqs = 1.0 / (
            theta ** (torch.arange(0, d, 2, dtype=torch.float64)[: d // 2] / d)
        )
        t = torch.arange(end, dtype=torch.float64)
        angles = torch.outer(t, freqs)
        return torch.cos(angles).float(), torch.sin(angles).float()

    f_cos, f_sin = _freqs(f_dim)
    h_cos, h_sin = _freqs(h_dim)
    w_cos, w_sin = _freqs(w_dim)
    return f_cos, f_sin, h_cos, h_sin, w_cos, w_sin


def build_rope_for_grid(
    f_cos,
    f_sin,
    h_cos,
    h_sin,
    w_cos,
    w_sin,
    f: int,
    h: int,
    w: int,
    temporal_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build full 3D RoPE cos/sin tensors for a (f, h, w) grid.

    Returns:
        rope_cos: (f*h*w, 1, head_dim/2) float32
        rope_sin: (f*h*w, 1, head_dim/2) float32
    """
    fc = (
        f_cos[temporal_offset : temporal_offset + f]
        .view(f, 1, 1, -1)
        .expand(f, h, w, -1)
    )
    fs = (
        f_sin[temporal_offset : temporal_offset + f]
        .view(f, 1, 1, -1)
        .expand(f, h, w, -1)
    )
    hc = h_cos[:h].view(1, h, 1, -1).expand(f, h, w, -1)
    hs = h_sin[:h].view(1, h, 1, -1).expand(f, h, w, -1)
    wc = w_cos[:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ws = w_sin[:w].view(1, 1, w, -1).expand(f, h, w, -1)

    rope_cos = torch.cat([fc, hc, wc], dim=-1).reshape(f * h * w, 1, -1)
    rope_sin = torch.cat([fs, hs, ws], dim=-1).reshape(f * h * w, 1, -1)
    return rope_cos, rope_sin


def apply_rope_real(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE using real-valued cos/sin (no complex numbers).

    x: (B, S, num_heads_per_rank * head_dim)
    rope_cos: (S, 1, head_dim/2) float32
    rope_sin: (S, 1, head_dim/2) float32
    """
    B, S, HD = x.shape
    half_head = rope_cos.shape[-1]
    num_heads_per_rank = HD // (half_head * 2)
    orig_dtype = x.dtype

    cos = rope_cos.squeeze(1).unsqueeze(0)
    sin = rope_sin.squeeze(1).unsqueeze(0)
    if num_heads_per_rank > 1:
        cos = cos.expand(B, S, half_head).repeat(1, 1, num_heads_per_rank)
        sin = sin.expand(B, S, half_head).repeat(1, 1, num_heads_per_rank)

    x = x.view(B, S, -1, 2)
    x_even = x[..., 0]
    x_odd = x[..., 1]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1)
    return out.view(B, S, HD).to(orig_dtype)


# -------------------------------------------------------------------
# TP helpers
# -------------------------------------------------------------------


def _get_tp_degree() -> int:
    if HAS_NXDI:
        try:
            return get_tensor_model_parallel_size()
        except Exception:
            return 1
    return 1


def make_column_parallel(
    in_f: int, out_f: int, bias: bool = True, gather_output: bool = False
) -> nn.Module:
    if HAS_NXDI and ColumnParallelLinear is not None:
        return ColumnParallelLinear(in_f, out_f, bias=bias, gather_output=gather_output)
    return nn.Linear(in_f, out_f, bias=bias)


def make_row_parallel(
    in_f: int, out_f: int, bias: bool = True, input_is_parallel: bool = True
) -> nn.Module:
    if HAS_NXDI and RowParallelLinear is not None:
        return RowParallelLinear(
            in_f, out_f, bias=bias, input_is_parallel=input_is_parallel
        )
    return nn.Linear(in_f, out_f, bias=bias)


# -------------------------------------------------------------------
# DistributedRMSNorm (all-reduce for global variance across TP ranks)
# -------------------------------------------------------------------


class DistributedRMSNorm(nn.Module):
    """RMSNorm with all-reduce for global variance computation across TP ranks.

    Standard RMSNorm on a TP-sharded hidden dimension only sees the local shard.
    This version computes sum-of-squares locally, all-reduces across ranks, then
    normalizes with the global RMS. Essential for QK-norm accuracy in TP>1.
    """

    def __init__(self, normalized_shape, eps=1e-5, tp_size=None, dtype=torch.bfloat16):
        super().__init__()
        if tp_size is None:
            tp_size = _get_tp_degree()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.eps = eps
        self.tp_size = tp_size
        self.local_dim = normalized_shape

        if HAS_NXDI:
            set_tensor_model_parallel_attributes(
                self.weight, is_parallel=True, dim=0, stride=1, num_partitions=tp_size
            )

    def forward(self, hidden_states):
        hidden_states_f32 = hidden_states.to(torch.float32)
        local_sum_sq = hidden_states_f32.pow(2).sum(dim=-1, keepdim=True)

        if self.tp_size > 1 and HAS_NXDI:
            import torch_xla.core.xla_model as xm

            global_sum_sq = xm.all_reduce(xm.REDUCE_SUM, local_sum_sq)
        else:
            global_sum_sq = local_sum_sq

        global_dim = self.local_dim * self.tp_size
        rms = torch.rsqrt(global_sum_sq / global_dim + self.eps)
        hidden_states = hidden_states_f32 * rms
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


# Register DistributedRMSNorm as a supported sharded module for NxD tracing.
if HAS_NXDI:
    _nxd_trace.__SUPPORTED_SHARDED_MODULES = (
        *_nxd_trace.__SUPPORTED_SHARDED_MODULES,
        DistributedRMSNorm,
    )


# -------------------------------------------------------------------
# NKI flash attention wrapper
# -------------------------------------------------------------------


def nki_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """NKI tiled flash attention -- never materializes the full S*S matrix.

    Input shapes (standard attention convention):
        q: [B, H, S_q, D]
        k: [B, H, S_kv, D]
        v: [B, H, S_kv, D]

    Returns: [B, H, S_q, D]
    """
    bs, n_head, q_len, d_head = q.shape
    kv_len = k.shape[2]

    # attention_cte layout: Q(BH, S_q, D), K(BH, D, S_kv), V(BH, S_kv, D)
    q_cte = q.reshape(bs * n_head, q_len, d_head)
    k_cte = k.permute(0, 1, 3, 2).reshape(bs * n_head, d_head, kv_len)
    v_cte = v.reshape(bs * n_head, kv_len, d_head)

    scale = 1.0 / math.sqrt(d_head)
    attn_output = _nkilib_attention_cte(
        q=q_cte,
        k=k_cte,
        v=v_cte,
        scale=scale,
        causal_mask=False,
    )

    return attn_output.reshape(bs, n_head, q_len, d_head)


# -------------------------------------------------------------------
# Self-attention (LCSA-capable)
# -------------------------------------------------------------------


class NeuronFlashVSRSelfAttention(nn.Module):
    """Traceable self-attention for NxDI compilation.

    Phase 1 (production): Dense NKI flash attention on full sequence.
    Phase 2 (USE_BLOCK_SPARSE_LCSA=True): Per-layer block-sparse LCSA.

    TP-sharded Q/K/V/O projections with DistributedRMSNorm for QK-norm.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = EPS):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        tp = _get_tp_degree()
        padded_heads = math.ceil(num_heads / tp) * tp
        self.padded_inner_dim = padded_heads * self.head_dim
        self.num_heads_per_rank = padded_heads // tp

        self.to_q = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_k = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_v = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_out = make_row_parallel(
            self.padded_inner_dim, dim, bias=True, input_is_parallel=True
        )

        shard_dim = self.num_heads_per_rank * self.head_dim
        self.norm_q = DistributedRMSNorm(shard_dim, eps=eps, tp_size=tp)
        self.norm_k = DistributedRMSNorm(shard_dim, eps=eps, tp_size=tp)

        # Phase 2: LCSA local mask (initialized via set_local_mask before compilation)
        self.local_attn_mask = None
        self.lcsa_max_active = LCSA_MAX_ACTIVE

    def set_local_mask(self, h: int, w: int):
        """Pre-compute and register the LCSA local spatial mask as a buffer.

        Must be called before Neuron compilation when USE_BLOCK_SPARSE_LCSA=True.
        """
        block_h = h // LCSA_WIN[1]
        block_w = w // LCSA_WIN[2]
        local_mask = build_local_block_mask(
            block_h,
            block_w,
            LCSA_LOCAL_RANGE,
            LCSA_LOCAL_RANGE,
            include_self=True,
        )
        self.register_buffer("_local_attn_mask", local_mask)
        self.local_attn_mask = local_mask

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        f: int,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, f*h*w, dim)
            rope_cos: (f*h*w, 1, head_dim/2) float32
            rope_sin: (f*h*w, 1, head_dim/2) float32
            attn_mask: (B, H, num_q_blocks, num_kv_blocks) -- unused in Phase 1
            f, h, w: post-patchify grid dimensions

        Returns:
            output: (B, f*h*w, dim)
            cache_k: windowed K for cache management
            cache_v: windowed V for cache management
        """
        B, L, D = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = apply_rope_real(q, rope_cos, rope_sin)
        k = apply_rope_real(k, rope_cos, rope_sin)

        n = self.num_heads_per_rank
        D_rank = q.shape[-1]
        d = self.head_dim

        if USE_BLOCK_SPARSE_LCSA and self.local_attn_mask is not None:
            # Phase 2: Block-sparse LCSA attention (requires TP=16, trn2.48xlarge)
            x_out, cache_k, cache_v = self._forward_block_sparse(
                q,
                k,
                v,
                f,
                h,
                w,
                B,
                n,
                d,
                D_rank,
            )
        else:
            # Phase 1 (production): Dense NKI flash attention
            q_4d = rearrange(q, "b s (n d) -> b n s d", n=n)
            k_4d = rearrange(k, "b s (n d) -> b n s d", n=n)
            v_4d = rearrange(v, "b s (n d) -> b n s d", n=n)

            if HAS_NKI_FLASH and q_4d.device.type == "xla":
                x_out = nki_flash_attention(q_4d, k_4d, v_4d)
            else:
                x_out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d)
            x_out = rearrange(x_out, "b n s d -> b s (n d)", n=n)

            # Cache output: windowed K/V for potential future use
            k_5d = k.view(B, f, h, w, D_rank)
            v_5d = v.view(B, f, h, w, D_rank)
            cache_k = WindowPartition3D.partition(k_5d, LCSA_WIN)
            cache_v = WindowPartition3D.partition(v_5d, LCSA_WIN)

        return self.to_out(x_out), cache_k, cache_v

    def _forward_block_sparse(
        self,
        q,
        k,
        v,
        f,
        h,
        w,
        B,
        n,
        d,
        D_rank,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Block-sparse LCSA attention.

        All ops are traceable on Neuron (topk, index_select, gather, SDPA).
        Requires trn2.48xlarge with TP=16 for sufficient HBM.
        """
        # Window partition
        q_5d = q.view(B, f, h, w, D_rank)
        k_5d = k.view(B, f, h, w, D_rank)
        v_5d = v.view(B, f, h, w, D_rank)

        q_w = WindowPartition3D.partition(q_5d, LCSA_WIN)
        k_w = WindowPartition3D.partition(k_5d, LCSA_WIN)
        v_w = WindowPartition3D.partition(v_5d, LCSA_WIN)

        num_q_blocks = q_w.shape[0]
        num_kv_blocks = k_w.shape[0]

        # Cache output
        cache_k = k_w
        cache_v = v_w

        # Generate block mask from this layer's Q/K
        seqlen = f // LCSA_WIN[0]
        block_h = h // LCSA_WIN[1]
        block_w = w // LCSA_WIN[2]
        spatial_blocks = block_h * block_w
        square_num = spatial_blocks * spatial_blocks

        avgpool_q = torch.mean(q_w, dim=1).float()
        avgpool_k = torch.mean(k_w, dim=1).float()

        avgpool_q = rearrange(avgpool_q, "s (h d) -> h s d", h=n)
        avgpool_k = rearrange(avgpool_k, "s (h d) -> h s d", h=n)

        scores = torch.bmm(avgpool_q, avgpool_k.transpose(1, 2))
        scores = scores / math.sqrt(d)

        # Apply local spatial constraint
        local_mask = self._local_attn_mask
        repeat_len_q = num_q_blocks // local_mask.shape[0]
        repeat_len_kv = num_kv_blocks // local_mask.shape[1]
        local_expanded = (
            local_mask.unsqueeze(1)
            .unsqueeze(0)
            .repeat(repeat_len_q, 1, repeat_len_kv, 1)
        )
        local_expanded = local_expanded.reshape(
            repeat_len_q * local_mask.shape[0],
            repeat_len_kv * local_mask.shape[1],
        )
        local_expanded = local_expanded.unsqueeze(0).expand(n, -1, -1)

        local_float = torch.where(
            local_expanded,
            torch.zeros(1, dtype=scores.dtype, device=scores.device),
            torch.tensor(-1e9, dtype=scores.dtype, device=scores.device),
        )
        scores = scores + local_float

        # Softmax + topk selection
        attn_map = torch.softmax(scores, dim=-1)
        attn_map_r = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
        loop_num, s1, s2 = attn_map_r.shape
        flat = attn_map_r.reshape(loop_num, -1)
        topk_k = min(flat.shape[1] - 1, max(int(square_num * LCSA_TOPK_RATIO), 1))
        topk_vals, _ = torch.topk(flat, k=topk_k + 1, dim=1, largest=True)
        thresholds = topk_vals[:, -1:]
        block_mask = (flat > thresholds).reshape(loop_num, s1, s2)
        block_mask = rearrange(block_mask, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)

        # Get kv_indices via topk on mask
        max_active = self.lcsa_max_active
        mask_float = block_mask.float()
        _, kv_indices = torch.topk(mask_float, k=max_active, dim=-1, largest=True)

        # Validity mask
        mask_2d = mask_float.reshape(-1, mask_float.shape[-1])
        idx_2d = kv_indices.reshape(-1, max_active)
        validity = torch.gather(mask_2d, 1, idx_2d)

        # Gather K/V
        flat_idx = kv_indices.reshape(-1)
        k_gathered = torch.index_select(k_w, 0, flat_idx)
        v_gathered = torch.index_select(v_w, 0, flat_idx)
        k_gathered = k_gathered.reshape(
            n, num_q_blocks, max_active, LCSA_WINDOW_TOKENS, D_rank
        )
        v_gathered = v_gathered.reshape(
            n, num_q_blocks, max_active, LCSA_WINDOW_TOKENS, D_rank
        )

        # Zero out padding blocks
        valid_mask = validity.to(k_gathered.dtype).reshape(
            n, num_q_blocks, max_active, 1, 1
        )
        k_gathered = (k_gathered * valid_mask).to(q.dtype)
        v_gathered = (v_gathered * valid_mask).to(q.dtype)

        # Chunked SDPA per head
        q_heads = rearrange(q_w, "nq p (h d) -> h nq p d", h=n)
        kv_len = max_active * LCSA_WINDOW_TOKENS
        num_chunks = num_q_blocks // LCSA_CTE_CHUNK_SIZE
        cs = LCSA_CTE_CHUNK_SIZE

        output_heads = []
        for head_i in range(n):
            hd_s = head_i * d
            hd_e = (head_i + 1) * d

            q_h = q_heads[head_i]
            k_h = k_gathered[head_i, :, :, :, hd_s:hd_e]
            v_h = v_gathered[head_i, :, :, :, hd_s:hd_e]

            k_flat = k_h.reshape(num_q_blocks, kv_len, d)
            v_flat = v_h.reshape(num_q_blocks, kv_len, d)

            q_chunks = torch.split(q_h, cs, dim=0)
            k_chunks = torch.split(k_flat, cs, dim=0)
            v_chunks = torch.split(v_flat, cs, dim=0)

            chunk_outputs = []
            for ci in range(num_chunks):
                q_sdpa = q_chunks[ci].unsqueeze(0)
                k_sdpa = k_chunks[ci].unsqueeze(0)
                v_sdpa = v_chunks[ci].unsqueeze(0)
                out_chunk = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
                chunk_outputs.append(out_chunk.squeeze(0))

            out_h = torch.cat(chunk_outputs, dim=0)
            output_heads.append(out_h)

        x_out = torch.cat(output_heads, dim=-1)
        x_out = WindowPartition3D.reverse(x_out, LCSA_WIN, (f, h, w))
        x_out = x_out.view(B, f * h * w, D_rank)

        return x_out, cache_k, cache_v


# -------------------------------------------------------------------
# Cross-attention (dense, text conditioning)
# -------------------------------------------------------------------


class NeuronFlashVSRCrossAttention(nn.Module):
    """Dense cross-attention for text conditioning.

    Cross-KV is computed INSIDE the compiled model from encoder_hidden_states.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = EPS):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        tp = _get_tp_degree()
        padded_heads = math.ceil(num_heads / tp) * tp
        self.padded_inner_dim = padded_heads * (dim // num_heads)
        self.num_heads_per_rank = padded_heads // tp

        self.to_q = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_k = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_v = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_out = make_row_parallel(
            self.padded_inner_dim, dim, bias=True, input_is_parallel=True
        )

        head_dim = dim // num_heads
        shard_dim = self.num_heads_per_rank * head_dim
        self.norm_q = DistributedRMSNorm(shard_dim, eps=eps, tp_size=tp)
        self.norm_k = DistributedRMSNorm(shard_dim, eps=eps, tp_size=tp)

    def forward(
        self, x: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = self.norm_q(q)
        k = self.norm_k(k)

        n = self.num_heads_per_rank
        q = rearrange(q, "b s (n d) -> b n s d", n=n)
        k = rearrange(k, "b s (n d) -> b n s d", n=n)
        v = rearrange(v, "b s (n d) -> b n s d", n=n)

        if HAS_NKI_FLASH and q.device.type == "xla" and not USE_BLOCK_SPARSE_LCSA:
            out = nki_flash_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n s d -> b s (n d)", n=n)
        return self.to_out(out)


# -------------------------------------------------------------------
# DiT Block
# -------------------------------------------------------------------


class NeuronFlashVSRBlock(nn.Module):
    """Single DiT block: AdaLN -> SelfAttn(LCSA) -> Gate -> CrossAttn -> AdaLN -> FFN -> Gate"""

    def __init__(
        self,
        dim: int = DIM,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = FFN_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.dim = dim

        self.self_attn = NeuronFlashVSRSelfAttention(dim, num_heads, eps)
        self.cross_attn = NeuronFlashVSRCrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

        # FFN (not TP-sharded)
        self.ffn_gelu_proj = nn.Linear(dim, ffn_dim, bias=True)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=True)

        # AdaLN modulation (6 vectors: shift/scale/gate x2)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        f: int,
        h: int,
        w: int,
        lq_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LQ conditioning residual
        if lq_residual is not None:
            x = x + lq_residual

        # AdaLN modulation
        mod = self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )

        # Self-attention
        normed = self.norm1(x) * (1 + scale_msa) + shift_msa
        sa_out, cache_k, cache_v = self.self_attn(
            normed, rope_cos, rope_sin, attn_mask, f, h, w
        )
        x = x + gate_msa * sa_out

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), encoder_hidden_states)

        # FFN
        normed = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        ffn_out = self.ffn_out(F.gelu(self.ffn_gelu_proj(normed), approximate="tanh"))
        x = x + gate_mlp * ffn_out

        return x, cache_k, cache_v


# -------------------------------------------------------------------
# Condition embedder (time + text)
# -------------------------------------------------------------------


class NeuronConditionEmbedder(nn.Module):
    """Time and text condition embedder for AdaLN modulation."""

    def __init__(
        self,
        dim: int = DIM,
        text_dim: int = TEXT_DIM,
        freq_dim: int = FREQ_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim

        self.time_embedder_linear_1 = nn.Linear(freq_dim, dim)
        self.time_embedder_act = nn.SiLU()
        self.time_embedder_linear_2 = nn.Linear(dim, dim)

        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)

        self.text_embedder_linear_1 = nn.Linear(text_dim, dim)
        self.text_embedder_act = nn.GELU(approximate="tanh")
        self.text_embedder_linear_2 = nn.Linear(dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sinusoidal time embedding
        position = timestep.to(torch.float64)
        sinusoid = torch.outer(
            position,
            torch.pow(
                10000,
                -torch.arange(
                    self.freq_dim // 2, dtype=torch.float64, device=timestep.device
                ).div(self.freq_dim // 2),
            ),
        )
        sin_emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        sin_emb = sin_emb.to(timestep.dtype)

        t = self.time_embedder_linear_2(
            self.time_embedder_act(self.time_embedder_linear_1(sin_emb))
        )
        t_mod = self.time_proj(self.act_fn(t)).unflatten(1, (6, self.dim))

        ctx = self.text_embedder_linear_2(
            self.text_embedder_act(self.text_embedder_linear_1(encoder_hidden_states))
        )

        t_emb = t.unsqueeze(1)
        return t_emb, t_mod, ctx


# -------------------------------------------------------------------
# Full FlashVSR DiT Transformer
# -------------------------------------------------------------------


class FlashVSRDiTConfig:
    """Configuration for FlashVSR DiT compilation."""

    def __init__(
        self,
        height: int = 768,
        width: int = 1280,
        num_latent_frames_first: int = 6,
        num_latent_frames_stream: int = 2,
        hidden_size: int = DIM,
        intermediate_size: int = FFN_DIM,
        num_attention_heads: int = NUM_HEADS,
        attention_head_dim: int = HEAD_DIM,
        num_hidden_layers: int = NUM_LAYERS,
        in_channels: int = IN_CHANNELS,
        tp_degree: int = 4,
    ):
        self.height = height
        self.width = width
        self.num_latent_frames_first = num_latent_frames_first
        self.num_latent_frames_stream = num_latent_frames_stream
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_hidden_layers = num_hidden_layers
        self.in_channels = in_channels
        self.tp_degree = tp_degree


class NeuronFlashVSRDiT(nn.Module):
    """Full 30-layer FlashVSR DiT transformer for Neuron compilation.

    Compiled via NxDI ModelBuilder. Takes:
    - 5D latent video tensor (patchified inside)
    - Timestep scalar
    - Text encoder hidden states
    - Precomputed RoPE cos/sin (float32)
    - Attention mask placeholder (block-level)
    - LQ conditioning residual for block 0

    Returns output video + per-layer KV caches.
    """

    def __init__(
        self,
        config=None,
        dim: int = DIM,
        ffn_dim: int = FFN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        patch_size: Tuple[int, int, int] = (PATCH_T, PATCH_H, PATCH_W),
        in_dim: int = IN_CHANNELS,
        out_dim: int = OUT_CHANNELS,
        text_dim: int = TEXT_DIM,
        freq_dim: int = FREQ_DIM,
        eps: float = EPS,
    ):
        super().__init__()

        if config is not None and hasattr(config, "hidden_size"):
            dim = getattr(config, "hidden_size", DIM)
            ffn_dim = getattr(config, "intermediate_size", FFN_DIM)
            num_heads = getattr(config, "num_attention_heads", NUM_HEADS)
            num_layers = getattr(config, "num_hidden_layers", NUM_LAYERS)
            in_dim = getattr(config, "in_channels", IN_CHANNELS)

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size

        self.condition_embedder = NeuronConditionEmbedder(dim, text_dim, freq_dim, eps)

        self.patch_embedding = nn.Conv3d(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                NeuronFlashVSRBlock(dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        # Auto-initialize LCSA masks when Phase 2 is enabled
        if USE_BLOCK_SPARSE_LCSA and config is not None:
            post_h = config.height // (8 * PATCH_H)
            post_w = config.width // (8 * PATCH_W)
            self.init_lcsa_masks(post_h, post_w)

    def init_lcsa_masks(self, h: int, w: int):
        """Initialize LCSA local masks on all self-attention layers (Phase 2 only)."""
        for block in self.blocks:
            block.self_attn.set_local_mask(h, w)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        lq_residual_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            hidden_states: (B, C_in, F, H_lat, W_lat) 5D latent video
            timestep: (B,) diffusion timestep
            encoder_hidden_states: (B, S_ctx, text_dim) text embeddings
            rope_cos: (L, 1, head_dim/2) float32
            rope_sin: (L, 1, head_dim/2) float32
            attn_mask: (B, H, NQ_blocks, NKV_blocks) block-level mask placeholder
            lq_residual_0: (B, S, dim) LQ conditioning for block 0

        Returns:
            Tuple: (output, cache_k_0, cache_v_0, ..., cache_k_29, cache_v_29)
        """
        B = hidden_states.shape[0]

        t_emb, t_mod, ctx_embedded = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        # Patchify: (B, C, F, H, W) -> (B, L, dim)
        x = self.patch_embedding(hidden_states)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        # Transformer blocks
        new_caches_k = []
        new_caches_v = []

        for i, block in enumerate(self.blocks):
            lq_res_i = lq_residual_0 if i == 0 else None
            x, ck, cv = block(
                x,
                t_mod,
                rope_cos,
                rope_sin,
                attn_mask,
                ctx_embedded,
                f,
                h,
                w,
                lq_residual=lq_res_i,
            )
            new_caches_k.append(ck)
            new_caches_v.append(cv)

        # Output head
        shift, scale = (
            self.scale_shift_table.to(dtype=t_emb.dtype, device=t_emb.device) + t_emb
        ).chunk(2, dim=1)
        x = self.proj_out(self.norm_out(x) * (1 + scale) + shift)

        # Unpatchify
        px, py, pz = self.patch_size
        x = rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=px,
            y=py,
            z=pz,
        )

        results = [x]
        for i in range(self.num_layers):
            results.append(new_caches_k[i])
            results.append(new_caches_v[i])
        return tuple(results)


# -------------------------------------------------------------------
# NxDI Application / ModelWrapper classes
# -------------------------------------------------------------------

try:
    from neuronx_distributed_inference.models.application_base import (
        NeuronApplicationBase,
    )
    from neuronx_distributed_inference.models.model_wrapper import (
        ModelWrapper,
        EncoderModelInstance,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig,
        NeuronConfig,
    )

    HAS_NXDI_INFERENCE = True
except ImportError:
    HAS_NXDI_INFERENCE = False


if HAS_NXDI_INFERENCE:

    class FlashVSRInferenceConfig(InferenceConfig):
        """Configuration for FlashVSR NxDI compilation."""

        def __init__(self, *args, **kwargs):
            self.attn_mode = kwargs.pop("attn_mode", "first")
            self.height = kwargs.pop("height", 768)
            self.width = kwargs.pop("width", 1280)
            self.num_latent_frames_first = kwargs.pop("num_latent_frames_first", 6)
            self.num_latent_frames_stream = kwargs.pop("num_latent_frames_stream", 2)
            self.hidden_size = kwargs.pop("hidden_size", DIM)
            self.intermediate_size = kwargs.pop("intermediate_size", FFN_DIM)
            self.num_attention_heads = kwargs.pop("num_attention_heads", NUM_HEADS)
            self.attention_head_dim = kwargs.pop("attention_head_dim", HEAD_DIM)
            self.num_hidden_layers = kwargs.pop("num_hidden_layers", NUM_LAYERS)
            self.in_channels = kwargs.pop("in_channels", IN_CHANNELS)
            super().__init__(*args, **kwargs)

        def get_required_attributes(self):
            return []

        def load_config(self):
            pass

    FIRST_FRAME_COUNTS = [6]
    STREAM_FRAME_COUNTS = [2]
    ALL_FRAME_COUNTS = [6, 2]

    class FlashVSRModelWrapper(ModelWrapper):
        """NxDI ModelWrapper for FlashVSR compilation."""

        def __init__(
            self,
            config,
            model_cls,
            tag="",
            compiler_args=None,
            priority_model_idx=None,
            model_init_kwargs=None,
        ):
            super().__init__(
                config,
                model_cls,
                tag,
                compiler_args,
                priority_model_idx=priority_model_idx,
                model_init_kwargs=model_init_kwargs,
            )
            self.mode = config.attn_mode
            if self.mode == "first":
                self.frame_counts = FIRST_FRAME_COUNTS
            elif self.mode == "stream":
                self.frame_counts = STREAM_FRAME_COUNTS
            else:
                self.frame_counts = ALL_FRAME_COUNTS

            self.base_freqs = precompute_freqs_cis_3d(HEAD_DIM)

        def get_model_instance(self):
            return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

        def input_generator(self):
            """Generate example inputs for each compilation bucket."""
            config = self.config
            lat_h = config.height // 8
            lat_w = config.width // 8
            batch = 1

            results = []
            for num_frames in self.frame_counts:
                post_f = num_frames // PATCH_T
                post_h = lat_h // PATCH_H
                post_w = lat_w // PATCH_W
                seq_len = post_f * post_h * post_w

                hidden_states = torch.randn(
                    batch, IN_CHANNELS, num_frames, lat_h, lat_w, dtype=torch.bfloat16
                )
                timestep = torch.tensor([1000.0], dtype=torch.bfloat16)
                encoder_hidden_states = torch.randn(
                    batch, 512, TEXT_DIM, dtype=torch.bfloat16
                )

                rope_cos, rope_sin = build_rope_for_grid(
                    *self.base_freqs, post_f, post_h, post_w
                )

                num_q_blocks = (
                    (post_f // LCSA_WIN[0])
                    * (post_h // LCSA_WIN[1])
                    * (post_w // LCSA_WIN[2])
                )
                num_kv_blocks = num_q_blocks
                attn_mask = torch.zeros(
                    batch, NUM_HEADS, num_q_blocks, num_kv_blocks, dtype=torch.bfloat16
                )

                lq_residual_0 = torch.zeros(batch, seq_len, DIM, dtype=torch.bfloat16)

                inputs = [
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    rope_cos,
                    rope_sin,
                    attn_mask,
                    lq_residual_0,
                ]
                results.append(tuple(inputs))

            return results

        def forward(self, *args, **kwargs):
            return self._forward(*args)

    # Compiler flags for production configuration
    FLASHVSR_COMPILER_ARGS = (
        "--auto-cast=none --model-type=transformer -O1 "
        "--internal-max-instruction-limit=15000000 "
        "--tensorizer-options='--enable-ccop-compute-overlap'"
    )

    class FlashVSRApplication(NeuronApplicationBase):
        """NxDI Application for FlashVSR DiT."""

        _model_cls = NeuronFlashVSRDiT

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_wrapper_cls = FlashVSRModelWrapper
            tag = f"FlashVSR_{self.config.attn_mode}"
            self.model = self.model_wrapper_cls(
                config=self.config,
                model_cls=self._model_cls,
                tag=tag,
                compiler_args=FLASHVSR_COMPILER_ARGS,
                priority_model_idx=0,
            )
            self.models.append(self.model)

        def forward(self, *model_inputs, **kwargs):
            return self.models[0](*model_inputs, **kwargs)

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, config):
            """Convert DiffSynth/HuggingFace state dict to NxDI naming."""
            from .weights import detect_format_and_convert

            tp_degree = getattr(config, "neuron_config", None)
            if tp_degree is not None:
                tp_degree = getattr(tp_degree, "tp_degree", 1)
            else:
                tp_degree = 1
            return detect_format_and_convert(state_dict, tp_degree=tp_degree)

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            return state_dict
