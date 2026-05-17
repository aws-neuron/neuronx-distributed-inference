"""
Sliding Tile Attention (STA) for NeuronCore — trace-compatible.

Provides an nn.Module that pre-computes all STA block mask groups and
gather/scatter indices at __init__ time, so the forward() method is
a static computation graph compatible with ModelBuilder.trace().

Usage:
    sta = STAAttention(
        canvas_thw=(2, 30, 53),   # latent T, H, W (before padding)
        tile_thw=(6, 8, 8),       # tile dimensions
        kernel_thw=(3, 3, 3),     # 3D neighborhood kernel
        text_len=320,             # text token count
        num_heads=4,              # local heads per TP rank
        head_dim=128,
    )

    # q, k, v: [B, local_heads, seq_len, head_dim] in BF16
    output = sta(q, k, v)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from functools import lru_cache
from typing import Tuple, List, Dict, Optional

# NKI attention_cte kernel — imported at module level for tracing.
# IMPORTANT: attention_cte is an nki.jit Kernel object. Call it DIRECTLY
# (not via nki_jit() wrapper from torch_neuronx). The Kernel.__call__
# detects torch tensors and dispatches through PyTorchXLAKernel with
# proper NKI backend context setup, which is required for NKI 0.3 GA
# operations like nl.ndarray(buffer=nl.shared_hbm).
# Using the deprecated nki_jit() wrapper causes "No backend set" errors.
try:
    from nkilib.core.attention.attention_cte import attention_cte
except ImportError:
    try:
        from neuronxcc.nki.kernels.attention import attention_cte
    except ImportError:
        attention_cte = None  # Will fail at forward() time if used


# ─── Static mask computation (runs at init time, not traced) ────────────────


@lru_cache(maxsize=64)
def create_sta_mask(canvas_thw, tile_thw, kernel_thw):
    """
    Create block-level STA mask with boundary clamping.

    Returns:
        block_mask: [N_blocks, N_blocks] boolean numpy array
    """
    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    kernel_t, kernel_h, kernel_w = kernel_thw

    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w
    N = n_t * n_h * n_w

    i_indices = np.arange(N)
    j_indices = np.arange(N)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="ij")

    q_t = i_grid // (n_h * n_w)
    q_h = (i_grid % (n_h * n_w)) // n_w
    q_w = i_grid % n_w

    kv_t = j_grid // (n_h * n_w)
    kv_h = (j_grid % (n_h * n_w)) // n_w
    kv_w = j_grid % n_w

    center_t = np.clip(q_t, kernel_t // 2, (n_t - 1) - kernel_t // 2)
    center_h = np.clip(q_h, kernel_h // 2, (n_h - 1) - kernel_h // 2)
    center_w = np.clip(q_w, kernel_w // 2, (n_w - 1) - kernel_w // 2)

    mask = (
        (np.abs(center_t - kv_t) <= kernel_t // 2)
        & (np.abs(center_h - kv_h) <= kernel_h // 2)
        & (np.abs(center_w - kv_w) <= kernel_w // 2)
    )
    return mask


def build_groups(block_mask, text_block_num=0):
    """
    Group query blocks by n_allowed (number of attended KV blocks).

    Returns:
        groups: dict n_allowed → {
            'block_indices': list of query block indices,
            'kv_indices': list of lists of KV block indices,
        }
        total_blocks: N_blocks + text_block_num
    """
    if isinstance(block_mask, np.ndarray):
        block_mask = torch.from_numpy(block_mask)

    N_img = block_mask.shape[0]
    total_blocks = N_img + text_block_num

    if text_block_num > 0:
        full_mask = torch.zeros(total_blocks, total_blocks, dtype=torch.bool)
        full_mask[:N_img, :N_img] = block_mask
        full_mask[:, N_img:] = True
        full_mask[N_img:, :] = True
    else:
        full_mask = block_mask

    n_allowed = full_mask.sum(dim=1)

    groups = {}
    for i in range(total_blocks):
        n = n_allowed[i].item()
        if n not in groups:
            groups[n] = {"block_indices": [], "kv_indices": []}
        groups[n]["block_indices"].append(i)
        allowed = full_mask[i].nonzero(as_tuple=True)[0].tolist()
        groups[n]["kv_indices"].append(allowed)

    return groups, total_blocks


def compute_padded_canvas(canvas_thw, tile_thw):
    """Compute padded canvas dimensions (divisible by tile dims)."""
    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    pad_t = (tile_t - t % tile_t) % tile_t
    pad_h = (tile_h - h % tile_h) % tile_h
    pad_w = (tile_w - w % tile_w) % tile_w
    return (t + pad_t, h + pad_h, w + pad_w), (pad_t, pad_h, pad_w)


# ─── Tiling ops (must be traceable — no einops, pure torch) ────────────────


def tile_tokens(x, canvas_thw, tile_thw):
    """
    Rearrange [B, H, T*Ht*W, D] from raster order into tile order.

    Uses pure torch ops (no einops) for traceability.
    Output: [B, H, N_tiles * tile_size, D]
    """
    b, h, s, d = x.shape
    t, ht, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    n_t = t // tile_t
    n_h = ht // tile_h
    n_w = w // tile_w

    # Reshape to spatial: [B, H, T, Ht, W, D]
    x = x.reshape(b, h, t, ht, w, d)

    # Reshape to tile grid: [B, H, n_t, tile_t, n_h, tile_h, n_w, tile_w, D]
    x = x.reshape(b, h, n_t, tile_t, n_h, tile_h, n_w, tile_w, d)

    # Permute to: [B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D]
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)

    # Flatten tiles: [B, H, N_tiles * tile_size, D]
    x = x.reshape(b, h, -1, d)

    return x


def untile_tokens(x, canvas_thw, tile_thw):
    """
    Reverse tile_tokens: [B, H, N_tiles * tile_size, D] → [B, H, T*Ht*W, D]
    """
    t, ht, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    n_t = t // tile_t
    n_h = ht // tile_h
    n_w = w // tile_w
    b, h, _, d = x.shape

    # Unflatten: [B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D]
    x = x.reshape(b, h, n_t, n_h, n_w, tile_t, tile_h, tile_w, d)

    # Permute back: [B, H, n_t, tile_t, n_h, tile_h, n_w, tile_w, D]
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)

    # Flatten spatial: [B, H, T*Ht*W, D]
    x = x.reshape(b, h, t * ht * w, d)

    return x


# ─── STA Attention Module ──────────────────────────────────────────────────


class STAAttention(nn.Module):
    """
    Sliding Tile Attention for NeuronCore — trace-compatible.

    Pre-computes all gather/scatter indices at __init__ time.
    The forward() is a static graph: tile → gather → attention_cte → untile.

    Key optimization (scatter-free): With boundary clamping, all image blocks
    have the same n_allowed. This means:
      - Q for image blocks is identity (blocks 0..N-1 in tile order) — no gather
      - Output is in tile order — no scatter needed
      - Only KV needs per-block gathering (different neighborhoods)
    Text blocks are processed separately and concatenated.

    This eliminates the global scatter_ operation that caused NCC_IBIR229
    SBUF overflow at 129-frame scale (64K+ tokens).
    """

    def __init__(
        self,
        canvas_thw: Tuple[int, int, int],
        tile_thw: Tuple[int, int, int] = (6, 8, 8),
        kernel_thw: Tuple[int, int, int] = (3, 3, 3),
        text_len: int = 320,
        num_heads: int = 4,
        head_dim: int = 128,
        max_batch_per_cte: int = 504,
    ):
        super().__init__()

        self.canvas_thw = canvas_thw
        self.tile_thw = tile_thw
        self.kernel_thw = kernel_thw
        self.text_len = text_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        tile_t, tile_h, tile_w = tile_thw
        self.block_size = tile_t * tile_h * tile_w
        img_len = canvas_thw[0] * canvas_thw[1] * canvas_thw[2]

        # Compute padded canvas
        self.padded_canvas, self.padding = compute_padded_canvas(canvas_thw, tile_thw)
        pad_t, pad_h, pad_w = self.padding
        pt, ph, pw = self.padded_canvas
        self.padded_img_len = pt * ph * pw

        # Text block padding
        self.text_block_num = (
            math.ceil(text_len / self.block_size) if text_len > 0 else 0
        )
        self.text_target = self.text_block_num * self.block_size
        self.text_pad = self.text_target - text_len

        # Total tokens after padding + tiling
        self.n_img_blocks = (self.padded_img_len) // self.block_size
        self.total_blocks = self.n_img_blocks + self.text_block_num
        self.total_tokens = self.total_blocks * self.block_size

        # Build STA mask and groups
        sta_mask = create_sta_mask(self.padded_canvas, tile_thw, kernel_thw)
        groups, total_blocks = build_groups(sta_mask, self.text_block_num)
        assert total_blocks == self.total_blocks

        # ─── Scatter-free optimization ────────────────────────────────
        # With boundary clamping in create_sta_mask, all image blocks have
        # the same n_allowed (kernel volume + text blocks). We exploit this:
        #   - Image blocks are processed as one contiguous group in tile order
        #   - Q is identity (no gather), output is identity (no scatter)
        #   - Only KV needs per-block gathering
        #   - Text block(s) use full attention and are handled separately
        #
        # This eliminates the global scatter_ that caused SBUF overflow.

        # Verify all image blocks have same n_allowed
        img_n_vals = set()
        for n_val, group_info in groups.items():
            for bi in group_info["block_indices"]:
                if bi < self.n_img_blocks:
                    img_n_vals.add(n_val)
        assert len(img_n_vals) == 1, (
            f"Expected all image blocks to have same n_allowed, got {img_n_vals}. "
            f"This can happen if the block grid is smaller than the kernel on some axis."
        )
        self._img_n_allowed = img_n_vals.pop()
        self._img_kv_len = self._img_n_allowed * self.block_size

        # Pre-compute KV gather indices for image blocks.
        # Each image block has a different set of KV blocks (different neighborhood),
        # but all have the same count (n_allowed).
        # We build a flat index: [n_img_blocks * kv_len] into the tiled token sequence.
        #
        # IMPORTANT: stored as Python list (not buffer) to avoid NKI dispatch issue.
        img_group = groups[self._img_n_allowed]
        img_block_indices = [
            bi for bi in img_group["block_indices"] if bi < self.n_img_blocks
        ]
        img_kv_lists = [
            kv
            for bi, kv in zip(img_group["block_indices"], img_group["kv_indices"])
            if bi < self.n_img_blocks
        ]

        # Sort by block index to ensure tile-order alignment
        sorted_pairs = sorted(zip(img_block_indices, img_kv_lists))
        img_block_indices = [p[0] for p in sorted_pairs]
        img_kv_lists = [p[1] for p in sorted_pairs]

        assert img_block_indices == list(range(self.n_img_blocks)), (
            "Image blocks must be 0..N-1 in order"
        )

        # Chunking for image blocks — KV gather indices stored PER-CHUNK
        # to avoid materializing the full 168*10752=1.8M element gather tensor
        # which exceeds SBUF at 129-frame scale.
        chunk_blocks = max_batch_per_cte // num_heads
        n_chunks = math.ceil(self.n_img_blocks / chunk_blocks)
        self._img_chunk_specs = []
        for c in range(n_chunks):
            c_start = c * chunk_blocks
            c_end = min(c_start + chunk_blocks, self.n_img_blocks)
            # Per-chunk KV gather: only indices for blocks [c_start, c_end)
            chunk_kv_gather = []
            for bi in range(c_start, c_end):
                for kj in img_kv_lists[bi]:
                    start = kj * self.block_size
                    chunk_kv_gather.extend(range(start, start + self.block_size))
            self._img_chunk_specs.append(
                {
                    "start_block": c_start,
                    "end_block": c_end,
                    "n_blocks": c_end - c_start,
                    "batch_size": (c_end - c_start) * num_heads,
                    "kv_gather_list": chunk_kv_gather,
                }
            )

        # Text block handling: text block attends to everything (n_allowed = total_blocks)
        # For the text block, we need to gather ALL tokens as KV.
        # Since it's just 1 block attending to all, no special handling needed.
        self._text_n_allowed = self.total_blocks  # full attention for text
        self._text_kv_len = self.total_blocks * self.block_size

        # Store group specs for compatibility with diagnostic printing
        self._group_specs = []
        self._group_specs.append(
            {
                "group_idx": 0,
                "n_val": self._img_n_allowed,
                "group_size": self.n_img_blocks,
                "kv_len": self._img_kv_len,
                "chunk_specs": self._img_chunk_specs,
            }
        )
        if self.text_block_num > 0:
            self._group_specs.append(
                {
                    "group_idx": 1,
                    "n_val": self._text_n_allowed,
                    "group_size": self.text_block_num,
                    "kv_len": self._text_kv_len,
                    "chunk_specs": [
                        {
                            "start_block": 0,
                            "end_block": self.text_block_num,
                            "batch_size": self.text_block_num * num_heads,
                        }
                    ],
                }
            )
        self.n_groups = len(self._group_specs)

        # Store seq_len for forward validation
        self.seq_len = img_len + text_len

        # Pre-compute padding index map for _pad_image / _unpad_image.
        # Stored as Python list (not registered buffer) to avoid the
        # index_select-with-buffer dispatch issue during XLA tracing.
        if self.padding != (0, 0, 0):
            orig_t, orig_h, orig_w = canvas_thw
            pt, ph, pw = self.padded_canvas

            pad_indices = []
            for ti in range(orig_t):
                for hi in range(orig_h):
                    for wi in range(orig_w):
                        padded_idx = ti * ph * pw + hi * pw + wi
                        pad_indices.append(padded_idx)
            self._pad_indices_list = pad_indices
        else:
            self._pad_indices_list = None

    def _pad_image(self, img_x):
        """
        Pad image tensor from [B, H, img_len, D] to [B, H, padded_img_len, D].
        Zero-pads along T, H, W dimensions using inline index tensors.
        """
        if self._pad_indices_list is None:
            return img_x

        b, h, _, d = img_x.shape

        # Create scatter index inline (becomes XLA graph constant during tracing)
        scatter_idx = torch.tensor(
            self._pad_indices_list, dtype=torch.long, device=img_x.device
        )

        # Allocate zero-filled padded tensor
        padded = torch.zeros(
            b, h, self.padded_img_len, d, dtype=img_x.dtype, device=img_x.device
        )

        # Scatter original pixels into padded positions
        scatter_idx_exp = scatter_idx.reshape(1, 1, -1, 1).expand(b, h, -1, d)
        padded.scatter_(2, scatter_idx_exp, img_x)

        return padded

    def _unpad_image(self, img_x):
        """
        Remove padding from image tensor: [B, H, padded_img_len, D] -> [B, H, img_len, D].
        Uses inline index tensor.
        """
        if self._pad_indices_list is None:
            return img_x

        gather_idx = torch.tensor(
            self._pad_indices_list, dtype=torch.long, device=img_x.device
        )
        return torch.index_select(img_x, 2, gather_idx)

    def forward(self, q, k, v):
        """
        STA forward pass — scatter-free, trace-compatible.

        Architecture:
          1. Split image/text, pad image, tile image tokens
          2. Image attention: Q is identity (tile order), gather KV per-block,
             call attention_cte in chunks, output is in tile order (no scatter)
          3. Text attention: full attention to all tokens (1 block)
          4. Untile image, unpad, concatenate with text output

        Args:
            q: [B, num_heads, seq_len, head_dim] in BF16
            k: [B, num_heads, seq_len, head_dim] in BF16
            v: [B, num_heads, seq_len, head_dim] in BF16

        Returns:
            output: [B, num_heads, seq_len, head_dim] in BF16
        """
        B, H, S, D = q.shape
        img_len = self.canvas_thw[0] * self.canvas_thw[1] * self.canvas_thw[2]

        # Split image and text
        img_q = q[:, :, :img_len]
        img_k = k[:, :, :img_len]
        img_v = v[:, :, :img_len]

        # Pad image to tile-divisible dimensions
        img_q = self._pad_image(img_q)
        img_k = self._pad_image(img_k)
        img_v = self._pad_image(img_v)

        # Tile image tokens: [B, H, N_img_blocks * block_size, D]
        img_q = tile_tokens(img_q, self.padded_canvas, self.tile_thw)
        img_k = tile_tokens(img_k, self.padded_canvas, self.tile_thw)
        img_v = tile_tokens(img_v, self.padded_canvas, self.tile_thw)

        # Handle text tokens (pad to block_size boundary)
        if self.text_len > 0:
            text_q = q[:, :, img_len:]
            text_k = k[:, :, img_len:]
            text_v = v[:, :, img_len:]

            if self.text_pad > 0:
                zero_pad = torch.zeros(
                    B, H, self.text_pad, D, dtype=q.dtype, device=q.device
                )
                text_q = torch.cat([text_q, zero_pad], dim=2)
                text_k = torch.cat([text_k, zero_pad], dim=2)
                text_v = torch.cat([text_v, zero_pad], dim=2)

        # Concatenate tiled image + text for KV source
        # This is the full token sequence that KV indices reference
        if self.text_len > 0:
            all_k = torch.cat([img_k, text_k], dim=2)
            all_v = torch.cat([img_v, text_v], dim=2)
        else:
            all_k = img_k
            all_v = img_v

        # Pre-scale Q for attention_cte (scale=1.0)
        img_q = img_q * self.scale

        # ─── Image block attention (scatter-free, per-chunk KV gather) ─
        # Q is already in tile order: [B, H, n_img_blocks * block_size, D]
        # No Q gather needed — identity.
        # KV is gathered per-chunk to avoid materializing the full
        # n_img_blocks * kv_len tensor which exceeds SBUF at 129-frame scale.

        kv_len = self._img_kv_len
        n_img = self.n_img_blocks
        H = self.num_heads

        # Reshape Q for chunked slicing: [B, H, n_img, block_size, D]
        q_by_block = img_q.reshape(B, H, n_img, self.block_size, D)

        # Call attention_cte in chunks, gathering KV per-chunk
        img_chunk_outputs = []
        for chunk in self._img_chunk_specs:
            cs = chunk["start_block"]
            ce = chunk["end_block"]
            n_blk = chunk["n_blocks"]

            # Q for this chunk: [B, H, n_blk, block_size, D] → [n_blk*H, block_size, D]
            q_c = q_by_block[:, :, cs:ce, :, :]  # [B, H, n_blk, block_size, D]
            q_c = q_c.permute(0, 2, 1, 3, 4)  # [B, n_blk, H, block_size, D]
            q_c = q_c.reshape(B * n_blk * H, self.block_size, D)

            # KV gather for this chunk only
            kv_idx = torch.tensor(
                chunk["kv_gather_list"], dtype=torch.long, device=q.device
            )
            # [B, H, n_blk * kv_len, D]
            k_c = torch.index_select(all_k, 2, kv_idx)
            v_c = torch.index_select(all_v, 2, kv_idx)

            # Reshape: [B, H, n_blk, kv_len, D] → [n_blk*H, kv_len, D]
            k_c = k_c.reshape(B, H, n_blk, kv_len, D)
            v_c = v_c.reshape(B, H, n_blk, kv_len, D)
            k_c = k_c.permute(0, 2, 1, 3, 4).reshape(B * n_blk * H, kv_len, D)
            v_c = v_c.permute(0, 2, 1, 3, 4).reshape(B * n_blk * H, kv_len, D)

            o_c = attention_cte(
                q=q_c,
                k=k_c,
                v=v_c,
                scale=1.0,
                causal_mask=False,
                tp_q=True,
                tp_k=True,
                tp_out=False,
            )
            img_chunk_outputs.append(o_c)

        # Concatenate: [n_img * H, block_size, D]
        img_out_batched = torch.cat(img_chunk_outputs, dim=0)

        # De-interleave: [B, n_img, H, block_size, D] → [B, H, n_img * block_size, D]
        img_out = img_out_batched.reshape(B, n_img, H, self.block_size, D)
        img_out = img_out.permute(0, 2, 1, 3, 4)  # [B, H, n_img, block_size, D]
        img_out_tiled = img_out.reshape(B, H, n_img * self.block_size, D)
        # ↑ Already in tile order — NO scatter needed!

        # Untile image output
        img_output = untile_tokens(img_out_tiled, self.padded_canvas, self.tile_thw)

        # Remove padding
        img_output = self._unpad_image(img_output)

        # ─── Text block attention ──────────────────────────────────────
        if self.text_len > 0:
            # Text block attends to all tokens (full attention).
            # Q: text tokens, KV: all tokens (img_tiled + text)
            text_q_scaled = text_q * self.scale

            # For text attention, KV is the full tiled+text sequence.
            # But that's total_tokens which could be large (64K+).
            # attention_cte can handle this — it's just 1 block × H heads.
            # batch = text_block_num * H, seqlen_q = block_size, seqlen_kv = total_tokens

            # Reshape text Q: [B, H, text_target, D] → [B, text_block_num, H, block_size, D]
            tq = text_q_scaled.reshape(B, H, self.text_block_num, self.block_size, D)
            tq = tq.permute(0, 2, 1, 3, 4).reshape(
                B * self.text_block_num * H, self.block_size, D
            )

            # KV for text: full sequence [B, H, total_tokens, D]
            # Concatenate img_k (tiled) + text_k for the full KV
            all_q_k = all_k  # already built above: [B, H, total_tokens, D]
            all_q_v = all_v

            # Expand KV for each text block (they all see the same KV)
            # [B, H, total_tokens, D] → [B, text_block_num, H, total_tokens, D]
            tk = all_q_k.unsqueeze(1).expand(
                B, self.text_block_num, H, self.total_tokens, D
            )
            tv = all_q_v.unsqueeze(1).expand(
                B, self.text_block_num, H, self.total_tokens, D
            )
            tk = tk.reshape(B * self.text_block_num * H, self.total_tokens, D)
            tv = tv.reshape(B * self.text_block_num * H, self.total_tokens, D)

            text_out = attention_cte(
                q=tq,
                k=tk,
                v=tv,
                scale=1.0,
                causal_mask=False,
                tp_q=True,
                tp_k=True,
                tp_out=False,
            )

            # Reshape: [text_block_num * H, block_size, D] → [B, H, text_target, D]
            text_out = text_out.reshape(B, self.text_block_num, H, self.block_size, D)
            text_out = text_out.permute(0, 2, 1, 3, 4).reshape(
                B, H, self.text_target, D
            )

            # Trim text padding
            text_output = text_out[:, :, : self.text_len, :]

            # Concatenate image + text
            final_output = torch.cat([img_output, text_output], dim=2)
        else:
            final_output = img_output

        return final_output


# ─── Factory function ────────────────────────────────────────────────────────


def create_sta_attention(
    canvas_thw: Tuple[int, int, int],
    tile_thw: Tuple[int, int, int] = (6, 8, 8),
    kernel_thw: Tuple[int, int, int] = (3, 3, 3),
    text_len: int = 320,
    num_heads: int = 4,
    head_dim: int = 128,
) -> STAAttention:
    """
    Create an STAAttention module with pre-computed indices.

    This is the main entry point for integration into the DiT wrapper.
    Call this at model construction time, before tracing.
    """
    return STAAttention(
        canvas_thw=canvas_thw,
        tile_thw=tile_thw,
        kernel_thw=kernel_thw,
        text_len=text_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
