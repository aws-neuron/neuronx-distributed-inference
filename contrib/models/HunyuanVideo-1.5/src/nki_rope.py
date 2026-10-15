"""
NKI Fused RoPE Kernel for HunyuanVideo-1.5 DiT.

Adapted from aws-neuron/nki-library (commit d2ad3a57, 2026-04-13).

Strategy: Reshape input on torch side to separate even/odd elements into
contiguous halves. NKI kernel operates on contiguous data with simple
element-wise multiply-add. Output tensor is passed as parameter (NKI
kernels cannot return values).

Usage:
    from nki_rope import nki_rope_apply
    img_q, img_k = nki_rope_apply(img_q, img_k, freqs_cos, freqs_sin)
"""

import torch
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from torch_neuronx.xla_impl.ops import nki_jit


# ============================================================================
# NKI RoPE Kernel — contiguous layout, pure element-wise, output parameter
# ============================================================================


@nki.jit
def rope_contiguous_kernel(
    x_in,
    cos,
    sin,
    x_out,
):
    """
    Apply RoPE rotation on contiguous-layout input.

    x_in:  [d_head, B*n_heads, S] @ HBM
        First half (0:d_head//2) = even elements, second half = odd elements
    cos:   [d_head//2, 1, S] @ HBM
    sin:   [d_head//2, 1, S] @ HBM
    x_out: [d_head, B*n_heads, S] @ HBM (pre-allocated output)
        Same layout: first half = even output, second half = odd output

    Math:
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_odd * cos + x_even * sin
    """
    d_head = x_in.shape[0]
    BH = x_in.shape[1]
    S = x_in.shape[2]
    half_d = d_head // 2

    i_p = nl.arange(half_d)[:, None]  # partition indices [0..63]
    i_f = nl.arange(S)[None, :]

    # Load cos/sin ONCE (shared across all batch-heads)
    cos_tile = nl.load(cos[i_p, 0, i_f])  # [half_d, S]
    sin_tile = nl.load(sin[i_p, 0, i_f])  # [half_d, S]

    # Process each batch-head
    for bh in nl.affine_range(BH):
        # Load even half and odd half
        x_even = nl.load(x_in[i_p, bh, i_f])  # [half_d, S]
        x_odd = nl.load(x_in[i_p + half_d, bh, i_f])  # [half_d, S]

        # Compute rotation
        out_even = x_even * cos_tile - x_odd * sin_tile  # [half_d, S]
        out_odd = x_odd * cos_tile + x_even * sin_tile  # [half_d, S]

        # Store to output
        nl.store(x_out[i_p, bh, i_f], value=out_even)
        nl.store(x_out[i_p + half_d, bh, i_f], value=out_odd)


# ============================================================================
# Torch Wrapper
# ============================================================================

_rope_kernel = nki_jit()(rope_contiguous_kernel)


def nki_rope_apply(q, k, freqs_cos, freqs_sin):
    """
    Apply NKI fused RoPE to q and k tensors.

    Args:
        q:          [B, L, n_heads, D]  (B=1, L=2880, n_heads=4, D=128)
        k:          [B, L, n_heads, D]
        freqs_cos:  [L, D]             (interleaved: [c0, c0, c1, c1, ...])
        freqs_sin:  [L, D]             (interleaved: [s0, s0, s1, s1, ...])

    Returns:
        q_out, k_out: same shapes as input [B, L, n_heads, D]

    Steps:
    1. De-interleave cos/sin: take every other element to get half-dim
    2. Separate even/odd elements of q/k into contiguous halves
    3. Call NKI kernel (pure element-wise on contiguous data)
    4. Re-interleave output back to original layout
    """
    B, L, n_heads, D = q.shape
    half_D = D // 2

    # --- Prepare cos/sin: [L, D] -> [D//2, 1, L] ---
    cos_half = freqs_cos[:, 0::2]  # [L, D//2]
    sin_half = freqs_sin[:, 0::2]  # [L, D//2]
    cos_nki = cos_half.permute(1, 0).unsqueeze(1).contiguous()  # [D//2, 1, L]
    sin_nki = sin_half.permute(1, 0).unsqueeze(1).contiguous()  # [D//2, 1, L]

    # --- Separate even/odd along D dimension ---
    q_pairs = q.reshape(B, L, n_heads, half_D, 2)
    q_even = q_pairs[..., 0]  # [B, L, H, D//2]
    q_odd = q_pairs[..., 1]  # [B, L, H, D//2]
    q_even_t = q_even.permute(3, 0, 2, 1).reshape(half_D, B * n_heads, L)
    q_odd_t = q_odd.permute(3, 0, 2, 1).reshape(half_D, B * n_heads, L)
    q_nki = torch.cat([q_even_t, q_odd_t], dim=0).contiguous()  # [D, B*H, L]

    k_pairs = k.reshape(B, L, n_heads, half_D, 2)
    k_even = k_pairs[..., 0]
    k_odd = k_pairs[..., 1]
    k_even_t = k_even.permute(3, 0, 2, 1).reshape(half_D, B * n_heads, L)
    k_odd_t = k_odd.permute(3, 0, 2, 1).reshape(half_D, B * n_heads, L)
    k_nki = torch.cat([k_even_t, k_odd_t], dim=0).contiguous()  # [D, B*H, L]

    # --- Pre-allocate outputs ---
    q_out_nki = torch.zeros_like(q_nki)
    k_out_nki = torch.zeros_like(k_nki)

    # --- Call NKI kernel ---
    _rope_kernel(q_nki, cos_nki, sin_nki, q_out_nki)
    _rope_kernel(k_nki, cos_nki, sin_nki, k_out_nki)

    # --- Re-interleave: separate halves back to interleaved pairs ---
    q_out_even = q_out_nki[:half_D]  # [D//2, B*H, L]
    q_out_odd = q_out_nki[half_D:]  # [D//2, B*H, L]
    q_out_even = q_out_even.reshape(half_D, B, n_heads, L).permute(
        1, 3, 2, 0
    )  # [B, L, H, D//2]
    q_out_odd = q_out_odd.reshape(half_D, B, n_heads, L).permute(
        1, 3, 2, 0
    )  # [B, L, H, D//2]
    q_out = torch.stack([q_out_even, q_out_odd], dim=-1).reshape(B, L, n_heads, D)

    k_out_even = k_out_nki[:half_D]
    k_out_odd = k_out_nki[half_D:]
    k_out_even = k_out_even.reshape(half_D, B, n_heads, L).permute(1, 3, 2, 0)
    k_out_odd = k_out_odd.reshape(half_D, B, n_heads, L).permute(1, 3, 2, 0)
    k_out = torch.stack([k_out_even, k_out_odd], dim=-1).reshape(B, L, n_heads, D)

    return q_out, k_out
