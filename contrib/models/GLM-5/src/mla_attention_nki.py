# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA (Multi-head Latent Attention) NKI kernel for token generation (decode).

Fuses the attention computation for GLM-5 / DeepSeek-V3 MLA architecture:
  1. Dual-score computation (RoPE + absorbed nope) in a single cache pass
  2. Online softmax across cache length
  3. V accumulation from compressed KV cache
  4. V absorption (matmul with v_absorb weight)

This replaces the ~8 separate kernel launches per layer in the standard
PyTorch MLA attention path during TKG (token generation).

Target: trn2 (NKI 0.3.0, nc_version >= 3)

Usage:
    from mla_attention_nki import mla_attention_tkg, mla_attention_tkg_reference, NKI_AVAILABLE
    # mla_attention_tkg is None if NKI SDK is not installed
    # mla_attention_tkg_reference always works (pure PyTorch)
"""

import torch

# NKI imports - only available on Neuron instances
NKI_AVAILABLE = False
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    NKI_AVAILABLE = True
except ImportError:
    pass


# === Self-contained utilities ===


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


# === Hardware constants ===
P_MAX = 128  # SBUF partition dimension max
PSUM_F_MAX = 512  # PSUM free dimension max (gen2/3)
CACHE_TILE_SIZE = 512  # Process 512 cache positions per tile


# === NKI Kernel (Neuron-only) ===
# The kernel is defined in mla_attention_nki_kernel.py to avoid parse errors
# on machines without the NKI SDK.
mla_attention_tkg = None

if NKI_AVAILABLE:
    from mla_attention_nki_kernel import mla_attention_tkg


# === PyTorch reference implementation (always available) ===


def mla_attention_tkg_reference(
    q_pe: torch.Tensor,  # [BH, d_rope]
    q_nope: torch.Tensor,  # [BH, d_c]
    kv_cache: torch.Tensor,  # [B, S, d_cache]
    v_absorb: torch.Tensor,  # [H, d_v, d_c]
    attn_mask: torch.Tensor,  # [BH, S] bool
    softmax_scale: float,
    n_heads: int,
    d_rope: int,
    d_c: int,
    d_v: int,
    seq_len: int,
) -> torch.Tensor:
    """
    PyTorch reference for MLA attention TKG.

    Returns:
        [BH, d_v] attention output after V absorption
    """
    BH = q_pe.shape[0]
    B = BH // n_heads

    output = torch.zeros(BH, d_v, dtype=q_pe.dtype, device=q_pe.device)

    # Cast all inputs to float32 for numerical stability
    q_pe_f = q_pe.float()
    q_nope_f = q_nope.float()
    kv_cache_f = kv_cache.float()
    v_absorb_f = v_absorb.float()

    for bh in range(BH):
        b = bh // n_heads
        h = bh % n_heads

        # Extract cache for this batch: [S, d_cache]
        cache = kv_cache_f[b, :seq_len, :]  # [S, d_cache]
        k_pe_cache = cache[:, :d_rope]  # [S, d_rope]
        c_kv_cache = cache[:, d_rope : d_rope + d_c]  # [S, d_c]

        # Dual scores: [1, S]
        rope_scores = q_pe_f[bh : bh + 1, :] @ k_pe_cache.T  # [1, S]
        nope_scores = q_nope_f[bh : bh + 1, :] @ c_kv_cache.T  # [1, S]
        scores = (rope_scores + nope_scores) * softmax_scale  # [1, S]

        # Mask
        mask = attn_mask[bh, :seq_len].unsqueeze(0).float()  # [1, S]
        scores = scores * mask + (1 - mask) * (-1e9)

        # Softmax
        weights = torch.softmax(scores, dim=-1)  # [1, S]

        # V accumulation: [1, S] @ [S, d_c] = [1, d_c]
        v_accum = weights @ c_kv_cache  # [1, d_c]

        # V absorption: [1, d_c] @ [d_c, d_v] = [1, d_v]
        # v_absorb[h] is [d_v, d_c], so v_absorb[h]^T = [d_c, d_v]
        attn_out = v_accum @ v_absorb_f[h].T  # [1, d_c] @ [d_c, d_v] = [1, d_v]

        output[bh] = attn_out.squeeze(0)

    return output
