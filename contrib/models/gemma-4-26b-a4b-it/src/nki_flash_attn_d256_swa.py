"""
Flash attention for d=256 with sliding window mask, optimized for Gemma4 E2B SWA layers.

Adapted from Qwen3-Coder-Next nki_flash_attn_d256_pipe.py with:
  - Sliding window mask (window_size=512) replacing full causal mask
  - Tile-skip optimization: only K tiles within the window are processed
  - No fused RoPE (E2B applies RoPE externally in prep_qkv_tensors)
  - NKI 0.3.0+ API (affine_select keyword offset)

Architecture: 2x128 QK tiling for head_dim=256, 3-stage software pipeline.
Called per (batch, kv_head) with pre-sliced post-RoPE Q/K/V in BHSD layout.

E2B SWA layer specifics:
  - head_dim = 256
  - num_kv_heads = 1 (GQA, 8 Q heads per KV head at full model, varies by TP)
  - sliding_window = 512
  - Q/K already have RoPE applied (theta=10000, full rotation)
  - V already has v_norm applied (V / RMS(V))

Layouts (per-call, single batch + single kv_head):
  Q: (1, q_h_per_k_h, seq_q, 256)  -- BHSD
  K: (1, 1, seq_k, 256)             -- BHSD
  V: (1, 1, seq_v, 256)             -- BHSD
  O: (1, q_h_per_k_h, seq_q, 256)   -- BHSD

Internal SBUF layout after DMA transpose of Q/K:
  Q_sb: (D_TILE=128, Q_GRP_SZ=128)  -- d on partition, seq on free
  K_sb: (D_TILE=128, K_TILE_SZ=512)  -- d on partition, seq on free
  V_sb: (V_TILE_SZ=128, D_HEAD=256)  -- seq on partition, d on free
"""

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import math
import nki.isa as nisa
import nki.language as nl
import nki

# ============================================================================
# Constants
# ============================================================================
D_HEAD = 256
D_TILE = 128  # partition dim tile for d-tiling (256 = 2 x 128)
Q_GRP_SZ = 128  # Q group size = partition dim max
K_TILE_SZ = 512  # K tile size for MM1 (free dim of K in matmul)
V_TILE_SZ = 128  # V tile size for MM2 (partition dim of transposed P)
LARGE_TILE_SZ = 2048  # Large tile grouping
EXP_TILE_SZ = 512  # Exp tile for activation_reduce
PSUM_FMAX = 512  # PSUM free dimension max
FLOAT32_MIN = -3.4028235e38


# ============================================================================
# ModularAllocator helpers
# ============================================================================


def _align32(addr):
    """Round up address to 32-byte alignment (required for DMA transpose)."""
    return (addr + 31) // 32 * 32


def _alloc_modular_1d(shape, dtype, block_dim, num_free_tiles, base_addr):
    """Allocate 1D modular buffer list."""
    base_addr = _align32(base_addr)
    tile_elems = 1
    for d in shape[1:]:
        tile_elems *= d
    dtype_size = 4 if dtype == nl.float32 else 2
    tile_bytes = _align32(tile_elems * dtype_size)

    tensors = []
    for i in range(block_dim):
        tensors.append(nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))
    next_addr = base_addr + num_free_tiles * tile_bytes
    return tensors, next_addr


def _alloc_modular_2d(
    shape, dtype, block_dim0, block_dim1, num_free0, num_free1, base_addr
):
    """Allocate 2D modular buffer."""
    base_addr = _align32(base_addr)
    tile_elems = 1
    for d in shape[1:]:
        tile_elems *= d
    dtype_size = 4 if dtype == nl.float32 else 2
    tile_bytes = _align32(tile_elems * dtype_size)

    tensors = []
    for i in range(block_dim0):
        row = []
        for j in range(block_dim1):
            row.append(nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))
        tensors.append(row)
    next_addr = base_addr + num_free0 * num_free1 * tile_bytes
    return tensors, next_addr


def _alloc_modular_3d(shape, dtype, dims, n_free, base_addr):
    """Allocate 3D modular buffer."""
    base_addr = _align32(base_addr)
    tile_elems = 1
    for d in shape[1:]:
        tile_elems *= d
    dtype_size = 4 if dtype == nl.float32 else 2
    tile_bytes = _align32(tile_elems * dtype_size)

    tensors = []
    for i in range(dims[0]):
        layer = []
        for j in range(dims[1]):
            row = []
            for k in range(dims[2]):
                row.append(nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))
            layer.append(row)
        tensors.append(layer)
    total_physical = n_free[0] * n_free[1] * n_free[2]
    next_addr = base_addr + total_physical * tile_bytes
    return tensors, next_addr


# ============================================================================
# Pipeline stage functions
# ============================================================================


def _pipe_load_q(
    grp_i,
    q_sb_lo,
    q_sb_hi,
    q_hbm,
    d_tile,
    seqlen_q,
    batch_id,
    q_head_idx,
    n_heads,
    d_head,
):
    """Load Q group from BHSD HBM into SBUF with DMA transpose to (D, S) layout."""
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)

    q_offset = (
        batch_id * n_heads * seqlen_q * d_head
        + q_head_idx * seqlen_q * d_head
        + q_start * d_head
    )

    # Lo half: D[0:128]
    nisa.dma_transpose(
        dst=q_sb_lo[grp_i].ap([[Q_GRP_SZ, d_tile], [1, 1], [1, 1], [1, num_q]]),
        src=q_hbm.ap(
            [[d_head, num_q], [1, 1], [1, 1], [1, d_tile]],
            offset=q_offset,
        ),
    )
    # Hi half: D[128:256]
    nisa.dma_transpose(
        dst=q_sb_hi[grp_i].ap([[Q_GRP_SZ, d_tile], [1, 1], [1, 1], [1, num_q]]),
        src=q_hbm.ap(
            [[d_head, num_q], [1, 1], [1, 1], [1, d_tile]],
            offset=q_offset + d_tile,
        ),
    )


def _pipe_qk_and_max(
    grp_i,
    q_sb_lo,
    q_sb_hi,
    k_sb_lo,
    k_sb_hi,
    mm1_masked,
    mm1_partial_max,
    mm1_psum,
    mm1_copy_sb,
    mm1_asel_sb,
    seqlen_q,
    seqlen_kv,
    scale,
    num_k_tiles,
    num_large_tiles,
    window_size,
):
    """Compute QK^T with d=256 tiling, sliding window mask, scale, and row-wise max.

    Sliding window mask: keep when q_pos - window_size < k_pos <= q_pos
    i.e., k_start + f <= q_start + p  (upper bound, same as causal)
    AND   k_start + f > q_start + p - window_size  (lower bound)

    Combined as affine_select with causal upper bound, plus tile-level skip
    for K tiles entirely outside the window.
    """
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)
    num_k_tiles_per_large = LARGE_TILE_SZ // K_TILE_SZ  # 4

    # Initialize partial max to -inf
    nisa.memset(mm1_partial_max[grp_i][...], value=FLOAT32_MIN)

    # Initialize mm1_masked to -inf (skipped K tiles → exp=0)
    for lt_idx in range(num_large_tiles):
        nisa.memset(mm1_masked[grp_i][lt_idx][...], value=FLOAT32_MIN)

    for large_tile_idx in range(num_large_tiles):
        for k_tile_local in range(num_k_tiles_per_large):
            k_tile_idx = large_tile_idx * num_k_tiles_per_large + k_tile_local
            if k_tile_idx >= num_k_tiles:
                continue

            k_start = k_tile_idx * K_TILE_SZ
            num_k = min(seqlen_kv - k_start, K_TILE_SZ)
            if num_k <= 0:
                continue

            # Tile-level skip: sliding window bounds
            # Upper bound: q_last < k_start means all Q positions before this K tile
            q_last = q_start + num_q - 1
            if q_last < k_start:
                continue

            # Lower bound: k_end <= q_first - window means K tile entirely before window
            q_first = q_start
            k_end = k_start + num_k - 1
            if k_end < q_first - window_size + 1:
                continue

            # MM1: QK = Q_lo^T @ K_lo + Q_hi^T @ K_hi
            psum_tile = mm1_psum[grp_i][large_tile_idx][k_tile_local]

            # First half: d[0:128]
            nisa.nc_matmul(
                psum_tile[:num_q, :num_k],
                q_sb_lo[grp_i][:D_TILE, :num_q],
                k_sb_lo[k_tile_idx][:D_TILE, :num_k],
            )
            # Second half: d[128:256] — accumulates into same PSUM
            nisa.nc_matmul(
                psum_tile[:num_q, :num_k],
                q_sb_hi[grp_i][:D_TILE, :num_q],
                k_sb_hi[k_tile_idx][:D_TILE, :num_k],
            )

            # Copy PSUM -> temp SBUF (unscaled)
            nisa.tensor_copy(
                mm1_copy_sb[:num_q, :num_k],
                psum_tile[:num_q, :num_k],
            )

            # Sliding window mask via affine_select (NKI 0.3.0 API)
            # Upper bound (causal): keep when (k_start+f) <= (q_start+p)
            # Pattern: (-1)*p + (1)*f + offset >= 0 => f <= p + offset
            # offset = q_start - k_start
            # This masks future tokens (same as causal)
            nisa.affine_select(
                dst=mm1_asel_sb[:num_q, :num_k],
                pattern=[[-1, num_k]],
                channel_multiplier=1,
                on_true_tile=mm1_copy_sb[:num_q, :num_k],
                on_false_value=FLOAT32_MIN,
                offset=q_start - k_start,
                cmp_op=nl.greater_equal,
            )

            # Lower bound: keep when k_pos >= q_pos - window_size + 1
            # where k_pos = k_start + f, q_pos = q_start + p
            # => (k_start + f) >= (q_start + p) - window_size + 1
            # => f >= p + (q_start - k_start) - window_size + 1
            # => f >= p - (k_start - q_start + window_size - 1)
            #
            # affine_select with ch_mul=-1, pattern=[[1, num_k]]:
            #   val = (-1)*p + (1)*f + offset >= 0
            #   => f >= p - offset
            # So offset = k_start - q_start + window_size - 1
            lower_offset = k_start - q_start + window_size - 1
            nisa.affine_select(
                dst=mm1_asel_sb[:num_q, :num_k],
                pattern=[[1, num_k]],
                channel_multiplier=-1,
                on_true_tile=mm1_asel_sb[:num_q, :num_k],
                on_false_value=FLOAT32_MIN,
                offset=lower_offset,
                cmp_op=nl.greater_equal,
            )

            # Scale + max extraction
            nisa.tensor_scalar_reduce(
                mm1_masked[grp_i][large_tile_idx][
                    :num_q, nl.ds(k_tile_local * K_TILE_SZ, num_k)
                ],
                data=mm1_asel_sb[:num_q, :num_k],
                op0=nl.multiply,
                operand0=scale,
                reduce_op=nl.maximum,
                reduce_res=mm1_partial_max[grp_i][:num_q, k_tile_idx],
            )


def _pipe_update_max(
    grp_i, mm1_partial_max, mm1_section_max, mm1_running_max, num_k_tiles, seqlen_q
):
    """Compute section max from partial maxes, store as -max (negated)."""
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)

    nisa.tensor_reduce(
        mm1_section_max[grp_i][:num_q, 0],
        nl.maximum,
        mm1_partial_max[grp_i][:num_q, :num_k_tiles],
        1,
        negate=True,
    )

    nisa.tensor_copy(mm1_running_max[:num_q, grp_i], mm1_section_max[grp_i][:num_q, 0])


def _pipe_exp(
    grp_i,
    mm1_masked,
    mm1_running_max,
    exp_sb,
    exp_partial_sum,
    exp_tp_sb,
    seqlen_q,
    seqlen_kv,
    num_large_tiles,
    num_k_tiles,
):
    """Compute exp(S - max), partial sums, and DMA transpose for MM2."""
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)
    num_exp_per_large = LARGE_TILE_SZ // EXP_TILE_SZ  # 4

    nisa.memset(exp_partial_sum[grp_i][...], value=0.0)

    for large_tile_idx in range(num_large_tiles):
        for exp_tile_idx in range(num_exp_per_large):
            kv_start = large_tile_idx * LARGE_TILE_SZ + exp_tile_idx * EXP_TILE_SZ
            num_kv = min(seqlen_kv - kv_start, EXP_TILE_SZ)
            if num_kv <= 0:
                continue

            nisa.activation_reduce(
                exp_sb[grp_i][large_tile_idx][
                    :num_q, nl.ds(exp_tile_idx * EXP_TILE_SZ, num_kv)
                ],
                op=nl.exp,
                data=mm1_masked[grp_i][large_tile_idx][
                    :num_q, nl.ds(exp_tile_idx * EXP_TILE_SZ, num_kv)
                ],
                reduce_op=nl.add,
                reduce_res=exp_partial_sum[grp_i][
                    :num_q,
                    large_tile_idx * num_exp_per_large + exp_tile_idx,
                ],
                bias=mm1_running_max[:num_q, grp_i],
            )

            # DMA transpose: exp_sb[Q=128, KV=512] -> exp_tp_sb[KV=128, Q=512]
            num_kv_outer = num_kv // V_TILE_SZ
            num_kv_inner = num_kv % V_TILE_SZ

            if num_kv_outer >= 1:
                nisa.dma_transpose(
                    dst=exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [K_TILE_SZ, V_TILE_SZ],
                            [1, 1],
                            [V_TILE_SZ, num_kv_outer],
                            [1, num_q],
                        ]
                    ),
                    src=exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [LARGE_TILE_SZ, num_q],
                            [1, 1],
                            [V_TILE_SZ, num_kv_outer],
                            [1, V_TILE_SZ],
                        ],
                        offset=exp_tile_idx * K_TILE_SZ,
                    ),
                )

            if num_kv_inner > 0:
                nisa.dma_transpose(
                    dst=exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [K_TILE_SZ, num_kv_inner],
                            [1, 1],
                            [V_TILE_SZ, 1],
                            [1, num_q],
                        ],
                        offset=num_kv_outer * V_TILE_SZ,
                    ),
                    src=exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [LARGE_TILE_SZ, num_q],
                            [1, 1],
                            [V_TILE_SZ, 1],
                            [1, num_kv_inner],
                        ],
                        offset=exp_tile_idx * K_TILE_SZ + num_kv_outer * V_TILE_SZ,
                    ),
                )


def _pipe_pv(
    grp_i,
    exp_tp_sb,
    v_sb,
    mm2_psum_lo,
    mm2_psum_hi,
    mm2_sb,
    seqlen_q,
    seqlen_kv,
    num_large_tiles,
    num_v_tiles,
):
    """Compute P@V (MM2) with d=256 split into lo/hi halves."""
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)
    num_mm2_grps_per_large = LARGE_TILE_SZ // K_TILE_SZ  # 4
    num_mm2_per_grp = K_TILE_SZ // V_TILE_SZ  # 4

    nisa.memset(mm2_sb[grp_i][...], value=0.0)

    for large_tile_idx in range(num_large_tiles):
        psum_tile_lo = mm2_psum_lo[grp_i][large_tile_idx]
        psum_tile_hi = mm2_psum_hi[grp_i][large_tile_idx]

        for mm2_grp_i in range(num_mm2_grps_per_large):
            exp_tp_tile = exp_tp_sb[grp_i][large_tile_idx][mm2_grp_i]

            for mm2_i in range(num_mm2_per_grp):
                v_tile_idx = (
                    large_tile_idx * num_mm2_grps_per_large * num_mm2_per_grp
                    + mm2_grp_i * num_mm2_per_grp
                    + mm2_i
                )
                kv_start = v_tile_idx * V_TILE_SZ
                num_kv = min(seqlen_kv - kv_start, V_TILE_SZ)
                if num_kv <= 0 or v_tile_idx >= num_v_tiles:
                    continue

                # MM2 lo: exp_tp^T @ V[:, :128]
                nisa.nc_matmul(
                    psum_tile_lo[:num_q, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q)],
                    v_sb[v_tile_idx][:num_kv, :D_TILE],
                )
                # MM2 hi: exp_tp^T @ V[:, 128:256]
                nisa.nc_matmul(
                    psum_tile_hi[:num_q, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q)],
                    v_sb[v_tile_idx][:num_kv, nl.ds(D_TILE, D_TILE)],
                )

        # Accumulate large tile results into SBUF
        if large_tile_idx == 0:
            nisa.tensor_copy(
                mm2_sb[grp_i][:num_q, :D_TILE], psum_tile_lo[:num_q, :D_TILE]
            )
            nisa.tensor_copy(
                mm2_sb[grp_i][:num_q, nl.ds(D_TILE, D_TILE)],
                psum_tile_hi[:num_q, :D_TILE],
            )
        else:
            nisa.tensor_tensor(
                mm2_sb[grp_i][:num_q, :D_TILE],
                mm2_sb[grp_i][:num_q, :D_TILE],
                psum_tile_lo[:num_q, :D_TILE],
                nl.add,
            )
            nisa.tensor_tensor(
                mm2_sb[grp_i][:num_q, nl.ds(D_TILE, D_TILE)],
                mm2_sb[grp_i][:num_q, nl.ds(D_TILE, D_TILE)],
                psum_tile_hi[:num_q, :D_TILE],
                nl.add,
            )


def _pipe_write_back(
    grp_i,
    mm2_sb,
    exp_partial_sum,
    exp_sum_recip,
    wb_exp_section_sum,
    wb_zero_bias,
    wb_o_bf16,
    o_hbm,
    seqlen_q,
    num_exp_tiles,
    batch_id,
    q_head_idx,
):
    """Write-back: normalize by 1/sum(exp), cast to bf16, DMA to HBM."""
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)

    nisa.tensor_reduce(
        wb_exp_section_sum[grp_i][:num_q, 0],
        nl.add,
        exp_partial_sum[grp_i][:num_q, :num_exp_tiles],
        axis=1,
    )

    nisa.reciprocal(
        exp_sum_recip[grp_i][:num_q, 0],
        wb_exp_section_sum[grp_i][:num_q, 0],
    )

    # Scale output and cast to bf16
    nisa.activation(
        wb_o_bf16[grp_i][:num_q, :D_HEAD],
        nl.copy,
        mm2_sb[grp_i][:num_q, :D_HEAD],
        scale=exp_sum_recip[grp_i][:num_q, 0],
        bias=wb_zero_bias[:num_q],
    )

    # DMA to HBM output
    nisa.dma_copy(
        dst=o_hbm[batch_id, q_head_idx, q_start : q_start + num_q, 0:D_HEAD],
        src=wb_o_bf16[grp_i][:num_q, :D_HEAD],
    )


# ============================================================================
# Main kernel
# ============================================================================


@nki.jit
def flash_attn_d256_swa(
    q,
    k,
    v,
    q_h_per_k_h=8,
    n_kv_heads=1,
    seqlen_q=512,
    seqlen_kv=512,
    window_size=512,
):
    """
    Flash attention for head_dim=256 with sliding window mask.

    Called per (batch, kv_head) pair with pre-sliced post-RoPE tensors.

    Args:
        q: (1, q_h_per_k_h, seq_q, 256)  -- bfloat16, BHSD, post-RoPE
        k: (1, 1, seq_k, 256)             -- bfloat16, BHSD, post-RoPE
        v: (1, 1, seq_v, 256)             -- bfloat16, BHSD, post-v_norm
        q_h_per_k_h: Q heads per KV head (8 for E2B full model)
        n_kv_heads: must be 1 (kernel processes one KV head at a time)
        seqlen_q: sequence length for Q
        seqlen_kv: sequence length for K/V
        window_size: sliding window size (512 for E2B SWA layers)

    Returns:
        o: (1, q_h_per_k_h, seq_q, 256)  -- bfloat16, BHSD
    """
    d = D_HEAD
    n_heads = q_h_per_k_h * n_kv_heads
    scale = 1.0 / math.sqrt(d)

    batch_id = 0
    kv_head_id = 0

    # Output allocation
    o = nl.ndarray((1, n_heads, seqlen_q, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    num_grps = (seqlen_q + Q_GRP_SZ - 1) // Q_GRP_SZ
    num_k_tiles = (seqlen_kv + K_TILE_SZ - 1) // K_TILE_SZ
    num_v_tiles = (seqlen_kv + V_TILE_SZ - 1) // V_TILE_SZ
    num_large_tiles = (seqlen_kv + LARGE_TILE_SZ - 1) // LARGE_TILE_SZ
    num_exp_per_large = LARGE_TILE_SZ // EXP_TILE_SZ
    num_exp_tiles = num_large_tiles * num_exp_per_large

    # =========================================================================
    # Buffer Allocation
    # =========================================================================
    sca = 0

    k_sb_lo, sca = _alloc_modular_1d(
        (D_TILE, K_TILE_SZ), nl.bfloat16, num_k_tiles, num_k_tiles, sca
    )
    k_sb_hi, sca = _alloc_modular_1d(
        (D_TILE, K_TILE_SZ), nl.bfloat16, num_k_tiles, num_k_tiles, sca
    )
    v_sb, sca = _alloc_modular_1d(
        (V_TILE_SZ, D_HEAD), nl.bfloat16, num_v_tiles, num_v_tiles, sca
    )
    q_sb_lo, sca = _alloc_modular_1d((D_TILE, Q_GRP_SZ), nl.bfloat16, num_grps, 2, sca)
    q_sb_hi, sca = _alloc_modular_1d((D_TILE, Q_GRP_SZ), nl.bfloat16, num_grps, 2, sca)

    # Masking temp buffers
    sca = _align32(sca)
    mm1_copy_sb = nl.ndarray((Q_GRP_SZ, K_TILE_SZ), dtype=nl.float32, buffer=nl.sbuf)
    sca += K_TILE_SZ * 4
    sca = _align32(sca)
    mm1_asel_sb = nl.ndarray((Q_GRP_SZ, K_TILE_SZ), dtype=nl.float32, buffer=nl.sbuf)
    sca += K_TILE_SZ * 4

    mm1_masked, sca = _alloc_modular_2d(
        (Q_GRP_SZ, LARGE_TILE_SZ),
        nl.float32,
        num_grps,
        num_large_tiles,
        2,
        num_large_tiles,
        sca,
    )
    mm1_partial_max, sca = _alloc_modular_1d(
        (Q_GRP_SZ, num_k_tiles), nl.float32, num_grps, 2, sca
    )
    mm1_section_max, sca = _alloc_modular_1d(
        (Q_GRP_SZ, 1), nl.float32, num_grps, 2, sca
    )

    sca = _align32(sca)
    mm1_running_max = nl.ndarray((Q_GRP_SZ, num_grps), dtype=nl.float32, buffer=nl.sbuf)
    sca += num_grps * 4

    exp_sb, sca = _alloc_modular_2d(
        (Q_GRP_SZ, LARGE_TILE_SZ),
        nl.bfloat16,
        num_grps,
        num_large_tiles,
        1,
        num_large_tiles,
        sca,
    )
    exp_partial_sum, sca = _alloc_modular_1d(
        (Q_GRP_SZ, num_exp_tiles), nl.float32, num_grps, 2, sca
    )
    exp_tp_sb, sca = _alloc_modular_3d(
        (V_TILE_SZ, K_TILE_SZ),
        nl.bfloat16,
        (num_grps, num_large_tiles, num_exp_per_large),
        (2, num_large_tiles, num_exp_per_large),
        sca,
    )
    mm2_sb, sca = _alloc_modular_1d((Q_GRP_SZ, D_HEAD), nl.float32, num_grps, 2, sca)
    exp_sum_recip, sca = _alloc_modular_1d((Q_GRP_SZ, 1), nl.float32, num_grps, 2, sca)

    wb_exp_section_sum, sca = _alloc_modular_1d(
        (Q_GRP_SZ, 1), nl.float32, num_grps, 2, sca
    )
    sca = _align32(sca)
    wb_zero_bias = nl.ndarray((Q_GRP_SZ, 1), dtype=nl.float32, buffer=nl.sbuf)
    sca += 1 * 4
    wb_o_bf16, sca = _alloc_modular_1d(
        (Q_GRP_SZ, D_HEAD), nl.bfloat16, num_grps, 2, sca
    )

    # =========================================================================
    # GQA outer loop
    # =========================================================================
    for i_q_h in range(q_h_per_k_h):
        q_head_idx = kv_head_id * q_h_per_k_h + i_q_h

        # PSUM allocations (per GQA iteration)
        mm1_psum = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                tile_row = []
                for kt_idx in range(4):
                    tile_row.append(
                        nl.ndarray(
                            (Q_GRP_SZ, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum
                        )
                    )
                grp_row.append(tile_row)
            mm1_psum.append(grp_row)

        mm2_psum_lo = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                grp_row.append(
                    nl.ndarray((Q_GRP_SZ, D_TILE), dtype=nl.float32, buffer=nl.psum)
                )
            mm2_psum_lo.append(grp_row)

        mm2_psum_hi = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                grp_row.append(
                    nl.ndarray((Q_GRP_SZ, D_TILE), dtype=nl.float32, buffer=nl.psum)
                )
            mm2_psum_hi.append(grp_row)

        # Load K and V (shared across Q heads in GQA)
        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * K_TILE_SZ
            num_k = min(seqlen_kv - k_start, K_TILE_SZ)
            k_offset = (
                batch_id * n_kv_heads * seqlen_kv * d
                + kv_head_id * seqlen_kv * d
                + k_start * d
            )
            nisa.dma_transpose(
                dst=k_sb_lo[k_idx].ap(
                    [[K_TILE_SZ, D_TILE], [1, 1], [1, 1], [1, num_k]]
                ),
                src=k.ap([[d, num_k], [1, 1], [1, 1], [1, D_TILE]], offset=k_offset),
            )
            nisa.dma_transpose(
                dst=k_sb_hi[k_idx].ap(
                    [[K_TILE_SZ, D_TILE], [1, 1], [1, 1], [1, num_k]]
                ),
                src=k.ap(
                    [[d, num_k], [1, 1], [1, 1], [1, D_TILE]], offset=k_offset + D_TILE
                ),
            )

        for v_idx in nl.affine_range(num_v_tiles):
            v_start = v_idx * V_TILE_SZ
            num_v = min(seqlen_kv - v_start, V_TILE_SZ)
            nisa.dma_copy(
                dst=v_sb[v_idx][:num_v, :D_HEAD],
                src=v[batch_id, kv_head_id, v_start : v_start + num_v, 0:D_HEAD],
            )

        nisa.memset(wb_zero_bias, value=0.0)

        # =====================================================================
        # Sequential execution (no pipelining for initial version — simpler, easier to debug)
        # Full pipeline can be added once correctness is validated.
        # =====================================================================
        for grp_i in range(num_grps):
            _pipe_load_q(
                grp_i,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
            )

            _pipe_qk_and_max(
                grp_i,
                q_sb_lo,
                q_sb_hi,
                k_sb_lo,
                k_sb_hi,
                mm1_masked,
                mm1_partial_max,
                mm1_psum,
                mm1_copy_sb,
                mm1_asel_sb,
                seqlen_q,
                seqlen_kv,
                scale,
                num_k_tiles,
                num_large_tiles,
                window_size,
            )

            _pipe_update_max(
                grp_i,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )

            _pipe_exp(
                grp_i,
                mm1_masked,
                mm1_running_max,
                exp_sb,
                exp_partial_sum,
                exp_tp_sb,
                seqlen_q,
                seqlen_kv,
                num_large_tiles,
                num_k_tiles,
            )

            _pipe_pv(
                grp_i,
                exp_tp_sb,
                v_sb,
                mm2_psum_lo,
                mm2_psum_hi,
                mm2_sb,
                seqlen_q,
                seqlen_kv,
                num_large_tiles,
                num_v_tiles,
            )

            _pipe_write_back(
                grp_i,
                mm2_sb,
                exp_partial_sum,
                exp_sum_recip,
                wb_exp_section_sum,
                wb_zero_bias,
                wb_o_bf16,
                o,
                seqlen_q,
                num_exp_tiles,
                batch_id,
                q_head_idx,
            )

    return o


# ============================================================================
# Unit test
# ============================================================================
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import time

    def reference_swa_attention(q, k, v, window_size):
        """CPU reference: sliding window attention.
        q(b,h,sq,d), k(b,h,sk,d), v(b,h,sk,d) -> (b,h,sq,d)
        """
        d = q.shape[3]
        q_t = q.float()
        k_t = k.float()
        v_t = v.float()
        scale = 1.0 / (d**0.5)
        attn = q_t @ k_t.transpose(-2, -1) * scale

        sq, sk = q_t.shape[2], k_t.shape[2]
        # Sliding window mask: position i attends to [i - window_size + 1, i]
        row_idx = torch.arange(sq).unsqueeze(1)  # (sq, 1)
        col_idx = torch.arange(sk).unsqueeze(0)  # (1, sk)
        # Mask out: future (col > row) OR too far past (col < row - window + 1)
        mask = (col_idx > row_idx) | (col_idx < row_idx - window_size + 1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return attn @ v_t

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    print("=" * 70)
    print("Flash Attention d=256 with Sliding Window Mask (E2B SWA)")
    print("=" * 70)

    tests = [
        {
            "seq": 512,
            "heads": 1,
            "kv_heads": 1,
            "window": 512,
            "label": "seq=512 w=512 1:1",
        },
        {
            "seq": 1024,
            "heads": 1,
            "kv_heads": 1,
            "window": 512,
            "label": "seq=1024 w=512 1:1",
        },
        {
            "seq": 1024,
            "heads": 8,
            "kv_heads": 1,
            "window": 512,
            "label": "seq=1024 w=512 GQA 8:1",
        },
        {
            "seq": 512,
            "heads": 8,
            "kv_heads": 1,
            "window": 512,
            "label": "seq=512 w=512 GQA 8:1",
        },
    ]

    for t in tests:
        seq_len = t["seq"]
        heads = t["heads"]
        kv_heads = t["kv_heads"]
        window = t["window"]
        d = 256
        print(f"\n=== Testing: {t['label']} ===")
        torch.manual_seed(42)
        q = torch.randn(1, heads, seq_len, d, dtype=torch.bfloat16)
        k = torch.randn(1, kv_heads, seq_len, d, dtype=torch.bfloat16)
        v = torch.randn(1, kv_heads, seq_len, d, dtype=torch.bfloat16)

        # CPU reference with GQA expansion
        ref_parts = []
        for h_idx in range(heads):
            kv_idx = h_idx // (heads // kv_heads)
            ref_h = reference_swa_attention(
                q[:, h_idx : h_idx + 1],
                k[:, kv_idx : kv_idx + 1],
                v[:, kv_idx : kv_idx + 1],
                window,
            )
            ref_parts.append(ref_h)
        ref = torch.cat(ref_parts, dim=1)

        # Run kernel
        q_dev = q.to(device)
        k_dev = k.to(device)
        v_dev = v.to(device)

        t0 = time.time()
        q_h_per_kv = heads // kv_heads
        out_parts = []
        for kv_h in range(kv_heads):
            q_slice = q_dev[:, kv_h * q_h_per_kv : (kv_h + 1) * q_h_per_kv, :, :]
            k_slice = k_dev[:, kv_h : kv_h + 1, :, :]
            v_slice = v_dev[:, kv_h : kv_h + 1, :, :]
            o_part = flash_attn_d256_swa(
                q_slice,
                k_slice,
                v_slice,
                q_h_per_k_h=q_h_per_kv,
                n_kv_heads=1,
                seqlen_q=seq_len,
                seqlen_kv=seq_len,
                window_size=window,
            )
            out_parts.append(o_part)
        out = torch.cat(out_parts, dim=1)
        xm.mark_step()
        out_cpu = out.cpu().float()
        t1 = time.time()

        cos_sim = F.cosine_similarity(
            ref.reshape(-1).unsqueeze(0), out_cpu.reshape(-1).unsqueeze(0)
        ).item()
        maxd = (ref - out_cpu).abs().max().item()
        print(f"  Time: {t1 - t0:.1f}s (includes compile)")
        print(f"  Cosine sim: {cos_sim:.6f}")
        print(f"  Max diff: {maxd:.6f}")
        print(f"  {'PASS' if cos_sim > 0.999 else 'FAIL'}")
