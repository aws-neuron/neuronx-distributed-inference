"""
Flash attention for d=256 with deferred softmax, 3-stage software pipeline, GQA, causal mask.

NKI Beta 2 API (`import nki`) with compiler-managed buffer placement.
Called per (batch, kv_head) pair with pre-sliced tensors (like deltanet pattern).
Integrated via Beta 2 PyTorchXLAKernel in the model for SPMD-traced-context.
3-stage software pipelining (QK+max | exp+transpose | PV+writeback).

**BHSD layout for Q/K**: Q and K are accepted in standard BHSD (B, H, S, D)
layout. The kernel uses DMA transpose during load to convert to (D, S) layout
in SBUF. This avoids torch.permute() in the model's perform_prefill, which
creates XLA lazy tensors that the Beta 2 tracer cannot resolve in SPMD context.

Architecture follows xpu-perf v4 (nki_flash_attn_bf16_pipe_opt) adapted for:
  - head_dim=256 (2x128 QK tiling, split PV output)
  - GQA (multiple Q heads per KV head)
  - Causal masking (affine_select with pattern/offset)

Layouts (per-call, single batch + single kv_head):
  Q: (1, q_h_per_k_h, seq_q, 256)  -- BHSD (seq on partition, d on free in HBM)
  K: (1, 1, seq_k, 256)             -- BHSD (seq on partition, d on free in HBM)
  V: (1, 1, seq_v, 256)             -- BHSD (seq on partition, d on free)
  O: (1, q_h_per_k_h, seq_q, 256)   -- BHSD (seq on partition, d on free)

Internal SBUF layout after DMA transpose of Q/K:
  Q_sb: (D_TILE=128, Q_GRP_SZ=128)  -- d on partition, seq on free
  K_sb: (D_TILE=128, K_TILE_SZ=512)  -- d on partition, seq on free
  V_sb: (V_TILE_SZ=128, D_HEAD=256)  -- seq on partition, d on free

Pipeline stages (for grp_i in main loop):
  Stage 1 (grp_i+2): load_q + qk_and_max  -- DMA + TensorEngine (MM1)
  Stage 2 (grp_i+1): exp + dma_transpose   -- VectorEngine + DMA
  Stage 3 (grp_i):   pv + write_back       -- TensorEngine (MM2) + DMA

Run on trn2:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python3 nki_flash_attn_d256_pipe.py  # unit test

Uses Beta 2 NKI API (`import nki`). Q/K accepted in BHSD layout; DMA transpose
during load converts to (D, S) SBUF layout needed for QK matmul.
"""

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import math
import numpy as np
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

# Partial RoPE constants (Qwen3.5: only 25% of head_dim=256 gets rotary)
ROPE_DIM = 64  # rope_dim = head_dim * partial_rotary_factor
ROPE_HALF = 32  # rope_dim // 2 — each half for rotate_half


# ============================================================================
# ModularAllocator helpers (layout structure preserved, compiler-managed placement)
# ============================================================================


def _align32(addr):
    """Round up address to 32-byte alignment (required for DMA transpose)."""
    return (addr + 31) // 32 * 32


def _alloc_modular_1d(shape, dtype, block_dim, num_free_tiles, base_addr):
    """Allocate 1D modular buffer list: block_dim entries, num_free_tiles physical.

    Elements at indices i and j share memory if i % num_free_tiles == j % num_free_tiles.
    Returns (list_of_tensors, next_address).
    """
    base_addr = _align32(base_addr)
    tile_elems = 1
    for d in shape[1:]:
        tile_elems *= d
    dtype_size = 4 if dtype == nl.float32 else 2
    tile_bytes = _align32(tile_elems * dtype_size)

    tensors = []
    for i in range(block_dim):
        addr = base_addr + (i % num_free_tiles) * tile_bytes
        tensors.append(nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))
    next_addr = base_addr + num_free_tiles * tile_bytes
    return tensors, next_addr


def _alloc_modular_2d(
    shape, dtype, block_dim0, block_dim1, num_free0, num_free1, base_addr
):
    """Allocate 2D modular buffer: [block_dim0][block_dim1], with modular addressing.

    Returns (nested_list, next_address).
    """
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
            idx = (i % num_free0) * num_free1 + (j % num_free1)
            addr = base_addr + idx * tile_bytes
            row.append(nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))
        tensors.append(row)
    next_addr = base_addr + num_free0 * num_free1 * tile_bytes
    return tensors, next_addr


def _alloc_modular_3d(shape, dtype, dims, n_free, base_addr):
    """Allocate 3D modular buffer: [d0][d1][d2], with modular addressing.

    dims = (block_dim0, block_dim1, block_dim2)
    n_free = (num_free0, num_free1, num_free2)
    Returns (nested_list, next_address).
    """
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
                idx = (
                    (i % n_free[0]) * n_free[1] * n_free[2]
                    + (j % n_free[1]) * n_free[2]
                    + (k % n_free[2])
                )
                addr = base_addr + idx * tile_bytes
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
    fuse_rope=False,
    cos_lo_q_sb=None,
    cos_hi_q_sb=None,
    sin_q_sb=None,
    rope_q_x1=None,
    rope_q_x2=None,
    rope_q_res1=None,
    rope_q_res2=None,
    cos_cache_hbm=None,
    sin_cache_hbm=None,
    rope_dim=64,
):
    """Load Q group from BHSD HBM into SBUF with DMA transpose to (D, S) layout.
    Optionally applies partial RoPE to the first rope_dim rows of q_sb_lo.

    Q_HBM layout: (1, H, S, D=256) -- BHSD
    Q_SB layout:  (D_TILE=128, Q_GRP_SZ=128) -- D on partition, S on free
    DMA transpose: (S, D) in HBM -> (D, S) in SBUF
    """
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)

    # Compute flat offset into Q HBM: q[batch_id, q_head_idx, q_start, 0]
    q_offset = (
        batch_id * n_heads * seqlen_q * d_head
        + q_head_idx * seqlen_q * d_head
        + q_start * d_head
    )

    # Lo half: D[0:128]
    # Source HBM: BHSD layout, stride between S rows = d_head (not d_tile!)
    # ap() format: [(stride, size), ...] — dim0 stride must be d_head=256
    # Dest SBUF: (D_TILE=128, Q_GRP_SZ=128) — transposed
    nisa.dma_transpose(
        dst=q_sb_lo[grp_i].ap([[Q_GRP_SZ, d_tile], [1, 1], [1, 1], [1, num_q]]),
        src=q_hbm.ap(
            [[d_head, num_q], [1, 1], [1, 1], [1, d_tile]],
            offset=q_offset,
        ),
    )
    # Hi half: D[128:256] — same stride, offset shifted by d_tile
    nisa.dma_transpose(
        dst=q_sb_hi[grp_i].ap([[Q_GRP_SZ, d_tile], [1, 1], [1, 1], [1, num_q]]),
        src=q_hbm.ap(
            [[d_head, num_q], [1, 1], [1, 1], [1, d_tile]],
            offset=q_offset + d_tile,
        ),
    )

    # Apply partial RoPE to first rope_dim rows of q_sb_lo
    if fuse_rope:
        _load_rope_cos_sin_q(
            grp_i,
            cos_lo_q_sb,
            cos_hi_q_sb,
            sin_q_sb,
            cos_cache_hbm,
            sin_cache_hbm,
            seqlen_q,
            rope_dim,
        )
        _apply_rope_q_sbuf(
            grp_i,
            q_sb_lo,
            cos_lo_q_sb,
            cos_hi_q_sb,
            sin_q_sb,
            rope_q_x1,
            rope_q_x2,
            rope_q_res1,
            rope_q_res2,
            seqlen_q,
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
    use_causal_mask,
):
    """Compute QK^T (MM1) with d=256 tiling, scale, causal mask, and row-wise max.

    Always applies causal masking via the nki-library pattern:
      1. tensor_copy PSUM -> mm1_copy_sb (temp SBUF)
      2. affine_select with pattern/offset -> mm1_asel_sb
      3. tensor_scalar_reduce: scale + max -> mm1_masked + mm1_partial_max
    """
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)
    num_k_tiles_per_large = LARGE_TILE_SZ // K_TILE_SZ  # 4

    # Initialize partial max to -inf
    nisa.memset(mm1_partial_max[grp_i][...], value=FLOAT32_MIN)

    # Initialize mm1_masked to -inf so that causally-skipped K tiles
    # produce exp(-inf) = 0 in the softmax (no contribution).
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

            # Causal skip: entire Q group before this K tile
            q_last = q_start + num_q - 1
            if q_last < k_start:
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

            # Causal mask via affine_select (nki-library pattern)
            # val = (-1)*p + (1)*f + offset >= 0 means f <= p + offset
            # For causal: keep when k_pos <= q_pos, i.e., (k_start+f) <= (q_start+p)
            # i.e., f <= p + (q_start - k_start)
            # So offset = q_start - k_start
            nisa.affine_select(
                dst=mm1_asel_sb[:num_q, :num_k],
                pattern=[[-1, num_k]],
                offset=q_start - k_start,
                channel_multiplier=1,
                cmp_op=nl.greater_equal,
                on_true_tile=mm1_copy_sb[:num_q, :num_k],
                on_false_value=FLOAT32_MIN,
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

    # Section max with negate=True (stores -max for use as bias in exp)
    nisa.tensor_reduce(
        mm1_section_max[grp_i][:num_q, 0],
        nl.maximum,
        mm1_partial_max[grp_i][:num_q, :num_k_tiles],
        1,
        negate=True,
    )

    # For single-section: running_max = section_max
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

            # activation_reduce: dst = exp(1.0*data + bias), reduce_res = sum(dst)
            # bias = mm1_running_max (which is -max)
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

    # Zero the output accumulator
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

                # MM2 lo: exp_tp^T @ V[:, :128] = [Q_GRP, 128]
                nisa.nc_matmul(
                    psum_tile_lo[:num_q, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q)],
                    v_sb[v_tile_idx][:num_kv, :D_TILE],
                )
                # MM2 hi: exp_tp^T @ V[:, 128:256] = [Q_GRP, 128]
                nisa.nc_matmul(
                    psum_tile_hi[:num_q, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q)],
                    v_sb[v_tile_idx][:num_kv, nl.ds(D_TILE, D_TILE)],
                )

        # Accumulate large tile results into SBUF
        if large_tile_idx == 0:
            nisa.tensor_copy(
                mm2_sb[grp_i][:num_q, :D_TILE],
                psum_tile_lo[:num_q, :D_TILE],
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


def _pipe_fused_qkmax_and_pv(
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
    exp_tp_sb,
    v_sb,
    mm2_psum_lo,
    mm2_psum_hi,
    mm2_sb,
    seqlen_q,
    seqlen_kv,
    scale,
    num_k_tiles,
    num_large_tiles,
    num_v_tiles,
    use_causal_mask,
):
    """Fused: QK+max for grp_i+2, PV for grp_i (interleaved MM1+MM2)."""
    qkmax_grp = grp_i + 2
    pv_grp = grp_i

    q_start_pv = pv_grp * Q_GRP_SZ
    num_q_pv = min(seqlen_q - q_start_pv, Q_GRP_SZ)
    q_start_qk = qkmax_grp * Q_GRP_SZ
    num_q_qk = min(seqlen_q - q_start_qk, Q_GRP_SZ)

    num_k_tiles_per_large = LARGE_TILE_SZ // K_TILE_SZ  # 4
    num_mm2_grps_per_large = LARGE_TILE_SZ // K_TILE_SZ  # 4
    num_mm2_per_grp = K_TILE_SZ // V_TILE_SZ  # 4

    # Init partial max for QK grp
    nisa.memset(mm1_partial_max[qkmax_grp][...], value=FLOAT32_MIN)
    # Init mm1_masked for QK grp to -inf (causally-skipped K tiles → exp=0)
    for lt_idx in range(num_large_tiles):
        nisa.memset(mm1_masked[qkmax_grp][lt_idx][...], value=FLOAT32_MIN)
    # Init MM2 accumulator for PV grp
    nisa.memset(mm2_sb[pv_grp][...], value=0.0)

    for large_tile_idx in range(num_large_tiles):
        # --- PV for pv_grp ---
        psum_tile_pv_lo = mm2_psum_lo[pv_grp][large_tile_idx]
        psum_tile_pv_hi = mm2_psum_hi[pv_grp][large_tile_idx]

        for mm2_grp_i in range(num_mm2_grps_per_large):
            exp_tp_tile = exp_tp_sb[pv_grp][large_tile_idx][mm2_grp_i]

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

                nisa.nc_matmul(
                    psum_tile_pv_lo[:num_q_pv, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q_pv)],
                    v_sb[v_tile_idx][:num_kv, :D_TILE],
                )
                nisa.nc_matmul(
                    psum_tile_pv_hi[:num_q_pv, :D_TILE],
                    exp_tp_tile[:num_kv, nl.ds(mm2_i * V_TILE_SZ, num_q_pv)],
                    v_sb[v_tile_idx][:num_kv, nl.ds(D_TILE, D_TILE)],
                )

        # Accumulate PV large tile
        if large_tile_idx == 0:
            nisa.tensor_copy(
                mm2_sb[pv_grp][:num_q_pv, :D_TILE],
                psum_tile_pv_lo[:num_q_pv, :D_TILE],
            )
            nisa.tensor_copy(
                mm2_sb[pv_grp][:num_q_pv, nl.ds(D_TILE, D_TILE)],
                psum_tile_pv_hi[:num_q_pv, :D_TILE],
            )
        else:
            nisa.tensor_tensor(
                mm2_sb[pv_grp][:num_q_pv, :D_TILE],
                mm2_sb[pv_grp][:num_q_pv, :D_TILE],
                psum_tile_pv_lo[:num_q_pv, :D_TILE],
                nl.add,
            )
            nisa.tensor_tensor(
                mm2_sb[pv_grp][:num_q_pv, nl.ds(D_TILE, D_TILE)],
                mm2_sb[pv_grp][:num_q_pv, nl.ds(D_TILE, D_TILE)],
                psum_tile_pv_hi[:num_q_pv, :D_TILE],
                nl.add,
            )

        # --- QK+max for qkmax_grp ---
        for k_tile_local in range(num_k_tiles_per_large):
            k_tile_idx = large_tile_idx * num_k_tiles_per_large + k_tile_local
            if k_tile_idx >= num_k_tiles:
                continue

            k_start = k_tile_idx * K_TILE_SZ
            num_k = min(seqlen_kv - k_start, K_TILE_SZ)
            if num_k <= 0:
                continue

            q_last = q_start_qk + num_q_qk - 1
            if q_last < k_start:
                continue

            psum_tile_qk = mm1_psum[qkmax_grp][large_tile_idx][k_tile_local]

            # d=256 tiled QK: two nc_matmul calls
            nisa.nc_matmul(
                psum_tile_qk[:num_q_qk, :num_k],
                q_sb_lo[qkmax_grp][:D_TILE, :num_q_qk],
                k_sb_lo[k_tile_idx][:D_TILE, :num_k],
            )
            nisa.nc_matmul(
                psum_tile_qk[:num_q_qk, :num_k],
                q_sb_hi[qkmax_grp][:D_TILE, :num_q_qk],
                k_sb_hi[k_tile_idx][:D_TILE, :num_k],
            )

            nisa.tensor_copy(
                mm1_copy_sb[:num_q_qk, :num_k],
                psum_tile_qk[:num_q_qk, :num_k],
            )

            nisa.affine_select(
                dst=mm1_asel_sb[:num_q_qk, :num_k],
                pattern=[[-1, num_k]],
                offset=q_start_qk - k_start,
                channel_multiplier=1,
                cmp_op=nl.greater_equal,
                on_true_tile=mm1_copy_sb[:num_q_qk, :num_k],
                on_false_value=FLOAT32_MIN,
            )

            nisa.tensor_scalar_reduce(
                mm1_masked[qkmax_grp][large_tile_idx][
                    :num_q_qk, nl.ds(k_tile_local * K_TILE_SZ, num_k)
                ],
                data=mm1_asel_sb[:num_q_qk, :num_k],
                op0=nl.multiply,
                operand0=scale,
                reduce_op=nl.maximum,
                reduce_res=mm1_partial_max[qkmax_grp][:num_q_qk, k_tile_idx],
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

    # Reduce partial exp sums to get total
    nisa.tensor_reduce(
        wb_exp_section_sum[grp_i][:num_q, 0],
        nl.add,
        exp_partial_sum[grp_i][:num_q, :num_exp_tiles],
        axis=1,
    )

    # Reciprocal
    nisa.reciprocal(
        exp_sum_recip[grp_i][:num_q, 0],
        wb_exp_section_sum[grp_i][:num_q, 0],
    )

    # Scale output and cast to bf16: copy(recip * mm2_sb + zero_bias) -> bf16
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
# RoPE helper functions (partial RoPE fusion)
# ============================================================================


def _apply_rope_q_sbuf(
    grp_i,
    q_sb_lo,
    cos_lo_sb,
    cos_hi_sb,
    sin_sb,
    rope_x1,
    rope_x2,
    rope_res1,
    rope_res2,
    seqlen_q,
):
    """Apply partial RoPE to Q group in SBUF (transposed D,S layout).

    Only the first ROPE_DIM=64 partition rows of q_sb_lo get rotated.
    Rows 64:128 (and all of q_sb_hi) are unchanged.

    In the (D, S) layout:
      q_sb_lo[0:32, :]   = X1 (first half of rope dims)
      q_sb_lo[32:64, :]  = X2 (second half of rope dims)

    RoPE formula:
      result[0:32]  = X1 * cos_lo - X2 * sin
      result[32:64] = X2 * cos_hi + X1 * sin

    All tensor_tensor ops use operands at partition row 0 to satisfy
    NCC_IBIR297 (both SB inputs must have same partition base).
    Results are computed entirely in the rope workspace buffers, then
    copied to q_sb_lo.

    Buffers (all ROPE_HALF=32 partition rows, partition start 0):
      cos_lo_sb: (32, S) — cos[0:32]
      cos_hi_sb: (32, S) — cos[32:64]
      sin_sb:    (32, S) — sin[0:32] (symmetric)
      rope_x1:   (32, S) — X1 copy
      rope_x2:   (32, S) — X2 copy
      rope_res1: (32, S) — intermediate/result
      rope_res2: (32, S) — intermediate
    """
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)

    # Save original X1 and X2 into partition-0 aligned buffers
    nisa.tensor_copy(
        dst=rope_x1[:ROPE_HALF, :num_q],
        src=q_sb_lo[grp_i][:ROPE_HALF, :num_q],
    )
    nisa.tensor_copy(
        dst=rope_x2[:ROPE_HALF, :num_q],
        src=q_sb_lo[grp_i][nl.ds(ROPE_HALF, ROPE_HALF), :num_q],
    )

    # --- First half: q[0:32] = X1 * cos_lo - X2 * sin ---
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_q],
        data1=rope_x1[:ROPE_HALF, :num_q],
        data2=cos_lo_sb[:ROPE_HALF, :num_q],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res2[:ROPE_HALF, :num_q],
        data1=rope_x2[:ROPE_HALF, :num_q],
        data2=sin_sb[:ROPE_HALF, :num_q],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_q],
        data1=rope_res1[:ROPE_HALF, :num_q],
        data2=rope_res2[:ROPE_HALF, :num_q],
        op=nl.subtract,
    )
    nisa.tensor_copy(
        dst=q_sb_lo[grp_i][:ROPE_HALF, :num_q],
        src=rope_res1[:ROPE_HALF, :num_q],
    )

    # --- Second half: q[32:64] = X2 * cos_hi + X1 * sin ---
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_q],
        data1=rope_x2[:ROPE_HALF, :num_q],
        data2=cos_hi_sb[:ROPE_HALF, :num_q],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res2[:ROPE_HALF, :num_q],
        data1=rope_x1[:ROPE_HALF, :num_q],
        data2=sin_sb[:ROPE_HALF, :num_q],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_q],
        data1=rope_res1[:ROPE_HALF, :num_q],
        data2=rope_res2[:ROPE_HALF, :num_q],
        op=nl.add,
    )
    nisa.tensor_copy(
        dst=q_sb_lo[grp_i][nl.ds(ROPE_HALF, ROPE_HALF), :num_q],
        src=rope_res1[:ROPE_HALF, :num_q],
    )


def _apply_rope_k_sbuf(
    k_tile_idx,
    k_sb_lo,
    cos_lo_sb,
    cos_hi_sb,
    sin_sb,
    rope_x1,
    rope_x2,
    rope_res1,
    rope_res2,
    seqlen_kv,
):
    """Apply partial RoPE to a K tile in SBUF (transposed D,S layout).
    Same algorithm as Q, adapted for K_TILE_SZ free dim.
    """
    k_start = k_tile_idx * K_TILE_SZ
    num_k = min(seqlen_kv - k_start, K_TILE_SZ)

    nisa.tensor_copy(
        dst=rope_x1[:ROPE_HALF, :num_k],
        src=k_sb_lo[k_tile_idx][:ROPE_HALF, :num_k],
    )
    nisa.tensor_copy(
        dst=rope_x2[:ROPE_HALF, :num_k],
        src=k_sb_lo[k_tile_idx][nl.ds(ROPE_HALF, ROPE_HALF), :num_k],
    )

    # First half: k[0:32] = X1 * cos_lo - X2 * sin
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_k],
        data1=rope_x1[:ROPE_HALF, :num_k],
        data2=cos_lo_sb[:ROPE_HALF, :num_k],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res2[:ROPE_HALF, :num_k],
        data1=rope_x2[:ROPE_HALF, :num_k],
        data2=sin_sb[:ROPE_HALF, :num_k],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_k],
        data1=rope_res1[:ROPE_HALF, :num_k],
        data2=rope_res2[:ROPE_HALF, :num_k],
        op=nl.subtract,
    )
    nisa.tensor_copy(
        dst=k_sb_lo[k_tile_idx][:ROPE_HALF, :num_k],
        src=rope_res1[:ROPE_HALF, :num_k],
    )

    # Second half: k[32:64] = X2 * cos_hi + X1 * sin
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_k],
        data1=rope_x2[:ROPE_HALF, :num_k],
        data2=cos_hi_sb[:ROPE_HALF, :num_k],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res2[:ROPE_HALF, :num_k],
        data1=rope_x1[:ROPE_HALF, :num_k],
        data2=sin_sb[:ROPE_HALF, :num_k],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_res1[:ROPE_HALF, :num_k],
        data1=rope_res1[:ROPE_HALF, :num_k],
        data2=rope_res2[:ROPE_HALF, :num_k],
        op=nl.add,
    )
    nisa.tensor_copy(
        dst=k_sb_lo[k_tile_idx][nl.ds(ROPE_HALF, ROPE_HALF), :num_k],
        src=rope_res1[:ROPE_HALF, :num_k],
    )


def _load_rope_cos_sin_q(
    grp_i,
    cos_lo_sb,
    cos_hi_sb,
    sin_sb,
    cos_cache_hbm,
    sin_cache_hbm,
    seqlen_q,
    rope_dim,
):
    """Load cos/sin for a Q group from HBM into SBUF with DMA transpose.

    HBM layout: cos_cache (S, rope_dim=64) — per batch (batch squeezed)
    SBUF layout: split into lo/hi (ROPE_HALF=32, Q_GRP_SZ) each

    cos_lo_sb: rows 0:32 of cos (D[0:32])
    cos_hi_sb: rows 32:64 of cos (D[32:64])
    sin_sb: rows 0:32 of sin (only first half — symmetric)
    """
    q_start = grp_i * Q_GRP_SZ
    num_q = min(seqlen_q - q_start, Q_GRP_SZ)
    rope_half = rope_dim // 2

    # cos lo: DMA transpose first rope_half cols of cos → (32, S)
    cos_offset = q_start * rope_dim
    nisa.dma_transpose(
        dst=cos_lo_sb.ap([[Q_GRP_SZ, rope_half], [1, 1], [1, 1], [1, num_q]]),
        src=cos_cache_hbm.ap(
            [[rope_dim, num_q], [1, 1], [1, 1], [1, rope_half]],
            offset=cos_offset,
        ),
    )

    # cos hi: DMA transpose second rope_half cols of cos → (32, S)
    nisa.dma_transpose(
        dst=cos_hi_sb.ap([[Q_GRP_SZ, rope_half], [1, 1], [1, 1], [1, num_q]]),
        src=cos_cache_hbm.ap(
            [[rope_dim, num_q], [1, 1], [1, 1], [1, rope_half]],
            offset=cos_offset + rope_half,
        ),
    )

    # sin: DMA transpose first rope_half cols → (32, S)
    sin_offset = q_start * rope_dim
    nisa.dma_transpose(
        dst=sin_sb.ap([[Q_GRP_SZ, rope_half], [1, 1], [1, 1], [1, num_q]]),
        src=sin_cache_hbm.ap(
            [[rope_dim, num_q], [1, 1], [1, 1], [1, rope_half]],
            offset=sin_offset,
        ),
    )


def _load_rope_cos_sin_k(
    k_tile_idx,
    cos_lo_sb,
    cos_hi_sb,
    sin_sb,
    cos_cache_hbm,
    sin_cache_hbm,
    seqlen_kv,
    rope_dim,
):
    """Load cos/sin for a K tile from HBM into SBUF with DMA transpose.
    Same as Q version but for K_TILE_SZ=512 free dim.
    """
    k_start = k_tile_idx * K_TILE_SZ
    num_k = min(seqlen_kv - k_start, K_TILE_SZ)
    rope_half = rope_dim // 2

    cos_offset = k_start * rope_dim
    nisa.dma_transpose(
        dst=cos_lo_sb.ap([[K_TILE_SZ, rope_half], [1, 1], [1, 1], [1, num_k]]),
        src=cos_cache_hbm.ap(
            [[rope_dim, num_k], [1, 1], [1, 1], [1, rope_half]],
            offset=cos_offset,
        ),
    )
    nisa.dma_transpose(
        dst=cos_hi_sb.ap([[K_TILE_SZ, rope_half], [1, 1], [1, 1], [1, num_k]]),
        src=cos_cache_hbm.ap(
            [[rope_dim, num_k], [1, 1], [1, 1], [1, rope_half]],
            offset=cos_offset + rope_half,
        ),
    )

    sin_offset = k_start * rope_dim
    nisa.dma_transpose(
        dst=sin_sb.ap([[K_TILE_SZ, rope_half], [1, 1], [1, 1], [1, num_k]]),
        src=sin_cache_hbm.ap(
            [[rope_dim, num_k], [1, 1], [1, 1], [1, rope_half]],
            offset=sin_offset,
        ),
    )


# ============================================================================
# Main kernel
# ============================================================================


@nki.jit
def flash_attn_d256_pipe(
    q,
    k,
    v,
    cos_cache=None,
    sin_cache=None,
    use_causal_mask=True,
    q_h_per_k_h=4,
    n_kv_heads=1,
    seqlen_q=512,
    seqlen_kv=512,
    rope_dim=64,
):
    """
    Flash attention for head_dim=256, 3-stage software pipelined, with fused partial RoPE.

    Called per (batch, kv_head) pair with pre-sliced tensors (like deltanet).
    The caller loops over (B, kv_heads) and passes single-element slices.

    Q and K are accepted in BHSD layout (standard PyTorch layout), and the
    kernel transposes them internally via DMA during load. This avoids the
    need for torch.permute() in the caller, which would create XLA tensors
    incompatible with the Beta 2 tracer in SPMD context.

    When cos_cache and sin_cache are provided, the kernel applies partial RoPE
    (first rope_dim=64 of 256 head dims) internally to Q and K after loading
    them into SBUF. This bypasses the Beta 2 tracer None-args issue where
    element-wise ops with model buffers (cos/sin caches) cause Q/K to resolve
    as None during KLIR tracing. By passing cos/sin as separate HBM inputs
    (which are NOT derived from element-wise ops), we avoid the issue entirely.

    Args:
        q: (1, q_h_per_k_h, seq_q, 256)  -- bfloat16, BHSD, PRE-RoPE Q heads for one KV head
        k: (1, 1, seq_k, 256)             -- bfloat16, BHSD, PRE-RoPE single KV head
        v: (1, 1, seq_v, 256)             -- bfloat16, BHSD, single KV head (no RoPE)
        cos_cache: (seq, rope_dim=64)     -- bfloat16, cos values (batch dim squeezed)
        sin_cache: (seq, rope_dim=64)     -- bfloat16, sin values (batch dim squeezed)
        use_causal_mask: bool
        q_h_per_k_h: Q heads per KV head (explicit, avoids .shape)
        n_kv_heads: must be 1 (kernel processes one KV head at a time)
        seqlen_q: sequence length for Q
        seqlen_kv: sequence length for K/V
        rope_dim: number of head dims that get rotary embedding (default 64)

    Returns:
        o: (1, q_h_per_k_h, seq_q, 256)  -- bfloat16, BHSD (post-RoPE attention output)
    """
    d = D_HEAD
    n_heads = q_h_per_k_h * n_kv_heads
    bs = 1

    scale = 1.0 / math.sqrt(d)

    # Fixed indices — caller pre-slices tensors per (batch, kv_head)
    batch_id = 0
    kv_head_id = 0

    # Output allocation
    o = nl.ndarray((1, n_heads, seqlen_q, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    num_grps = (seqlen_q + Q_GRP_SZ - 1) // Q_GRP_SZ
    num_k_tiles = (seqlen_kv + K_TILE_SZ - 1) // K_TILE_SZ
    num_v_tiles = (seqlen_kv + V_TILE_SZ - 1) // V_TILE_SZ
    num_large_tiles = (seqlen_kv + LARGE_TILE_SZ - 1) // LARGE_TILE_SZ
    num_exp_per_large = LARGE_TILE_SZ // EXP_TILE_SZ  # 4
    num_exp_tiles = num_large_tiles * num_exp_per_large

    # =========================================================================
    # Buffer Allocation (ModularAllocator-style)
    # =========================================================================
    sca = 0  # SBUF current address

    # K lo buffers: [128, K_TILE_SZ=512] x num_k_tiles (all loaded)
    k_sb_lo, sca = _alloc_modular_1d(
        (D_TILE, K_TILE_SZ),
        nl.bfloat16,
        block_dim=num_k_tiles,
        num_free_tiles=num_k_tiles,
        base_addr=sca,
    )
    # K hi buffers: [128, K_TILE_SZ=512] x num_k_tiles (all loaded)
    k_sb_hi, sca = _alloc_modular_1d(
        (D_TILE, K_TILE_SZ),
        nl.bfloat16,
        block_dim=num_k_tiles,
        num_free_tiles=num_k_tiles,
        base_addr=sca,
    )

    # V buffers: [V_TILE_SZ=128, D_HEAD=256] x num_v_tiles (all loaded)
    v_sb, sca = _alloc_modular_1d(
        (V_TILE_SZ, D_HEAD),
        nl.bfloat16,
        block_dim=num_v_tiles,
        num_free_tiles=num_v_tiles,
        base_addr=sca,
    )

    # Q lo buffers: [128, Q_GRP_SZ=128] x num_grps (modular 2)
    q_sb_lo, sca = _alloc_modular_1d(
        (D_TILE, Q_GRP_SZ),
        nl.bfloat16,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )
    # Q hi buffers: [128, Q_GRP_SZ=128] x num_grps (modular 2)
    q_sb_hi, sca = _alloc_modular_1d(
        (D_TILE, Q_GRP_SZ),
        nl.bfloat16,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # Causal masking temp buffers (shared, reused per K tile)
    # mm1_copy_sb: [Q_GRP_SZ, K_TILE_SZ=512] -- PSUM copy for masking
    sca = _align32(sca)
    mm1_copy_sb = nl.ndarray(
        (Q_GRP_SZ, K_TILE_SZ),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    sca += K_TILE_SZ * 4  # f32

    # mm1_asel_sb: [Q_GRP_SZ, K_TILE_SZ=512] -- affine_select output
    sca = _align32(sca)
    mm1_asel_sb = nl.ndarray(
        (Q_GRP_SZ, K_TILE_SZ),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    sca += K_TILE_SZ * 4  # f32

    # mm1_masked: [Q_GRP_SZ, LARGE_TILE_SZ=2048] x [num_grps, num_large_tiles]
    mm1_masked, sca = _alloc_modular_2d(
        (Q_GRP_SZ, LARGE_TILE_SZ),
        nl.float32,
        num_grps,
        num_large_tiles,
        2,
        num_large_tiles,
        sca,
    )

    # mm1_partial_max: [Q_GRP_SZ, num_k_tiles] x num_grps (modular 2)
    mm1_partial_max, sca = _alloc_modular_1d(
        (Q_GRP_SZ, num_k_tiles),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # mm1_section_max: [Q_GRP_SZ, 1] x num_grps (modular 2)
    mm1_section_max, sca = _alloc_modular_1d(
        (Q_GRP_SZ, 1),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # mm1_running_max: [Q_GRP_SZ, num_grps] -- persistent
    sca = _align32(sca)
    mm1_running_max = nl.ndarray(
        (Q_GRP_SZ, num_grps),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    sca += num_grps * 4

    # exp_sb: [Q_GRP_SZ, LARGE_TILE_SZ] x [num_grps, num_large_tiles]
    exp_sb, sca = _alloc_modular_2d(
        (Q_GRP_SZ, LARGE_TILE_SZ),
        nl.bfloat16,
        num_grps,
        num_large_tiles,
        1,
        num_large_tiles,
        sca,
    )

    # exp_partial_sum: [Q_GRP_SZ, num_exp_tiles] x num_grps (modular 2)
    exp_partial_sum, sca = _alloc_modular_1d(
        (Q_GRP_SZ, num_exp_tiles),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # exp_tp_sb: [V_TILE_SZ=128, K_TILE_SZ=512] x [grps, large, exp_per_large]
    exp_tp_sb, sca = _alloc_modular_3d(
        (V_TILE_SZ, K_TILE_SZ),
        nl.bfloat16,
        (num_grps, num_large_tiles, num_exp_per_large),
        (2, num_large_tiles, num_exp_per_large),
        sca,
    )

    # mm2_sb: [Q_GRP_SZ, D_HEAD=256] x num_grps (modular 2)
    mm2_sb, sca = _alloc_modular_1d(
        (Q_GRP_SZ, D_HEAD),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # exp_sum_recip: [Q_GRP_SZ, 1] x num_grps (modular 2)
    exp_sum_recip, sca = _alloc_modular_1d(
        (Q_GRP_SZ, 1),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # Write-back buffers
    wb_exp_section_sum, sca = _alloc_modular_1d(
        (Q_GRP_SZ, 1),
        nl.float32,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )
    sca = _align32(sca)
    wb_zero_bias = nl.ndarray(
        (Q_GRP_SZ, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    sca += 1 * 4
    wb_o_bf16, sca = _alloc_modular_1d(
        (Q_GRP_SZ, D_HEAD),
        nl.bfloat16,
        block_dim=num_grps,
        num_free_tiles=2,
        base_addr=sca,
    )

    # =========================================================================
    # RoPE SBUF buffers (only allocated when cos_cache/sin_cache provided)
    # All ROPE_HALF=32 partition rows starting at 0 to satisfy NCC_IBIR297.
    # =========================================================================
    fuse_rope = cos_cache is not None and sin_cache is not None

    if fuse_rope:
        rope_half = rope_dim // 2  # 32

        # Q RoPE buffers: all (ROPE_HALF=32, Q_GRP_SZ=128) bf16
        sca = _align32(sca)
        cos_lo_q_sb = nl.ndarray(
            (ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += Q_GRP_SZ * 2
        cos_hi_q_sb = nl.ndarray(
            (ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += Q_GRP_SZ * 2
        sin_q_sb = nl.ndarray((ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
        sca += Q_GRP_SZ * 2
        rope_q_x1 = nl.ndarray((ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
        sca += Q_GRP_SZ * 2
        rope_q_x2 = nl.ndarray((ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
        sca += Q_GRP_SZ * 2
        rope_q_res1 = nl.ndarray(
            (ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += Q_GRP_SZ * 2
        rope_q_res2 = nl.ndarray(
            (ROPE_HALF, Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += Q_GRP_SZ * 2

        # K RoPE buffers: all (ROPE_HALF=32, K_TILE_SZ=512) bf16
        # Reused across K tiles (sequential processing)
        sca = _align32(sca)
        cos_lo_k_sb = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
        cos_hi_k_sb = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
        sin_k_sb = nl.ndarray((ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf)
        sca += K_TILE_SZ * 2
        rope_k_x1 = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
        rope_k_x2 = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
        rope_k_res1 = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
        rope_k_res2 = nl.ndarray(
            (ROPE_HALF, K_TILE_SZ), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        sca += K_TILE_SZ * 2
    else:
        cos_lo_q_sb = cos_hi_q_sb = sin_q_sb = None
        rope_q_x1 = rope_q_x2 = rope_q_res1 = rope_q_res2 = None
        cos_lo_k_sb = cos_hi_k_sb = sin_k_sb = None
        rope_k_x1 = rope_k_x2 = rope_k_res1 = rope_k_res2 = None

    # =========================================================================
    # GQA outer loop: iterate over Q heads sharing this KV head
    # =========================================================================
    for i_q_h in range(q_h_per_k_h):
        q_head_idx = kv_head_id * q_h_per_k_h + i_q_h

        # PSUM allocations (compiler-managed placement)
        # Allocated per-GQA-iteration to avoid NCC_ISCH715 accumulation conflicts
        # MM1 PSUM: QK matmul results
        mm1_psum = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                tile_row = []
                for kt_idx in range(4):
                    tile = nl.ndarray(
                        (Q_GRP_SZ, PSUM_FMAX),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    tile_row.append(tile)
                grp_row.append(tile_row)
            mm1_psum.append(grp_row)

        # MM2 PSUM lo: PV result for d[0:128]
        mm2_psum_lo = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                tile = nl.ndarray(
                    (Q_GRP_SZ, D_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                grp_row.append(tile)
            mm2_psum_lo.append(grp_row)

        # MM2 PSUM hi: PV result for d[128:256]
        mm2_psum_hi = []
        for grp_idx in range(num_grps):
            grp_row = []
            for lt_idx in range(num_large_tiles):
                tile = nl.ndarray(
                    (Q_GRP_SZ, D_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                grp_row.append(tile)
            mm2_psum_hi.append(grp_row)

        # Load K and V (shared across Q heads in same GQA group)
        # K is BHSD: (1, 1, S, D=256). DMA transpose to (D=128, S=512) in SBUF.
        # Flat offset: k[batch_id, kv_head_id, k_start, 0]
        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * K_TILE_SZ
            num_k = min(seqlen_kv - k_start, K_TILE_SZ)
            k_offset = (
                batch_id * n_kv_heads * seqlen_kv * d
                + kv_head_id * seqlen_kv * d
                + k_start * d
            )
            # Lo half: D[0:128], transpose (S=num_k, D=128) -> (D=128, S=num_k)
            # ap() dim0 stride must be d=256 (full head dim), not D_TILE=128
            nisa.dma_transpose(
                dst=k_sb_lo[k_idx].ap(
                    [[K_TILE_SZ, D_TILE], [1, 1], [1, 1], [1, num_k]]
                ),
                src=k.ap(
                    [[d, num_k], [1, 1], [1, 1], [1, D_TILE]],
                    offset=k_offset,
                ),
            )
            # Hi half: D[128:256], transpose (S=num_k, D=128) -> (D=128, S=num_k)
            nisa.dma_transpose(
                dst=k_sb_hi[k_idx].ap(
                    [[K_TILE_SZ, D_TILE], [1, 1], [1, 1], [1, num_k]]
                ),
                src=k.ap(
                    [[d, num_k], [1, 1], [1, 1], [1, D_TILE]],
                    offset=k_offset + D_TILE,
                ),
            )

        for v_idx in nl.affine_range(num_v_tiles):
            v_start = v_idx * V_TILE_SZ
            num_v = min(seqlen_kv - v_start, V_TILE_SZ)
            nisa.dma_copy(
                dst=v_sb[v_idx][:num_v, :D_HEAD],
                src=v[batch_id, kv_head_id, v_start : v_start + num_v, 0:D_HEAD],
            )

        # Apply RoPE to K tiles (sequential — shared cos/sin/tmp buffers)
        if fuse_rope:
            for k_idx in nl.sequential_range(num_k_tiles):
                _load_rope_cos_sin_k(
                    k_idx,
                    cos_lo_k_sb,
                    cos_hi_k_sb,
                    sin_k_sb,
                    cos_cache,
                    sin_cache,
                    seqlen_kv,
                    rope_dim,
                )
                _apply_rope_k_sbuf(
                    k_idx,
                    k_sb_lo,
                    cos_lo_k_sb,
                    cos_hi_k_sb,
                    sin_k_sb,
                    rope_k_x1,
                    rope_k_x2,
                    rope_k_res1,
                    rope_k_res2,
                    seqlen_kv,
                )

        # Zero the bias buffer once
        nisa.memset(wb_zero_bias, value=0.0)

        # =====================================================================
        # Software Pipeline
        # =====================================================================
        if num_grps <= 1:
            # Single group -- no pipelining
            _pipe_load_q(
                0,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
                fuse_rope=fuse_rope,
                cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                sin_q_sb=sin_q_sb if fuse_rope else None,
                rope_q_x1=rope_q_x1 if fuse_rope else None,
                rope_q_x2=rope_q_x2 if fuse_rope else None,
                rope_q_res1=rope_q_res1 if fuse_rope else None,
                rope_q_res2=rope_q_res2 if fuse_rope else None,
                cos_cache_hbm=cos_cache if fuse_rope else None,
                sin_cache_hbm=sin_cache if fuse_rope else None,
                rope_dim=rope_dim,
            )
            _pipe_qk_and_max(
                0,
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
                use_causal_mask,
            )
            _pipe_update_max(
                0,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )
            _pipe_exp(
                0,
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
                0,
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
                0,
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

        elif num_grps == 2:
            # Two groups -- partial pipelining
            _pipe_load_q(
                0,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
                fuse_rope=fuse_rope,
                cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                sin_q_sb=sin_q_sb if fuse_rope else None,
                rope_q_x1=rope_q_x1 if fuse_rope else None,
                rope_q_x2=rope_q_x2 if fuse_rope else None,
                rope_q_res1=rope_q_res1 if fuse_rope else None,
                rope_q_res2=rope_q_res2 if fuse_rope else None,
                cos_cache_hbm=cos_cache if fuse_rope else None,
                sin_cache_hbm=sin_cache if fuse_rope else None,
                rope_dim=rope_dim,
            )
            _pipe_qk_and_max(
                0,
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
                use_causal_mask,
            )
            _pipe_update_max(
                0,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )
            _pipe_exp(
                0,
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

            _pipe_load_q(
                1,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
                fuse_rope=fuse_rope,
                cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                sin_q_sb=sin_q_sb if fuse_rope else None,
                rope_q_x1=rope_q_x1 if fuse_rope else None,
                rope_q_x2=rope_q_x2 if fuse_rope else None,
                rope_q_res1=rope_q_res1 if fuse_rope else None,
                rope_q_res2=rope_q_res2 if fuse_rope else None,
                cos_cache_hbm=cos_cache if fuse_rope else None,
                sin_cache_hbm=sin_cache if fuse_rope else None,
                rope_dim=rope_dim,
            )
            _pipe_qk_and_max(
                1,
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
                use_causal_mask,
            )
            _pipe_update_max(
                1,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )

            _pipe_pv(
                0,
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
                0,
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

            _pipe_exp(
                1,
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
                1,
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
                1,
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

        else:
            # Full 3-stage pipelining (num_grps >= 3)

            # Prologue: prime groups 0 and 1
            _pipe_load_q(
                0,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
                fuse_rope=fuse_rope,
                cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                sin_q_sb=sin_q_sb if fuse_rope else None,
                rope_q_x1=rope_q_x1 if fuse_rope else None,
                rope_q_x2=rope_q_x2 if fuse_rope else None,
                rope_q_res1=rope_q_res1 if fuse_rope else None,
                rope_q_res2=rope_q_res2 if fuse_rope else None,
                cos_cache_hbm=cos_cache if fuse_rope else None,
                sin_cache_hbm=sin_cache if fuse_rope else None,
                rope_dim=rope_dim,
            )
            _pipe_qk_and_max(
                0,
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
                use_causal_mask,
            )
            _pipe_update_max(
                0,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )
            _pipe_exp(
                0,
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

            _pipe_load_q(
                1,
                q_sb_lo,
                q_sb_hi,
                q,
                D_TILE,
                seqlen_q,
                batch_id,
                q_head_idx,
                n_heads,
                d,
                fuse_rope=fuse_rope,
                cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                sin_q_sb=sin_q_sb if fuse_rope else None,
                rope_q_x1=rope_q_x1 if fuse_rope else None,
                rope_q_x2=rope_q_x2 if fuse_rope else None,
                rope_q_res1=rope_q_res1 if fuse_rope else None,
                rope_q_res2=rope_q_res2 if fuse_rope else None,
                cos_cache_hbm=cos_cache if fuse_rope else None,
                sin_cache_hbm=sin_cache if fuse_rope else None,
                rope_dim=rope_dim,
            )
            _pipe_qk_and_max(
                1,
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
                use_causal_mask,
            )
            _pipe_update_max(
                1,
                mm1_partial_max,
                mm1_section_max,
                mm1_running_max,
                num_k_tiles,
                seqlen_q,
            )

            # Main pipelined loop
            for grp_i in range(0, num_grps - 2):
                _pipe_load_q(
                    grp_i + 2,
                    q_sb_lo,
                    q_sb_hi,
                    q,
                    D_TILE,
                    seqlen_q,
                    batch_id,
                    q_head_idx,
                    n_heads,
                    d,
                    fuse_rope=fuse_rope,
                    cos_lo_q_sb=cos_lo_q_sb if fuse_rope else None,
                    cos_hi_q_sb=cos_hi_q_sb if fuse_rope else None,
                    sin_q_sb=sin_q_sb if fuse_rope else None,
                    rope_q_x1=rope_q_x1 if fuse_rope else None,
                    rope_q_x2=rope_q_x2 if fuse_rope else None,
                    rope_q_res1=rope_q_res1 if fuse_rope else None,
                    rope_q_res2=rope_q_res2 if fuse_rope else None,
                    cos_cache_hbm=cos_cache if fuse_rope else None,
                    sin_cache_hbm=sin_cache if fuse_rope else None,
                    rope_dim=rope_dim,
                )
                _pipe_exp(
                    grp_i + 1,
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
                _pipe_fused_qkmax_and_pv(
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
                    exp_tp_sb,
                    v_sb,
                    mm2_psum_lo,
                    mm2_psum_hi,
                    mm2_sb,
                    seqlen_q,
                    seqlen_kv,
                    scale,
                    num_k_tiles,
                    num_large_tiles,
                    num_v_tiles,
                    use_causal_mask,
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
                _pipe_update_max(
                    grp_i + 2,
                    mm1_partial_max,
                    mm1_section_max,
                    mm1_running_max,
                    num_k_tiles,
                    seqlen_q,
                )

            # Epilogue: drain last 2 groups
            _pipe_pv(
                num_grps - 2,
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
                num_grps - 2,
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

            _pipe_exp(
                num_grps - 1,
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
                num_grps - 1,
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
                num_grps - 1,
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

    def reference_causal_attention(q, k, v):
        """CPU reference: q(b,h,sq,d), k(b,h,sk,d), v(b,h,sk,d) -> (b,h,sq,d)

        All inputs in BHSD layout.
        """
        d = q.shape[3]
        q_t = q.float()
        k_t = k.float()
        v_t = v.float()
        scale = 1.0 / (d**0.5)
        attn = q_t @ k_t.transpose(-2, -1) * scale
        mask = torch.triu(
            torch.ones(q_t.shape[2], k_t.shape[2], dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return attn @ v_t

    def rotate_half(x):
        """Standard rotate_half for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_partial_rope(q, k, cos, sin, rope_dim=64):
        """Apply partial RoPE to Q and K (only first rope_dim dimensions).

        q: (B, H, S, D=256) - BHSD
        k: (B, Hkv, S, D=256) - BHSD
        cos: (S, rope_dim=64)
        sin: (S, rope_dim=64)
        Returns: post-RoPE q, k with same shape
        """
        # Expand cos/sin to broadcast: (1, 1, S, rope_dim)
        cos_exp = cos.unsqueeze(0).unsqueeze(0)
        sin_exp = sin.unsqueeze(0).unsqueeze(0)

        # Split rope/pass-through portions
        q_rope = q[..., :rope_dim]
        q_pass = q[..., rope_dim:]
        k_rope = k[..., :rope_dim]
        k_pass = k[..., rope_dim:]

        # Apply RoPE
        q_rope = q_rope * cos_exp + rotate_half(q_rope) * sin_exp
        k_rope = k_rope * cos_exp + rotate_half(k_rope) * sin_exp

        # Reassemble
        q_out = torch.cat([q_rope, q_pass], dim=-1)
        k_out = torch.cat([k_rope, k_pass], dim=-1)
        return q_out, k_out

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    # ====================================================================
    # Part 1: Original tests (post-RoPE inputs, no cos/sin — backward compat)
    # ====================================================================
    print("=" * 70)
    print("PART 1: Post-RoPE inputs (backward compatible, no fused RoPE)")
    print("=" * 70)

    tests_basic = [
        {"seq": 512, "bs": 1, "heads": 1, "kv_heads": 1, "label": "seq=512 1:1"},
        {"seq": 1024, "bs": 1, "heads": 1, "kv_heads": 1, "label": "seq=1024 1:1"},
        {"seq": 512, "bs": 1, "heads": 4, "kv_heads": 1, "label": "seq=512 GQA 4:1"},
        {"seq": 1024, "bs": 1, "heads": 4, "kv_heads": 1, "label": "seq=1024 GQA 4:1"},
    ]

    for t in tests_basic:
        seq_len = t["seq"]
        bs = t["bs"]
        heads = t["heads"]
        kv_heads = t["kv_heads"]
        d = 256
        print(f"\n=== Testing: {t['label']} ===")
        torch.manual_seed(42)
        q = torch.randn(bs, heads, seq_len, d, dtype=torch.bfloat16)
        k = torch.randn(bs, kv_heads, seq_len, d, dtype=torch.bfloat16)
        v = torch.randn(bs, kv_heads, seq_len, d, dtype=torch.bfloat16)

        ref_parts = []
        for h_idx in range(heads):
            kv_idx = h_idx // (heads // kv_heads)
            ref_h = reference_causal_attention(
                q[:, h_idx : h_idx + 1],
                k[:, kv_idx : kv_idx + 1],
                v[:, kv_idx : kv_idx + 1],
            )
            ref_parts.append(ref_h)
        ref = torch.cat(ref_parts, dim=1)

        q_dev = q.to(device)
        k_dev = k.to(device)
        v_dev = v.to(device)
        t0 = time.time()
        q_h_per_kv = heads // kv_heads
        out_parts = []
        for b in range(bs):
            for kv_h in range(kv_heads):
                q_slice = q_dev[
                    b : b + 1, kv_h * q_h_per_kv : (kv_h + 1) * q_h_per_kv, :, :
                ]
                k_slice = k_dev[b : b + 1, kv_h : kv_h + 1, :, :]
                v_slice = v_dev[b : b + 1, kv_h : kv_h + 1, :, :]
                o_part = flash_attn_d256_pipe(
                    q_slice,
                    k_slice,
                    v_slice,
                    use_causal_mask=True,
                    q_h_per_k_h=q_h_per_kv,
                    n_kv_heads=1,
                    seqlen_q=seq_len,
                    seqlen_kv=seq_len,
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

    # ====================================================================
    # Part 2: Fused RoPE tests (pre-RoPE inputs + cos/sin caches)
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 2: Fused RoPE (pre-RoPE inputs + cos/sin caches)")
    print("=" * 70)

    rope_dim = 64

    tests_rope = [
        {"seq": 512, "bs": 1, "heads": 1, "kv_heads": 1, "label": "ROPE seq=512 1:1"},
        {"seq": 1024, "bs": 1, "heads": 1, "kv_heads": 1, "label": "ROPE seq=1024 1:1"},
        {
            "seq": 512,
            "bs": 1,
            "heads": 4,
            "kv_heads": 1,
            "label": "ROPE seq=512 GQA 4:1",
        },
        {
            "seq": 1024,
            "bs": 1,
            "heads": 4,
            "kv_heads": 1,
            "label": "ROPE seq=1024 GQA 4:1",
        },
    ]

    for t in tests_rope:
        seq_len = t["seq"]
        bs = t["bs"]
        heads = t["heads"]
        kv_heads = t["kv_heads"]
        d = 256
        print(f"\n=== Testing: {t['label']} ===")
        torch.manual_seed(42)

        # Generate PRE-RoPE Q and K
        q_pre = torch.randn(bs, heads, seq_len, d, dtype=torch.bfloat16)
        k_pre = torch.randn(bs, kv_heads, seq_len, d, dtype=torch.bfloat16)
        v = torch.randn(bs, kv_heads, seq_len, d, dtype=torch.bfloat16)

        # Generate cos/sin caches: (S, rope_dim=64)
        # Use realistic RoPE frequencies (theta=10M, rope_dim=64)
        inv_freq = 1.0 / (
            10_000_000 ** (torch.arange(0, rope_dim, 2).float() / rope_dim)
        )
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (S, rope_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (S, rope_dim)
        cos_cache = emb.cos().to(torch.bfloat16)  # (S, 64)
        sin_cache = emb.sin().to(torch.bfloat16)  # (S, 64)

        # Apply partial RoPE on CPU to get post-RoPE Q, K (reference)
        q_post, k_post = apply_partial_rope(
            q_pre, k_pre, cos_cache, sin_cache, rope_dim
        )

        # CPU reference attention with post-RoPE Q, K
        ref_parts = []
        for h_idx in range(heads):
            kv_idx = h_idx // (heads // kv_heads)
            ref_h = reference_causal_attention(
                q_post[:, h_idx : h_idx + 1],
                k_post[:, kv_idx : kv_idx + 1],
                v[:, kv_idx : kv_idx + 1],
            )
            ref_parts.append(ref_h)
        ref = torch.cat(ref_parts, dim=1)

        # Run kernel with PRE-RoPE inputs + cos/sin
        q_dev = q_pre.to(device)
        k_dev = k_pre.to(device)
        v_dev = v.to(device)
        cos_dev = cos_cache.to(device)
        sin_dev = sin_cache.to(device)

        t0 = time.time()
        q_h_per_kv = heads // kv_heads
        out_parts = []
        for b in range(bs):
            for kv_h in range(kv_heads):
                q_slice = q_dev[
                    b : b + 1, kv_h * q_h_per_kv : (kv_h + 1) * q_h_per_kv, :, :
                ]
                k_slice = k_dev[b : b + 1, kv_h : kv_h + 1, :, :]
                v_slice = v_dev[b : b + 1, kv_h : kv_h + 1, :, :]
                o_part = flash_attn_d256_pipe(
                    q_slice,
                    k_slice,
                    v_slice,
                    cos_cache=cos_dev,
                    sin_cache=sin_dev,
                    use_causal_mask=True,
                    q_h_per_k_h=q_h_per_kv,
                    n_kv_heads=1,
                    seqlen_q=seq_len,
                    seqlen_kv=seq_len,
                    rope_dim=rope_dim,
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
