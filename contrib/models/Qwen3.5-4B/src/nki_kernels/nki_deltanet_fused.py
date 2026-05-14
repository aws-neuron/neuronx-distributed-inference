# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused single-kernel DeltaNet chunked forward for CTE (context encoding).

SSD-style architecture: processes ALL chunks for one (batch, head) pair in
a single NKI kernel call.  State (128x128) persists in SBUF across chunks —
no HBM round-trips for inter-chunk state propagation.

Key optimizations over nki_deltanet_chunked.py:
  1. Single kernel call per (B,H) instead of B*H*num_chunks calls
  2. State in SBUF across all chunks (no HBM state read/write per chunk)
  3. In-kernel cumsum via tensor_tensor_scan (no PyTorch cumsum)
  4. Masks and constants loaded once, reused across chunks
  5. Uses tensor_scalar for partition-broadcast (no explicit broadcast loops)
  6. nc_transpose (Vector Engine) for all 128x128 transposes instead of
     nc_matmul(moving=eye) (Tensor Engine) — frees TE for actual math
  7. Forms decay as exp(cumsum(g)_i - cumsum(g)_j), never as split
     exp(cumsum(g)_i) * exp(-cumsum(g)_j)

NKI 0.3.0 (SDK 2.29). k_dim = v_dim = 128 = P_MAX exactly.
Chunk size = 128 = P_MAX (one tile per chunk).

Mathematical framework:
  Per-chunk blocked triangular solve for intra-chunk correction:
    A = -QK_decay * lower_mask
    N = inv(I - A), computed by 64x64 forward substitution plus one
        64->128 lower-block merge
    value_corr = N @ v_beta
    k_cumdecay = N @ (k_beta * exp(gc))

  Inter-chunk state propagation:
    v_prime = k_cumdecay @ state
    v_new = value_corr - v_prime
    attn_inter = (q * exp(gc)) @ state
    attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    output = attn_inter + attn_intra @ v_new
    state = state * exp(g_last) + (k * exp(g_last - gc))^T @ v_new
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128  # Partition dim = chunk_size = k_dim = v_dim
CHUNK_SIZE = 128

# Broadcast partition 0 to all partitions in a 32-wide group
_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    """Strict lower triangular (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1)


def _make_lower_mask_diag():
    """Lower triangular with diagonal (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0)


def _make_identity():
    """Identity matrix (128x128) as numpy constant."""
    return np.eye(CHUNK_SIZE, dtype=np.float32)


@nki.jit
def deltanet_fused_chunked_fwd(
    query: nl.ndarray,  # (S, 128) float32 — l2-normed and scaled
    key: nl.ndarray,  # (S, 128) float32 — l2-normed
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 1)   float32 — per-token log-decay (NOT cumsum)
    beta_in: nl.ndarray,  # (S, 1)   float32 — per-token write gate
    lower_mask: nl.ndarray,  # (128, 128) float32 — strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 — identity
    lower_mask_diag: nl.ndarray,  # (128, 128) float32 — lower tri with diag
):
    """Fused chunked DeltaNet forward — single kernel call per (batch, head).

    Processes all chunks sequentially within the kernel, keeping the recurrent
    state (128x128) in SBUF across chunks.  Returns per-token output and
    final state.

    Input requirements:
      - S must be divisible by 128 (pad before calling)
      - query must be l2-normed and scaled by 1/sqrt(k_dim)
      - key must be l2-normed
      - g_in is RAW log-decay (cumsum computed in-kernel via tensor_tensor_scan)
      - beta_in is sigmoid(b) (write gate)

    Returns:
        output:      (S, 128) float32
        final_state: (128, 128) float32
    """
    seq_len = query.shape[0]
    dim = query.shape[1]  # 128
    num_chunks = seq_len // CHUNK_SIZE

    # Output tensors in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # ================================================================
    # Load constant masks into SBUF once (reused across all chunks)
    # ================================================================
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # Ones vector for cumsum scan: (1, CHUNK_SIZE)
    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    # Zero initial for cumsum scan
    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    # ================================================================
    # Initialize recurrent state in SBUF — persists across ALL chunks
    # ================================================================
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # ================================================================
    # Sequential chunk processing
    # ================================================================
    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK_SIZE

        # ---- Load chunk data from HBM ----
        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_c,
            src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_c,
            src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_c,
            src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        # g: (CHUNK_SIZE, 1) — raw log-decay per token
        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # beta: (CHUNK_SIZE, 1) — write gate scalar per token
        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # ---- In-kernel cumsum of g via tensor_tensor_scan ----
        # Need g as (1, CHUNK_SIZE) for scan along free dim.
        # Transpose: (CHUNK_SIZE, 1) -> (1, CHUNK_SIZE) via nc_transpose
        g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=g_padded, value=0.0)
        nisa.tensor_copy(
            dst=g_padded[0:CHUNK_SIZE, 0:1],
            src=g_chunk_p[0:CHUNK_SIZE, 0:1],
        )

        g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=g_row[0:1, 0:CHUNK_SIZE],
            src=g_tp_psum[0:1, 0:CHUNK_SIZE],
        )

        # cumsum: gc_row[t] = 1.0 * gc_row[t-1] + g_row[t]
        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        # Transpose gc back to (CHUNK_SIZE, 1) partition layout
        gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gc_padded, value=0.0)
        nisa.tensor_copy(
            dst=gc_padded[0:1, 0:CHUNK_SIZE],
            src=gc_row[0:1, 0:CHUNK_SIZE],
        )

        gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

        # gc_p: (P_MAX, 1) — cumulative sum of g per token in this chunk
        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gc_p[0:CHUNK_SIZE, 0:1],
            src=gc_tp_psum[0:CHUNK_SIZE, 0:1],
        )

        # g_last = gc[-1] (scalar) — needed for state decay
        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1],
            src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
        )

        # ---- Build stable decay factors from cumulative log-decay ----
        #
        # Pairwise decays are computed as exp(gc[i] - gc[j]) under the causal
        # mask.  The older split form exp(gc[i]) * exp(-gc[j]) is algebraically
        # equivalent, but can overflow/underflow before the multiply.
        #
        # The one-vector exp(gc) and exp(g_last) factors are still required by
        # the chunk recurrence; these are non-positive GDN decays and therefore
        # bounded above by one.

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        # g_last: scalar, then broadcast raw and exp(g_last) to (P_MAX, 1)
        gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_11,
            op=nl.exp,
            data=gl_11,
            bias=None,
            scale=1.0,
        )

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        # Broadcast gc row-wise so row i, column j can form gc[i] - gc[j].
        gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:P_MAX],
                dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        # Strict-lower decay for the KKT solve: exp(gc[i] - gc[j]) where i > j.
        # Mask before exp by zeroing non-causal differences, then mask again
        # after exp so exp(0) from non-causal positions does not contribute.
        gc_col_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_col_strict,
            data=Lmask,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        gc_row_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_row_strict, data1=gc_row_broadcast, data2=Lmask, op=nl.multiply
        )
        g_diff_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=g_diff_strict,
            data1=gc_col_strict,
            data2=gc_row_strict,
            op=nl.subtract,
        )
        decay_strict_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=decay_strict_raw,
            op=nl.exp,
            data=g_diff_strict,
            bias=None,
            scale=1.0,
        )
        decay_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_strict, data1=decay_strict_raw, data2=Lmask, op=nl.multiply
        )

        # Lower-with-diagonal decay for intra-chunk attention: exp(gc[i] - gc[j])
        # where i >= j.
        gc_col_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_col_diag,
            data=Lmask_d,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        gc_row_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_row_diag, data1=gc_row_broadcast, data2=Lmask_d, op=nl.multiply
        )
        g_diff_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=g_diff_diag,
            data1=gc_col_diag,
            data2=gc_row_diag,
            op=nl.subtract,
        )
        decay_diag_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=decay_diag_raw,
            op=nl.exp,
            data=g_diff_diag,
            bias=None,
            scale=1.0,
        )
        decay_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_diag, data1=decay_diag_raw, data2=Lmask_d, op=nl.multiply
        )

        # ============================================================
        # k_beta = K * beta, v_beta = V * beta
        # tensor_scalar broadcasts beta_p (P_MAX, 1) across free dim
        # ============================================================
        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v_beta,
            data=v_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        # ============================================================
        # Phase 1: Build A matrix (intra-chunk correction)
        # Transpose K and K_beta for matmul
        # ============================================================
        kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
        kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_c)
        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        # QK = k_beta^T @ k  (contract over features)
        QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
        QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK, src=QK_psum)

        # QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j]) for i > j.
        # This is the same causal decay as the split-exp form, but numerically
        # bounded by construction.
        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)

        # A = -QK_decay * lower_mask
        neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_QK_decay,
            data=QK_decay,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )
        A_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_mat, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

        # ============================================================
        # Stable triangular solve: N = inv(I - A_mat)
        #
        # A_mat is strictly lower triangular.  Solve two 64x64 diagonal
        # blocks row-by-row:
        #   N[i, :] = e_i + sum_{j<i} A_mat[i, j] * N[j, :]
        # then merge:
        #   N21 = N22 @ A21 @ N11
        #
        # This is mathematically equivalent to the triangular inverse but avoids
        # repeated squaring of A.
        # ============================================================
        P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=P_acc, value=0.0)

        A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=A_T_psum, data=A_mat)
        A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_T, src=A_T_psum)

        col_mask_left_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=col_mask_left_row, value=0.0)
        nisa.memset(dst=col_mask_left_row[0:1, 0:64], value=1.0)
        col_mask_left = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=col_mask_left_row[0:1, 0:P_MAX],
                dst=col_mask_left[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        col_mask_right_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=col_mask_right_row, value=0.0)
        nisa.memset(dst=col_mask_right_row[0:1, 64:P_MAX], value=1.0)
        col_mask_right = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=col_mask_right_row[0:1, 0:P_MAX],
                dst=col_mask_right[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        block_row_mask_bottom = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=block_row_mask_bottom[0:P_MAX, 0:1],
            src=Lmask_d[0:P_MAX, 64:65],
        )

        # Top-left block: N11 = inv(I - A11)
        for solve_i in nl.static_range(64):
            row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=P_acc)
            row_prod = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=row_prod, src=row_psum)

            row_with_eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_with_eye,
                data1=row_prod,
                data2=eye,
                op=nl.add,
            )

            row_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_col_masked,
                data1=row_with_eye,
                data2=col_mask_left,
                op=nl.multiply,
            )

            row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=row_mask[0:P_MAX, 0:1],
                src=eye[0:P_MAX, solve_i : solve_i + 1],
            )
            row_update = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=row_update,
                data=row_col_masked,
                op0=nl.multiply,
                operand0=row_mask,
                engine=nisa.vector_engine,
            )

            P_next = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=P_next,
                data1=P_acc,
                data2=row_update,
                op=nl.add,
            )
            nisa.tensor_copy(dst=P_acc, src=P_next)

        # Bottom-right block: N22 = inv(I - A22)
        for solve_i in nl.static_range(64):
            row_idx = 64 + solve_i

            row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=P_acc)
            row_prod = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=row_prod, src=row_psum)

            row_with_eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_with_eye,
                data1=row_prod,
                data2=eye,
                op=nl.add,
            )

            row_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_col_masked,
                data1=row_with_eye,
                data2=col_mask_right,
                op=nl.multiply,
            )

            row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=row_mask[0:P_MAX, 0:1],
                src=eye[0:P_MAX, row_idx : row_idx + 1],
            )
            row_update = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=row_update,
                data=row_col_masked,
                op0=nl.multiply,
                operand0=row_mask,
                engine=nisa.vector_engine,
            )

            P_next = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=P_next,
                data1=P_acc,
                data2=row_update,
                op=nl.add,
            )
            nisa.tensor_copy(dst=P_acc, src=P_next)

        # Merge lower-left block: N21 = N22 @ A21 @ N11.
        N_diag_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=N_diag_T_psum, data=P_acc)
        N_diag_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=N_diag_T, src=N_diag_T_psum)

        tmp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tmp_psum, stationary=N_diag_T, moving=A_mat)
        tmp = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp, src=tmp_psum)

        tmp_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tmp_T_psum, data=tmp)
        tmp_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_T, src=tmp_T_psum)

        N21_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=N21_psum, stationary=tmp_T, moving=P_acc)
        N21 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=N21, src=N21_psum)

        N21_col_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=N21_col_masked,
            data1=N21,
            data2=col_mask_left,
            op=nl.multiply,
        )
        N21_block = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=N21_block,
            data=N21_col_masked,
            op0=nl.multiply,
            operand0=block_row_mask_bottom,
            engine=nisa.vector_engine,
        )
        nisa.tensor_tensor(dst=P_acc, data1=P_acc, data2=N21_block, op=nl.add)

        # ============================================================
        # Apply N: value_corr = N @ v_beta
        #          k_cumdecay = N @ (k_beta * exp(gc))
        # ============================================================
        N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=N_T_psum, data=P_acc)
        N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=N_T, src=N_T_psum)

        vc_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta)
        value_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=value_corr, src=vc_psum)

        # k_beta * exp(gc): row-scaled
        kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kb_exp_gc,
            data=k_beta,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        kcd_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kcd_psum, stationary=N_T, moving=kb_exp_gc)
        k_cumdecay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_cumdecay, src=kcd_psum)

        # ============================================================
        # Phase 2: Inter-chunk state propagation
        # attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
        # ============================================================
        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_c)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        # attn_intra[i,j] = qk[i,j] * exp(gc[i] - gc[j]) for i >= j.
        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=attn_intra, data1=qk_raw, data2=decay_diag, op=nl.multiply)

        # ============================================================
        # v_prime = k_cumdecay @ state   (state is in SBUF!)
        # ============================================================
        kcd_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kcd_T_psum, data=k_cumdecay)
        kcd_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kcd_T, src=kcd_T_psum)

        vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=vp_psum, stationary=kcd_T, moving=state)
        v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_prime, src=vp_psum)

        v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_new, data1=value_corr, data2=v_prime, op=nl.subtract)

        # ============================================================
        # attn_inter = (q * exp(gc)) @ state   (state is in SBUF!)
        # ============================================================
        q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_exp,
            data=q_c,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qe_T_psum, data=q_exp)
        qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

        ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
        attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=attn_inter, src=ai_psum)

        # ============================================================
        # attn_intra @ v_new
        # ============================================================
        ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
        ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

        intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
        intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=intra_out, src=intra_psum)

        # ============================================================
        # chunk_output = attn_inter + intra_out
        # ============================================================
        chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

        # Store output chunk to HBM
        nisa.dma_copy(
            dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            src=chunk_out,
        )

        # ============================================================
        # State update: state = exp(g_last) * (state + k_raw_decay^T @ v_new)
        # state is updated IN-PLACE in SBUF — no HBM round-trip!
        # ============================================================

        # k_raw_decay contributes as k * exp(g_last - gc), with one bounded
        # exponential instead of exp(g_last) * exp(-gc).
        gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gl_minus_gc_p,
            data1=gl_p,
            data2=gc_p,
            op=nl.subtract,
        )
        exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_minus_gc_p,
            op=nl.exp,
            data=gl_minus_gc_p,
            bias=None,
            scale=1.0,
        )

        # k_raw_decay = k * exp(g_last - gc)
        k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_c,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc_p,
            engine=nisa.vector_engine,
        )

        # k_raw_decay^T @ v_new → (dim, dim) outer product sum
        # nc_matmul: result[M,N] = sum_K stationary[K,M] * moving[K,N]
        # stationary=k_raw_decay (P_MAX, dim), moving=v_new (P_MAX, dim)
        # Result: sum over tokens -> (dim, dim)
        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)
        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        # state = state * exp(g_last) + kv_outer
        # tensor_scalar broadcasts exp_gl_p (P_MAX, 1) across free dim.
        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )
        nisa.tensor_tensor(dst=state, data1=state_decayed, data2=kv_outer, op=nl.add)

    # ---- Write final state to HBM ----
    nisa.dma_copy(dst=final_state_out, src=state)

    return output, final_state_out
