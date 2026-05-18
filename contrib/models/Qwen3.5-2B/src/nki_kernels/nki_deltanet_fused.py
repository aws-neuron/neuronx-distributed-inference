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

Numerical stability:
  - Decay mask uses exp(gc[i] - gc[j]) computed as a SINGLE exp of the
    difference, NOT exp(gc[i]) * exp(-gc[j]).  Since gc is monotonically
    decreasing (all g < 0) and the mask is lower-triangular (i >= j),
    gc[i] - gc[j] <= 0, so exp(gc[i]-gc[j]) in (0, 1].  Never overflows.
  - State decay uses exp(g_last - gc[t]) instead of exp(-gc[t]).
    Since g_last = gc[-1] <= gc[t] for all t, the argument is <= 0.
  - exp(gc[t]) is always safe because gc[t] <= 0 → exp(gc[t]) in (0, 1].

NKI 0.3.0 GA (SDK 2.29.1). k_dim = v_dim = 128 = P_MAX exactly.
Chunk size = 128 = P_MAX (one tile per chunk).

Mathematical framework:
  Per-chunk Neumann-series power-doubling for intra-chunk correction:
    A = -QK_decay * lower_mask   (QK_decay[i,j] = (k_beta^T @ k)[i,j] * exp(gc[i]-gc[j]))
    N = (I+A)(I+A^2)(I+A^4)...(I+A^64)  [6 rounds]
    value_corr = N @ v_beta
    k_cumdecay = N @ (k_beta * exp(gc))

  Inter-chunk state propagation:
    v_prime = k_cumdecay @ state
    v_new = value_corr - v_prime
    attn_inter = (q * exp(gc)) @ state
    attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    output = attn_inter + attn_intra @ v_new
    state = exp(g_last) * state + (k * exp(g_last - gc))^T @ v_new
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

        # ---- Compute exp(gc) as (P_MAX, 1) ----
        # gc <= 0 always (all g values are negative), so exp(gc) in (0, 1].
        # Safe from overflow.
        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        # ---- Compute exp(g_last - gc) as (P_MAX, 1) ----
        # g_last = gc[-1] is the most negative value in gc (monotonically decreasing).
        # g_last - gc[t] <= 0 for all t, so exp(g_last - gc) in (0, 1].
        # Used for state update: k * exp(g_last - gc) replaces k * exp(-gc).
        gl_broadcast = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gl_minus_gc, data1=gl_broadcast, data2=gc_p, op=nl.subtract
        )

        exp_gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_minus_gc[0:P_MAX, 0:1],
            op=nl.exp,
            data=gl_minus_gc[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        # exp(g_last): scalar, then broadcast to (P_MAX, 1)
        # g_last <= 0 so exp(g_last) in (0, 1]. Safe.
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
        #
        # NUMERICALLY STABLE decay mask:
        #   decay[i,j] = exp(gc[i] - gc[j])
        # Computed as: build gc_diff matrix = gc[i] - gc[j], clamp <= 0, exp.
        # Since gc is monotonically decreasing and lower-tri has i >= j:
        #   gc[i] - gc[j] <= 0, so exp() in (0, 1]. Never overflows.
        # ============================================================

        # Build gc_diff[i,j] = gc[i] - gc[j] as a (128x128) matrix
        # gc_mat_rows[i,j] = gc[i] for all j (broadcast gc_p across free dim)
        ones_PP = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_PP, value=1.0)

        gc_mat_rows = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_mat_rows,
            data=ones_PP,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )

        # gc_mat_cols[i,j] = gc[j] for all i (broadcast gc_row across partitions)
        gc_mat_cols = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:CHUNK_SIZE],
                dst=gc_mat_cols[i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK_SIZE],
                shuffle_mask=_BROADCAST_MASK,
            )

        # gc_diff[i,j] = gc[i] - gc[j]
        gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_diff, data1=gc_mat_rows, data2=gc_mat_cols, op=nl.subtract
        )

        # Clamp to <= 0 for numerical safety (upper triangle would be positive
        # but gets masked out anyway; clamping prevents any overflow in exp)
        gc_diff_clamped = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_diff_clamped,
            data=gc_diff,
            op0=nl.minimum,
            operand0=0.0,
            engine=nisa.vector_engine,
        )

        # exp(gc_diff_clamped) — always in (0, 1], never overflows
        exp_gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_diff, op=nl.exp, data=gc_diff_clamped, bias=None, scale=1.0
        )

        # ---- Transpose K and K_beta for matmul ----
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

        # QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j])
        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=exp_gc_diff, op=nl.multiply)

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
        # Neumann power-doubling: N = (I+A)(I+A^2)...(I+A^{64})
        # 6 rounds → resolves rank up to 2^6 = 64 (sufficient for chunk=128
        # since A entries are bounded by ~1.0: Q,K are l2-normed so |QK|<=1,
        # and exp(gc[i]-gc[j])<=1, so A is convergent)
        # ============================================================
        P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=P_acc, data1=eye, data2=A_mat, op=nl.add)

        A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_pow, src=A_mat)

        for _round in nl.sequential_range(6):
            # A_pow = A_pow^2: transpose A_pow, then matmul
            Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=Ap_T_psum, data=A_pow)
            Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

            Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
            nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

            # P_acc = (I + A_pow) @ P_acc: transpose IpA, then matmul
            IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

            IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=IpA_T_psum, data=IpA)
            IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

            Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
            nisa.tensor_copy(dst=P_acc, src=Pacc_psum)

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

        # k_beta * exp(gc): row-scaled (exp(gc) in (0,1] — safe)
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
        #
        # Uses the SAME numerically stable exp_gc_diff matrix for decay.
        # ============================================================
        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_c)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        # Apply stable decay: qk_decay[i,j] = qk_raw[i,j] * exp(gc[i] - gc[j])
        qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=qk_decay, data1=qk_raw, data2=exp_gc_diff, op=nl.multiply
        )

        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply
        )

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
        # exp(gc) in (0,1] — safe from overflow
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
        # State update:
        #   state_new = exp(g_last) * state + (k * exp(g_last - gc))^T @ v_new
        #
        # Mathematically equivalent to:
        #   state_new = exp(g_last) * (state + (k * exp(-gc))^T @ v_new)
        # But the original exp(-gc) OVERFLOWS because gc is very negative
        # (gc = cumsum of negative g values, so -gc is very positive).
        #
        # Instead we use exp(g_last - gc[t]):
        #   g_last = gc[-1] <= gc[t] for all t (gc is monotonically decreasing)
        #   So g_last - gc[t] <= 0, meaning exp(g_last - gc) in (0, 1].
        #
        # Then: state = exp(g_last) * state + (k * exp(g_last - gc))^T @ v_new
        # This is equivalent because:
        #   exp(g_last) * (k * exp(-gc))^T = (k * exp(g_last - gc))^T
        # So we fold the exp(g_last) factor into the key scaling for the
        # outer product term, and apply it separately to the state decay.
        # ============================================================

        # k_state_decay = k * exp(g_last - gc)  — always in (0, 1], safe
        k_state_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_state_decay,
            data=k_c,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc,
            engine=nisa.vector_engine,
        )

        # k_state_decay^T @ v_new → (dim, dim) outer product sum
        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_psum, stationary=k_state_decay, moving=v_new)
        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        # state = exp(g_last) * state + kv_outer
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
