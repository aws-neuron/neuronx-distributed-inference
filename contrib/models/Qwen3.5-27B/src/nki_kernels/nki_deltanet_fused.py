# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused single-kernel DeltaNet chunked forward for CTE (context encoding).

v14: 8-block (16x16) forward substitution with 4-round Neumann per block.

ROOT CAUSE of v4/v12 failure: Neumann power-doubling on large blocks (64x64 or
128x128) suffers CATASTROPHIC CANCELLATION in fp32. Intermediate matrices
A^16/A^32/A^64 have entries up to 10^13-10^57, but the final resolvent has
max ~1.0. The matmul (I + A^k) @ P involves terms of magnitude 10^7+ that
must cancel to ~1.0, but fp32's 7 significant digits lose this cancellation.

SOLUTION: Split 128x128 resolvent into 8 blocks of 16x16. Each 16x16 block
needs 4 rounds of Neumann (A^16 = 0 for 16x16 SLT), max intermediate ~2300,
well within fp32 precision (error < 2e-4 even for worst case).

Algorithm:
  For (I - A)^{-1} @ b where A is 128x128 strictly lower triangular:

  Partition into 8 blocks of 16 rows each. Sequential forward solve:
    x_accum = 0
    For i = 0 to 7:
      cross_i = (A @ x_accum) masked to block i rows
      b_adj_i = b[block_i] + cross_i
      x_i = N_i @ b_adj_i  (N_i = 4-round Neumann on diagonal block i)
      x_accum += x_i

Optimization: Compute all 8 block resolvents ONCE (before the forward solve),
then reuse them for both value_corr and k_cumdecay solves.

Performance: ~107 matmuls per chunk → ~107µs. Negligible vs 50ms TTFT.
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
CHUNK_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8
NEUMANN_ROUNDS = 4

_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1)


def _make_lower_mask_diag():
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0)


def _make_identity():
    return np.eye(CHUNK_SIZE, dtype=np.float32)


def _make_block_masks():
    """8 diagonal block masks, packed as (8, 128, 128)."""
    masks = np.zeros((NUM_BLOCKS, CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
    for i in range(NUM_BLOCKS):
        start = i * BLOCK_SIZE
        masks[i, start : start + BLOCK_SIZE, start : start + BLOCK_SIZE] = 1.0
    return masks


def _make_row_masks():
    """8 row selection masks, packed as (8, 128, 1)."""
    masks = np.zeros((NUM_BLOCKS, CHUNK_SIZE, 1), dtype=np.float32)
    for i in range(NUM_BLOCKS):
        start = i * BLOCK_SIZE
        masks[i, start : start + BLOCK_SIZE, 0] = 1.0
    return masks


@nki.jit
def deltanet_fused_chunked_fwd(
    query: nl.ndarray,
    key: nl.ndarray,
    value: nl.ndarray,
    g_in: nl.ndarray,
    beta_in: nl.ndarray,
    lower_mask: nl.ndarray,
    identity: nl.ndarray,
    lower_mask_diag: nl.ndarray,
    block_masks: nl.ndarray,  # (8, 128, 128)
    row_masks: nl.ndarray,  # (8, 128, 1)
):
    """Fused chunked DeltaNet forward — 8-block forward substitution resolvent."""
    seq_len = query.shape[0]
    dim = query.shape[1]
    num_chunks = seq_len // CHUNK_SIZE

    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load constants
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # Load block masks
    blk_masks = [None] * NUM_BLOCKS
    for bi in nl.static_range(NUM_BLOCKS):
        blk_masks[bi] = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk_masks[bi], src=block_masks[bi, 0:P_MAX, 0:P_MAX])

    # Load row masks
    r_masks = [None] * NUM_BLOCKS
    for bi in nl.static_range(NUM_BLOCKS):
        r_masks[bi] = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_masks[bi][0:P_MAX, 0:1], src=row_masks[bi, 0:P_MAX, 0:1])

    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK_SIZE

        # ---- Load chunk data ----
        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_c, src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim])

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_c, src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim])

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_c, src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim])

        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # ---- Cumsum of g ----
        g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=g_padded, value=0.0)
        nisa.tensor_copy(
            dst=g_padded[0:CHUNK_SIZE, 0:1], src=g_chunk_p[0:CHUNK_SIZE, 0:1]
        )

        g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=g_row[0:1, 0:CHUNK_SIZE], src=g_tp_psum[0:1, 0:CHUNK_SIZE])

        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gc_padded, value=0.0)
        nisa.tensor_copy(
            dst=gc_padded[0:1, 0:CHUNK_SIZE], src=gc_row[0:1, 0:CHUNK_SIZE]
        )

        gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gc_p[0:CHUNK_SIZE, 0:1], src=gc_tp_psum[0:CHUNK_SIZE, 0:1])

        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1], src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE]
        )

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        gl_broadcast = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )
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

        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_gl_11, op=nl.exp, data=gl_11, bias=None, scale=1.0)

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        # ---- k_beta, v_beta ----
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

        # ---- Build decay mask ----
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

        gc_mat_cols = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:CHUNK_SIZE],
                dst=gc_mat_cols[i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK_SIZE],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_diff, data1=gc_mat_rows, data2=gc_mat_cols, op=nl.subtract
        )

        gc_diff_clamped = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_diff_clamped,
            data=gc_diff,
            op0=nl.minimum,
            operand0=0.0,
            engine=nisa.vector_engine,
        )

        exp_gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_diff, op=nl.exp, data=gc_diff_clamped, bias=None, scale=1.0
        )

        # ---- QK and A matrix ----
        kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
        kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_c)
        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
        QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK, src=QK_psum)

        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=exp_gc_diff, op=nl.multiply)

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

        # ---- Transpose A for cross-block matmuls ----
        A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=A_T_psum, data=A_mat)
        A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_T, src=A_T_psum)

        # ================================================================
        # PRE-COMPUTE 8 BLOCK RESOLVENTS (4-round Neumann each)
        # Store transposed resolvents for nc_matmul application.
        # ================================================================
        blk_resolvents_T = [None] * NUM_BLOCKS
        for bi in nl.static_range(NUM_BLOCKS):
            # Extract diagonal block: A_diag = A_mat * blk_mask[bi]
            A_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=A_diag, data1=A_mat, data2=blk_masks[bi], op=nl.multiply
            )

            # Neumann: P = (I+A)(I+A^2)(I+A^4)(I+A^8)
            P_blk = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=P_blk, data1=eye, data2=A_diag, op=nl.add)

            A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_pow, src=A_diag)

            for _r in nl.sequential_range(NEUMANN_ROUNDS):
                # A_pow = A_pow @ A_pow
                Ap_T_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=Ap_T_ps, data=A_pow)
                Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Ap_T, src=Ap_T_ps)

                Ap_sq_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=Ap_sq_ps, stationary=Ap_T, moving=A_pow)
                nisa.tensor_copy(dst=A_pow, src=Ap_sq_ps)

                # P = (I + A_pow) @ P
                IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

                IpA_T_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=IpA_T_ps, data=IpA)
                IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=IpA_T, src=IpA_T_ps)

                P_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=P_ps, stationary=IpA_T, moving=P_blk)
                nisa.tensor_copy(dst=P_blk, src=P_ps)

            # Transpose resolvent for later nc_matmul application
            blk_resolvents_T[bi] = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            P_T_ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=P_T_ps, data=P_blk)
            nisa.tensor_copy(dst=blk_resolvents_T[bi], src=P_T_ps)

        # ================================================================
        # FORWARD SOLVE 1: value_corr = (I-A)^{-1} @ v_beta
        # Uses static_range to allow list indexing of block resolvents.
        # The sequential dependency (x_accum) is maintained by data flow.
        # ================================================================
        vc_accum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=vc_accum, value=0.0)

        for bi in nl.static_range(NUM_BLOCKS):
            # Cross-block
            cross_ps = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=cross_ps, stationary=A_T, moving=vc_accum)
            cross = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cross, src=cross_ps)

            cross_m = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=cross_m,
                data=cross,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            # b_adj
            b_m = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=b_m,
                data=v_beta,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            b_adj = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=b_adj, data1=b_m, data2=cross_m, op=nl.add)

            # Apply resolvent
            x_i_ps = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=x_i_ps, stationary=blk_resolvents_T[bi], moving=b_adj)
            x_i = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x_i, src=x_i_ps)

            x_i_m = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=x_i_m,
                data=x_i,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            nisa.tensor_tensor(dst=vc_accum, data1=vc_accum, data2=x_i_m, op=nl.add)

        value_corr = vc_accum

        # ================================================================
        # FORWARD SOLVE 2: k_cumdecay = (I-A)^{-1} @ kb_exp_gc
        # Uses static_range for list indexing.
        # ================================================================
        kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kb_exp_gc,
            data=k_beta,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        kcd_accum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=kcd_accum, value=0.0)

        for bi in nl.static_range(NUM_BLOCKS):
            cross_ps2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=cross_ps2, stationary=A_T, moving=kcd_accum)
            cross2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cross2, src=cross_ps2)

            cross_m2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=cross_m2,
                data=cross2,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            b_m2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=b_m2,
                data=kb_exp_gc,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            b_adj2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=b_adj2, data1=b_m2, data2=cross_m2, op=nl.add)

            x_i_ps2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=x_i_ps2, stationary=blk_resolvents_T[bi], moving=b_adj2)
            x_i2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x_i2, src=x_i_ps2)

            x_i_m2 = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=x_i_m2,
                data=x_i2,
                op0=nl.multiply,
                operand0=r_masks[bi],
                engine=nisa.vector_engine,
            )

            nisa.tensor_tensor(dst=kcd_accum, data1=kcd_accum, data2=x_i_m2, op=nl.add)

        k_cumdecay = kcd_accum

        # ================================================================
        # Intra-chunk attention (same as v4)
        # ================================================================
        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_c)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=qk_decay, data1=qk_raw, data2=exp_gc_diff, op=nl.multiply
        )

        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply
        )

        # v_prime = k_cumdecay @ state
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

        # attn_inter = (q * exp(gc)) @ state
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

        # intra_out = attn_intra @ v_new
        ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
        ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

        intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
        intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=intra_out, src=intra_psum)

        # Output
        chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)
        nisa.dma_copy(
            dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim], src=chunk_out
        )

        # State update
        k_state_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_state_decay,
            data=k_c,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc,
            engine=nisa.vector_engine,
        )

        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_psum, stationary=k_state_decay, moving=v_new)
        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )

        nisa.tensor_tensor(dst=state, data1=state_decayed, data2=kv_outer, op=nl.add)

    nisa.dma_copy(dst=final_state_out, src=state)
    return output, final_state_out
