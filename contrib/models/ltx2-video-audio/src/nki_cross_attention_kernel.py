"""
NKI Cross-Attention Kernel for LTX-2 attn2 (text cross-attention)
=================================================================

Custom NKI kernel for cross-attention with a 1D additive mask over K positions.

K_seq = 1024, d = 128, Q_seq = 6144 (or 24576 for Stage 2).
Mask: [batch_heads, 1, 1024] additive bias {0, -10000}.

Uses ONLY nisa.* (ISA-level) APIs for mode='torchxla' compatibility.
All loads/stores via nisa.dma_copy, transposes via nisa.nc_transpose,
math via nisa.tensor_tensor / nisa.tensor_scalar / nisa.tensor_reduce /
nisa.activation_reduce / nisa.nc_matmul.

Module-level imports of nki.language and nki.isa are REQUIRED so the
NKI AST tracer can resolve them from the kernel function's __globals__.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# ── Constants ───────────────────────────────────────────────────────────────
_XATTN_Q_GRP = 128  # Q tile size
_XATTN_K_CHUNK = 128  # K chunk size (= V tile)
_XATTN_D = 128  # head_dim


@nki.jit(mode="torchxla")
def cross_attn_kernel(q_ref, k_ref, v_ref, mask_ref, out_ref):
    """NKI masked cross-attention: softmax(Q@K^T + mask) @ V.

    q_ref: [batch_heads, q_seq, 128] pre-scaled
    k_ref: [batch_heads, 1024, 128]
    v_ref: [batch_heads, 1024, 128]
    mask_ref: [batch_heads, 1, 1024] additive bias
    out_ref: [batch_heads, q_seq, 128]
    """
    batch_heads = q_ref.shape[0]
    q_seq = q_ref.shape[1]
    n_q_grps = q_seq // _XATTN_Q_GRP

    # Ones column for mask broadcasting via outer product
    ones_col = nl.ndarray((1, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_col, 1.0)

    for bh in nl.affine_range(batch_heads):
        # ── Load & transpose K: 8 × [128,128] -> [d_par, 128] ──────
        kc0_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc0_raw, src=k_ref[bh, 0:128, :])
        kc0_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc0_p, kc0_raw)
        kc0 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc0, kc0_p)

        kc1_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc1_raw, src=k_ref[bh, 128:256, :])
        kc1_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc1_p, kc1_raw)
        kc1 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc1, kc1_p)

        kc2_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc2_raw, src=k_ref[bh, 256:384, :])
        kc2_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc2_p, kc2_raw)
        kc2 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc2, kc2_p)

        kc3_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc3_raw, src=k_ref[bh, 384:512, :])
        kc3_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc3_p, kc3_raw)
        kc3 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc3, kc3_p)

        kc4_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc4_raw, src=k_ref[bh, 512:640, :])
        kc4_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc4_p, kc4_raw)
        kc4 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc4, kc4_p)

        kc5_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc5_raw, src=k_ref[bh, 640:768, :])
        kc5_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc5_p, kc5_raw)
        kc5 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc5, kc5_p)

        kc6_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc6_raw, src=k_ref[bh, 768:896, :])
        kc6_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc6_p, kc6_raw)
        kc6 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc6, kc6_p)

        kc7_raw = nl.ndarray(
            (_XATTN_K_CHUNK, _XATTN_D), dtype=k_ref.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=kc7_raw, src=k_ref[bh, 896:1024, :])
        kc7_p = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(kc7_p, kc7_raw)
        kc7 = nl.ndarray((_XATTN_D, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(kc7, kc7_p)

        # ── Load V: 8 × [128_par, 128_free] ────────────────────────
        vt0 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt0, src=v_ref[bh, 0:128, :])
        vt1 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt1, src=v_ref[bh, 128:256, :])
        vt2 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt2, src=v_ref[bh, 256:384, :])
        vt3 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt3, src=v_ref[bh, 384:512, :])
        vt4 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt4, src=v_ref[bh, 512:640, :])
        vt5 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt5, src=v_ref[bh, 640:768, :])
        vt6 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt6, src=v_ref[bh, 768:896, :])
        vt7 = nl.ndarray((_XATTN_K_CHUNK, _XATTN_D), dtype=v_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt7, src=v_ref[bh, 896:1024, :])

        # ── Broadcast masks: [1,128] -> [128,128] via outer product ─
        m0_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m0_raw, src=mask_ref[bh, 0:1, 0:128])
        m0_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m0_p, ones_col, m0_raw)
        m0 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m0, m0_p)

        m1_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m1_raw, src=mask_ref[bh, 0:1, 128:256])
        m1_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m1_p, ones_col, m1_raw)
        m1 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m1, m1_p)

        m2_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m2_raw, src=mask_ref[bh, 0:1, 256:384])
        m2_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m2_p, ones_col, m2_raw)
        m2 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m2, m2_p)

        m3_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m3_raw, src=mask_ref[bh, 0:1, 384:512])
        m3_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m3_p, ones_col, m3_raw)
        m3 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m3, m3_p)

        m4_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m4_raw, src=mask_ref[bh, 0:1, 512:640])
        m4_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m4_p, ones_col, m4_raw)
        m4 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m4, m4_p)

        m5_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m5_raw, src=mask_ref[bh, 0:1, 640:768])
        m5_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m5_p, ones_col, m5_raw)
        m5 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m5, m5_p)

        m6_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m6_raw, src=mask_ref[bh, 0:1, 768:896])
        m6_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m6_p, ones_col, m6_raw)
        m6 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m6, m6_p)

        m7_raw = nl.ndarray((1, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m7_raw, src=mask_ref[bh, 0:1, 896:1024])
        m7_p = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(m7_p, ones_col, m7_raw)
        m7 = nl.ndarray(
            (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(m7, m7_p)

        # ── Process each Q group ──────────────────────────────────
        for qg in nl.sequential_range(n_q_grps):
            q_start = qg * _XATTN_Q_GRP

            # Load Q chunk and transpose
            q_raw = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_D), dtype=q_ref.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=q_raw, src=q_ref[bh, q_start : q_start + _XATTN_Q_GRP, :])
            q_t_p = nl.ndarray(
                (_XATTN_D, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(q_t_p, q_raw)
            q_t = nl.ndarray((_XATTN_D, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(q_t, q_t_p)

            # Phase 1: Q@K^T + mask
            s0_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s0_p, q_t, kc0)
            s0 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s0, s0_p)
            nisa.tensor_tensor(s0, s0, m0, op=nl.add)

            s1_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s1_p, q_t, kc1)
            s1 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s1, s1_p)
            nisa.tensor_tensor(s1, s1, m1, op=nl.add)

            s2_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s2_p, q_t, kc2)
            s2 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s2, s2_p)
            nisa.tensor_tensor(s2, s2, m2, op=nl.add)

            s3_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s3_p, q_t, kc3)
            s3 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s3, s3_p)
            nisa.tensor_tensor(s3, s3, m3, op=nl.add)

            s4_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s4_p, q_t, kc4)
            s4 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s4, s4_p)
            nisa.tensor_tensor(s4, s4, m4, op=nl.add)

            s5_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s5_p, q_t, kc5)
            s5 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s5, s5_p)
            nisa.tensor_tensor(s5, s5, m5, op=nl.add)

            s6_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s6_p, q_t, kc6)
            s6 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s6, s6_p)
            nisa.tensor_tensor(s6, s6, m6, op=nl.add)

            s7_p = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(s7_p, q_t, kc7)
            s7 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(s7, s7_p)
            nisa.tensor_tensor(s7, s7, m7, op=nl.add)

            # Row-wise max across all 8 chunks
            max0 = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(max0, nl.maximum, s0, axis=1)
            maxt = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(maxt, nl.maximum, s1, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s2, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s3, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s4, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s5, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s6, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)
            nisa.tensor_reduce(maxt, nl.maximum, s7, axis=1)
            nisa.tensor_tensor(max0, max0, maxt, op=nl.maximum)

            # Negate max for activation_reduce bias
            neg_max = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(neg_max, max0, nl.multiply, -1.0)

            # Phase 2: exp(scores - max) with fused sum
            e0 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            sum0 = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(
                e0,
                op=nl.exp,
                data=s0,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sum0,
            )
            e1 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            sumt = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(
                e1,
                op=nl.exp,
                data=s1,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e2 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e2,
                op=nl.exp,
                data=s2,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e3 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e3,
                op=nl.exp,
                data=s3,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e4 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e4,
                op=nl.exp,
                data=s4,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e5 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e5,
                op=nl.exp,
                data=s5,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e6 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e6,
                op=nl.exp,
                data=s6,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)
            e7 = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_K_CHUNK), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation_reduce(
                e7,
                op=nl.exp,
                data=s7,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=sumt,
            )
            nisa.tensor_tensor(sum0, sum0, sumt, op=nl.add)

            sum_recip = nl.ndarray((_XATTN_Q_GRP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(sum_recip, sum0)

            # Phase 3: P @ V (transpose exp, matmul with V)
            e0_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e0_t_p, e0)
            e0_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e0_t, e0_t_p)
            e1_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e1_t_p, e1)
            e1_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e1_t, e1_t_p)
            e2_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e2_t_p, e2)
            e2_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e2_t, e2_t_p)
            e3_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e3_t_p, e3)
            e3_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e3_t, e3_t_p)
            e4_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e4_t_p, e4)
            e4_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e4_t, e4_t_p)
            e5_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e5_t_p, e5)
            e5_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e5_t, e5_t_p)
            e6_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e6_t_p, e6)
            e6_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e6_t, e6_t_p)
            e7_t_p = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(e7_t_p, e7)
            e7_t = nl.ndarray(
                (_XATTN_K_CHUNK, _XATTN_Q_GRP), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(e7_t, e7_t_p)

            # P@V matmuls and accumulation
            o_p = nl.ndarray((_XATTN_Q_GRP, _XATTN_D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(o_p, e0_t, vt0)
            out_acc = nl.ndarray(
                (_XATTN_Q_GRP, _XATTN_D), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(out_acc, o_p)
            tmp = nl.ndarray((_XATTN_Q_GRP, _XATTN_D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_matmul(o_p, e1_t, vt1)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e2_t, vt2)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e3_t, vt3)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e4_t, vt4)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e5_t, vt5)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e6_t, vt6)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)
            nisa.nc_matmul(o_p, e7_t, vt7)
            nisa.tensor_copy(tmp, o_p)
            nisa.tensor_tensor(out_acc, out_acc, tmp, op=nl.add)

            # Phase 4: Normalize and write
            nisa.tensor_tensor(out_acc, out_acc, sum_recip, op=nl.multiply)
            nisa.dma_copy(
                dst=out_ref[bh, q_start : q_start + _XATTN_Q_GRP, :],
                src=out_acc,
            )
