"""Qwen3.6-specific Q/K RMSNorm + partial-RoPE NKI kernel.

The model's full-attention layers use head_dim=256 and partial RoPE over the
first 64 dimensions.  This kernel consumes projected Q/K tensors in BSD layout
and returns normalized/rotated Q/K tensors in BHSD layout, replacing the
separate move_heads_front + q/k RMSNorm + partial RoPE torch ops.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
D_HEAD = 256
ROPE_DIM = 64
ROPE_HALF = 32
_BROADCAST_MASK = [0] * 32


def _broadcast_row_to_tile(row, out):
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=row[0:1, 0:D_HEAD],
            dst=out[i_shuf * 32 : i_shuf * 32 + 32, 0:D_HEAD],
            shuffle_mask=_BROADCAST_MASK,
        )


def _normalize_rope_store(
    proj,
    gamma,
    cos_cache,
    sin_cache,
    out,
    eps,
):
    batch_size, seq_len, width = proj.shape
    num_heads = width // D_HEAD
    gamma_2d = gamma.reshape((1, D_HEAD))

    gamma_row = nl.ndarray((1, D_HEAD), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_row, src=gamma_2d[0:1, 0:D_HEAD])
    gamma_tile = nl.ndarray((P_MAX, D_HEAD), dtype=gamma.dtype, buffer=nl.sbuf)
    _broadcast_row_to_tile(gamma_row, gamma_tile)

    num_seq_tiles = (seq_len + P_MAX - 1) // P_MAX
    for b_idx in nl.sequential_range(batch_size):
        for h_idx in nl.sequential_range(num_heads):
            col_start = h_idx * D_HEAD
            for tile_idx in nl.affine_range(num_seq_tiles):
                seq_start = tile_idx * P_MAX
                p_size = min(P_MAX, seq_len - seq_start)

                x = nl.ndarray((P_MAX, D_HEAD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=x[0:p_size, 0:D_HEAD],
                    src=proj[
                        b_idx,
                        seq_start : seq_start + p_size,
                        col_start : col_start + D_HEAD,
                    ],
                )

                square = nl.ndarray((P_MAX, D_HEAD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=square[0:p_size, 0:D_HEAD],
                    data1=x[0:p_size, 0:D_HEAD],
                    data2=x[0:p_size, 0:D_HEAD],
                    op=nl.multiply,
                )

                sumsq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=sumsq[0:p_size, 0:1],
                    data=square[0:p_size, 0:D_HEAD],
                    op=nl.add,
                    axis=1,
                )

                variance = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=variance[0:p_size, 0:1],
                    data=sumsq[0:p_size, 0:1],
                    op0=nl.multiply,
                    operand0=(1.0 / D_HEAD),
                )

                variance_eps = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=variance_eps[0:p_size, 0:1],
                    data=variance[0:p_size, 0:1],
                    op0=nl.add,
                    operand0=eps,
                )

                inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=inv_rms[0:p_size, 0:1],
                    data=variance_eps[0:p_size, 0:1],
                    op=nl.rsqrt,
                )

                normed = nl.ndarray((P_MAX, D_HEAD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=normed[0:p_size, 0:D_HEAD],
                    data=x[0:p_size, 0:D_HEAD],
                    op0=nl.multiply,
                    operand0=inv_rms[0:p_size, 0:1],
                    engine=nisa.vector_engine,
                )
                nisa.tensor_tensor(
                    dst=normed[0:p_size, 0:D_HEAD],
                    data1=normed[0:p_size, 0:D_HEAD],
                    data2=gamma_tile[0:p_size, 0:D_HEAD],
                    op=nl.multiply,
                )

                nisa.dma_copy(
                    dst=out[b_idx, h_idx, seq_start : seq_start + p_size, 0:D_HEAD],
                    src=normed[0:p_size, 0:D_HEAD],
                )

                cos_tile = nl.ndarray((P_MAX, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
                sin_tile = nl.ndarray((P_MAX, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=cos_tile[0:p_size, 0:ROPE_DIM],
                    src=cos_cache[b_idx, seq_start : seq_start + p_size, 0:ROPE_DIM],
                )
                nisa.dma_copy(
                    dst=sin_tile[0:p_size, 0:ROPE_DIM],
                    src=sin_cache[b_idx, seq_start : seq_start + p_size, 0:ROPE_DIM],
                )

                neg_hi = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=neg_hi[0:p_size, 0:ROPE_HALF],
                    data=normed[0:p_size, ROPE_HALF:ROPE_DIM],
                    op0=nl.multiply,
                    operand0=-1.0,
                )

                lo_cos = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                hi_sin_neg = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                rope_lo = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=lo_cos[0:p_size, 0:ROPE_HALF],
                    data1=normed[0:p_size, 0:ROPE_HALF],
                    data2=cos_tile[0:p_size, 0:ROPE_HALF],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=hi_sin_neg[0:p_size, 0:ROPE_HALF],
                    data1=neg_hi[0:p_size, 0:ROPE_HALF],
                    data2=sin_tile[0:p_size, 0:ROPE_HALF],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=rope_lo[0:p_size, 0:ROPE_HALF],
                    data1=lo_cos[0:p_size, 0:ROPE_HALF],
                    data2=hi_sin_neg[0:p_size, 0:ROPE_HALF],
                    op=nl.add,
                )

                hi_cos = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                lo_sin = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                rope_hi = nl.ndarray((P_MAX, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=hi_cos[0:p_size, 0:ROPE_HALF],
                    data1=normed[0:p_size, ROPE_HALF:ROPE_DIM],
                    data2=cos_tile[0:p_size, ROPE_HALF:ROPE_DIM],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=lo_sin[0:p_size, 0:ROPE_HALF],
                    data1=normed[0:p_size, 0:ROPE_HALF],
                    data2=sin_tile[0:p_size, ROPE_HALF:ROPE_DIM],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=rope_hi[0:p_size, 0:ROPE_HALF],
                    data1=hi_cos[0:p_size, 0:ROPE_HALF],
                    data2=lo_sin[0:p_size, 0:ROPE_HALF],
                    op=nl.add,
                )

                nisa.dma_copy(
                    dst=out[b_idx, h_idx, seq_start : seq_start + p_size, 0:ROPE_HALF],
                    src=rope_lo[0:p_size, 0:ROPE_HALF],
                )
                nisa.dma_copy(
                    dst=out[
                        b_idx,
                        h_idx,
                        seq_start : seq_start + p_size,
                        ROPE_HALF:ROPE_DIM,
                    ],
                    src=rope_hi[0:p_size, 0:ROPE_HALF],
                )


@nki.jit
def qwen_qk_norm_partial_rope_kernel(
    q_proj: nl.ndarray,
    k_proj: nl.ndarray,
    q_gamma: nl.ndarray,
    k_gamma: nl.ndarray,
    cos_cache: nl.ndarray,
    sin_cache: nl.ndarray,
    eps: float,
):
    batch_size, seq_len, q_width = q_proj.shape
    _, _, k_width = k_proj.shape
    q_heads = q_width // D_HEAD
    k_heads = k_width // D_HEAD

    q_out = nl.ndarray(
        (batch_size, q_heads, seq_len, D_HEAD),
        dtype=q_proj.dtype,
        buffer=nl.shared_hbm,
    )
    k_out = nl.ndarray(
        (batch_size, k_heads, seq_len, D_HEAD),
        dtype=k_proj.dtype,
        buffer=nl.shared_hbm,
    )

    _normalize_rope_store(q_proj, q_gamma, cos_cache, sin_cache, q_out, eps)
    _normalize_rope_store(k_proj, k_gamma, cos_cache, sin_cache, k_out, eps)

    return q_out, k_out
