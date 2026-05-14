"""
NKI Flash Attention kernel supporting head_dim > 128 (up to 512).

Designed for NxDI integration: accepts the same I/O layout as NxDI's
standard flash attention kernel path (tp_q=True, tp_k=True).

Input layout:
    Q: (B*H, seqlen, d)       -- tp_q=True layout
    K: (B*H_kv, seqlen, d)    -- tp_k=True layout
    V: (B*H_kv, seqlen, d)

Output layout:
    O: (B*H, d, seqlen)       -- tp_out=True (what NxDI expects from kernel)

The kernel tiles the QK matmul contraction dimension in chunks of 128
(the hardware partition axis max). For d=256: 2 chunks, for d=512: 4 chunks.
The PV matmul places d on the free axis (max 512), supporting up to d=512.

Supports:
    - Causal masking
    - GQA (B*H > B*H_kv, with B*H divisible by B*H_kv)
    - Sliding window attention (optional)
    - head_dim = 128, 256, or 512

Based on the proven nki_flash_attn_d256.py kernel (cosine > 0.999 validated).
"""

import math
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

B_P = 128  # partition dim max (nl.tile_size.pmax)
B_F = 512  # free dim max for matmul moving operand (nl.tile_size.gemm_moving_fmax)
D_TILE = 128  # head_dim tile size for QK contraction
NEG_INF = -9984.0  # bfloat16-safe negative infinity


@nki.jit
def flash_attn_large_d(
    q,
    k,
    v,
    scale: float = 1.0,
    use_causal_mask: bool = True,
    sliding_window: int = 0,
):
    """
    Flash attention for head_dim up to 512.

    Args:
        q: (bs, seqlen_q, d)  -- bfloat16, tp_q=True layout (B*H merged into batch)
        k: (bs_kv, seqlen_k, d)  -- bfloat16, tp_k=True layout (B*H_kv merged)
        v: (bs_kv, seqlen_k, d)  -- bfloat16
        scale: float, scaling factor (already applied to Q by NxDI: Q = Q / sqrt(d))
        use_causal_mask: bool
        sliding_window: int, 0 means no sliding window

    Returns:
        o: (bs, d, seqlen_q)  -- bfloat16, tp_out=True (BHDS after unmerge)
    """
    bs, seqlen_q, d = q.shape
    bs_kv, seqlen_k, _ = k.shape

    assert d <= 512, f"head_dim must be <= 512, got {d}"
    assert d % D_TILE == 0, f"head_dim must be divisible by {D_TILE}, got {d}"
    assert seqlen_q % B_P == 0, f"seqlen_q must be divisible by {B_P}, got {seqlen_q}"
    assert seqlen_k % B_F == 0 or seqlen_k % B_P == 0, (
        f"seqlen_k must be divisible by {B_F} or {B_P}, got {seqlen_k}"
    )

    num_d_chunks = d // D_TILE  # 1 for d=128, 2 for d=256, 4 for d=512
    q_h_per_k_h = bs // bs_kv  # GQA ratio

    # Output: (bs, d, seqlen_q) -- transposed layout for NxDI
    o = nl.ndarray((bs, d, seqlen_q), dtype=q.dtype, buffer=nl.shared_hbm)

    batch_id = nl.program_id(axis=0)

    n_q_tiles = seqlen_q // B_P
    # K/V tiles: use B_F if possible, else B_P
    kv_tile_size = B_F if seqlen_k % B_F == 0 else B_P
    n_kv_tiles = seqlen_k // kv_tile_size

    for i_q_h in nl.affine_range(q_h_per_k_h):
        q_batch = batch_id * q_h_per_k_h + i_q_h
        k_batch = batch_id

        for qi in nl.sequential_range(n_q_tiles):
            # Accumulators
            o_acc = nl.zeros((par_dim(B_P), d), dtype=np.float32, buffer=nl.sbuf)
            m_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)
            l_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)

            # Load Q tile: num_d_chunks chunks of (D_TILE, B_P)
            # Q is (bs, seqlen_q, d), we need it transposed to (d_chunk, seqlen_chunk)
            # for the QK matmul: Q^T @ K where Q is (D_TILE, B_P) contraction on D_TILE
            q_chunks = nl.ndarray(
                (num_d_chunks, par_dim(D_TILE), B_P), dtype=nl.bfloat16
            )
            for dc in nl.affine_range(num_d_chunks):
                # Load from (bs, seqlen, d) -> need to transpose (seqlen_tile, d_chunk)
                # to (d_chunk, seqlen_tile) for matmul contraction
                q_tile_raw = nl.ndarray((par_dim(B_P), D_TILE), dtype=nl.bfloat16)
                q_tile_raw[:, :] = nl.load(
                    q[q_batch, nl.ds(qi * B_P, B_P), nl.ds(dc * D_TILE, D_TILE)]
                )
                # Transpose: (B_P, D_TILE) -> (D_TILE, B_P) via nc_transpose
                # But nc_transpose output goes to PSUM, need to copy to SBUF
                q_t_psum = nl.ndarray(
                    (par_dim(D_TILE), B_P), dtype=np.float32, buffer=nl.psum
                )
                q_t_psum[:, :] = nisa.nc_transpose(q_tile_raw)
                q_chunks[dc, :, :] = nl.copy(q_t_psum, dtype=nl.bfloat16)

            # Scale Q (scale is already applied by NxDI, but if scale != 1.0)
            if scale != 1.0:
                for dc in nl.affine_range(num_d_chunks):
                    q_chunks[dc, :, :] = nl.multiply(q_chunks[dc], scale)

            for kvi in nl.sequential_range(n_kv_tiles):
                kv_start = kvi * kv_tile_size
                num_kv = kv_tile_size  # actual KV tokens in this tile

                # Causal mask: skip if Q tile is entirely after K tile (no Q can attend to K)
                if use_causal_mask:
                    q_end = (qi + 1) * B_P - 1
                    skip_condition = q_end < kv_start
                else:
                    skip_condition = False

                # Sliding window: skip if K tile is entirely before the window
                # of the FIRST Q position in this tile
                if sliding_window > 0 and use_causal_mask:
                    q_start = qi * B_P
                    kv_end = kv_start + kv_tile_size - 1
                    # Window for q_start covers [q_start - sw + 1, q_start]
                    # Skip if kv_end < q_start - sw + 1
                    skip_sw = kv_end < (q_start - sliding_window + 1)
                    skip_condition = skip_condition or skip_sw

                if not skip_condition:
                    # Load K tile: (num_d_chunks, par_dim(D_TILE), kv_tile_size)
                    # K is (bs_kv, seqlen_k, d), need transposed to (d_chunk, seqlen_chunk)
                    k_chunks = nl.ndarray(
                        (num_d_chunks, par_dim(D_TILE), kv_tile_size),
                        dtype=nl.bfloat16,
                    )
                    for dc in nl.affine_range(num_d_chunks):
                        if kv_tile_size <= B_P:
                            # Small tile: load (B_P, D_TILE) and transpose
                            k_raw = nl.ndarray(
                                (par_dim(kv_tile_size), D_TILE), dtype=nl.bfloat16
                            )
                            k_raw[:, :] = nl.load(
                                k[
                                    k_batch,
                                    nl.ds(kv_start, kv_tile_size),
                                    nl.ds(dc * D_TILE, D_TILE),
                                ]
                            )
                            k_t_psum = nl.ndarray(
                                (par_dim(D_TILE), kv_tile_size),
                                dtype=np.float32,
                                buffer=nl.psum,
                            )
                            k_t_psum[:, :] = nisa.nc_transpose(k_raw)
                            k_chunks[dc, :, :] = nl.copy(k_t_psum, dtype=nl.bfloat16)
                        else:
                            # Large tile (B_F=512): load in sub-tiles of B_P and transpose each
                            n_sub = kv_tile_size // B_P
                            for si in nl.affine_range(n_sub):
                                k_raw = nl.ndarray(
                                    (par_dim(B_P), D_TILE), dtype=nl.bfloat16
                                )
                                k_raw[:, :] = nl.load(
                                    k[
                                        k_batch,
                                        nl.ds(kv_start + si * B_P, B_P),
                                        nl.ds(dc * D_TILE, D_TILE),
                                    ]
                                )
                                k_t_psum = nl.ndarray(
                                    (par_dim(D_TILE), B_P),
                                    dtype=np.float32,
                                    buffer=nl.psum,
                                )
                                k_t_psum[:, :] = nisa.nc_transpose(k_raw)
                                k_chunks[dc, :, nl.ds(si * B_P, B_P)] = nl.copy(
                                    k_t_psum, dtype=nl.bfloat16
                                )

                    # Tiled QK matmul: accumulate over d-chunks
                    # Each chunk: (D_TILE, B_P)^T @ (D_TILE, kv_tile_size)
                    #           = (B_P, kv_tile_size)
                    qk = nl.ndarray(
                        (par_dim(B_P), kv_tile_size),
                        dtype=np.float32,
                        buffer=nl.psum,
                    )
                    qk[:, :] = nl.matmul(q_chunks[0], k_chunks[0], transpose_x=True)
                    for dc in nl.affine_range(num_d_chunks - 1):
                        qk[:, :] += nl.matmul(
                            q_chunks[dc + 1], k_chunks[dc + 1], transpose_x=True
                        )

                    # Move to SBUF for masking/softmax
                    qk_sbuf = nl.ndarray(
                        (par_dim(B_P), kv_tile_size), dtype=np.float32, buffer=nl.sbuf
                    )

                    # Apply causal mask (and sliding window if enabled)
                    if use_causal_mask:
                        i_q, i_k = nl.mgrid[0:B_P, 0:kv_tile_size]
                        q_pos = qi * B_P + i_q
                        k_pos = kv_start + i_k
                        pred_causal = q_pos >= k_pos

                        qk_sbuf[:, :] = nisa.affine_select(
                            pred=pred_causal,
                            on_true_tile=qk,
                            on_false_value=NEG_INF,
                            dtype=np.float32,
                        )

                        if sliding_window > 0:
                            # Apply sliding window mask on top of causal mask
                            pred_sw = (q_pos - k_pos) < sliding_window
                            qk_sw = nl.ndarray(
                                (par_dim(B_P), kv_tile_size),
                                dtype=np.float32,
                                buffer=nl.sbuf,
                            )
                            qk_sw[:, :] = nisa.affine_select(
                                pred=pred_sw,
                                on_true_tile=qk_sbuf,
                                on_false_value=NEG_INF,
                                dtype=np.float32,
                            )
                            qk_sbuf = qk_sw
                    else:
                        qk_sbuf[:, :] = nl.copy(qk, dtype=np.float32)

                    # Row max for online softmax
                    new_max = nisa.tensor_reduce(
                        np.max, qk_sbuf, axis=(1,), dtype=np.float32, negate=False
                    )

                    m_prev = nl.copy(m_acc[:, 0])
                    m_acc[:, 0] = nl.maximum(m_prev, new_max)
                    m_cur = m_acc[:, 0]

                    # Rescale previous output
                    alpha = nisa.activation(np.exp, m_cur, bias=m_prev, scale=-1.0)
                    o_acc[...] = nl.multiply(o_acc, alpha)

                    # exp(qk - max) and row sum
                    p = nl.ndarray((par_dim(B_P), kv_tile_size), dtype=nl.bfloat16)
                    p_sum = nl.ndarray((par_dim(B_P), 1), dtype=np.float32)
                    p[:, :] = nisa.activation_reduce(
                        np.exp,
                        qk_sbuf,
                        bias=-1 * m_cur,
                        scale=1.0,
                        reduce_op=nl.add,
                        reduce_res=p_sum[:, 0],
                        dtype=nl.bfloat16,
                    )

                    # Load V tile: (kv_tile_size // B_P, par_dim(B_P), d)
                    n_v_sub = kv_tile_size // B_P
                    v_tile = nl.ndarray((n_v_sub, par_dim(B_P), d), dtype=nl.bfloat16)
                    for vi in nl.affine_range(n_v_sub):
                        v_tile[vi, :, :] = nl.load(
                            v[k_batch, nl.ds(kv_start + vi * B_P, B_P), :],
                            dtype=nl.bfloat16,
                        )

                    # Transpose p for PV matmul: need p as (par_dim, kv_tile_size)
                    # in the right layout for contraction
                    p_t = nl.ndarray((par_dim(B_P), kv_tile_size), dtype=nl.bfloat16)
                    for ti in nl.affine_range(kv_tile_size // B_P):
                        p_t_psum = nl.ndarray(
                            (par_dim(B_P), B_P),
                            dtype=np.float32,
                            buffer=nl.psum,
                        )
                        p_t_psum[:, :] = nisa.nc_transpose(p[:, nl.ds(ti * B_P, B_P)])
                        p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                            p_t_psum, dtype=nl.bfloat16
                        )

                    # PV matmul: (B_P, kv_tile_size) @ (kv_tile_size, d) -> (B_P, d)
                    # d is on the free axis of PSUM (max 512)
                    pv = nl.zeros(
                        (par_dim(B_P), d),
                        dtype=np.float32,
                        buffer=nl.psum,
                        lazy_initialization=True,
                    )
                    for vi in nl.affine_range(n_v_sub):
                        pv[:, :] += nl.matmul(
                            p_t[:, nl.ds(vi * B_P, B_P)],
                            v_tile[vi, :, :],
                            transpose_x=True,
                        )

                    o_acc[:, :] = nl.add(o_acc, pv)

                    # Update log-sum-exp
                    exp_l = nisa.activation(nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0)
                    l_acc[:, 0] = nl.add(
                        m_cur, nisa.activation(nl.log, exp_l, bias=p_sum[:, 0])
                    )

            # Final rescale
            final_exp = nisa.activation(
                np.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
            )
            out = nl.multiply(o_acc, final_exp, dtype=nl.bfloat16)

            # Store output: (B_P, d) -> o[batch, d, seqlen] (transposed)
            # We need to write to (bs, d, seqlen_q) layout
            # out is (par_dim(B_P), d), we need (d, B_P) in HBM
            # Transpose: (B_P, d) -> (d, B_P) via nc_transpose in chunks
            # d can be up to 512, B_P is 128
            # nc_transpose takes (par_dim(P), F) -> (par_dim(F), P) but F <= 512
            # We need d on par_dim for the output, but d can be > 128
            # Instead, store in d-chunks: each chunk is (B_P, D_TILE)
            # Transpose each to (D_TILE, B_P) and store
            for dc in nl.affine_range(num_d_chunks):
                out_chunk = out[:, nl.ds(dc * D_TILE, D_TILE)]
                out_t_psum = nl.ndarray(
                    (par_dim(D_TILE), B_P), dtype=np.float32, buffer=nl.psum
                )
                out_t_psum[:, :] = nisa.nc_transpose(out_chunk)
                out_t = nl.ndarray((par_dim(D_TILE), B_P), dtype=nl.bfloat16)
                out_t[:, :] = nl.copy(out_t_psum, dtype=nl.bfloat16)
                nl.store(
                    o[q_batch, nl.ds(dc * D_TILE, D_TILE), nl.ds(qi * B_P, B_P)],
                    out_t,
                )

    return o
