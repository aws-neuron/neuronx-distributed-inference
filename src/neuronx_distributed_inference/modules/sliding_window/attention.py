"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

Adapted from https://github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/reference/attention.py
"""

import math
from dataclasses import dataclass

import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.language import par_dim


B_P_SIZE = nl.tile_size.pmax  # 128
B_F_SIZE = nl.tile_size.gemm_moving_fmax  # 512
NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
MIN_SLIDING_WINDOW_SEQ_TILE_SIZE = 512
DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE = 2048


@dataclass(frozen=True)
class FlashConfig:
    """
    Config class for flash attention with default values
    """
    seq_tile_size: int = DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE
    attn_core_tile_size: int = 256
    should_transpose_v: bool = False
    lse_dtype: str = ""
    windowed_context_encoding: bool = False  # if True, uses offset-ed mask for WCTE window = sliding window


@nki.jit(mode="trace")
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, use_dma_transpose=False):
    for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
        # Temporarily disable use_dma_tranpose by default until we stablized it
        if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), buffer=nl.sbuf, dtype=p_local.dtype)
        else:
            p_local_t_tmp = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), buffer=nl.psum, dtype=np.float32)

        for j in nl.affine_range(B_F_SIZE // B_P_SIZE):
            j_128_slice = nl.ds(j * B_P_SIZE, B_P_SIZE)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * B_P_SIZE, B_P_SIZE)

            if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(p_local[:, i_j_128_slice])
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(p_local[:, i_j_128_slice])

        p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
            p_local_t_tmp, dtype=p_local_transposed.dtype
        )


@nki.jit(mode="trace")
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    q_tile_idx,
    local_k_large_tile_idx,
    kernel_dtype,
    acc_type,
    flash_config: FlashConfig,
    use_causal_mask,
    sliding_window,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
    The q_local_tile has (B_P_SIZE, B_D_SIZE), which is loaded into the SBUF already. The block size of K and V
    is defined in the seq_tile_size of the flash_config. The results are stored in the following three buffers
    o_buffer: (B_P_SIZE, d)
    l_buffer: (B_P_SIZE, 1)
    m_buffer: (B_P_SIZE, 1)
    """
    NEG_INFINITY = -9984.0  # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
    LARGE_TILE_SZ = flash_config.seq_tile_size
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        qk_psum = nl.ndarray(
            (par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
        )  # (128, 512)
        if use_causal_mask:
            if flash_config.windowed_context_encoding:
                multiplication_required_selection = (
                    q_tile_idx * B_P_SIZE + sliding_window >= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
                )
            else:
                multiplication_required_selection = (
                    q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
                )
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
        else:
            qk_psum[:, :] = 0

        if use_causal_mask:
            diagonal_and_left_selection = (
                q_tile_idx + 1
            ) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE

            i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
            q_pos = q_tile_idx * B_P_SIZE + i_q_p
            k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
            pred_causal = q_pos >= k_pos  # casual mask
            pred_sliding = k_pos > q_pos - sliding_window  # sliding window mask
            if flash_config.windowed_context_encoding:
                pred_causal = q_pos + sliding_window >= k_pos  # causal mask
                pred_sliding = q_pos < k_pos  # sliding window mask

            # Apply causal mask
            qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                pred=pred_causal,
                on_true_tile=qk_psum,
                on_false_value=NEG_INFINITY,
                dtype=acc_type,
            )
            if sliding_window > 0:  # Apply sliding window mask
                qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                    pred=pred_sliding,
                    on_true_tile=qk_res_buf[:, k_i_b_f_slice],
                    on_false_value=NEG_INFINITY,
                    dtype=acc_type,
                    mask=diagonal_and_left_selection,
                )
        else:
            # Simply send psum result back to sbuf
            qk_res_buf[:, k_i_b_f_slice] = nl.copy(qk_psum, dtype=acc_type)

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type, negate=False
        )

    max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,), dtype=acc_type, negate=False)

    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

    m_current = m_buffer[:, 0]
    # Compute scaling factor
    alpha = nisa.activation(np.exp, m_current, bias=m_previous, scale=-1.0)
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk-max)
        # Compute partial row-tile sum of exp(qk-max))
        # FIXME: Use activation accumulate to accumulate over k_r_i loop?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    transpose_p_local(
        p_local_transposed=p_local_transposed, p_local=p_local, LARGE_TILE_SZ=LARGE_TILE_SZ
    )

    pv_psum = nl.zeros(
        (par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum, lazy_initialization=True
    )
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)], v[k_i, :, :], transpose_x=True
        )  # (128, 128) (p(Br), d)

    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

    exp = nisa.activation(nl.exp, m_current, bias=l_buffer[:, 0], scale=-1.0)
    l_buffer[:, 0] = nl.add(m_current, nisa.activation(nl.log, exp, bias=ps))


@nki.jit(mode="trace")
def load_v_tile(v_hbm_tile, cur_v_tile, j, v_i, config):
    LARGE_TILE_SZ = config.seq_tile_size
    B_P_SIZE = 128

    if not config.should_transpose_v:
        cur_v_tile[v_i, :, :] = nl.load(
            v_hbm_tile[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
            dtype=cur_v_tile.dtype,
        )
        return

    if nisa.get_nc_version() >= nisa.nc_version.gen3:
        cur_v_tile_transposed = nisa.dma_transpose(
            v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)]
        )
        cur_v_tile[v_i, :, :] = nisa.tensor_copy(cur_v_tile_transposed, dtype=cur_v_tile.dtype)
        return

    cur_v_tile[v_i, :, :] = nl.load_transpose2d(
        v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)], dtype=cur_v_tile.dtype
    )


@nki.jit
def flash_fwd(
    q,
    k,
    v,
    softmax_scale=None,
    use_causal_mask=True,
    window_size=(-1, -1),  # -1 means infinite context window
    mixed_precision=True,
    config=None,
):
    """
    Flash Attention Forward kernel

    IO tensor layouts:
      - q: shape   (bs, n_heads, d, seq_q)
      - k: shape   (bs, nk_heads, d, seq_k)
      - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
      - o: shape (bs, n_heads, seq_q, d)
      - This kernel requires seq_k == seq_v

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype
      - If mixed_precision is True, then all Tensor Engine operation will be performed in
        bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
        will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default is set to `true`, if false, we use same precision as input types
      - causal_mask: flag to set causal masking
      - config: Instance of :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
          seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
          training: bool to indicate training vs inference `default=True`

    Performance Notes:
      For better performance, the kernel is tiled to be of size `config.seq_tile_size`, and Flash attention math techniques are applied in unit
      of `config.seq_tile_size`. Seqlen that is not divisible by `config.seq_tile_size` is not supported at the moment.

      For large seqlen, `o_buffer` will overflow the statebuf. the kernel is tile `o_buffer` based on the value of `config.attn_core_tile_size`.
      This is a tradeoff between memory usage and performance. The default value of `config.attn_core_tile_size` is 256, which means the `o_buffer`
      will roughly take half of the statebuf. The computes are also tiled accordingly. DMA will be rematerialized
      `seqlen_q // B_P_SIZE // attn_core_tile_size times`.



    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Masking support Notes:
      3 masking options are supported w/
            use_causal_mask and window_size=(left_window_size, right_window_size):
        1. use_causal_mask=False, ()=-1: full (no masking)
        2. use_causal_mask=True, left_window_size=-1: causal
        3. use_causal_mask={True/False}, left_window_size >= 0: causal & sliding window
            - excluding current token, attend only the previous `left_window_size` tokens
            - given left_window_size >= 0, use_causal_mask is overriden to be True
                i.e. no support for bidirectional sliding window

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    config = config or FlashConfig()
    b, h, d, seqlen_q = q.shape
    B_D_SIZE = d
    _, k_h, _, seqlen_k = k.shape
    if config.should_transpose_v:
        assert tuple(v.shape) == (
            b,
            k_h,
            d,
            seqlen_k,
        ), f"Expect shape of V to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {v.shape}"
        assert tuple(k.shape) == (
            b,
            k_h,
            d,
            seqlen_k,
        ), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
    else:
        assert tuple(v.shape) == (
            b,
            k_h,
            seqlen_k,
            d,
        ), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} (batch, heads, seqlen_k, d_head) but got {v.shape}"
        assert tuple(k.shape) == (
            b,
            k_h,
            d,
            seqlen_k,
        ), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    left_window_size, right_window_size = window_size
    assert right_window_size < 0, "right sliding window is currently not supported"
    use_causal_mask = (
        True if left_window_size > 0 else use_causal_mask
    )  # setting sliding window assumes causal
    # WCTE can only be used SWA
    if config.windowed_context_encoding:
        assert left_window_size > 0, "SWA must be turned on if WCTE is turned on."
    sliding_window = left_window_size + 1  # sliding_window includes current token
    kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    n_tile_q = seqlen_q // B_P_SIZE  # since q will be loaded on tensor engine

    LARGE_TILE_SZ = config.seq_tile_size
    attn_core_tile_size = config.attn_core_tile_size

    # FIXME: Add masking for different seqlen values.
    assert (
        config.seq_tile_size >= MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
    ), f" seq tile_size {config.seq_tile_size} cannot be less than {MIN_SLIDING_WINDOW_SEQ_TILE_SIZE}"
    assert (
        seqlen_k % LARGE_TILE_SZ == 0
    ), f"Need seqlen_k to be divisible by {LARGE_TILE_SZ} but got {seqlen_k}"
    num_large_k_tile = seqlen_k // LARGE_TILE_SZ

    q_h_per_k_h = h // k_h

    n_remat = math.ceil(n_tile_q / attn_core_tile_size)
    attn_core_tile_size = min(n_tile_q, attn_core_tile_size)

    for i_q_h in nl.affine_range(q_h_per_k_h):
        # =============== Global Flash Attention accumulators ====================== #
        l_buffer = nl.full(
            (par_dim(B_P_SIZE), n_tile_q),
            fill_value=-9984.0,
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=False,
        )
        # =============== Global Flash Attention accumulators END ================== #

        for i0 in nl.sequential_range(n_remat):
            # =============== Global Flash Attention accumulators ====================== #
            o_buffer = nl.zeros(
                (attn_core_tile_size, par_dim(B_P_SIZE), d),
                dtype=acc_type,
                buffer=nl.sbuf,
                lazy_initialization=False,
            )
            m_buffer = nl.full(
                (attn_core_tile_size, par_dim(B_P_SIZE), 1),
                fill_value=-9984.0,
                dtype=acc_type,
                buffer=nl.sbuf,
                lazy_initialization=False,
            )
            # =============== Global Flash Attention accumulators END ================== #

            for j in nl.sequential_range(0, num_large_k_tile):
                cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
                cur_v_tile = nl.ndarray(
                    (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype
                )

                cur_k_tile[:, :] = nl.load(
                    k[batch_id, head_id, :, nl.ds(j * LARGE_TILE_SZ, LARGE_TILE_SZ)]
                )

                load_tile_size = B_P_SIZE

                v_hbm_tile = v[batch_id, head_id]
                for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
                    load_v_tile(
                        v_hbm_tile=v_hbm_tile, cur_v_tile=cur_v_tile, j=j, v_i=v_i, config=config
                    )

                for i1 in nl.affine_range(attn_core_tile_size):
                    i = i0 * attn_core_tile_size + i1
                    # mask are used to only apply computation to the lower half of the matrix,
                    # which reduce the arthimetic intensity by half.
                    # forward_mask imply initialize, i.e. if forward_mask is false, initialize will
                    # be false as well
                    if use_causal_mask and sliding_window < 0:
                        causal_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
                        sliding_mask = True
                    elif sliding_window > 0:
                        if config.windowed_context_encoding:
                            causal_mask = i * B_P_SIZE + sliding_window >= j * LARGE_TILE_SZ
                            sliding_mask = ((j + 1) * LARGE_TILE_SZ - 1) > ((i * B_P_SIZE))
                        else:
                            causal_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
                            sliding_mask = ((j + 1) * LARGE_TILE_SZ - 1) > (
                                (i * B_P_SIZE) - sliding_window
                            )
                    else:
                        casual_mask = True  # noqa: F841
                        sliding_mask = True

                    if (i < n_tile_q) & causal_mask & sliding_mask:
                        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                        q_hbm_tile = q[batch_id, head_id * q_h_per_k_h + i_q_h]
                        q_sbuf_tile = nl.load(
                            q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)], dtype=kernel_dtype
                        )  # load (d, 128) tile in SBUF
                        q_tile[:, :] = q_sbuf_tile * softmax_scale

                        _flash_attention_core(
                            q_local_tile=q_tile,
                            k=cur_k_tile,
                            v=cur_v_tile,
                            o_buffer=o_buffer[i1],
                            l_buffer=l_buffer[:, i],
                            m_buffer=m_buffer[i1],
                            q_tile_idx=i,
                            local_k_large_tile_idx=j,
                            kernel_dtype=kernel_dtype,
                            acc_type=acc_type,
                            flash_config=config,
                            use_causal_mask=use_causal_mask,
                            sliding_window=sliding_window,
                            B_D_SIZE=B_D_SIZE,
                        )

            # -------- write output to buffer on HBM ------------ #
            for i1 in nl.affine_range(attn_core_tile_size):
                i = i0 * attn_core_tile_size + i1

                if i < n_tile_q:
                    exp = nisa.activation(
                        np.exp, l_buffer[:, i], bias=m_buffer[i1, :, :], scale=-1.0
                    )
                    out = nl.multiply(o_buffer[i1, :, :], exp, dtype=kernel_dtype)

                    nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h, nl.ds(i * B_P_SIZE, B_P_SIZE), :, ],
                             out)

    return o
