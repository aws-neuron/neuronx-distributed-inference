"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

"""

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

from neuronx_distributed_inference.modules.chunked_prefill.utils import B_P_SIZE


@nki.jit
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, B_F_SIZE=512):
    """
    Transpose local probability matrix with hardware-specific optimizations.

    This function performs matrix transposition on the local probability
    matrix, in order to prepare for the V multiplication. The shapes of input
    and output are the same, but the data inside has already been transposed.

    Args:
        p_local_transposed (ndarray): Output buffer for the transposed matrix
            (par_dim(B_P_SIZE), LARGE_TILE_SZ), in sbuf
        p_local (ndarray): Input local probability matrix to transpose
            (par_dim(B_P_SIZE), LARGE_TILE_SZ), in sbuf
        LARGE_TILE_SZ (int): Size of the large tile to process
        B_F_SIZE (int, optional): Block feature size, defaults to 512

    Notes:
        - For NC gen3: Uses shared buffer (sbuf) and DMA transpose
        - For other NC versions: Uses parallel sum buffer (psum) and NC
            transpose
        - Processes data in blocks of B_F_SIZE (512) with sub-blocks of
            B_P_SIZE (128)
    """
    # Process the matrix in blocks of B_F_SIZE (512)
    for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            # for dma_transpose(), output will be in sbuf
            p_local_t_tmp = nl.ndarray(
                (par_dim(B_P_SIZE), B_F_SIZE), buffer=nl.sbuf, dtype=p_local.dtype
            )
        else:
            # for nc_transpose(), output will be in psum, and dtype will be fp32
            p_local_t_tmp = nl.ndarray(
                (par_dim(B_P_SIZE), B_F_SIZE), buffer=nl.psum, dtype=np.float32
            )

        # Process sub-blocks of B_P_SIZE (128) elements
        for j in nl.affine_range(B_F_SIZE // B_P_SIZE):
            j_128_slice = nl.ds(j * B_P_SIZE, B_P_SIZE)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * B_P_SIZE, B_P_SIZE)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(p_local[:, i_j_128_slice])
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(p_local[:, i_j_128_slice])

        p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
            p_local_t_tmp, dtype=p_local_transposed.dtype
        )


@nki.jit
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    q_tile_idx=None,
    initialize=False,
    LARGE_TILE_SZ=2048,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.

    Args:
        q_local_tile (ndarray): Local query tile
            (B_D_SIZE, B_P_SIZE) in sbuf
        k (ndarray): Key matrix
            (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE) in sbuf
        v (ndarray): Value matrix
            (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE) in sbuf
        o_buffer (ndarray): Output buffer for attention results
            (B_P_SIZE, d)
        l_buffer (ndarray): Buffer for storing log sum
            (B_P_SIZE, 1)
        m_buffer (ndarray): Buffer for storing maximum values
            (B_P_SIZE, 1)
        kernel_dtype: Data type for kernel computations
        acc_type: Accumulation data type
        tile_mask: Mask for tile computations
            (B_P_SIZE, LARGE_KV_TILE_SIZE)
        use_causal_mask (bool): Whether to use causal masking. False for prior
            part, true for active part.
        q_tile_idx (int, optional): Query tile index. None for prior part.
        initialize (bool): Whether to initialize buffers. False by default.
        LARGE_TILE_SZ (int): Size of large tiles. KV tile size for prior part,
            Q tile size for the active part.
        B_P_SIZE (int): Parition dim size.
        B_F_SIZE (int): Free dim size.
        B_D_SIZE (int): Data dim size, maps to the head dim for attention module.

    The results are stored in the following three buffers
        o_buffer: (B_P_SIZE, d)
        l_buffer: (B_P_SIZE, 1)
        m_buffer: (B_P_SIZE, 1)
    """
    # ============= Step 1: QK matmul, masking and max reduce ============= #
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    qk_psum = nl.ndarray(
        (num_k_tile_per_large_tile, par_dim(B_P_SIZE), B_F_SIZE),
        buffer=nl.psum,
        dtype=acc_type,
    )

    # max_local is to store the max qk per tile
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        # masks are only applied to computation on the lower half of the matrix,
        # which reduces the arithmetic intensity by half
        if use_causal_mask:
            # active part
            multiplication_required_selection = q_tile_idx * B_P_SIZE >= k_i * B_F_SIZE
        else:
            # prior part
            multiplication_required_selection = True

        if multiplication_required_selection:
            # Step 1.1: QK matmul
            qk_psum[k_i, :, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(B_P_SIZE), B_F_SIZE)
            # Step 1.2: masking
            nisa.tensor_copy_predicated(
                src=-9984.0,
                dst=qk_psum[k_i, :, :],
                predicate=tile_mask[:, k_i_b_f_slice],
                reverse_pred=True,
            )  # masking with tile_mask
        else:
            qk_psum[k_i, :, :] = -9984.0

        # Step 1.3: Max reduce for current tile on free dim using VectorE
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_psum[k_i, :, :],
            axis=(1,),
            dtype=acc_type,
            negate=False,
        )

    # Step 1.4: Max reduce across tiles on free dim using VectorE
    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    # ============= Step 2: Update M buffers ============= #
    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

    if initialize:
        m_buffer[:, 0] = nl.copy(max_)
        m_current = max_
    else:
        m_previous = nl.copy(m_buffer[:, 0])
        m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

        m_current = m_buffer[:, 0]
        # Compute scaling factor using ScalarE
        alpha = nisa.activation(
            np.exp,
            m_previous,
            bias=-1 * m_current,
            scale=1.0,
        )
        o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    # ============= Step 3: Compute denominator for softmax ============= #
    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)

    REDUCTION_TILE = B_F_SIZE
    p_partial_sum = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE),
        dtype=acc_type,
    )

    # Step 3.1: compute the partial sum per tile
    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # Compute tile sum of exp(qk - max)) for each tile
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_psum[k_r_i, :, :],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )
    # Step 3.2: sum across tiles
    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    # ============= Step 4: PV matmul ============= #
    # Step 4.1: transpose the partial sum to prepare for PV matmul
    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
        B_F_SIZE=B_F_SIZE,
    )

    # Step 4.2: compute PV matmul
    pv_psum = nl.zeros(
        (par_dim(B_P_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    # Step 4.3: write outputs to aggregation buffers
    if initialize:
        o_buffer[:, :] = nl.copy(pv_psum[:, :])
        l_buffer[:, 0] = nl.add(nl.log(ps), max_)
    else:
        o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

        l_prev = l_buffer[:, 0]
        l_exp = nl.add(
            nl.exp(nl.subtract(l_prev, m_current)),
            ps,
        )
        l_buffer[:, 0] = nl.add(m_current, nl.log(l_exp))
