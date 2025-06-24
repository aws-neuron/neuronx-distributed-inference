import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.language import par_dim

from neuronx_distributed_inference.modules.chunked_prefill.flash_attn_core import (
    _flash_attention_core,
)
from neuronx_distributed_inference.modules.chunked_prefill.paged_cache import (
    load_block_tables,
    load_kv_tile_from_cache,
    transform_block_tables_for_indirect_load,
)
from neuronx_distributed_inference.modules.chunked_prefill.utils import (
    B_F_SIZE,
    B_P_SIZE,
    ceil_div,
    is_power_of_2,
)


@nki.jit
def load_v_tile(v_hbm_tile, cur_v_tile, large_tile_idx, v_i, LARGE_TILE_SZ):
    """
    Load the value for the active part, from HBM to SBUF

    Args:
        v_hbm_tile (ndarray): Source value tensor in HBM memory
            (seqlen_v, head_dim)
        cur_v_tile (ndarray): Destination tensor in SBUF
            (par_dim(B_P_SIZE), LARGE_Q_TILE_SIZE // B_P_SIZE * B_D_SIZE)
        large_tile_idx (int): Index of the current large tile being processed
        v_i (int): Value index within the tile
        LARGE_TILE_SZ (int): Size of each large q tile
    """
    B_D_SIZE = v_hbm_tile.shape[-1]
    cur_v_tile[:, nl.ds(v_i * B_D_SIZE, B_D_SIZE)] = nl.load(
        v_hbm_tile[
            nl.ds(large_tile_idx * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE),
            :,
        ],
        dtype=cur_v_tile.dtype,
    )


@nki.jit
def validate_inputs(
    tile_masks,
    query,
    key,
    value,
    key_cache,
    value_cache,
    B_F_SIZE,
):
    """
    Validate inputs have the expected shape
    """
    b, h, d, seqlen_q = query.shape
    NUM_LARGE_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape

    assert seqlen_q % LARGE_Q_TILE_SIZE == 0

    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    assert (
        d >= 16 and d <= 128 and is_power_of_2(d)
    ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"

    num_blocks, k_h, block_size, _ = key_cache.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{key_cache.shape=} mismatch!"
    assert tuple(value_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{value_cache.shape=} mismatch!"
    assert key is None or tuple(key.shape) == (
        1,
        k_h,
        d,
        seqlen_q,
    ), f"key shape {key.shape} mismatch!"
    assert value is None or tuple(value.shape) == (
        1,
        k_h,
        seqlen_q,
        d,
    ), f"value shape {value.shape} mismatch!"

    assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"

    assert LARGE_Q_TILE_SIZE % B_P_SIZE == 0
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"Need LARGE_KV_TILE_SIZE ({LARGE_KV_TILE_SIZE=}) to be divisible by B_P_SIZE ({B_F_SIZE=})"

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"


def init_oml_buffers(
    q_h_per_k_h,
    d,
    n_large_q_tile,
    n_small_in_large_q_tile,
    d_type,
):
    """
    Construct and init aggregation buffers for flash attention, which includes
    o_buffer, m_buffer, l_buffer

    FlashAttention paper: https://arxiv.org/abs/2205.14135

    Args:
        q_h_per_k_h (int): Number of query heads per KV head
        d (int): Head dimension
        n_large_q_tile (int): number of large q tiles
        n_small_in_large_q_tile (int): number of small tile in a large q tile
        d_type (dtype): dtype for the constructed buffers
    """
    NEG_INF = -9984.0  # Magic number to replace -inf similar to Tensorizer

    o_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * d),
        dtype=d_type,
        buffer=nl.hbm,
    )
    m_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * 1),
        dtype=d_type,
        buffer=nl.hbm,
    )
    # L buffer stores LSE + M
    # L_0 = LSE_0 + M_0 = log(sum([])) + max([]) = -inf + -inf
    # TODO: since we only do inference, we should consider save SumExp instead of LSE + M
    l_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * 1),
        dtype=d_type,
        buffer=nl.hbm,
    )

    for large_q_idx in nl.affine_range(n_large_q_tile):
        nl.store(dst=o_buffer[:, large_q_idx], value=0.0)
        nl.store(dst=m_buffer[:, large_q_idx], value=NEG_INF)
        nl.store(dst=l_buffer[:, large_q_idx], value=NEG_INF + NEG_INF)

    return o_buffer, m_buffer, l_buffer


@nki.jit
def flash_paged_attention_with_schedule(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    softmax_scale=None,
    mixed_precision=True,
):
    """
    Flash PagedAttention Forward Kernel.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_heads, d, seq_q)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - value_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - tile_q_indices: (num_large_tiles,)
      - tile_block_tables: (num_large_tiles, num_blocks_per_tile)
      - tile_masks: (num_large_tiles, large_tile_size_q, large_tile_size_k)
      - active_mask: (seq_q, seq_q)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is always 1, and different
        requests are concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for block_tables (int32) and mask (int32)
      - If mixed_percision is True, then all Tensor Engine operation will be performed in
        bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
        will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """

    # ================== Step 1: Validate Inputs ================== #

    NUM_LARGE_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    b, h, d, seqlen_q = query.shape
    num_blocks, k_h, block_size, _ = key_cache.shape

    B_D_SIZE = d

    validate_inputs(
        tile_masks,
        query,
        key,
        value,
        key_cache,
        value_cache,
        B_F_SIZE,
    )
    n_large_q_tile = seqlen_q // LARGE_Q_TILE_SIZE
    query = query.reshape((b, h, d, n_large_q_tile, LARGE_Q_TILE_SIZE))

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // B_P_SIZE
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(
        num_blocks_per_large_tile
    ), f"{num_blocks_per_large_tile=} is expected of be power of 2"

    # init output
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)

    # ================== Step 2: Load Auxiliary Inputs ================== #

    tile_q_indices_sbuf = nl.load(tile_q_indices.reshape((1, NUM_LARGE_TILE)), dtype=nl.int32)

    # block_tables_sbuf shape:
    # (num_large_tiles, num_blocks_per_large_tile)
    # -> (num_partitions, B_P_SIZE, num_blocks_per_large_tile)
    block_tables_sbuf = load_block_tables(
        block_tables_hbm=tile_block_tables,
        num_tiles=NUM_LARGE_TILE,
        num_blocks_per_tile=num_blocks_per_large_tile,
    )
    if num_blocks_per_large_tile < B_P_SIZE:
        # we checked num_blocks_per_tile is a power of 2
        assert B_P_SIZE % num_blocks_per_large_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
        assert block_size % block_size_tiling_factor == 0
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # block_tables_sbuf shape:
    # (num_partitions, B_P_SIZE, num_blocks_per_large_tile)
    # -> (num_loads, B_P_SIZE, num_large_tiles_rounded_to_128)
    block_tables_sbuf = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
    )

    # flatten KV cache to be 2D for loading into SBUF
    new_cache_shape = (
        num_blocks * k_h * block_size_tiling_factor,
        tiled_block_size * d,
    )
    key_cache = key_cache.reshape(new_cache_shape)
    value_cache = value_cache.reshape(new_cache_shape)

    # ========= Step 3: Init Global Flash Attention Accumulators =========== #

    q_h_per_k_h = h // k_h
    o_buffer, m_buffer, l_buffer = init_oml_buffers(
        q_h_per_k_h=q_h_per_k_h,
        d=d,
        n_large_q_tile=n_large_q_tile,
        n_small_in_large_q_tile=n_small_in_large_q_tile,
        d_type=acc_type,
    )

    # =========== Step 4: Attention Computation For Prior Part ================ #

    num_loads = ceil_div(num_blocks_per_large_tile, B_P_SIZE)
    # XXX: Work around a DMA skipping correctness issue:
    #      If nl.ndarray is used to allocate buffer for DMA skipping,
    #      kernel does not produce correct results.
    k_load_buffer = nl.zeros(
        (num_loads, par_dim(B_P_SIZE), tiled_block_size * B_D_SIZE),
        dtype=kernel_dtype,
    )
    v_load_buffer = nl.zeros(
        (num_loads, par_dim(B_P_SIZE), tiled_block_size * B_D_SIZE),
        dtype=kernel_dtype,
    )

    for large_tile_idx in nl.sequential_range(0, NUM_LARGE_TILE):

        # Step 4.1: load kv cache into cur_k_tile, cur_v_tile
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
            dtype=kernel_dtype,
        )
        load_kv_tile_from_cache(
            cur_k_tile=cur_k_tile,
            cur_v_tile=cur_v_tile,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=large_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            B_P_SIZE=B_P_SIZE,
            B_D_SIZE=B_D_SIZE,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
        )

        large_q_idx = tile_q_indices_sbuf[0, large_tile_idx]

        # Step 4.2: load aggregation buffers from HBM per q tile
        m_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        l_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        o_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, d),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        # reshape just return a new view, and no copy will occur
        m_sbuf_tile_flattened = m_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * 1)
        )
        l_sbuf_tile_flattened = l_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * 1)
        )
        o_sbuf_tile_flattened = o_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * d)
        )
        m_sbuf_tile_flattened[...] = nl.load(m_buffer[:, large_q_idx])
        l_sbuf_tile_flattened[...] = nl.load(l_buffer[:, large_q_idx])
        o_sbuf_tile_flattened[...] = nl.load(o_buffer[:, large_q_idx])

        q_sbuf_tile = nl.ndarray(
            (q_h_per_k_h, par_dim(B_D_SIZE), LARGE_Q_TILE_SIZE), dtype=kernel_dtype
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            q_hbm_tile = nl.load(
                query[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    :,
                    large_q_idx,
                    :,
                ]
            )
            if kernel_dtype != query.dtype:
                q_hbm_tile = nl.copy(q_hbm_tile, dtype=kernel_dtype)
            q_sbuf_tile[i_q_h, :, :] = q_hbm_tile

        # Step 4.3: flash attention computation
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            cur_mask = nl.load(
                tile_masks[large_tile_idx, nl.ds(small_q_idx * B_P_SIZE, B_P_SIZE), :],
                dtype=tile_masks.dtype,
            )
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_tile[:, :] = (
                    q_sbuf_tile[i_q_h, :, nl.ds(small_q_idx * B_P_SIZE, B_P_SIZE)] * softmax_scale
                )

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_sbuf_tile[:, small_q_idx, i_q_h],
                    l_buffer=l_sbuf_tile[:, small_q_idx, i_q_h],
                    m_buffer=m_sbuf_tile[:, small_q_idx, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=False,
                    q_tile_idx=None,
                    initialize=False,
                    LARGE_TILE_SZ=LARGE_KV_TILE_SIZE,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )
        # Step 4.4: save intermediate results in aggregation buffers
        nl.store(m_buffer[:, large_q_idx], m_sbuf_tile_flattened)
        nl.store(l_buffer[:, large_q_idx], l_sbuf_tile_flattened)
        nl.store(o_buffer[:, large_q_idx], o_sbuf_tile_flattened)

    # ========== Step 5: Attention Computation For Active Part ============= #

    # Step 5.1: Load l, m, o from HBM to SBUF
    # No need to load KV cache because it is for the active part
    o_buffer_sbuf = nl.ndarray(
        (n_large_q_tile, n_small_in_large_q_tile, q_h_per_k_h, par_dim(B_P_SIZE), d),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (n_large_q_tile, n_small_in_large_q_tile, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (n_large_q_tile, n_small_in_large_q_tile, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
    )
    for i0 in nl.affine_range(n_large_q_tile):
        for i1 in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                offset = i1 * q_h_per_k_h + i_q_h
                o_buffer_sbuf[i0, i1, i_q_h] = nl.load(
                    o_buffer[:, i0, nl.ds(offset * B_D_SIZE, B_D_SIZE)]
                )
                l_buffer_sbuf[i0, i1, i_q_h] = nl.load(l_buffer[:, i0, nl.ds(offset, 1)])
                m_buffer_sbuf[i0, i1, i_q_h] = nl.load(m_buffer[:, i0, nl.ds(offset, 1)])

    # Step 5.2: compute attention between input query, key and value.
    if key is not None and value is not None:
        b_f_size = min(seqlen_q, B_F_SIZE)
        LARGE_Q_TILE_SIZE = seqlen_q
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_Q_TILE_SIZE), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), LARGE_Q_TILE_SIZE // B_P_SIZE * B_D_SIZE), dtype=kernel_dtype
        )

        cur_k_tile[:, :] = nl.load(key[batch_id, head_id, :, :], dtype=cur_k_tile.dtype)

        v_hbm_tile = value[batch_id, head_id]
        # load at granularity of B_P_SIZE
        for v_i in nl.affine_range(LARGE_Q_TILE_SIZE // B_P_SIZE):
            load_v_tile(
                v_hbm_tile=v_hbm_tile,
                cur_v_tile=cur_v_tile,
                large_tile_idx=0,
                v_i=v_i,
                LARGE_TILE_SZ=LARGE_Q_TILE_SIZE,
            )

        for i0 in nl.affine_range(n_large_q_tile):
            for i1 in nl.affine_range(n_small_in_large_q_tile):
                i = i0 * n_small_in_large_q_tile + i1
                cur_mask = nl.load(
                    active_mask[
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                        nl.ds(0, LARGE_Q_TILE_SIZE),
                    ],
                    dtype=active_mask.dtype,
                )
                for i_q_h in nl.sequential_range(q_h_per_k_h):
                    q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                    q_hbm_tile = query[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        :,
                        i0,
                        nl.ds(i1 * B_P_SIZE, B_P_SIZE),
                    ]
                    q_sbuf_tile = nl.load(
                        q_hbm_tile, dtype=kernel_dtype
                    )  # load (d, 128) tile in SBUF
                    q_tile[:, :] = q_sbuf_tile * softmax_scale
                    _flash_attention_core(
                        q_local_tile=q_tile,
                        k=cur_k_tile,
                        v=cur_v_tile,
                        o_buffer=o_buffer_sbuf[i0, i1, i_q_h],
                        l_buffer=l_buffer_sbuf[i0, i1, i_q_h],
                        m_buffer=m_buffer_sbuf[i0, i1, i_q_h],
                        kernel_dtype=kernel_dtype,
                        acc_type=acc_type,
                        tile_mask=cur_mask,
                        use_causal_mask=True,
                        q_tile_idx=i,
                        initialize=False,
                        LARGE_TILE_SZ=LARGE_Q_TILE_SIZE,
                        B_P_SIZE=B_P_SIZE,
                        B_F_SIZE=b_f_size,
                        B_D_SIZE=B_D_SIZE,
                    )

    # Step 5.3: write output to buffer on HBM
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i0 in nl.affine_range(n_large_q_tile):
            for i1 in nl.affine_range(n_small_in_large_q_tile):
                i = i0 * n_small_in_large_q_tile + i1
                out = nl.multiply(
                    o_buffer_sbuf[i0, i1, i_q_h],
                    nl.exp(m_buffer_sbuf[i0, i1, i_q_h] - l_buffer_sbuf[i0, i1, i_q_h]),
                    dtype=kernel_dtype,
                )

                nl.store(
                    o[batch_id, head_id * q_h_per_k_h + i_q_h, nl.ds(i * B_P_SIZE, B_P_SIZE), :],
                    out,
                )
    return o


def flash_attn_varlen_blocksparse_nkifunc(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    n_kv_head=None,
    head_size=None,
    mixed_precision=True,
):
    """
    A wrapper for flash_paged_attention_with_schedule

    This wrapper derives the kernel grid for users automatically.
    """
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        active_mask=active_mask,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
    )

    return flash_paged_attention_with_schedule[1, n_kv_head](**kwargs)
