import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim

from neuronx_distributed_inference.modules.chunked_prefill.utils import B_P_SIZE, ceil_div


@nki.jit
def load_block_tables(block_tables_hbm, num_tiles, num_blocks_per_tile):
    """
    Load tiled block table from HBM to SBUF.

    Args:
        block_tables_hbm (ndarray): Input block table data in HBM, which has a
            shape of (num_tiles, num_blocks_per_tile)
        num_tiles (int): Number of tiles in the block table
        num_blocks_per_tile (int): Number of blocks per tile

    Returns:
        ndarray: Reshaped block table data in SBUF format with dimensions
                (ceil_div(num_tiles, B_P_SIZE), B_P_SIZE, num_blocks_per_tile)

    Raises:
        AssertionError: If input dimensions don't match expected shapes

    """
    if len(block_tables_hbm.shape) == 1:
        (num_total_blocks,) = block_tables_hbm.shape
        assert num_blocks_per_tile * num_tiles == num_total_blocks
        block_tables_hbm = block_tables_hbm.reshape((num_tiles, num_blocks_per_tile))
    else:
        assert tuple(block_tables_hbm.shape) == (num_tiles, num_blocks_per_tile)

    # Reshape block_tables_sbuf to ensure high DMA efficiency
    block_tables_sbuf = nl.zeros(
        (ceil_div(num_tiles, B_P_SIZE), par_dim(B_P_SIZE), num_blocks_per_tile),
        dtype=nl.int32,
    )
    for i in nl.affine_range(ceil_div(num_tiles, B_P_SIZE)):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(num_blocks_per_tile)[None, :]
        block_tables_sbuf[i, i_p, i_f] = nl.load(
            block_tables_hbm[i_p + i * B_P_SIZE, i_f],
            dtype=nl.int32,
            mask=(i_p + i * B_P_SIZE < num_tiles),
        )
    return block_tables_sbuf


@nki.jit
def transform_block_tables_for_indirect_load(
    block_tables,
    block_size_tiling_factor,
    num_head,
    head_id,
):
    """
    Transform block tables for vector DGE (indirect load) by fusing kv_head
    dim and tiling block size. Note that the KV cache will be flattened
    to 2D, so indices in the block tables needs to account for that.

    For Vector DGE, this function will map the indices to the partition
    dimension.

    Args:
        block_tables (ndarray): Input block tables with shape
            (ceil_div(num_tiles, B_P_SIZE), B_P_SIZE, num_blocks_per_tile)
        block_size_tiling_factor (int): Factor for tiling the block size dimension
        num_head (int): Number of attention kv heads
        head_id (int): Identifier for the current attention head

    Returns:
        ndarray: Transformed block tables with shape
            (num_loads, B_P_SIZE, num_partitions * num_tiles_per_partition)

    Raises:
        AssertionError: If dimensions don't match expected configurations
    """
    num_partitions, num_tiles_per_partition, num_blocks_per_tile = block_tables.shape
    assert num_tiles_per_partition == B_P_SIZE

    # num_partitions * num_tiles_per_partition is actually num_tiles rounded
    # to B_P_SIZE (128)
    num_loads = ceil_div(num_blocks_per_tile, B_P_SIZE)
    block_tables_transposed = nl.ndarray(
        (num_loads, par_dim(B_P_SIZE), num_partitions * num_tiles_per_partition),
        dtype=nl.int32,
    )

    # prepare iota ahead of time to avoid repeatedly using Gpsimd
    if num_head > 1:
        head_id = nisa.iota(head_id, dtype=nl.int32).reshape((1, 1))
        head_id = nl.transpose(head_id.broadcast_to((1, num_tiles_per_partition)))
        if num_blocks_per_tile > 1:
            head_id = head_id.broadcast_to((num_tiles_per_partition, num_blocks_per_tile))

    if block_size_tiling_factor > 1:
        broadcast_shape = (
            num_tiles_per_partition,
            num_blocks_per_tile,
            block_size_tiling_factor,
        )
        offset = nisa.iota(
            nl.arange(block_size_tiling_factor)[None, None, :], dtype=nl.int32
        ).broadcast_to(broadcast_shape)

    for partition_id in nl.affine_range(num_partitions):
        # shape: (B_P_SIZE, num_blocks_per_tile)
        block_tables_partition = block_tables[partition_id]

        # update index value to account for the kv_head_dim, since KV cache
        # will be flattened to 2D and its 1st dim is (num_blocks * num_kv_heads
        # * block_size_tiling_factor)
        if num_head > 1:
            # fuse num_block and num_head dimension
            block_tables_partition = block_tables_partition * num_head + head_id

        # update index value to account for the tiling_factor, since KV cache
        # will be flattened to 2D and its 1st dim is (num_blocks * num_kv_heads
        # * block_size_tiling_factor)
        if block_size_tiling_factor > 1:
            assert num_blocks_per_tile * block_size_tiling_factor == B_P_SIZE
            block_tables_partition = (
                (block_tables_partition * block_size_tiling_factor)
                .reshape((num_tiles_per_partition, num_blocks_per_tile, 1))
                .broadcast_to(broadcast_shape)
            )
            new_block_tables = block_tables_partition + offset
            new_block_tables = new_block_tables.reshape((num_tiles_per_partition, B_P_SIZE))
        else:
            new_block_tables = block_tables_partition

        # transpose the block table so that it can be used by vector DGE
        for i in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = (
                partition_id * num_tiles_per_partition + nl.arange(num_tiles_per_partition)[None, :]
            )
            block_tables_transposed[i, i_p, i_f] = nl.transpose(
                new_block_tables[:, nl.ds(i * B_P_SIZE, B_P_SIZE)]
            )
    return block_tables_transposed


@nki.jit
def load_kv_tile_from_cache(
    cur_k_tile,
    cur_v_tile,
    key_cache,
    value_cache,
    block_tables,
    large_k_tile_idx,
    num_blocks_per_large_tile,
    tiled_block_size,
    B_P_SIZE,
    B_D_SIZE,
    k_load_buffer=None,
    v_load_buffer=None,
):
    """
    Load KV cache from HBM to SBUF using block tables for indirect addressing.

    This function also transpose the K cache to prepare for QK matmul later.

    Args:
        cur_k_tile (ndarray): Key tile on SBUF to store loaded data
            (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE)
        cur_v_tile (ndarray): Value tile on SBUF to store loaded data
            (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE)
        key_cache (ndarray): Key cache on HBM
        value_cache (ndarray): Value cache on HBM
        block_tables (ndarray): block table for indirect addressing
        large_k_tile_idx (int): Index of the large key tile
        num_blocks_per_large_tile (int): Number of blocks in each large tile
        tiled_block_size (int): Size of each tiled block
        k_load_buffer (ndarray, optional): Buffer for key loading for DMA
            skipping
        v_load_buffer (ndarray, optional): Buffer for value loading for DMA
            skipping
    """
    num_loads = ceil_div(num_blocks_per_large_tile, B_P_SIZE)

    # load key cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        if k_load_buffer is None:
            loaded = nl.load(
                key_cache[block_tables[load_idx, i_p, large_k_tile_idx], i_f],
                dtype=cur_k_tile.dtype,
                mode=oob_mode.error,
            )
        else:
            k_load_buffer[load_idx, i_p, i_f] = nl.load(
                key_cache[block_tables[load_idx, i_p, large_k_tile_idx], i_f],
                dtype=cur_k_tile.dtype,
                mode=oob_mode.skip,  # DMA skipping
            )
            loaded = k_load_buffer[load_idx]

        # Transpose SBUF tensor using PE
        for tb_i in nl.affine_range(tiled_block_size):
            cur_k_tile[
                :,
                nl.ds(
                    load_idx * B_P_SIZE * tiled_block_size + tb_i * B_P_SIZE,
                    B_P_SIZE,
                ),
            ] = nl.transpose(loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)])

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        if v_load_buffer is None:
            loaded = nl.load(
                value_cache[block_tables[load_idx, i_p, large_k_tile_idx], i_f],
                dtype=cur_v_tile.dtype,
                mode=oob_mode.error,
            )
        else:
            v_load_buffer[load_idx, i_p, i_f] = nl.load(
                value_cache[block_tables[load_idx, i_p, large_k_tile_idx], i_f],
                mode=oob_mode.skip,  # DMA skipping
            )
            loaded = v_load_buffer[load_idx]
        cur_v_tile[
            :,
            nl.ds(
                load_idx * tiled_block_size * B_D_SIZE,
                tiled_block_size * B_D_SIZE,
            ),
        ] = loaded
