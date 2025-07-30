import numpy as np
import torch

from neuronx_distributed_inference.modules.chunked_prefill.utils import B_P_SIZE, ceil_div
from neuronx_distributed_inference.models.config import NeuronConfig


def get_max_num_tiles_for_chunked_prefill_schedule(neuron_config: NeuronConfig):
    cp_config = neuron_config.chunked_prefill_config
    assert cp_config is not None
    q_tile_size = cp_config.kernel_q_tile_size
    kv_tile_size = cp_config.kernel_kv_tile_size

    max_q_len = neuron_config.max_context_length
    max_kv_cache_len = neuron_config.max_length * cp_config.max_num_seqs

    assert max_q_len % q_tile_size == 0
    assert max_kv_cache_len % kv_tile_size == 0
    max_num_tiles = int((max_q_len // q_tile_size) * (max_kv_cache_len // kv_tile_size))
    return max_num_tiles


class FlashPASchedule:
    """
    A class to store the schedule for chunked prefill attention kernel.

    The chunked prefill attention kernel only computes for necessary tiles,
    and it is this schedule that decides which tile is necessary.

    This schedule will be computed as a result by
    FlashPagedAttentionSchedulerBase.

    Currently this class uses np to work with NKI baremetal mode
    TODO: use torch to replace np ops
    """

    def __init__(
        self,
        tile_q_indices: torch.Tensor,
        tile_block_table_offsets: torch.Tensor,
        tile_q_seq_ids: torch.Tensor,
        tile_kv_seq_ids: torch.Tensor,
        tile_kv_skip_indices: torch.Tensor,
        block_size: int,
    ):
        """
        Initialize the schedule with tile information and configurations.

        Args:
            tile_q_indices: (num_tiles,)
                A list of tile index on q dim for each necessary tile
            tile_block_table_offsets: (num_tiles,)
                A list of beginning block index on kv dim for each necessary
                tile. Same as tile_kv_indices * (num_blocks_per_kv_tile).
            tile_q_seq_ids: (num_tiles, tile_size_q)
                A list of seq ids on the q dim for each necessary tile, one seq
                id for each row.
            tile_kv_seq_ids: (num_large_tiles, tile_size_kv)
                A list of seq ids on the kv dim for each necessary tile, one
                seq id for each column.
            tile_kv_skip_indices: (num_tiles_to_skip_KV_loading,)
                A list of tile ids which can skip loading KV cache.
            block_size: int
                block size of the KV cache

        Raises:
            AssertionError: If tensor shapes or types are incompatible
        """
        self.tile_q_indices = tile_q_indices.numpy()
        self.tile_block_table_offsets = tile_block_table_offsets.numpy()
        self.tile_q_seq_ids = tile_q_seq_ids.numpy()
        self.tile_kv_seq_ids = tile_kv_seq_ids.numpy()
        self.tile_kv_skip_indices = tile_kv_skip_indices.numpy()
        self.block_size = block_size
        num_tiles = len(self.tile_q_indices)

        for x in [
            self.tile_q_indices,
            self.tile_block_table_offsets,
            self.tile_q_seq_ids,
            self.tile_kv_seq_ids,
        ]:
            assert isinstance(x, np.ndarray)
            assert x.dtype == np.int32
            assert x.shape[0] == num_tiles

        if len(self.tile_kv_skip_indices) > 0:
            assert self.tile_kv_skip_indices[0] > 0, "Can't skip the first tile"
            assert np.all(self.tile_kv_skip_indices < num_tiles)

    @property
    def num_tiles(self):
        """Number of tiles in the schedule"""
        return len(self.tile_q_indices)

    def pad_schedule(self, target_num_tiles: int):
        """
        Pad the number of tiles in the schedule to `target_num_tiles`

        Args:
            target_num_tiles (int): Target number of tiles after padding

        Returns:
            FlashPASchedule: A new schedule with padded arrays

        Note:
            Returns original schedule if padding size is smaller than current size
        """
        num_tiles = self.num_tiles
        if target_num_tiles <= num_tiles:
            return self

        def pad(x, pad_to, pad_value=0):
            shape = x.shape
            pad_width = [(0, pad_to - shape[0])] + [(0, 0)] * (len(shape) - 1)
            return np.pad(x, pad_width, mode="constant", constant_values=pad_value)

        tile_q_indices = pad(self.tile_q_indices, target_num_tiles)
        tile_block_tables_offsets = pad(self.tile_block_table_offsets, target_num_tiles)
        # pad different value for q and kv seq ids so that sequence affiliation mask is False
        tile_q_seq_ids = pad(self.tile_q_seq_ids, target_num_tiles, pad_value=0)
        tile_kv_seq_ids = pad(self.tile_kv_seq_ids, target_num_tiles, pad_value=1)
        tile_kv_skip_indices = np.array(
            self.tile_kv_skip_indices.tolist() + list(range(num_tiles, target_num_tiles)),
            dtype=np.int32,
        )
        return FlashPASchedule(
            torch.tensor(tile_q_indices),
            torch.tensor(tile_block_tables_offsets),
            torch.tensor(tile_q_seq_ids),
            torch.tensor(tile_kv_seq_ids),
            torch.tensor(tile_kv_skip_indices),
            self.block_size,
        )

    def build_tile_masks(self, b_p_size=B_P_SIZE):
        """
        Build a list of mask for each tile for prior attention computation.

        Args:
            b_p_size: the partition dimension of the tile. By default it is 128.

        Output:
            tile_mask: (num_tiles, tile_size_q, tile_size_kv)
                A list of a boolean mask for each tile
        """
        tile_kv_seq_ids = self.tile_kv_seq_ids
        num_tiles, tile_size_kv = tile_kv_seq_ids.shape
        assert tile_size_kv % b_p_size == 0 and tile_size_kv % self.block_size == 0

        dim1 = max(b_p_size, tile_size_kv // self.block_size)
        dim2 = tile_size_kv // dim1
        # To maximize the DMA efficiency, we can rearange tile_kv_seq_ids
        # when applicable.
        #
        # In that case, its layout will change like
        #    (num_tiles, tile_size_kv)
        # -> (num_tiles, dim_a, dim_b, dim_c)
        # where dim_a * dim_b * dim_c = tile_size_kv, and dim_b needs to be
        # equal to max partition size in the tensor engine (b_p_size).
        if dim2 > 1:
            tile_kv_seq_ids = tile_kv_seq_ids.reshape(
                (
                    num_tiles,
                    dim1 // b_p_size,
                    b_p_size,
                    dim2,
                )
            )
            tile_kv_seq_ids = tile_kv_seq_ids.transpose(0, 1, 3, 2).reshape(
                (num_tiles, tile_size_kv)
            )
        tile_masks = np.expand_dims(self.tile_q_seq_ids, 2) == np.expand_dims(tile_kv_seq_ids, 1)
        return torch.from_numpy(tile_masks)

    def build_tile_block_tables(
        self, block_tables: torch.Tensor, skip_value: int,
    ):
        """
        Construct block tables for each tile for KV cache access.

        Args:
            block_tables (torch.Tensor): Input block tables
            skip_value (int): Value to use as kv block id to skip
                loading, and it should be out-of-bound.

        Returns:
            torch.Tensor: Shape (num_tiles, num_blocks_per_kv_tile) block tables
        """
        tile_size_kv = self.tile_kv_seq_ids.shape[1]
        assert tile_size_kv % self.block_size == 0
        num_blocks_per_tile = tile_size_kv // self.block_size

        # block_table could be empty at the begining since it doesn't need to
        # read KV cache
        if block_tables.shape[0] == 0:
            return torch.zeros((0, num_blocks_per_tile), dtype=torch.int32)

        block_table_len = block_tables.shape[0]

        block_tables = block_tables.numpy()
        in_tile_offset = np.arange(num_blocks_per_tile)
        indices = self.tile_block_table_offsets.reshape(-1, 1) + in_tile_offset.reshape(1, -1)

        # to avoid OOB for padding
        indices[indices >= block_table_len] = 0

        tile_block_tables = block_tables[indices]

        if len(self.tile_kv_skip_indices) > 0:
            tile_block_tables[self.tile_kv_skip_indices, :] = skip_value

        # Always return a 2D tensor
        if tile_block_tables.ndim == 1:
            tile_block_tables = tile_block_tables.reshape(-1, num_blocks_per_tile)
        return torch.from_numpy(tile_block_tables)

    def get_tile_q_indices(self):
        return torch.from_numpy(self.tile_q_indices)


class FlashPagedAttentionSchedulerBase:
    """
    A base class to generate a schedule for flash attention kernel for
    chunked prefill. The schedule should be an instance of FlashPASchedule.
    """

    def __init__(
        self,
        prompt_lens: torch.Tensor,
        context_lens: torch.Tensor,
        tile_size_q: int,
        tile_size_kv: int,
        block_size: int,
    ):
        """
        Args:
            prompt_lens: query lens for each seq
            context_lens: prior kv lens for each seq. This doesn't include the
                active kv lens.
            tile_size_q: tile size on the q dim
            tile_size_kv: tile size on the kv dim
            block_size: block size of kv cache

        # TODO: replace np with torch
        """
        prompt_lens = prompt_lens.numpy()
        context_lens = context_lens.numpy()
        assert self._check_np_int_array(prompt_lens, context_lens)
        assert len(prompt_lens) == len(
            context_lens
        ), "prompt_lens and context_lens must have the same length"
        self.num_seq = len(prompt_lens)
        assert self.num_seq > 0, "prompt_lens and context_lens must be non-empty"
        self.prompt_lens = prompt_lens.astype(np.int32)
        self.context_lens = context_lens.astype(np.int32)
        self.tile_size_q = tile_size_q
        self.tile_size_kv = tile_size_kv
        self.block_size = block_size

    def _check_np_int_array(self, *arrays):
        for a in arrays:
            if not isinstance(a, np.ndarray) or a.dtype not in (np.int32, np.int64):
                return False
        return True

    def compute_schedule(self) -> FlashPASchedule:
        """
        Abstract method to generate a schedule
        """
        raise NotImplementedError


class GridTileScheduler(FlashPagedAttentionSchedulerBase):
    """
    A scheduler for chunked prefill attn kernel based on tile grid. This
    class compares a tile grid with an attn mask, and generate a schedule
    that only pick necessary tiles for attn computation.
    """

    def __init__(
        self,
        prompt_lens: torch.Tensor,
        context_lens: torch.Tensor,
        tile_size_q: int,
        tile_size_kv: int,
        block_size: int,
        column_order: bool = True,
    ):
        """
        Construct a scheduler that compute schedules based on a grid.

        If we want to use DMA skipping to skip loading the same KV cache
        for different querys (same column but different rows), we need to
        set column_order to True. This will pick up the tiles column by
        column , so we cleary know which tile can skip the loading.
        """
        super(__class__, self).__init__(
            prompt_lens,
            context_lens,
            tile_size_q,
            tile_size_kv,
            block_size,
        )
        self.column_order = column_order

    def _get_seq_start_end(self, seqlens, padded_seqlens=None):
        """
        For chunked prefil, we concatenate multiple prompts to a single prompt
        for computation.
        This function computes the start, end indices & total length of a
        prompt, once we flatten a list of input prompt lengths.

        Args:
            seqlens (np.ndarray): Array of original sequence lengths
            padded_seqlens (np.ndarray, optional): Array of padded sequence
                lengths. If None, uses original seqlens

        Returns:
            tuple: A tuple containing:
                - seqlens_starts (np.ndarray): Starting indices for each sequence
                - seqlens_ends (np.ndarray): Ending indices for each sequence
                - total_padded_length (int): Total length of concatenated
                    sequences, including padding

        For example, assuming
            seq_lens = [31,19,13]
        then the output is
            seqlen_starts = [0, 31, 50]
            seqlen_ends = [31, 50, 63]
            total_length = 63
        """
        if padded_seqlens is None:
            padded_seqlens = seqlens
        cu_seqlen = np.cumsum(padded_seqlens)
        seqlens_starts = np.concatenate(([0], cu_seqlen[:-1]))
        seqlens_ends = seqlens_starts + seqlens
        return seqlens_starts, seqlens_ends, cu_seqlen[-1]

    def compute_schedule(self):
        """
        Generate a grid-based schedule for the block sparse flash attention
        kernel.

        Based on the query lens, kv lens and block size, it can figure out
        which areas in attention mask need computation. And then it split
        the computation into tiles based on a fixed-size grid, and only
        pick up the tiles that needs computation.

        This scheduling only works for the prior part, so the active part is
        not considered here.

        # TODO: replace np with torch and add it inside neff
        """

        # Assuming:
        #   prompt_lens = [10, 1, 3]
        #   context_lens = [3, 6, 5]
        #   tile_size_q = 6
        #   tile_size_kv = 8
        #   block_size = 4
        #   column_order = True

        # =========== Step 1: Find the required tiles =========== #

        # This func computes the required tiles based on the following
        # tile_needed = max(q_id_start, kv_id_start) < min(q_id_end, kv_id_end)

        # Step 1.1: Compute starting and ending indices for prompts and
        # contexts. Need to pad the context because KV cache is in discret
        # blocks.
        num_context_blocks = ceil_div(self.context_lens, self.block_size)
        padded_context_lens = num_context_blocks * self.block_size
        context_starts, context_ends, total_seqlen_kv = self._get_seq_start_end(
            self.context_lens, padded_seqlens=padded_context_lens
        )
        prompt_starts, prompt_ends, total_seqlen_q = self._get_seq_start_end(self.prompt_lens)
        # Results till here:
        #   padded_context_lens=[ 4, 8, 8]
        #   context_starts=[ 0,  4, 12], context_ends=[ 3, 10, 17], total_seqlen_kv=20
        #   prompt_starts= [ 0, 10, 11], prompt_ends= [10, 11, 14], total_seqlen_q= 14

        # Step 1.2: Get the first and last (excluding) seq id on q dimension
        tile_q_starts = np.arange(0, total_seqlen_q, self.tile_size_q)
        tile_q_ends = tile_q_starts + self.tile_size_q
        # tile_q_seq_starts: the first seq id in each tile on q dim
        tile_q_seq_starts = np.searchsorted(prompt_ends, tile_q_starts, side="right")
        # tile_q_seq_ends: the (last seq id + 1) in each tile on q dim
        tile_q_seq_ends = np.searchsorted(prompt_starts, tile_q_ends, side="left")
        # Results till here:
        #   tile_q_starts: [ 0,  6, 12]
        #   tile_q_ends:   [ 6, 12, 18]
        #   tile_q_seq_starts: [0, 0, 2]
        #   tile_q_seq_ends:   [1, 3, 3]

        # Step 1.3: Get the first and last (excluding) seq id on kv dimension
        tile_kv_starts = np.arange(0, total_seqlen_kv, self.tile_size_kv)
        tile_kv_ends = tile_kv_starts + self.tile_size_kv
        # tile_kv_seq_starts: the first seq id in each tile on kv dim
        tile_kv_seq_starts = np.searchsorted(context_ends, tile_kv_starts, side="right")
        # tile_kv_seq_ends: the (last seq id + 1) in each tile on kv dim
        tile_kv_seq_ends = np.searchsorted(context_starts, tile_kv_ends, side="left")
        # Results till here:
        #   tile_kv_starts: [ 0,  8, 16]
        #   tile_kv_ends:   [ 8, 16, 24]
        #   tile_kv_seq_starts: [0, 1, 2]
        #   tile_kv_seq_ends:   [2, 3, 3]

        # Step 1.4: Get the required tile ids
        tile_seq_starts = np.maximum(
            tile_q_seq_starts.reshape(-1, 1), tile_kv_seq_starts.reshape(1, -1)
        )
        tile_seq_ends = np.minimum(tile_q_seq_ends.reshape(-1, 1), tile_kv_seq_ends.reshape(1, -1))
        # A boolean 2D matrix to show if a tile is necessary for attn computation
        # shape (num_tiles_on_q_dim, num_tiles_on_kv_dim)
        tile_needed = tile_seq_starts < tile_seq_ends
        tile_q_indices, tile_kv_indices = np.nonzero(tile_needed)
        # Results till here:
        #   tile_seq_starts: [[0, 1, 2],[0, 1, 2],[2, 2, 2]]
        #   tile_seq_ends:   [[1, 1, 1],[2, 3, 3],[2, 3, 3]]
        #   tile_needed:     [[T, F, F],[T, T, T],[F, T, T]]
        #   tile_q_indices:  [0, 1, 1, 1, 2, 2]
        #   tile_kv_indices: [0, 0, 1, 2, 1, 2]

        # =========== Step 2: Compute auxiliary inputs =========== #

        # Step 2.1: Get seq_id per slot on q dim and kv dim
        num_q_tiles = len(tile_q_starts)
        num_kv_tiles = len(tile_kv_starts)
        # A 2D matrix to show the seq id per row for each q tile, and it uses
        # num_seq as padding value.
        # shape (num_q_tiles, q_tile_size)
        q_seq_ids = np.repeat(
            np.arange(self.num_seq + 1, dtype=np.int32),
            np.concatenate((self.prompt_lens, [num_q_tiles * self.tile_size_q - total_seqlen_q])),
        ).reshape((num_q_tiles, self.tile_size_q))
        # Results till here:
        #   q_seq_ids: [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 2],[2, 2, 3, 3, 3, 3]]

        # A 2D matrix to show the seq id per col for each kv tile, and it
        # uses num_seq + 1 as padding, to avoid conflict with padding for
        # q_seq_ids.
        # shape (num_kv_tiles, kv_tile_size)
        kv_seq_ids = np.repeat(
            np.stack(
                (
                    np.arange(self.num_seq, dtype=np.int32),
                    # use num_seq + 1 as padding, to avoid conflict with
                    # padding for q_seq_ids
                    np.full((self.num_seq,), self.num_seq + 1, dtype=np.int32),
                )
            ).flatten("F"),
            np.stack(
                (
                    self.context_lens,
                    padded_context_lens - self.context_lens,
                )
            ).flatten("F"),
        )
        # Results till here:
        #   kv_seq_ids: [0, 0, 0, 4, 1, 1, 1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4]

        kv_seq_ids = np.concatenate(
            (
                kv_seq_ids,
                np.full(
                    (num_kv_tiles * self.tile_size_kv - total_seqlen_kv,),
                    self.num_seq + 1,
                    dtype=np.int32,
                ),
            )
        ).reshape((num_kv_tiles, self.tile_size_kv))
        # Results till here:
        #   kv_seq_ids:
        #       [[0, 0, 0, 4, 1, 1, 1, 1],
        #        [1, 1, 4, 4, 2, 2, 2, 2],
        #        [2, 4, 4, 4, 4, 4, 4, 4]]

        # Step 2.2: Reorder tiles orders in a column-first mannar, so it
        # provides opportunites for DMA skipping.
        if self.column_order:
            sort_indices = np.argsort(tile_kv_indices, kind="stable")
            tile_q_indices = tile_q_indices[sort_indices]
            tile_kv_indices = tile_kv_indices[sort_indices]
            # Results till here:
            #   sort_indices:    [0, 1, 2, 4, 3, 5]
            #   tile_q_indices:  [0, 1, 1, 2, 1, 2]
            #   tile_kv_indices: [0, 0, 1, 1, 2, 2]

        tile_q_indices = tile_q_indices.astype(np.int32)
        # get block table starting ids, after padding them to block size
        tile_kv_offsets = tile_kv_indices.astype(np.int32) * self.tile_size_kv
        tile_bt_offsets = tile_kv_offsets // self.block_size
        # reorder seq ids for each tile after sorting them in column order
        tile_q_seq_ids = q_seq_ids[tile_q_indices]
        tile_kv_seq_ids = kv_seq_ids[tile_kv_indices]
        # Results till here:
        #   tile_q_indices= [0, 1, 1, 2, 1, 2]
        #   tile_kv_offsets=[0, 0, 8, 8, 16, 16]
        #   tile_bt_offsets=[0, 0, 2, 2, 4, 4]
        #   tile_q_seq_ids=
        #         [[0, 0, 0, 0, 0, 0],
        #          [0, 0, 0, 0, 1, 2],
        #          [0, 0, 0, 0, 1, 2],
        #          [2, 2, 3, 3, 3, 3],
        #          [0, 0, 0, 0, 1, 2],
        #          [2, 2, 3, 3, 3, 3]]
        #   tile_kv_seq_ids=
        #         [[0, 0, 0, 4, 1, 1, 1, 1],
        #          [0, 0, 0, 4, 1, 1, 1, 1],
        #          [1, 1, 4, 4, 2, 2, 2, 2],
        #          [1, 1, 4, 4, 2, 2, 2, 2],
        #          [2, 4, 4, 4, 4, 4, 4, 4],
        #          [2, 4, 4, 4, 4, 4, 4, 4]]

        # Step 2.3: Calculate load mask for kv, for DMA skipping
        prev_kv_indices = np.concatenate(([-1], tile_kv_indices[:-1]))
        tile_kv_skip_indices = np.nonzero(tile_kv_indices == prev_kv_indices)[0].astype(np.int32)
        # Results till here:
        #   prev_kv_indices=[-1,  0,  0,  1,  1,  2]
        #   tile_kv_skip_indices=[1, 3, 5]

        return FlashPASchedule(
            tile_q_indices=torch.tensor(tile_q_indices),
            tile_block_table_offsets=torch.tensor(tile_bt_offsets),
            tile_q_seq_ids=torch.tensor(tile_q_seq_ids),
            tile_kv_seq_ids=torch.tensor(tile_kv_seq_ids),
            tile_kv_skip_indices=torch.tensor(tile_kv_skip_indices),
            block_size=self.block_size,
        )
