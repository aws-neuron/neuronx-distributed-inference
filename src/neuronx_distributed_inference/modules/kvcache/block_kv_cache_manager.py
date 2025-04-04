from math import ceil, log2
from typing import List

import torch
from torch import Tensor

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager


class BlockKVCacheManager(KVCacheManager):
    """
    Key Value cache management with block layout

    It stores KV cache as a parameter list of the shape (num_blocks, block_size, num_kv_head_per_rank, head_dim),
    and vends out read and write operations.

    """

    # Reserve extra blocks to serve as the destination of non-active KV during update
    _NUM_EXTRA_RESERVED_BLOCK = 1

    def __init__(self, config: InferenceConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.pa_num_blocks = config.neuron_config.pa_num_blocks
        self.pa_block_size = config.neuron_config.pa_block_size

        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill
        self.is_prefix_caching = config.neuron_config.is_prefix_caching

    def _init_kv_shape(self, config: InferenceConfig):
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        block_size = config.neuron_config.pa_block_size
        max_num_blocks_per_seq = (config.neuron_config.max_length + block_size - 1) // block_size
        if config.neuron_config.is_prefix_caching and max_num_blocks_per_seq < 128:
            # Enable tiling on block_size dimension to avoid V cache transpose.
            # The tiling factor is the smallest power of 2 that's larger than or equal to
            # 128 / max_num_blocks_per_seq, so that the block_size dimension can be
            # correctly tiled (assuming the block_size is always a power of 2).
            tiling_factor = BlockKVCacheManager._find_next_power_2(128 / max_num_blocks_per_seq)
            self.kv_shape = (
                config.neuron_config.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK,
                tiling_factor,
                config.neuron_config.pa_block_size // tiling_factor,
                num_kv_heads_per_rank,
                hidden_dim_per_head,
            )
            self.block_tiling = True
            self.block_tiling_factor = tiling_factor
        else:
            self.kv_shape = (
                config.neuron_config.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK,
                config.neuron_config.pa_block_size,
                num_kv_heads_per_rank,
                hidden_dim_per_head,
            )
            self.block_tiling = False
            self.block_tiling_factor = -1

    @staticmethod
    def _find_next_power_2(x):
        return 2 ** ceil(log2(x))

    def get_cache(self, **kwargs):
        """
        Get cache for paged attention using an active block table.

        An active block table will only have padding block at the end, not
        between blocks.
        """
        active_block_table = kwargs.get("active_block_table")
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            k_cache, v_cache = self.get_kv_by_layer_id(key_layer_idx)

            if self.is_prefix_caching:
                key_state = self._get_cache_from_block_table_pc(k_cache, active_block_table)
                value_state = self._get_cache_from_block_table_pc(v_cache, active_block_table)
            elif self.is_chunked_prefill:
                key_state = self._get_cache_from_block_table_cp(k_cache, active_block_table)
                value_state = self._get_cache_from_block_table_cp(v_cache, active_block_table)
            else:
                raise ValueError("Can't find a proper way to read block KV cache.")

            past_key_values.append([key_state, value_state])

        return past_key_values

    def get_kv_by_layer_id(self, key_layer_idx, gather_index=None, slice_index=None):
        # Hide the extra blocks from being accessed during cache read
        k_cache = self.past_key_values[key_layer_idx][: self.pa_num_blocks]
        v_cache = self.past_key_values[key_layer_idx + 1][: self.pa_num_blocks]
        return k_cache, v_cache

    def _get_cache_from_block_table_pc(self, cache: Tensor, active_block_table: Tensor):
        """
        Reorder the cache based on the table indices from active_block_table, and return
        them in BHSD layout.

        This is for prefix caching only.

        Args:
            cache: cache in block layout in shape (max_blocks, block_size, num_heads_per_rank, head_dimension)
            active_block_table: indices of precomputed cache blocks in shape (batch_size, max_blocks_per_seq)

        Returns:
            cache: reordered cache in BHSD layout
        """
        num_heads_per_rank, head_dimension = cache.shape[-2], cache.shape[-1]
        batch_size, _ = active_block_table.shape

        if self.block_tiling:
            _, _, num_block_tiles, num_heads_per_rank, head_dimension = cache.shape
            cache_reshaped = cache.reshape(-1, num_block_tiles, num_heads_per_rank, head_dimension)
            index_array = active_block_table.reshape(-1) * self.block_tiling_factor
            index_array = index_array.unsqueeze(-1) + torch.arange(self.block_tiling_factor)
            selected_cache = cache_reshaped.index_select(
                dim=0, index=index_array.reshape(-1)
            ).reshape(batch_size, -1, num_heads_per_rank, head_dimension)
        else:
            selected_cache = cache.index_select(
                dim=0, index=active_block_table.reshape(-1)
            ).reshape(batch_size, -1, num_heads_per_rank, head_dimension)

        selected_cache = selected_cache.permute((0, 2, 1, 3))  # BSHD to BHSD
        return selected_cache

    def _get_cache_from_block_table_cp(self, cache: Tensor, active_block_table: Tensor):
        """
        Read KV cache and return it in block layout.

        This is for chunked prefill only.
        """
        selected_cache = cache.index_select(dim=0, index=active_block_table)
        return selected_cache

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        new_key_values: List[Tensor],
        seq_len: int,
        scatter_index=None,
        **kwargs,
    ):
        """
        Write the KV cache for paged attention

        The slot_mapping will be passed as scatter_index
        """
        slot_mapping = scatter_index
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            k_cache = self._update_cache_into_block_layout(
                latest=kv_per_layer[0],
                cache=self.past_key_values[idx * 2],
                slot_mapping=slot_mapping,
            )
            v_cache = self._update_cache_into_block_layout(
                latest=kv_per_layer[1],
                cache=self.past_key_values[idx * 2 + 1],
                slot_mapping=slot_mapping,
            )

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return updated_kv_cache

    def _update_cache_into_block_layout(self, latest, cache, slot_mapping, padding_id=-1):
        """
        Write the latest KV into cache, where the cache is in block layout

        Args:
            latest: the newly generated KV cache in shape (batch_size, num_heads_per_rank, n_active_tokens, head_dim)
            cache: the KV cache to be updated in block layout in shape (max_blocks, block_size, num_heads_per_rank, head_dimension)
            slot_mapping: the mapping of position to block slot in shape (batch_size, n_active_tokens)
            padding_id: the padding id for non-active slots in slot_mapping

        Returns:
            cache: updated KV cache in block layout in shape (max_blocks, block_size, num_heads_per_rank, head_dimension)
        """
        batch_size, num_heads_per_rank, n_active_tokens, head_dim = latest.shape
        latest = latest.permute((0, 2, 1, 3))
        latest = latest.reshape((batch_size * n_active_tokens, num_heads_per_rank * head_dim))

        if self.block_tiling:
            num_blocks, block_tiling_factor, num_block_tiles, num_heads_per_rank, head_dim = (
                cache.shape
            )
        else:
            num_blocks, block_size, num_heads_per_rank, head_dim = cache.shape
        cache = cache.reshape((-1, num_heads_per_rank * head_dim))

        slot_mapping = slot_mapping.reshape((batch_size * n_active_tokens, 1))
        # Ensure the non-active KV are scattered to the extra reserved blocks
        # instead of pollute existing blocks.
        dtype = slot_mapping.dtype
        device = slot_mapping.device

        if self.block_tiling:
            pad_dest_index = torch.tensor(
                (num_blocks - 1) * block_tiling_factor * num_block_tiles, device=device, dtype=dtype
            )
        else:
            pad_dest_index = torch.tensor((num_blocks - 1) * block_size, device=device, dtype=dtype)

        slot_mapping = torch.where(
            slot_mapping == padding_id,
            pad_dest_index,
            slot_mapping,
        )
        slot_mapping = slot_mapping.expand(
            (batch_size * n_active_tokens, num_heads_per_rank * head_dim)
        )

        cache = torch.scatter(input=cache, dim=0, index=slot_mapping, src=latest)
        if self.block_tiling:
            cache = cache.reshape(
                (num_blocks, block_tiling_factor, num_block_tiles, num_heads_per_rank, head_dim)
            )
        else:
            cache = cache.reshape((num_blocks, block_size, num_heads_per_rank, head_dim))

        return cache
