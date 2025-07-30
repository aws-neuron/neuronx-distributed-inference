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

        # tiling attributes for prefix caching
        self.block_tiling = False
        self.block_tiling_factor = -1

        # self._init_kv_shape() will be invoked in parent's init method, so
        # param needed inside self._init_kv_shape() should be initialized
        # beforehand to avoid override.
        super().__init__(config, **kwargs)

        self.pa_num_blocks = config.neuron_config.pa_num_blocks
        self.pa_block_size = config.neuron_config.pa_block_size

        self.is_chunked_prefill = config.neuron_config.is_chunked_prefill
        self.is_prefix_caching = config.neuron_config.is_prefix_caching

    def _init_kv_shape(self, config: InferenceConfig, layer_to_cache_size_mapping=None):
        # This func is called before finishing the invocation of the
        # self.__init__(), so we can't use child attributes like
        # self.is_chunked_prefill here.
        if config.neuron_config.is_prefix_caching:
            self._init_kv_shape_for_prefix_caching(config)
        elif config.neuron_config.is_chunked_prefill:
            self._init_kv_shape_for_chunked_prefill(config)

    def _init_kv_shape_for_prefix_caching(self, config: InferenceConfig):
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        block_size = config.neuron_config.pa_block_size
        max_num_blocks_per_seq = (config.neuron_config.max_length + block_size - 1) // block_size
        if config.neuron_config.is_prefix_caching and max_num_blocks_per_seq < 128 and \
                not (config.neuron_config.attn_tkg_nki_kernel_enabled or config.neuron_config.attn_block_tkg_nki_kernel_enabled):
            # Enable tiling on block_size dimension to avoid V cache transpose.
            # The tiling factor is the smallest power of 2 that's larger than or equal to
            # 128 / max_num_blocks_per_seq, so that the block_size dimension can be
            # correctly tiled (assuming the block_size is always a power of 2).
            tiling_factor = BlockKVCacheManager._find_next_power_2(128 / max_num_blocks_per_seq)
            self.k_shape = self.v_shape = (
                config.neuron_config.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK,
                tiling_factor,
                config.neuron_config.pa_block_size // tiling_factor,
                num_kv_heads_per_rank,
                hidden_dim_per_head,
            )
            self.block_tiling = True
            self.block_tiling_factor = tiling_factor
        else:
            self.k_shape = self.v_shape = (
                config.neuron_config.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK,
                config.neuron_config.pa_block_size,
                num_kv_heads_per_rank,
                hidden_dim_per_head,
            )

    def _init_kv_shape_for_chunked_prefill(self, config: InferenceConfig):
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        self.k_shape = self.v_shape = (
            config.neuron_config.pa_num_blocks,
            num_kv_heads_per_rank,
            config.neuron_config.pa_block_size,
            hidden_dim_per_head,
        )

    @staticmethod
    def _find_next_power_2(x):
        return 2 ** ceil(log2(x))

    def get_cache(self, active_block_table, kvcache_buffer=None, **kwargs):
        """
        Get cache for paged attention using an active block table.

        An active block table will only have padding block at the end, not
        between blocks.
        """
        past_key_values = []
        for idx in range(len(self.past_key_values) // 2):
            k_cache, v_cache = self.get_kv_by_layer_id(
                idx, active_block_table, kvcache_buffer=kvcache_buffer, **kwargs,
            )
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def _fetch_cache(self, idx: int, kvcache_buffer=None):
        if kvcache_buffer is None:
            k_cache = self.past_key_values[2 * idx]
            v_cache = self.past_key_values[2 * idx + 1]
        else:
            if (
                len(kvcache_buffer) == len(self.past_key_values) // 2
                and len(kvcache_buffer[0]) == 2
            ):
                k_cache = kvcache_buffer[idx][0]
                v_cache = kvcache_buffer[idx][1]
            elif len(kvcache_buffer) == len(self.past_key_values):
                k_cache = kvcache_buffer[2 * idx]
                v_cache = kvcache_buffer[2 * idx + 1]
            else:
                raise ValueError(
                    f"Received kvcache_buffer has length {len(kvcache_buffer)}"
                    f"kvcache_buffer must be a list of 2 element tuples of length {len(self.past_key_values) // 2}"
                    f"or a flat list of length {len(self.past_key_values)}"
                )
        return k_cache, v_cache

    def get_kv_by_layer_id(self, idx, active_block_table, kvcache_buffer=None, **kwargs):
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer=kvcache_buffer)
        if self.is_prefix_caching:
            key_state = self._get_block_cache_and_reshape_bhsd(k_cache, active_block_table)
            value_state = self._get_block_cache_and_reshape_bhsd(v_cache, active_block_table)
        elif self.is_chunked_prefill:
            is_for_context_encoding = kwargs.get("is_for_context_encoding", False)
            key_state = self._get_cache_for_chunked_prefill(k_cache, active_block_table, is_for_context_encoding)
            value_state = self._get_cache_for_chunked_prefill(v_cache, active_block_table, is_for_context_encoding)
        else:
            raise ValueError("Can't find a proper way to read block KV cache.")

        return key_state, value_state

    def _get_block_cache_and_reshape_bhsd(self, cache: Tensor, active_block_table: Tensor):
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

    def _get_cache_for_chunked_prefill(
        self,
        cache: Tensor,
        active_block_table: Tensor,
        is_for_context_encoding: bool
    ):
        """
        Read KV cache for chunked prefill.

        For CTE, it return the whole cache as it is. For TKG, it selects
        specific KV cache based on the block table, and reshape into bhsd
        before returning.
        """
        # CTE usecase
        if is_for_context_encoding:
            return cache

        # TKG usecase
        batch_size, _ = active_block_table.shape
        num_blocks, num_heads_per_rank, block_size, head_dimension = cache.shape

        cache = cache.reshape(num_blocks * num_heads_per_rank, block_size * head_dimension)

        indices = torch.arange(num_heads_per_rank).reshape(1, -1, 1) \
            + active_block_table.reshape(batch_size, 1, -1) * num_heads_per_rank
        indices = indices.reshape(-1)

        selected_cache = cache[indices]
        selected_cache = selected_cache.reshape(batch_size, num_heads_per_rank, -1, head_dimension)
        return selected_cache

    def update_cache(
        self,
        new_key_values: List[Tensor],
        scatter_index=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        """
        Write the KV cache for paged attention

        The slot_mapping will be passed as scatter_index
        """
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            k_cache, v_cache = self.update_kv_by_layer_id(
                idx=idx,
                kv_per_layer=kv_per_layer,
                scatter_index=scatter_index,
                kvcache_buffer=kvcache_buffer,
            )
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return updated_kv_cache

    def update_kv_by_layer_id(
        self,
        idx,
        kv_per_layer: List[Tensor],
        scatter_index=None,
        kvcache_buffer=None,
        **kwargs,
    ):
        k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer=kvcache_buffer)
        slot_mapping = scatter_index
        k_cache = self._update_cache_into_block_layout(
            latest=kv_per_layer[0],
            cache=k_cache,
            slot_mapping=slot_mapping,
        )
        v_cache = self._update_cache_into_block_layout(
            latest=kv_per_layer[1],
            cache=v_cache,
            slot_mapping=slot_mapping,
        )
        return k_cache, v_cache

    def _update_cache_into_block_layout(self, latest, cache, slot_mapping, padding_id=-1):
        if self.is_prefix_caching:
            return self._update_cache_with_reshape(latest, cache, slot_mapping, padding_id)
        elif self.is_chunked_prefill:
            return self._update_cache_with_index_put(latest, cache, slot_mapping, padding_id)

    def _update_cache_with_reshape(self, latest, cache, slot_mapping, padding_id=-1):
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

    def _update_cache_with_index_put(
        self,
        latest,
        cache,
        slot_mapping,
        padding_id=-1,
    ):
        """
        Update KV cache with index_put

        This avoids reshaping the whole KV cache
        """
        batch_size, num_heads_per_rank, n_active_tokens, head_dim = latest.shape
        num_blocks, num_heads_per_rank, block_size, head_dim = cache.shape

        dtype = slot_mapping.dtype
        device = slot_mapping.device

        pad_dest_index = torch.tensor(num_blocks * block_size - 1, device=device, dtype=dtype)

        slot_mapping = torch.where(
            slot_mapping == padding_id,
            pad_dest_index,
            slot_mapping,
        )

        block_id = slot_mapping // self.pa_block_size
        block_id = block_id.view(batch_size, 1, n_active_tokens)
        block_id = block_id.expand(batch_size, num_heads_per_rank, n_active_tokens)

        block_offset = slot_mapping % self.pa_block_size
        block_offset = block_offset.view(batch_size, 1, n_active_tokens)
        block_offset = block_offset.expand(batch_size, num_heads_per_rank, n_active_tokens)

        indices_on_h = torch.arange(num_heads_per_rank, dtype=dtype, device=device)
        indices_on_h = indices_on_h.view(1, num_heads_per_rank, 1)
        indices_on_h = indices_on_h.expand(batch_size, num_heads_per_rank, n_active_tokens)

        cache = torch.index_put(
            input=cache,
            indices=[block_id, indices_on_h, block_offset],
            values=latest
        )
        return cache


def generate_tokengen_slot_mapping(
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: torch.Tensor,
):
    B = position_ids.shape[0]

    # Determine active sequences from slot mapping -1 pad
    active_mask = (slot_mapping >= 0)

    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device)
    block_indices = (position_ids // block_size).squeeze()

    block_number = block_table[row_indices, block_indices]
    block_offset = (position_ids % block_size).squeeze()
    cur_slots = block_size * block_number + block_offset
    cur_slots = cur_slots.unsqueeze(dim=1)

    # Mask out inactive sequences
    inactive_slots = torch.ones_like(cur_slots) * -1
    final_slots = torch.where(active_mask, cur_slots, inactive_slots)

    return final_slots


def generate_fusedspec_slot_mapping(
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: torch.Tensor,
):
    B = position_ids.shape[0]
    speculation_length = slot_mapping.shape[1]

    # Determine active sequences from slot mapping -1 pad
    active_mask = ~torch.all(slot_mapping < 0, dim=1).unsqueeze(dim=1)
    expanded_active_mask = torch.tile(active_mask, (1, speculation_length))

    # Generate all speculative positions through outer sum.
    relative_speculative_positions = torch.arange(speculation_length, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=0)
    expanded_positions = position_ids + relative_speculative_positions

    row_indices = torch.arange(B, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(dim=1)
    expanded_row_indices = torch.tile(row_indices, (1, speculation_length))

    expanded_block_indices = (expanded_positions // block_size)
    block_number = block_table[expanded_row_indices, expanded_block_indices]
    block_offset = (expanded_positions % block_size)
    cur_slots = block_size * block_number + block_offset

    # Mask out inactive sequences
    inactive_slots = torch.ones_like(cur_slots) * -1
    final_slots = torch.where(expanded_active_mask, cur_slots, inactive_slots)

    return final_slots
