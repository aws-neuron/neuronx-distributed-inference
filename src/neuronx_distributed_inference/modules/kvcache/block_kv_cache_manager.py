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

    def __init__(self, config: InferenceConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        self.is_paged_attention = config.neuron_config.is_paged_attention
        self.pa_num_blocks = config.neuron_config.pa_num_blocks
        self.pa_block_size = config.neuron_config.pa_block_size


    def _init_kv_shape(self, config: InferenceConfig):
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        # block layout for paged attention
        self.kv_shape = (
            config.neuron_config.pa_num_blocks,
            config.neuron_config.pa_block_size,
            num_kv_heads_per_rank,
            hidden_dim_per_head,
        )


    def get_cache(self, seq_len: int, **kwargs):
        """
        Get cache for paged attention using an active block table.

        An active block table will only have padding block at the end, not
        between blocks.
        """
        block_table = kwargs.get("block_table")
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            k_cache, v_cache = self.get_kv_by_layer_id(key_layer_idx)

            key_state = self._get_cache_from_block_table(k_cache, block_table)
            value_state = self._get_cache_from_block_table(v_cache, block_table)

            past_key_values.append([key_state, value_state])

        return past_key_values


    def _get_cache_from_block_table(self, cache: Tensor, block_table: Tensor):
        selected_cache = cache.index_select(dim=0, index=block_table)
        return selected_cache
    

    def update_cache(
        self, 
        is_for_context_encoding: bool, 
        seq_ids: Tensor, 
        position_ids: Tensor, 
        past_key_values: List[Tensor], 
        seq_len: int, 
        scatter_index=None
    ):
        """
        Write the KV cache for paged attention

        The slot_mapping will be passed as scatter_index
        """
        slot_mapping = scatter_index
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
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


    def _update_cache_into_block_layout(self, latest, cache, slot_mapping):
        """
        Write the latest KV into cache, where the cache is in block layout
        """
        batch_size, num_heads_per_rank, seq_len, head_dim = latest.shape
        latest = latest.permute((0, 2, 1, 3))
        latest = latest.reshape((batch_size*seq_len, num_heads_per_rank*head_dim))

        num_blocks, block_size, num_heads_per_rank, head_dim = cache.shape
        cache = cache.reshape((num_blocks*block_size, num_heads_per_rank*head_dim))

        slot_mapping = slot_mapping.reshape((batch_size*seq_len, 1))
        slot_mapping = slot_mapping.expand((batch_size*seq_len, num_heads_per_rank*head_dim))

        cache = torch.scatter(input=cache, dim=0, index=slot_mapping, src=latest)
        cache = cache.reshape((num_blocks, block_size, num_heads_per_rank, head_dim))

        return cache
