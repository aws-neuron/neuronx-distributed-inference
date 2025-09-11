import torch

from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed.parallel_layers.layers import SPMDRank


class DataParallelKVCacheManager(KVCacheManager):
    def __init__(self, config: InferenceConfig, global_rank: SPMDRank, **kwargs):
        super().__init__(config, **kwargs)

        self.global_rank = global_rank
        self.config = config

    def get_cache_update_index_for_seq_ids(self, seq_ids):
        assert self.kv_cache_padding_size > 0, "kv_cache_padding_size must be > 0 for data-parallel tokengen"
        # handle out-of-bound seq_ids
        garbage_pos = self.kv_cache_batch_size + self.kv_cache_padding_size - 1

        dp_rank = torch.div(
            self.get_rank(device=seq_ids.device),
            self.config.neuron_config.tp_degree // self.config.neuron_config.attention_dp_degree,
            rounding_mode="floor",
        ).to(torch.int32)

        kv_range_start = dp_rank * self.kv_cache_batch_size
        kv_range_end = kv_range_start + self.kv_cache_batch_size
        valid_mask = torch.logical_and(
            seq_ids >= kv_range_start,
            seq_ids < kv_range_end,
        )

        seq_ids = torch.where(
            valid_mask, seq_ids - kv_range_start, garbage_pos
        )
        return seq_ids
