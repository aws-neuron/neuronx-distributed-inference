import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.kvcache.data_parallel_kv_cache_manager import DataParallelKVCacheManager


class MockSPMDRank:
    def __init__(self, rank):
        self._rank = rank
    
    def get_rank(self):
        return torch.tensor(self._rank)

@pytest.fixture
def create_manager():
    def _create(rank=0, tp_degree=8, attention_dp_degree=2, batch_size=32):
        config = InferenceConfig(NeuronConfig(
            tp_degree = tp_degree,
            cp_degree = attention_dp_degree,
            attention_dp_degree = attention_dp_degree,
            batch_size = batch_size,
            is_continuous_batching = True,
        ))
        config.num_attention_heads = 40
        config.hidden_size = 128
        config.num_hidden_layers = 2

        spmd_rank = MockSPMDRank(rank)
        manager = DataParallelKVCacheManager(config=config, global_rank=spmd_rank, num_kv_head=8)
        return manager
    return _create


def test_correct_seq_id_writes(create_manager):
    # With tp_degree=8, dp_degree=2, we have 4 ranks per sub-tp group
    
    for rank in [0, 1, 2, 3]:
        manager = create_manager(rank=rank, batch_size=32)
        
        # seq_ids 0-15 should be valid for the first group
        seq_ids = torch.tensor([0, 7, 15])
        result = manager.get_cache_update_index_for_seq_ids(seq_ids)
        expected = torch.tensor([0, 7, 15])
        assert torch.all(result == expected), f"Rank {rank}: Expected {expected}, got {result}"
        
        # seq_ids 16-31 should be invalid for the first group
        seq_ids = torch.tensor([16, 23, 31])
        result = manager.get_cache_update_index_for_seq_ids(seq_ids)
        garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
        expected = torch.tensor([garbage_pos, garbage_pos, garbage_pos])
        assert torch.all(result == expected), f"Rank {rank}: Expected {expected}, got {result}"
    
    for rank in [4, 5, 6, 7]:
        manager = create_manager(rank=rank)
        
        # seq_ids 16-31 should be valid for the second group
        seq_ids = torch.tensor([16, 23, 31])
        result = manager.get_cache_update_index_for_seq_ids(seq_ids)
        expected = torch.tensor([0, 7, 15])
        assert torch.all(result == expected), f"Rank {rank}: Expected {expected}, got {result}"
        
        # seq_ids 0-15 should be invalid for the second group
        seq_ids = torch.tensor([0, 7, 15])
        result = manager.get_cache_update_index_for_seq_ids(seq_ids)
        garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
        expected = torch.tensor([garbage_pos, garbage_pos, garbage_pos])
        assert torch.all(result == expected), f"Rank {rank}: Expected {expected}, got {result}"


def test_mixed_seq_ids(create_manager):
    manager = create_manager(rank=0)
    seq_ids = torch.tensor([0, 15, 16, 31])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
    expected = torch.tensor([0, 15, garbage_pos, garbage_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"
    
    manager = create_manager(rank=4)
    seq_ids = torch.tensor([15, 16, 31, 32])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
    expected = torch.tensor([garbage_pos, 0, 15, garbage_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"