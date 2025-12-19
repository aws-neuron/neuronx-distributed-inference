import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.kvcache.data_parallel_kv_cache_manager import DataParallelKVCacheManager
from neuronx_distributed_inference.modules.kvcache.utils import write_kv_cache_at_batch_kernel

from torch_xla.core import xla_model as xm
from neuronxcc.nki.language import nc 
class MockSPMDRank:
    def __init__(self, rank):
        self._rank = rank
    
    def get_rank(self):
        return torch.tensor(self._rank)

@pytest.fixture
def create_manager():
    def _create(rank=0, tp_degree=8, attention_dp_degree=2, batch_size=32, legacy_flow=False):
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
        if legacy_flow:
            config.neuron_config.kv_cache_padding_size = 1  #Tests legacy KV cache update flow for Trn1

        spmd_rank = MockSPMDRank(rank)
        manager = DataParallelKVCacheManager(config=config, global_rank=spmd_rank, num_kv_head=8)
        return manager
    return _create


def test_correct_seq_id_writes_legacy(create_manager):
    # With tp_degree=8, dp_degree=2, we have 4 ranks per sub-tp group
    
    for rank in [0, 1, 2, 3]:
        manager = create_manager(rank=rank, batch_size=32, legacy_flow=True)
        
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
        manager = create_manager(rank=rank, legacy_flow=True)
        
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


def test_mixed_seq_ids_legacy(create_manager):
    manager = create_manager(rank=0, legacy_flow=True)
    seq_ids = torch.tensor([0, 15, 16, 31])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
    expected = torch.tensor([0, 15, garbage_pos, garbage_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"
    
    manager = create_manager(rank=4, legacy_flow=True)
    seq_ids = torch.tensor([15, 16, 31, 32])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    garbage_pos = manager.kv_cache_batch_size + manager.kv_cache_padding_size - 1
    expected = torch.tensor([garbage_pos, 0, 15, garbage_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_kernel_update_oob_seq_ids(create_manager):
    manager = create_manager(rank=0)
    oob_pos = manager.kv_cache_batch_size
    seq_ids = torch.tensor([0, 15, 16, 31])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    expected = torch.tensor([0, 15, oob_pos, oob_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"
    
    manager = create_manager(rank=4)
    oob_pos = manager.kv_cache_batch_size
    seq_ids = torch.tensor([15, 16, 31, 32])
    result = manager.get_cache_update_index_for_seq_ids(seq_ids)
    expected = torch.tensor([oob_pos, 0, 15, oob_pos])
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "batch_size,  num_heads, seq_len, prior_seq_len, head_dim, idx, lnc",  [
    #Qwen3 MoE
    [2,           2,         10240,   10240,         128,      1,   2], #Without bucketing
    [2,           2,         10240,   16384,         128,      1,   2], #With bucketing
    ])
def test_write_kv_cache_at_batch_kernel_execution(batch_size, num_heads, seq_len, prior_seq_len, head_dim, idx, lnc):
    """Test actual execution of write_kv_cache_at_batch_kernel function."""
    device = xm.xla_device()
    
    # Create input K and V tensors with known values
    K = torch.ones(1, num_heads, seq_len, head_dim, dtype=torch.float16, device=device) * 2.0
    V = torch.ones(1, num_heads, seq_len, head_dim, dtype=torch.float16, device=device) * 3.0

    # Create prior K and V cache tensors (initially zeros)
    K_prior = torch.zeros(batch_size, num_heads, prior_seq_len, head_dim, dtype=torch.float16, device=device)
    V_prior = torch.zeros(batch_size, num_heads, prior_seq_len, head_dim, dtype=torch.float16, device=device)

    # Create batch indices
    batch_idx = torch.tensor([idx], dtype=torch.int32, device=device)

    def golden(X, Y, X_prior, Y_prior, idx):
        X_prior[idx[0],:,:X.shape[2],:]=X
        Y_prior[idx[0],:,:Y.shape[2],:]=Y
        return X_prior, Y_prior
    
    K_golden, V_golden = golden(K.cpu(), V.cpu(), K_prior.cpu().clone(), V_prior.cpu().clone(), batch_idx.cpu())

    K_prior = K_prior.to(device)
    V_prior = V_prior.to(device)

    # Call the kernel function directly
    grid = (nc(lnc),)
    K_prior, V_prior = write_kv_cache_at_batch_kernel[grid](K, 
                                                    V, 
                                                    K_prior,
                                                    V_prior, 
                                                    batch_idx)
    
    assert torch.allclose(K_prior.cpu(), K_golden, atol=1e-6), "K cache positions differs from golden"
    assert torch.allclose(V_prior.cpu(), V_golden, atol=1e-6), "V cache positions differs from golden"
