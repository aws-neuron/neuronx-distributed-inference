import pytest
import torch

from neuronx_distributed_inference.modules.attention.utils import apply_seq_id_mask


def test_basic_functionality():
    position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    seq_ids = torch.tensor([1, -1])
    pad_constant = 100
    
    expected = torch.tensor([[0, 1, 2], [100, 100, 100]])
    result = apply_seq_id_mask(position_ids, seq_ids, pad_constant)
    
    assert torch.equal(result, expected)

def test_with_chunk_size():
    position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    seq_ids = torch.tensor([1, -1])
    pad_constant = 100
    chunk_size = 2
    
    expected = torch.tensor([[0, 1, 0], [100, 100, 100]])
    result = apply_seq_id_mask(position_ids, seq_ids, pad_constant, chunk_size)
    
    assert torch.equal(result, expected)

def test_large_chunk_size():
    position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    seq_ids = torch.tensor([1, -1])
    pad_constant = 100
    chunk_size = 10
    
    expected = torch.tensor([[0, 1, 2], [100, 100, 100]])
    result = apply_seq_id_mask(position_ids, seq_ids, pad_constant, chunk_size)
    
    assert torch.equal(result, expected)