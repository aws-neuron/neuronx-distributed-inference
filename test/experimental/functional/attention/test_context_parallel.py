import pytest
import torch
from unittest.mock import patch, MagicMock

from neuronx_distributed_inference.experimental.functional.attention.context_parallel import (
    gather_kv_context_parallel, 
    split_input_for_context_parallel,
)

torch.manual_seed(0)


@pytest.fixture
def mock_spmd_rank():
    """
    Fixture that returns a mock SPMDRank instance.
    Set return value in test using: mock_spmd_rank.get_rank.return_value = torch.tensor([rank])
    """
    mock = MagicMock()
    mock.get_rank = MagicMock()
    return mock


@pytest.mark.parametrize("x,dim,world_size,cp_degree,rank,expected", [
    (torch.arange(8), 0, 8, 2, 3, torch.tensor([0, 1, 2, 3])),
    (torch.arange(8), 0, 8, 2, 5, torch.tensor([4, 5, 6, 7])),
    (torch.arange(16).view(4, 4), 0, 16, 4, 4, torch.tensor([[4, 5, 6, 7]])),
    (torch.arange(16).view(4, 4), 1, 16, 4, 4, torch.tensor([[1], [5], [9], [13]])),
    (torch.arange(32).view(2, 4, 4), 1, 16, 4, 7, torch.tensor([[[4, 5, 6, 7]], [[20, 21, 22, 23]]])),
])
def test_split_input_for_context_parallel(mock_spmd_rank, x, dim, world_size, cp_degree, rank, expected):
    mock_spmd_rank.get_rank.return_value = torch.tensor([rank], dtype=torch.int32)

    actual = split_input_for_context_parallel(x, dim, world_size, cp_degree, mock_spmd_rank)

    assert torch.equal(actual, expected)


def test_split_input_for_context_parallel_invalid_cp_degree(mock_spmd_rank):
    x = torch.zeros(8)
    dim = 0
    world_size = 8
    cp_degree = 9
    mock_spmd_rank.get_rank.return_value = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(AssertionError, match="Cp degree size should be <= world_size"):
        split_input_for_context_parallel(x, dim, world_size, cp_degree, mock_spmd_rank)


def test_gather_kv_context_parallel():
    bsz = 16
    seq_lem = 8
    num_heads = 2
    hidden_dim = 32
    cp_degree = 4
    K = torch.randn(bsz, num_heads, seq_lem // cp_degree, hidden_dim)
    V = torch.randn(bsz, num_heads, seq_lem // cp_degree, hidden_dim)

    mock_process_group = MagicMock()
    mock_process_group.size.return_value = cp_degree
    
    def mock_gather_implementation(tensor, gather_dim, process_group):
        return tensor.repeat(1, 1, 1, process_group.size(), 1)

    with patch('neuronx_distributed_inference.experimental.functional.attention.context_parallel.gather_from_tensor_model_parallel_region_with_dim',
              side_effect=mock_gather_implementation) as mock_gather:
        K_actual, V_actual = gather_kv_context_parallel(K, V, dim=2, process_group=mock_process_group)
        
        K_expected = torch.cat((K, K, K, K), 2)
        V_expected = torch.cat((V, V, V, V), 2)
        assert torch.equal(K_actual, K_expected)
        assert torch.equal(V_actual, V_expected)
        expected_stacked_KV_for_gather = torch.stack([K, V], dim=0)
        mock_gather.assert_called_once_with(
            TensorMatcher(expected_stacked_KV_for_gather),
            gather_dim=3,
            process_group=mock_process_group
        )


def test_gather_kv_context_parallel_kv_shape_mismatch():
    # K and V have different shapes, so the gather should fail
    bsz = 16
    k_len = 8
    v_len = 10
    num_heads = 2
    hidden_dim = 32
    K = torch.randn(bsz, num_heads, k_len, hidden_dim)
    V = torch.randn(bsz, num_heads, v_len, hidden_dim)
    mock_process_group = MagicMock()

    with pytest.raises(AssertionError, match="K and V tensor should be the same shape"):
        gather_kv_context_parallel(K, V, dim=2, process_group=mock_process_group)


# Wrapper to allow matching tensor-valued args in mock assertions. By default,
# torch tensor equality check returns a element-wise tensor of booleans
class TensorMatcher:
    def __init__(self, expected_tensor):
        self.expected_tensor = expected_tensor

    def __eq__(self, other):
        return torch.equal(self.expected_tensor, other)
