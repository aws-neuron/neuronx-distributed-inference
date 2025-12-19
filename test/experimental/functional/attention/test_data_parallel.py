import pytest
import torch
from unittest.mock import MagicMock

from neuronx_distributed_inference.experimental.functional.attention.data_parallel import (
    split_input_for_data_parallel,
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


@pytest.mark.parametrize("x,dim,world_size,dp_degree,rank,expected", [
    (torch.arange(8), 0, 8, 2, 3, torch.tensor([0, 1, 2, 3])),
    (torch.arange(8), 0, 8, 2, 5, torch.tensor([4, 5, 6, 7])),
    (torch.arange(16).view(4, 4), 0, 16, 4, 4, torch.tensor([[4, 5, 6, 7]])),
    (torch.arange(16).view(4, 4), 1, 16, 4, 4, torch.tensor([[1], [5], [9], [13]])),
    (torch.arange(32).view(2, 4, 4), 1, 16, 4, 7, torch.tensor([[[4, 5, 6, 7]], [[20, 21, 22, 23]]])),
])
def test_split_input_for_data_parallel(mock_spmd_rank, x, dim, world_size, dp_degree, rank, expected):
    mock_spmd_rank.get_rank.return_value = torch.tensor([rank], dtype=torch.int32)

    actual = split_input_for_data_parallel(x, dim, world_size, dp_degree, mock_spmd_rank)

    assert torch.equal(actual, expected)


def test_split_input_for_data_parallel_invalid_dp_degree(mock_spmd_rank):
    x = torch.zeros(8)
    dim = 0
    world_size = 8
    dp_degree = 9
    mock_spmd_rank.get_rank.return_value = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(AssertionError, match="DP degree size should be <= world_size"):
        split_input_for_data_parallel(x, dim, world_size, dp_degree, mock_spmd_rank)
