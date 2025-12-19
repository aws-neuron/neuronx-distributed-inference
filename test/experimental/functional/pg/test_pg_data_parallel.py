import pytest
import torch

import neuronx_distributed_inference.experimental.functional.pg.data_parallel as dp


@pytest.mark.parametrize("rank,world_size,dp_degree,expected_result", [
    (torch.tensor(1), 16, 4, torch.tensor(0)),
    (torch.tensor(11), 16, 4, torch.tensor(2)),
    (torch.tensor(0), 16, 8, torch.tensor(0)),
    (torch.tensor(15), 16, 8, torch.tensor(7)),
    (torch.tensor(1), 64, 8, torch.tensor(0)),
    (torch.tensor(31), 64, 8, torch.tensor(2)),
])
def test_get_dp_rank(rank, world_size, dp_degree, expected_result):
    actual = dp.get_dp_rank(rank=rank, world_size=world_size, dp_degree=dp_degree)

    assert actual.item() == expected_result.item()
    assert actual.dtype == torch.int32


def test_get_dp_rank_invalid_dp_degree():
    rank = torch.tensor(1)
    world_size = 8
    dp_degree = 9

    with pytest.raises(AssertionError, match="DP degree size should be <= world_size"):
        dp.get_dp_rank(rank=rank, world_size=world_size, dp_degree=dp_degree)


@pytest.mark.parametrize("rank,world_size,dp_degree", [
    (torch.tensor(8), 8, 2),
    (torch.tensor(-1), 8, 2),
])
def test_get_dp_rank_invalid_rank(rank, world_size, dp_degree):
    rank = torch.tensor(10)
    world_size = 8
    dp_degree = 2

    with pytest.raises(AssertionError, match="Rank should be between 0 and"):
        dp.get_dp_rank(rank=rank, world_size=world_size, dp_degree=dp_degree)
