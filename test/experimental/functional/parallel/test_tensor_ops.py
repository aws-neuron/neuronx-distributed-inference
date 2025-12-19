import pytest
import torch


from neuronx_distributed_inference.experimental.functional.parallel import tensor_ops


@pytest.mark.parametrize("x,dim,rank,num_partitions,expected", [
    (torch.arange(8), 0, 0, 4, torch.tensor([0, 1])),
    (torch.arange(8), 0, 1, 4, torch.tensor([2, 3])),
    (torch.arange(8), 0, 2, 4, torch.tensor([4, 5])),
    (torch.arange(8), 0, 3, 4, torch.tensor([6, 7])),
    (torch.arange(16).view(4, 4), 0, 1, 4, torch.tensor([[4, 5, 6, 7]])),
    (torch.arange(16).view(4, 4), 1, 1, 4, torch.tensor([[1], [5], [9], [13]])),
    (torch.arange(32).view(2, 4, 4), 1, 1, 4, torch.tensor([[[4, 5, 6, 7]], [[20, 21, 22, 23]]])),
])
def test_split_along_dim(x, dim, rank, num_partitions, expected):
    actual = tensor_ops.split_along_dim(x, dim, rank, num_partitions)

    assert torch.equal(actual, expected)


def test_split_along_dim_invalid_rank():
    x = torch.zeros(8)
    dim = 0
    rank = 4
    num_partitions = 4

    with pytest.raises(AssertionError, match="Rank"):
        tensor_ops.split_along_dim(x, dim, rank, num_partitions)


@pytest.mark.parametrize("x,dim,rank,num_partitions,expected", [
    (torch.zeros(8), 0, 0, 4, torch.tensor([0, 1])),
    (torch.zeros(8), 0, 1, 4, torch.tensor([2, 3])),
    (torch.zeros(8), 0, 2, 4, torch.tensor([4, 5])),
    (torch.zeros(8), 0, 3, 4, torch.tensor([6, 7])),
    (torch.zeros(16).view(4, 4), 0, 1, 4, torch.tensor([1])),
    (torch.zeros(16).view(4, 4), 1, 1, 4, torch.tensor([1])),
    (torch.zeros(64).view(2, 8, 4), 1, 1, 4, torch.tensor([2, 3])),
])
def test_indices_split_along_dim(x, dim, rank, num_partitions, expected):
    actual = tensor_ops._indices_split_along_dim(x, dim, rank, num_partitions)

    assert torch.equal(actual, expected)
