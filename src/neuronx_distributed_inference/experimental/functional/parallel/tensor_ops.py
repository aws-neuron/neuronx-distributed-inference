"""
This file contains general tensor operations that can be used when implementing
various parallelism techniques
"""

import torch


def split_along_dim(x: torch.Tensor, dim: int, rank: int, num_partitions: int) -> torch.Tensor:
    r"""
    Partitions a tensor into num_partitions and selects the rank'th partition.

    :param x: Input tensor to partition
    :param dim: Dimension to partition along
    :param rank: Rank to select
    :param num_partitions: Number of partitions to use to partition the tensor

    :return: Rank'th partition of the provided tensor along the provided dimension
    """
    assert rank < num_partitions, f"Rank {rank} should be < number of partitions {num_partitions}"

    indices = _indices_split_along_dim(x, dim, rank, num_partitions)
    x = torch.index_select(x, dim=dim, index=indices)

    return x


# - private functions go below


def _indices_split_along_dim(tensor: torch.Tensor, dim: int, rank: int, num_partitions: int) -> torch.Tensor:
    r"""
    Calculates indices for partitioning a tensor along a specified dimension.

    Example:
        For a tensor of size 8 along dim with num_partitions=4:
        - rank 0: returns indices [0, 1]
        - rank 1: returns indices [2, 3]
        - rank 2: returns indices [4, 5]
        - rank 3: returns indices [6, 7]

    :param tensor: Input tensor to be partitioned.
    :param dim: Dimension along which to split the tensor.
    :param rank: Process rank determining which partition to create (0 to num_partitions-1).
    :param num_partitions: Total number of partitions to create.

    :return: Tensor containing indices for the specified partition
    """
    # Validate partition dimension is evenly disible by the number of partitions
    dim_size = tensor.size(dim)
    assert dim_size % num_partitions == 0, "{} is not divisible by {}".format(dim_size, num_partitions)

    partition_size = dim_size // num_partitions
    start_idx = rank * partition_size
    indices = torch.arange(partition_size, device=tensor.device) + start_idx

    return indices
