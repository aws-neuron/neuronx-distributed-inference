from torch import Tensor

from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed_inference.experimental.functional.parallel.tensor_ops import split_along_dim
from neuronx_distributed_inference.experimental.functional.pg import get_dp_rank


def split_input_for_data_parallel(x: Tensor, dim: int, world_size: int, dp_degree: int, rank_util: SPMDRank):
    r"""
    Splits `x` along the `dim` dimension into dp_degree number
    of partitions. The i'th tp-group will be assigned the i-th partition. For data
    parallelism use-cases, the dim specified will generally be the sequence dim.

    Example (dim=0, world_size=8, dp_degree=2)
        x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    If rank = 0/1/2/3 (results in dp_rank = 0)
        Returns torch.tensor([0, 1, 2, 3])
    If rank = 4/5/6/7 (results in dp_rank = 1)
        Returns torch.tensor([4, 5, 6, 7])

    :param x: Input tensor that is split for data parallel attention
    :param dim: Dimension to split the input tensor
    :param world_size: World size of program
    :param dp_degree: Size of each data parallel process group
    :param rank_util: SPMDRank for retrieving the current rank

    :return: Result of splitting the input tensor along the provided dimension, based on the rank
    """
    assert dp_degree <= world_size, f"DP degree size should be <= world_size, but got {dp_degree=} and {world_size=}"

    dp_rank = get_dp_rank(rank_util.get_rank(), world_size, dp_degree)
    return split_along_dim(x, dim, dp_rank, num_partitions=dp_degree)
