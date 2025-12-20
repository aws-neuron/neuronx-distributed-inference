import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed_inference.experimental.functional.parallel.tensor_ops import split_along_dim
from neuronx_distributed_inference.experimental.functional.pg import get_cp_rank

from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
)


def split_input_for_context_parallel(x: Tensor, dim: int, world_size: int, cp_degree: int, rank_util: SPMDRank):
    r"""
    Splits `x` along the `dim` dimension into cp_degree number
    of partitions. The i'th tp-group will be assigned the i-th partition. For context
    parallelism use-cases, the dim specified will generally be the sequence dim.

    Example (dim=0, world_size=8, cp_degree=2)
        x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    If rank = 0/1/2/3 (results in cp_rank = 0)
        Returns torch.tensor([0, 1, 2, 3])
    If rank = 4/5/6/7 (results in cp_rank = 1)
        Returns torch.tensor([4, 5, 6, 7])

    :param x: Input tensor that is split for context parallel attention
    :param dim: Dimension to split the input tensor
    :param world_size: World size of program
    :param cp_degree: Size of each context parallel process group
    :param rank_util: SPMDRank for retrieving the current rank

    :return: Result of splitting the input tensor along the provided dimension, based on the rank
    """
    assert cp_degree <= world_size, f"Cp degree size should be <= world_size, but got {cp_degree=} and {world_size=}"

    cp_rank = get_cp_rank(rank_util.get_rank(), world_size, cp_degree)
    return split_along_dim(x, dim, cp_rank, num_partitions=cp_degree)


def gather_kv_context_parallel(K: Tensor, V: Tensor, dim: int, process_group: ProcessGroup) -> tuple[Tensor, Tensor]:
    r"""
    Stacks KV, gathers along the specified dim and unstacks KV.

    :param K: Partial (S / CP length) K tensor
    :param V: Partial (S / CP length) V tensor
    :param dim: Dimension along which to gather on, expected to be the sequence dim
    :param process_group: ProcessGroup to gather along.

    :return: Tuple containing K and V tensors that have been gathered along provided dimension
    """
    assert K.shape == V.shape, f"K and V tensor should be the same shape, but got {K.shape=} and {V.shape=}"

    stacked_kv = torch.stack([K, V], dim=0)

    stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
        stacked_kv,
        # We increment by 1 since we stacked K and V in a new dimension 0 before gathering
        gather_dim=dim + 1,
        process_group=process_group,
    )

    K, V = torch.unbind(stacked_kv, dim=0)

    return K, V
