import os

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.utils import divide


def get_init_world_size() -> int:
    """Get world size set by distributed launcher (torchrun or mpirun)"""
    for var in ["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"]:
        if var in os.environ and os.environ[var] != "":
            return int(os.environ[var])
    return -1


def get_init_rank() -> int:
    """Get rank set by distributed launcher (torchrun or mpirun)"""
    for var in ["RANK", "OMPI_COMM_WORLD_RANK"]:
        if var in os.environ and os.environ[var] != "":
            return int(os.environ[var])
    return -1


def get_tp_group(config):
    """Get TP process group. Handle override."""
    if not hasattr(config.neuron_config, "use_draft_group"):
        return None
    if config.neuron_config.use_draft_group:
        return parallel_state.get_speculative_draft_group(as_list=False)
    return parallel_state.get_tensor_model_parallel_group(as_list=False)


def get_dp_rank_spmd(global_rank: torch.tensor, tp_degree: int):
    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor",
    ).to(torch.int32)
    return dp_rank


def get_cp_rank(global_rank: torch.tensor, tp_degree: int):
    cp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return cp_rank


def get_dp_rank(global_rank: torch.tensor, tp_degree: int):
    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return dp_rank


def split_along_dim(tensor, dim, rank, num_partitions):
    if tensor is None:
        return None

    num_per_partition = divide(tensor.size(dim), num_partitions)
    indices = torch.arange(0, num_per_partition, device=tensor.device)
    indices = indices + (rank * num_per_partition)
    tensor = torch.index_select(tensor, dim=dim, index=indices)

    return tensor
