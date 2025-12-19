import os

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.utils import divide
from neuronx_distributed_inference.modules.attention.attention_process_groups import tp_mesh_8_by_8


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


def get_cp_rank(global_rank: torch.tensor, tp_degree: int, cp_degree: int = None, switch_cc: bool = False):
    if cp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(global_rank, switch_cc)
    cp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return cp_rank


def get_dp_rank(global_rank: torch.tensor, tp_degree: int, dp_degree: int = None, switch_cc: bool = False):
    if dp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(global_rank, switch_cc)
    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return dp_rank


def get_kv_head_group_number(global_rank: torch.tensor, tp_degree: int):
    tp_cp_rank = torch.remainder(global_rank, tp_degree).to(torch.int32)
    return tp_cp_rank


def split_along_dim(tensor: torch.tensor, dim: int, rank: int, num_partitions: int):
    if tensor is None:
        return None

    num_per_partition = divide(tensor.size(dim), num_partitions)
    indices = torch.arange(0, num_per_partition, device=tensor.device)
    indices = indices + (rank * num_per_partition)
    tensor = torch.index_select(tensor, dim=dim, index=indices)

    return tensor


def get_rank_8_by_8(global_rank, switch_cc: bool = False):
    """
    Get the row index of a global rank in an 8x8 mesh topology.
    Args:
        global_rank: The global rank to locate in the mesh
        switch_cc: If True, use PDS topology; otherwise use base TRN2 topology
    Returns:
        torch.Tensor: The row index (0-7) where the global rank is found in the 8x8 mesh
    """
    mesh = torch.tensor(tp_mesh_8_by_8(switch_cc), device=global_rank.device)
    matches = (mesh == global_rank).any(dim=1)
    return torch.argmax(matches.int())
