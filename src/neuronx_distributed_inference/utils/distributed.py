import os
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.utils import divide
from neuronx_distributed_inference.modules.attention.attention_process_groups import tp_mesh_8_by_8
from neuronxcc.nki._pre_prod_kernels.util.kernel_helpers import get_verified_program_sharding_info


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


@nki.jit
def split_along_dim0_kernel(tensor, rank, num_partitions):
    """
    NKI kernel that returns the rank-th partition of tensor split along dimension 0.
    Equivalent to split_along_dim(tensor, 0, rank, num_partitions).

    :param tensor: Input tensor in HBM
    :param rank: Single-element tensor (shape (1,)) indicating which partition to return
    :param num_partitions: Number of partitions to split tensor into
    :return: Tensor partition with shape (tensor.shape[0] // num_partitions, *tensor.shape[1:])
    """
    num_per_partition = divide(tensor.shape[0], num_partitions)
    output_shape = (num_per_partition, *tensor.shape[1:])
    rank_sb = nl.load(rank.reshape((1, 1)))

    # Size of the remaining dim after leading num_partitions dim.
    F = num_per_partition * math.prod(tensor.shape[1:])
    tensor = tensor.reshape((num_partitions, F))
    out_tensor = nl.ndarray((1, F), dtype=tensor.dtype, buffer=nl.shared_hbm)

    # LNC-shard on remaining dim F, last NC takes all remainder in case F indivisible by n_prgs.
    _, n_prgs, prg_id = get_verified_program_sharding_info("split_along_dim0_kernel", (0, 1), 2)
    sharded_F = F // n_prgs
    F_offset = sharded_F * prg_id
    i_p, i_f = nl.mgrid[:1, :F]
    if prg_id == n_prgs - 1:
        i_p, i_f = nl.mgrid[:1, :(F - (n_prgs - 1) * sharded_F)]

    nki.isa.dma_copy(src=tensor[rank_sb, F_offset + i_f], dst=out_tensor[i_p, F_offset + i_f])

    return out_tensor.reshape(output_shape)


def split_along_dim0_kernel_wrapper(tensor, rank, num_partitions, lnc):
    """Wrapper for split_along_dim0_kernel and fall back to flat-torch if on CPU"""
    if tensor.device == torch.device("cpu"):
        return split_along_dim(tensor, 0, rank, num_partitions)
    return split_along_dim0_kernel[(nl.nc(lnc),)](tensor, rank, num_partitions)


def get_rank_8_by_8(global_rank, switch_cc: bool = False):
    """
    Get the row index of a global rank in an 8x8 mesh topology.
    Args:
        global_rank: The global rank to locate in the mesh
        switch_cc: If True, use switch topology; otherwise use base TRN2 topology
    Returns:
        torch.Tensor: The row index (0-7) where the global rank is found in the 8x8 mesh
    """
    mesh = torch.tensor(tp_mesh_8_by_8(switch_cc), device=global_rank.device)
    return _get_rank_row_index(mesh, global_rank)


def _get_rank_row_index(mesh, global_rank):
    """
    Get the row index of a global rank in a mesh topology.

    Args:
        mesh: The mesh
        global_rank: The global rank to locate in the mesh

    Returns:
        torch.Tensor: The row index where the global rank is found in the mesh
    """
    matches = (mesh == global_rank).any(dim=1)
    return torch.argmax(matches.int()).to(torch.int32)
