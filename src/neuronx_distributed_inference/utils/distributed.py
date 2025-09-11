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


def get_cp_rank(global_rank: torch.tensor, tp_degree: int, cp_degree: int = None):
    if cp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(global_rank, tp_degree)

    cp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return cp_rank


def get_dp_rank(global_rank: torch.tensor, tp_degree: int, dp_degree: int = None):
    if dp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(global_rank, tp_degree)

    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return dp_rank


def split_along_dim(tensor: torch.tensor, dim: int, rank: int, num_partitions: int):
    if tensor is None:
        return None

    num_per_partition = divide(tensor.size(dim), num_partitions)
    indices = torch.arange(0, num_per_partition, device=tensor.device)
    indices = indices + (rank * num_per_partition)
    tensor = torch.index_select(tensor, dim=dim, index=indices)

    return tensor


def get_rank_8_by_8(global_rank, tp_degree):
    """
    # Using the 8x8 mesh as an example with 31 as our input:
    # The pattern repeats every 2 rows, i.e. odd and even row structure, call this pattern a "block"
    # Calculate which block we fall into 31 // 16 = 1
    # Assuming we didn't have the interleaving pattern, compute it's position 31 % 16 = 15
    # Simply assume we fall into an even row to begin with, block 1 = rows 2, 3 assume we're in 2
    # When we account for the partial contiguity, the positions that fall into an odd row are 4 - 11 (half width to 3 * half_width)
    # Check if the position id falls into that range and offset the initial even row assumption
    """

    block_size = 2 * tp_degree
    block_idx = torch.div(global_rank, block_size, rounding_mode="floor").to(torch.int32)
    pos_in_block = global_rank % block_size
    half_width = torch.div(tp_degree, 2, rounding_mode="floor").to(torch.int32)

    # Calculate row indices
    row_idx = block_idx * 2  # Start with even row indices

    # Numbers in odd rows have positions between half_width and 3*half_width-1
    mask_odd_row = (pos_in_block >= half_width) & (pos_in_block < 3 * half_width)
    row_idx = row_idx + mask_odd_row.int()

    return row_idx
