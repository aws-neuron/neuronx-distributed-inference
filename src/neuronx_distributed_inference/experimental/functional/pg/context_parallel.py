"""
File containing ProcessGroup management APIs for context parallelism
"""

import torch
from torch.distributed import ProcessGroup


# Global state management
_ATTENTION_TP_CP_GROUP = None
_ATTENTION_CP_GROUP = None

# Follows the 8x8 paradigm for TRN2 topology
_TRN2_8_BY_8_TP_MESH = [
    [0, 1, 2, 3, 12, 13, 14, 15],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [16, 17, 18, 19, 28, 29, 30, 31],
    [20, 21, 22, 23, 24, 25, 26, 27],
    [32, 33, 34, 35, 44, 45, 46, 47],
    [36, 37, 38, 39, 40, 41, 42, 43],
    [48, 49, 50, 51, 60, 61, 62, 63],
    [52, 53, 54, 55, 56, 57, 58, 59],
]

# -- public functions at the top


def initialize_context_parallel_tp_group(world_size: int, cp_degree: int) -> None:
    """
    Creates the tp-mesh and creates a new group.

    Args:
        world_size: world size
        cp_degree: cp degree
    """
    global _ATTENTION_TP_CP_GROUP

    assert world_size % cp_degree == 0, f"World size ({world_size}) must be evenly divisble by CP degree ({cp_degree})"

    if _ATTENTION_TP_CP_GROUP is None:
        tp_cp_group_mesh: list[list[int]] = get_context_parallel_tp_mesh(world_size, cp_degree)
        tp_cp_group: ProcessGroup = torch.distributed.new_group(
            tp_cp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": tp_cp_group_mesh}}
        )

        _ATTENTION_TP_CP_GROUP = tp_cp_group


def initialize_context_parallel_cp_group(world_size: int, cp_degree: int) -> None:
    """
    Creates the cp-mesh and creates a new group.

    Args:
        world_size: world size
        cp_degree: cp degree
    """
    global _ATTENTION_CP_GROUP

    assert world_size % cp_degree == 0, f"World size ({world_size}) must be evenly divisble by CP degree ({cp_degree})"

    if _ATTENTION_CP_GROUP is None:
        cp_group_mesh: list[list[int]] = get_context_parallel_cp_mesh(world_size, cp_degree)
        cp_group: ProcessGroup = torch.distributed.new_group(
            cp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": cp_group_mesh}}
        )

        _ATTENTION_CP_GROUP = cp_group


def initialize_context_parallel_process_groups(world_size: int, cp_degree: int) -> None:
    """
    Creates the tp-mesh, cp-mesh and creates both groups.

    Args:
        world_size: world size
        cp_degree: cp degree
    """
    initialize_context_parallel_tp_group(world_size, cp_degree)
    initialize_context_parallel_cp_group(world_size, cp_degree)


def get_context_parallel_tp_mesh(world_size: int, cp_degree: int) -> list[list[int]]:
    """
    Returns a List of Lists indicating the mesh used for TP in context parallel.
    Handles special use-cases such as 8x8.

    Args:
        world_size: world size
        cp_degree: cp degree
    """
    tp_degree = world_size // cp_degree

    if cp_degree == 8 and tp_degree == 8:
        return _TRN2_8_BY_8_TP_MESH

    # Generate a contiguous TP mesh based on cp_group size
    tp_cp_group_mesh = [
        list(range(world_size))[i : i + tp_degree]
        for i in range(0, world_size, tp_degree)
    ]

    return tp_cp_group_mesh


def get_context_parallel_cp_mesh(world_size: int, cp_degree: int) -> list:
    """
    Returns a List of Lists indicating the mesh used for KV gather in context parallel.
    Handles special use-cases such as 8x8. This is an inverse of the above mesh.

    Args:
        world_size: world size
        cp_degree: cp degree
    """
    tp_degree = world_size // cp_degree

    # We transpose the TP mesh here
    tp_cp_group_mesh = get_context_parallel_tp_mesh(world_size, cp_degree)
    cp_group_mesh = [[row[i] for row in tp_cp_group_mesh] for i in range(tp_degree)]

    return cp_group_mesh


def get_context_parallel_tp_group() -> ProcessGroup:
    """
    Returns the context parallel tp-group, throws an error if not initialized.
    """
    assert _ATTENTION_TP_CP_GROUP is not None, "_ATTENTION_TP_CP_GROUP is not initialized"

    return _ATTENTION_TP_CP_GROUP


def get_context_parallel_cp_group() -> ProcessGroup:
    """
    Returns the context parallel cp-group, throws an error if not initialized.
    """
    assert _ATTENTION_CP_GROUP is not None, "_ATTENTION_CP_GROUP is not initialized"

    return _ATTENTION_CP_GROUP


def get_cp_rank(rank: torch.Tensor, world_size: int, cp_degree: int) -> torch.Tensor:
    r"""
    Returns the cp_rank based on the passed in rank and TP/CP configuration.

    For example, if world_size is 8 and cp_degree is 2, there are two cp_ranks: 0 and 1.
    Ranks 0-3 will map to cp_rank 0, and ranks 4-7 will map to cp_rank 1

    :param rank: Rank in the world group
    :param world_size: Number of ranks in the world
    :param cp_degree: Context parallel group size

    :return: CP rank for the provided world rank
    """
    assert cp_degree <= world_size, f"Cp degree size should be <= world_size, but got {cp_degree=} and {world_size=}"
    assert rank >= 0 and rank < world_size, f"Rank should be between 0 and {world_size} but got {rank=}"

    tp_degree = world_size // cp_degree

    if cp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(rank)

    cp_rank = torch.div(
        rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return cp_rank


def get_rank_8_by_8(global_rank: torch.Tensor) -> torch.Tensor:
    r"""
    Special handling to get cp_rank or dp_rank for 8x8 due to 8x8 being non-contiguous.

    Implementation as below (avoid for loops for XLA tracing):
    Using the 8x8 mesh as an example with 31 as our input:
    The pattern repeats every 2 rows, i.e. odd and even row structure, call this pattern a "block"
    Calculate which block we fall into 31 // 16 = 1
    Assuming we didn't have the interleaving pattern, compute it's position 31 % 16 = 15
    Simply assume we fall into an even row to begin with, block 1 = rows 2, 3 assume we're in 2
    When we account for the partial contiguity, the positions that fall into an odd row are 4 - 11 (half width to 3 * half_width)
    Check if the position id falls into that range and offset the initial even row assumption

    :param global_rank: global_rank

    :return: cp_rank or dp_rank for 8x8 case
    """
    assert global_rank >= 0 and global_rank <= 63, f"Rank must be between 0 and 63, but got {global_rank}"

    tp_degree = 8

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


# -- private functions go below
