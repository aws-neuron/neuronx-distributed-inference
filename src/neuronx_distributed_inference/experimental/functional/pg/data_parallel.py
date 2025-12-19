import torch

from neuronx_distributed_inference.experimental.functional.pg import get_rank_8_by_8


def get_dp_rank(rank: torch.Tensor, world_size: int, dp_degree: int) -> torch.Tensor:
    r"""
    Returns the DP rank based on the passed in rank and TP/DP configuration.

    For example, if world_size is 8 and dp_degree is 2, there are two dp_ranks: 0 and 1.
    Ranks 0-3 will map to dp_rank 0, and ranks 4-7 will map to dp_rank 1

    :param rank: Rank in the world group
    :param world_size: Number of ranks in the world
    :param dp_degree: Data parallel group size

    :return: DP rank for the provided world rank
    """
    assert dp_degree <= world_size, f"DP degree size should be <= world_size, but got {dp_degree=} and {world_size=}"
    assert rank >= 0 and rank < world_size, f"Rank should be between 0 and {world_size} but got {rank=}"

    tp_degree = world_size // dp_degree

    if dp_degree == 8 and tp_degree == 8:
        return get_rank_8_by_8(rank)

    dp_rank = torch.div(
        rank,
        tp_degree,
        rounding_mode="floor"
    ).to(torch.int32)

    return dp_rank
