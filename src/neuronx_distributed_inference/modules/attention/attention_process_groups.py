import torch

_ATTENTION_TP_CP_GROUP = None
_ATTENTION_CP_GROUP = None
_ATTENTION_TP_DP_GROUP = None
_ATTENTION_DP_GROUP = None

# TODO: Add mesh validation per instance to fail fast on non-working TP, CP configurations


def tp_mesh_8_by_8():
    """
    Follows the 8x8 paradigm for TRN2 topology.
    """
    mesh = [
        [0, 1, 2, 3, 12, 13, 14, 15],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [16, 17, 18, 19, 28, 29, 30, 31],
        [20, 21, 22, 23, 24, 25, 26, 27],
        [32, 33, 34, 35, 44, 45, 46, 47],
        [36, 37, 38, 39, 40, 41, 42, 43],
        [48, 49, 50, 51, 60, 61, 62, 63],
        [52, 53, 54, 55, 56, 57, 58, 59]
    ]

    return mesh


def _fully_contiguous_tp_mesh(tp_degree, cp_degree):
    tp_cp_group_size = tp_degree // cp_degree

    tp_cp_group_mesh = [
        list(range(tp_degree))[i : i + tp_cp_group_size]
        for i in range(0, tp_degree, tp_cp_group_size)
    ]

    return tp_cp_group_mesh


def get_tp_cp_group_mesh(tp_degree, cp_degree):

    if cp_degree == 8 and (tp_degree // cp_degree) == 8:
        return tp_mesh_8_by_8()

    return _fully_contiguous_tp_mesh(tp_degree, cp_degree)


def get_flattened_inverted_tp_cp_group_mesh(tp_degree, cp_degree):
    """
    Flattens the CP mesh and then inverts it. Useful in cases such as when doing SP with a non contiguous mesh.
    The inverted mesh dictates how to reorder the tensor to be contiguous after doing a non-contiguous gather.
    """

    mesh = sum(get_tp_cp_group_mesh(tp_degree, cp_degree), [])

    n = len(mesh)
    inv_mesh = [0] * n

    for idx, rank in enumerate(mesh):
        inv_mesh[rank] = idx

    return inv_mesh


def get_cp_group_mesh(tp_degree, cp_degree):
    tp_cp_group_size = tp_degree // cp_degree

    tp_cp_group_mesh = get_tp_cp_group_mesh(tp_degree, cp_degree)
    cp_group_mesh = [[row[i] for row in tp_cp_group_mesh] for i in range(tp_cp_group_size)]

    return cp_group_mesh


def init_context_parallel_attention_process_groups(config):
    """
    initializes process groups needed to run context parallel attention

    example: TP = 8, CP = 4

    Attention will run in TP = 8 // 4 = 2

    _ATTENTION_TP_CP_GROUP = [[0, 1], [2, 3], [4, 5], [6, 7]]
    _ATTENTION_CP_GROUP = [[0, 2, 4, 6], [1, 3, 5, 7]]
    """

    global _ATTENTION_TP_CP_GROUP
    global _ATTENTION_CP_GROUP

    tp_degree = config.neuron_config.tp_degree
    cp_degree = config.neuron_config.cp_degree

    if cp_degree > 1 and _ATTENTION_CP_GROUP is None and _ATTENTION_TP_CP_GROUP is None:
        tp_cp_group_mesh = get_tp_cp_group_mesh(tp_degree, cp_degree)
        tp_cp_group = torch.distributed.new_group(
            tp_cp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": tp_cp_group_mesh}}
        )
        _ATTENTION_TP_CP_GROUP = tp_cp_group

        cp_group_mesh = get_cp_group_mesh(tp_degree, cp_degree)
        cp_group = torch.distributed.new_group(
            cp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": cp_group_mesh}}
        )
        _ATTENTION_CP_GROUP = cp_group


def get_context_parallel_attention_tp_group():
    assert _ATTENTION_TP_CP_GROUP is not None, "_ATTENTION_TP_CP_GROUP is not initialized"

    return _ATTENTION_TP_CP_GROUP


def get_context_parallel_attention_cp_group():
    assert _ATTENTION_CP_GROUP is not None, "_ATTENTION_CP_GROUP is not initialized"

    return _ATTENTION_CP_GROUP


def init_data_parallel_attention_process_groups(config):
    """
    initializes process groups needed to run data parallel attention

    example: TP = 8, DP = 4

    Attention will run in TP = 8 // 4 = 2

    _ATTENTION_TP_DP_GROUP = [[0, 1], [2, 3], [4, 5], [6, 7]]
    _ATTENTION_DP_GROUP = [[0, 2, 4, 6], [1, 3, 5, 7]]
    """

    global _ATTENTION_TP_DP_GROUP
    global _ATTENTION_DP_GROUP

    tp_degree = config.neuron_config.tp_degree
    dp_degree = config.neuron_config.attention_dp_degree

    if dp_degree > 1 and _ATTENTION_DP_GROUP is None and _ATTENTION_TP_DP_GROUP is None:
        tp_dp_group_mesh = get_tp_cp_group_mesh(tp_degree, dp_degree)
        tp_dp_group = torch.distributed.new_group(
            tp_dp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": tp_dp_group_mesh}}
        )
        _ATTENTION_TP_DP_GROUP = tp_dp_group

        dp_group_mesh = get_cp_group_mesh(tp_degree, dp_degree)
        dp_group = torch.distributed.new_group(
            dp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": dp_group_mesh}}
        )
        _ATTENTION_DP_GROUP = dp_group


def get_data_parallel_attention_tp_group():
    assert _ATTENTION_TP_DP_GROUP is not None, "_ATTENTION_TP_DP_GROUP is not initialized"

    return _ATTENTION_TP_DP_GROUP


def get_data_parallel_attention_dp_group():
    assert _ATTENTION_DP_GROUP is not None, "_ATTENTION_DP_GROUP is not initialized"

    return _ATTENTION_DP_GROUP
