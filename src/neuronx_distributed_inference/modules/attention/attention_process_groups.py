import torch

from neuronx_distributed_inference.models.config import InferenceConfig

_ATTENTION_TP_CP_GROUP = None
_ATTENTION_CP_GROUP = None

# TODO: Add mesh validation per instance to fail fast on non-working TP, CP configurations


def get_tp_cp_group_mesh(tp_degree, cp_degree):
    tp_cp_group_size = tp_degree // cp_degree
    tp_cp_group_mesh = [
        list(range(tp_degree))[i : i + tp_cp_group_size]
        for i in range(0, tp_degree, tp_cp_group_size)
    ]

    return tp_cp_group_mesh


def get_cp_group_mesh(tp_degree, cp_degree):
    tp_cp_group_size = tp_degree // cp_degree

    tp_cp_group_mesh = get_tp_cp_group_mesh(tp_degree, cp_degree)
    cp_group_mesh = [[row[i] for row in tp_cp_group_mesh] for i in range(tp_cp_group_size)]

    return cp_group_mesh


def init_context_parallel_attention_process_groups(config: InferenceConfig):
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
