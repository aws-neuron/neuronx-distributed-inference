import os

from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402


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
