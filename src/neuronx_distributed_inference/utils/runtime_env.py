import collections
import os
from neuronx_distributed_inference.models.config import NeuronConfig


LONG_CONTEXT_RUNTIME_ENV_VARS = {
    "NEURON_RT_EXEC_TIMEOUT": "600",  # sometimes long context neff can take time to load/execute
    "NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE": "1",  # LNC2 related scratchpad optimization
}


def set_env_vars(neuron_config: NeuronConfig) -> None:
    """
    Set environment variables if they're not already set.

    Args:
        neuron_config (NeuronConfig): config contains env var info
    """

    env_vars = collections.defaultdict()
    if neuron_config.enable_long_context_mode:
        env_vars.update(LONG_CONTEXT_RUNTIME_ENV_VARS)

    if neuron_config.scratchpad_page_size:
        env_vars.update({"NEURON_SCRATCHPAD_PAGE_SIZE": f"{neuron_config.scratchpad_page_size}"})

    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = str(value)
