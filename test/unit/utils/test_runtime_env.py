from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.runtime_env import set_env_vars, LONG_CONTEXT_RUNTIME_ENV_VARS
import os
import unittest

LONG_CONTEXT_LEN = 32 * 1024

def clear_flags():
    for var in ["NEURON_RT_EXEC_TIMEOUT", "NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE", "NEURON_SCRATCHPAD_PAGE_SIZE"]:
        if var in os.environ:
            del os.environ[var]


class TestRuntimeEnv(unittest.TestCase):

    def test_set_env_vars_scratchpad_sz(self):
        # Setup
        clear_flags()
        neuron_config = NeuronConfig(scratchpad_page_size=1024)

        # Act
        set_env_vars(neuron_config)

        # Verify
        assert "NEURON_SCRATCHPAD_PAGE_SIZE" in os.environ
        assert os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] == "1024"
        for var in LONG_CONTEXT_RUNTIME_ENV_VARS:
            assert var not in os.environ

        # Cleanup
        clear_flags()

    def test_set_env_vars_long_context(self):
        # Setup
        clear_flags()
        neuron_config = NeuronConfig(max_context_length=LONG_CONTEXT_LEN, max_length=LONG_CONTEXT_LEN)

        # Act
        set_env_vars(neuron_config)

        # Verify
        assert os.environ.get("NEURON_RT_EXEC_TIMEOUT") == "600"
        assert os.environ.get("NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE") == "1"
        assert os.environ.get('NEURON_SCRATCHPAD_PAGE_SIZE') == '1024'

        # Clean up
        clear_flags()

    def test_set_env_vars_long_context_and_scratchpad_sz(self):
        # Setup
        clear_flags()
        neuron_config = NeuronConfig(max_context_length=LONG_CONTEXT_LEN, max_length=LONG_CONTEXT_LEN, scratchpad_page_size=4096)

        # Act
        set_env_vars(neuron_config)

        # Verify
        assert os.environ.get('NEURON_RT_EXEC_TIMEOUT') == '600'
        assert os.environ.get('NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE') == '1'
        assert os.environ.get('NEURON_SCRATCHPAD_PAGE_SIZE') == '4096'

        # Cleanup
        clear_flags()


if __name__ == '__main__':
    unittest.main()