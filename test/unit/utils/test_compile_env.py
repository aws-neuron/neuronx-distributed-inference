from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.compile_env import set_compile_env_vars
import os
import unittest

def clear_compile_flags():
    if "DISABLE_NUMERIC_CC_TOKEN" in os.environ:
        del os.environ["DISABLE_NUMERIC_CC_TOKEN"]

class TestCompileEnv(unittest.TestCase):

    def test_set_compile_env_vars_disabled_numeric_token(self):
        # Setup
        clear_compile_flags()
        neuron_config = NeuronConfig(disable_numeric_cc_token=True)

        # Act
        set_compile_env_vars(neuron_config)

        # Verify
        assert "DISABLE_NUMERIC_CC_TOKEN" in os.environ
        assert os.environ["DISABLE_NUMERIC_CC_TOKEN"] == "1"

        # Cleanup
        clear_compile_flags()

    def test_set_compile_env_vars_enabled_numeric_token(self):
        # Setup
        clear_compile_flags()
        neuron_config = NeuronConfig(disable_numeric_cc_token=False)

        # Act
        set_compile_env_vars(neuron_config)

        # Verify
        assert "DISABLE_NUMERIC_CC_TOKEN" not in os.environ

        # Cleanup
        clear_compile_flags()

    def test_set_compile_env_vars_default_config(self):
        # Setup
        clear_compile_flags()
        neuron_config = NeuronConfig()

        # Act
        set_compile_env_vars(neuron_config)

        # Verify
        assert "DISABLE_NUMERIC_CC_TOKEN" not in os.environ

        # Cleanup
        clear_compile_flags()


if __name__ == '__main__':
    unittest.main()