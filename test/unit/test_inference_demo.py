from unittest.mock import patch, MagicMock
import sys
from neuronx_distributed_inference.inference_demo import parse_args, create_neuron_config
from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM


class TestConfig(InferenceConfig):
    """Minimal test config class that implements required methods"""

    def __init__(self, neuron_config, load_config):
        self.neuron_config = neuron_config

    def load_config(self):
        pass

    def get_required_attributes(self):
        return []

    def get_neuron_config(self):
        return self.neuron_config


def test_attention_kernel_configuration():
    test_cases = [
        {
            "args": [
                "--model-type", "llama",
                "--task-type", "causal-lm",
                "run",
                "--model-path", "dummy/path",
                "--compiled-model-path", "dummy/compiled/path",
                "--prompt", "test"
            ],
            "expected_enabled": None,
            "desc": "Default behavior - kernel should be enabled (None)"
        },
        {
            "args": [
                "--model-type", "llama",
                "--task-type", "causal-lm",
                "run",
                "--model-path", "dummy/path",
                "--compiled-model-path", "dummy/compiled/path",
                "--prompt", "test",
                "--attn-kernel-enabled"
            ],
            "expected_enabled": True,
            "desc": "With --attn-kernel-enabled - kernel should be enabled"
        },
        {
            "args": [
                "--model-type", "llama",
                "--task-type", "causal-lm",
                "run",
                "--model-path", "dummy/path",
                "--compiled-model-path", "dummy/compiled/path",
                "--prompt", "test",
                "--no-attn-kernel-enabled"
            ],
            "expected_enabled": False,
            "desc": "With --no-attn-kernel-enabled - kernel should be disabled"
        }
    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['desc']}")

        # Set up command line arguments
        with patch.object(sys, 'argv', ['inference_demo.py'] + test_case["args"]):
            # Parse arguments
            args = parse_args()
            _, neuron_config = create_neuron_config(NeuronLlamaForCausalLM, args)

            # Check if attention kernel is configured as expected
            assert neuron_config.attn_kernel_enabled == test_case["expected_enabled"], \
                f"Test failed: {test_case['desc']}\n" \
                f"Expected attn_kernel_enabled={test_case['expected_enabled']}, " \
                f"got {neuron_config.attn_kernel_enabled}"

            print(f"attn_kernel_enabled={neuron_config.attn_kernel_enabled}")
            print(f"Test passed: {test_case['desc']}")


def main():
    test_attention_kernel_configuration()


if __name__ == "__main__":
    main()