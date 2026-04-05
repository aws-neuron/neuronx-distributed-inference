import logging
import os
import pytest
import tempfile

import torch
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Config,
    Gemma3MLP,
)

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import (
    NeuronGemma3ForCausalLM,
    Gemma3InferenceConfig,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, destroy_mp


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS1 baseline TP2 configuration
BASELINE_NEURON_CONFIG = NeuronConfig(
    tp_degree=2,
    cp_degree=1,
    attention_dp_degree=1,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    max_context_length=1024,
    seq_len=1024,
    sequence_parallel_enabled=False,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=False,
    fused_qkv=False,
    torch_dtype=torch.float32,
)


class CPUGemma3MLPModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = Gemma3MLP(config=config)

    def forward(self, hidden_states):
        outputs = self.mlp(hidden_states)
        return outputs


class NeuronGemma3MLPModule(nn.Module):
    """Neuron implementation of Gemma3 MLP module for testing"""

    def __init__(self, config):
        super().__init__()
        self.mlp = NeuronLlamaMLP(config=config)

    def forward(self, hidden_states):
        outputs = self.mlp(hidden_states)
        return outputs[0]  # Return only the hidden states


def create_random_checkpoint(path, config, dtype, rand_range=0.05):
    """
    Create a random checkpoint for Gemma3 MLP module.

    Args:
        path: Path to save the checkpoint
        config: Model configuration
        dtype: Data type for tensors
        rand_range: Range for random values (default: 0.05)

    Returns:
        Dictionary containing random state dict
    """
    if os.path.exists(path):
        os.remove(path)

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    random_sd = {}

    # Create random weights for MLP layers
    # Gate projection weights
    random_sd["mlp.gate_proj.weight"] = (
        torch.randn(intermediate_size, hidden_size) * rand_range
    ).to(dtype)

    # Up projection weights
    random_sd["mlp.up_proj.weight"] = (torch.randn(intermediate_size, hidden_size) * rand_range).to(
        dtype
    )

    # Down projection weights
    random_sd["mlp.down_proj.weight"] = (
        torch.randn(hidden_size, intermediate_size) * rand_range
    ).to(dtype)

    torch.save(random_sd, path)

    logger.info(f"Created random checkpoints at {path}")
    return random_sd


def load_cpu_model(config, checkpoint_path):
    """Load and return the CPU model with checkpoint loaded"""
    # Set random seed for reproducibility
    set_random_seed(0)

    # Create model directly
    model = CPUGemma3MLPModule(config).to(config.torch_dtype).eval()

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model


def load_neuron_model(neuron_model_config, checkpoint_path, example_inputs, artifacts_path):
    """Load and return the Neuron model with checkpoint loaded"""
    os.environ["NXD_CPU_MODE"] = "0"
    destroy_mp()
    set_random_seed(0)

    # Create a wrapper class that sets the model to eval mode
    class EvalNeuronGemma3MLPModule(NeuronGemma3MLPModule):
        def __init__(self, config):
            super().__init__(config)
            self.eval()  # Set to evaluation mode for inference

    # Build and trace the model using utility function
    neuron_model = build_module(
        module_cls=EvalNeuronGemma3MLPModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": neuron_model_config},
        compiler_args=NeuronGemma3ForCausalLM(
            checkpoint_path,
            neuron_model_config,
        ).get_compiler_args(),
        compiler_workdir=os.path.join(artifacts_path, "compiler_workdir_mlp"),
        checkpoint_path=checkpoint_path,
    )

    return neuron_model


def check_results(test_name, actual_output, expected_output, rtol=1e-5):
    print("-" * 20)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    assert check_accuracy_embeddings(
        actual_output, expected_output, plot_outputs=False, rtol=rtol, atol=0
    )
    print("-" * 20)


@pytest.mark.parametrize(
    "neuron_config, rtol",
    [
        (BASELINE_NEURON_CONFIG, 5e-2),
    ],
)
def test_gemma3_mlp(neuron_config, rtol):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(0)

    dtype = neuron_config.torch_dtype

    # Create configs
    config_path = os.path.join(CURR_DIR, "config.json")
    model_config = Gemma3Config.from_pretrained(config_path).get_text_config()

    neuron_model_config = Gemma3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=model_config),
    )

    logger.info(f"Running Gemma3 MLP test with dtype {dtype}...")

    # Create test inputs
    batch_size = neuron_config.batch_size
    seq_len = neuron_config.seq_len
    hidden_size = model_config.hidden_size

    # Generate test inputs with controlled random range (0.05)
    hidden_states = (torch.randn((batch_size, seq_len, hidden_size)) * 0.05).to(dtype)

    artifacts_tempdir = tempfile.TemporaryDirectory()
    artifacts_path = artifacts_tempdir.name

    # Create random checkpoint before running tests
    checkpoint_path = os.path.join(artifacts_path, f"checkpoint_mlp_{str(dtype).split('.')[1]}.pt")
    create_random_checkpoint(checkpoint_path, neuron_model_config, dtype=dtype, rand_range=0.05)

    # Load CPU model
    cpu_model = load_cpu_model(config=model_config, checkpoint_path=checkpoint_path)

    # Run CPU inference (always use float32)
    logger.info("Running inference on CPU model")
    with torch.no_grad():
        cpu_output = cpu_model(hidden_states)

    # Create example inputs tuple for Neuron model
    hidden_states.to(dtype)
    example_inputs = [(hidden_states,)]

    # Load Neuron model
    neuron_model = load_neuron_model(
        neuron_model_config=neuron_model_config,
        checkpoint_path=checkpoint_path,
        example_inputs=example_inputs,
        artifacts_path=artifacts_path,
    )

    # Run Neuron inference
    logger.info("Running inference on Neuron model")
    neuron_output = neuron_model(hidden_states)

    # Check results with different rtol based on dtype
    check_results(f"mlp_tp64_{str(dtype).split('.')[1]}", neuron_output, cpu_output, rtol=rtol)

    # Clean up
    artifacts_tempdir.cleanup()


if __name__ == "__main__":
    test_gemma3_mlp(BASELINE_NEURON_CONFIG, 5e-2)
