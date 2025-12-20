import logging
import os
import json
import pytest
import tempfile
from functools import partial

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeSparseMoeBlock,
)

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    NeuronQwen3MoeForCausalLM,
    Qwen3MoeInferenceConfig,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, destroy_mp, init_cpu_env


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CPUQwen3MoeMoEMLPModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = Qwen3MoeSparseMoeBlock(config=config)

    def forward(self, hidden_states):
        outputs, router_logits = self.mlp(hidden_states)
        return outputs


class NeuronQwen3MoeMoEMLPModule(nn.Module):
    """Neuron implementation of Qwen3MoE MLP module for testing"""

    def __init__(self, config):
        super().__init__()
        self.mlp = initialize_moe_module(
            config=config,
        )

    def forward(self, hidden_states):
        outputs = self.mlp(hidden_states)
        return outputs[0]  # Return only the hidden states


def create_random_checkpoint(path, config, dtype, rand_range=0.05):
    """
    Create a random checkpoint for Qwen3MoE MoE MLP module.

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
    intermediate_size = config.moe_intermediate_size
    num_experts = config.num_experts

    random_sd = {}

    # Create random weights for each expert
    for e in range(num_experts):
        # Gate projection weights
        random_sd[f"mlp.experts.{e}.gate_proj.weight"] = (
            torch.randn(intermediate_size, hidden_size) * rand_range
        ).to(dtype)

        # Up projection weights
        random_sd[f"mlp.experts.{e}.up_proj.weight"] = (
            torch.randn(intermediate_size, hidden_size) * rand_range
        ).to(dtype)

        # Down projection weights
        random_sd[f"mlp.experts.{e}.down_proj.weight"] = (
            torch.randn(hidden_size, intermediate_size) * rand_range
        ).to(dtype)

    # Router weights
    random_sd["mlp.gate.weight"] = (torch.randn(num_experts, hidden_size) * rand_range).to(dtype)

    # Create Neuron weights copy
    # Copy router weights
    random_sd["mlp.router.linear_router.weight"] = random_sd["mlp.gate.weight"].detach().clone()

    intermediate_size, hidden_size = random_sd["mlp.experts.0.gate_proj.weight"].shape
    device = random_sd["mlp.experts.0.gate_proj.weight"].device
    dtype = random_sd["mlp.experts.0.gate_proj.weight"].dtype

    # copy the MLP parameters
    gate_up_proj = torch.empty(
        config.num_experts,
        hidden_size,
        2 * intermediate_size,
        dtype=dtype,
        device=device,
    )
    for e in range(config.num_experts):
        # Copy gate_proj and up_proj after concatenation
        gate_proj_weights = random_sd[f"mlp.experts.{e}.gate_proj.weight"].T.detach().clone()
        up_proj_weights = random_sd[f"mlp.experts.{e}.up_proj.weight"].T.detach().clone()

        gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
        gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
        gate_proj_slice.copy_(gate_proj_weights)
        up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
        up_proj_slice.copy_(up_proj_weights)

    random_sd["mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

    down_proj = torch.empty(
        config.num_experts,
        intermediate_size,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    for e in range(config.num_experts):
        # Copy down_proj
        down_proj_weights = random_sd[f"mlp.experts.{e}.down_proj.weight"].T.detach().clone()
        down_proj_slice = torch.narrow(down_proj, 0, e, 1)
        down_proj_slice.copy_(down_proj_weights)
    random_sd["mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

    torch.save(random_sd, path)

    logger.info(f"Created random checkpoints at {path}")
    return random_sd


def load_cpu_model(config, checkpoint_path):
    """Load and return the CPU model with checkpoint loaded"""
    # Set random seed for reproducibility
    set_random_seed(0)

    # Create model directly
    model = CPUQwen3MoeMoEMLPModule(config).to(config.torch_dtype).eval()

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

    # Build and trace the model using utility function
    neuron_model = build_module(
        module_cls=NeuronQwen3MoeMoEMLPModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": neuron_model_config},
        compiler_args=NeuronQwen3MoeForCausalLM(
            checkpoint_path,
            neuron_model_config,
        ).get_compiler_args(),
        compiler_workdir=os.path.join(artifacts_path, "compiler_workdir_moe_mlp"),
        checkpoint_path=checkpoint_path,
    )

    return neuron_model


# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS16 EP32/TP2 for MOE
# TODO: Increase seq len after Teacher forcing is supported
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_optimized.json"), "r") as f:
    ep32_tp2_json = json.load(f)
    ep32_tp2_json["seq_len"] = 512
    ep32_tp2_json["max_context_length"] = 512
MOE_EP32_TP2_NEURON_CONFIG = MoENeuronConfig(**ep32_tp2_json)


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
        (MOE_EP32_TP2_NEURON_CONFIG, 5e-2),
    ],
)
def test_qwen3_moe_moe_mlp(neuron_config, rtol):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(0)

    logger.info("Running Qwen3MoE MoE MLP tp64 test ...")
    dtype = neuron_config.torch_dtype

    # Create configs
    config_path = os.path.join(CURR_DIR, "config.json")
    model_config = Qwen3MoeConfig.from_pretrained(config_path, torch_dtype=dtype)
    print(model_config)

    neuron_model_config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=model_config),
    )

    logger.info(f"Running Qwen3MoE MoE MLP test with dtype {dtype}...")

    # Create test inputs
    batch_size = neuron_config.batch_size
    seq_len = neuron_config.seq_len
    hidden_size = model_config.hidden_size

    # Generate test inputs with controlled random range (0.05)
    hidden_states = (torch.randn((batch_size, seq_len, hidden_size)) * 0.05).to(dtype)

    artifacts_tempdir = tempfile.TemporaryDirectory()
    artifacts_path = artifacts_tempdir.name

    # Create random checkpoint before running tests
    checkpoint_path = os.path.join(
        artifacts_path, f"checkpoint_moe_mlp_{str(dtype).split('.')[1]}.pt"
    )
    create_random_checkpoint(checkpoint_path, model_config, dtype=dtype, rand_range=0.05)

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
    check_results(
        f"moe_mlp_tp64_{str(dtype).split('.')[1]}", neuron_output, cpu_output, rtol=rtol
    )

    # Clean up
    artifacts_tempdir.cleanup()


if __name__ == "__main__":
    test_qwen3_moe_moe_mlp(MOE_EP32_TP2_NEURON_CONFIG, 5e-2)
