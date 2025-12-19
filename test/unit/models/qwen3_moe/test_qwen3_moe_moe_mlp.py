import logging
import os
import tempfile

import pytest
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

# Set random seed for reproducibility
torch.manual_seed(0)
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


class TestQwen3MoeMoEMLP:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        pass

    def check_results(self, test_name, actual_output, expected_output, rtol=1e-5):
        print("-" * 20)
        print(f"Test result of {test_name}:")
        print("actual_output shape:", actual_output.shape)
        print("expected_output shape:", expected_output.shape)
        assert check_accuracy_embeddings(
            actual_output, expected_output, plot_outputs=False, rtol=rtol, atol=0
        )
        print("-" * 20)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_moe_mlp(self, dtype):
        self.model_config = Qwen3MoeConfig(
            attention_bias=False,
            attention_dropout=0.0,
            decoder_sparse_step=1,
            bos_token_id=151643,
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            hidden_size=4096,
            initializer_range=0.02,
            intermediate_size=12288,
            max_position_embeddings=40960,
            max_window_layers=94,
            mlp_only_layers=None,
            moe_intermediate_size=1536,
            norm_topk_prob=True,
            num_attention_heads=32,
            num_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=94,
            num_key_value_heads=4,
            output_router_logits=False,
            rms_norm_eps=1e-6,
            rope_scaling=None,
            rope_theta=1000000.0,
            router_aux_loss_coef=0.001,
            sliding_window=None,
            tie_word_embeddings=False,
            torch_dtype=str(dtype).split(".")[1],
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151936,
        )

        # TODO: Add parameterize for the kernel
        self.neuron_config = MoENeuronConfig(
            batch_size=16,
            ctx_batch_size=1,
            tp_degree=1,
            vocab_size=151936,
            max_context_length=512,
            seq_len=512,
            torch_dtype=str(dtype).split(".")[1],
        )

        self.neuron_model_config = Qwen3MoeInferenceConfig(
            self.neuron_config,
            load_config=load_pretrained_config(hf_config=self.model_config),
        )

        logger.info(f"Running Qwen3MoE MoE MLP test with dtype {dtype}...")

        # Create test inputs
        batch_size = self.neuron_config.batch_size
        seq_length = self.neuron_config.seq_len
        hidden_size = self.model_config.hidden_size

        # Generate test inputs with controlled random range (0.05)
        hidden_states = (torch.randn((batch_size, seq_length, hidden_size)) * 0.05).to(dtype)

        artifacts_tempdir = tempfile.TemporaryDirectory()
        artifacts_path = artifacts_tempdir.name

        # Create random checkpoint before running tests
        checkpoint_path = os.path.join(
            artifacts_path, f"checkpoint_moe_mlp_{str(dtype).split('.')[1]}.pt"
        )
        create_random_checkpoint(checkpoint_path, self.model_config, dtype=dtype, rand_range=0.05)

        # Load CPU model
        cpu_model = load_cpu_model(config=self.model_config, checkpoint_path=checkpoint_path)

        # Run CPU inference (always use float32)
        logger.info("Running inference on CPU model")
        with torch.no_grad():
            cpu_output = cpu_model(hidden_states)

        # Create example inputs tuple for Neuron model
        hidden_states.to(dtype)
        example_inputs = [(hidden_states,)]

        # Load Neuron model
        neuron_model = load_neuron_model(
            neuron_model_config=self.neuron_model_config,
            checkpoint_path=checkpoint_path,
            example_inputs=example_inputs,
            artifacts_path=artifacts_path,
        )

        # Run Neuron inference
        logger.info("Running inference on Neuron model")
        neuron_output = neuron_model(hidden_states)

        # Check results with different rtol based on dtype
        rtol = 0.05 if dtype == torch.bfloat16 else 1e-5
        self.check_results(
            f"moe_mlp_{str(dtype).split('.')[1]}", neuron_output, cpu_output, rtol=rtol
        )

        # Clean up
        artifacts_tempdir.cleanup()
