import logging
import os
import time
import unittest
from functools import partial

import torch
import torch.nn as nn
from torch_neuronx.testing.validation import custom_allclose
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
)
from transformers.cache_utils import DynamicCache

from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
    NeuronQwen3MoEAttention,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, init_cpu_env, destroy_mp

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("matplotlib not found. Install via `pip install matplotlib`.")
    matplotlib = None
    plt = None

ARTIFACTS_FOLDER = "/tmp/qwen3_moe/artifacts/"
DTYPE = torch.float32
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

torch.manual_seed(0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Qwen3MoeAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Qwen3MoeAttention(config=config, layer_idx=0)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value=None):
        # Compute position embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Run attention and return all outputs
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=True,
        )

        return outputs[0]


class NeuronQwen3MoeAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NeuronQwen3MoEAttention(config=config)

    def forward(self, hidden_states, attention_mask, position_ids, past_k=None, past_v=None):
        # Run attention and return all outputs
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(past_k, past_v) if past_k is not None and past_v is not None else None,
        )
        return (outputs[0], outputs[1][0], outputs[1][1])


def build_4d_causal_mask(attn_2d: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    bsz, seq_len = attn_2d.shape
    device      = attn_2d.device
    minus_inf   = torch.finfo(dtype).min

    # ① causal pattern (lower-triangular ones) – shape (1,1,S,S)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))[None, None]

    # ② broadcast padding mask to (B,1,1,S)
    pad = attn_2d.to(dtype)[:, None, None, :]

    # ③ allow only positions that are both causal AND non-padding
    keep = causal * pad          # (B,1,S,S)

    # ④ convert to additive form expected by attention kernels
    mask_4d = torch.where(keep.bool(), torch.zeros_like(keep), torch.full_like(keep, minus_inf))
    return mask_4d


def create_context_attn_mask(attention_mask):
    # Lower triangle causal mask for classic attention
    batch_size, n_positions = attention_mask.shape
    mask = torch.full(
        (n_positions, n_positions), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)

    expanded_mask = (
        attention_mask[:, None, None, :]
        .expand(batch_size, 1, n_positions, n_positions)
        .to(torch.bool)
    )
    return torch.logical_and(mask, expanded_mask)


def create_random_checkpoint(path, config, dtype=torch.float32, rand_range=0.05):
    """
    Create a random checkpoint for Qwen3MoE attention module.

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
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    random_sd = {
        # Query projection weights
        "attention.q_proj.weight": (torch.randn(num_attention_heads * head_dim, hidden_size) * rand_range).to(dtype),

        # Key and value projection weights
        "attention.k_proj.weight": (torch.randn(num_key_value_heads * head_dim, hidden_size) * rand_range).to(dtype),
        "attention.v_proj.weight": (torch.randn(num_key_value_heads * head_dim, hidden_size) * rand_range).to(dtype),

        # Output projection
        "attention.o_proj.weight": (torch.randn(hidden_size, num_attention_heads * head_dim) * rand_range).to(dtype),

        # Normalization weights (RMSNorm)
        "attention.q_norm.weight": (torch.randn(head_dim) * rand_range).to(torch.float32),
        "attention.k_norm.weight": (torch.randn(head_dim) * rand_range).to(torch.float32),
    }
    random_sd["attention.q_layernorm.weight"] = random_sd["attention.q_norm.weight"]
    random_sd["attention.k_layernorm.weight"] = random_sd["attention.k_norm.weight"]

    torch.save(random_sd, path)
    logger.info(f"Created random checkpoint at {path}")
    return random_sd


def get_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    # Add flags for cc-overlap
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    compiler_args += " --auto-cast=none"
    # Enable vector-offset DGE
    compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
    print(f"compiler_args: {compiler_args}")
    return compiler_args


def load_cpu_model(config, checkpoint_path):
    """Load and return the CPU model with checkpoint loaded"""
    # Initialize CPU environment
    init_cpu_env()

    # Set random seed for reproducibility
    set_random_seed(0)

    # Create model directly
    model = Qwen3MoeAttentionModule(config).eval()

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model


def load_neuron_model(model_config, neuron_config, checkpoint_path, example_inputs):
    """Load and return the Neuron model with checkpoint loaded"""
    os.environ["NXD_CPU_MODE"] = "0"
    destroy_mp()
    set_random_seed(0)

    # Build and trace the model using utility function
    neuron_model = build_module(
        module_cls=NeuronQwen3MoeAttentionModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": model_config},
        tp_degree=neuron_config.tp_degree,
        compiler_args=get_compiler_args(),
        compiler_workdir=os.path.join(ARTIFACTS_FOLDER, "compiler_workdir"),
        checkpoint_path=checkpoint_path,
    )

    return neuron_model


def check_accuracy(
    actual_output: torch.Tensor,
    expected_output: torch.Tensor,
    plot_outputs: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
):
    assert (
        expected_output.dtype == actual_output.dtype
    ), f"dtypes {expected_output.dtype} and {actual_output.dtype} does not match!"
    dtype = expected_output.dtype

    # Set default rtol, atol based on dtype if not provided
    if not rtol:
        if dtype == torch.bfloat16:
            rtol = 0.05
        elif dtype == torch.float32:
            rtol = 0.01
        else:
            NotImplementedError(f"Specify rtol for dtype {dtype}")
    print(f"Using rtol = {rtol} for dtype {dtype}")
    print(f"Using atol = {atol}")

    if plot_outputs and matplotlib and plt:
        # Save plot, expecting a y=x straight line
        matplotlib.rcParams["agg.path.chunksize"] = 10000
        matplotlib.rcParams["path.simplify_threshold"] = 1.0
        plt.plot(
            actual_output.float().detach().numpy().reshape(-1),
            expected_output.float().detach().numpy().reshape(-1),
        )
        plt.xlabel("Actual Output")
        plt.ylabel("Expected Output")
        plot_path = "plot.png"
        plt.savefig(plot_path, format="png")
        print(f"Saved outputs plot to {plot_path}.")

    # NxD logit validation tests uses this method
    # equivalent to torch.allclose except rtol is multiplied by absolute max, not abs
    # this matches the behavior of the compiler's birsim-to-xla_infergoldens verification
    passed, max_err = custom_allclose(expected_output, actual_output, atol=atol, rtol=rtol)
    print(f"Accuracy validation passed: {passed}, max_err: {max_err}")
    return passed


class TestQwen3Moe(unittest.TestCase):
    def setUp(self):
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
            num_attention_heads=64,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=94,
            num_key_value_heads=4,
            output_router_logits=False,
            rms_norm_eps=1e-6,
            rope_scaling=None,
            rope_theta=1000000.0,
            router_aux_loss_coef=0.001,
            sliding_window=None,
            tie_word_embeddings=False,
            torch_dtype="float32",
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151936,
        )

        self.neuron_config = MoENeuronConfig(
            tp_degree=1,
            batch_size=1,
            max_context_length=512,
            seq_len=512*10,
            torch_dtype="float32",
        )

        self.neuron_model_config = Qwen3MoeInferenceConfig(
            self.neuron_config,
            load_config=load_pretrained_config(hf_config=self.model_config),
        )

    def check_results(self, test_name, actual_output, expected_output, rtol=1e-5):
        print("-" * 20)
        print(f"Test result of {test_name}:")
        print("actual_output shape:", actual_output.shape)
        print("expected_output shape:", expected_output.shape)
        self.assertTrue(
            check_accuracy(
                actual_output, expected_output, plot_outputs=False, rtol=rtol, atol=0
            )
        )
        print("-" * 20)

    def test_attention(self):
        logger.info("Running Qwen3MoeAttention test ...")

        # Create test inputs
        batch_size = 1
        context_length = 16  # Length for context/prefill phase
        decode_length = 1    # Length for decode phase
        hidden_size = 4096

        # Generate test inputs with controlled random range (0.05)
        context_hidden_states = (torch.randn((batch_size, context_length, hidden_size)) * 0.05).to(DTYPE)
        decode_hidden_states = (torch.randn((batch_size, decode_length, hidden_size)) * 0.05).to(DTYPE)

        # Context phase masks and positions
        context_attention_mask_2d = torch.tensor([[1]*context_length])
        context_attention_mask = create_context_attn_mask(context_attention_mask_2d)
        context_position_ids = torch.arange(0, context_length, dtype=torch.int32).unsqueeze(0)

        # Create random checkpoint before running tests
        ckpt_path = os.path.join(ARTIFACTS_FOLDER, "checkpoint.pt")
        create_random_checkpoint(ckpt_path, self.model_config, dtype=DTYPE, rand_range=0.05)

        #############################################
        # Context/Prefill Phase
        #############################################
        logger.info("Testing context/prefill phase...")

        # Create a DynamicCache for CPU model
        cpu_cache = DynamicCache()

        # Load CPU model
        cpu_model = load_cpu_model(
            config=self.model_config,
            checkpoint_path=ckpt_path
        )

        # Run CPU inference for context phase
        logger.info("Running inference on CPU model - context phase")
        with torch.no_grad():
            cpu_output = cpu_model(
                hidden_states=context_hidden_states,
                attention_mask=build_4d_causal_mask(context_attention_mask_2d),
                position_ids=context_position_ids,
                past_key_value=cpu_cache
            )

        # Create example inputs tuple for Neuron model - context phase
        context_example_inputs = [(
            context_hidden_states,
            context_attention_mask,
            context_position_ids
        )]

        # Load Neuron model for context phase
        context_neuron_model = load_neuron_model(
            model_config=self.neuron_model_config,
            neuron_config=self.neuron_config,
            checkpoint_path=ckpt_path,
            example_inputs=context_example_inputs
        )

        # Run Neuron inference for context phase
        logger.info("Running inference on Neuron model - context phase")
        neuron_context_output = context_neuron_model(
            context_hidden_states,
            context_attention_mask,
            context_position_ids
        )

        # Check context phase results
        self.check_results("context_phase", neuron_context_output[0], cpu_output, rtol=1e-5)
        self.check_results("k cache", neuron_context_output[1], cpu_cache[0][0], rtol=2e-5)
        self.check_results("v cache", neuron_context_output[2], cpu_cache[0][1], rtol=2e-5)

        #############################################
        # Decode Phase
        #############################################
        logger.info("Testing decode phase...")

        # Decode phase positions and masks
        decode_position_ids = torch.tensor([[context_length]], dtype=torch.int32)
        decode_attention_mask = build_4d_causal_mask(torch.tensor([[1]]))
        decode_attention_mask_neuron = create_context_attn_mask(torch.tensor([[1]]))

        # Run CPU inference for decode phase
        logger.info("Running inference on CPU model - decode phase")
        with torch.no_grad():
            cpu_decode_output = cpu_model(
                hidden_states=decode_hidden_states,
                attention_mask=decode_attention_mask,
                position_ids=decode_position_ids,
                past_key_value=cpu_cache
            )

        # Create example inputs tuple for Neuron model - decode phase
        decode_example_inputs = [(
            decode_hidden_states,
            decode_attention_mask_neuron,
            decode_position_ids,
            neuron_context_output[1],
            neuron_context_output[2],
        )]

        # Load Neuron model for decode phase
        decode_neuron_model = load_neuron_model(
            model_config=self.neuron_model_config,
            neuron_config=self.neuron_config,
            checkpoint_path=ckpt_path,
            example_inputs=decode_example_inputs
        )

        # Run Neuron inference for decode phase
        logger.info("Running inference on Neuron model - decode phase")
        neuron_decode_output = decode_neuron_model(
            decode_hidden_states,
            decode_attention_mask_neuron,
            decode_position_ids,
            neuron_context_output[1],
            neuron_context_output[2],
        )

        # Check decode phase results
        self.check_results("decode_phase", neuron_decode_output[0], cpu_decode_output, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
