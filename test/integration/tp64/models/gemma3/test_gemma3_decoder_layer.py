import logging
import os
import pytest
import tempfile
from functools import partial
import copy

import torch
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Config,
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
)
from transformers.cache_utils import DynamicCache

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import (
    NeuronGemma3ForCausalLM,
    Gemma3InferenceConfig,
    NeuronGemma3DecoderLayer,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module
from neuronx_distributed_inference.modules.attention.utils import (
    stride_tensor,
    order_strided_tensor,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CPUGemma3DecoderLayerModule(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer = Gemma3DecoderLayer(config=config, layer_idx=layer_idx)
        # Transformers 4.56.* style
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

    def forward(
        self, hidden_states, attention_mask, position_ids, past_key_value=None, local_mask=None
    ):
        # Compute position embeddings
        cos, sin = self.rotary_emb_local(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Run decoder layer and return all outputs
        outputs = self.layer(
            hidden_states,
            position_embeddings_global=None,
            position_embeddings_local=position_embeddings,
            attention_mask=local_mask,
            past_key_value=past_key_value,
            use_cache=True,
        )

        return outputs[0]


class NeuronGemma3DecoderLayerModule(nn.Module):
    """Neuron implementation of Gemma3 Decoder Layer module for testing"""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer = NeuronGemma3DecoderLayer(config=config, layer_idx=layer_idx)

    def forward(self, hidden_states, attention_mask, position_ids, local_mask=None):
        # Run decoder layer and return all outputs
        outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            local_mask=local_mask,
            position_ids=position_ids,
            past_key_value=None,
        )
        return (
            outputs[0],
            outputs[1][0] if outputs[1] else None,
            outputs[1][1] if outputs[1] else None,
        )


def build_4d_causal_mask(
    attn_2d: torch.Tensor, *, dtype: torch.dtype | None = None, sliding_window=None
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    bsz, seq_len = attn_2d.shape
    device = attn_2d.device
    minus_inf = torch.finfo(dtype).min

    # ① causal pattern (lower-triangular ones) – shape (1,1,S,S)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))[None, None]

    # ② Apply sliding window if specified
    if sliding_window is not None and sliding_window > 0:
        # Create sliding window mask - only allow attention within the window
        sliding_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        for i in range(seq_len):
            start_pos = max(0, i - sliding_window + 1)
            sliding_mask[i, start_pos : i + 1] = 1.0
        sliding_mask = sliding_mask[None, None]
        causal = causal * sliding_mask

    # ③ broadcast padding mask to (B,1,1,S)
    pad = attn_2d.to(dtype)[:, None, None, :]

    # ④ allow only positions that are both causal AND non-padding
    keep = causal * pad  # (B,1,S,S)

    # ⑤ convert to additive form expected by attention kernels
    mask_4d = torch.where(keep.bool(), torch.zeros_like(keep), torch.full_like(keep, minus_inf))
    return mask_4d


def create_context_attn_mask(attention_mask, sliding_window=None):
    # Lower triangle causal mask for attention
    batch_size, n_positions = attention_mask.shape
    mask = torch.full((n_positions, n_positions), True).tril(diagonal=0)

    # Apply sliding window if specified
    if sliding_window is not None and sliding_window > 0:
        for i in range(n_positions):
            start_pos = max(0, i - sliding_window + 1)
            mask[i, :start_pos] = False

    mask = mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)

    expanded_mask = (
        attention_mask[:, None, None, :]
        .expand(batch_size, 1, n_positions, n_positions)
        .to(torch.bool)
    )
    return torch.logical_and(mask, expanded_mask)


def create_random_checkpoint(path, config, dtype, layer_idx=0, rand_range=0.05):
    """
    Create a random checkpoint for Gemma3 Decoder Layer module.

    Args:
        path: Path to save the checkpoint
        config: Model configuration
        dtype: Data type for tensors
        layer_idx: Layer index for the decoder layer
        rand_range: Range for random values (default: 0.05)

    Returns:
        Dictionary containing random state dict
    """
    if os.path.exists(path):
        os.remove(path)

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    random_sd = {}

    # Create random weights for attention components
    # Q, K, V projection weights
    random_sd["layer.self_attn.q_proj.weight"] = (
        torch.randn(num_attention_heads * head_dim, hidden_size) * rand_range
    ).to(dtype)
    random_sd["layer.self_attn.k_proj.weight"] = (
        torch.randn(num_key_value_heads * head_dim, hidden_size) * rand_range
    ).to(dtype)
    random_sd["layer.self_attn.v_proj.weight"] = (
        torch.randn(num_key_value_heads * head_dim, hidden_size) * rand_range
    ).to(dtype)

    # Output projection weight
    random_sd["layer.self_attn.o_proj.weight"] = (
        torch.randn(hidden_size, num_attention_heads * head_dim) * rand_range
    ).to(dtype)

    # Q and K norm weights (Gemma3 specific)
    random_sd["layer.self_attn.q_norm.weight"] = (torch.randn(head_dim) * rand_range).to(dtype)
    random_sd["layer.self_attn.k_norm.weight"] = (torch.randn(head_dim) * rand_range).to(dtype)

    # Create random weights for MLP components
    # Gate projection weights
    random_sd["layer.mlp.gate_proj.weight"] = (
        torch.randn(intermediate_size, hidden_size) * rand_range
    ).to(dtype)
    # Up projection weights
    random_sd["layer.mlp.up_proj.weight"] = (
        torch.randn(intermediate_size, hidden_size) * rand_range
    ).to(dtype)
    # Down projection weights
    random_sd["layer.mlp.down_proj.weight"] = (
        torch.randn(hidden_size, intermediate_size) * rand_range
    ).to(dtype)

    # Create random weights for RMSNorm layers
    random_sd["layer.input_layernorm.weight"] = (torch.randn(hidden_size) * rand_range).to(dtype)
    random_sd["layer.post_attention_layernorm.weight"] = (torch.randn(hidden_size) * rand_range).to(
        dtype
    )
    random_sd["layer.pre_feedforward_layernorm.weight"] = (
        torch.randn(hidden_size) * rand_range
    ).to(dtype)
    random_sd["layer.post_feedforward_layernorm.weight"] = (
        torch.randn(hidden_size) * rand_range
    ).to(dtype)

    torch.save(random_sd, path)

    logger.info(f"Created random checkpoints at {path}")
    return random_sd


def checkpoint_loader_fn(neuron_model_config, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")

    # Rename q_norm and k_norm to q_layernorm and k_layernorm
    if "layer.self_attn.q_norm.weight" in sd:
        sd["layer.self_attn.q_layernorm.weight"] = sd.pop("layer.self_attn.q_norm.weight")
    if "layer.self_attn.k_norm.weight" in sd:
        sd["layer.self_attn.k_layernorm.weight"] = sd.pop("layer.self_attn.k_norm.weight")

    if neuron_model_config.neuron_config.fused_qkv:
        sd["layer.self_attn.Wqkv.weight"] = torch.cat(
            [
                sd.pop("layer.self_attn.q_proj.weight"),
                sd.pop("layer.self_attn.k_proj.weight"),
                sd.pop("layer.self_attn.v_proj.weight"),
            ]
        )

    # To facilitate rank usage in attention
    sd["layer.self_attn.rank_util.rank"] = torch.arange(
        0, neuron_model_config.neuron_config.tp_degree, dtype=torch.int32
    )

    return sd


def load_cpu_model(config, checkpoint_path, layer_idx=0):
    """Load and return the CPU model with checkpoint loaded"""
    # Set random seed for reproducibility
    set_random_seed(0)

    # Create model directly
    model = CPUGemma3DecoderLayerModule(config, layer_idx=layer_idx).to(config.torch_dtype).eval()

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model


def load_neuron_model(
    neuron_model_config, checkpoint_path, example_inputs, artifacts_path, layer_idx=0
):
    """Load and return the Neuron model with checkpoint loaded"""
    os.environ["NXD_CPU_MODE"] = "0"
    set_random_seed(0)

    # Create a wrapper class that sets the model to eval mode
    class EvalNeuronGemma3DecoderLayerModule(NeuronGemma3DecoderLayerModule):
        def __init__(self, config, layer_idx=0):
            super().__init__(config, layer_idx)
            self.eval()  # Set to evaluation mode for inference

    # Build and trace the model using utility function
    neuron_model = build_module(
        module_cls=EvalNeuronGemma3DecoderLayerModule,
        example_inputs=example_inputs,
        module_init_kwargs={"config": neuron_model_config, "layer_idx": layer_idx},
        compiler_args=NeuronGemma3ForCausalLM(
            checkpoint_path,
            neuron_model_config,
        ).get_compiler_args(),
        compiler_workdir=os.path.join(
            artifacts_path, f"compiler_workdir_decoder_layer_{layer_idx}"
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_loader_fn=partial(checkpoint_loader_fn, neuron_model_config),
        logical_nc_config=neuron_model_config.neuron_config.logical_nc_config,
    )

    return neuron_model


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


def check_results(test_name, actual_output, expected_output, rtol=1e-5):
    line_break = "-" * 150
    print(line_break)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    passed, _ = check_accuracy_embeddings(
        actual_output, expected_output, plot_outputs=False, rtol=rtol, atol=0
    )
    assert passed, f"Logits failed accuracy validation \n {line_break}"
    print("Logits passed accuracy validation")
    print(line_break)


@pytest.mark.parametrize(
    "neuron_config, layer_idx, rtol",
    [
        (BASELINE_NEURON_CONFIG, 1, 2e-2),  # Sliding window layer (layer 0)
        # (BASELINE_NEURON_CONFIG, 5, 2e-2),  # Global attention layer (layer 5, (5+1)%6 == 0)
    ],
)
def test_gemma3_decoder_layer(neuron_config, layer_idx, rtol):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(42)

    logger.info(f"Running Gemma3 Decoder Layer test for layer {layer_idx}...")
    dtype = neuron_config.torch_dtype

    # Create configs
    # Disable SP in module-level testing
    neuron_config.sequence_parallel_enabled = False
    neuron_config.is_prefill_stage = True

    config_path = os.path.join(CURR_DIR, "config.json")

    # Create Gemma3Config with text config
    model_config = Gemma3Config.from_pretrained(config_path, torch_dtype=dtype).get_text_config()
    model_config._attn_implementation = "eager"  # Necessary when initialized config without model

    # Determine if this layer uses sliding window attention
    is_sliding_window_layer = (layer_idx + 1) % 6 != 0
    sliding_window = (
        getattr(model_config, "sliding_window", None) if is_sliding_window_layer else None
    )

    neuron_model_config = Gemma3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=model_config)
    )

    # Override sliding window for this specific layer test
    neuron_model_config.sliding_window = sliding_window

    logger.info(
        f"Testing context/prefill phase for layer {layer_idx} (sliding_window={sliding_window})..."
    )

    # Create test inputs
    batch_size = neuron_config.ctx_batch_size
    seq_len = neuron_config.max_context_length
    hidden_size = neuron_model_config.hidden_size

    # Generate test inputs with controlled random range (0.05)
    hidden_states = (torch.randn((batch_size, seq_len, hidden_size)) * 0.05).to(dtype)

    # Context phase masks and positions
    attention_mask_2d = torch.tensor([[1] * seq_len])
    attention_mask = create_context_attn_mask(attention_mask_2d, sliding_window)
    position_ids = torch.arange(0, seq_len, dtype=torch.int32).unsqueeze(0)

    # Build local mask for sliding window attention if needed
    local_mask = None
    if sliding_window is not None:
        local_mask = create_context_attn_mask(attention_mask_2d, sliding_window)

    artifacts_tempdir = tempfile.TemporaryDirectory()
    artifacts_path = artifacts_tempdir.name

    # Create random checkpoint before running tests
    checkpoint_path = os.path.join(
        artifacts_path, f"checkpoint_decoder_layer_{layer_idx}_{str(dtype).split('.')[1]}.pt"
    )
    create_random_checkpoint(
        checkpoint_path, neuron_model_config, dtype=dtype, layer_idx=layer_idx, rand_range=0.05
    )

    # Load CPU model
    cpu_model = load_cpu_model(
        config=model_config, checkpoint_path=checkpoint_path, layer_idx=layer_idx
    )

    # Create a DynamicCache for CPU model
    cpu_cache = DynamicCache()

    # Run CPU inference
    logger.info("Running inference on CPU model")
    with torch.no_grad():
        cpu_output = cpu_model(
            hidden_states=hidden_states,
            attention_mask=build_4d_causal_mask(attention_mask_2d, sliding_window=sliding_window),
            local_mask=(
                build_4d_causal_mask(attention_mask_2d, sliding_window=sliding_window)
                if local_mask is not None
                else None
            ),
            position_ids=position_ids,
            past_key_value=cpu_cache,
        )

    # Create example inputs tuple for Neuron model
    example_inputs = [
        (
            torch.ones_like(hidden_states),
            torch.ones_like(attention_mask),
            torch.ones_like(position_ids),
            torch.ones_like(local_mask) if local_mask is not None else None,
        )
    ]

    # Load Neuron model
    neuron_model = load_neuron_model(
        neuron_model_config=neuron_model_config,
        checkpoint_path=checkpoint_path,
        example_inputs=example_inputs,
        artifacts_path=artifacts_path,
        layer_idx=layer_idx,
    )

    # Run Neuron inference
    logger.info("Running inference on Neuron model")

    # Stage 2 strided CP FA kernel expected inputs to be strided. This logic is inside NeuronBaseModel.forward()
    if neuron_config.strided_context_parallel_kernel_enabled and neuron_config.is_prefill_stage:
        logging.info("strided_context_parallel_kernel_enabled enabled, shuffling inputs")

        # The strided CP FA kernel expected inputs to be strided, due to SP happening in model_base
        hidden_states = stride_tensor(hidden_states, 1, neuron_config.cp_degree)
        position_ids = stride_tensor(position_ids, 1, neuron_config.cp_degree)

    neuron_output = neuron_model(hidden_states, attention_mask, position_ids, local_mask)

    # Check results
    neuron_output_hidden_states = neuron_output[0]
    if neuron_config.strided_context_parallel_kernel_enabled and neuron_config.is_prefill_stage:
        logging.info(
            "strided_context_parallel_kernel_enabled enabled, reorder output hidden_states"
        )
        neuron_output_hidden_states = order_strided_tensor(
            neuron_output_hidden_states, 1, neuron_config.cp_degree
        )

    check_results(
        f"decoder_layer_{layer_idx}_{str(dtype).split('.')[1]}",
        neuron_output_hidden_states,
        cpu_output,
        rtol=rtol,
    )

    # Clean up
    artifacts_tempdir.cleanup()

    logger.info(f"Gemma3 Decoder Layer test for layer {layer_idx} completed successfully!")


if __name__ == "__main__":
    # Test sliding window layer (layer 0)
    test_gemma3_decoder_layer(BASELINE_NEURON_CONFIG, 1, 2e-2)

    print("Gemma3 decoder layer test completed!")
