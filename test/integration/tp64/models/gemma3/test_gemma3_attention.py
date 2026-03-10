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
    Gemma3Attention,
    Gemma3RotaryEmbedding,
)
from transformers.cache_utils import DynamicCache

from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import (
    Gemma3InferenceConfig,
    NeuronGemma3Attention,
    NeuronGemma3ForCausalLM,
)
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, build_cpu_model
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.modules.attention.utils import (
    stride_tensor,
    order_strided_tensor,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Reading neuron_config test cases
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


class Gemma3AttentionModule(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.attention = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.is_sliding_window = ((layer_idx + 1) % 6) != 0
        # Transformers 4.56.* style
        self.rotary_emb_global = Gemma3RotaryEmbedding(config=config)
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value=None):
        # Compute position embeddings
        if self.is_sliding_window:
            cos, sin = self.rotary_emb_local(hidden_states, position_ids)
        else:
            cos, sin = self.rotary_emb_global(hidden_states, position_ids)
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


class NeuronGemma3AttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NeuronGemma3Attention(config=config)

    def forward(self, hidden_states, attention_mask, position_ids, past_k=None, past_v=None):
        # Run attention and return all outputs
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(past_k, past_v) if past_k is not None and past_v is not None else None,
        )
        return (outputs[0], outputs[1][0], outputs[1][1])


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


def checkpoint_loader_fn(neuron_model_config, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")

    # Rename q_norm and k_norm to q_layernorm and k_layernorm
    if "attention.q_norm.weight" in sd:
        sd["attention.q_layernorm.weight"] = sd.pop("attention.q_norm.weight")
    if "attention.k_norm.weight" in sd:
        sd["attention.k_layernorm.weight"] = sd.pop("attention.k_norm.weight")

    if neuron_model_config.neuron_config.fused_qkv:
        sd["attention.Wqkv.weight"] = torch.cat(
            [
                sd.pop("attention.q_proj.weight"),
                sd.pop("attention.k_proj.weight"),
                sd.pop("attention.v_proj.weight"),
            ]
        )

    # To facilitate rank usage in attention
    sd["attention.rank_util.rank"] = torch.arange(
        0, neuron_model_config.neuron_config.tp_degree, dtype=torch.int32
    )

    return sd


def check_results(test_name, actual_output, expected_output, plot_outputs=False, rtol=1e-5):
    line_break = "-" * 150
    print(line_break)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    passed, _ = check_accuracy_embeddings(
        actual_output, expected_output, plot_outputs=plot_outputs, rtol=rtol, atol=1e-5
    )
    assert passed, f"Logits failed accuracy validation \n {line_break}"
    print("Logits passed accuracy validation")
    print(line_break)


@pytest.mark.parametrize(
    "neuron_config, layer_idx, rtol",
    [
        (BASELINE_NEURON_CONFIG, 0, 2e-3),  # Sliding window layer (layer 0)
        # (BASELINE_NEURON_CONFIG, 5, 2e-3),  # Global attention layer (layer 5, (5+1)%6 == 0)
    ],
)
def test_gemma3_attention(neuron_config, layer_idx, rtol):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(42)

    logger.info(f"Running Gemma3 Attention test for layer {layer_idx}...")

    # Create configs
    # Disable SP in module-level testing
    # Because when both CP and SP are both enabled, we skip two collectives before and after attention
    # This causes a shape mismatch if only running attention module
    neuron_config.sequence_parallel_enabled = False
    neuron_config.is_prefill_stage = True

    config_path = os.path.join(CURR_DIR, "config.json")

    # Create Gemma3Config with text config
    model_config = Gemma3Config.from_pretrained(config_path).get_text_config()
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

    #############################################
    # Context/Prefill Phase
    #############################################
    logger.info(
        f"Testing context/prefill phase for layer {layer_idx} (sliding_window={sliding_window})..."
    )

    # Create context phase test inputs
    # Generate test inputs with controlled random range (0.05)
    context_hidden_states = (
        torch.randn(
            (
                neuron_config.ctx_batch_size,
                neuron_config.max_context_length,
                neuron_model_config.hidden_size,
            )
        )
        * 0.05
    ).to(neuron_config.torch_dtype)

    # Context phase masks and positions
    context_attention_mask_2d = torch.tensor([[1] * neuron_config.max_context_length])
    context_attention_mask = create_context_attn_mask(context_attention_mask_2d, sliding_window)
    context_position_ids = torch.arange(
        0, neuron_config.max_context_length, dtype=torch.int32
    ).unsqueeze(0)

    # Create a DynamicCache for CPU model
    cpu_cache = DynamicCache()

    # Build CPU model and save random checkpoint
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name

    def init_cpu_model_with_layer_idx(config):
        return Gemma3AttentionModule(config, layer_idx=layer_idx)

    cpu_model, ckpt_path = build_cpu_model(
        init_cpu_model_with_layer_idx,
        model_config,
        dtype=neuron_config.torch_dtype,
        checkpoint_dir=model_path,
    )

    # Run CPU inference for context phase
    logger.info("Running inference on CPU model - context phase")
    with torch.no_grad():
        cpu_output = cpu_model(
            hidden_states=context_hidden_states,
            attention_mask=build_4d_causal_mask(
                context_attention_mask_2d, sliding_window=sliding_window
            ),
            position_ids=context_position_ids,
            past_key_value=cpu_cache,
        )

    # Create example inputs tuple for Neuron model - context phase
    context_example_inputs = [
        (
            torch.ones_like(context_hidden_states),
            torch.ones_like(context_attention_mask),
            torch.ones_like(context_position_ids),
        )
    ]

    # Build and trace Neuron model for context phase
    context_neuron_model = build_module(
        module_cls=NeuronGemma3AttentionModule,
        example_inputs=context_example_inputs,
        module_init_kwargs={"config": neuron_model_config},
        tp_degree=neuron_config.tp_degree,
        compiler_args=NeuronGemma3ForCausalLM(
            ckpt_path,
            neuron_model_config,
        ).get_compiler_args(),
        checkpoint_path=ckpt_path,
        checkpoint_loader_fn=partial(checkpoint_loader_fn, neuron_model_config),
        logical_nc_config=neuron_config.logical_nc_config,
    )

    # Run Neuron inference for context phase
    logger.info("Running inference on Neuron model - context phase")

    # Stage 2 strided CP FA kernel expected inputs to be strided. This logic is inside NeuronBaseModel.forward()
    if neuron_config.strided_context_parallel_kernel_enabled and neuron_config.is_prefill_stage:
        logging.info("strided_context_parallel_kernel_enabled enabled, shuffling inputs")

        # The strided CP FA kernel expected inputs to be strided, due to SP happening in model_base
        # stride here rather than in attention to order it before we move the inputs to SP region
        context_hidden_states = stride_tensor(context_hidden_states, 1, neuron_config.cp_degree)
        context_position_ids = stride_tensor(context_position_ids, 1, neuron_config.cp_degree)

    neuron_context_output = context_neuron_model(
        context_hidden_states, context_attention_mask, context_position_ids
    )

    # Check context phase results
    neuron_context_output_hidden_states = neuron_context_output[0]
    if neuron_config.strided_context_parallel_kernel_enabled and neuron_config.is_prefill_stage:
        logging.info(
            "strided_context_parallel_kernel_enabled enabled, reorder output hidden_states"
        )
        neuron_context_output_hidden_states = order_strided_tensor(
            neuron_context_output_hidden_states, 1, neuron_config.cp_degree
        )

    check_results(
        f"context_phase_layer_{layer_idx}",
        neuron_context_output_hidden_states,
        cpu_output,
        plot_outputs=True,
        rtol=rtol,
    )

    # Check v cache (k cache comparison is skipped as noted in original test)
    # TODO: Fix since we are currently getting a shape mismatch
    # check_results(f"v_cache_layer_{layer_idx}", neuron_context_output[2], cpu_cache[0][1][:, [0], :, :], plot_outputs=False, rtol=rtol)

    logger.info(f"Gemma3 Attention test for layer {layer_idx} completed successfully!")


if __name__ == "__main__":
    # Test sliding window layer (layer 0)
    test_gemma3_attention(BASELINE_NEURON_CONFIG, 0, 2e-3)

    # Test global attention layer (layer 5, where (5+1) % 6 == 0)
    # test_gemma3_attention(BASELINE_NEURON_CONFIG, 5, 2e-3)

    print("All Gemma3 attention tests completed!")
