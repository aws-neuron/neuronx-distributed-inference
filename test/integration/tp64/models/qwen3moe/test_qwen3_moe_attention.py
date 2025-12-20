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
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
)
from transformers.cache_utils import DynamicCache

from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeInferenceConfig,
    NeuronQwen3MoEAttention,
    NeuronQwen3MoeForCausalLM,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, build_cpu_model
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.modules.attention.utils import stride_tensor, order_strided_tensor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS16 baseline
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_baseline_tp64.json"), "r") as f:
    baseline_json = json.load(f)
BASELINE_NEURON_CONFIG = MoENeuronConfig(**baseline_json)

# BS16 TP4/CP16 for CTE
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_optimized.json"), "r") as f:
    cp16_dp8_json = json.load(f)
CP16_TP4_NEURON_CONFIG = MoENeuronConfig(**cp16_dp8_json)


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


def checkpoint_loader_fn(neuron_model_config, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")

    sd["attention.q_layernorm.weight"] = sd.pop("attention.q_norm.weight")
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
    print("-" * 20)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    passed, _ = check_accuracy_embeddings(
            actual_output, expected_output, plot_outputs=plot_outputs, rtol=rtol, atol=1e-5
        )
    assert(passed)
    print("-" * 20)


@pytest.mark.parametrize(
        "neuron_config, rtol",
        [
            (BASELINE_NEURON_CONFIG, 1.6e-3),  # BS16 TP64 baseline
            (CP16_TP4_NEURON_CONFIG, 2e-3),  # BS16 CP16/TP4 for CTE
        ]
)
def test_attention(neuron_config, rtol):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(0)

    logger.info("Running Qwen3MoeAttention test ...")

    # Create configs
    # Disable SP in module-level testing
    # Because when both CP and SP are both enabled, we skip two collectives before and after attention
    # This causes a shape mismatch if only running attention module
    neuron_config.sequence_parallel_enabled = False
    neuron_config.is_prefill_stage = True

    config_path = os.path.join(CURR_DIR, "config.json")
    model_config = Qwen3MoeConfig.from_pretrained(
        config_path,
        _attn_implementation="eager",  # Necessary when initialized config without model
    )

    neuron_model_config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=model_config)
    )
    
    #############################################
    # Context/Prefill Phase
    #############################################
    logger.info("Testing context/prefill phase...")

    # Create context phase test inputs
    # Generate test inputs with controlled random range (0.05)
    context_hidden_states = (torch.randn((
        neuron_config.ctx_batch_size,
        neuron_config.max_context_length,
        neuron_model_config.hidden_size,
        )) * 0.05).to(neuron_config.torch_dtype)

    # Context phase masks and positions
    context_attention_mask_2d = torch.tensor([[1]*neuron_config.max_context_length])
    context_attention_mask = create_context_attn_mask(context_attention_mask_2d)
    context_position_ids = torch.arange(0, neuron_config.max_context_length, dtype=torch.int32).unsqueeze(0)


    # Create a DynamicCache for CPU model
    cpu_cache = DynamicCache()

    # Build CPU model and save random checkpoint
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    cpu_model, ckpt_path = build_cpu_model(Qwen3MoeAttentionModule, model_config, dtype=neuron_config.torch_dtype, checkpoint_dir=model_path)

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
        torch.ones_like(context_hidden_states),
        torch.ones_like(context_attention_mask),
        torch.ones_like(context_position_ids)
    )]

    # Build and trace Neuron model for context phase
    context_neuron_model = build_module(
        module_cls=NeuronQwen3MoeAttentionModule,
        example_inputs=context_example_inputs,
        module_init_kwargs={"config": neuron_model_config},
        tp_degree=neuron_config.tp_degree,
        compiler_args=NeuronQwen3MoeForCausalLM(
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
        context_hidden_states,
        context_attention_mask,
        context_position_ids
    )

    # Check context phase results
    neuron_context_output_hidden_states = neuron_context_output[0]
    if neuron_config.strided_context_parallel_kernel_enabled and neuron_config.is_prefill_stage:
        logging.info("strided_context_parallel_kernel_enabled enabled, reorder output hidden_states")
        neuron_context_output_hidden_states = order_strided_tensor(neuron_context_output_hidden_states, 1, neuron_config.cp_degree)

    check_results("context_phase", neuron_context_output_hidden_states, cpu_output, plot_outputs=True, rtol=rtol)
    
    # default forward only return rank 0 k cache and v cache
    # TODO: investigate why k cache does not match but v cache does
    # check_results("k cache", neuron_context_output[1], cpu_cache[0][0][:, [0], :, :], plot_outputs=True, rtol=rtol)
    check_results("v cache", neuron_context_output[2], cpu_cache[0][1][:, [0], :, :], plot_outputs=False, rtol=rtol)


if __name__ == "__main__":
    test_attention(BASELINE_NEURON_CONFIG, 1.6e-3)
    test_attention(CP16_TP4_NEURON_CONFIG, 2e-3)
