from argparse import Namespace
import copy
import os
import pytest
import tempfile
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, Gemma3Config

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import (
    Gemma3InferenceConfig,
    NeuronGemma3ForCausalLM,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from torch_neuronx.testing.validation import (
    DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
    DEFAULT_TOLERANCE_MAP,
)
from neuronx_distributed_inference.utils.random import set_random_seed

# Gemma3 is SWA for the first 5 layers and swaps to standard attention for layer 6
# This is also a different local/global RoPE

NEURON_CONFIG_TP2_FLAT = NeuronConfig(
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
    mlp_kernel_enabled=False,
    is_continuous_batching=False,
    fused_qkv=False,
    torch_dtype=torch.float32,
)

# At high BS theres certain tokens with larger divergence and tolerance errors, most tokens are accurate.
# The kernels also do not support fp32 and have precision differences.
KERNELS_TOL_MAP = DEFAULT_TOLERANCE_MAP.copy()
KERNELS_TOL_MAP[5] = (1e-5, 0.04)

LARGER_LAYER_TOLERANCE_MAP = DEFAULT_TOLERANCE_MAP.copy()

for key in LARGER_LAYER_TOLERANCE_MAP.keys():
    LARGER_LAYER_TOLERANCE_MAP[key] = (
        LARGER_LAYER_TOLERANCE_MAP[key][0],
        LARGER_LAYER_TOLERANCE_MAP[key][1] + 0.04,
    )


@pytest.mark.tp64
@pytest.mark.context_parallel
@pytest.mark.data_parallel
@pytest.mark.parametrize(
    "neuron_config, latency_threshold, throughput_threshold, divergence_tolerance, tolerance_map, number_of_layers",
    [
        (NEURON_CONFIG_TP2_FLAT, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, DEFAULT_TOLERANCE_MAP, 4),
        (NEURON_CONFIG_TP2_FLAT, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, LARGER_LAYER_TOLERANCE_MAP, 6)
    ],
)
def test_qemma3_x_layers(
    neuron_config, latency_threshold, throughput_threshold, divergence_tolerance, tolerance_map, number_of_layers
):
    # For reproducibility
    set_random_seed(42)

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path, number_of_layers)
    model_path = model_tempdir.name

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)

    config = Gemma3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    if config.neuron_config.on_device_sampling_config is None:
        validate_accuracy(
            model_path, config, generation_config, divergence_tolerance, tolerance_map
        )

    if throughput_threshold > 0:
        validate_performance(
            model_path, config, generation_config, latency_threshold, throughput_threshold
        )


def save_checkpoint(config_path, number_of_layers):
    hf_config = Gemma3Config.from_pretrained(config_path).get_text_config()
    hf_config.num_hidden_layers = number_of_layers  # number_of_layers layer test
    hf_config.layer_types = hf_config.layer_types[:number_of_layers]
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=hf_config.dtype)
    hf_model.eval()  # Set HuggingFace CPU golden model to evaluation mode

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, divergence_tolerance, tolerance_map):
    input_len = 16
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronGemma3ForCausalLM(model_path, config)
    model.eval()  # Set to evaluation mode for inference
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        num_tokens_to_check=20 - input_len,
        inputs=inputs,
        divergence_difference_tol=divergence_tolerance,
        tol_map=tolerance_map,
    )


# TODO: TAKE A LOOK AT THIS BELOW AND FIX IT
def validate_performance(
    model_path, config, generation_config, latency_threshold, throughput_threshold
):
    config = copy.deepcopy(config)
    config.neuron_config.on_device_sampling_config = OnDeviceSamplingConfig()

    model = NeuronGemma3ForCausalLM(model_path, config)
    model.eval()  # Set to evaluation mode for inference
    compiled_model_path = model_path + "/compiled_checkpoint_perf"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    benchmark_results = benchmark_sampling(model, generation_config=generation_config)
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert (
        latency < latency_threshold
    ), f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert (
        throughput > throughput_threshold
    ), f"throughput ({throughput}) is below threshold ({throughput_threshold})"


if __name__ == "__main__":
    # For ease of running the testing
    test_qemma3_x_layers(NEURON_CONFIG_TP2_FLAT, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, DEFAULT_TOLERANCE_MAP, 4)
    test_qemma3_x_layers(NEURON_CONFIG_TP2_FLAT, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, LARGER_LAYER_TOLERANCE_MAP, 6)
