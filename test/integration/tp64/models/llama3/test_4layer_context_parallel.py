from argparse import Namespace
import copy
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from torch_neuronx.testing.validation import DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

SP_NEURON_CONFIG_CP16_TP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    torch_dtype=torch.float32,
)

SP_NEURON_CONFIG_CP4_TP16 = copy.deepcopy(SP_NEURON_CONFIG_CP16_TP4)
SP_NEURON_CONFIG_CP4_TP16.cp_degree = 4

SP_DISABLED_NEURON_CONFIG_CP16_TP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=False,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    torch_dtype=torch.float32,
)

SP_DISABLED_NEURON_CONFIG_CP4_TP16 = copy.deepcopy(SP_DISABLED_NEURON_CONFIG_CP16_TP4)
SP_DISABLED_NEURON_CONFIG_CP4_TP16.cp_degree = 4

PERF_CONFIG_CP16_TP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    on_device_sampling_config=OnDeviceSamplingConfig(),
    attn_kernel_enabled=False,
    torch_dtype=torch.bfloat16,
)

PERF_CONFIG_CP4_TP16 = copy.deepcopy(PERF_CONFIG_CP16_TP4)
PERF_CONFIG_CP4_TP16.cp_degree = 4

KERNEL_CONFIG_CP16_TP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    batch_size=1,
    max_context_length=4096,
    seq_len=4096,
    sequence_parallel_enabled=False,
    logical_nc_config=2,
    torch_dtype=torch.float32,
)

KERNEL_CONFIG_CP4_TP16 = copy.deepcopy(KERNEL_CONFIG_CP16_TP4)
KERNEL_CONFIG_CP4_TP16.cp_degree = 4

@pytest.fixture(scope="module", autouse=True)
def model_path():
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    yield model_path

    model_tempdir.cleanup()

# TODO: The divergence is a higher than expected error for fp32 in the kernel case, investigate why the kernel causes this increase.
@pytest.mark.tp64
@pytest.mark.context_parallel
@pytest.mark.parametrize(
    "neuron_config, latency_threshold, throughput_threshold, check_performance, divergence_tolerance",
    # fmt: off
    [
        pytest.param(SP_NEURON_CONFIG_CP16_TP4, float('inf'), 0, False, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=[pytest.mark.key_config_test]),
        pytest.param(SP_DISABLED_NEURON_CONFIG_CP16_TP4, float('inf'), 0, False, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        pytest.param(PERF_CONFIG_CP16_TP4, 410.25, 645.02, True, None),
        pytest.param(KERNEL_CONFIG_CP16_TP4, float('inf'), 0, True, 0.015),
        pytest.param(SP_NEURON_CONFIG_CP4_TP16, float('inf'), 0, False, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        pytest.param(SP_DISABLED_NEURON_CONFIG_CP4_TP16, float('inf'), 0, False, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        pytest.param(PERF_CONFIG_CP4_TP16, 311.78, 803.67, True, None),
        pytest.param(KERNEL_CONFIG_CP4_TP16, float('inf'), 0, True, 0.01),
    ],
    # fmt: on
)
def test_llama_4layer_context_parallel(model_path, neuron_config, latency_threshold, throughput_threshold, check_performance, divergence_tolerance):
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    if config.neuron_config.on_device_sampling_config is None:
        validate_accuracy(model_path, config, generation_config, divergence_tolerance)

    if check_performance:
        validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold)

def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)

    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

def validate_accuracy(model_path, config, generation_config, divergence_tolerance):
    input_len = 16
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        num_tokens_to_check=256 - input_len,
        inputs=inputs,
        divergence_difference_tol=divergence_tolerance,
    )

def validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold):
    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_perf"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    benchmark_results = benchmark_sampling(model, generation_config=generation_config)
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"
