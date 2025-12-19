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
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

ATTN_CHUNK_SIZE = 2048

FA_KERNEL_CONFIG = NeuronConfig(
    tp_degree=64,
    cp_degree=1,
    batch_size=1,
    max_context_length=4096,
    seq_len=4096,
    sequence_parallel_enabled=True,
    fused_qkv=True,
    attn_kernel_enabled=True,
    logical_nc_config=2,
    torch_dtype=torch.float32,
    is_continuous_batching=True,
)
PERF_CONFIG = NeuronConfig(
    tp_degree=64,
    cp_degree=1,
    batch_size=1,
    max_context_length=4096,
    seq_len=4096,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    fused_qkv=True,
    attn_kernel_enabled=True,
    on_device_sampling_config=OnDeviceSamplingConfig(),
    torch_dtype=torch.bfloat16,
)

@pytest.fixture(scope="module", autouse=True)
def model_path():
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config_chunked_attention.json"
    
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    yield model_path

    model_tempdir.cleanup()
@pytest.mark.tp64
@pytest.mark.chunked_attention
@pytest.mark.parametrize(
    "input_start_offsets, cp_degree, input_len, check_perf, latency_threshold, throughput_threshold, divergence_difference_tol",
    # fmt: off
    [
        ([ATTN_CHUNK_SIZE], 1, 16, False, float('inf'), 0, None),
        ([ATTN_CHUNK_SIZE], 16, 16, False, float('inf'), 0, None),
        ([ATTN_CHUNK_SIZE], 16, 130, False, float('inf'), 0, 0.0018), # input_len > s/cp
        ([0], 1, 16, False, float('inf'), 0, None),
        ([0], 16, 16, False, float('inf'), 0, None),
        ([0], 16, 130, False, float('inf'), 0, 0.0018), # input_len > s/cp
        ([ATTN_CHUNK_SIZE], 1, 16, True, 3143, 923, None),
    ],
    # fmt: on
)
def test_llama_4layer_chunked_attention_flash_attention_kernel(model_path, input_start_offsets, cp_degree, input_len, check_perf, latency_threshold, throughput_threshold, divergence_difference_tol):
    # Load model from config, and save with random weights.
    neuron_config = copy.deepcopy(PERF_CONFIG) if check_perf else copy.deepcopy(FA_KERNEL_CONFIG)
    neuron_config.cp_degree = cp_degree
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    if check_perf:
        validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold)
    else:
        validate_accuracy(model_path, config, generation_config, input_start_offsets=input_start_offsets, input_len=input_len, divergence_difference_tol=divergence_difference_tol)


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, input_start_offsets = [0], input_len = 16, divergence_difference_tol = None):
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
        num_tokens_to_check=256 - input_len,
        inputs=inputs,
        input_start_offsets=input_start_offsets,
        pad_token_id=128009,
        divergence_difference_tol=divergence_difference_tol,
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
