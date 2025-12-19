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

NEURON_CONFIG_CP16_DP16 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    attention_dp_degree=16,
    batch_size=16,
    ctx_batch_size=1,
    tkg_batch_size=16,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP4_DP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=4,
    attention_dp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP8_DP8 = NeuronConfig(
    tp_degree=64,
    cp_degree=8,
    attention_dp_degree=8,
    batch_size=8,
    ctx_batch_size=1,
    tkg_batch_size=8,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP8_DP8_WITHOUT_SP = NeuronConfig(
    tp_degree=64,
    cp_degree=8,
    attention_dp_degree=8,
    batch_size=8,
    ctx_batch_size=1,
    tkg_batch_size=8,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=False,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP16_DP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    attention_dp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP8_DP4 = NeuronConfig(
    tp_degree=64,
    cp_degree=8,
    attention_dp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP16_DP8 = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    attention_dp_degree=8,
    batch_size=8,
    ctx_batch_size=1,
    tkg_batch_size=8,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

NEURON_CONFIG_CP16_DP8_MULTI_BATCH = NeuronConfig(
    tp_degree=64,
    cp_degree=16,
    attention_dp_degree=8,
    batch_size=16,
    ctx_batch_size=1,
    tkg_batch_size=16,
    max_context_length=128,
    seq_len=128,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

# TKG attention kernel doesn't run in fp32, and downcasts most of the compute, some precision loss is expected.
NEURON_CONFIG_CP4_DP4_TKG_ATTN_KERNEL = NeuronConfig(
    tp_degree=64,
    cp_degree=4,
    attention_dp_degree=4,
    batch_size=4,
    ctx_batch_size=1,
    tkg_batch_size=4,
    seq_len=2048,
    sequence_parallel_enabled=True,
    logical_nc_config=2,
    attn_kernel_enabled=False,
    is_continuous_batching=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
    attn_block_tkg_nki_kernel_enabled=True,
    attn_block_tkg_nki_kernel_cache_update=False,
    qkv_kernel_enabled=True,
)


@pytest.mark.tp64
@pytest.mark.context_parallel
@pytest.mark.data_parallel
@pytest.mark.parametrize(
    "neuron_config, num_kv_heads, latency_threshold, throughput_threshold, divergence_tolerance",
    # fmt: off
    [
        (NEURON_CONFIG_CP4_DP4, 8, 304, 1309, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        (NEURON_CONFIG_CP16_DP16, 8, 385, 5871, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        (NEURON_CONFIG_CP16_DP4, 8, 288, 1494, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        (NEURON_CONFIG_CP16_DP4, 4, 308, 1254, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE), # TP 4 CTE, TP 16 Decode, 16 copies of KV in prefill, 16 copies needed in decode
        pytest.param(NEURON_CONFIG_CP8_DP8, 8, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=pytest.mark.xfail),
        pytest.param(NEURON_CONFIG_CP8_DP8_WITHOUT_SP, 8, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=pytest.mark.xfail),
        pytest.param(NEURON_CONFIG_CP8_DP4, 8, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=pytest.mark.xfail),
        pytest.param(NEURON_CONFIG_CP16_DP8, 8, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=pytest.mark.xfail),
        pytest.param(NEURON_CONFIG_CP16_DP8_MULTI_BATCH, 8, float("inf"), 0, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, marks=pytest.mark.xfail),
        (NEURON_CONFIG_CP4_DP4_TKG_ATTN_KERNEL, 8, float("inf"), 0, 0.08),
    ],
    # fmt: on
)
def test_llama_4layer_context_parallel(neuron_config, num_kv_heads, latency_threshold, throughput_threshold, divergence_tolerance):
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path, num_kv_heads)
    model_path = model_tempdir.name

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    config.num_key_value_heads = num_kv_heads

    if config.neuron_config.on_device_sampling_config is None:
        validate_accuracy(model_path, config,
                          generation_config, divergence_tolerance)

    if throughput_threshold > 0:
        validate_performance(model_path, config, generation_config,
                             latency_threshold, throughput_threshold)


def save_checkpoint(config_path, num_kv_heads):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_config.num_key_value_heads = num_kv_heads

    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, divergence_tolerance):
    input_len = 16
    input_ids = torch.rand(
        (config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones(
        (config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        num_tokens_to_check=128 - input_len,
        inputs=inputs,
        divergence_difference_tol=divergence_tolerance,
    )


def validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold):
    config = copy.deepcopy(config)
    config.neuron_config.on_device_sampling_config = OnDeviceSamplingConfig()

    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_perf"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    benchmark_results = benchmark_sampling(
        model, generation_config=generation_config)
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"