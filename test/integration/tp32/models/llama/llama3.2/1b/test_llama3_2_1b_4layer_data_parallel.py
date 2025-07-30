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

neuron_config = NeuronConfig(
    tp_degree=32,
    attention_dp_degree=4,
    batch_size=4,
    tkg_batch_size=4,
    ctx_batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=True,
    is_continuous_batching=True,
    torch_dtye=torch.float32,
)

@pytest.mark.tp32
@pytest.mark.data_parallel
@pytest.mark.parametrize(
    "neuron_config, num_kv_heads, latency_threshold, throughput_threshold, divergence_tol",
    # fmt: off
    [
        (neuron_config, 8, 264, 5104, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE),
        (neuron_config, 4, 300, 4104, 0.008),
    ],
    # fmt: on
)
def test_llama3_2_1b_4layer_data_parallel(neuron_config, num_kv_heads, latency_threshold, throughput_threshold, divergence_tol):
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path, num_kv_heads)
    model_path = model_tempdir.name
    
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    config.num_key_value_heads = num_kv_heads

    validate_accuracy(model_path, config, generation_config, divergence_tol)
    validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold)

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path, num_kv_heads):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_config.num_key_value_heads = num_kv_heads

    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, divergence_tol):
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
        num_tokens_to_check=config.neuron_config.max_context_length - input_len,
        inputs=inputs,
        divergence_difference_tol=divergence_tol,
    )


def validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold):
    config = copy.deepcopy(config)
    config.neuron_config.on_device_sampling_config = OnDeviceSamplingConfig()

    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_perf"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    benchmark_results = benchmark_sampling(model, generation_config=generation_config)
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"