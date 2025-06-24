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



ATTN_CHUNK_SIZE = 128


@pytest.mark.tp32
@pytest.mark.chunked_attention
@pytest.mark.parametrize(
    "batch_size, seq_len, input_start_offsets, latency_threshold, throughput_threshold",
    # fmt: off
    [
        (1, 256, [ATTN_CHUNK_SIZE], 252, 1000),  # seq_len divisible by chunk size
        (1, 192, [ATTN_CHUNK_SIZE], 252, 1100),  # seq_len not divisible by chunk size
        (2, 256, [ATTN_CHUNK_SIZE], 270, 2000),  # bs 2
        (2, 256, [ATTN_CHUNK_SIZE, 0], 270, 2000),  # bs 2, mixed input offset seq1: [..., pad, pad, 1, 2, 3, 4, 5], seq2: [1, 2, 3, 4, 5, pad, pad]
        (1, 256, [0], 252, 1000),  # base case, input start at 0
    ],
    # fmt: on
)
def test_llama3_2_1b_4layer_chunked_attention(
    batch_size, seq_len, input_start_offsets, latency_threshold, throughput_threshold
):
    # Load model from config, and save with random weights.
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        seq_len=seq_len,
        sequence_parallel_enabled=True,
    )
    config_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/config_chunked_attention.json"
    )
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    validate_accuracy(model_path, config, generation_config, input_start_offsets=input_start_offsets)
    validate_performance(
        model_path, config, generation_config, latency_threshold, throughput_threshold
    )

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.float16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, input_start_offsets = 0):
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
        prompt="",
        num_tokens_to_check=ATTN_CHUNK_SIZE - input_len,
        inputs=inputs,
        input_start_offsets=input_start_offsets,
        pad_token_id=128009,
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