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

@pytest.mark.tp32
def test_llama3_2_1b_4layer_neuron_on_device_sampling():
    """
    # with logit enabled:
    {
        "e2e_model": {
            "latency_ms_p50": 279.6483039855957,
            "latency_ms_p90": 284.64014530181885,
            "latency_ms_p95": 286.96556091308594,
            "latency_ms_p99": 288.54488372802734,
            "latency_ms_p100": 288.9397144317627,
            "latency_ms_avg": 279.3704152107239,
            "throughput": 1832.6922684845065
        },
        "context_encoding_model": {
            "latency_ms_p50": 2.121567726135254,
            "latency_ms_p90": 2.168107032775879,
            "latency_ms_p95": 2.1848320960998535,
            "latency_ms_p99": 2.1879124641418457,
            "latency_ms_p100": 2.1886825561523438,
            "latency_ms_avg": 2.1269917488098145,
            "throughput": 120357.77766568586
        },
        "token_generation_model": {
            "latency_ms_p50": 1.3585090637207031,
            "latency_ms_p90": 1.399064064025879,
            "latency_ms_p95": 1.4117002487182615,
            "latency_ms_p99": 1.6343665122985847,
            "latency_ms_p100": 2.619028091430664,
            "latency_ms_avg": 1.362218068340632,
            "throughput": 1479.754290699961
        }
    }
    # with logit disabled:
    {
        "e2e_model": {
            "latency_ms_p50": 134.83953475952148,
            "latency_ms_p90": 135.31560897827148,
            "latency_ms_p95": 135.44069528579712,
            "latency_ms_p99": 136.7614483833313,
            "latency_ms_p100": 137.09163665771484,
            "latency_ms_avg": 134.82286930084229,
            "throughput": 3797.575312371737
        },
        "context_encoding_model": {
            "latency_ms_p50": 1.4035701751708984,
            "latency_ms_p90": 1.419973373413086,
            "latency_ms_p95": 1.4309048652648926,
            "latency_ms_p99": 1.4405083656311035,
            "latency_ms_p100": 1.4429092407226562,
            "latency_ms_avg": 1.4043450355529785,
            "throughput": 182291.38389711812
        },
        "token_generation_model": {
            "latency_ms_p50": 0.6699562072753906,
            "latency_ms_p90": 0.6871223449707031,
            "latency_ms_p95": 0.6928563117980956,
            "latency_ms_p99": 0.7776403427124026,
            "latency_ms_p100": 3.000497817993164,
            "latency_ms_avg": 0.6692040623642328,
            "throughput": 3012.1574940454207
        }
    }
    """

    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=2,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )
    # latency is a bit flaky hence increasing to 300 ms
    run_llama3_2_1b_4layer(neuron_config, cpu_mode=False, run_accuracy=True, run_perf=True,
                           latency_threshold=300 * 1.1, throughput_threshold = 1832 * 0.9)

@pytest.mark.tp32
def test_llama3_2_1b_4layer_continuous_batching_neuron_on_device_sampling():
    """
    # the test send requests in batches instead of interleaving multiple requests (continuous batching)
    {
        "e2e_model": {
            "latency_ms_p50": 309.39459800720215,
            "latency_ms_p90": 314.711856842041,
            "latency_ms_p95": 316.727352142334,
            "latency_ms_p99": 323.05768966674805,
            "latency_ms_p100": 324.64027404785156,
            "latency_ms_avg": 309.7065806388855,
            "throughput": 1653.1776591372673
        },
        "context_encoding_model": {
            "latency_ms_p50": 3.7462711334228516,
            "latency_ms_p90": 3.8457393646240234,
            "latency_ms_p95": 3.901815414428712,
            "latency_ms_p99": 4.5845699310302725,
            "latency_ms_p100": 4.755258560180664,
            "latency_ms_avg": 3.7874341011047363,
            "throughput": 67591.9351112482
        },
        "token_generation_model": {
            "latency_ms_p50": 1.3948678970336914,
            "latency_ms_p90": 1.4429092407226562,
            "latency_ms_p95": 1.4600753784179688,
            "latency_ms_p99": 1.5731883049011248,
            "latency_ms_p100": 8.270740509033203,
            "latency_ms_avg": 1.4003710483941505,
            "throughput": 1439.4385215315501
        }
    }
    """
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=2,
        ctx_batch_size=1,
        max_context_length=128,
        seq_len=256,
        is_continuous_batching=True,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )
    run_llama3_2_1b_4layer(neuron_config, cpu_mode=False, run_accuracy=True, run_perf=True,
                           latency_threshold=309 * 1.1, throughput_threshold = 1653 * 0.9)

@pytest.mark.tp1
@pytest.mark.cpu
def test_llama3_2_1b_4layer_cpu_accuracy():
    neuron_config = NeuronConfig(
        tp_degree=1,  # constraint: tp_degree > 1 is only supported when using 'torchrun'
        batch_size=2,
        seq_len=256,
        max_context_length=256, # constraint: max_context_length should be equal to seq_len in CPU mode
        torch_dtype=torch.float32,  # optional: cpu mode backend with tp > 1 supports only float32. Tp=1 supports bfloat16 as well.
        on_cpu=True,
    )
    run_llama3_2_1b_4layer(neuron_config, cpu_mode=True, run_accuracy=True, run_perf=False)

def run_llama3_2_1b_4layer(neuron_config, cpu_mode=False, run_accuracy=True, run_perf=True,
                           latency_threshold=0, throughput_threshold=0):
    """Run Llama 3.2 1B 4-layer model tests with the specified configuration.
    
    This function loads a Llama 3.2 1B model with 4 layers from a configuration file,
    initializes it with random weights, and runs accuracy and/or performance tests
    based on the provided parameters.

    Args:
        neuron_config (NeuronConfig): Configuration object containing Neuron-specific 
            settings such as tensor parallelism degree, batch size, sequence length, etc.
        cpu_mode (bool, optional): only support in accuracy tests
            Whether to run the model on CPU instead of Neuron cores.
            When True, the model is loaded to CPU memory. When False, the model is compiled
            and loaded to Neuron devices. Defaults to False.
        run_accuracy (bool, optional): Whether to run accuracy validation tests.
            Validates model outputs against expected results. Defaults to True.
        run_perf (bool, optional): Whether to run performance benchmarking tests.
            Measures throughput and latency metrics. Only runs when not in CPU mode.
            Defaults to True.
    """
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    if run_accuracy:
        validate_accuracy(model_path, config, generation_config, cpu_mode=cpu_mode)
    if run_perf:
        validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold)

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, cpu_mode=False):
    input_len = 16
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronLlamaForCausalLM(model_path, config)
    
    if cpu_mode:
        print("\nLoading model to CPU...")
        model.to_cpu()
    else:
        print("\nCompiling and loading model to device ...")
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        model.compile(compiled_model_path)
        model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        num_tokens_to_check=config.neuron_config.max_context_length - input_len,
        inputs=inputs,
    )


def validate_performance(model_path, config, generation_config, latency_threshold, throughput_threshold):
    config = copy.deepcopy(config)

    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_perf"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    benchmark_results = benchmark_sampling(model, generation_config=generation_config)
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"



if __name__ == "__main__":
    test_llama3_2_1b_4layer_neuron_on_device_sampling()
    test_llama3_2_1b_4layer_continuous_batching_neuron_on_device_sampling()
    test_llama3_2_1b_4layer_cpu_accuracy()