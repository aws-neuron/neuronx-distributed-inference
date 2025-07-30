"""
Tests for Llama 3.1 8B model with different data types.
"""

import os
import pytest
import torch

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from transformers import GenerationConfig


from test.integration.utils.test_utils import (
    save_checkpoint,
    load_text_model_inputs,
    validate_e2e_performance,
)

# Set config_path as a global constant
CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + "/config.json"


# Combined parameterization data for accuracy and performance thresholds
NEURON_TEST_PARAMS = [
    pytest.param(
        torch.bfloat16,
        {50: (1e-5, 0.04)},  # tol_map
        0.13,  # divergence_difference_tol
        306 * 1.2,  # latency_threshold
        1665 * 0.8,  # throughput_threshold
        id="bfloat16"
    ),
    pytest.param(
        torch.float16,
        None,  # tol_map
        0.02,  # divergence_difference_tol
        289 * 1.2,  # latency_threshold
        1753 * 0.8,  # throughput_threshold
        id="float16"
    ),
    pytest.param(
        torch.float32,
        None,  # tol_map
        0.001,  # divergence_difference_tol
        466 * 1.2,  # latency_threshold
        1113 * 0.8,  # throughput_threshold
        id="float32"
    ),
]

# CPU-specific accuracy thresholds (only bfloat16 and float32 are supported)
CPU_ACCURACY_THRESHOLDS = [
    pytest.param(
        torch.bfloat16,
        {50: (1e-5, 0.04)},  # tol_map
        0.13,  # divergence_difference_tol
        id="bfloat16"
    ),
    pytest.param(
        torch.float32,
        None,  # tol_map
        0.001,  # divergence_difference_tol
        id="float32"
    ),
]


def build_and_load_model(model_class, model_path, config, cpu_mode=False):
    """
    Load a model to either CPU or Neuron device.

    Args:
        model_class: The model class to instantiate.
        model_path (str): Path to the model checkpoint.
        config: Model configuration object.
        cpu_mode (bool, optional): Whether to load the model to CPU. Defaults to False.

    Returns:
        The loaded model.
    """
    model = model_class(model_path, config)

    if cpu_mode:
        print("\nLoading model to CPU...")
        model.to_cpu()
    else:
        print("\nCompiling and loading model to device ...")
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        model.compile(compiled_model_path)
        model.load(compiled_model_path)

    return model


def run_model_accuracy_or_perf_tests(
    config_path,
    neuron_config,
    model_class,
    inference_config_class,
    cpu_mode=False,
    run_accuracy=True,
    run_perf=True,
    latency_threshold=0,
    throughput_threshold=0,
    tol_map=None,
    num_tokens_to_check=None,
    divergence_difference_tol=0.001,
    input_len=16,
    **config_kwargs,
):
    """Run model tests with the specified configuration.

    This function loads a model from a configuration file, initializes it with random weights,
    and runs accuracy and/or performance tests based on the provided parameters.

    Args:
        config_path (str): Path to the model configuration file.
        neuron_config (NeuronConfig): Configuration object containing Neuron-specific
            settings such as tensor parallelism degree, batch size, sequence length, etc.
        model_class: The model class to instantiate (e.g., NeuronLlamaForCausalLM).
        inference_config_class: The inference config class to use (e.g., LlamaInferenceConfig).
        cpu_mode (bool, optional): Whether to run the model on CPU instead of Neuron cores.
            When True, the model is loaded to CPU memory. When False, the model is compiled
            and loaded to Neuron devices. Defaults to False.
        run_accuracy (bool, optional): Whether to run accuracy validation tests.
            Validates model outputs against expected results. Defaults to True.
        run_perf (bool, optional): Whether to run performance benchmarking tests.
            Measures throughput and latency metrics.
            Defaults to True.
        latency_threshold (float, optional): Maximum allowed latency in ms. Defaults to 0 (no check).
        throughput_threshold (float, optional): Minimum required throughput. Defaults to 0 (no check).
        tol_map (dict, optional): Tolerance map for accuracy checking. Defaults to None.
        num_tokens_to_check (int, optional): Number of tokens to check for accuracy.
            If None, will use neuron_config.max_context_length - input_len. Defaults to None.
        divergence_difference_tol (float, optional): Tolerance for divergence difference.
            Defaults to 0.001.
        input_len (int, optional): Length of input sequence. Defaults to 16.
        **config_kwargs: Additional keyword arguments to pass to AutoConfig.from_pretrained.
            Can be used to override any config attributes as needed.

    Returns:
        tuple: (model_tempdir, benchmark_results) where:
            - model_tempdir: Temporary directory containing the model checkpoint.
              The caller is responsible for calling cleanup() on this object when done.
            - benchmark_results: Results from benchmark_sampling if run_perf is True,
              otherwise None.
    """
    model_tempdir = save_checkpoint(config_path, dtype=neuron_config.torch_dtype, **config_kwargs)
    model_path = model_tempdir.name

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)

    config = inference_config_class(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    benchmark_results = None

    # Load the model once and reuse it for both accuracy and performance testing
    model = build_and_load_model(model_class, model_path, config, cpu_mode)

    if run_accuracy:
        # Load model inputs
        inputs = load_text_model_inputs(model, input_len=input_len)

        # Use the inputs for accuracy checking
        if num_tokens_to_check is None:
            num_tokens_to_check = config.neuron_config.max_context_length - input_len

        check_accuracy_logits(
            model,
            generation_config=generation_config,
            num_tokens_to_check=num_tokens_to_check,
            inputs=inputs,  # prompt can be fed as `prompt=TEST_PROMPT` if inputs is None
            input_start_offsets=None,
            pad_token_id=0,
            tol_map=tol_map,
            divergence_difference_tol=divergence_difference_tol,
        )

    if run_perf:
        # Run benchmarking with the already loaded model
        benchmark_results = benchmark_sampling(model, generation_config=generation_config)

        # Validate performance if thresholds are provided
        if latency_threshold > 0 or throughput_threshold > 0:
            validate_e2e_performance(benchmark_results, latency_threshold, throughput_threshold)

    return model_tempdir, benchmark_results


@pytest.mark.tp32
@pytest.mark.llama31_8b
@pytest.mark.parametrize("torch_dtype,tol_map,divergence_difference_tol,latency_threshold,throughput_threshold", NEURON_TEST_PARAMS)
def test_llama3_1_8b_4layer_neuron_on_device_sampling(
    torch_dtype, tol_map, divergence_difference_tol, latency_threshold, throughput_threshold
):
    """
    Test Llama 3.1 8B 4-layer model with different torch data types.

    This test runs the model with on-device sampling using different data types:
    - torch.bfloat16
    - torch.float16
    - torch.float32

    """
    print(f"\nRunning test with torch_dtype: {torch_dtype}")

    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=2,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        torch_dtype=torch_dtype,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )

    input_len = 16
    num_tokens_to_check = None  # test up to the context length

    model_tempdir, _ = run_model_accuracy_or_perf_tests(
        config_path=CONFIG_PATH,
        neuron_config=neuron_config,
        model_class=NeuronLlamaForCausalLM,
        inference_config_class=LlamaInferenceConfig,
        cpu_mode=False,
        run_accuracy=True,
        run_perf=True,
        latency_threshold=latency_threshold,
        throughput_threshold=throughput_threshold,
        tol_map=tol_map,
        input_len=input_len,
        num_tokens_to_check=num_tokens_to_check,
        divergence_difference_tol=divergence_difference_tol,
    )

    # Clean up the model checkpoint only if the test passes
    model_tempdir.cleanup()


@pytest.mark.tp1
@pytest.mark.cpu
@pytest.mark.llama31_8b
@pytest.mark.parametrize("torch_dtype,tol_map,divergence_difference_tol", CPU_ACCURACY_THRESHOLDS)
def test_llama3_1_8b_4layer_cpu_accuracy(torch_dtype, tol_map, divergence_difference_tol):
    """
    Test Llama 3.1 8B 4-layer model on CPU with different torch data types.

    This test runs the model on CPU using different data types:
    - torch.float32
    - torch.bfloat16

    Note: CPU mode with tp > 1 (supported in torchrun) only supports float32, but tp=1 supports both float32 and bfloat16.
    """
    print(f"\nRunning CPU test with torch_dtype: {torch_dtype}")

    neuron_config = NeuronConfig(
        tp_degree=1,  # constraint: tp_degree > 1 is only supported when using 'torchrun'
        batch_size=2,
        seq_len=256,
        max_context_length=256,  # constraint: max_context_length should be equal to seq_len in CPU mode
        torch_dtype=torch_dtype,
        on_cpu=True,
    )

    input_len = 16
    num_tokens_to_check = 64  # full sequence takes long time on CPU

    model_tempdir, _ = run_model_accuracy_or_perf_tests(
        config_path=CONFIG_PATH,
        neuron_config=neuron_config,
        model_class=NeuronLlamaForCausalLM,
        inference_config_class=LlamaInferenceConfig,
        cpu_mode=True,
        run_accuracy=True,
        run_perf=False,
        tol_map=tol_map,
        input_len=input_len,
        num_tokens_to_check=num_tokens_to_check,
        divergence_difference_tol=divergence_difference_tol,
    )

    # Clean up the model checkpoint only if the test passes
    model_tempdir.cleanup()


if __name__ == "__main__":
    # Run tests with different dtypes using the combined parameterized data
    for param in NEURON_TEST_PARAMS:
        torch_dtype, tol_map, divergence_difference_tol, latency_threshold, throughput_threshold = param.values
        test_llama3_1_8b_4layer_neuron_on_device_sampling(
            torch_dtype, tol_map, divergence_difference_tol, latency_threshold, throughput_threshold
        )

    # Run CPU tests with supported dtypes
    for param in CPU_ACCURACY_THRESHOLDS:
        torch_dtype, tol_map, divergence_difference_tol = param.values
        test_llama3_1_8b_4layer_cpu_accuracy(torch_dtype, tol_map, divergence_difference_tol)
