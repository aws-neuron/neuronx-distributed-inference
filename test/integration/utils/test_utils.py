"""
Utility functions for integration testing of models.
"""
import os
from argparse import Namespace
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


def save_checkpoint(config_path, dtype=torch.bfloat16, **kwargs):
    """
    Load model configuration with specified parameters and save a model with random weights.

    Args:
        config_path (str): Path to the model configuration file.
        dtype (torch.dtype, optional): Data type for model weights. Defaults to torch.bfloat16.
        **kwargs: Additional keyword arguments to override config from AutoConfig.from_pretrained
            Can be used to override any config attributes as needed.
            Or create new attributes as specified by the users

    Returns:
        tempfile.TemporaryDirectory: Temporary directory containing the saved model.
    """
    # Get the config from the path
    hf_config = AutoConfig.from_pretrained(config_path)

    # Apply any config overrides from kwargs
    for key, value in kwargs.items():
        if hasattr(hf_config, key):
            original_value = getattr(hf_config, key)
            print(f"Overriding {key} from {original_value} to {value}")
        else:
            print(f"Adding new attribute {key} with value {value}")
        setattr(hf_config, key, value)

    # Create the model with the updated config
    hf_model = AutoModel.from_config(hf_config, torch_dtype=dtype)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path} using dtype {dtype}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def load_text_model_inputs(
    model,
    input_len=16,
):
    """
    Load model inputs for inference or testing.

    Args:
        model: Pre-loaded model instance.
        input_len (int, optional): Length of input sequence. Defaults to 16.

    Returns:
        Namespace: Object containing input_ids and attention_mask tensors.
    """
    config = model.config

    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)

    return Namespace(input_ids=input_ids, attention_mask=attention_mask)


def validate_e2e_performance(benchmark_results, latency_threshold=0, throughput_threshold=0):
    """
    Validate end-to-end model performance by comparing benchmark results against thresholds.

    Args:
        benchmark_results (dict): Results from benchmark_sampling.
        latency_threshold (float, optional): Maximum allowed latency in ms. Defaults to 0 (no check).
        throughput_threshold (float, optional): Minimum required throughput. Defaults to 0 (no check).

    Returns:
        bool: True if all thresholds are met, False otherwise.
    """
    if "e2e_model" not in benchmark_results:
        raise ValueError("e2e_model not found in benchmark results")

    if latency_threshold > 0:
        latency = benchmark_results["e2e_model"]["latency_ms_p50"]
        assert (
            latency < latency_threshold
        ), f"latency ({latency}) is above threshold ({latency_threshold})"

    if throughput_threshold > 0:
        throughput = benchmark_results["e2e_model"]["throughput"]
        assert (
            throughput > throughput_threshold
        ), f"throughput ({throughput}) is below threshold ({throughput_threshold})"

    return True

