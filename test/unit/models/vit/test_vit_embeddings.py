import logging

import pytest
import torch
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPatchEmbeddings

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.vit.modeling_vit import (
    NeuronViTEmbeddings,
    NeuronViTPatchEmbeddings,
    ViTInferenceConfig,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from .test_vit_utils import VIT_CONFIG_PATH, get_rtol, run_on_cpu, run_on_neuron, setup_debug_env

# Set flags for debugging
setup_debug_env()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "dtype,batch_size,model_path",
    [
        pytest.param(
            dtype,
            batch_size,
            model_path,
            id=f"dtype_{str(dtype).split('.')[-1]}_batch_size_{batch_size}_model_path_{model_path.split('/')[-1]}",
        )
        for dtype in [torch.float32, torch.float16]
        for batch_size in [1, 4]
        for model_path in VIT_CONFIG_PATH
    ],
)
def test_patch_embeddings(dtype, batch_size, model_path):
    logger.info(f"\n\n Testing dtype {dtype}, batch_size {batch_size}, model_path {model_path}")
    hf_config = ViTConfig.from_pretrained(model_path)

    neuron_config = NeuronConfig(
        tp_degree=2,
        torch_dtype=dtype,
    )
    inference_config = ViTInferenceConfig(
        neuron_config=neuron_config, load_config=load_pretrained_config(model_path)
    )

    test_inputs = (
        torch.randn(
            [batch_size, 3, inference_config.image_size, inference_config.image_size], dtype=dtype
        ),  # pixel_values
        torch.tensor(False),  # interpolate_pos_encoding
    )
    golden_output = run_on_cpu(test_inputs, ViTPatchEmbeddings, hf_config, dtype=dtype)
    logger.info(f"golden_output, {golden_output.shape}, {golden_output}")

    neuron_output = run_on_neuron(test_inputs, NeuronViTPatchEmbeddings, inference_config)
    logger.info(f"neuron_output, {neuron_output.shape}, {neuron_output}")

    passed, max_err = check_accuracy_embeddings(
        neuron_output, golden_output, plot_outputs=False, rtol=get_rtol(dtype), atol=1e-5
    )
    logger.info(f"\n\n results {model_path} {passed}, {max_err}")
    assert (
        passed
    ), f"NeuronViTPatchEmbeddings did not pass dtype {dtype}, batch_size {batch_size}, model_path {model_path}, max error is {max_err}"


@pytest.mark.parametrize(
    "dtype,batch_size,model_path",
    [
        pytest.param(
            dtype,
            batch_size,
            model_path,
            id=f"dtype_{str(dtype).split('.')[-1]}_batch_size_{batch_size}_model_path_{model_path.split('/')[-1]}",
        )
        for dtype in [torch.float32, torch.float16]
        for batch_size in [1, 4]
        for model_path in VIT_CONFIG_PATH
    ],
)
def test_embeddings(dtype, batch_size, model_path):
    logger.info(f"\n\n Testing dtype {dtype}, batch_size {batch_size}, model_path {model_path}")
    hf_config = ViTConfig.from_pretrained(model_path)

    neuron_config = NeuronConfig(
        tp_degree=2,
        torch_dtype=dtype,
    )
    inference_config = ViTInferenceConfig(
        neuron_config=neuron_config, load_config=load_pretrained_config(model_path)
    )

    test_inputs = (
        torch.randn(
            [batch_size, 3, inference_config.image_size, inference_config.image_size], dtype=dtype
        ),  # pixel_values
    )

    golden_output = run_on_cpu(test_inputs, ViTEmbeddings, hf_config, dtype=dtype)
    logger.info(f"golden_output, {golden_output.shape}, {golden_output}")

    neuron_output = run_on_neuron(
        test_inputs,
        NeuronViTEmbeddings,
        inference_config,
    )
    logger.info(f"neuron_output, {neuron_output.shape}, {neuron_output}")

    passed, max_err = check_accuracy_embeddings(
        neuron_output, golden_output, plot_outputs=False, rtol=get_rtol(dtype), atol=1e-5
    )
    logger.info(f"\n\n results {model_path} {passed}, {max_err}")
    assert (
        passed
    ), f"NeuronViTEmbeddings did not pass dtype {dtype}, batch_size {batch_size}, model_path {model_path}, max error is {max_err}"
