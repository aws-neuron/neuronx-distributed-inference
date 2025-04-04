import logging
import unittest
from functools import partial

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

from .test_vit_utils import run_on_cpu, run_on_neuron, setup_debug_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAMES = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch32-384",
    "google/vit-huge-patch14-224-in21k",
]


class TestViTEmbeddings(unittest.TestCase):
    def test_patch_embeddings_different(self):
        for model_name in MODEL_NAMES:
            hf_config = ViTConfig.from_pretrained(model_name)

            neuron_config = NeuronConfig(
                tp_degree=2,
                torch_dtype=torch.float32,
            )
            inference_config = ViTInferenceConfig(
                neuron_config=neuron_config, load_config=load_pretrained_config(model_name)
            )

            test_inputs = (
                torch.randn(
                    1, 3, inference_config.image_size, inference_config.image_size
                ),  # pixel_values
                torch.tensor(False),  # interpolate_pos_encoding
            )
            golden_output = run_on_cpu(test_inputs, ViTPatchEmbeddings, hf_config)

            neuron_output = run_on_neuron(test_inputs, NeuronViTPatchEmbeddings, inference_config)

            passed, max_err = check_accuracy_embeddings(
                neuron_output, golden_output, plot_outputs=False, rtol=1.3e-6, atol=1e-5
            )
            logger.info(f"\n\n results {model_name} {passed}, {max_err}")
            assert (
                passed
            ), f"test_patch_embeddings did not pass model {model_name}, max error is {max_err}"

    def test_embeddings(self):
        for model_name in MODEL_NAMES:
            hf_config = ViTConfig.from_pretrained(model_name)

            neuron_config = NeuronConfig(
                tp_degree=2,
                torch_dtype=torch.float32,
            )
            inference_config = ViTInferenceConfig(
                neuron_config=neuron_config, load_config=load_pretrained_config(model_name)
            )

            test_inputs = (
                torch.randn(
                    1, 3, inference_config.image_size, inference_config.image_size
                ),  # pixel_values
                torch.ones(
                    [1, (inference_config.image_size // inference_config.patch_size) ** 2],
                    dtype=torch.bool,
                ),  # bool_masked_pos
                torch.tensor(True),  # interpolate_pos_encoding
            )

            golden_output = run_on_cpu(test_inputs, ViTEmbeddings, hf_config, use_mask_token=True)

            neuron_output = run_on_neuron(
                test_inputs, NeuronViTEmbeddings, inference_config, use_mask_token=True
            )

            passed, max_err = check_accuracy_embeddings(
                neuron_output, golden_output, plot_outputs=False, rtol=1.3e-6, atol=1e-5
            )
            logger.info(f"\n\n results {model_name} {passed}, {max_err}")
            assert (
                passed
            ), f"test_patch_embeddings did not pass model {model_name}, max error is {max_err}"


if __name__ == "__main__":
    # Set flags for debugging
    setup_debug_env()

    unittest.main()
