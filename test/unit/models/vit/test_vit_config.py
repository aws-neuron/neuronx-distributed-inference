import logging
import unittest

from transformers import ViTConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.vit.modeling_vit import ViTInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from .test_vit_utils import setup_debug_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAMES = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch16-224",
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch32-384",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch16-224",
    "google/vit-large-patch16-384",
    "google/vit-large-patch32-224-in21k",
    "google/vit-large-patch32-384",
    "google/vit-huge-patch14-224-in21k",
]


class TestViTInferenceConfig(unittest.TestCase):
    def test_HF_config(self):
        for model_name in MODEL_NAMES:
            hf_config = ViTConfig.from_pretrained(model_name)

            neuron_config = NeuronConfig()
            inference_config = ViTInferenceConfig(
                neuron_config=neuron_config, load_config=load_pretrained_config(model_name)
            )

            for key in hf_config.to_diff_dict().keys():
                logger.info(f"{model_name}, {key}")
                assert hasattr(
                    inference_config, key
                ), f"Model {model_name} HF ViT config {key} is not in ViTInferenceConfig"

    def test_added_config(self):
        model_name = MODEL_NAMES[0]
        neuron_config = NeuronConfig()
        inference_config = ViTInferenceConfig(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(model_name),
            use_mask_token=True,
            add_pooling_layer=True,
            interpolate_pos_encoding=True,
        )
        assert (
            inference_config.use_mask_token
        ), f"inference_config.use_mask_token is {inference_config.use_mask_token}"
        assert (
            inference_config.add_pooling_layer
        ), f"inference_config.add_pooling_layer is {inference_config.add_pooling_layer}"
        assert (
            inference_config.interpolate_pos_encoding
        ), f"inference_config.interpolate_pos_encoding is {inference_config.interpolate_pos_encoding}"


if __name__ == "__main__":
    # Set flags for debugging
    setup_debug_env()

    unittest.main()
