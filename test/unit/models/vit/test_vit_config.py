import logging

from transformers import ViTConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.vit.modeling_vit import ViTInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from .test_vit_utils import VIT_CONFIG_PATH, setup_debug_env

# Set flags for debugging
setup_debug_env()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_HF_config():
    for model_path in VIT_CONFIG_PATH:
        hf_config = ViTConfig.from_pretrained(model_path)

        neuron_config = NeuronConfig()
        inference_config = ViTInferenceConfig(
            neuron_config=neuron_config, load_config=load_pretrained_config(model_path)
        )

        for key in hf_config.to_diff_dict().keys():
            logger.info(f"{model_path}, {key}")
            assert hasattr(
                inference_config, key
            ), f"Model {model_path} HF ViT config {key} is not in ViTInferenceConfig"


def test_added_config():
    model_name = VIT_CONFIG_PATH[0]
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
