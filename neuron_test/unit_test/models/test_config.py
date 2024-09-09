import tempfile
from pathlib import Path

import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

TEST_CONFIG_PATH = Path(__file__).parent.parent / "resources"


def test_validate_config():
    class ValidatingInferenceConfig(InferenceConfig):
        def get_required_attributes(self):
            return ["hidden_size"]

    neuron_config = NeuronConfig()
    with pytest.raises(AssertionError, match=r"Config must define"):
        _ = ValidatingInferenceConfig(neuron_config)


def test_serialize_deserialize_basic_inference_config():
    neuron_config = NeuronConfig()
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    assert config.hidden_size == 4096
    assert neuron_config.tp_degree == 1

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.hidden_size == 4096
    assert deserialized_config.neuron_config.tp_degree == 1


def test_serialize_deserialize_inference_config_with_nested_config():
    lora_config = LoraServingConfig(max_lora_rank=32)
    neuron_config = NeuronConfig(lora_config=lora_config)
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    assert config.neuron_config.lora_config.max_lora_rank == 32

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.neuron_config.lora_config.max_lora_rank == 32


def test_serialize_deserialize_pretrained_config_adapter():
    neuron_config = NeuronConfig()
    config = InferenceConfig(neuron_config, load_config=load_pretrained_config(TEST_CONFIG_PATH))

    # Assert that an attribute from config.json is set on the config.
    assert config.model_type == "llama"

    # Assert that torch_dtype is copied to neuron_config correctly.
    assert not hasattr(config, "torch_dtype")
    assert neuron_config.torch_dtype == torch.bfloat16

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.model_type == "llama"
    assert not hasattr(deserialized_config, "torch_dtype")
    assert deserialized_config.neuron_config.torch_dtype == torch.bfloat16


def test_kwargs_override_load_config():
    neuron_config = NeuronConfig()
    config = InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(TEST_CONFIG_PATH),
        pad_token_id=2,
    )
    assert config.pad_token_id == 2


def verify_serialize_deserialize(config: InferenceConfig):
    """Verify that the config is identical after being serialized and deserialized."""
    with tempfile.TemporaryDirectory() as model_path:
        config.save(model_path)
        deserialized_config = InferenceConfig.load(model_path)
        assert config.to_json_string() == deserialized_config.to_json_string()
        return deserialized_config
