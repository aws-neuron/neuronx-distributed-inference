import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Type
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig,
    InferenceConfig,
    NeuronConfig,
    get_platform_lnc,
)
from neuronx_distributed_inference.models.mllama.modeling_mllama import (
    MllamaInferenceConfig,
    MultimodalVisionNeuronConfig,
)
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

TEST_CONFIG_PATH = Path(__file__).parent.parent / "resources"
TEST_MM_CONFIG_PATH = Path(__file__).parent.parent / "resources_multi_modal"


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


def test_serialize_deserialize_inference_config_with_fused_spec_config():
    neuron_config = NeuronConfig()
    draft_config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=1024,
    )
    fused_spec_config = FusedSpecNeuronConfig(
        NeuronBaseModel, draft_config=draft_config, draft_model_path="draft_model_path"
    )
    config = InferenceConfig(
        neuron_config=neuron_config,
        fused_spec_config=fused_spec_config,
        hidden_size=4096,
    )
    assert config.hidden_size == 4096
    assert neuron_config.tp_degree == 1
    assert config.fused_spec_config.draft_config.hidden_size == 1024
    assert config.fused_spec_config.draft_config.neuron_config.tp_degree == 1

    deserialized_config = verify_serialize_deserialize(config)

    assert deserialized_config.hidden_size == 4096
    assert deserialized_config.neuron_config.tp_degree == 1
    assert deserialized_config.fused_spec_config.draft_config.hidden_size == 1024
    assert deserialized_config.fused_spec_config.draft_config.neuron_config.tp_degree == 1


def test_neuron_config_logical_neuron_cores_backward_compatible():
    with warnings.catch_warnings(record=True) as w:
        neuron_config = NeuronConfig(logical_neuron_cores=2)
        config = InferenceConfig(
            neuron_config=neuron_config,
            hidden_size=4096,
        )
        assert config.hidden_size == 4096
        assert neuron_config.logical_nc_config == 2
        assert neuron_config.logical_neuron_cores == 2

        # Validate that the deprecated "logical_neuron_cores" attr isn't serialized.
        config_json = json.loads(config.to_json_string())
        print(config_json)
        assert "logical_neuron_cores" not in config_json["neuron_config"]

        deserialized_config = verify_serialize_deserialize(config)
        assert deserialized_config.neuron_config.logical_nc_config == 2
        assert deserialized_config.neuron_config.logical_neuron_cores == 2

        # Validate the deprecation warning is triggered three times (once in constructor, twice on access).
        lnc_deprecation_warning_count = 0
        for warning in w:
            message = str(warning.message)
            if "Unexpected keyword arguments" in message:
                continue

            if (
                issubclass(warning.category, DeprecationWarning)
                and "deprecated" in message
                and "logical_neuron_cores" in message
            ):
                lnc_deprecation_warning_count += 1
        assert lnc_deprecation_warning_count == 3


def test_serialize_deserialize_pretrained_config_adapter():
    neuron_config = NeuronConfig()
    config = InferenceConfig(neuron_config, load_config=load_pretrained_config(TEST_CONFIG_PATH))

    # Assert that an attribute from config.json is set on the config.
    assert config.model_type == "llama"

    # Assert that torch_dtype is copied to neuron_config correctly.
    assert not hasattr(config, "torch_dtype")
    assert neuron_config.torch_dtype == torch.bfloat16
    assert not neuron_config.overrides_torch_dtype

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


def test_serialize_deserialize_pretrained_config_adapter_where_neuron_config_overrides_dtype():
    neuron_config = NeuronConfig(torch_dtype=torch.float32)
    config = InferenceConfig(neuron_config, load_config=load_pretrained_config(TEST_CONFIG_PATH))
    assert neuron_config.torch_dtype == torch.float32
    assert neuron_config.overrides_torch_dtype

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.neuron_config.torch_dtype == torch.float32
    assert deserialized_config.neuron_config.overrides_torch_dtype


def test_preloaded_pretrained_config():
    hf_config = AutoConfig.from_pretrained(TEST_CONFIG_PATH)
    neuron_config = NeuronConfig()
    config = InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    # Assert that an attribute from config.json is set on the config.
    assert config.model_type == "llama"

    # Assert that torch_dtype is copied to neuron_config correctly.
    assert not hasattr(config, "torch_dtype")
    assert neuron_config.torch_dtype == torch.bfloat16
    assert not neuron_config.overrides_torch_dtype


def test_multi_modal_preloaded_pretrained_config():
    hf_config = AutoConfig.from_pretrained(TEST_MM_CONFIG_PATH)
    neuron_config = NeuronConfig()
    config = MllamaInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    # Assert that an attribute from config.json is set on the config.
    assert config.checkpoint == "META"

    # Asset nested configs are set correctly
    assert hasattr(config, "text_config")
    assert hasattr(config, "vision_config")
    assert isinstance(config.text_config, InferenceConfig)
    assert isinstance(config.vision_config, InferenceConfig)

    # Assert that torch_dtype is copied to neuron_config correctly.
    assert not hasattr(config, "torch_dtype")
    assert not hasattr(config.text_config, "torch_dtype")
    assert not hasattr(config.vision_config, "torch_dtype")
    assert neuron_config.torch_dtype == torch.bfloat16
    assert not neuron_config.overrides_torch_dtype


def test_get_text_config_multi_modal():
    hf_config = AutoConfig.from_pretrained(TEST_MM_CONFIG_PATH)
    neuron_config = NeuronConfig()
    config = MllamaInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    text_config = config.get_text_config()

    assert text_config != config
    assert text_config.vocab_size == 128256


def test_get_text_config_text_model():
    hf_config = AutoConfig.from_pretrained(TEST_CONFIG_PATH)
    neuron_config = NeuronConfig()
    config = InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    text_config = config.get_text_config()

    assert text_config == config
    assert text_config.vocab_size == 32000


def test_serialize_deserialize_mllama_inference_config():
    hf_config = AutoConfig.from_pretrained(TEST_MM_CONFIG_PATH)
    neuron_config = MultimodalVisionNeuronConfig()
    config = MllamaInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    assert config.text_config.hidden_size == 4096
    assert config.vision_config.attention_heads == 16
    assert config.neuron_config.tp_degree == 1

    deserialized_config = verify_serialize_deserialize(config, MllamaInferenceConfig)

    assert deserialized_config.text_config.hidden_size == 4096
    assert deserialized_config.vision_config.attention_heads == 16
    assert deserialized_config.neuron_config.tp_degree == 1


def verify_serialize_deserialize(
    config: InferenceConfig, config_cls: Type[InferenceConfig] = InferenceConfig
):
    """Verify that the config is identical after being serialized and deserialized."""
    with tempfile.TemporaryDirectory() as model_path:
        config.save(model_path)
        deserialized_config = config_cls.load(model_path)
        assert config.to_json_string() == deserialized_config.to_json_string()
        return deserialized_config


class TestGetPlatformLNC(unittest.TestCase):
    @patch("neuronx_distributed_inference.models.config.get_platform_target")
    def test_get_platform_lnc(self, get_platform_target_mock):
        get_platform_target_mock.return_value = "trn1"
        assert get_platform_lnc() == 1

        get_platform_target_mock.return_value = "inf2"
        assert get_platform_lnc() == 1

        get_platform_target_mock.return_value = "trn2"
        assert get_platform_lnc() == 2

        assert get_platform_target_mock.call_count == 3
