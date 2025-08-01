import os
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Type
from unittest.mock import patch

from neuronx_distributed_inference.models.config import LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE
import pytest
import torch
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig,
    InferenceConfig,
    NeuronConfig,
    ChunkedPrefillConfig,
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

def test_validate_chunked_attention_config():
    neuron_config = NeuronConfig(cp_degree=2, tp_degree=2, padding_side="left")
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    config.attention_chunk_size = 4
    with pytest.raises(ValueError, match=r"The Neuron config padding_side: left is not yet supported with chunked attention"):
        _ = config._validate_chunked_attention_support()
    config.neuron_config.padding_side = "right"
    config.neuron_config.cp_degree = 3
    config.neuron_config.tp_degree = 3
    with pytest.raises(AssertionError, match=r"attention_chunk_size: 4 must be divisible by cp_degree: 3"):
        _ = config._validate_chunked_attention_support()
    config.neuron_config.seq_len = 5
    config.attention_chunk_size = 3
    with pytest.raises(AssertionError, match=r"The last chunk must be divisible by cp_degree: 3"):
        _ = config._validate_chunked_attention_support()
    ## acceptable chunked attention config case
    config.attention_chunk_size = 2
    config.neuron_config.seq_len = 4
    config.neuron_config.cp_degree = 2
    config.neuron_config.tp_degree = 2
    config._validate_chunked_attention_support()

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


def test_serialize_deserialize_inference_config_with_nested_lora_config():
    lora_config = LoraServingConfig(max_lora_rank=32)
    neuron_config = NeuronConfig(lora_config=lora_config)
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    assert config.neuron_config.lora_config.max_lora_rank == 32

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.neuron_config.lora_config.max_lora_rank == 32


def test_serialize_deserialize_inference_config_with_nested_chunked_prefill_config():
    chunked_prefill_config = ChunkedPrefillConfig(
        max_num_seqs=8,
        tkg_model_enabled=True,
        kernel_q_tile_size=256,
        kernel_kv_tile_size=2048,
    )
    neuron_config = NeuronConfig(
        chunked_prefill_config=chunked_prefill_config,
        is_block_kv_layout=True,
        batch_size=1,
    )
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    actual = config.neuron_config.chunked_prefill_config
    assert actual.max_num_seqs == 8
    assert actual.tkg_model_enabled
    assert actual.kernel_q_tile_size == 256
    assert actual.kernel_kv_tile_size == 2048

    deserialized_config = verify_serialize_deserialize(config)
    deserialized_actual = deserialized_config.neuron_config.chunked_prefill_config
    assert deserialized_actual.max_num_seqs == 8
    assert deserialized_actual.tkg_model_enabled
    assert deserialized_actual.kernel_q_tile_size == 256
    assert deserialized_actual.kernel_kv_tile_size == 2048


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
    assert type(deserialized_config.fused_spec_config.worker_cls) is type(NeuronBaseModel)


def test_neuron_config_lnc():
    # Capture original env var so that it can be reset at end of test
    original_env_value = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    platform_lnc = get_platform_lnc()
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(platform_lnc - 1)
    expected_lnc_value = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", platform_lnc))

    with warnings.catch_warnings(record=True) as w:
        neuron_config = NeuronConfig(logical_nc_config=platform_lnc)
        assert neuron_config.logical_nc_config == expected_lnc_value

        # Validate the deprecation warning is triggered once.
        lnc_mismatch_warning_count = 0
        for warning in w:
            message = str(warning.message)
            if "does not match provided logical_nc_config" in message:
                lnc_mismatch_warning_count += 1

        assert lnc_mismatch_warning_count == 1
    
    # Reset env variable at the end of the test
    del os.environ["NEURON_LOGICAL_NC_CONFIG"]
    if original_env_value is not None:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = original_env_value


def test_neuron_config_logical_neuron_cores_backward_compatible():

    # Unset environment variable
    original_env_value = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    if original_env_value is not None:
        del os.environ["NEURON_LOGICAL_NC_CONFIG"]

    with warnings.catch_warnings(record=True) as w:
        logical_neuron_cores=1
        neuron_config = NeuronConfig(logical_neuron_cores=logical_neuron_cores)
        config = InferenceConfig(
            neuron_config=neuron_config,
            hidden_size=4096,
        )
        expected_lnc_value = logical_neuron_cores
        assert config.hidden_size == 4096
        assert neuron_config.logical_nc_config == expected_lnc_value
        assert neuron_config.logical_neuron_cores == expected_lnc_value

        # Validate that the deprecated "logical_neuron_cores" attr isn't serialized.
        config_json = json.loads(config.to_json_string())
        print(config_json)
        assert "logical_neuron_cores" not in config_json["neuron_config"]

        deserialized_config = verify_serialize_deserialize(config)
        assert deserialized_config.neuron_config.logical_nc_config == expected_lnc_value
        assert deserialized_config.neuron_config.logical_neuron_cores == expected_lnc_value

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

    # Reset environment variable
    if original_env_value is not None:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = original_env_value


def test_serialize_deserialize_pretrained_config_adapter():
    neuron_config = NeuronConfig()
    config = InferenceConfig(neuron_config, load_config=load_pretrained_config(TEST_CONFIG_PATH))

    # Assert that an attribute from config.json is set on the config.
    assert config.model_type == "llama"
    assert config.transformers_version == "4.31.0"

    # Assert that torch_dtype is copied to neuron_config correctly.
    assert not hasattr(config, "torch_dtype")
    assert neuron_config.torch_dtype == torch.bfloat16
    assert not neuron_config.overrides_torch_dtype

    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.model_type == "llama"
    assert not hasattr(deserialized_config, "torch_dtype")
    assert deserialized_config.neuron_config.torch_dtype == torch.bfloat16

def test_serialize_deserialize_long_context_neuron_config():
    neuron_config = NeuronConfig(max_context_length=32 * 1024, seq_len=32 * 1024)
    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=4096,
    )
    assert neuron_config.enable_long_context_mode
    assert neuron_config.scratchpad_page_size == LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE
    
    deserialized_config = verify_serialize_deserialize(config)
    assert deserialized_config.neuron_config.enable_long_context_mode
    assert deserialized_config.neuron_config.scratchpad_page_size == LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE

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


def test_kv_cache_tiling_disabling():
    hf_config = AutoConfig.from_pretrained(TEST_CONFIG_PATH)
    neuron_config = NeuronConfig(max_length = 2048, enable_fused_speculation=True, disable_kv_cache_tiling=True)
    config = InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    # Assert that kv cache tiling has been turned off by the user.
    assert not config.neuron_config.kv_cache_tiling


def test_inference_config_from_json_string_invalid_fused_spec_worker_cls():
    worker_cls_name = "NotReal"
    worker_cls_module = "not.real"
    config = {
        "fused_spec_config": {
            "worker_cls": {
                "__name__": worker_cls_name,
                "__module__": worker_cls_module,
            },
        },
    }
    config_json = json.dumps(config)
    with pytest.raises(ModuleNotFoundError) as e_info:
        config = InferenceConfig.from_json_string(config_json)
        message = str(e_info)
        assert worker_cls_name in message and worker_cls_module in message

def test_neuron_config_invalid_dtype_setting():
    # Should have cast_type as-declared also set when attention_dtype != torch_dtype
    with pytest.raises(AssertionError):
        NeuronConfig(torch_dtype = torch.bfloat16, attention_dtype = torch.float32) # Exception
    
    NeuronConfig(torch_dtype = torch.bfloat16, attention_dtype = torch.float32, cast_type = 'as-declared') # Works

def test_invalid_cast_type():
    with pytest.raises(ValueError):
        NeuronConfig(cast_type = 'user-defined') # Exception
    
    NeuronConfig(cast_type = 'as-declared') # Works
    NeuronConfig(cast_type = 'config') # Works

def test_valid_cp_dp_configurations():
    NeuronConfig(tp_degree = 32, cp_degree = 4)
    NeuronConfig(tp_degree = 32, attention_dp_degree = 4, batch_size = 4, ctx_batch_size = 1, tkg_batch_size = 4, is_continuous_batching = True)
    NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 4, batch_size = 4, ctx_batch_size = 1, tkg_batch_size = 4, is_continuous_batching = True)
    NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 2, batch_size = 2, ctx_batch_size = 1, tkg_batch_size = 2, is_continuous_batching = True) # cp != dp, cp > dp

def test_invalid_cp_tp_configuration():
    with pytest.raises(ValueError):
        NeuronConfig(tp_degree = 32, cp_degree = 7) # tp % cp != 0

def test_invalid_cp_dp_configuration():
    with pytest.raises(ValueError):
        NeuronConfig(tp_degree = 32, attention_dp_degree = 2, batch_size = 2, ctx_batch_size = 1, tkg_batch_size = 2, is_continuous_batching = True) # only DP set
        NeuronConfig(tp_degree = 32, cp_degree = 7, attention_dp_degree = 7) # tp % cp or tp % dp != 0
        NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 8, batch_size = 2, ctx_batch_size = 1, tkg_batch_size = 2, is_continuous_batching = True) # cp != dp, cp < dp
        NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 2, batch_size = 2, ctx_batch_size = 1, tkg_batch_size = 2) # CB = True not set
        NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 2, batch_size = 2, ctx_batch_size = 2, tkg_batch_size = 5, is_continuous_batching = True) # tkg batch size % dp != 0
        NeuronConfig(tp_degree = 32, cp_degree = 4, attention_dp_degree = 3, batch_size = 3, ctx_batch_size = 1, tkg_batch_size = 3, is_continuous_batching = True) # cp % dp != 0

