import os
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Type
from unittest.mock import patch

import pytest
import torch
from torch.nn import RMSNorm
import torch.nn.functional as F
from transformers import AutoConfig

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
)

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (
    GptOssInferenceConfig, GptOssNeuronConfig, GptOssRMSNormV2Padded, GptOssRMSNormV3PaddedShuffled,
)
from neuronx_distributed_inference.models.gpt_oss.mx_layout_transform import shuffle_hidden_dim, unshuffle_hidden_dim


def test_serialize_deserialize_gpt_oss_inference_config():
    
    neuron_config = GptOssNeuronConfig(
        padded_hidden_size=4096,
        padded_intermediate_size=10240
    )
    
    gpt_oss_inference_config = GptOssInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=1,
        num_local_experts=1,
        num_experts_per_tok=1,
        vocab_size=1000,
        head_dim=64,
        num_attention_heads=1,
        num_key_value_heads=1,
        sliding_window=1024,
        initial_context_length=1024,
        rope_theta=10000.0,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=32.0,
        pad_token_id=0,
    )
    
    # Test that original sizes are preserved and padded sizes are used
    assert gpt_oss_inference_config.original_hidden_size == 3072
    assert gpt_oss_inference_config.original_intermediate_size == 8192
    assert gpt_oss_inference_config.hidden_size == 4096 
    assert gpt_oss_inference_config.intermediate_size == 10240
    assert neuron_config.tp_degree == 1

    deserialized_config = verify_serialize_deserialize(gpt_oss_inference_config, GptOssInferenceConfig)
    assert deserialized_config.original_hidden_size == 3072
    assert deserialized_config.original_intermediate_size == 8192
    assert deserialized_config.hidden_size == 4096
    assert deserialized_config.intermediate_size == 10240
    assert deserialized_config.neuron_config.padded_hidden_size == 4096
    assert deserialized_config.neuron_config.padded_intermediate_size == 10240


def verify_serialize_deserialize(
    config: InferenceConfig, config_cls: Type[InferenceConfig] = InferenceConfig
):
    """Verify that the config is identical after being serialized and deserialized."""
    with tempfile.TemporaryDirectory() as model_path:
        config.save(model_path)
        deserialized_config = config_cls.load(model_path)
        assert config.to_json_string() == deserialized_config.to_json_string()
        return deserialized_config


def _pad_tensor(X, pad_to, pad_value=0):
    """
    Pads dim i to pad_to[i] using constant fill value. Uses RHS padding only.
    """

    padding = []
    for i in reversed(range(len(X.shape))):
        pad_left = 0
        pad_right = pad_to[i] - X.shape[i]
        padding.extend([pad_left, pad_right])

    padding = tuple(padding)

    return F.pad(X, padding, "constant", pad_value)


def _shuffle_weight(x):
    """
    Apply shuffling: view(H/4, 4) -> transpose -> reshape(H)
    """
    hidden_size = x.shape[0]
    return x.view(hidden_size // 4, 4).transpose(-2, -1).reshape(hidden_size)


@pytest.mark.parametrize(
    "batch_size,seq_len", [
        pytest.param(1, 10240),
        pytest.param(1, 1),
        pytest.param(128, 1),
        pytest.param(1, 4),
        pytest.param(128, 4),
    ]
)
@pytest.mark.parametrize(
    "unpadded_size, padded_size", [
        pytest.param(2880, 2880),
        pytest.param(2880, 3072),
        pytest.param(3072, 3072),
    ]
)
def test_rmsnorm_torch_rmsnorm_v3_equivalence(batch_size, seq_len, unpadded_size, padded_size):
    """
    Test that torch.nn.RMSNorm(input) = unshuffle(GptOssRMSNormV3PaddedShuffled(shuffle(input)))
    """
    
    # Input
    input_unpadded = torch.rand((batch_size, seq_len, unpadded_size), dtype=torch.float32)

    # Norm eps / weight
    eps = 1e-6
    weight_unpadded = torch.rand((unpadded_size), dtype=torch.float32)
    
    # Compute golden
    torch_norm = RMSNorm([unpadded_size], eps=eps)
    torch_norm.weight.data = weight_unpadded
    golden_output = torch_norm(input_unpadded)

    # Pad and shuffle inputs, weight
    input_padded = _pad_tensor(input_unpadded.detach().clone(), [batch_size, seq_len, padded_size])
    weight_padded = _pad_tensor(weight_unpadded.detach().clone(), [padded_size])
    input_padded_shuffled = shuffle_hidden_dim(input_padded, dim=-1)
    weight_padded_shuffled = _shuffle_weight(weight_padded)

    # Padded and shuffled norm func
    cpu_padded_shuffled_norm = GptOssRMSNormV3PaddedShuffled(padded_hidden_size=padded_size, unpadded_hidden_size=unpadded_size, eps=eps)
    cpu_padded_shuffled_norm.weight.data = weight_padded_shuffled

    # Compute padded/shuffled output
    cpu_output = cpu_padded_shuffled_norm(input_padded_shuffled)

    # Validate accuracy
    cpu_output_unshuffled_unpadded = unshuffle_hidden_dim(cpu_output, dim=-1)[..., :unpadded_size]
    torch.testing.assert_close(cpu_output_unshuffled_unpadded, golden_output)
    print("Test passes!")


@pytest.mark.parametrize(
    "batch_size,seq_len", [
        pytest.param(1, 10240),
        pytest.param(1, 1),
        pytest.param(128, 1),
        pytest.param(1, 4),
        pytest.param(128, 4),
    ]
)
@pytest.mark.parametrize(
    "unpadded_size, padded_size", [
        pytest.param(2880, 2880),
        pytest.param(2880, 3072),
        pytest.param(3072, 3072),
    ]
)
def test_rmsnorm_v2_rmsnorm_v3_equivalence(batch_size, seq_len, unpadded_size, padded_size):
    """
    Test that GptOssRMSNormV2Padded(input) = unshuffle(GptOssRMSNormV3PaddedShuffled(shuffle(input)))
    """
    
    # Init norms
    eps = 1e-6
    v2_norm = GptOssRMSNormV2Padded(padded_size, unpadded_size, eps)
    v3_norm = GptOssRMSNormV3PaddedShuffled(padded_size, unpadded_size, eps)
    v3_norm.weight.data = _shuffle_weight(v2_norm.weight.data.clone())
    
    # Create input (only unpadded region has non-zero values)
    input_tensor = torch.randn(batch_size, seq_len, padded_size)
    input_tensor[..., unpadded_size:] = 0.0
    
    # V2 path
    v2_output = v2_norm(input_tensor)
    
    # V3 path
    shuffled_input = shuffle_hidden_dim(input_tensor, dim=-1)
    v3_output = v3_norm(shuffled_input)
    unshuffled_output = unshuffle_hidden_dim(v3_output, dim=-1)
    
    # Test accuracy
    torch.testing.assert_close(unshuffled_output, v2_output)
    print("Test passes!")

@pytest.mark.parametrize(
    "batch_size,seq_len", [
        pytest.param(1, 10240),
        pytest.param(1, 1),
        pytest.param(128, 1),
        pytest.param(1, 4),
        pytest.param(128, 4),
    ]
)
@pytest.mark.parametrize(
    "unpadded_size, padded_size", [
        pytest.param(2880, 2880),
        pytest.param(2880, 3072),
        pytest.param(3072, 3072),
    ]
)
def test_rmsnormv3_standalone(batch_size, seq_len, unpadded_size, padded_size):
    """Test GptOssRMSNormV3PaddedShuffled properties: padded regions zeroed, proper normalization"""
    eps = 1e-6
    v3_norm = GptOssRMSNormV3PaddedShuffled(padded_size, unpadded_size, eps)
    
    # Create shuffled input with non-zero values in both padded and unpadded regions
    input_tensor = torch.randn(batch_size, seq_len, padded_size)
    
    output = v3_norm(input_tensor)
    
    # Check padded regions are zero after reshaping back
    *shape, _ = output.shape
    reshaped = output.view(*shape, 4, padded_size // 4)
    assert torch.allclose(reshaped[..., unpadded_size // 4:], torch.zeros_like(reshaped[..., unpadded_size // 4:]))
    
    # Check output dtype matches input dtype
    assert output.dtype == input_tensor.dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnormv3_dtype_preservation(dtype):
    """Test that GptOssRMSNormV3PaddedShuffled preserves input dtype"""
    unpadded_size, padded_size = 16, 20
    v3_norm = GptOssRMSNormV3PaddedShuffled(padded_size, unpadded_size, 1e-6)
    
    input_tensor = torch.randn(2, 5, padded_size, dtype=dtype)
    output = v3_norm(input_tensor)
    
    assert output.dtype == dtype

if __name__ == "__main__":
    pytest.main([__file__, '-v'])