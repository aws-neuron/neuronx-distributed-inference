import pytest
import torch
from transformers import PretrainedConfig

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4 import NeuronLlama4ForCausalLM, Llama4InferenceConfig, Llama4NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_wrapper import IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


LLAMA4_SCOUT_MINIMAL_MOCK_CONFIG = {
  "architectures": [
    "Llama4ForConditionalGeneration"
  ],
  "text_config": {
    "hidden_act": "silu",
    "hidden_size": 5120,
    "max_position_embeddings": 10485760,
    "num_attention_heads": 40,
    "num_experts_per_tok": 1,
    "num_hidden_layers": 48,
    "num_key_value_heads": 8,
    "num_local_experts": 16,
    "output_router_logits": False,
    "pad_token_id": 200018,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "vocab_size": 202048,
  },
  "vision_config": {
    "hidden_size": 1408,
    "image_size": 336,
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 34,
    "patch_size": 14,
    "pixel_shuffle_ratio": 0.5,
    "rope_theta": 10000,
    "vision_output_dim": 4096
  }
}


def generate_text_config(batch_size, seq_len, **kwargs):
    return Llama4NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.float16,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=64,
        cp_degree=1,
        on_device_sampling_config=OnDeviceSamplingConfig(dynamic=True, top_k=1),
        world_size=64,
        capacity_factor=None,
        fused_qkv=False,
        attention_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        cast_type="as-declared",
        logical_neuron_cores=2,
        **kwargs
    )


def generate_vision_config(batch_size, seq_len, **kwargs):
    return Llama4NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.float16,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=64,
        cp_degree=1,
        on_device_sampling_config=OnDeviceSamplingConfig(dynamic=True, top_k=1),
        dp_degree=4,
        world_size=64,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        attn_kernel_enabled=True,
        mlp_kernel_enabled=True,
        enable_bucketing=False,
        logical_neuron_cores=2,
        **kwargs
    )


def generate_mock_llama4_scout(batch_size, seq_len, context_encoding_buckets, token_generation_buckets):
    pretrained_config = PretrainedConfig.from_dict(LLAMA4_SCOUT_MINIMAL_MOCK_CONFIG)
    load_config = load_pretrained_config(hf_config=pretrained_config)
    text_neuron_config = generate_text_config(batch_size=batch_size, seq_len=seq_len,
                                              context_encoding_buckets=context_encoding_buckets,
                                              token_generation_buckets=token_generation_buckets)
    vision_neuron_config = generate_vision_config(batch_size=1, seq_len=seq_len)

    inference_config = Llama4InferenceConfig(text_neuron_config=text_neuron_config,
                                             vision_neuron_config=vision_neuron_config,
                                             load_config=load_config)


    mock_model = NeuronLlama4ForCausalLM("mock_path", inference_config)

    return mock_model


def test_convert_input_dict_to_ordered_tuple():
    batch_size = 4
    seq_len = 8192
    hidden_size = 5120
    mock_model = generate_mock_llama4_scout(batch_size=batch_size,
                                            seq_len=seq_len,
                                            context_encoding_buckets=[seq_len],
                                            token_generation_buckets=[seq_len])

    input_ids = torch.arange(1000, 1000 + seq_len, dtype=torch.int32).repeat(batch_size, 1)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32)
    position_ids = torch.arange(seq_len, dtype=torch.int32).repeat(batch_size, 1)
    seq_ids = torch.arange(batch_size, dtype=torch.int32)
    sampling_params = torch.ones((batch_size, 3), dtype=torch.int32)
    vision_embeddings = torch.zeros(input_ids.shape[0], seq_len, hidden_size, dtype=torch.float32)
    vision_mask = torch.full(size=(input_ids.shape[0], seq_len, 1), fill_value=1, dtype=torch.int32)

    mock_input_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "seq_ids": seq_ids,
        "sampling_params": sampling_params,
        "vision_embeddings": vision_embeddings,
        "vision_mask": vision_mask,
    }

    mock_args = mock_model._convert_input_dict_to_ordered_tuple(mock_input_dict)

    assert len(mock_args) == len(IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS)
    assert torch.allclose(mock_args[0], input_ids)
    assert torch.allclose(mock_args[1], attention_mask)
    assert torch.allclose(mock_args[2], position_ids)
    assert torch.allclose(mock_args[3], seq_ids)
    assert torch.allclose(mock_args[4], sampling_params)
    assert torch.allclose(mock_args[22], vision_embeddings)
    assert torch.allclose(mock_args[23], vision_mask)

    for i in range(5, 22):
        assert mock_args[i].numel() == 0
