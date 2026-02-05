import logging
import pytest
from typing import Dict, OrderedDict

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipEncoderLayer
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        print(key)
        if key.startswith("self_attn.qkv_proj."):
            hf_state_dict[key.replace("qkv_proj.", "")] = tensor
        elif key.startswith("self_attn.o_proj."):
            hf_state_dict[key.replace("o_proj.o_proj.", "out_proj.")] = tensor
        elif key.endswith("rank"):
            logger.info(f"Skipping neuron-related key: {key}")
        else:
            hf_state_dict[key] = tensor
    return hf_state_dict

config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
hf_config = AutoModel.from_config(config=config.vision_config).config


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    ])
def test_encoder_layer(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    encoder_layer = NeuronSiglipEncoderLayer(config=config)
    encoder_layer.eval()

    with torch.no_grad():
        output_cpu, *_ = encoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        encoder_layer = encoder_layer.to(device=device)
        mark_step()
        output_nrn, *_ = encoder_layer(
            hidden_states=hidden_states.to(device=device),
            attention_mask=attention_mask.to(device=device),
        )
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Encoder layer outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_encoder_layer_vs_transformers_implementation(random_seed) -> None:
    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    encoder_layer = NeuronSiglipEncoderLayer(config=config)
    encoder_layer.eval()

    reference_model = SiglipEncoderLayer(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(convert_to_hf_state_dict(encoder_layer.state_dict()), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output, *_ = reference_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        output, *_ = encoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Encoder layer outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
