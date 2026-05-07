import logging
import pytest
from typing import Dict, OrderedDict

import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipAttention

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipAttention
from test.utils import (
    assert_tensor_all_close,
    mark_step,
    FP32_TOLERANCES,
    FP16_TOLERANCES,
    BF16_TOLERANCES
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("qkv_proj."):
            hf_state_dict[key.replace("qkv_proj.", "")] = tensor
        elif key.startswith("o_proj."):
            hf_state_dict[key.replace("o_proj.o_proj.", "out_proj.")] = tensor
        else:
            logger.info(f"Skipping unexpected input key: {key}")
    return hf_state_dict


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_attention_layer(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, seq_len, hidden_size = 2, 32, hf_config.vision_config.hidden_size
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

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    attn_layer = NeuronSiglipAttention(config=config)
    attn_layer.eval()

    with torch.no_grad():
        output_cpu, *_ = attn_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        attn_layer = attn_layer.to(device=device)
        mark_step()
        output_nrn, *_ = attn_layer(
            hidden_states=hidden_states.to(device=device),
            attention_mask=attention_mask.to(device=device),
        )
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Attention outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


# Note: As HuggingFace Transformers supports left padding only, we can only test the NxDI implementation of the attention layer
# and therefore the SWA implementation, for left padding only
def test_nxdi_attn_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, seq_len, hidden_size = 2, 32, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    attn_layer = NeuronSiglipAttention(config=config)
    attn_layer.eval()

    hf_config.vision_config._attn_implementation = "eager"
    reference_model = SiglipAttention(config=hf_config.vision_config).to(dtype=model_dtype)
    reference_model.load_state_dict(convert_to_hf_state_dict(attn_layer.state_dict()), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output, *_ = reference_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        output, *_ = attn_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Attention outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
