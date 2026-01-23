# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from models.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipVisionModel
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES

config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
hf_config = AutoModel.from_config(config=config.vision_config).config
hf_config.num_hidden_layers = 5    # lower num_hidden_layers for faster testing


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_vision_model(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))
    
    batch_size, num_channels, image_size = 2, 3, 896
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    vision_model = NeuronSiglipVisionModel(config=config)
    vision_model.eval()
    
    with torch.no_grad():
        output_cpu = vision_model(pixel_values=pixel_values).last_hidden_state

        vision_model = vision_model.to(device=device)
        mark_step()
        output_nrn = vision_model(pixel_values=pixel_values.to(device=device)).last_hidden_state
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Vision model outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_vision_model_vs_transformers_implementation(random_seed) -> None:
    batch_size, num_channels, image_size = 2, 3, 896
    inputs_dtype = model_dtype = torch.float32

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    vision_model = NeuronSiglipVisionModel(config=config)
    vision_model.eval()

    reference_model = SiglipVisionModel(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(vision_model.state_dict(), strict=True)
    reference_model.eval()    

    with torch.no_grad():
        ref_output = reference_model(pixel_values=pixel_values).last_hidden_state
        output = vision_model(pixel_values=pixel_values).last_hidden_state

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Vision model outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)

