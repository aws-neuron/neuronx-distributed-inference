import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipVisionEmbeddings
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_vision_embed(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
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

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    vision_embed = NeuronSiglipVisionEmbeddings(config=config)
    vision_embed.eval()

    with torch.no_grad():
        output_cpu = vision_embed(pixel_values=pixel_values)

        vision_embed = vision_embed.to(device=device)
        mark_step()
        output_nrn = vision_embed(pixel_values=pixel_values.to(device=device))
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Vision embedding outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_vision_embedding_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, num_channels, image_size = 2, 3, 896
    inputs_dtype = model_dtype = torch.float32

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    vision_embed = NeuronSiglipVisionEmbeddings(config=config)
    vision_embed.eval()

    reference_model = SiglipVisionEmbeddings(config=hf_config.vision_config).to(dtype=model_dtype)
    reference_model.load_state_dict(vision_embed.state_dict(), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(pixel_values=pixel_values)
        output = vision_embed(pixel_values=pixel_values)

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Vision embedding outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
