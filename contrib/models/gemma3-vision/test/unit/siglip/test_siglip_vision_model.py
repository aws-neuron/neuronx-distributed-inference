import logging
import pytest
from typing import Dict, OrderedDict

import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipVisionModel
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    """Convert NeuronSiglipVisionModel state dict to HuggingFace SiglipVisionModel format.

    Key mappings:
    - vision_model.encoder.layers.{i}.self_attn.qkv_proj.{q,k,v}_proj.{weight,bias}
      → vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
    - vision_model.encoder.layers.{i}.self_attn.o_proj.o_proj.{weight,bias}
      → vision_model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}
    - vision_model.encoder.layers.{i}.self_attn.rank_util.rank (skip - internal tracking)
    """
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        if "rank_util.rank" in key:
            # Skip internal rank tracking tensors
            logger.debug(f"Skipping internal key: {key}")
            continue
        elif ".qkv_proj." in key:
            # qkv_proj.q_proj.weight → q_proj.weight
            hf_key = key.replace(".qkv_proj.", ".")
            hf_state_dict[hf_key] = tensor
        elif ".o_proj.o_proj." in key:
            # o_proj.o_proj.weight → out_proj.weight
            hf_key = key.replace(".o_proj.o_proj.", ".out_proj.")
            hf_state_dict[hf_key] = tensor
        else:
            hf_state_dict[key] = tensor
    return hf_state_dict


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    ])
def test_vision_model(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, num_channels, image_size = 2, 3, 896
    hf_config.vision_config.num_hidden_layers = 5    # lower num_hidden_layers for faster testing
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        torch_dtype=model_dtype,
        attn_kernel_enabled=False,  # Otherwise, a NKI kernel is automatically selected due to the sequence length (cannot run on CPU)
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

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


def test_nxdi_vision_model_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, num_channels, image_size = 2, 3, 896
    hf_config.vision_config.num_hidden_layers = 5    # lower num_hidden_layers for faster testing
    inputs_dtype = model_dtype = torch.float32

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        torch_dtype=model_dtype,
        attn_kernel_enabled=False, # Otherwise, a NKI kernel is automatically selected due to the sequence length (cannot run on CPU)
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    vision_model = NeuronSiglipVisionModel(config=config)
    vision_model.eval()

    hf_config.vision_config._attn_implementation = "eager"
    reference_model = SiglipVisionModel(config=hf_config.vision_config).to(dtype=model_dtype)
    reference_model.load_state_dict(convert_to_hf_state_dict(vision_model.state_dict()), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(pixel_values=pixel_values).last_hidden_state
        output = vision_model(pixel_values=pixel_values).last_hidden_state

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Vision model outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
