import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipVisionTransformer
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES


def convert_neuron_to_hf_state_dict(neuron_state_dict):
    """Convert Neuron model state dict to HuggingFace compatible format.

    Neuron model structure:
    - encoder.layers.X.self_attn.qkv_proj.{q,k,v}_proj.{weight,bias}
    - encoder.layers.X.self_attn.o_proj.o_proj.{weight,bias}
    - encoder.layers.X.self_attn.rank_util.rank (excluded)

    HuggingFace model structure:
    - encoder.layers.X.self_attn.{q,k,v}_proj.{weight,bias}
    - encoder.layers.X.self_attn.out_proj.{weight,bias}
    """
    hf_state_dict = {}

    for key, value in neuron_state_dict.items():
        # Skip rank_util parameters
        if 'rank_util' in key:
            continue

        # Convert qkv_proj paths
        if '.qkv_proj.q_proj.' in key:
            new_key = key.replace('.qkv_proj.q_proj.', '.q_proj.')
        elif '.qkv_proj.k_proj.' in key:
            new_key = key.replace('.qkv_proj.k_proj.', '.k_proj.')
        elif '.qkv_proj.v_proj.' in key:
            new_key = key.replace('.qkv_proj.v_proj.', '.v_proj.')
        # Convert o_proj paths
        elif '.o_proj.o_proj.' in key:
            new_key = key.replace('.o_proj.o_proj.', '.out_proj.')
        else:
            new_key = key

        hf_state_dict[new_key] = value

    return hf_state_dict


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    ])
def test_vision_transformer(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, num_channels, image_size = 2, 3, 896
    hf_config.vision_config.num_hidden_layers = 3    # lower num_hidden_layers for faster testing
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

    vision_transformer = NeuronSiglipVisionTransformer(config=config)
    vision_transformer.eval()

    with torch.no_grad():
        output_cpu = vision_transformer(pixel_values=pixel_values).last_hidden_state

        vision_transformer = vision_transformer.to(device=device)
        mark_step()
        output_nrn = vision_transformer(pixel_values=pixel_values.to(device=device)).last_hidden_state
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Vision transformer outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_vision_transformer_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, num_channels, image_size = 2, 3, 896
    hf_config.vision_config.num_hidden_layers = 3
    inputs_dtype = model_dtype = torch.float32

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        torch_dtype=model_dtype,
        attn_kernel_enabled=False,  # Otherwise, a NKI kernel is automatically selected due to the sequence length (cannot run on CPU)
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    vision_transformer = NeuronSiglipVisionTransformer(config=config)
    vision_transformer.eval()

    hf_config.vision_config._attn_implementation = "eager"
    reference_model = SiglipVisionTransformer(config=hf_config.vision_config).to(dtype=model_dtype)
    hf_compatible_state_dict = convert_neuron_to_hf_state_dict(vision_transformer.state_dict())
    reference_model.load_state_dict(hf_compatible_state_dict, strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(pixel_values=pixel_values).last_hidden_state
        output = vision_transformer(pixel_values=pixel_values).last_hidden_state

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Vision transformer outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
