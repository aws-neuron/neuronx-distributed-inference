import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipEncoder

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipEncoder
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES


def convert_neuron_siglip_encoder_state_dict_to_hf(neuron_state_dict: dict) -> dict:
    """
    Convert Neuron SigLIP encoder state dict to HuggingFace format.

    Neuron model has:
    - layers.X.self_attn.qkv_proj.{q,k,v}_proj.{weight,bias}
    - layers.X.self_attn.o_proj.o_proj.{weight,bias}
    - layers.X.self_attn.rank_util.rank (not needed in HF)

    HuggingFace model expects:
    - layers.X.self_attn.{q,k,v}_proj.{weight,bias}
    - layers.X.self_attn.out_proj.{weight,bias}
    """
    hf_state_dict = {}

    for key, value in neuron_state_dict.items():
        # Skip rank_util parameters (not needed in HF)
        if "rank_util" in key:
            continue

        # Convert qkv_proj paths
        if "qkv_proj.q_proj" in key:
            new_key = key.replace("qkv_proj.q_proj", "q_proj")
            hf_state_dict[new_key] = value
        elif "qkv_proj.k_proj" in key:
            new_key = key.replace("qkv_proj.k_proj", "k_proj")
            hf_state_dict[new_key] = value
        elif "qkv_proj.v_proj" in key:
            new_key = key.replace("qkv_proj.v_proj", "v_proj")
            hf_state_dict[new_key] = value
        # Convert o_proj path
        elif "o_proj.o_proj" in key:
            new_key = key.replace("o_proj.o_proj", "out_proj")
            hf_state_dict[new_key] = value
        else:
            # Keep other parameters as-is
            hf_state_dict[key] = value

    return hf_state_dict


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    ])
def test_encoder(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, seq_len, hidden_size = 2, 32, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    encoder = NeuronSiglipEncoder(config=config)
    encoder.eval()

    with torch.no_grad():
        output_cpu = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        encoder = encoder.to(device=device)
        mark_step()
        output_nrn = encoder(
            inputs_embeds=inputs_embeds.to(device=device),
            attention_mask=attention_mask.to(device=device),
        ).last_hidden_state
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Encoder last hidden states", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_encoder_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, seq_len, hidden_size = 2, 32, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    encoder = NeuronSiglipEncoder(config=config)
    encoder.eval()

    hf_config.vision_config._attn_implementation = "eager"
    reference_model = SiglipEncoder(config=hf_config.vision_config).to(dtype=model_dtype)
    hf_state_dict = convert_neuron_siglip_encoder_state_dict_to_hf(encoder.state_dict())
    reference_model.load_state_dict(hf_state_dict, strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        output = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Encoder last hidden states", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
