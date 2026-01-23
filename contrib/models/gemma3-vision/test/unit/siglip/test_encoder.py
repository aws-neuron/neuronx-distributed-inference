
import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.siglip.modeling_siglip import SiglipEncoder

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipEncoder
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES

config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
hf_config = AutoModel.from_config(config=config.vision_config).config
hf_config.num_hidden_layers = 5    # lower num_hidden_layers for faster testing


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_encoder(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))
    
    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

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


def test_nxdi_encoder_vs_transformers_implementation(random_seed) -> None:
    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    encoder = NeuronSiglipEncoder(config=config)
    encoder.eval()

    reference_model = SiglipEncoder(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(encoder.state_dict(), strict=True)
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

