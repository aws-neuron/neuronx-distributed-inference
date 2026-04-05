import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers.models.siglip.modeling_siglip import SiglipMLP

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipMLP
from test.utils import assert_tensor_all_close, mark_step, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_mlp_layer(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    batch_size, seq_len, hidden_size = 2, 32, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    x = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    mlp_layer = NeuronSiglipMLP(config).to(dtype=model_dtype)
    mlp_layer.eval()

    with torch.no_grad():
        cpu_output = mlp_layer(x)

        mlp_layer = mlp_layer.to(device=device)
        mark_step()
        nrn_output = mlp_layer(x.to(device=device))
        mark_step()
        nrn_output = nrn_output.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="MLP outputs", computed_value=nrn_output, reference_value=cpu_output, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_mlp_vs_transformers_implementation(random_seed, hf_config) -> None:
    batch_size, seq_len = 2, 32
    inputs_dtype = model_dtype = torch.float32

    x = torch.randn(batch_size, seq_len, hf_config.vision_config.hidden_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.vision_config.to_dict())

    mlp_layer = NeuronSiglipMLP(config=config).to(dtype=model_dtype)
    mlp_layer.eval()

    reference_model = SiglipMLP(config=hf_config.vision_config).to(dtype=model_dtype)
    reference_model.load_state_dict(mlp_layer.state_dict(), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(hidden_states=x)
        output = mlp_layer(hidden_states=x)

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="MLP outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
