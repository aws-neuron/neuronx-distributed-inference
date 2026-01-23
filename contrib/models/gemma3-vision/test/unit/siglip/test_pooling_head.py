
from typing import Dict, OrderedDict

import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipMultiheadAttentionPoolingHead
from test.utils import (
    assert_tensor_all_close,
    mark_step, 
    FP32_TOLERANCES, 
    FP16_TOLERANCES, 
    BF16_TOLERANCES
)

config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
hf_config = AutoModel.from_config(config=config.vision_config).config
# gemma3 does not use head, but setting head to True for unit test
hf_config.vision_use_head = True


def convert_qkv_proj_to_in_proj(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    """
    Merges the separate Q, K, and V projection weights and biases into a single 
    'in_proj' format, as used by PyTorch's native MultiheadAttention layer.
    """
    q_proj_weight, q_proj_bias = state_dict["attention.q_proj.weight"], state_dict["attention.q_proj.bias"]
    k_proj_weight, k_proj_bias = state_dict["attention.k_proj.weight"], state_dict["attention.k_proj.bias"]
    v_proj_weight, v_proj_bias = state_dict["attention.v_proj.weight"], state_dict["attention.v_proj.bias"]
    
    state_dict["attention.in_proj_weight"] = torch.concat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    state_dict["attention.in_proj_bias"] = torch.concat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)

    keys_to_remove = [
        "attention.q_proj.weight", "attention.q_proj.bias",
        "attention.k_proj.weight", "attention.k_proj.bias",
        "attention.v_proj.weight", "attention.v_proj.bias",
    ]

    for key in keys_to_remove:
        del state_dict[key]

    return state_dict


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_pooling_head_layer(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))
    
    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32
    device = xm.xla_device()

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    pooling_head_layer = NeuronSiglipMultiheadAttentionPoolingHead(config=config)
    pooling_head_layer.eval()
    
    with torch.no_grad():
        output_cpu = pooling_head_layer(
            hidden_state=hidden_states,
        )

        pooling_head_layer = pooling_head_layer.to(device=device)
        mark_step()
        output_nrn = pooling_head_layer(
            hidden_state=hidden_states.to(device=device),
        )
        mark_step()
        output_nrn = output_nrn.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Multihead attention pooling head outputs", computed_value=output_nrn, reference_value=output_cpu, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_pooling_head_vs_transformers_implementation(random_seed) -> None:
    batch_size, seq_len, hidden_size = 2, 32, hf_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)

    neuron_config = NeuronSiglipConfig(
        tp_degree=2,
        batch_size=batch_size,
        max_context_length=seq_len,
        torch_dtype=model_dtype,
    )

    config = SiglipInferenceConfig(neuron_config=neuron_config, **hf_config.to_dict())

    pooling_head_layer = NeuronSiglipMultiheadAttentionPoolingHead(config=config)
    pooling_head_layer.eval()

    reference_model = SiglipMultiheadAttentionPoolingHead(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(convert_qkv_proj_to_in_proj(pooling_head_layer.state_dict()), strict=True)
    reference_model.eval()    

    with torch.no_grad():
        ref_output = reference_model(
            hidden_state=hidden_states,
        )
        output = pooling_head_layer(
            hidden_state=hidden_states,
        )

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Multihead attention pooling head outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)

