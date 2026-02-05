
import logging
from typing import Dict, OrderedDict

import pytest
import torch
import torch.nn.functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.testing import init_cpu_env
from neuronx_distributed.utils import cpu_mode

from gemma3_vision.modeling_gemma3_text import NeuronGemma3Attention, NeuronGemma3TextModel, get_rmsnorm_cls
from gemma3_vision.modeling_causal_lm_gemma3 import TextGemma3InferenceConfig
from test.utils import (
    assert_tensor_all_close,
    create_cache_position,
    create_hf_attention_mask_4d,
    create_hidden_states,
    create_position_ids,
    create_rope,
    FP32_TOLERANCES,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("qkv_proj."):
            hf_state_dict[key.replace("qkv_proj.", "")] = tensor
        elif key.startswith("o_proj."):
            hf_state_dict["o_proj.weight"] = tensor
        elif key.startswith("q_layernorm."):
            hf_state_dict["q_norm.weight"] = tensor
        elif key.startswith("k_layernorm."):
            hf_state_dict["k_norm.weight"] = tensor
        else:
            logger.info(f"Skipping unexpected input key: {key}")

    return hf_state_dict


@pytest.mark.parametrize("layer_idx", [
    0, # sliding
    1, # non-sliding
    ])
def test_nxdi_attn_layer_vs_transformers_implementation_prefill(random_seed, monkeypatch, hf_config, layer_idx) -> None:
    # TODO: Move to a fixture
    monkeypatch.setenv("NXD_CPU_MODE", "1")
    init_cpu_env()
    assert cpu_mode() is True
    padding_side = "left" # HuggingFace reference only supports left padding
    bucket_size, sliding_window_size, sliding_window_pattern = 8, 4, 2

    is_swa_layer = (layer_idx + 1) % sliding_window_pattern != 0

    hf_text_config = hf_config.text_config
    hf_text_config.sliding_window = sliding_window_size
    hf_text_config.sliding_window_pattern = sliding_window_pattern
    # Make test faster on CPU
    head_dim = 2
    hf_text_config.num_attention_heads = 2
    hf_text_config.num_key_value_heads = 1
    hf_text_config.head_dim = head_dim
    hf_text_config.hidden_size = 4
    hf_text_config._attn_implementation = "eager"
    hf_text_config.query_pre_attn_scalar = head_dim

    attention_mask_2d = torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 1, 1, 1],
                                      [0, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.int32)

    batch_size, max_input_seq_len = attention_mask_2d.shape
    inputs_dtype = model_dtype = torch.float32

    attention_mask_2d = F.pad(attention_mask_2d, (0, bucket_size - max_input_seq_len), "constant", 0)

    position_ids = create_position_ids(attention_mask_2d=attention_mask_2d, is_for_context_encoding=True)
    cache_position = create_cache_position(attention_mask_2d=attention_mask_2d, is_for_context_encoding=True)

    cos, sin = create_rope(position_ids=position_ids, hf_config=hf_text_config)
    hidden_states = create_hidden_states(attention_mask_2d=attention_mask_2d, hf_config=hf_text_config, is_for_context_encoding=True)

    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=batch_size,
        max_context_length=bucket_size,
        seq_len=bucket_size,
        torch_dtype=model_dtype,
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        padding_side=padding_side,
    )

    config = TextGemma3InferenceConfig(
        neuron_config=neuron_config,
        **hf_text_config.to_dict()
        )

    nrn_model = NeuronGemma3TextModel(config=config)

    sliding_window = sliding_window_size if is_swa_layer else None
    rms_norm_cls = get_rmsnorm_cls()
    rms_norm_eps = getattr(config, "rms_norm_eps", None)
    q_norm = rms_norm_cls(config.head_dim, rms_norm_eps) if rms_norm_eps else rms_norm_cls(config.head_dim)
    k_norm = rms_norm_cls(config.head_dim, rms_norm_eps) if rms_norm_eps else rms_norm_cls(config.head_dim)

    nrn_attn_layer = NeuronGemma3Attention(
        config=config,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        sliding_window=sliding_window,
        use_qk_norm=False,
        q_layernorm=q_norm,
        k_layernorm=k_norm,
        rotary_emb=NeuronGemma3Attention.get_rope(config=config, is_swa_layer=is_swa_layer),
        )
    nrn_attn_layer.eval()

    hf_attn_layer = Gemma3Attention(config=hf_text_config, layer_idx=layer_idx).to(dtype=model_dtype)
    hf_attn_layer.load_state_dict(convert_to_hf_state_dict(nrn_attn_layer.state_dict()), strict=True)
    hf_attn_layer.eval()

    # Attention mask creation
    attention_mask_4d_hf = create_hf_attention_mask_4d(
        attention_mask_2d=attention_mask_2d,
        cache_position=cache_position,
        is_for_context_encoding=True,
        dtype=inputs_dtype,
        is_swa_layer=is_swa_layer,
        sliding_window_size=sliding_window_size,
    )

    if not is_swa_layer:
        # Global attention mask
        attention_mask_4d = nrn_model._create_context_attn_mask(
            attention_mask=attention_mask_2d,
        )
    else:
        # Sliding window attention (SWA) mask
        #   Note: As of Neuron 2.26, NeuronBaseModel._create_windowed_attn_mask_cte does not support
        #   left padding we therefore use the HF left-padded mask to create the Neuron attention mask
        attention_mask_4d = (attention_mask_4d_hf == 0)

    with torch.no_grad():
        ref_output, *_ = hf_attn_layer(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask_4d_hf,
            )

        output = nrn_attn_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask_4d,
            cos_cache=cos,
            sin_cache=sin,
            position_ids=position_ids,
        )
        output = output.hidden_states

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Attention outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
