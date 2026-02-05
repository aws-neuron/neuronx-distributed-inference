import copy
import logging
from typing import Dict, OrderedDict

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RotaryEmbedding

from gemma3_vision.modeling_gemma3_text import NeuronGemma3DecoderLayer
from test.utils import assert_tensor_all_close, causal_mask, window_mask, mark_step, cpu_setup, create_neuron_config, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("self_attn"):
            splits = key.split(".")
            if len(splits) == 4:
                # q/k/v/o projection
                hf_state_dict[f"self_attn.{splits[-2]}.{splits[-1]}"] = tensor
            else:
                # norm weights
                # in Gemma3RMSNorm, weights are initialized with torch.zeros
                # while Neuron's CustomRMSNorms initializes with torch.ones
                hf_state_dict["self_attn.q_norm.weight"] = torch.zeros_like(tensor)
                hf_state_dict["self_attn.k_norm.weight"] = torch.zeros_like(tensor)
        elif key.find("_layernorm.") != -1:
            hf_state_dict[key] = torch.zeros_like(tensor)
        else:
            hf_state_dict[key] = tensor
    return hf_state_dict


@pytest.mark.parametrize("layer_idx", [0, 5])
def test_nxdi_decoder_layer_cpu_vs_transformers_implementation(random_seed, layer_idx, hf_config) -> None:
    inputs_dtype = model_dtype = torch.float32
    batch_size, max_seq_len = 2, 64
    hf_config.text_config.sliding_window = 10
    hf_config.text_config._attn_implementation = "eager"
    hf_config.text_config.query_pre_attn_scalar = hf_config.text_config.head_dim

    # --- Set NxDI Model ---
    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        torch_dtype=model_dtype,
        tp_degree=1,
        hf_config=hf_config
    )

    cpu_setup(model_dtype)
    decoder_layer = NeuronGemma3DecoderLayer(config=nrn_config.text_config, layer_idx=layer_idx).to(dtype=model_dtype)
    decoder_layer.eval()

    # --- Set Transformers Model ---
    hf_text_config = hf_config.text_config

    reference_model = Gemma3DecoderLayer(hf_text_config, layer_idx=layer_idx)
    reference_model.load_state_dict(convert_to_hf_state_dict(decoder_layer.state_dict()), strict=True)
    reference_model.eval()

    # --- Set Inputs ---
    batch_size, seq_len, hidden_size = 2, 15, hf_text_config.hidden_size
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)

    attention_mask = causal_mask(batch_size, seq_len).to(dtype=inputs_dtype)
    local_mask = None
    if decoder_layer.is_swa_layer:
        local_mask = window_mask(batch_size, seq_len, decoder_layer.sliding_window)

    attention_mask_nrn = local_mask if local_mask is not None else attention_mask
    attention_mask_hf = torch.where(attention_mask_nrn.to(bool), 0.0, torch.finfo(inputs_dtype).min).to(inputs_dtype)

    ## Required only for the reference model
    rotary_emb = Gemma3RotaryEmbedding(config=hf_text_config)
    position_embeddings_global = rotary_emb(hidden_states, position_ids)

    hf_text_config_copy = copy.deepcopy(hf_text_config)
    hf_text_config_copy.rope_theta = hf_text_config_copy.rope_local_base_freq
    hf_text_config_copy.rope_scaling = {"rope_type": "default"}
    rotary_emb_local = Gemma3RotaryEmbedding(config=hf_text_config_copy)
    position_embeddings_local = rotary_emb_local(hidden_states, position_ids)

    with torch.no_grad():
        device = torch.device("cpu")
        ref_output, *_ = reference_model(
            hidden_states=hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask_hf,
            cache_position=torch.arange(0, seq_len) # required for sliding-window layers
        )
        output, *_ = decoder_layer(
            hidden_states=hidden_states.to(device=device),
            attention_mask=attention_mask.to(device=device),
            local_mask=local_mask.to(device=device) if local_mask is not None else None,
            position_ids=position_ids.to(device=device)
        )

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Gemma3 decoder - nxdi (cpu) vs huggingface", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
