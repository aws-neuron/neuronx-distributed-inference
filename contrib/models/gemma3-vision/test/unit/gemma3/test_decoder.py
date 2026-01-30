import os
import copy
import logging
from typing import Dict, OrderedDict

import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RotaryEmbedding, eager_attention_forward
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp
from neuronx_distributed_inference.models.model_base import NeuronBaseModel

from gemma3_vision.modeling_gemma3_text import NeuronGemma3DecoderLayer
from test.unit.gemma3.test_config import get_gemma3_config
from test.unit.gemma3.utils import causal_mask, window_mask, create_windowed_attn_mask_cte
from test.utils import assert_tensor_all_close, mark_step, cpu_setup, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES

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


# 
# @pytest.mark.parametrize("layer_idx", [0, 5])
# @pytest.mark.parametrize("tolerances, compiler_flags", [
#     (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
#     # (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
#     # (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
#     ])
# def test_decoder_layer(monkeypatch, base_compiler_flags, layer_idx, tolerances, compiler_flags) -> None:
#     monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

#     # --- Input and Configurations ---
#     text_config = get_gemma3_config(
#         tkg_batch_size=2,
#         text_tp_degree=1,
#         vision_tp_degree=1,
#         text_seq_length=64,
#         vision_seq_len=64,
#     ).text_config

#     batch_size, seq_len, hidden_size = 2, 2048, text_config.hidden_size
#     inputs_dtype = model_dtype = torch.float32

#     hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
#     attention_mask = causal_mask(batch_size, seq_len).to(dtype=inputs_dtype)
#     position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)

#     sliding_window_pattern = 6
#     is_sliding = bool((layer_idx + 1) % sliding_window_pattern)
#     logger.info(f"layer_idx: {layer_idx}, is_sliding: {is_sliding}")

#     local_mask = None
#     if is_sliding:
#         local_mask = create_windowed_attn_mask_cte(batch_size, attention_mask, text_config).to(dtype=inputs_dtype)
 
#     # --- CPU Reference Execution ---
#     # Note: We explicitly set 'NXD_CPU_MODE' to force a CPU-only environment.
#     #       This is critical because the module's initialization logic (in
#     #       get_rmsnorm_cls) checks this variable to choose between the
#     #       CPU and Neuron-specific RMSNorm implementations.
#     cpu_setup(model_dtype)
#     cpu_decoder_layer = NeuronGemma3DecoderLayer(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     cpu_decoder_layer.eval()

#     with torch.no_grad():
#         cpu_output, *_ = cpu_decoder_layer(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             # local_mask=local_mask,
#             position_ids=position_ids
#         )

#     # --- Neuron Device Execution ---
#     # Note: Tear down CPU environment and switch to NeuronCore mode
#     destroy_mp()
#     os.environ.setdefault("NXD_CPU_MODE", "0")
#     set_random_seed(0)

#     nrn_decoder_layer = NeuronGemma3DecoderLayer(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     nrn_decoder_layer.eval()

#     with torch.no_grad():
#         device = xm.xla_device()
#         nrn_decoder_layer = nrn_decoder_layer.to(device=device)
#         mark_step()
#         nrn_output, *_ = nrn_decoder_layer(
#             hidden_states=hidden_states.to(device=device),
#             attention_mask=attention_mask.to(device=device),
#             local_mask=local_mask.to(device=device) if local_mask else None,
#             position_ids=position_ids.to(device=device)
#         )
#         mark_step()
#         nrn_output = nrn_output.cpu()

#     rtol, atol = tolerances.rtol, tolerances.atol
#     assert_tensor_all_close(test_objective="Gemma3 decoder - cpu vs neuron", computed_value=nrn_output, reference_value=cpu_output, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("layer_idx", [0, 5])
def _test_nxdi_decoder_layer_cpu_vs_transformers_implementation(random_seed, layer_idx) -> None:
    inputs_dtype = model_dtype = torch.float32

    # --- Set NxDI Model ---
    text_config = get_gemma3_config(
        tkg_batch_size=2,
        text_tp_degree=1,
        vision_tp_degree=1,
        text_seq_length=64,
        vision_seq_len=64,
    ).text_config
    text_config.sliding_window = 10

    cpu_setup(model_dtype)
    decoder_layer = NeuronGemma3DecoderLayer(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
    decoder_layer.eval()

    # --- Set Transformers Model ---
    hf_text_config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config  # nosec B615
    hf_text_config.sliding_window = 10

    reference_model = Gemma3DecoderLayer(hf_text_config, layer_idx=layer_idx)
    reference_model.load_state_dict(convert_to_hf_state_dict(decoder_layer.state_dict()), strict=True)
    reference_model.eval()    

    # --- Set Inputs ---
    batch_size, seq_len, hidden_size = 2, 15, 5376
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)

    attention_mask = causal_mask(batch_size, seq_len).to(dtype=inputs_dtype)
    local_mask = None
    if decoder_layer.is_swa_layer:
        local_mask = window_mask(batch_size, seq_len, decoder_layer.sliding_window)
        # local_mask = create_windowed_attn_mask_cte(batch_size, attention_mask, text_config).to(dtype=inputs_dtype)

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


# @pytest.mark.parametrize("tolerances, compiler_flags, layer_idx", [
#     # (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"], 0), # sliding
#     (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"], 5), # non-sliding
#     ])
# def test_nxdi_decoder_layer_vs_transformers_implementation(random_seed, monkeypatch, base_compiler_flags, tolerances, compiler_flags, layer_idx) -> None:
#     monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

#     # --- Set Inputs ---
#     batch_size, seq_len, hidden_size = 2, 15, 5376
#     inputs_dtype = model_dtype = torch.float32

#     hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
#     attention_mask = causal_mask(batch_size, seq_len).to(dtype=inputs_dtype)
#     position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)

#     sliding_window_pattern = 6
#     is_sliding = bool((layer_idx + 1) % sliding_window_pattern)
#     logger.info(f"layer_idx: {layer_idx}, is_sliding: {is_sliding}")
      
#     # --- Set NxDI Model ---
#     text_config = get_gemma3_config(
#         tkg_batch_size=2,
#         text_tp_degree=1,
#         vision_tp_degree=1,
#         text_seq_length=64,
#         vision_seq_len=64,
#     ).text_config

#     decoder_layer = NeuronGemma3DecoderLayer(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     decoder_layer.eval()
#     decoder_layer.to(device=xm.xla_device())

#     # --- Set Transformers Model ---
#     hf_text_config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config
#     reference_model = Gemma3DecoderLayer(hf_text_config, layer_idx=layer_idx)
#     reference_model.load_state_dict(
#         convert_to_hf_state_dict(decoder_layer.state_dict()), strict=True
#     )
#     reference_model.eval()    

#     ## Required only for the reference model
#     rotary_emb = Gemma3RotaryEmbedding(config=hf_text_config)
#     position_embeddings_global = rotary_emb(hidden_states, position_ids)
    
#     hf_text_config_copy = copy.deepcopy(hf_text_config)
#     hf_text_config_copy.rope_theta = hf_text_config_copy.rope_local_base_freq
#     hf_text_config_copy.rope_scaling = {"rope_type": "default"}
#     rotary_emb_local = Gemma3RotaryEmbedding(config=hf_text_config_copy)
#     position_embeddings_local = rotary_emb_local(hidden_states, position_ids)
    
#     # Attention masks preparation
#     local_mask = None
#     if is_sliding:
#         local_mask = create_windowed_attn_mask_cte(batch_size, attention_mask, text_config).to(dtype=inputs_dtype)
        
#     attention_mask_nrn = local_mask if local_mask else attention_mask
#     attention_mask_hf = torch.where(attention_mask_nrn.to(bool), 0.0, torch.finfo(inputs_dtype).min).to(inputs_dtype)

#     with torch.no_grad():
#         device = xm.xla_device()
#         ref_output, *_ = reference_model(
#             hidden_states=hidden_states,
#             position_embeddings_global=position_embeddings_global,
#             position_embeddings_local=position_embeddings_local,
#             attention_mask=attention_mask_hf,
#         )
#         output, *_ = decoder_layer(
#             hidden_states=hidden_states.to(device=device),
#             attention_mask=attention_mask.to(device=device),
#             local_mask=local_mask.to(device=device) if local_mask else None,
#             position_ids=position_ids.to(device=device)
#         )
#         output = output.cpu()

#     rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
#     assert_tensor_all_close(test_objective="Gemma3 decoder - nxdi vs huggingface", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)