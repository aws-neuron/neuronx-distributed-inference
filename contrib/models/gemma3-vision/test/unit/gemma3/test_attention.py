
import os
import logging
from typing import Dict, OrderedDict

import pytest
import torch
import torch.nn.functional as F
import torch_xla
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import DynamicCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3RotaryEmbedding, eager_attention_forward
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp
from neuronx_distributed_inference.models.config import NeuronConfig

from gemma3_vision.modeling_gemma3_text import NeuronGemma3Attention, NeuronGemma3TextModel
from gemma3_vision.modeling_causal_lm_gemma3 import TextGemma3InferenceConfig
from test.unit.gemma3.test_config import get_gemma3_config
# from test.unit.gemma3.utils import (
#     create_context_attn_mask, create_windowed_attn_mask_cte,
#     apply_sliding_window_to_hf_attn_mask_with_cache_position,
#     create_simple_attn_mask,
#     causal_mask, window_mask, 
#     create_simple_attn_mask, create_windowed_attn_mask_tkg, 
#     prepare_4d_causal_attention_mask_with_cache_position, apply_sliding_window_to_hf_attn_mask
# )
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


# @pytest.mark.forked
# @pytest.mark.parametrize("tolerances, compiler_flags", [
#     (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
#     (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
#     (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
#     ])
# def test_attention_layer(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
#     monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

#     # --- Input and Configurations ---
#     text_config = get_gemma3_config(
#         tkg_batch_size=2,
#         text_tp_degree=1,
#         vision_tp_degree=1,
#         text_seq_length=64,
#         vision_seq_len=64,
#     ).text_config

#     layer_idx = 5 # global attention layer
#     batch_size, seq_len, hidden_size = 2, 2048, text_config.hidden_size
#     inputs_dtype = model_dtype = torch.float32

#     hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype=inputs_dtype)
#     attention_mask = create_context_attn_mask(batch_size, seq_len).to(dtype=inputs_dtype)
#     position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)

#     # --- CPU Reference Execution ---
#     # Note: We explicitly set 'NXD_CPU_MODE' to force a CPU-only environment.
#     #       This is critical because the module's initialization logic (in
#     #       get_rmsnorm_cls) checks this variable to choose between the
#     #       CPU and Neuron-specific RMSNorm implementations.
#     cpu_setup(model_dtype)
#     cpu_attn_layer = NeuronGemma3Attention(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     cpu_attn_layer.eval()

#     with torch.no_grad():
#         cpu_output, *_ = cpu_attn_layer(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids
#         )

#     # --- Neuron Device Execution ---
#     # Note: Tear down CPU environment and switch to NeuronCore mode
#     destroy_mp()
#     os.environ.setdefault("NXD_CPU_MODE", "0")
#     set_random_seed(0)

#     nrn_attn_layer = NeuronGemma3Attention(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     nrn_attn_layer.eval()

#     with torch.no_grad():
#         device = xm.xla_device()
#         nrn_attn_layer = nrn_attn_layer.to(device=device)
#         mark_step()
#         nrn_output, *_ = nrn_attn_layer(
#             hidden_states=hidden_states.to(device=device),
#             attention_mask=attention_mask.to(device=device),
#             position_ids=position_ids.to(device=device)
#         )
#         mark_step()
#         nrn_output = nrn_output.cpu()

#     rtol, atol = tolerances.rtol, tolerances.atol
#     assert_tensor_all_close(test_objective="Gemma3 global attention - cpu vs neuron", computed_value=nrn_output, reference_value=cpu_output, rtol=rtol, atol=atol, equal_nan=True)


# @pytest.mark.parametrize("tolerances, compiler_flags", [
#     (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
#     ])
# def test_nxdi_attention_context_encode_vs_transformers_eager_attention_forward(random_seed, monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
#     monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

#     inputs_dtype = model_dtype = torch.float32

#     # --- Set NxDI Model ---
#     text_config = get_gemma3_config(
#         tkg_batch_size=2,
#         text_tp_degree=1,
#         vision_tp_degree=1,
#         text_seq_length=64,
#         vision_seq_len=64,
#     ).text_config
    
#     layer_idx = 5 # global attention layer (attention_context_encode is for global attn)
#     global_attn_layer = NeuronGemma3Attention(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     global_attn_layer.eval()
#     global_attn_layer.to(device=xm.xla_device())

#     # --- Set Transformers Model ---
#     hf_text_config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config
#     reference_model = Gemma3Attention(hf_text_config, layer_idx=layer_idx)
#     reference_model.load_state_dict(convert_to_hf_state_dict(global_attn_layer.state_dict()), strict=True)
#     reference_model.eval()    

#     # --- Set Inputs ---
#     batch_size, seq_len = 2, 32
#     Q = torch.randn(batch_size, global_attn_layer.num_attention_heads, seq_len, global_attn_layer.head_dim).to(dtype=inputs_dtype)
#     K = torch.randn(batch_size, global_attn_layer.num_key_value_heads, seq_len, global_attn_layer.head_dim).to(dtype=inputs_dtype)
#     V = torch.randn(batch_size, global_attn_layer.num_key_value_heads, seq_len, global_attn_layer.head_dim).to(dtype=inputs_dtype)
#     attention_mask = create_context_attn_mask(batch_size, seq_len).to(dtype=inputs_dtype)
#     attention_mask_hf = torch.where(attention_mask.to(bool), 0.0, torch.finfo(inputs_dtype).min).to(inputs_dtype)

#     with torch.no_grad():
#         device = xm.xla_device()
#         ref_output, *_ = eager_attention_forward(
#             reference_model,
#             Q, K, V,
#             attention_mask=attention_mask_hf,
#             dropout=0.0,
#             scaling=reference_model.scaling,
#             sliding_window=None,
#         )
#         output, *_ = global_attn_layer.attention_context_encode(
#             Q.to(device=device), 
#             K.to(device=device), 
#             V.to(device=device), 
#             seq_len, batch_size, 
#             attention_mask=attention_mask.to(device=device)
#         )
#         output = output.cpu()

#     rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
#     assert_tensor_all_close(test_objective="attention_context_encode vs eager_attention_forward", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)

from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.utils.testing import init_cpu_env


@pytest.mark.parametrize("layer_idx", [
    0, # sliding
    1, # non-sliding
    ])
def test_nxdi_attn_layer_vs_transformers_implementation_prefill(random_seed, monkeypatch, hf_text_config, layer_idx) -> None:
    # TODO: Move to a fixture
    monkeypatch.setenv("NXD_CPU_MODE", "1")
    init_cpu_env()
    assert cpu_mode() is True
    padding_side = "left" # HuggingFace reference only supports left padding
    bucket_size, sliding_window_size, sliding_window_pattern = 8, 4, 2

    is_swa_layer = (layer_idx + 1) % sliding_window_pattern != 0
    
    hf_text_config.sliding_window = sliding_window_size
    hf_text_config.sliding_window_pattern = sliding_window_pattern
    # Make test faster on CPU
    hf_text_config.num_attention_heads = 2
    hf_text_config.num_key_value_heads = 1
    hf_text_config.head_dim = 2
    hf_text_config.hidden_size = 4

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

    nrn_attn_layer = NeuronGemma3Attention(config=config, layer_idx=layer_idx)
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


# @pytest.mark.parametrize("tolerances, compiler_flags, layer_idx", [
#     # (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"], 0), # sliding
#     (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"], 5), # non-sliding
#     ])
# def test_nxdi_attn_layer_vs_transformers_implementation_token_generation(random_seed, monkeypatch, base_compiler_flags, tolerances, compiler_flags, layer_idx) -> None:
#     monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

#     device = xm.xla_device()
#     inputs_dtype = model_dtype = torch.float32

#     # --- Set NxDI Model ---
#     text_config = get_gemma3_config(
#         tkg_batch_size=2,
#         text_tp_degree=1,
#         vision_tp_degree=1,
#         text_seq_length=2048,
#         vision_seq_len=2048,
#     ).text_config

#     cpu_setup(model_dtype)
#     attn_layer = NeuronGemma3Attention(config=text_config, layer_idx=layer_idx).to(dtype=model_dtype)
#     attn_layer.eval()
#     attn_layer.to(device=xm.xla_device())

#     logger.info(f"[Neuron] layer_idx: {layer_idx}, sliding_window: {attn_layer.sliding_window}")

#     # --- Set Transformers Model ---
#     hf_text_config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config

#     reference_model = Gemma3Attention(hf_text_config, layer_idx=layer_idx)
#     reference_model.load_state_dict(convert_to_hf_state_dict(attn_layer.state_dict()), strict=True)
#     reference_model.eval()    

#     logger.info(f"[Transformers] layer_idx: {layer_idx}, sliding_window: {reference_model.sliding_window}")

#     assert attn_layer.is_sliding == reference_model.is_sliding, "Attention type does not match (sliding vs global)"

#     # --- Set Inputs ---
#     batch_size, hidden_size, past_seen_tokens = 1, 5376, 2000
#     hidden_states = torch.randn(batch_size, 1, hidden_size).to(dtype=inputs_dtype)
#     position_ids = torch.tensor([[past_seen_tokens]], dtype=torch.long).expand(batch_size, 1)
#     cache_position = torch.arange(past_seen_tokens, past_seen_tokens+1)

#     attention_mask = torch.ones(batch_size, 1)
#     attention_mask = create_simple_attn_mask(attention_mask, 1)
#     attention_mask_hf = torch.where(attention_mask.to(bool), 0.0, torch.finfo(inputs_dtype).min).to(inputs_dtype)
    
#     if attn_layer.is_sliding:
#         attention_mask = create_windowed_attn_mask_tkg(
#             attention_mask, 
#             window_size=text_config.sliding_window, 
#             position_ids=position_ids
#         )
#         attention_mask_hf_2d = torch.ones(batch_size, past_seen_tokens + 1)
#         attention_mask_hf = prepare_4d_causal_attention_mask_with_cache_position(
#             attention_mask=attention_mask_hf_2d,
#             sequence_length=1,
#             target_length=past_seen_tokens + 1,
#             cache_position=cache_position,  
#             batch_size=batch_size,
#             dtype=inputs_dtype
#         )
#         attention_mask_hf = apply_sliding_window_to_hf_attn_mask_with_cache_position(
#             attention_mask=attention_mask_hf,
#             sliding_window=text_config.sliding_window,
#             cache_position=cache_position,
#         )

#     ## Required only for the reference model
#     if attn_layer.sliding_window:
#         hf_text_config.rope_theta = hf_text_config.rope_local_base_freq
#         hf_text_config.rope_scaling = {"rope_type": "default"}
#         rotary_emb_local = Gemma3RotaryEmbedding(config=hf_text_config)
#         position_embeddings = rotary_emb_local(hidden_states, position_ids)
#     else:
#         rotary_emb = Gemma3RotaryEmbedding(config=hf_text_config)
#         position_embeddings = rotary_emb(hidden_states, position_ids)

#     # KV cache initialization: we assume this token generation step takes place after the prefill step 
#     key_states = torch.arange(0, past_seen_tokens, dtype=torch.float32)[None, None, :, None]\
#         .expand(batch_size, text_config.num_key_value_heads, -1, text_config.head_dim)
#     value_states = key_states + 1 

#     kv_cache_manager_hf = DynamicCache()
#     kv_cache_manager_hf.update(
#         key_states=key_states,
#         value_states=value_states,
#         layer_idx=layer_idx,
#         cache_kwargs={
#             "sliding_window": hf_text_config.sliding_window,
#         }
#     )

#     past_key_value_nrn = (
#         kv_cache_manager_hf.key_cache[layer_idx].clone().to(device=device), 
#         kv_cache_manager_hf.value_cache[layer_idx].clone().to(device=device)
#     )

#     with torch.no_grad():
#         ref_output, *_ = reference_model(
#             hidden_states=hidden_states,
#             position_embeddings=position_embeddings,
#             attention_mask=attention_mask_hf,
#             past_key_value=kv_cache_manager_hf,
#         )
#         output = attn_layer(
#             hidden_states=hidden_states.to(device=device),
#             attention_mask=attention_mask.to(device=device),
#             position_ids=position_ids.to(device=device),
#             past_key_value=past_key_value_nrn,
#         )

#         output = output.hidden_states.cpu()

#     rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
#     assert_tensor_all_close(test_objective="Gemma3 attention token gen - nxdi vs huggingface", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
