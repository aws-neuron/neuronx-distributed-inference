import os
import copy
import logging
from typing import Dict, OrderedDict

import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from transformers import AutoConfig, AutoModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3RotaryEmbedding, eager_attention_forward
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp
from neuronx_distributed_inference.models.model_base import NeuronBaseModel

from gemma3_vision.modeling_gemma3_text import NeuronGemma3TextModel
from test.unit.gemma3.test_config import get_gemma3_config
from test.unit.gemma3.utils import causal_mask, window_mask, create_windowed_attn_mask_cte
from test.utils import (
    assert_tensor_all_close, mark_step, cpu_setup, 
    FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES,
    MockKVCacheManager
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_to_hf_state_dict(state_dict: OrderedDict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        if key.find('self_attn.') != -1:
            if key.find("qk_norm.") != -1:
                # in Gemma3RMSNorm, weights are initialized with torch.zeros
                # while Neuron's CustomRMSNorms initializes with torch.ones
                hf_state_dict[key.replace('qk_norm.', 'q_norm.')] = torch.zeros_like(tensor)
                hf_state_dict[key.replace('qk_norm.', 'k_norm.')] = torch.zeros_like(tensor)
            else:
                # q/k/v/o projection weight
                parts = key.split('.')
                del parts[-3]
                key = '.'.join(parts)
                hf_state_dict[key] = tensor
        elif key.find("_layernorm.") != -1 or key == "norm.weight":
            hf_state_dict[key] = torch.zeros_like(tensor)
        else:
            hf_state_dict[key] = tensor
    return hf_state_dict


def test_nxdi_text_model_cpu_vs_transformers_implementation(random_seed) -> None:
    inputs_dtype = model_dtype = torch.float32

    # --- Set NxDI Model ---
    text_config = get_gemma3_config(
        tkg_batch_size=2,
        text_tp_degree=1,
        vision_tp_degree=1,
        text_seq_length=32,
        vision_seq_len=32,
    ).text_config
    text_config.sliding_window = 10
    text_config.num_hidden_layers = 1 # smaller network for quick testing

    cpu_setup(model_dtype)
    text_model = NeuronGemma3TextModel(config=text_config, optimize_inference=False).to(dtype=model_dtype)
    text_model.kv_mgr = MockKVCacheManager(config=text_config, num_kv_head=text_config.num_key_value_heads)
    text_model.eval()

    # --- Set Transformers Model ---
    hf_text_config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config  # nosec B615
    hf_text_config.sliding_window = 10
    hf_text_config.num_hidden_layers = 1

    reference_model = Gemma3TextModel(hf_text_config)
    reference_model.load_state_dict(convert_to_hf_state_dict(text_model.state_dict()), strict=False)
    reference_model.eval()    

    # --- Set Inputs ---
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, hf_text_config.vocab_size, (batch_size, seq_len)).to(dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=inputs_dtype)
    seq_ids = torch.arange(batch_size).to(dtype=inputs_dtype)
    attention_mask = causal_mask(batch_size, seq_len).to(dtype=inputs_dtype)
    attention_mask_hf = torch.where(attention_mask.to(bool), 0.0, torch.finfo(inputs_dtype).min).to(inputs_dtype)

    with torch.no_grad():
        device = torch.device("cpu")
        ref_last_hidden_state = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask_hf,
            position_ids=position_ids,
            use_cache=None
        ).last_hidden_state

        # pass through lm_head manually as logit calculation happens at a higher model class (Gemma3ForCausalLM) in HF
        lm_head = torch.nn.Linear(hf_text_config.hidden_size, hf_text_config.vocab_size, bias=False)
        lm_head.load_state_dict({"weight": text_model.state_dict()["lm_head.weight"]}, strict=True) 
        ref_output = lm_head(ref_last_hidden_state[:, -1:, :]) 

        output, *_ = text_model(
            input_ids=input_ids.to(device=device),
            attention_mask=attention_mask.to(device=device),
            position_ids=position_ids.to(device=device),
            seq_ids=seq_ids.to(device=device),
            sampling_params=None,
            kv_cache=None
        ) # first item is logits when on_device_sampling is off

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    print((ref_output - output).abs().max())
    assert_tensor_all_close(test_objective="Gemma3 text model - nxdi (cpu) vs huggingface", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)