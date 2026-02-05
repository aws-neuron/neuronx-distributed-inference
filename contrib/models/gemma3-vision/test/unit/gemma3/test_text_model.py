import logging
from typing import Dict, OrderedDict

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

from gemma3_vision.modeling_gemma3_text import NeuronGemma3TextModel
from test.utils import (
    assert_tensor_all_close, mark_step, cpu_setup, create_neuron_config, causal_mask, window_mask,
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


def test_nxdi_text_model_cpu_vs_transformers_implementation(random_seed, hf_config) -> None:
    model_dtype = torch.float32
    batch_size, seq_len = 2, 32
    hf_config.text_config.sliding_window = 10
    hf_config.text_config.query_pre_attn_scalar = hf_config.text_config.head_dim
    hf_config.text_config.num_hidden_layers = 1 # smaller network for quick testing

    # --- Set NxDI Model ---

    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=seq_len,
        torch_dtype=model_dtype,
        tp_degree=1,
        hf_config=hf_config
    )

    cpu_setup(model_dtype)
    text_model = NeuronGemma3TextModel(config=nrn_config.text_config, optimize_inference=False).to(dtype=model_dtype)
    text_model.kv_mgr = MockKVCacheManager(config=nrn_config.text_config, num_kv_head=nrn_config.text_config.num_key_value_heads)
    text_model.eval()

    # --- Set Transformers Model ---
    reference_model = Gemma3TextModel(hf_config.text_config)
    reference_model.load_state_dict(convert_to_hf_state_dict(text_model.state_dict()), strict=False)
    reference_model.eval()

    # --- Set Inputs ---
    input_ids = torch.randint(0, hf_config.text_config.vocab_size, (batch_size, seq_len)).to(dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(dtype=torch.long)
    seq_ids = torch.arange(batch_size).to(dtype=torch.long)
    attention_mask = causal_mask(batch_size, seq_len).to(dtype=torch.long)
    attention_mask_hf = torch.ones((batch_size, seq_len)).to(dtype=torch.bool)

    with torch.no_grad():
        device = torch.device("cpu")
        ref_last_hidden_state = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask_hf,
            position_ids=position_ids,
            use_cache=None
        ).last_hidden_state

        # pass through lm_head manually as logit calculation happens at a higher model class (Gemma3ForCausalLM) in HF
        lm_head = torch.nn.Linear(hf_config.text_config.hidden_size, hf_config.text_config.vocab_size, bias=False)
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
