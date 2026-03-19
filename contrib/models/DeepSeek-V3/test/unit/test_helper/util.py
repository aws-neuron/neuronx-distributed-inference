import os
from enum import Enum
from functools import partial

import torch
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Config

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3InferenceConfig,
    custom_compiler_args,
)
from neuronx_distributed_inference.models.deepseek.rope_util import DeepseekV3YarnRotaryEmbedding
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.testing import build_module

TEST_YARN_ROPE_CONFIG = {
    "max_position_embeddings": 128,
    "max_seq_len": 256,
    "beta_fast": 32,
    "beta_slow": 1,
    "rope_theta": 10000.0,
    "factor": 40,
    "mscale": 1,
    "mscale_all_dim": 1,
}

class STAGE(Enum):
    PREFILL = 1,
    DECODE = 2


class PrefillAttnModel(torch.nn.Module):
    """ Prefill Attention Module """

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = DeepseekV3Attention(cfg, 0)

    def forward(self, x, mask, position_ids, cos, sin):
        attn_res = self.self_attn(hidden_states=x, attention_mask=mask, position_ids=position_ids,
                                  cos_cache=cos, sin_cache=sin)
        attn_out, kv = attn_res[0], attn_res[1]
        return attn_out, kv

class DecodeAttnModel(torch.nn.Module):
    """ Decode Attention Module """

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = DeepseekV3Attention(cfg, 0)

    def forward(self, x, mask, position_ids, past_key_value, cos, sin):
        attn_res = self.self_attn(hidden_states=x, attention_mask=mask, position_ids=position_ids,
                                  past_key_value=past_key_value,
                                  cos_cache=cos, sin_cache=sin)
        attn_out, kv = attn_res[0], attn_res[1]
        return attn_out, kv


def create_context_attn_mask(attention_mask):
    # Lower triangle causal mask for classic attention
    batch_size, n_positions = attention_mask.shape
    mask = torch.full(
        (n_positions, n_positions), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)

    expanded_mask = (
        attention_mask[:, None, None, :]
        .expand(batch_size, 1, n_positions, n_positions)
        .to(torch.bool)
    )
    return torch.logical_and(mask, expanded_mask)

def create_dummy_sd(PATH, args, dtype):
    if os.path.exists(PATH):
        os.remove(PATH)
    q_lora_rank = args.q_lora_rank
    kv_lora_rank = args.kv_lora_rank
    dummy_sd = {
        "q_a_proj.weight": torch.randn(q_lora_rank, args.dim).to(dtype), # wq_a
        "kv_a_layernorm.weight": torch.randn(kv_lora_rank,).to(torch.float32), # kv_norm
        "q_a_layernorm.weight": torch.randn(q_lora_rank,).to(torch.float32), # q_norm
        "q_b_proj.weight": torch.randn(192*args.n_heads, q_lora_rank).to(dtype), # wq_b
        "kv_a_proj_with_mqa.weight": torch.randn(576, args.dim).to(dtype), # wkv_a
        "kv_b_proj.weight": torch.randn(256*args.n_heads, kv_lora_rank).to(dtype), # wkv_b
        "o_proj.weight": torch.randn(args.dim, 2048).to(dtype), # wo
    }
    torch.save(dummy_sd, PATH)
    return dummy_sd

def convert_to_reference_ckpt(dummy_weights, dtype):
    test_sd = {}
    mapping = {
        "q_a_proj.weight": "wq_a.weight",
        "kv_a_layernorm.weight": "kv_norm.weight",
        "q_a_layernorm.weight": "q_norm.weight",
        "q_b_proj.weight": "wq_b.weight",
        "kv_a_proj_with_mqa.weight": "wkv_a.weight",
        "kv_b_proj.weight": "wkv_b.weight",
        "o_proj.weight": "wo.weight"
    }
    for k, v in dummy_weights.items():
        if k in mapping.keys():
            test_sd[mapping[k]] = v.to(dtype)
    return test_sd

def get_reference_cpu_res(mla, freqs_table, seq_len, bsz, x_prefill, x_decode, dtype):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cpu")
    freqs_cis_table = torch.polar(torch.ones_like(freqs_table), freqs_table)

    # prefill
    start_pos = 0
    end_pos = start_pos + seq_len
    freqs_cis = freqs_cis_table[start_pos: end_pos]
    ref_prefill_res = mla(x_prefill, start_pos, freqs_cis, torch.full((seq_len, seq_len), float("-inf")).triu_(1))
    ref_prefill_pe_cache = mla.pe_cache[:bsz, :seq_len, :]
    ref_prefill_kv_cache = mla.kv_cache[:bsz, :seq_len, :]
    prefill_out = (ref_prefill_res, ref_prefill_pe_cache, ref_prefill_kv_cache)

    # decode
    start_pos = seq_len
    end_pos = seq_len + 1
    freqs_cis = freqs_cis_table[start_pos: end_pos]
    ref_decode_res = mla(x_decode, start_pos, freqs_cis, None)
    ref_decode_pe_cache = mla.pe_cache[:bsz, start_pos:end_pos, :]
    ref_decode_kv_cache = mla.kv_cache[:bsz, start_pos:end_pos, :]
    decode_out = (ref_decode_res, ref_decode_pe_cache, ref_decode_kv_cache)

    return prefill_out, decode_out

def create_rope(rope_dim, yarn_config):
    return DeepseekV3YarnRotaryEmbedding(
        dim=rope_dim,
        scaling_factor=yarn_config["factor"],
        base=yarn_config["rope_theta"],
        original_max_position_embeddings = yarn_config["max_position_embeddings"],
        max_position_embeddings = yarn_config["max_seq_len"],
        mscale=yarn_config["mscale"],
        mscale_all_dim=yarn_config["mscale_all_dim"],
        beta_fast=yarn_config["beta_fast"],
        beta_slow=yarn_config["beta_slow"],
    )

def setup_mini_deepseek_config(tp_degree, bsz, max_len, n_heads, hidden_dim, dtype):
    hf_cfg = DeepseekV3Config()
    hf_cfg.rope_scaling = TEST_YARN_ROPE_CONFIG
    hf_cfg.hidden_size = hidden_dim
    hf_cfg.num_attention_heads = n_heads
    hf_cfg.num_key_value_heads = n_heads
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=bsz,
        max_context_length=max_len,
        seq_len=max_len,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        torch_dtype=dtype
    )
    return DeepseekV3InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_cfg),
    )

def _convert_to_neuron_test_sd(file, dtype):
    dummy_weights = torch.load(file)
    test_sd = {}
    for k, v in dummy_weights.items():
        if "norm" not in k:
            test_sd["self_attn." + k] = v.to(dtype)
        else:
            test_sd["self_attn." + k] = v.to(torch.float32)
    return test_sd

def trace_attention(stage: STAGE, example_inputs, cfg, tp, dtype, ckpt_path):
    os.environ['NXD_CPU_MODE'] = '0'
    if stage == STAGE.PREFILL:
        model_cls = partial(
            PrefillAttnModel,
            cfg=cfg
        )
    elif stage == STAGE.DECODE:
        model_cls = partial(
            DecodeAttnModel,
            cfg=cfg
        )
    else:
        raise TypeError("Test stage should be either prefill or decode.")
    test_ckpt_path = os.path.join("/tmp/test_mla_ckpt.pt")
    torch.save(_convert_to_neuron_test_sd(ckpt_path, dtype), test_ckpt_path)
    return build_module(model_cls, example_inputs, tp_degree=tp, compiler_args=custom_compiler_args(),
                        checkpoint_path=test_ckpt_path)
