import os

import pytest
import torch
from torch_neuronx.testing import assert_close

from neuronx_distributed_inference.models.deepseek.modeling_deepseek import DeepseekV3Attention
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp, init_cpu_env

from .test_helper.reference_model import MLA, ModelArgs
from .test_helper.util import (
    STAGE,
    TEST_YARN_ROPE_CONFIG,
    convert_to_reference_ckpt,
    create_context_attn_mask,
    create_dummy_sd,
    create_rope,
    get_reference_cpu_res,
    setup_mini_deepseek_config,
    trace_attention,
)

"""
Compares reference and our attention for prefill and decoding
"""

SEQ_LEN = 3
MODEL_ARGS = ModelArgs()
HIDDEN_DIM = MODEL_ARGS.dim
N_HEADS = MODEL_ARGS.n_heads
DUMMY_CKPT_PATH = "/tmp/dummy_weights.pt"


def _cpu_setup(dtype):
    set_random_seed(0)
    os.environ.setdefault("NXD_CPU_MODE", "1")
    init_cpu_env()
    torch.set_default_dtype(dtype)
    torch.set_default_device("cpu")


@pytest.mark.parametrize("dtype, bsz, max_len",
 # fmt: off
 [
     (torch.float32, 1, 128),
     (torch.float32, 1, 1024),
     (torch.float32, 2, 128)
 ])
def test_attn_cpu_e2e(dtype, bsz, max_len):

    # Setup
    _cpu_setup(dtype)
    dummy_sd = create_dummy_sd(DUMMY_CKPT_PATH, MODEL_ARGS, dtype)

    # Random input
    X = torch.randn(bsz, SEQ_LEN, HIDDEN_DIM, dtype=dtype)
    X_DECODE = torch.randn(bsz, 1, HIDDEN_DIM, dtype=dtype)

    rope_dim = MODEL_ARGS.qk_rope_head_dim

    rope_max_seq_len = TEST_YARN_ROPE_CONFIG["max_seq_len"]
    rotary_emb = create_rope(rope_dim, TEST_YARN_ROPE_CONFIG)
    assert rotary_emb._mscale == 1.0, "default yarn config should produce value of 1 for _mscale."
    freqs_table = torch.randn(rope_max_seq_len, rope_dim//2).to(torch.float32)

    # Get reference CPU results
    mla = MLA(MODEL_ARGS)
    mla.load_state_dict(convert_to_reference_ckpt(dummy_sd, dtype), strict=False)
    prefill_out, decode_out = get_reference_cpu_res(mla, freqs_table, SEQ_LEN, bsz, X, X_DECODE, dtype)
    ref_prefill_res, ref_prefill_pe_cache, ref_prefill_kv_cache = prefill_out
    ref_decode_res, ref_decode_pe_cache, ref_decode_kv_cache = decode_out

    # DeepseekV3Attention cpu
    # Prefill
    config = setup_mini_deepseek_config(1, bsz, max_len, N_HEADS, HIDDEN_DIM, dtype)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(dim=0)
    self_attn = DeepseekV3Attention(config, 0)
    self_attn.load_state_dict(dummy_sd, strict=False)
    cos, sin = rotary_emb(freqs_table, rope_max_seq_len, freqs_table)
    mask = create_context_attn_mask(torch.tensor([[1]*SEQ_LEN]))
    output = self_attn(hidden_states=X,
                       attention_mask=mask,
                       position_ids=position_ids,
                       cos_cache=cos,
                       sin_cache=sin)
    cpu_res, cache_prefill = output[0], output[1]
    test_pe_cache, test_kv_cache = cache_prefill
    torch.testing.assert_close(ref_prefill_res, cpu_res)
    torch.testing.assert_close(ref_prefill_pe_cache, test_pe_cache)
    torch.testing.assert_close(ref_prefill_kv_cache, test_kv_cache)

    # Decoding
    position_ids = torch.tensor([[SEQ_LEN]])
    rope_dim, latent_dim = MODEL_ARGS.qk_rope_head_dim, MODEL_ARGS.kv_lora_rank
    kv_cache_tensor = torch.empty(bsz, SEQ_LEN+1, latent_dim + rope_dim)
    kv_cache_tensor[:, :SEQ_LEN, :rope_dim] = test_pe_cache
    kv_cache_tensor[:, :SEQ_LEN, rope_dim:] = test_kv_cache
    output = self_attn(hidden_states=X_DECODE,
                       attention_mask=torch.tensor([[[[True, True, True, False]]]]),
                       position_ids=position_ids,
                       past_key_value=kv_cache_tensor,
                       cos_cache=cos,
                       sin_cache=sin)
    cpu_res, cache_decode = output[0], output[1]
    test_pe_cache, test_kv_cache = cache_decode
    torch.testing.assert_close(ref_decode_res, cpu_res)
    torch.testing.assert_close(ref_decode_pe_cache, test_pe_cache)
    torch.testing.assert_close(ref_decode_kv_cache, test_kv_cache)

@pytest.mark.parametrize("dtype, tp, bsz, max_len",
 # fmt: off
 [
     (torch.float32, 2, 1, 128),
     (torch.float32, 2, 2, 128),
     (torch.float32, 2, 1, 1024),
 ])
def test_attn_neuron_e2e(dtype, tp, bsz, max_len):

    # Env setup
    destroy_mp()
    os.environ.setdefault("NXD_CPU_MODE", "0")
    set_random_seed(0)

    dummy_sd = create_dummy_sd(DUMMY_CKPT_PATH, MODEL_ARGS, dtype)
    rope_max_seq_len = TEST_YARN_ROPE_CONFIG["max_seq_len"]
    rope_dim = MODEL_ARGS.qk_rope_head_dim
    rotary_emb = create_rope(rope_dim, TEST_YARN_ROPE_CONFIG)
    assert rotary_emb._mscale == 1.0, "default yarn config should produce value of 1 for _mscale."
    freqs_table = torch.randn(rope_max_seq_len, rope_dim//2).to(torch.float32)

    # Random input
    X = torch.randn(bsz, SEQ_LEN, HIDDEN_DIM, dtype=dtype)
    X_DECODE = torch.randn(bsz, 1, HIDDEN_DIM, dtype=dtype)

    # Get reference CPU results
    mla = MLA(MODEL_ARGS)
    mla.load_state_dict(convert_to_reference_ckpt(dummy_sd, dtype), strict=False)
    prefill_out, decode_out = get_reference_cpu_res(mla, freqs_table, SEQ_LEN, bsz, X, X_DECODE, dtype)
    ref_prefill_res, ref_prefill_pe_cache, ref_prefill_kv_cache = prefill_out
    ref_decode_res, ref_decode_pe_cache, ref_decode_kv_cache = decode_out

    config = setup_mini_deepseek_config(tp, bsz, max_len, N_HEADS, HIDDEN_DIM, dtype)


    ################ test scenario: prefill ################
    position_ids = torch.arange(SEQ_LEN).unsqueeze(dim=0)
    cos, sin = rotary_emb(freqs_table, rope_max_seq_len, freqs_table)
    mask = create_context_attn_mask(torch.tensor([[1]*SEQ_LEN]))

    # ours neuron v.s. reference
    inputs = [(X, mask, position_ids, cos, sin),]
    neuron_model = trace_attention(STAGE.PREFILL, inputs, config, tp, dtype, DUMMY_CKPT_PATH)
    neuron_output = neuron_model(*inputs[0])
    neuron_res, neuron_kv_cache = neuron_output[0], neuron_output[1]
    neuron_pe_cache, neuron_kv_cache = neuron_kv_cache

    assert_close(ref_prefill_kv_cache, neuron_kv_cache, rtol=1e-4)
    assert_close(ref_prefill_pe_cache, neuron_pe_cache)
    assert_close(ref_prefill_res, neuron_res, rtol=1e-4)

    # ################ test scenario: decode  ################
    cache_from_prefill = torch.cat((neuron_pe_cache, neuron_kv_cache), dim=-1)
    position_ids = torch.tensor([[SEQ_LEN]])
    attn_mask = torch.tensor([[[[True]*SEQ_LEN]]])

    # ours neuron v.s. reference
    inputs = [(X_DECODE, attn_mask, position_ids, cache_from_prefill, cos, sin),]
    neuron_model = trace_attention(STAGE.DECODE, inputs, config, tp, dtype, DUMMY_CKPT_PATH)
    neuron_output = neuron_model(*(inputs[0]))
    neuron_res, neuron_kv_cache = neuron_output[0], neuron_output[1]
    neuron_pe_cache, neuron_kv_cache = neuron_kv_cache

    assert_close(ref_decode_kv_cache, neuron_kv_cache, rtol=1e-5)
    assert_close(ref_decode_pe_cache, neuron_pe_cache)
    assert_close(ref_decode_res, neuron_res, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])