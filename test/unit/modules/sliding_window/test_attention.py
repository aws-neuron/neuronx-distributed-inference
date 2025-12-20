"""
Copyright (c) 2023, Amazon.com. All Rights Reserved
Adapted from https://github.com/aws-neuron/nki-samples/blob/main/test/unit/test_flash_attn_fwd.py
"""
import math
import pytest
from time import perf_counter

import torch
import numpy as np
from torch_xla.core import xla_model as xm
import neuronxcc.nki.language as nl
from neuronxcc.nki import benchmark
from neuronx_distributed_inference.modules.sliding_window.attention import flash_fwd, FlashConfig


torch.manual_seed(0)
bench_func = benchmark(warmup=5, iters=10)(flash_fwd)
 
def causal_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def window_mask(batch_size: int, seq_len: int, window_size: int):
    """create a causal, window attention mask"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
    for i in range(seq_len):
        if i >= window_size:
            mask[i, : i - window_size + 1] = False
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def attn(q, k, v, causal, sliding_window=None, wce=False):
    b, _, s, _ = q.shape
    attn = torch.einsum('bnsd,bntd->bnst', q / math.sqrt(q.shape[-1]), k)
    if wce:
        pmask = torch.ones(sliding_window,sliding_window).triu(diagonal=1)[None, None, :, :].expand(b, 1, sliding_window, sliding_window).bool()
        amask = torch.ones(sliding_window,sliding_window).tril(diagonal=0)[None, None, :, :].expand(b, 1, sliding_window, sliding_window).bool()
        fmask = torch.concat((pmask,amask), dim=3)
        attn = attn.masked_fill(~fmask, float('-inf'))
    else:
        if causal:
            attn = attn.masked_fill(~causal_mask(b,s), float('-inf'))
        if sliding_window > 0:
            attn = attn.masked_fill(~window_mask(b,s,sliding_window), float('-inf'))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    attn = torch.einsum('bnts,bnsd->bntd', attn, v)
    return attn
 
class TestAttention:
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, \
                              sliding_window, mixed_precision, tile_size, kv_heads, should_transpose_v, latency", [
    # Causal vs. Sliding - test sliding window is faster
    [1, 1, 16*1024, 16*1024, 96, torch.bfloat16, True, -1, True, 2048, None, False, 12000], 
    [1, 1, 16*1024, 16*1024, 96, torch.bfloat16, True, 4096, True, 2048, None, False, 10000],
    ])
    def test_flash_attn_fwd_perf(self, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask,
                                 sliding_window, mixed_precision, tile_size, kv_heads, should_transpose_v,latency):
        q = torch.randn(bs, nheads, seqlen_q, d, dtype=dtype)
        k = torch.randn(bs, nheads, seqlen_k, d, dtype=dtype)
        v = torch.randn(bs, nheads, seqlen_k, d, dtype=dtype)

        config = FlashConfig(**{'seq_tile_size':tile_size, 'should_transpose_v':should_transpose_v})
        heads = nheads if kv_heads is None else kv_heads

        device = xm.xla_device()
        q = q.permute(0, 1, 3, 2).to(device) # (bnds)
        k = k.permute(0, 1, 3, 2).to(device)
        v = v.to(device)
        for _ in range(2):  # warmup
            attn_out = flash_fwd[bs, heads](q, k, v, use_causal_mask=use_causal_mask, window_size=(sliding_window-1,-1), config=config)
            print(attn_out.to('cpu'))  # barrier to trigger xla comp. and run

        t0 = perf_counter()
        attn_out = flash_fwd[bs, heads](q, k, v, use_causal_mask=use_causal_mask, window_size=(sliding_window-1,-1), config=config) 
        print(attn_out.to('cpu'))
        actual_latency = (perf_counter() - t0) * 1e6
        assert actual_latency <= latency
    
    @pytest.mark.simulation
    @pytest.mark.parametrize("bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, \
                              sliding_window, tile_size, kv_heads, should_transpose_v, wce", [
    [1, 1, 4096, 4096, 128, torch.bfloat16, True, -1, 2048, None, False, False],  # causal correctness
    [1, 1, 4096, 4096, 128, torch.bfloat16, True, 1024, 2048, None, False, False],  # sliding correctness
    [1, 1, 4096, 4096, 128, torch.bfloat16, False, 1024, 2048, None, False, False],  # test that causal flag is ignored. set to true and do sliding
    [1, 1, 1024, 2048, 128, torch.bfloat16, True, 1024, 2048, None, False, True],  # for wcte and swa
    ])
    def test_flash_attn_fwd_numerical(self, bs, nheads, seqlen_q, seqlen_k, d, dtype, use_causal_mask, 
                                      sliding_window, tile_size, kv_heads, should_transpose_v, wce):
        q = torch.randn(bs, nheads, seqlen_q, d, dtype=dtype)
        k = torch.randn(bs, nheads, seqlen_k, d, dtype=dtype)
        v = torch.randn(bs, nheads, seqlen_k, d, dtype=dtype)

        config = FlashConfig(**{'seq_tile_size':tile_size, 'should_transpose_v':should_transpose_v, 'windowed_context_encoding':wce})
        heads = nheads if kv_heads is None else kv_heads

        attn_gold = attn(q, k, v, use_causal_mask, sliding_window, wce) # (bnsd)

        device = xm.xla_device()
        q = q.permute(0, 1, 3, 2).to(device) # (bnds)
        k = k.permute(0, 1, 3, 2).to(device)
        v = v.to(device)
        attn_nki = flash_fwd[bs, heads](q, k, v, use_causal_mask=use_causal_mask, window_size=(sliding_window-1,-1), config=config)

        torch.testing.assert_close(attn_nki.to('cpu'), attn_gold, atol=1e-2, rtol=1e-2)
