"""
Blenderbot NeuronX Port - Encoder-Decoder Model (Part 1: Core Modules)

Ports facebook/blenderbot-3B to NeuronX using the Whisper pattern.
See modeling_blenderbot.py in transformers for the HF reference.

Architecture (facebook/blenderbot-3B):
  - d_model=2560, encoder_layers=2, decoder_layers=24
  - 32 attention heads, head_dim=80, ffn_dim=10240, GELU
  - PRE-LayerNorm (LN before sublayer, residual add after)
  - Final LayerNorm on encoder and decoder outputs
  - Learned positional embeddings (no offset in current HF version)
  - Shared token embeddings, vocab_size=8008, max_position_embeddings=128
"""

import math
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase

from .configuration_blenderbot_neuron import BlenderbotInferenceConfig


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class LayerNorm(nn.LayerNorm):
    """Cast to float32 before LN for precision (Whisper pattern)."""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class BlenderbotMLP(nn.Module):
    """fc1 -> GELU -> fc2. Ref: BlenderbotEncoderLayer/DecoderLayer."""
    def __init__(self, d_model: int, ffn_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, ffn_dim, bias=True, gather_output=False, dtype=dtype)
        self.fc2 = RowParallelLinear(ffn_dim, d_model, bias=True, input_is_parallel=True, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BlenderbotSelfAttention(nn.Module):
    """MHA with optional KV cache. Ref: BlenderbotAttention (self-attn path)."""
    def __init__(self, d_model: int, n_head: int, batch_size: int, seq_len: int,
                 dtype: torch.dtype = torch.float32, kvcache: bool = True):
        super().__init__()
        self.head_dim = d_model // n_head
        tp = parallel_state.get_tensor_model_parallel_group().size()
        self.n_heads = ceil_div(n_head, tp)
        self.q_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.k_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.out_proj = RowParallelLinear(n_head * self.head_dim, d_model, bias=True, input_is_parallel=True, dtype=dtype)
        # KV cache in float32 for torch.scatter compatibility (see learnings §7)
        self.cache_k = nn.Parameter(torch.zeros(batch_size, self.n_heads, seq_len, self.head_dim, dtype=torch.float32), requires_grad=False) if kvcache else None
        self.cache_v = nn.Parameter(torch.zeros(batch_size, self.n_heads, seq_len, self.head_dim, dtype=torch.float32), requires_grad=False) if kvcache else None

    def forward(self, x: Tensor, last_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.cache_k is not None:
            if seq_len > 1:
                indices = torch.arange(seq_len, dtype=torch.int64, device=q.device).view(1, 1, seq_len, 1).expand(bsz, self.n_heads, seq_len, self.head_dim)
            else:
                indices = last_pos.view(bsz, 1, 1, 1).expand_as(k).to(torch.int64)
            updated_kcache = torch.scatter(self.cache_k, 2, indices, k.float())
            updated_vcache = torch.scatter(self.cache_v, 2, indices, v.float())
            k = updated_kcache.to(q.dtype)
            v = updated_vcache.to(q.dtype)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = torch.where(mask, scores, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        if self.cache_k is not None:
            return self.out_proj(output), updated_kcache, updated_vcache
        return self.out_proj(output)


class BlenderbotCrossAttention(nn.Module):
    """Cross-attention: Q from decoder, K/V from encoder. Cache populated once at prefill."""
    def __init__(self, d_model: int, n_head: int, batch_size: int, kv_seq_len: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.head_dim = d_model // n_head
        tp = parallel_state.get_tensor_model_parallel_group().size()
        self.n_heads = ceil_div(n_head, tp)
        self.q_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.k_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(d_model, n_head * self.head_dim, bias=True, gather_output=False, dtype=dtype)
        self.out_proj = RowParallelLinear(n_head * self.head_dim, d_model, bias=True, input_is_parallel=True, dtype=dtype)
        self.cache_k = nn.Parameter(torch.zeros(batch_size, self.n_heads, kv_seq_len, self.head_dim, dtype=torch.float32), requires_grad=False)
        self.cache_v = nn.Parameter(torch.zeros(batch_size, self.n_heads, kv_seq_len, self.head_dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x: Tensor, xa: Tensor, cross_attn_mask: Optional[Tensor] = None):
        """Cross-attention: Q from decoder hidden, K/V from encoder output.

        Always computes K/V from xa and updates cache. This ensures xa remains
        alive in the Neuron trace graph during decode (avoids dead code elimination
        that would drop the encoder output input tensor).

        Args:
            cross_attn_mask: [bsz, 1, 1, kv_seq_len] bool mask. True = attend, False = mask out.
        """
        bsz, seq_len, _ = x.shape
        kv_seq_len = xa.shape[1]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(xa).view(bsz, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(xa).view(bsz, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Always update cache from xa (during decode this re-populates with same values)
        indices = torch.arange(kv_seq_len, dtype=torch.int64, device=q.device).view(1, 1, kv_seq_len, 1).expand(bsz, self.n_heads, kv_seq_len, self.head_dim)
        updated_kcache = torch.scatter(self.cache_k, 2, indices, k.float())
        updated_vcache = torch.scatter(self.cache_v, 2, indices, v.float())
        k = updated_kcache.to(q.dtype)
        v = updated_vcache.to(q.dtype)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if cross_attn_mask is not None:
            scores = torch.where(cross_attn_mask, scores, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(output), updated_kcache, updated_vcache


class BlenderbotEncoderLayer(nn.Module):
    """PRE-LayerNorm encoder layer: LN -> self_attn -> residual, LN -> FFN -> residual."""
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, batch_size: int, seq_len: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.self_attn = BlenderbotSelfAttention(d_model, n_head, batch_size, seq_len, dtype=dtype, kvcache=False)
        self.self_attn_layer_norm = LayerNorm(d_model)
        self.mlp = BlenderbotMLP(d_model, ffn_dim, dtype=dtype)
        self.final_layer_norm = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = residual + x
        residual = x
        x = self.mlp(self.final_layer_norm(x))
        x = residual + x
        return x


class BlenderbotDecoderLayer(nn.Module):
    """PRE-LayerNorm decoder layer: self_attn + cross_attn + FFN, each with LN->sublayer->residual."""
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, batch_size: int,
                 dec_seq_len: int, enc_seq_len: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.self_attn = BlenderbotSelfAttention(d_model, n_head, batch_size, dec_seq_len, dtype=dtype, kvcache=True)
        self.self_attn_layer_norm = LayerNorm(d_model)
        self.encoder_attn = BlenderbotCrossAttention(d_model, n_head, batch_size, enc_seq_len, dtype=dtype)
        self.encoder_attn_layer_norm = LayerNorm(d_model)
        self.mlp = BlenderbotMLP(d_model, ffn_dim, dtype=dtype)
        self.final_layer_norm = LayerNorm(d_model)

    def forward(self, x: Tensor, xa: Tensor, last_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None, cross_attn_mask: Optional[Tensor] = None):
        # Self-attention
        residual = x
        h, sk, sv = self.self_attn(self.self_attn_layer_norm(x), last_pos=last_pos, mask=mask)
        x = residual + h
        # Cross-attention
        residual = x
        h, ck, cv = self.encoder_attn(self.encoder_attn_layer_norm(x), xa, cross_attn_mask=cross_attn_mask)
        x = residual + h
        # FFN
        residual = x
        x = residual + self.mlp(self.final_layer_norm(x))
        return x, sk, sv, ck, cv
