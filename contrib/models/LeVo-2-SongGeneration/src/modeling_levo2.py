# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LeVo 2 (SongGeneration v2) -- Neuron-optimized text-to-music pipeline.

Supports two model variants:
  - v2-medium: 2.83B params (dim=1536, 12 heads, 28 primary layers)
  - v2-large:  5.12B params (dim=2048, 16 heads, 36 primary layers)

Three-stage architecture:
  1. LeLM: Dual-Llama autoregressive language model (primary + secondary)
     with on-device KV cache via ModelBuilder. Generates codec tokens [B, 3, T].
  2. Diffusion: GPT2-RoPE CFM backbone (16L) with Euler ODE solver. Converts
     codec tokens to latents [B, 64, T] via RVQ dequantization + denoising.
  3. VAE: Stable Audio decoder. Converts latents to stereo 48kHz audio.

The LeLM transformers use on-device KV cache (torch.scatter for in-HBM updates)
compiled via neuronx_distributed.ModelBuilder. The GPT2 and VAE are compiled via
torch_neuronx.trace().

Key differences from v1 SongGeneration:
  - PREFILL_LEN: 602 -> 952 (description=600 + prompt_audio=252 + type_info=100)
  - rope_theta: 100000 -> 500000 for both primary and secondary
  - version='v2' for model loading (adds musicality tokens)
  - Configurable batch size (B=1..N, model uses 2*B for CFG)
  - Two model variants (v2-medium, v2-large) via LeVo2Config

Reference: https://huggingface.co/lglg666/SongGeneration-v2-medium
"""

import os
import sys
import time
import math
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LeVo2Config:
    """Configuration for LeVo 2 Neuron pipeline."""

    # Model variant
    variant: str = "v2-medium"

    # Model paths (must be set before compile/load)
    model_path: str = ""
    config_path: str = ""
    safetensors_path: str = ""
    prompt_path: str = ""

    # LeLM architecture
    dim: int = 1536
    num_heads: int = 12
    head_dim: int = 128
    primary_layers: int = 28
    secondary_layers: int = 12
    vocab_size: int = 16385
    code_depth: int = 3
    primary_rope_theta: float = 500000.0
    secondary_rope_theta: float = 500000.0

    # Diffusion architecture (shared across variants)
    gpt2_hidden_size: int = 2200
    gpt2_num_layers: int = 16
    gpt2_num_heads: int = 20

    # VAE
    latent_dim: int = 64
    sample_rate: int = 48000
    samples_per_frame: int = 1920

    # Compilation
    max_seq_len: int = 2048
    batch_size: int = 2  # CFG doubles batch: real_batch * 2
    real_batch: int = 1  # Actual number of songs per generate() call
    prefill_len: int = (
        952  # v2 prepend: description(600) + prompt_audio(252) + type_info(100)
    )
    compiler_args: str = "--auto-cast matmult --model-type transformer"
    tp_degree: int = 1

    # Generation defaults
    default_duration_sec: float = 5.0
    default_genre: str = "Pop"
    default_temp: float = 1.0
    default_top_k: int = 5000
    default_cfg_coef: float = 3.0
    default_num_diffusion_steps: int = 10
    default_guidance_scale: float = 1.5

    # Codeclm source path (on the instance)
    codeclm_path: str = "/mnt/models/songgeneration"

    @classmethod
    def v2_medium(cls, **kwargs) -> "LeVo2Config":
        """Create config for v2-medium (2.83B params)."""
        defaults = dict(
            variant="v2-medium",
            dim=1536,
            num_heads=12,
            head_dim=128,
            primary_layers=28,
            secondary_layers=12,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def v2_large(cls, **kwargs) -> "LeVo2Config":
        """Create config for v2-large (5.12B params)."""
        defaults = dict(
            variant="v2-large",
            dim=2048,
            num_heads=16,
            head_dim=128,
            primary_layers=36,
            secondary_layers=12,
        )
        defaults.update(kwargs)
        return cls(**defaults)


# ============================================================================
# CUDA -> CPU patches (required because upstream code assumes CUDA)
# ============================================================================


def _patch_cuda_to_cpu():
    """Redirect all CUDA calls to CPU. Required for the upstream codeclm codebase."""

    def _cpu_cuda(self, *a, **k):
        return self

    torch.Tensor.cuda = _cpu_cuda
    torch.cuda.is_available = lambda: False
    nn.Module.cuda = lambda self, *a, **k: self
    _orig_to = nn.Module.to
    _dev = torch.device

    def _patched_to(self, *a, **k):
        na = []
        for x in a:
            if isinstance(x, str) and "cuda" in x:
                x = "cpu"
            elif isinstance(x, _dev) and x.type == "cuda":
                x = torch.device("cpu")
            na.append(x)
        if "device" in k:
            d = k["device"]
            if isinstance(d, str) and "cuda" in d:
                k["device"] = "cpu"
            elif isinstance(d, _dev) and d.type == "cuda":
                k["device"] = torch.device("cpu")
        return _orig_to(self, *na, **k)

    nn.Module.to = _patched_to


# ============================================================================
# RoPE helper (Neuron-compatible, no complex numbers)
# ============================================================================


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================================
# On-Device KV Cache Wrappers
# ============================================================================


class _NeuronPrimaryTransformer(nn.Module):
    """Primary Llama transformer with on-device KV cache.

    KV cache is stored as registered buffers and updated via torch.scatter,
    keeping the cache in Neuron HBM without PCIe round-trips.
    Supports both prefill (seq_len > 1) and decode (seq_len = 1).
    """

    def __init__(self, causal_lm, config: LeVo2Config):
        super().__init__()
        self.num_layers = config.primary_layers
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.tp_degree = config.tp_degree
        self.num_heads_per_rank = config.num_heads // config.tp_degree
        self.hidden_dim_per_rank = self.num_heads_per_rank * config.head_dim

        self.layers = nn.ModuleList(list(causal_lm.model.layers))
        self.norm = causal_lm.model.norm
        self.lm_head = causal_lm.lm_head

        inv_freq = 1.0 / (
            config.primary_rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        t = torch.arange(config.max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=True)
        self.register_buffer("rope_sin", emb.sin(), persistent=True)

        for i in range(self.num_layers):
            self.register_buffer(
                f"cache_k_{i}",
                torch.zeros(
                    config.batch_size,
                    self.num_heads_per_rank,
                    config.max_seq_len,
                    config.head_dim,
                ),
            )
            self.register_buffer(
                f"cache_v_{i}",
                torch.zeros(
                    config.batch_size,
                    self.num_heads_per_rank,
                    config.max_seq_len,
                    config.head_dim,
                ),
            )

    def forward(self, inputs_embeds, position_ids, cache_position, attn_mask):
        hidden_states = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        num_heads = self.num_heads_per_rank
        head_dim = self.head_dim
        hidden_dim = self.hidden_dim_per_rank
        cos_pos = self.rope_cos[position_ids].unsqueeze(1)
        sin_pos = self.rope_sin[position_ids].unsqueeze(1)

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = getattr(self, f"cache_k_{i}")
            v_cache = getattr(self, f"cache_v_{i}")

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            bsz = hidden_states.size(0)

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, seq_len, num_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(
                1, 2
            )
            value_states = value_states.view(
                bsz, seq_len, num_heads, head_dim
            ).transpose(1, 2)

            query_states = (query_states * cos_pos) + (
                _rotate_half(query_states) * sin_pos
            )
            key_states = (key_states * cos_pos) + (_rotate_half(key_states) * sin_pos)

            idx = cache_position.view(1, 1, seq_len, 1).expand(
                bsz, num_heads, seq_len, head_dim
            )
            setattr(self, f"cache_k_{i}", torch.scatter(k_cache, 2, idx, key_states))
            setattr(self, f"cache_v_{i}", torch.scatter(v_cache, 2, idx, value_states))

            k_cache = getattr(self, f"cache_k_{i}")
            v_cache = getattr(self, f"cache_v_{i}")

            attn_weights = torch.matmul(query_states, k_cache.transpose(2, 3)) / (
                head_dim**0.5
            )
            attn_weights = attn_weights + attn_mask
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, v_cache)

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .reshape(bsz, seq_len, hidden_dim)
            )
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        return hidden_states, logits


class _NeuronFusedSecondary(nn.Module):
    """Fused secondary transformer + MLP bridge + output heads.

    On-device KV cache via torch.scatter, same pattern as primary.
    Supports both prefill (seq_len > 1) and decode (seq_len = 1).
    """

    def __init__(self, causal_lm, mlp_bridge, output_linears, config: LeVo2Config):
        super().__init__()
        self.num_layers = config.secondary_layers
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.code_depth = config.code_depth
        self.tp_degree = config.tp_degree
        self.num_heads_per_rank = config.num_heads // config.tp_degree
        self.hidden_dim_per_rank = self.num_heads_per_rank * config.head_dim

        self.mlp_bridge = mlp_bridge
        self.layers = nn.ModuleList(list(causal_lm.model.layers))
        self.norm = causal_lm.model.norm
        self.output_linears = nn.ModuleList(list(output_linears))

        inv_freq = 1.0 / (
            config.secondary_rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        t = torch.arange(config.max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=True)
        self.register_buffer("rope_sin", emb.sin(), persistent=True)

        for i in range(self.num_layers):
            self.register_buffer(
                f"cache_k_{i}",
                torch.zeros(
                    config.batch_size,
                    self.num_heads_per_rank,
                    config.max_seq_len,
                    config.head_dim,
                ),
            )
            self.register_buffer(
                f"cache_v_{i}",
                torch.zeros(
                    config.batch_size,
                    self.num_heads_per_rank,
                    config.max_seq_len,
                    config.head_dim,
                ),
            )

    def forward(
        self, fused_input2, primary_hidden, position_ids, cache_position, attn_mask
    ):
        bridge_input = torch.cat([fused_input2, primary_hidden], dim=-1)
        hidden_states = self.mlp_bridge(bridge_input)
        seq_len = fused_input2.shape[1]
        num_heads = self.num_heads_per_rank
        head_dim = self.head_dim
        hidden_dim = self.hidden_dim_per_rank

        cos_pos = self.rope_cos[position_ids].unsqueeze(1)
        sin_pos = self.rope_sin[position_ids].unsqueeze(1)

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = getattr(self, f"cache_k_{i}")
            v_cache = getattr(self, f"cache_v_{i}")

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            bsz = hidden_states.size(0)

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, seq_len, num_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(
                1, 2
            )
            value_states = value_states.view(
                bsz, seq_len, num_heads, head_dim
            ).transpose(1, 2)

            query_states = (query_states * cos_pos) + (
                _rotate_half(query_states) * sin_pos
            )
            key_states = (key_states * cos_pos) + (_rotate_half(key_states) * sin_pos)

            idx = cache_position.view(1, 1, seq_len, 1).expand(
                bsz, num_heads, seq_len, head_dim
            )
            setattr(self, f"cache_k_{i}", torch.scatter(k_cache, 2, idx, key_states))
            setattr(self, f"cache_v_{i}", torch.scatter(v_cache, 2, idx, value_states))

            k_cache = getattr(self, f"cache_k_{i}")
            v_cache = getattr(self, f"cache_v_{i}")

            attn_weights = torch.matmul(query_states, k_cache.transpose(2, 3)) / (
                head_dim**0.5
            )
            attn_weights = attn_weights + attn_mask
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, v_cache)

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .reshape(bsz, seq_len, hidden_dim)
            )
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)

        res_logits = torch.stack(
            [
                self.output_linears[k](hidden_states).float()
                for k in range(self.code_depth - 1)
            ],
            dim=1,
        )
        return res_logits


# ============================================================================
# GPT2-RoPE Neuron Wrappers (for diffusion backbone)
# ============================================================================


def _precompute_freqs_sincos(dim, end, constant=10000.0):
    freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary_emb_sincos(xq, cos_vals, sin_vals):
    x_even = xq.float()[..., 0::2]
    x_odd = xq.float()[..., 1::2]
    cos_v = cos_vals[: xq.shape[1]].unsqueeze(0).unsqueeze(2)
    sin_v = sin_vals[: xq.shape[1]].unsqueeze(0).unsqueeze(2)
    out_even = x_even * cos_v - x_odd * sin_v
    out_odd = x_even * sin_v + x_odd * cos_v
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2).type_as(xq)


class _NeuronGPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        from transformers.pytorch_utils import Conv1D

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, attention_mask=None, rope_cos=None, rope_sin=None):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        def split_heads(t):
            return t.view(t.size()[:-1] + (self.num_heads, self.head_dim)).permute(
                0, 2, 1, 3
            )

        query, key, value = split_heads(query), split_heads(key), split_heads(value)

        query = _apply_rotary_emb_sincos(
            query.transpose(1, 2), rope_cos, rope_sin
        ).transpose(1, 2)
        key = _apply_rotary_emb_sincos(
            key.transpose(1, 2), rope_cos, rope_sin
        ).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(float(value.size(-1)))
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1).type(value.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(attn_output.size(0), -1, self.embed_dim)
        )
        return self.resid_dropout(self.c_proj(attn_output))


class _NeuronGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = _NeuronGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        from transformers.pytorch_utils import Conv1D
        from transformers.activations import ACT2FN

        self.mlp_c_fc = Conv1D(inner_dim, hidden_size)
        self.mlp_c_proj = Conv1D(hidden_size, inner_dim)
        self.mlp_act = ACT2FN[config.activation_function]
        self.mlp_dropout = nn.Dropout(0.0)

        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        time_step_emb=None,
        rope_cos=None,
        rope_sin=None,
    ):
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + time_step_emb.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states) * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states.squeeze(1)
        attn_output = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = attn_output * gate_msa + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = self.mlp_act(self.mlp_c_fc(hidden_states))
        hidden_states = self.mlp_dropout(self.mlp_c_proj(hidden_states))
        return hidden_states * gate_mlp + residual


class _NeuronTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, flow_t_size=512):
        super().__init__()
        self.flow_t_size = flow_t_size
        from diffusers.models.embeddings import TimestepEmbedding

        self.timestep_embedder = TimestepEmbedding(
            in_channels=flow_t_size, time_embed_dim=hidden_size
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def timestep_embedding(self, timesteps, max_period=10000, scale=1000):
        half = self.flow_t_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=timesteps.device).float()
            / half
        )
        args = timesteps[:, None].float() * freqs[None] * scale
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.flow_t_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timestep, hidden_dtype):
        timesteps_proj = self.timestep_embedding(timestep)
        embedded_timestep = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype)
        )
        adaln_params = self.linear(self.silu(embedded_timestep))
        return adaln_params, embedded_timestep


class _NeuronGPT2Model(nn.Module):
    """Neuron-traceable GPT2 backbone for CFM diffusion."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList(
            [_NeuronGPT2Block(config, layer_idx=i) for i in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.embed_dim) / self.embed_dim**0.5
        )
        self.timestep_emb = _NeuronTimestepEmbedding(self.embed_dim)

    def forward(self, inputs_embeds, attention_mask, time_step):
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )
        hidden_states = inputs_embeds + self.wpe(position_ids)

        head_dim = self.embed_dim // self.h[0].attn.num_heads
        rope_cos, rope_sin = _precompute_freqs_sincos(dim=head_dim, end=seq_len)
        rope_cos = rope_cos.to(device=device, dtype=hidden_states.dtype)
        rope_sin = rope_sin.to(device=device, dtype=hidden_states.dtype)

        processed_mask = (
            1.0 - attention_mask.to(dtype=hidden_states.dtype)
        ) * torch.finfo(hidden_states.dtype).min

        adaln_params, embedded_timestep = self.timestep_emb(
            time_step, hidden_dtype=hidden_states.dtype
        )

        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=processed_mask,
                time_step_emb=adaln_params,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        hidden_states = self.ln_f(hidden_states) * (1 + scale) + shift
        return self.proj_out(hidden_states)


def _load_gpt2_weights(neuron_model, original_model):
    """Copy weights from original GPT2Model to _NeuronGPT2Model."""
    state = {}
    orig_sd = original_model.state_dict()

    for key in [
        "wte.weight",
        "wpe.weight",
        "ln_f.weight",
        "ln_f.bias",
        "proj_out.weight",
        "proj_out.bias",
        "scale_shift_table",
    ]:
        state[key] = orig_sd[key]

    for orig_key, new_key in [
        (
            "adaln_single.emb.timestep_embedder.linear_1.weight",
            "timestep_emb.timestep_embedder.linear_1.weight",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_1.bias",
            "timestep_emb.timestep_embedder.linear_1.bias",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "timestep_emb.timestep_embedder.linear_2.weight",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_2.bias",
            "timestep_emb.timestep_embedder.linear_2.bias",
        ),
        ("adaln_single.linear.weight", "timestep_emb.linear.weight"),
        ("adaln_single.linear.bias", "timestep_emb.linear.bias"),
    ]:
        state[new_key] = orig_sd[orig_key]

    for i in range(neuron_model.num_layers):
        p = f"h.{i}."
        for suffix in ["ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"]:
            state[p + suffix] = orig_sd[p + suffix]
        for suffix in [
            "attn.c_attn.weight",
            "attn.c_attn.bias",
            "attn.c_proj.weight",
            "attn.c_proj.bias",
        ]:
            state[p + suffix] = orig_sd[p + suffix]
        state[p + "mlp_c_fc.weight"] = orig_sd[p + "mlp.c_fc.weight"]
        state[p + "mlp_c_fc.bias"] = orig_sd[p + "mlp.c_fc.bias"]
        state[p + "mlp_c_proj.weight"] = orig_sd[p + "mlp.c_proj.weight"]
        state[p + "mlp_c_proj.bias"] = orig_sd[p + "mlp.c_proj.bias"]
        state[p + "scale_shift_table"] = orig_sd[p + "scale_shift_table"]

    neuron_model.load_state_dict(state, strict=False)
    return neuron_model


# ============================================================================
# VAE Decoder Wrapper
# ============================================================================


class _VAEDecoderWrapper(nn.Module):
    """Wrapper for Stable Audio VAE decoder. Removes weight_norm before tracing."""

    def __init__(self, vae_model):
        super().__init__()
        self.decoder = vae_model.decoder
        self.pretransform = vae_model.pretransform

    def forward(self, latents):
        decoded = self.decoder(latents)
        if self.pretransform is not None:
            decoded = self.pretransform.decode(decoded)
        return decoded


def _remove_all_weight_norm(model):
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_g"):
            try:
                torch.nn.utils.remove_weight_norm(module)
                count += 1
            except ValueError:
                pass
    return count


# ============================================================================
# CPU-side diffusion components
# ============================================================================


class _Feature1DProcessor(nn.Module):
    def __init__(self, dim=64, power_std=1.0):
        super().__init__()
        self.dim = dim
        self.power_std = power_std
        self.register_buffer("counts", torch.zeros(1))
        self.register_buffer("sum_x", torch.zeros(dim))
        self.register_buffer("sum_x2", torch.zeros(dim))
        self.register_buffer("sum_target_x2", torch.zeros(dim))

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return torch.zeros_like(mean) if self.counts.item() < 10 else mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return torch.ones_like(std) if self.counts.item() < 10 else std

    def project_sample(self, x):
        rescale = (1.0 / self.std.clamp(min=1e-12)) ** self.power_std
        return (x - self.mean.view(1, -1, 1)) * rescale.view(1, -1, 1)

    def return_sample(self, x):
        rescale = self.std**self.power_std
        return x * rescale.view(1, -1, 1) + self.mean.view(1, -1, 1)


class _RVQDequantizer(nn.Module):
    def __init__(self, codebook_size=16384, codebook_dim=32, output_dim=1024):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.out_proj = nn.Conv1d(codebook_dim, output_dim, kernel_size=1)

    def forward(self, codes):
        z_p = F.embedding(codes[:, 0, :], self.codebook.weight).transpose(1, 2)
        return self.out_proj(z_p)


def _load_rvq_weights(rvq_module, state_dict, prefix):
    rvq_module.codebook.weight.data.copy_(
        state_dict[f"{prefix}.quantizers.0.codebook.weight"]
    )
    g = state_dict[f"{prefix}.quantizers.0.out_proj.weight_g"]
    v = state_dict[f"{prefix}.quantizers.0.out_proj.weight_v"]
    norm = torch.norm(v, dim=[1, 2], keepdim=True)
    rvq_module.out_proj.weight.data.copy_(g * v / norm)
    rvq_module.out_proj.bias.data.copy_(
        state_dict[f"{prefix}.quantizers.0.out_proj.bias"]
    )


def _load_cpu_diffusion_components(safetensors_path):
    from safetensors.torch import load_file

    sd = load_file(safetensors_path)

    rvq_vocal = _RVQDequantizer(codebook_size=16384, codebook_dim=32, output_dim=1024)
    _load_rvq_weights(rvq_vocal, sd, prefix="rvq_bestrq_emb")
    rvq_vocal.eval()

    rvq_bgm = _RVQDequantizer(codebook_size=16384, codebook_dim=32, output_dim=1024)
    _load_rvq_weights(rvq_bgm, sd, prefix="rvq_bestrq_bgm_emb")
    rvq_bgm.eval()

    normfeat = _Feature1DProcessor(dim=64)
    normfeat.counts.copy_(sd["normfeat.counts"])
    normfeat.sum_x.copy_(sd["normfeat.sum_x"])
    normfeat.sum_x2.copy_(sd["normfeat.sum_x2"])
    normfeat.sum_target_x2.copy_(sd["normfeat.sum_target_x2"])
    normfeat.eval()

    mask_emb = nn.Embedding(3, 24)
    mask_emb.weight.data.copy_(sd["mask_emb.weight"])
    mask_emb.eval()

    zero_cond = sd["zero_cond_embedding1"]
    return rvq_vocal, rvq_bgm, normfeat, mask_emb, zero_cond


# ============================================================================
# Euler ODE Solver for Diffusion
# ============================================================================


def _solve_euler(
    x,
    latent_mask_input,
    incontext_x,
    incontext_length,
    t_span,
    mu,
    attention_mask,
    guidance_scale,
    neuron_gpt2,
    sigma_min=1e-4,
):
    dt = t_span[1:] - t_span[:-1]
    t = t_span[:-1]
    B = x.shape[0]
    x_next = x.clone()
    noise = x.clone()

    if guidance_scale > 1.0:
        attention_mask_2b = torch.cat([attention_mask, attention_mask], 0)

    for i in range(len(dt)):
        ti = t[i]
        x_next[:, :incontext_length] = (1 - (1 - sigma_min) * ti) * noise[
            :, :incontext_length
        ] + ti * incontext_x[:, :incontext_length]

        if guidance_scale > 1.0:

            def double(z):
                return torch.cat([z, z], 0) if z is not None else None

            model_input = torch.cat(
                [
                    double(latent_mask_input),
                    double(incontext_x),
                    torch.cat([torch.zeros_like(mu), mu], 0),
                    double(x_next),
                ],
                dim=2,
            )
            timestep = ti.expand(2 * B)
            mask_for_model = attention_mask_2b
        else:
            model_input = torch.cat([latent_mask_input, incontext_x, mu, x_next], dim=2)
            timestep = ti.expand(B)
            mask_for_model = attention_mask

        v = neuron_gpt2(model_input, mask_for_model, timestep)
        v = v[..., -x.shape[2] :]

        if guidance_scale > 1.0:
            v_uncond, v_cond = v.chunk(2, 0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)

        x_next = x_next + dt[i] * v

    return x_next


# ============================================================================
# Attention mask helper
# ============================================================================


def _build_attn_mask(cache_position, max_seq, batch_size):
    """Build causal attention mask.

    For decode (cache_position is int): mask shape [B, 1, 1, max_seq]
    For prefill (cache_position is tensor of len P): mask shape [B, 1, P, max_seq]
    """
    if isinstance(cache_position, int):
        mask = torch.full((1, 1, 1, max_seq), float("-inf"), dtype=torch.float32)
        mask[:, :, :, : cache_position + 1] = 0.0
    else:
        P = cache_position.shape[0]
        mask = torch.full((1, 1, P, max_seq), float("-inf"), dtype=torch.float32)
        for q in range(P):
            mask[:, :, q, : cache_position[q].item() + 1] = 0.0
    return mask.expand(batch_size, -1, -1, -1)


# ============================================================================
# Top-k sampling
# ============================================================================


def _sample_top_k(probs, k=5000):
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    sample_indices = torch.multinomial(top_k_probs.view(-1, k), num_samples=1)
    sample_indices = sample_indices.view(probs.shape[0], probs.shape[1], 1)
    next_token = torch.gather(top_k_indices, -1, sample_indices)
    return next_token


# ============================================================================
# Main Pipeline Class
# ============================================================================


class LeVo2Neuron:
    """Neuron-optimized LeVo 2 (SongGeneration v2) text-to-music pipeline.

    Supports both v2-medium and v2-large via LeVo2Config.

    Usage:
        config = LeVo2Config.v2_medium(
            model_path="/path/to/model.pt",
            config_path="/path/to/config.yaml",
            safetensors_path="/path/to/model_2.safetensors",
            prompt_path="/path/to/encode-s12k.pt",
        )
        model = LeVo2Neuron(config)
        model.compile()
        model.save("/path/to/compiled")
        # ... later ...
        model = LeVo2Neuron(config)
        model.load("/path/to/compiled")
        audio, sample_rate = model.generate(
            lyrics="[verse] Hello world",
            descriptions="pop, uplifting",
            genre="Pop",
            duration_sec=5.0,
        )
    """

    def __init__(self, config: LeVo2Config):
        self.config = config
        self._lelm_model = None
        self._primary_neuron = None
        self._secondary_neuron = None
        self._neuron_gpt2 = None
        self._neuron_vae = None
        self._rvq_vocal = None
        self._rvq_bgm = None
        self._normfeat = None
        self._mask_emb = None
        self._zero_cond = None
        self._prompt_data = None
        self._compiled = False

    def _setup_codeclm_paths(self):
        """Add codeclm source paths and patch CUDA."""
        _patch_cuda_to_cpu()
        base = self.config.codeclm_path
        for p in [
            base,
            os.path.join(base, "codeclm/tokenizer"),
            os.path.join(base, "codeclm/tokenizer/Flow1dVAE"),
        ]:
            if p not in sys.path:
                sys.path.insert(0, p)
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(base, "third_party/hub")

    def _load_lelm_cpu(self):
        """Load LeLM model on CPU from checkpoint."""
        self._setup_codeclm_paths()
        from omegaconf import OmegaConf

        prev_cwd = os.getcwd()
        os.chdir(self.config.codeclm_path)

        OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
        OmegaConf.register_new_resolver(
            "concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True
        )
        OmegaConf.register_new_resolver("get_fname", lambda: "test", replace=True)
        OmegaConf.register_new_resolver(
            "load_yaml", lambda x: list(OmegaConf.load(x)), replace=True
        )

        cfg = OmegaConf.load(self.config.config_path)
        cfg.lm.use_flash_attn_2 = False

        from codeclm.models.builders import get_lm_model

        # CRITICAL: version='v2' adds musicality tokens to match v2 checkpoints
        model = get_lm_model(cfg, version="v2")

        sd = torch.load(self.config.model_path, map_location="cpu", weights_only=False)
        if "best_state" in sd:
            sd = sd["best_state"]
        stripped = {
            k[len("audiolm.") :] if k.startswith("audiolm.") else k: v
            for k, v in sd.items()
        }
        model.load_state_dict(stripped, strict=False)
        model.eval()
        os.chdir(prev_cwd)
        self._lelm_model = model
        return model

    def _build_attn_mask(self, cache_position):
        """Build causal attention mask using current config dimensions."""
        return _build_attn_mask(
            cache_position, self.config.max_seq_len, self.config.batch_size
        )

    def compile(self):
        """Compile all pipeline components on Neuron.

        Traces the LeLM primary + secondary via ModelBuilder (on-device KV),
        and the GPT2 + VAE via torch_neuronx.trace().
        Takes ~15-25 minutes depending on variant.
        """
        import torch_neuronx
        from neuronx_distributed import ModelBuilder

        cfg = self.config
        T_frames = int(cfg.default_duration_sec * 25)

        # Compute max_seq_len: must include prepend tokens + pattern steps + headroom
        required = cfg.prefill_len + T_frames + 260 + 10
        for candidate in [512, 768, 1024, 1536, 2048, 3072, 4096, 8192]:
            if candidate >= required:
                cfg.max_seq_len = candidate
                break

        # 1. Load LeLM on CPU
        print(f"[1/5] Loading LeLM {cfg.variant} model on CPU...")
        lm_model = self._load_lelm_cpu()

        # 2. Build primary transformer with on-device KV
        print(f"[2/5] Building primary ({cfg.primary_layers}L) with on-device KV...")
        primary_wrapper = _NeuronPrimaryTransformer(lm_model.transformer, cfg)
        primary_wrapper.eval()

        builder = ModelBuilder(model=primary_wrapper)
        example_kwargs = {
            "inputs_embeds": torch.randn(cfg.batch_size, 1, cfg.dim),
            "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
            "cache_position": torch.tensor([0], dtype=torch.long),
            "attn_mask": self._build_attn_mask(0),
        }
        builder.trace(kwargs=example_kwargs, tag="decode")

        if cfg.prefill_len > 0:
            prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
            prefill_kwargs = {
                "inputs_embeds": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "position_ids": prefill_positions.unsqueeze(0).expand(
                    cfg.batch_size, -1
                ),
                "cache_position": prefill_positions,
                "attn_mask": self._build_attn_mask(prefill_positions),
            }
            builder.trace(kwargs=prefill_kwargs, tag="prefill")

        self._primary_neuron = builder.compile(
            priority_model_key="decode", compiler_args=cfg.compiler_args
        )
        self._primary_neuron.set_weights([primary_wrapper.state_dict()])
        self._primary_neuron.to_neuron()

        # 3. Build secondary transformer with on-device KV
        print(
            f"[3/5] Building secondary ({cfg.secondary_layers}L) with on-device KV..."
        )
        secondary_wrapper = _NeuronFusedSecondary(
            lm_model.transformer2, lm_model.mlp, lm_model.linears, cfg
        )
        secondary_wrapper.eval()

        builder = ModelBuilder(model=secondary_wrapper)
        example_kwargs = {
            "fused_input2": torch.randn(cfg.batch_size, 1, cfg.dim),
            "primary_hidden": torch.randn(cfg.batch_size, 1, cfg.dim),
            "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
            "cache_position": torch.tensor([0], dtype=torch.long),
            "attn_mask": self._build_attn_mask(0),
        }
        builder.trace(kwargs=example_kwargs, tag="decode")

        if cfg.prefill_len > 0:
            prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
            prefill_kwargs = {
                "fused_input2": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "primary_hidden": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "position_ids": prefill_positions.unsqueeze(0).expand(
                    cfg.batch_size, -1
                ),
                "cache_position": prefill_positions,
                "attn_mask": self._build_attn_mask(prefill_positions),
            }
            builder.trace(kwargs=prefill_kwargs, tag="prefill")

        self._secondary_neuron = builder.compile(
            priority_model_key="decode", compiler_args=cfg.compiler_args
        )
        self._secondary_neuron.set_weights([secondary_wrapper.state_dict()])
        self._secondary_neuron.to_neuron()

        # 4. Trace GPT2 diffusion backbone
        print("[4/5] Tracing GPT2 diffusion backbone (fp32)...")
        self._setup_codeclm_paths()
        sys.path.insert(
            0,
            os.path.join(
                cfg.codeclm_path, "codeclm/tokenizer/Flow1dVAE/models_gpt/models"
            ),
        )
        from gpt2_config import GPT2Config
        from gpt2_rope2_time_new_correct_mask_noncasual_reflow import (
            GPT2Model as OrigGPT2Model,
        )
        from safetensors.torch import load_file

        gpt2_config = GPT2Config(
            n_positions=1000,
            n_layer=cfg.gpt2_num_layers,
            n_head=cfg.gpt2_num_heads,
            n_embd=cfg.gpt2_hidden_size,
            n_inner=cfg.gpt2_hidden_size * 2,  # 4400
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
        gpt2_config._attn_implementation = "eager"

        full_sd = load_file(cfg.safetensors_path)
        gpt2_sd = {
            k[len("cfm_wrapper.estimator.") :]: v
            for k, v in full_sd.items()
            if k.startswith("cfm_wrapper.estimator.")
        }
        orig_gpt2 = OrigGPT2Model(gpt2_config)
        orig_gpt2.load_state_dict(gpt2_sd, strict=False)
        orig_gpt2.eval()

        neuron_gpt2 = _NeuronGPT2Model(gpt2_config)
        _load_gpt2_weights(neuron_gpt2, orig_gpt2)
        neuron_gpt2.eval()

        # GPT2 diffusion MUST use --auto-cast none (fp32).
        # The Euler solver amplifies per-step errors exponentially.
        B_diff = 2  # CFG doubles batch for diffusion (always 2 for B=1)
        example_inputs = (
            torch.randn(B_diff, T_frames, cfg.gpt2_hidden_size),
            torch.ones(B_diff, 1, T_frames, T_frames),
            torch.tensor([0.5] * B_diff),
        )
        self._neuron_gpt2 = torch_neuronx.trace(
            neuron_gpt2,
            example_inputs,
            compiler_args=["--auto-cast", "none", "--model-type", "transformer"],
        )

        # 5. Trace VAE decoder
        print("[5/5] Tracing VAE decoder...")
        sys.path.insert(
            0, os.path.join(cfg.codeclm_path, "codeclm/tokenizer/Flow1dVAE")
        )
        from tools.get_1dvae_large import get_model as get_vae_model

        vae_config_path = os.path.join(
            os.path.dirname(cfg.safetensors_path),
            "../vae/stable_audio_1920_vae.json",
        )
        vae_weights_path = os.path.join(
            os.path.dirname(cfg.safetensors_path),
            "../vae/autoencoder_music_1320k.ckpt",
        )
        vae = get_vae_model(vae_config_path, vae_weights_path)
        vae.eval()
        _remove_all_weight_norm(vae)

        vae_wrapper = _VAEDecoderWrapper(vae)
        vae_wrapper.eval()
        self._neuron_vae = torch_neuronx.trace(
            vae_wrapper,
            (torch.randn(1, cfg.latent_dim, T_frames),),
            compiler_args=["--auto-cast", "matmult"],
        )

        # Load CPU diffusion components
        print("[+] Loading CPU diffusion components...")
        (
            self._rvq_vocal,
            self._rvq_bgm,
            self._normfeat,
            self._mask_emb,
            self._zero_cond,
        ) = _load_cpu_diffusion_components(cfg.safetensors_path)

        # Load prompt data
        self._prompt_data = torch.load(
            cfg.prompt_path, map_location="cpu", weights_only=False
        )

        self._compiled = True
        print(f"Compilation complete ({cfg.variant}).")

    def save(self, model_dir: str):
        """Save all compiled Neuron models to disk."""
        os.makedirs(model_dir, exist_ok=True)

        torch.jit.save(self._neuron_gpt2, os.path.join(model_dir, "gpt2_neuron.pt"))
        torch.jit.save(self._neuron_vae, os.path.join(model_dir, "vae_neuron.pt"))

        self._primary_neuron.save(os.path.join(model_dir, "primary_neuron"))
        self._secondary_neuron.save(os.path.join(model_dir, "secondary_neuron"))

        torch.save(
            {
                "rvq_vocal": self._rvq_vocal.state_dict(),
                "rvq_bgm": self._rvq_bgm.state_dict(),
                "normfeat": self._normfeat.state_dict(),
                "mask_emb": self._mask_emb.state_dict(),
                "zero_cond": self._zero_cond,
            },
            os.path.join(model_dir, "cpu_components.pt"),
        )

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        print(f"Saved compiled pipeline to {model_dir}")

    def load(self, model_dir: str):
        """Load pre-compiled Neuron models from disk."""
        import torch_neuronx
        from neuronx_distributed import ModelBuilder

        with open(os.path.join(model_dir, "config.json")) as f:
            saved_config = json.load(f)
        for k, v in saved_config.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self._neuron_gpt2 = torch.load(os.path.join(model_dir, "gpt2_neuron.pt"))
        self._neuron_vae = torch.load(os.path.join(model_dir, "vae_neuron.pt"))

        self._setup_codeclm_paths()
        lm_model = self._load_lelm_cpu()

        cfg = self.config
        primary_wrapper = _NeuronPrimaryTransformer(lm_model.transformer, cfg)
        primary_wrapper.eval()
        builder = ModelBuilder(model=primary_wrapper)
        self._primary_neuron = builder.load(os.path.join(model_dir, "primary_neuron"))
        self._primary_neuron.to_neuron()

        secondary_wrapper = _NeuronFusedSecondary(
            lm_model.transformer2, lm_model.mlp, lm_model.linears, cfg
        )
        secondary_wrapper.eval()
        builder = ModelBuilder(model=secondary_wrapper)
        self._secondary_neuron = builder.load(
            os.path.join(model_dir, "secondary_neuron")
        )
        self._secondary_neuron.to_neuron()

        cpu_state = torch.load(
            os.path.join(model_dir, "cpu_components.pt"), map_location="cpu"
        )
        self._rvq_vocal = _RVQDequantizer(
            codebook_size=16384, codebook_dim=32, output_dim=1024
        )
        self._rvq_vocal.load_state_dict(cpu_state["rvq_vocal"])
        self._rvq_vocal.eval()

        self._rvq_bgm = _RVQDequantizer(
            codebook_size=16384, codebook_dim=32, output_dim=1024
        )
        self._rvq_bgm.load_state_dict(cpu_state["rvq_bgm"])
        self._rvq_bgm.eval()

        self._normfeat = _Feature1DProcessor(dim=64)
        self._normfeat.load_state_dict(cpu_state["normfeat"])
        self._normfeat.eval()

        self._mask_emb = nn.Embedding(3, 24)
        self._mask_emb.load_state_dict(cpu_state["mask_emb"])
        self._mask_emb.eval()

        self._zero_cond = cpu_state["zero_cond"]

        self._prompt_data = torch.load(
            cfg.prompt_path, map_location="cpu", weights_only=False
        )

        self._compiled = True
        print(f"Loaded compiled pipeline from {model_dir}")

    def warmup(self, n_warmup: int = 5):
        """Warm up all Neuron models (prefill + decode + GPT2 + VAE)."""
        cfg = self.config
        T_frames = int(cfg.default_duration_sec * 25)

        for _ in range(3):
            self._neuron_gpt2(
                torch.randn(2, T_frames, cfg.gpt2_hidden_size),
                torch.ones(2, 1, T_frames, T_frames),
                torch.tensor([0.5, 0.5]),
            )
            self._neuron_vae(torch.randn(1, cfg.latent_dim, T_frames))

        if cfg.prefill_len > 0:
            prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
            prefill_pos_ids = prefill_positions.unsqueeze(0).expand(cfg.batch_size, -1)
            prefill_mask = self._build_attn_mask(prefill_positions)
            for _ in range(2):
                primary_out = self._primary_neuron(
                    torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                    prefill_pos_ids,
                    prefill_positions,
                    prefill_mask,
                    model_name="prefill",
                )
                primary_hidden = (
                    primary_out[0] if isinstance(primary_out, tuple) else primary_out[0]
                )
                self._secondary_neuron(
                    torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                    primary_hidden,
                    prefill_pos_ids,
                    prefill_positions,
                    prefill_mask,
                    model_name="prefill",
                )

        for i in range(n_warmup):
            pos_ids = torch.full((cfg.batch_size, 1), i, dtype=torch.long)
            cp = torch.tensor([i], dtype=torch.long)
            am = self._build_attn_mask(i)
            self._primary_neuron(
                torch.randn(cfg.batch_size, 1, cfg.dim),
                pos_ids,
                cp,
                am,
                model_name="decode",
            )
            self._secondary_neuron(
                torch.randn(cfg.batch_size, 1, cfg.dim),
                torch.randn(cfg.batch_size, 1, cfg.dim),
                pos_ids,
                cp,
                am,
                model_name="decode",
            )

    # ------------------------------------------------------------------
    # Stage 1: LeLM AR generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stage1_lelm(
        self, texts, descriptions, audio_qt_embs, T_frames, temp, top_k, cfg_coef
    ):
        """Autoregressive token generation using on-device KV cache."""
        cfg = self.config
        model = self._lelm_model
        primary_neuron = self._primary_neuron
        secondary_neuron = self._secondary_neuron

        code_depth = model.code_depth
        code_size = model.code_size
        B = cfg.real_batch
        BATCH_SIZE = cfg.batch_size
        PREFILL_LEN = cfg.prefill_len

        condition_tensors = model.prepare_condition_tensors(
            batch_size=B,
            text=texts,
            descriptions=descriptions,
            audio_qt_emb=audio_qt_embs,
            prepare_null_condition=True,
        )

        pattern = model.pattern_provider.get_pattern(T_frames)
        unknown_token = -1

        gen_codes = torch.full(
            (B, code_depth, T_frames), unknown_token, dtype=torch.long, device="cpu"
        )
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(
            gen_codes, model.special_token_id
        )
        output_codes = torch.full_like(gen_sequence, code_size)

        start_offset_sequence = pattern.get_first_step_with_timesteps(0)
        assert start_offset_sequence is not None

        is_end = torch.zeros((B, code_depth, 1)).bool()
        ignore_tokens = audio_qt_embs[0][0]
        ignore_tokens = ignore_tokens[ignore_tokens < 16384]

        record_token_pool = []
        gen_sequence_len = gen_sequence.shape[-1]
        prev_offset = 0
        neuron_position = 0

        t_gen_start = time.time()

        with model.streaming():
            for offset in range(start_offset_sequence, gen_sequence_len):
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                S = curr_sequence.shape[-1]

                curr_sequence_cfg = torch.cat([curr_sequence, curr_sequence], dim=0)

                input_1 = model.emb[0](curr_sequence_cfg[:, 0])
                input_2 = sum(
                    [
                        model.layer2_emb[k](curr_sequence_cfg[:, k])
                        for k in range(1, code_depth)
                    ]
                )

                fused_input1_cfg, fused_input2_cfg = model.fuser(
                    input_1, input_2, condition_tensors
                )

                fused_S = fused_input1_cfg.shape[1]

                if fused_S > 1 and fused_S == PREFILL_LEN + 1:
                    # Prefill optimization: process prepend tokens in one shot
                    prefill_len = PREFILL_LEN

                    prefill_input1 = fused_input1_cfg[:, :prefill_len, :]
                    prefill_input2 = fused_input2_cfg[:, :prefill_len, :]
                    prefill_positions = torch.arange(prefill_len, dtype=torch.long)
                    prefill_pos_ids = prefill_positions.unsqueeze(0).expand(
                        BATCH_SIZE, -1
                    )
                    prefill_cache_pos = prefill_positions
                    prefill_attn_mask = self._build_attn_mask(prefill_positions)

                    primary_out = primary_neuron(
                        prefill_input1,
                        prefill_pos_ids,
                        prefill_cache_pos,
                        prefill_attn_mask,
                        model_name="prefill",
                    )
                    primary_hidden = (
                        primary_out[0]
                        if isinstance(primary_out, tuple)
                        else primary_out[0]
                    )
                    primary_logits = (
                        primary_out[1]
                        if isinstance(primary_out, tuple)
                        else primary_out[1]
                    )

                    secondary_out = secondary_neuron(
                        prefill_input2,
                        primary_hidden,
                        prefill_pos_ids,
                        prefill_cache_pos,
                        prefill_attn_mask,
                        model_name="prefill",
                    )
                    fused_res_logits = (
                        secondary_out[0]
                        if isinstance(secondary_out, tuple)
                        else secondary_out
                    )

                    neuron_position = prefill_len

                    # Process the 1 remaining actual token via decode
                    token_input1 = fused_input1_cfg[:, prefill_len : prefill_len + 1, :]
                    position_ids = torch.full(
                        (BATCH_SIZE, 1), neuron_position, dtype=torch.long
                    )
                    cache_position = torch.tensor([neuron_position], dtype=torch.long)
                    attn_mask = self._build_attn_mask(neuron_position)

                    primary_out = primary_neuron(
                        token_input1,
                        position_ids,
                        cache_position,
                        attn_mask,
                        model_name="decode",
                    )
                    primary_hidden = (
                        primary_out[0]
                        if isinstance(primary_out, tuple)
                        else primary_out[0]
                    )
                    primary_logits = (
                        primary_out[1]
                        if isinstance(primary_out, tuple)
                        else primary_out[1]
                    )

                    token_input2 = fused_input2_cfg[:, prefill_len : prefill_len + 1, :]
                    secondary_out = secondary_neuron(
                        token_input2,
                        primary_hidden,
                        position_ids,
                        cache_position,
                        attn_mask,
                        model_name="decode",
                    )
                    fused_res_logits = (
                        secondary_out[0]
                        if isinstance(secondary_out, tuple)
                        else secondary_out
                    )

                    neuron_position += 1

                else:
                    # Normal decode: token by token
                    for s_idx in range(fused_S):
                        token_input1 = fused_input1_cfg[:, s_idx : s_idx + 1, :]
                        position_ids = torch.full(
                            (BATCH_SIZE, 1), neuron_position, dtype=torch.long
                        )
                        cache_position = torch.tensor(
                            [neuron_position], dtype=torch.long
                        )
                        attn_mask = self._build_attn_mask(neuron_position)

                        primary_out = primary_neuron(
                            token_input1,
                            position_ids,
                            cache_position,
                            attn_mask,
                            model_name="decode",
                        )
                        primary_hidden = (
                            primary_out[0]
                            if isinstance(primary_out, tuple)
                            else primary_out[0]
                        )
                        primary_logits = (
                            primary_out[1]
                            if isinstance(primary_out, tuple)
                            else primary_out[1]
                        )

                        token_input2 = fused_input2_cfg[:, s_idx : s_idx + 1, :]
                        secondary_out = secondary_neuron(
                            token_input2,
                            primary_hidden,
                            position_ids,
                            cache_position,
                            attn_mask,
                            model_name="decode",
                        )
                        fused_res_logits = (
                            secondary_out[0]
                            if isinstance(secondary_out, tuple)
                            else secondary_out
                        )

                        neuron_position += 1

                # CFG logits
                logits_cb0 = primary_logits
                cond_logits_cb0, uncond_logits_cb0 = logits_cb0.split(B, dim=0)
                logits_cb0 = (
                    uncond_logits_cb0 + (cond_logits_cb0 - uncond_logits_cb0) * cfg_coef
                )

                cond_res, uncond_res = fused_res_logits.split(B, dim=0)
                res_logits = uncond_res + (cond_res - uncond_res) * cfg_coef

                logits = torch.cat([logits_cb0.unsqueeze(1), res_logits], dim=1)
                logits = logits[:, :, :, :code_size]
                logits = logits[..., -1, :]

                # Repetition penalty
                if record_token_pool and len(record_token_pool) > 0:
                    pool = torch.stack(record_token_pool[-150:], -1)
                    for b in range(B):
                        for q in range(code_depth):
                            q_count = torch.bincount(torch.unique(pool[b, q]))
                            tmp = min(q_count.shape[-1], code_size - 1)
                            logits[b, q, :tmp] /= 1.1 ** q_count[:tmp]

                # Ignore prompt tokens
                if ignore_tokens is not None and len(ignore_tokens) > 0:
                    logits[:, 0, ignore_tokens.to(torch.int)] = float("-inf")

                # Sampling
                if temp > 0:
                    probs = torch.softmax(logits / temp, dim=-1)
                    next_cb0 = _sample_top_k(probs[:, [0], :], k=top_k)
                    next_res = _sample_top_k(probs[:, 1:, :], k=1)
                    next_token = torch.cat([next_cb0, next_res], dim=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                valid_mask = mask[..., offset : offset + 1].expand(B, -1, -1)
                next_token[~valid_mask] = model.special_token_id

                next_token[is_end] = model.special_token_id
                is_end = is_end | (next_token == model.eos_token_id)

                gen_sequence[..., offset : offset + 1] = torch.where(
                    gen_sequence[..., offset : offset + 1] == unknown_token,
                    next_token,
                    gen_sequence[..., offset : offset + 1],
                )

                record_token_pool.append(next_token.squeeze(-1))

                if torch.all(is_end):
                    gen_sequence = gen_sequence[..., : offset + 1]
                    break

                prev_offset = offset

        gen_time = time.time() - t_gen_start
        gen_steps = neuron_position

        max_gen_len_actual = gen_sequence.shape[-1]
        output_codes[..., :max_gen_len_actual] = gen_sequence
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            output_codes, special_token=unknown_token
        )

        return out_codes, gen_time, gen_steps

    # ------------------------------------------------------------------
    # Stage 2: Diffusion decode
    # ------------------------------------------------------------------

    def _stage2_diffusion(self, tokens, num_diffusion_steps, guidance_scale, seed):
        """Diffusion decode: per-song loop (GPT2 NEFF compiled for B=1 CFG)."""
        cfg = self.config
        B_total = tokens.shape[0]
        all_latents = []
        t_start = time.time()

        for b_idx in range(B_total):
            single_tokens = tokens[b_idx : b_idx + 1]
            codes_vocal = single_tokens[:, [1], :].clamp(0, 16383)
            codes_bgm = single_tokens[:, [2], :].clamp(0, 16383)
            T = codes_vocal.shape[2]
            B = 1

            quantized_vocal = self._rvq_vocal(codes_vocal).permute(0, 2, 1).contiguous()
            quantized_bgm = self._rvq_bgm(codes_bgm).permute(0, 2, 1).contiguous()

            latent_masks = torch.full((B, T), 2, dtype=torch.int64)

            zero_cond_reshaped = self._zero_cond.reshape(1, 1, 1024)
            mask_active = (latent_masks > 0.5).unsqueeze(-1)
            mask_inactive = (latent_masks < 0.5).unsqueeze(-1)
            quantized_vocal = (
                mask_active * quantized_vocal + mask_inactive * zero_cond_reshaped
            )
            quantized_bgm = (
                mask_active * quantized_bgm + mask_inactive * zero_cond_reshaped
            )

            torch.manual_seed(seed + b_idx)
            true_latents = torch.randn(B, T, 64)
            true_latents_perm = true_latents.permute(0, 2, 1).contiguous()
            true_latents_norm = self._normfeat.project_sample(true_latents_perm)
            true_latents_norm = true_latents_norm.permute(0, 2, 1).contiguous()

            incontext_mask = (
                ((latent_masks > 0.5) & (latent_masks < 1.5)).unsqueeze(-1).float()
            )
            incontext_latents = true_latents_norm * incontext_mask

            attn_1d = latent_masks > 0.5
            attn_2d = attn_1d.view(B, 1, T) * attn_1d.view(B, T, 1)
            attention_mask = attn_2d.unsqueeze(1).float()

            latent_mask_input = self._mask_emb(latent_masks)
            mu = torch.cat([quantized_vocal, quantized_bgm], dim=2)

            torch.manual_seed(seed + 1000 + b_idx)
            latents = torch.randn(B, T, 64)

            t_span = torch.linspace(0, 1, num_diffusion_steps + 1)

            latents = _solve_euler(
                latents,
                latent_mask_input,
                incontext_latents,
                0,
                t_span,
                mu,
                attention_mask,
                guidance_scale,
                self._neuron_gpt2,
            )

            latents = latents.permute(0, 2, 1).contiguous()
            latents = self._normfeat.return_sample(latents)
            all_latents.append(latents)

        decode_time = time.time() - t_start
        return torch.cat(all_latents, dim=0), decode_time

    # ------------------------------------------------------------------
    # Stage 3: VAE decode
    # ------------------------------------------------------------------

    def _stage3_vae(self, latents):
        """VAE decode: per-song loop (VAE NEFF compiled for batch=1)."""
        B_total = latents.shape[0]
        all_audio = []
        t_start = time.time()
        for b_idx in range(B_total):
            audio = self._neuron_vae(latents[b_idx : b_idx + 1])
            all_audio.append(audio)
        decode_time = time.time() - t_start
        return torch.cat(all_audio, dim=0), decode_time

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        lyrics: str,
        descriptions: str = ".",
        genre: str = None,
        duration_sec: float = None,
        temp: float = None,
        top_k: int = None,
        cfg_coef: float = None,
        num_diffusion_steps: int = None,
        guidance_scale: float = None,
        seed: int = 42,
        lang: str = "en",
    ) -> Tuple[torch.Tensor, int]:
        """Generate audio from lyrics and style description.

        Args:
            lyrics: Lyrics with structural tags.
                Format: "[intro-short] ; [verse] Line one.Line two ; [chorus] ..."
            descriptions: Style description, e.g. "pop, uplifting, piano".
                Musicality prefix is added automatically if missing.
            genre: Genre for prompt selection (Pop, R&B, etc.).
            duration_sec: Audio duration in seconds.
            temp: Sampling temperature.
            top_k: Top-k sampling parameter.
            cfg_coef: Classifier-free guidance coefficient for LeLM.
            num_diffusion_steps: Number of Euler solver steps.
            guidance_scale: CFG scale for diffusion.
            seed: Random seed.
            lang: Language for prompt selection ("en" or "zh").

        Returns:
            Tuple of (audio_tensor [B, 2, samples], sample_rate).
        """
        assert self._compiled, "Call compile() or load() before generate()"

        cfg = self.config
        genre = genre or cfg.default_genre
        duration_sec = duration_sec or cfg.default_duration_sec
        temp = temp if temp is not None else cfg.default_temp
        top_k = top_k or cfg.default_top_k
        cfg_coef = cfg_coef or cfg.default_cfg_coef
        num_diffusion_steps = num_diffusion_steps or cfg.default_num_diffusion_steps
        guidance_scale = guidance_scale or cfg.default_guidance_scale

        T_frames = int(duration_sec * 25)

        # Load prompt
        prompt_data = self._prompt_data
        genre_data = prompt_data[genre]
        if isinstance(genre_data, dict) and lang in genre_data:
            prompt_list = genre_data[lang]
        elif isinstance(genre_data, list):
            prompt_list = genre_data
        else:
            prompt_list = genre_data
        prompt_tensor = prompt_list[0]
        if isinstance(prompt_tensor, list):
            prompt_tensor = prompt_tensor[0]

        audio_qt_embs = (
            prompt_tensor.unsqueeze(0) if prompt_tensor.dim() == 2 else prompt_tensor
        )
        if cfg.real_batch > 1 and audio_qt_embs.shape[0] == 1:
            audio_qt_embs = audio_qt_embs.expand(cfg.real_batch, -1, -1).contiguous()

        # Format description with musicality prefix
        description_text = descriptions.lower() if descriptions else "."
        if description_text != "." and not description_text.startswith("[musicality"):
            description_text = f"[Musicality-very-high]{description_text}"

        texts_list = [lyrics] * cfg.real_batch
        desc_list = [description_text] * cfg.real_batch

        # Stage 1: LeLM
        tokens, lelm_time, lelm_steps = self._stage1_lelm(
            texts_list, desc_list, audio_qt_embs, T_frames, temp, top_k, cfg_coef
        )

        # Stage 2: Diffusion
        latents, diff_time = self._stage2_diffusion(
            tokens, num_diffusion_steps, guidance_scale, seed
        )

        # Stage 3: VAE
        audio, vae_time = self._stage3_vae(latents)

        total_time = lelm_time + diff_time + vae_time
        audio_duration = audio.shape[-1] / cfg.sample_rate

        print(
            f"Generated {audio_duration:.1f}s audio in {total_time:.1f}s "
            f"(RTF: {total_time / audio_duration:.2f}x)"
        )

        return audio, cfg.sample_rate

    def generate_timed(self, lyrics: str, **kwargs) -> Dict:
        """Generate audio and return timing breakdown.

        Returns:
            Dict with keys: audio, sample_rate, timings.
        """
        assert self._compiled, "Call compile() or load() before generate_timed()"

        cfg = self.config
        genre = kwargs.get("genre", cfg.default_genre)
        duration_sec = kwargs.get("duration_sec", cfg.default_duration_sec)
        temp = kwargs.get("temp", cfg.default_temp)
        top_k = kwargs.get("top_k", cfg.default_top_k)
        cfg_coef = kwargs.get("cfg_coef", cfg.default_cfg_coef)
        num_diffusion_steps = kwargs.get(
            "num_diffusion_steps", cfg.default_num_diffusion_steps
        )
        guidance_scale = kwargs.get("guidance_scale", cfg.default_guidance_scale)
        seed = kwargs.get("seed", 42)
        lang = kwargs.get("lang", "en")
        descriptions = kwargs.get("descriptions", ".")

        T_frames = int(duration_sec * 25)

        # Load prompt
        prompt_data = self._prompt_data
        genre_data = prompt_data[genre]
        if isinstance(genre_data, dict) and lang in genre_data:
            prompt_list = genre_data[lang]
        elif isinstance(genre_data, list):
            prompt_list = genre_data
        else:
            prompt_list = genre_data
        prompt_tensor = prompt_list[0]
        if isinstance(prompt_tensor, list):
            prompt_tensor = prompt_tensor[0]

        audio_qt_embs = (
            prompt_tensor.unsqueeze(0) if prompt_tensor.dim() == 2 else prompt_tensor
        )
        if cfg.real_batch > 1 and audio_qt_embs.shape[0] == 1:
            audio_qt_embs = audio_qt_embs.expand(cfg.real_batch, -1, -1).contiguous()

        description_text = descriptions.lower() if descriptions else "."
        if description_text != "." and not description_text.startswith("[musicality"):
            description_text = f"[Musicality-very-high]{description_text}"

        texts_list = [lyrics] * cfg.real_batch
        desc_list = [description_text] * cfg.real_batch

        tokens, lelm_time, lelm_steps = self._stage1_lelm(
            texts_list, desc_list, audio_qt_embs, T_frames, temp, top_k, cfg_coef
        )

        latents, diff_time = self._stage2_diffusion(
            tokens, num_diffusion_steps, guidance_scale, seed
        )

        audio, vae_time = self._stage3_vae(latents)

        total_time = lelm_time + diff_time + vae_time

        return {
            "audio": audio,
            "sample_rate": cfg.sample_rate,
            "timings": {
                "lelm_s": lelm_time,
                "lelm_steps": lelm_steps,
                "diffusion_s": diff_time,
                "vae_s": vae_time,
                "total_s": total_time,
            },
        }
