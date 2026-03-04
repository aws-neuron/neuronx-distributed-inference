# coding=utf-8
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
NeuronX CodeGen model - v2 (proper NXDI structure).
Algorithmic fixes from v1:
1. Partial RoPE (32/64 dims) with GPT-J rotate_every_two convention
2. Pre-interleaved cos/sin cache (Neuron-safe, no strided slicing)
3. QKV weight mapping with mp_num=4 and (Q,V,K) order
Structural fixes:
4. Proper setup_attr_for_model/init_model pattern (no __init__ override)
5. Correct decoder layer forward: 5-tuple return
6. MLP returns (output, None) tuple
7. Weight paths: self_attn.qkv_proj.{q,k,v}_proj
"""

import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase

logger = logging.getLogger("Neuron")


# ---------- GPT-J / CodeGen style partial rotary embeddings ----------

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """GPT-J style rotation: interleaved pairs (-x_odd, x_even).
    Uses reshape instead of strided slicing (::2) for Neuron compatibility."""
    *leading, d = x.shape
    x = x.reshape(*leading, d // 2, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).reshape(*leading, d)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply GPT-J style partial RoPE to first rotary_dim dimensions only.

    Q/K shape: [batch, num_heads, seq_len, head_dim]
    cos/sin shape: [batch, seq_len, rotary_dim] (pre-interleaved)
    """
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Expand cos/sin for head broadcasting: [batch, seq, rotary_dim] -> [batch, 1, seq, rotary_dim]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    q_rot = (q_rot * cos) + (rotate_every_two(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_every_two(k_rot) * sin)

    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class CodeGenRotaryEmbedding(nn.Module):
    """Rotary embeddings for CodeGen partial RoPE (rotary_dim=32 out of head_dim=64).
    Pre-interleaves cos/sin in cache to avoid runtime repeat_interleave."""

    def __init__(self, rotary_dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = torch.matmul(t, self.inv_freq.unsqueeze(0))  # [seq, rotary_dim//2]
        # Pre-interleave: [seq, rotary_dim//2] -> [seq, rotary_dim]
        freqs_interleaved = freqs.unsqueeze(-1).expand(-1, -1, 2).reshape(seq_len, -1)
        self.register_buffer("cos_cached", freqs_interleaved.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_interleaved.sin(), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """Base class calls rotary_emb(V, position_ids). x is unused (V tensor)."""
        cos = self.cos_cached[position_ids]  # [batch, seq, rotary_dim]
        sin = self.sin_cached[position_ids]  # [batch, seq, rotary_dim]
        return cos, sin


# ---------- Config ----------

class CodeGenInferenceConfig(InferenceConfig):
    def __init__(self, neuron_config=None, **kwargs):
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("use_cache", True)
        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.sliding_window = None

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rotary_dim",
            "layer_norm_epsilon",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "CodeGenInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            params = json.load(f)

        n_head = params.get("n_head", 16)
        n_inner = params.get("n_inner", None)
        n_embd = params.get("n_embd", 1024)
        config_dict = {
            "hidden_size": n_embd,
            "num_attention_heads": n_head,
            "num_key_value_heads": n_head,  # CodeGen is MHA
            "num_hidden_layers": params.get("n_layer", 20),
            "vocab_size": params.get("vocab_size", 51200),
            "max_position_embeddings": params.get("n_positions", 2048),
            "rotary_dim": params.get("rotary_dim", 32),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-5),
            "hidden_act": params.get("activation_function", "gelu_new"),
            "intermediate_size": n_inner if n_inner is not None else 4 * n_embd,
            "pad_token_id": params.get("pad_token_id", None),
            "bos_token_id": params.get("bos_token_id", 1),
            "eos_token_id": params.get("eos_token_id", 50256),
        }
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


# ---------- Attention ----------

class NeuronCodeGenAttention(NeuronAttentionBase):
    """CodeGen attention with partial RoPE (32/64 dims) and GPT-J rotation convention."""

    def __init__(self, config: CodeGenInferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = config.rotary_dim

        rotary_emb = CodeGenRotaryEmbedding(
            rotary_dim=config.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # MHA
            head_dim=head_dim,
            rotary_emb=rotary_emb,
        )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Override: partial CodeGen RoPE on first rotary_dim dimensions only."""
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb_partial(Q, K, cos_cache, sin_cache, self.rotary_dim)
        return Q, K, cos_cache, sin_cache


# ---------- MLP ----------

class NeuronCodeGenMLP(nn.Module):
    def __init__(self, config: CodeGenInferenceConfig):
        super().__init__()
        intermediate_size = config.intermediate_size
        self.fc_in = ColumnParallelLinear(
            config.hidden_size, intermediate_size,
            bias=True, gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.fc_out = RowParallelLinear(
            intermediate_size, config.hidden_size,
            bias=True, input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act = nn.GELU(approximate='tanh')

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states, None  # (output, unused) — matches NXDI expected tuple


# ---------- Decoder block ----------

class NeuronCodeGenBlock(nn.Module):
    """CodeGen block with PARALLEL residual: output = attn(ln(x)) + mlp(ln(x)) + x"""

    def __init__(self, config: CodeGenInferenceConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NeuronCodeGenAttention(config)
        self.mlp = NeuronCodeGenMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # Attention
        attn_output = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        attn_hidden = attn_output.hidden_states
        present_key_value = attn_output.present_key_value
        cos_cache = attn_output.cos_cache
        sin_cache = attn_output.sin_cache

        # MLP (parallel with attention, both on normed input)
        mlp_output, _ = self.mlp(normed)

        # Parallel residual
        hidden_states = attn_hidden + mlp_output + residual

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------- Base model ----------

class NeuronCodeGenModel(NeuronBaseModel):
    """CodeGen model — relies on base class __init__ to call setup_attr_for_model/init_model."""

    def setup_attr_for_model(self, config: CodeGenInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # MHA
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: CodeGenInferenceConfig):
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [NeuronCodeGenBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)


# ---------- CausalLM wrapper ----------

class NeuronCodeGenForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronCodeGenModel

    @classmethod
    def get_config_cls(cls):
        return CodeGenInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: CodeGenInferenceConfig) -> dict:
        """Convert HF CodeGen state dict to NeuronX format.

        FIX #3: Correct QKV weight mapping with mp_num=4 and (Q,V,K) order.
        The HF CodeGen model has a fused qkv_proj with interleaved layout:
        [mp_num=4 chunks] where each chunk has [Q_chunk, V_chunk, K_chunk]
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        hidden_size = config.hidden_size

        # CodeGen-specific constant for QKV layout
        mp_num = 4
        local_dim = head_dim * num_heads // mp_num

        # Token embeddings
        if "transformer.wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["transformer.wte.weight"]

        # Final layer norm
        if "transformer.ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["transformer.ln_f.weight"]
        if "transformer.ln_f.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["transformer.ln_f.bias"]

        # LM head (CodeGen has separate lm_head, not tied)
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
        if "lm_head.bias" in state_dict:
            neuron_state_dict["lm_head.bias"] = state_dict["lm_head.bias"]

        # Decoder layers
        for i in range(config.num_hidden_layers):
            hf = f"transformer.h.{i}"
            nx = f"layers.{i}"

            # Layer norm
            if f"{hf}.ln_1.weight" in state_dict:
                neuron_state_dict[f"{nx}.input_layernorm.weight"] = state_dict[f"{hf}.ln_1.weight"]
            if f"{hf}.ln_1.bias" in state_dict:
                neuron_state_dict[f"{nx}.input_layernorm.bias"] = state_dict[f"{hf}.ln_1.bias"]

            # QKV projection - decompose fused qkv_proj with mp_num=4 and (Q,V,K) order
            if f"{hf}.attn.qkv_proj.weight" in state_dict:
                qkv_weight = state_dict[f"{hf}.attn.qkv_proj.weight"]  # [3*hidden_size, hidden_size]

                # Reshape to [mp_num, 3, local_dim, hidden_size]
                qkv_reshaped = qkv_weight.reshape(mp_num, 3, local_dim, hidden_size)

                # Extract Q, K, V based on the ACTUAL split order: (query, value, key)
                # Index 0 = Query, Index 1 = Value, Index 2 = Key
                q_weight = qkv_reshaped[:, 0, :, :].reshape(hidden_size, hidden_size)
                v_weight = qkv_reshaped[:, 1, :, :].reshape(hidden_size, hidden_size)
                k_weight = qkv_reshaped[:, 2, :, :].reshape(hidden_size, hidden_size)

                neuron_state_dict[f"{nx}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"{nx}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"{nx}.self_attn.qkv_proj.v_proj.weight"] = v_weight

            # Output projection
            if f"{hf}.attn.out_proj.weight" in state_dict:
                neuron_state_dict[f"{nx}.self_attn.o_proj.weight"] = state_dict[f"{hf}.attn.out_proj.weight"]

            # MLP
            if f"{hf}.mlp.fc_in.weight" in state_dict:
                neuron_state_dict[f"{nx}.mlp.fc_in.weight"] = state_dict[f"{hf}.mlp.fc_in.weight"]
            if f"{hf}.mlp.fc_in.bias" in state_dict:
                neuron_state_dict[f"{nx}.mlp.fc_in.bias"] = state_dict[f"{hf}.mlp.fc_in.bias"]
            if f"{hf}.mlp.fc_out.weight" in state_dict:
                neuron_state_dict[f"{nx}.mlp.fc_out.weight"] = state_dict[f"{hf}.mlp.fc_out.weight"]
            if f"{hf}.mlp.fc_out.bias" in state_dict:
                neuron_state_dict[f"{nx}.mlp.fc_out.bias"] = state_dict[f"{hf}.mlp.fc_out.bias"]

            neuron_state_dict[f"{nx}.self_attn.rank_util.rank"] = \
                torch.arange(0, tp_degree, dtype=torch.int32)

        # Top-level rank_util for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if config.neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = \
                torch.arange(0, config.neuron_config.local_ranks_size, dtype=torch.int32)

        return neuron_state_dict
