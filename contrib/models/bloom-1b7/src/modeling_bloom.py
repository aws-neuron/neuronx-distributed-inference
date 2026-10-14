# coding=utf-8
# Copyright 2022 BigScience workshop and HuggingFace Inc. team.
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
"""Bloom model port for NeuronX Distributed Inference."""

import json
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    FlashAttentionStrategy,
    repeat_kv,
    manual_softmax,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group


def _build_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each attention head.

    Returns a 1-D tensor of shape [num_heads] with the per-head slope values.
    For a power-of-2 num_heads: slopes = 2^(-8/n * i) for i in 1..n.
    For non-power-of-2: uses the interleaved formula from the paper.
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.float32)
    slopes = torch.pow(torch.tensor(base, dtype=torch.float32), powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining, 2, dtype=torch.float32)
        slopes = torch.cat([slopes, torch.pow(torch.tensor(extra_base, dtype=torch.float32), extra_powers)])

    return slopes


class BloomInferenceConfig(InferenceConfig):
    """Configuration for Bloom model inference on NeuronX."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.sliding_window = None

        if not hasattr(self, 'intermediate_size') or self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        # Bloom uses MHA
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.output_attentions = False
        self.output_hidden_states = False
        self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "layer_norm_epsilon",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "BloomInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        n_head = params.get("num_attention_heads", params.get("n_head", 16))
        hidden_size = params.get("hidden_size", params.get("n_embed", 2048))

        config_dict = {
            "hidden_size": hidden_size,
            "num_attention_heads": n_head,
            "num_key_value_heads": n_head,
            "num_hidden_layers": params.get("num_hidden_layers", params.get("n_layer", 24)),
            "vocab_size": params.get("vocab_size", 250880),
            "max_position_embeddings": params.get("seq_length", 2048),
            "intermediate_size": params.get("n_inner", None),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-5),
            "bos_token_id": params.get("bos_token_id", 1),
            "eos_token_id": params.get("eos_token_id", 2),
            "pad_token_id": params.get("pad_token_id", 3),
        }

        config_dict.update(kwargs)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronBloomAttention(NeuronAttentionBase):
    """Bloom attention with ALiBi positional encoding.

    ALiBi adds a linear bias to attention scores instead of using position
    embeddings. Each head has a geometric slope that scales the distance
    between query and key positions.

    The base class always divides Q by sqrt(head_dim) in both prefill and
    token-gen paths. Bloom uses standard scaled attention (1/sqrt(d_k)),
    so we don't need to compensate like GPT-Neo does.

    For ALiBi, we override perform_prefill and compute_for_token_gen to
    add the position-dependent bias to attention scores.
    """

    def __init__(self, config: BloomInferenceConfig, layer_idx: int = None, **kwargs):
        self.layer_idx = layer_idx

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            clip_qkv=None,
            qkv_bias=True,
            o_bias=True,
            sliding_window=None,
            **kwargs,
        )

        # ALiBi slopes: [num_heads_per_rank]
        # For TP, each rank handles a shard of heads
        full_slopes = _build_alibi_slopes(config.num_attention_heads)
        self.heads_per_rank = config.num_attention_heads // config.neuron_config.tp_degree
        self.register_buffer("alibi_slopes", full_slopes, persistent=False)

    def _slice_alibi_for_tp(self, alibi: torch.Tensor) -> torch.Tensor:
        """Slice ALiBi bias along head dimension for tensor parallelism."""
        if self.heads_per_rank < self.config.num_attention_heads:
            rank = self.rank_util.get_rank()
            start = rank * self.heads_per_rank
            end = start + self.heads_per_rank
            alibi = alibi[:, start:end, :, :]
        return alibi

    def _get_alibi_bias_prefill(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build ALiBi bias for context encoding (prefill).

        Returns tensor of shape [1, num_heads, seq_len, seq_len] where
        bias[h, i, j] = slope_h * (j - i) for j <= i, -inf for j > i (causal).
        """
        # positions: [seq_len]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # relative positions: [seq_len, seq_len], rel[i,j] = j - i
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]

        # slopes: [num_heads, 1, 1]
        slopes = self.alibi_slopes.to(device).unsqueeze(-1).unsqueeze(-1)

        # bias: [num_heads, seq_len, seq_len]
        alibi = slopes * rel_pos.unsqueeze(0)  # [H, S, S]

        return alibi.unsqueeze(0).to(dtype)  # [1, H, S, S]

    def _get_alibi_bias_token_gen(
        self, position_ids: torch.Tensor, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build ALiBi bias for token generation.

        position_ids: [batch, q_len] - current position(s) being generated
        kv_len: total length of the KV cache (prior tokens)

        Returns tensor of shape [batch, num_heads, q_len, kv_len]
        """
        # kv positions: [kv_len]
        kv_positions = torch.arange(kv_len, device=device, dtype=torch.float32)

        # For each query position, compute distance to each kv position
        # position_ids: [batch, q_len] -> [batch, 1, q_len, 1]
        q_pos = position_ids.float().unsqueeze(1).unsqueeze(-1)
        # kv_positions: [1, 1, 1, kv_len]
        kv_pos = kv_positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # relative distances: [batch, 1, q_len, kv_len]
        rel_pos = kv_pos - q_pos  # negative for past positions

        # slopes: [1, num_heads, 1, 1]
        slopes = self.alibi_slopes.to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # bias: [batch, num_heads, q_len, kv_len]
        alibi = slopes * rel_pos

        return alibi.to(dtype)

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
        """Override prefill to add ALiBi bias. Always uses non-flash path."""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # Compute scaled QK: [batch, heads, q_len, kv_len]
        active_scores = self.scaled_qk(Q, K_active, attention_mask)

        # Add ALiBi bias
        alibi = self._get_alibi_bias_prefill(q_len, Q.device, torch.float32)
        active_scores = active_scores + self._slice_alibi_for_tp(alibi)

        active_scores = F.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_output = torch.matmul(active_scores, V_active)

        return attn_output, FlashAttentionStrategy.NONE

    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask, is_prefix_caching=False,
    ):
        """Override token gen to add ALiBi bias."""
        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1

        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / math.sqrt(self.head_dim)

        # Pad attention mask if needed
        if prior_scores.shape[-1] > attention_mask.shape[-1] and self.neuron_config.apply_seq_ids_mask:
            attention_mask = F.pad(attention_mask, (0, prior_scores.shape[-1] - attention_mask.shape[-1]), "constant", 0)

        # Add ALiBi bias to prior scores
        prior_kv_len = prior_scores.shape[-1]
        alibi_prior = self._get_alibi_bias_token_gen(position_ids, prior_kv_len, Q.device, torch.float32)
        prior_scores = prior_scores + self._slice_alibi_for_tp(alibi_prior)

        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # Active (current token) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        # ALiBi for active: distance is 0 (current token attending to itself)
        # So alibi contribution is slope * 0 = 0; no bias needed for active
        if is_speculation or is_prefix_caching:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )

        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output


class NeuronBloomMLP(nn.Module):
    """Bloom MLP: Linear -> GELU -> Linear."""

    def __init__(self, config: BloomInferenceConfig):
        super().__init__()
        intermediate_size = config.intermediate_size or (4 * config.hidden_size)

        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        self.act = nn.GELU(approximate='tanh')

        self.dense_4h_to_h = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states, None


class NeuronBloomBlock(nn.Module):
    """Bloom transformer block: LN -> Attn -> Residual -> LN -> MLP -> Residual.

    Bloom uses pre-LayerNorm residual connections
    (apply_residual_connection_post_layernorm=False for bloom-1b7).
    """

    def __init__(self, config: BloomInferenceConfig, layer_idx: int = None):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = NeuronBloomAttention(
            config, layer_idx=layer_idx, tensor_model_parallel_group=get_tp_group(config)
        )
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = NeuronBloomMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs
    ):
        # Attention with pre-norm residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states + residual

        # MLP with pre-norm residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, _ = self.mlp(hidden_states)
        hidden_states = mlp_output + residual

        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronBloomModel(NeuronBaseModel):
    """Bloom base model for NeuronX.

    Bloom does NOT use position embeddings. Instead, ALiBi biases are
    added directly in the attention module. An embedding LayerNorm is
    applied after token embeddings.
    """

    def setup_attr_for_model(self, config: BloomInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: BloomInferenceConfig):
        self.padding_idx = getattr(config, 'pad_token_id', 3)

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        # Embedding LayerNorm (applied after token embeddings, before transformer blocks)
        self.word_embeddings_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        # LM head (tied to embeddings)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            bias=False,
            pad=True,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            NeuronBloomBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def get_model_output(self, input_ids, position_ids=None, **kwargs):
        """Override to apply embedding LayerNorm (no position embeddings for ALiBi)."""
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Apply embedding LayerNorm
        inputs_embeds = self.word_embeddings_layernorm(inputs_embeds)

        kwargs['inputs_embeds'] = inputs_embeds
        return super().get_model_output(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )


class NeuronBloomForCausalLM(NeuronBaseForCausalLM):
    """Bloom model with language modeling head for NeuronX."""

    _model_cls = NeuronBloomModel
    _STATE_DICT_MODEL_PREFIX = "transformer."

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert HF Bloom weights to NeuronX format.

        After framework strips "transformer." prefix, keys look like:
        - word_embeddings.weight
        - word_embeddings_layernorm.{weight,bias}
        - h.0.self_attention.query_key_value.{weight,bias}  (fused QKV)
        - h.0.self_attention.dense.{weight,bias}
        - h.0.input_layernorm.{weight,bias}
        - h.0.post_attention_layernorm.{weight,bias}
        - h.0.mlp.dense_h_to_4h.{weight,bias}
        - h.0.mlp.dense_4h_to_h.{weight,bias}
        - ln_f.{weight,bias}

        The fused QKV weight has shape [3*hidden_size, hidden_size] with
        interleaved layout: [num_heads, 3, head_dim, hidden_size]. We need
        to split it into separate Q, K, V projections.
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        # Token embeddings
        if "word_embeddings.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["word_embeddings.weight"]

        # Embedding LayerNorm
        for param in ["weight", "bias"]:
            key = f"word_embeddings_layernorm.{param}"
            if key in state_dict:
                neuron_state_dict[f"word_embeddings_layernorm.{param}"] = state_dict[key]

        # Final layer norm
        for param in ["weight", "bias"]:
            if f"ln_f.{param}" in state_dict:
                neuron_state_dict[f"norm.{param}"] = state_dict[f"ln_f.{param}"]

        for i in range(config.num_hidden_layers):
            hf = f"h.{i}"
            nxd = f"layers.{i}"

            # Layer norms
            for ln_name in ["input_layernorm", "post_attention_layernorm"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.{ln_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.{ln_name}.{param}"] = state_dict[key]

            # Split fused QKV into separate Q, K, V
            # HF layout: [num_heads * 3 * head_dim, hidden_size]
            # Interleaved as: [num_heads, 3, head_dim, hidden_size]
            qkv_w_key = f"{hf}.self_attention.query_key_value.weight"
            qkv_b_key = f"{hf}.self_attention.query_key_value.bias"
            attn_nxd = f"{nxd}.attn"

            if qkv_w_key in state_dict:
                qkv_weight = state_dict[qkv_w_key]  # [3*H, H]
                # Reshape to [num_heads, 3, head_dim, hidden_size]
                qkv_weight = qkv_weight.view(num_heads, 3, head_dim, config.hidden_size)
                q_weight = qkv_weight[:, 0, :, :].reshape(config.hidden_size, config.hidden_size)
                k_weight = qkv_weight[:, 1, :, :].reshape(config.hidden_size, config.hidden_size)
                v_weight = qkv_weight[:, 2, :, :].reshape(config.hidden_size, config.hidden_size)

                neuron_state_dict[f"{attn_nxd}.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"{attn_nxd}.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"{attn_nxd}.qkv_proj.v_proj.weight"] = v_weight

            if qkv_b_key in state_dict:
                qkv_bias = state_dict[qkv_b_key]  # [3*H]
                qkv_bias = qkv_bias.view(num_heads, 3, head_dim)
                q_bias = qkv_bias[:, 0, :].reshape(config.hidden_size)
                k_bias = qkv_bias[:, 1, :].reshape(config.hidden_size)
                v_bias = qkv_bias[:, 2, :].reshape(config.hidden_size)

                neuron_state_dict[f"{attn_nxd}.qkv_proj.q_proj.bias"] = q_bias
                neuron_state_dict[f"{attn_nxd}.qkv_proj.k_proj.bias"] = k_bias
                neuron_state_dict[f"{attn_nxd}.qkv_proj.v_proj.bias"] = v_bias

            # Output projection: dense -> o_proj (preshard_hook adds extra o_proj level)
            for param in ["weight", "bias"]:
                key = f"{hf}.self_attention.dense.{param}"
                if key in state_dict:
                    neuron_state_dict[f"{attn_nxd}.o_proj.{param}"] = state_dict[key]

            # rank_util for attention
            neuron_state_dict[f"{attn_nxd}.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

            # MLP
            for hf_name, nxd_name in [
                ("dense_h_to_4h", "dense_h_to_4h"),
                ("dense_4h_to_h", "dense_4h_to_h"),
            ]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.mlp.{hf_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.mlp.{nxd_name}.{param}"] = state_dict[key]

        # LM head uses tied weights
        if "word_embeddings.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["word_embeddings.weight"].clone()

        # rank_util for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @classmethod
    def get_config_cls(cls):
        return BloomInferenceConfig
