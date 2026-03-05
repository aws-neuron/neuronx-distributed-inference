# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
"""MPT (MosaicML Pretrained Transformer) model for NeuronX Distributed Inference.

Key architecture notes:
- ALiBi (Attention with Linear Biases) instead of positional embeddings
- Fused QKV projection (Wqkv)
- GELU activation (not SwiGLU)
- LayerNorm without bias (no_bias=True)
- Standard Multi-Head Attention (not GQA)

ALiBi implementation strategy:
  NXDI has no native ALiBi support. We store per-head ALiBi slopes as a
  weight parameter (alibi_slopes) that gets TP-sharded just like attention
  heads. At runtime we compute the position bias from these slopes.
  Flash attention must be disabled since NKI kernels don't accept additive bias.
"""

import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import (
    manual_softmax,
    repeat_kv,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


def compute_alibi_slopes(num_heads, alibi_bias_max=8):
    """Compute ALiBi slopes for all heads.

    Returns: tensor of shape (num_heads,) with per-head slopes.
    """
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(
        1, num_heads_power_of_2 + 1, dtype=torch.float32
    )
    base = base * (alibi_bias_max / num_heads_power_of_2)
    slopes = 1.0 / torch.pow(2, base)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]

    return slopes


class MptInferenceConfig(InferenceConfig):
    """Configuration for MPT model inference on NeuronX."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.sliding_window = None

        if not hasattr(self, "intermediate_size") or self.intermediate_size is None:
            self.intermediate_size = self.expansion_ratio * self.hidden_size

        # MPT uses MHA
        if not hasattr(self, "num_key_value_heads") or self.num_key_value_heads is None:
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
            "max_position_embeddings",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "MptInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            params = json.load(f)

        attn_config = params.get("attn_config", {})

        config_dict = {
            "hidden_size": params.get("d_model", 4096),
            "num_attention_heads": params.get("n_heads", 32),
            "num_key_value_heads": params.get("n_heads", 32),
            "num_hidden_layers": params.get("n_layers", 32),
            "vocab_size": params.get("vocab_size", 50432),
            "max_position_embeddings": params.get("max_seq_len", 2048),
            "intermediate_size": params.get("expansion_ratio", 4) * params.get("d_model", 4096),
            "expansion_ratio": params.get("expansion_ratio", 4),
            "layer_norm_epsilon": params.get("layer_norm_epsilon", 1e-5),
            "no_bias": params.get("no_bias", True),
            "alibi": attn_config.get("alibi", True),
            "alibi_bias_max": attn_config.get("alibi_bias_max", 8),
            "clip_qkv": attn_config.get("clip_qkv", None),
            "softmax_scale": attn_config.get("softmax_scale", None),
            "bos_token_id": params.get("bos_token_id", None),
            "eos_token_id": params.get("eos_token_id", None),
        }

        config_dict.update(kwargs)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronMptAttention(NeuronAttentionBase):
    """MPT attention with ALiBi position biases for NeuronX.

    ALiBi slopes are stored as a parameter (alibi_slopes) with shape
    (local_num_heads,). The slopes get TP-sharded via convert_hf_to_neuron_state_dict
    so each rank holds only its local heads' slopes.

    At runtime we compute: bias[h, pos] = alibi_slopes[h] * (pos - seq_len + 1)
    """

    def __init__(self, config: MptInferenceConfig, layer_idx: int = None, **kwargs):
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        no_bias = getattr(config, "no_bias", True)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            head_dim=head_dim,
            rotary_emb=None,  # ALiBi, not RoPE
            clip_qkv=getattr(config, "clip_qkv", None),
            qkv_bias=not no_bias,
            o_bias=not no_bias,
            sliding_window=None,
            **kwargs,
        )

        # Register ALiBi slopes as a parameter so they get TP-sharded.
        # Shape: (local_num_heads,) after sharding.
        # Initialized with placeholder; actual values come from state dict.
        self.alibi_slopes = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32),
            requires_grad=False,
        )

    def _build_alibi_bias(self, seq_length, device):
        """Build ALiBi bias from stored slopes.

        slopes shape: (local_num_heads,)
        Returns: (1, local_num_heads, 1, seq_length) for broadcasting.
        """
        # positions: [1-seq_length, 2-seq_length, ..., 0]
        positions = torch.arange(
            1 - seq_length, 1, dtype=torch.float32, device=device
        )
        # slopes: (local_num_heads,) -> (local_num_heads, 1)
        slopes = self.alibi_slopes.unsqueeze(-1)
        # bias: (local_num_heads, seq_length) -> (1, local_num_heads, 1, seq_length)
        bias = slopes * positions.unsqueeze(0)
        return bias.unsqueeze(0).unsqueeze(2)

    def scaled_qk(self, Q, K, attention_mask):
        """Override: add ALiBi bias to attention scores during prefill.

        Q shape: (batch, local_num_heads, q_len, head_dim)
        K shape: (batch, local_num_heads, k_len, head_dim)
        """
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Add ALiBi position bias
        k_len = K.shape[2]
        alibi = self._build_alibi_bias(k_len, Q.device)
        QK = QK + alibi

        if attention_mask is not None:
            QK = torch.where(attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min)

        return QK

    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask,
        is_prefix_caching=False,
    ):
        """Override: add ALiBi bias to prior and active scores during token gen."""
        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1

        # -- Prior (cached) KV --
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        if not self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)
        prior_scores = torch.matmul(Q, K_prior) / math.sqrt(self.head_dim)

        # Add ALiBi to prior scores
        prior_k_len = prior_scores.shape[-1]
        alibi_prior = self._build_alibi_bias(prior_k_len, Q.device)
        prior_scores = prior_scores + alibi_prior

        # Pad attention mask if KV cache is padded
        if prior_scores.shape[-1] > attention_mask.shape[-1] and self.neuron_config.apply_seq_ids_mask:
            attention_mask = F.pad(
                attention_mask, (0, prior_scores.shape[-1] - attention_mask.shape[-1]), "constant", 0
            )

        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # -- Active (current) KV --
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)

        # ALiBi for active token: bias = 0 (most recent position)

        if is_speculation or is_prefix_caching:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        # -- Combine via manual softmax --
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )

        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output


class NeuronMptMLP(nn.Module):
    """MPT MLP: up_proj -> GELU -> down_proj."""

    def __init__(self, config: MptInferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        no_bias = getattr(config, "no_bias", True)

        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not no_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        self.act = nn.GELU(approximate="none")

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=not no_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.act(self.up_proj(hidden_states))
        hidden_states = self.down_proj(hidden_states)
        return hidden_states, None


class NeuronMptBlock(nn.Module):
    """MPT transformer block: LN -> Attn -> Residual -> LN -> MLP -> Residual."""

    def __init__(self, config: MptInferenceConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        no_bias = getattr(config, "no_bias", True)

        self.norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if no_bias:
            self.norm_1.bias = None

        self.attn = NeuronMptAttention(
            config, layer_idx=layer_idx, tensor_model_parallel_group=get_tp_group(config)
        )

        self.norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if no_bias:
            self.norm_2.bias = None

        self.ffn = NeuronMptMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        mlp_output, _ = self.ffn(hidden_states)
        hidden_states = mlp_output + residual

        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronMptModel(NeuronBaseModel):
    """MPT base model for NeuronX."""

    def setup_attr_for_model(self, config: MptInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MptInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
            bias=False,
            pad=True,
        )

        self.layers = nn.ModuleList([
            NeuronMptBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        no_bias = getattr(config, "no_bias", True)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if no_bias:
            self.norm.bias = None


class NeuronMptForCausalLM(NeuronBaseForCausalLM):
    """MPT model with language modeling head for NeuronX."""

    _model_cls = NeuronMptModel
    _STATE_DICT_MODEL_PREFIX = "transformer."

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Convert HF MPT weights to NeuronX format.

        After framework strips 'transformer.' prefix, keys look like:
          wte.weight, norm_f.weight,
          blocks.{i}.attn.Wqkv.weight, blocks.{i}.attn.out_proj.weight,
          blocks.{i}.ffn.up_proj.weight, blocks.{i}.ffn.down_proj.weight,
          blocks.{i}.norm_1.weight, blocks.{i}.norm_2.weight
        """
        neuron_state_dict = {}
        tp_degree = config.neuron_config.tp_degree
        num_heads = config.num_attention_heads
        alibi_bias_max = getattr(config, "alibi_bias_max", 8)

        # Compute ALiBi slopes for all heads
        full_slopes = compute_alibi_slopes(num_heads, alibi_bias_max)

        # Token embeddings
        if "wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["wte.weight"]

        # Final layer norm
        if "norm_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["norm_f.weight"]
        if "norm_f.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["norm_f.bias"]

        for i in range(config.num_hidden_layers):
            hf = f"blocks.{i}"
            nxd = f"layers.{i}"

            # Layer norms
            for ln_name in ["norm_1", "norm_2"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.{ln_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.{ln_name}.{param}"] = state_dict[key]

            # Attention: split fused Wqkv into separate Q, K, V
            wqkv_key = f"{hf}.attn.Wqkv.weight"
            if wqkv_key in state_dict:
                wqkv = state_dict[wqkv_key]
                q_weight, k_weight, v_weight = wqkv.chunk(3, dim=0)
                neuron_state_dict[f"{nxd}.attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"{nxd}.attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"{nxd}.attn.qkv_proj.v_proj.weight"] = v_weight

            # Output projection (GQA_O wraps as o_proj.o_proj internally)
            out_key = f"{hf}.attn.out_proj.weight"
            if out_key in state_dict:
                neuron_state_dict[f"{nxd}.attn.o_proj.o_proj.weight"] = state_dict[out_key]

            # ALiBi slopes: same for all layers, sharded across TP ranks
            # Shape: (num_heads,) -> each TP rank gets (num_heads/tp_degree,)
            neuron_state_dict[f"{nxd}.attn.alibi_slopes"] = full_slopes.clone()

            # rank_util for attention
            neuron_state_dict[f"{nxd}.attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

            # MLP
            for layer_name in ["up_proj", "down_proj"]:
                for param in ["weight", "bias"]:
                    key = f"{hf}.ffn.{layer_name}.{param}"
                    if key in state_dict:
                        neuron_state_dict[f"{nxd}.ffn.{layer_name}.{param}"] = state_dict[key]

        # LM head (tied to embeddings)
        if "wte.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["wte.weight"].clone()

        # rank_util for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @classmethod
    def get_config_cls(cls):
        return MptInferenceConfig
