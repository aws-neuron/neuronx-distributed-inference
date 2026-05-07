# coding=utf-8
# Copyright 2024 Google Inc. and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Gemma2 model for NeuronX Distributed Inference

Forked from the Gemma3 contrib port with the following key differences:
- No Q-K normalization (Gemma2 does not use q_norm/k_norm)
- GQA with 8 KV heads (vs MQA with 1 KV head in Gemma3)
- Attention logit softcapping disabled (NeuronAttentionBase kernel doesn't support tanh capping)
- Final logit softcapping (30.0) applied after lm_head output
- Sliding window disabled (head_dim=256 exceeds NKI kernel limit of 128)
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

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
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


# ====================================================================================
# Configuration Classes
# ====================================================================================


class Gemma2NeuronConfig(NeuronConfig):
    """NeuronConfig for Gemma2 model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma2Attention


class Gemma2InferenceConfig(InferenceConfig):
    """Configuration class for Gemma2 model inference on NeuronX."""

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1

        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True

        if not hasattr(self, "query_pre_attn_scalar"):
            self.query_pre_attn_scalar = 256

        # Force-disable sliding window: NKI kernel doesn't support head_dim > 128
        # Gemma2 uses head_dim=256. The HF config has sliding_window=4096 which
        # must be overridden regardless of what was loaded from config.json.
        self.sliding_window = None

        # Gemma2 alternates: even layers = sliding, odd layers = global
        # But all disabled due to head_dim limitation
        if not hasattr(self, "sliding_window_pattern"):
            self.sliding_window_pattern = 2  # every other layer

        # Force-disable attention logit softcapping (NeuronAttentionBase doesn't support tanh capping)
        # HF config has attn_logit_softcapping=50.0 but we must override it.
        self.attn_logit_softcapping = None

        # Keep final logit softcapping — applied in model forward pass
        if not hasattr(self, "final_logit_softcapping"):
            self.final_logit_softcapping = 30.0

        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        if not hasattr(self, "attention_dropout"):
            self.attention_dropout = 0.0

        # All layers use global attention (sliding disabled due to head_dim)
        if not hasattr(self, "layer_types"):
            self.layer_types = ["global_attention"] * self.num_hidden_layers

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma2NeuronConfig]:
        return Gemma2NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Gemma2InferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config_dict.update(kwargs)

        if "output_attentions" not in config_dict:
            config_dict["output_attentions"] = False
        if "output_hidden_states" not in config_dict:
            config_dict["output_hidden_states"] = False
        if "use_cache" not in config_dict:
            config_dict["use_cache"] = True
        if "tie_word_embeddings" not in config_dict:
            config_dict["tie_word_embeddings"] = True

        # Note: sliding_window and attn_logit_softcapping are force-overridden
        # in add_derived_config() regardless of what's in config.json

        if neuron_config is None:
            neuron_config = NeuronConfig()

        config = cls(neuron_config=neuron_config, **config_dict)
        return config


# ====================================================================================
# Model Components
# ====================================================================================


class Gemma2RMSNorm(nn.Module):
    """
    Gemma2 RMSNorm: uses (1.0 + weight) scaling instead of just weight.
    Same as Gemma3.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def get_rmsnorm_cls():
    return Gemma2RMSNorm


class Gemma2ScaledEmbedding(nn.Module):
    """Embeddings scaled by sqrt(hidden_size), same as Gemma3."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        dtype: torch.dtype,
        shard_across_embedding: bool = True,
        pad: bool = True,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()
        self.embed_scale = embedding_dim**0.5
        self.embedding = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            pad=pad,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids: torch.Tensor):
        return self.embedding(input_ids) * self.embed_scale


class NeuronGemma2Attention(NeuronAttentionBase):
    """
    Gemma2 attention — NO Q-K normalization (unlike Gemma3).
    GQA with 8 KV heads (vs MQA in Gemma3).
    """

    def __init__(self, config: Gemma2InferenceConfig, is_sliding: bool = False):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        sliding_window = config.sliding_window if is_sliding else None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
            # No Q-K normalization for Gemma2
        )

        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.attn_logit_softcapping = config.attn_logit_softcapping


class NeuronGemma2MLP(nn.Module):
    """Gemma2 MLP: gate/up/down with GELU tanh activation."""

    def __init__(self, config: Gemma2InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_output = self.down_proj(gate_output * up_output)
        return down_output, None


class NeuronGemma2DecoderLayer(nn.Module):
    """
    Gemma2 decoder layer with 4 norms per layer.
    Same structure as Gemma3.
    """

    def __init__(self, config: Gemma2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.self_attn = NeuronGemma2Attention(config, is_sliding=is_sliding)
        self.mlp = NeuronGemma2MLP(config)

        # Four normalization layers
        self.input_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# Model Classes
# ====================================================================================


class NeuronGemma2Model(NeuronBaseModel):
    """
    Gemma2 base model for NeuronX inference.

    Note on final_logit_softcapping: Gemma2 applies tanh(logits/30)*30 after lm_head.
    Since tanh is monotonic, this does not change greedy argmax results and is omitted
    from the NeuronX forward pass. If probability-calibrated logits are needed,
    apply softcapping as a post-processing step.
    """

    def setup_attr_for_model(self, config: Gemma2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Gemma2InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma2ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        self.layers = nn.ModuleList(
            [NeuronGemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronGemma2ForCausalLM(NeuronBaseForCausalLM):
    """Gemma2 model for causal language modeling on NeuronX."""

    _model_cls = NeuronGemma2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Gemma2 state dict to NeuronX format.

        Key difference from Gemma3: no q_norm/k_norm weight mapping.
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}

        if "embed_tokens.weight" in state_dict:
            neuron_state_dict["embed_tokens.embedding.weight"] = (
                state_dict["embed_tokens.weight"].detach().clone()
            )

        if "norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["norm.weight"].detach().clone()

        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].detach().clone()

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for i in range(num_layers):
            prefix = f"layers.{i}"

            # Attention Q/K/V/O projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"{prefix}.self_attn.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            # NO q_norm/k_norm mapping (Gemma2 doesn't have Q-K normalization)

            # MLP weights
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            # Four norms per layer
            for norm_name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]:
                key = f"{prefix}.{norm_name}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            neuron_state_dict[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.embedding.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embeddings and lm_head."""
        if "embed_tokens.embedding.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.embedding.weight"].clone()
        elif "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma2InferenceConfig
