# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Qwen3 model for NXD inference
"""
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return Qwen3RMSNorm if cpu_mode() else CustomRMSNorm


class Qwen3NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attn_cls = NeuronQwen3Attention


class Qwen3InferenceConfig(InferenceConfig):

    def add_derived_config(self):
        self.num_cores_per_group = 1

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
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen3NeuronConfig]:
        return Qwen3NeuronConfig


class NeuronQwen3Attention(NeuronAttentionBase):

    def __init__(self, config: Qwen3InferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            q_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
        )


class NeuronQwen3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Qwen3InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3Attention(config)
        self.mlp = NeuronLlamaMLP(config)  # can reuse LlamaMLP module
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

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

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen3Model(NeuronBaseModel):

    def setup_attr_for_model(self, config: Qwen3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen3InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        self.layers = nn.ModuleList(
            [NeuronQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


class NeuronQwen3ForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as Qwen3ForCausalLM
    """

    _model_cls = NeuronQwen3Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3ForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            # To facilitate rank usage in attention
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # Rename q_norm and k_norm
            state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                state_dict[f"layers.{i}.self_attn.q_norm.weight"].detach().clone()
            )
            state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                state_dict[f"layers.{i}.self_attn.k_norm.weight"].detach().clone()
            )
        # To facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Qwen3InferenceConfig
