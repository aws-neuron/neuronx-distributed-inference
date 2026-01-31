# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch InternLM3 model for NXD inference."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


class InternLM3RMSNorm(nn.Module):
    """
    InternLM3 RMSNorm implementation for Neuron.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3RMSNorm
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM3MLP(nn.Module):
    """
    InternLM3 MLP implementation for Neuron using parallel layers.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3MLP
    """
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_proj = self.down_proj(gate_output * up_output)
        return down_proj


class InternLM3Attention(NeuronAttentionBase):
    """
    InternLM3 Attention implementation for Neuron using GQA.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3Attention
    """
    def __init__(self, config: InferenceConfig, layer_idx: Optional[int] = None):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_emb = RotaryEmbedding(
            head_dim,
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
            num_cores_per_group=1,
            qkv_bias=config.qkv_bias,
            o_bias=config.bias,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.layer_idx = layer_idx


class InternLM3DecoderLayer(nn.Module):
    """
    InternLM3 Decoder Layer implementation for Neuron.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3DecoderLayer
    """
    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = InternLM3Attention(config=config, layer_idx=layer_idx)
        self.mlp = InternLM3MLP(config)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class InternLM3Model(NeuronBaseModel):
    """
    InternLM3 Model implementation for Neuron.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3Model
    """
    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        
        self.layers = nn.ModuleList(
            [InternLM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )


class InternLM3ForCausalLM(NeuronBaseForCausalLM):
    """
    InternLM3 For Causal LM implementation for Neuron.
    Reference: transformers/src/transformers/models/internlm3/modeling_internlm3.py::InternLM3ForCausalLM
    """
    _model_cls = InternLM3Model

    @staticmethod
    def convert_hf_to_neuron_state_dict(hf_state_dict, config: InferenceConfig):
        """
        Convert HuggingFace state dict to Neuron state dict format.
        """
        neuron_state_dict = {}
        
        for key, value in hf_state_dict.items():
            new_key = key
            
            if config.neuron_config.fused_qkv and "self_attn" in key and any(x in key for x in ["q_proj", "k_proj", "v_proj"]):
                continue
            
            neuron_state_dict[new_key] = value
        
        if config.neuron_config.fused_qkv:
            for layer_idx in range(config.num_hidden_layers):
                q_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
                k_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
                v_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
                
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                neuron_state_dict[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"] = qkv_weight
                
                if config.qkv_bias:
                    q_bias = hf_state_dict.get(f"model.layers.{layer_idx}.self_attn.q_proj.bias")
                    k_bias = hf_state_dict.get(f"model.layers.{layer_idx}.self_attn.k_proj.bias")
                    v_bias = hf_state_dict.get(f"model.layers.{layer_idx}.self_attn.v_proj.bias")
                    if q_bias is not None:
                        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                        neuron_state_dict[f"model.layers.{layer_idx}.self_attn.qkv_proj.bias"] = qkv_bias
        
        return neuron_state_dict
