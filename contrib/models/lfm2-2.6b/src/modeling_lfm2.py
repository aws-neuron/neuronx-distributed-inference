# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied,
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch LFM2 model for NXD inference
"""
from typing import List, Optional, Tuple, Type

import torch
import gc
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class Lfm2NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronLfm2Attention


class Lfm2InferenceConfig(InferenceConfig):

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = False
        self.o_bias = False

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "norm_eps",
            "intermediate_size",
            "layer_types",
            "conv_L_cache",
            "conv_bias",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Lfm2NeuronConfig]:
        return Lfm2NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        import os
        import json
        
        neuron_config = kwargs.pop("neuron_config", None)
        
        model_path = os.path.expanduser(model_path)
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        final_config = {
            "hidden_size": config_dict.get("hidden_size", 2048),
            "num_attention_heads": config_dict.get("num_attention_heads", 32),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 30),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 8),
            "vocab_size": config_dict.get("vocab_size", 65536),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 128000),
            "norm_eps": config_dict.get("norm_eps", 1e-05),
            "intermediate_size": config_dict.get("intermediate_size", 10752),
            "layer_types": config_dict.get("layer_types", ["conv"] * 30),
            "conv_L_cache": config_dict.get("conv_L_cache", 3),
            "conv_bias": config_dict.get("conv_bias", False),
            "rope_theta": config_dict.get("rope_theta", 1000000.0),
            "pad_token_id": config_dict.get("pad_token_id", 0),
            "bos_token_id": config_dict.get("bos_token_id", 1),
            "eos_token_id": config_dict.get("eos_token_id", 7),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", True),
            "hidden_act": "silu",
            "rms_norm_eps": config_dict.get("norm_eps", 1e-05),
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True,
        }
        
        final_config.update(kwargs)
        
        config = cls(neuron_config=neuron_config, **final_config)
        return config


class NeuronLfm2Attention(NeuronAttentionBase):

    def __init__(self, config: Lfm2InferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 1000000.0),
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
        )
        
        self.q_layernorm = get_rmsnorm_cls()(head_dim, eps=config.norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(head_dim, eps=config.norm_eps)


class NeuronLfm2ShortConv(nn.Module):

    def __init__(self, config: Lfm2InferenceConfig):
        super().__init__()
        self.config = config
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        self.hidden_size = config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        seqlen = hidden_states.shape[1]
        
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        
        Bx = B * x
        conv_out = self.conv(Bx)[..., :seqlen]
        
        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        
        # Conv layers don't use KV cache, return past_key_value unchanged
        return y, past_key_value, None, None, None


class NeuronLfm2DecoderLayer(nn.Module):

    def __init__(self, config: Lfm2InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = NeuronLfm2Attention(config)
        else:
            self.conv = NeuronLfm2ShortConv(config)
        
        self.mlp = NeuronLlamaMLP(config)
        self.operator_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_eps)
        
        # For conv layers, store dimensions for dummy KV cache
        if not self.is_attention_layer:
            self.num_key_value_heads = config.num_key_value_heads
            self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)

        if self.is_attention_layer:
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        else:
            hidden_states, _, _, _, _ = self.conv(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                **kwargs,
            )
            # Create dummy KV cache for conv layers
            bsz, seq_len, _ = hidden_states.shape
            dummy_k = torch.zeros(bsz, self.num_key_value_heads, seq_len, self.head_dim, 
                                 dtype=hidden_states.dtype, device=hidden_states.device)
            dummy_v = torch.zeros(bsz, self.num_key_value_heads, seq_len, self.head_dim,
                                 dtype=hidden_states.dtype, device=hidden_states.device)
            present_key_value = (dummy_k, dummy_v)
            cos_cache = None
            sin_cache = None
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronLfm2Model(NeuronBaseModel):

    def setup_attr_for_model(self, config: Lfm2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Lfm2InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronLfm2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronLfm2ForCausalLM(NeuronBaseForCausalLM):

    _model_cls = NeuronLfm2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_config = config.neuron_config

        # Rename HF keys to match Neuron model structure
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            if key.startswith('model.'):
                key = key[6:]
            
            # Rename embedding_norm to norm
            if key == 'embedding_norm.weight':
                key = 'norm.weight'
            
            # Rename out_proj to o_proj for attention layers
            if '.self_attn.out_proj.' in key:
                key = key.replace('.self_attn.out_proj.', '.self_attn.o_proj.')
            
            # Rename feed_forward to mlp and map w1/w2/w3 to gate_proj/down_proj/up_proj
            if '.feed_forward.' in key:
                key = key.replace('.feed_forward.', '.mlp.')
                # w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
                if '.mlp.w1.' in key:
                    key = key.replace('.mlp.w1.', '.mlp.gate_proj.')
                elif '.mlp.w2.' in key:
                    key = key.replace('.mlp.w2.', '.mlp.down_proj.')
                elif '.mlp.w3.' in key:
                    key = key.replace('.mlp.w3.', '.mlp.up_proj.')
            
            new_state_dict[key] = value
        
        state_dict = new_state_dict

        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            if config.layer_types[i] == "full_attention":
                state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                    0, tp_degree, dtype=torch.int32
                )

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Lfm2InferenceConfig

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict, cfg: InferenceConfig):
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):
        if cfg.layer_types[l] == "full_attention":
            _helper_concat_and_delete_qkv(state_dict, l, "weight")
            if (
                cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
            ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
                _helper_concat_and_delete_qkv(state_dict, l, "scale")

    gc.collect()

    return state_dict
