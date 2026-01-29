# coding=utf-8
# Copyright 2024 OpenBMB and HuggingFace Inc. team. All rights reserved.
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
PyTorch MiniCPM model for NXD inference
Based on transformers/src/transformers/models/minicpm/modeling_minicpm.py
"""
from typing import List, Optional, Tuple, Type
import math

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn

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


class MiniCPMLongRoPE(RotaryEmbedding):
    """
    MiniCPM LongRoPE implementation for NeuronX.
    Applies position-dependent scaling factors to the inverse frequencies.
    Based on HuggingFace MiniCPMLongRoPE.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=32768,
    ):
        super().__init__(dim, max_position_embeddings, base)
        self.short_factor = torch.tensor(short_factor, dtype=torch.float32) if short_factor else None
        self.long_factor = torch.tensor(long_factor, dtype=torch.float32) if long_factor else None
        self.original_max_position_embeddings = original_max_position_embeddings

        # Compute scaling factor as in HF implementation
        scale = max_position_embeddings / original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    def get_inv_freqs(self, device=None):
        """Get inverse frequencies with LongRoPE scaling factors applied."""
        # Base inverse frequencies
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        base_inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))
        return base_inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = self.get_inv_freqs(x.device)

        seq_len = position_ids.shape[-1]

        # Choose factors based on sequence length
        if seq_len > self.original_max_position_embeddings:
            ext_factors = self.long_factor.to(x.device) if self.long_factor is not None else torch.ones_like(self.inv_freq)
        else:
            ext_factors = self.short_factor.to(x.device) if self.short_factor is not None else torch.ones_like(self.inv_freq)

        # Apply LongRoPE: freqs = outer(t, 1/ext_factors) * inv_freq
        # Equivalent to modifying inv_freq: scaled_inv_freq = inv_freq / ext_factors
        scaled_inv_freq = self.inv_freq / ext_factors

        inv_freq_expanded = scaled_inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply scaling factor to cos/sin
        cos = emb.cos() * self.scaling_factor
        sin = emb.sin() * self.scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiniCPMNeuronConfig(NeuronConfig):
    """Custom Neuron configuration for MiniCPM - REQUIRED for token generation"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronMiniCPMAttention


class MiniCPMInferenceConfig(InferenceConfig):
    """Configuration class for MiniCPM inference on NeuronX"""

    def add_derived_config(self):
        """Add derived configuration parameters required by framework"""
        self.num_cores_per_group = 1

        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

        self.qkv_bias = getattr(self, 'attention_bias', False)
        self.o_bias = getattr(self, 'attention_bias', False)

        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

        # Handle rope_scaling for LongRoPE
        rope_scaling = getattr(self, 'rope_scaling', None)
        if rope_scaling and isinstance(rope_scaling, dict):
            self.rope_type = rope_scaling.get('rope_type', 'default')
            self.rope_short_factor = rope_scaling.get('short_factor', None)
            self.rope_long_factor = rope_scaling.get('long_factor', None)
            self.original_max_position_embeddings = rope_scaling.get(
                'original_max_position_embeddings', self.max_position_embeddings
            )
        else:
            self.rope_type = 'default'
            self.rope_short_factor = None
            self.rope_long_factor = None
            self.original_max_position_embeddings = self.max_position_embeddings

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MiniCPMNeuronConfig]:
        """Return custom NeuronConfig class - CRITICAL for token generation"""
        return MiniCPMNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from pretrained model"""
        import json
        import os
        
        neuron_config = kwargs.pop("neuron_config", None)
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 4096),
            "num_attention_heads": hf_config.get("num_attention_heads", 32),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 32),
            "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", 32)),
            "vocab_size": hf_config.get("vocab_size", 32000),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 2048),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "intermediate_size": hf_config.get("intermediate_size", 11008),
            "attention_bias": hf_config.get("attention_bias", False),
            "scale_emb": hf_config.get("scale_emb", 1),
            "dim_model_base": hf_config.get("dim_model_base", 1),
            "scale_depth": hf_config.get("scale_depth", 1),
            "pad_token_id": hf_config.get("pad_token_id"),
            "rope_scaling": hf_config.get("rope_scaling", None),
        }
        
        config_dict.update(kwargs)
        
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronMiniCPMAttention(NeuronAttentionBase):
    """
    MiniCPM attention using NeuronAttentionBase
    Based on transformers MiniCPMAttention
    """

    def __init__(self, config: MiniCPMInferenceConfig):
        # Use LongRoPE if config specifies it
        rope_type = getattr(config, 'rope_type', 'default')
        if rope_type == 'longrope' and hasattr(config, 'rope_short_factor') and config.rope_short_factor:
            rotary_emb = MiniCPMLongRoPE(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                short_factor=config.rope_short_factor,
                long_factor=config.rope_long_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
            )
        else:
            rotary_emb = RotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
        )


class NeuronMiniCPMDecoderLayer(nn.Module):
    """
    MiniCPM decoder layer with NeuronX components
    Based on transformers MiniCPMDecoderLayer
    """

    def __init__(self, config: MiniCPMInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniCPMAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

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
        
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronMiniCPMModel(NeuronBaseModel):
    """
    MiniCPM base model for NeuronX
    Based on transformers MiniCPMModel
    """

    def setup_attr_for_model(self, config: MiniCPMInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MiniCPMInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.scale_emb = config.scale_emb
        self.dim_model_base = config.dim_model_base

        self._embed_tokens_base = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        self.layers = nn.ModuleList(
            [NeuronMiniCPMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Create a custom lm_head wrapper that applies scaling
        self._lm_head_base = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )
    
    @property
    def embed_tokens(self):
        """Property to apply MiniCPM scaling to embeddings"""
        class ScaledEmbedding(nn.Module):
            def __init__(self, embed, scale_emb):
                super().__init__()
                self._embed = embed
                self.scale_emb = scale_emb
            
            def forward(self, input_ids, **kwargs):
                return self._embed(input_ids, **kwargs) * self.scale_emb
        
        return ScaledEmbedding(self._embed_tokens_base, self.scale_emb)
    
    @property
    def lm_head(self):
        """Property to apply MiniCPM scaling before lm_head"""
        class ScaledLMHead(nn.Module):
            def __init__(self, lm_head, hidden_size, dim_model_base):
                super().__init__()
                self._lm_head = lm_head
                self.hidden_size = hidden_size
                self.dim_model_base = dim_model_base
                self.gather_output = lm_head.gather_output
                self.tensor_parallel_group = lm_head.tensor_parallel_group
                if hasattr(lm_head, 'pad_size'):
                    self.pad_size = lm_head.pad_size
            
            def forward(self, hidden_states):
                scaled_hidden = hidden_states / (self.hidden_size / self.dim_model_base)
                return self._lm_head(scaled_hidden)
        
        return ScaledLMHead(self._lm_head_base, self.hidden_size, self.dim_model_base)


class NeuronMiniCPMForCausalLM(NeuronBaseForCausalLM):
    """
    MiniCPM causal language model for NeuronX inference
    """

    _model_cls = NeuronMiniCPMModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert HuggingFace weights to NeuronX format"""
        neuron_config = config.neuron_config
        
        # Debug: Print first few keys to understand structure
        print(f"DEBUG: First 10 keys received: {list(state_dict.keys())[:10]}")
        
        neuron_state_dict = {}
        
        # First pass: Copy all weights
        for key, value in state_dict.items():
            neuron_state_dict[key] = value
        
        # Second pass: Restructure QKV weights for non-fused attention
        # The framework expects qkv_proj.q_proj structure when fused_qkv=False
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            # Check if this layer has separate Q/K/V projections
            q_key = f"layers.{i}.self_attn.q_proj.weight"
            k_key = f"layers.{i}.self_attn.k_proj.weight"
            v_key = f"layers.{i}.self_attn.v_proj.weight"
            
            if q_key in neuron_state_dict:
                # Pop original keys
                q_weight = neuron_state_dict.pop(q_key)
                k_weight = neuron_state_dict.pop(k_key)
                v_weight = neuron_state_dict.pop(v_key)
                
                # Add with qkv_proj intermediate level
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.weight"] = v_weight
                
                # Note: o_proj stays as is - it's not part of qkv_proj
        
        # Handle embed_tokens weight mapping for MiniCPM's scaled embeddings
        if "embed_tokens.weight" in neuron_state_dict:
            neuron_state_dict["_embed_tokens_base.weight"] = neuron_state_dict.pop("embed_tokens.weight")
        
        # Handle lm_head weight mapping for MiniCPM's scaled lm_head
        if "lm_head.weight" in neuron_state_dict:
            neuron_state_dict["_lm_head_base.weight"] = neuron_state_dict.pop("lm_head.weight")
        
        # Add rank utilities for distributed training
        if neuron_config.vocab_parallel:
            neuron_state_dict["_embed_tokens_base.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Update state dict for tied weights between embed_tokens and lm_head"""
        state_dict["_lm_head_base.weight"] = state_dict["_embed_tokens_base.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return MiniCPMInferenceConfig

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args


def convert_state_dict_to_fused_qkv(state_dict: dict, config: InferenceConfig) -> dict:
    """Convert separate Q, K, V weights to fused QKV format"""
    num_layers = config.num_hidden_layers
    
    for i in range(num_layers):
        q_weight = state_dict.pop(f"layers.{i}.self_attn.q_proj.weight")
        k_weight = state_dict.pop(f"layers.{i}.self_attn.k_proj.weight")
        v_weight = state_dict.pop(f"layers.{i}.self_attn.v_proj.weight")
        
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        state_dict[f"layers.{i}.self_attn.qkv_proj.weight"] = qkv_weight
        
        if config.qkv_bias:
            q_bias = state_dict.pop(f"layers.{i}.self_attn.q_proj.bias")
            k_bias = state_dict.pop(f"layers.{i}.self_attn.k_proj.bias")
            v_bias = state_dict.pop(f"layers.{i}.self_attn.v_proj.bias")
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            state_dict[f"layers.{i}.self_attn.qkv_proj.bias"] = qkv_bias
    
    return state_dict
