# coding=utf-8
# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Persimmon model for NeuronX Distributed Inference.

Key architectural differences from Llama:
- Fused QKV projection (query_key_value)
- QK LayerNorm after projection
- Partial rotary embeddings (partial_rotary_factor=0.5)
- Standard LayerNorm (not RMSNorm)
- relu2 activation (relu squared)
"""

import logging
from typing import List, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


# Persimmon uses relu2 activation (relu squared)
def relu2(x):
    return torch.relu(x) ** 2


class PersimmonNeuronConfig(NeuronConfig):
    """Custom NeuronConfig for Persimmon - REQUIRED for token generation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from modeling_persimmon import NeuronPersimmonAttention
        self.attn_cls = NeuronPersimmonAttention
        self.qk_layernorm = True  # Persimmon uses QK LayerNorm


class PersimmonInferenceConfig(InferenceConfig):
    """Inference configuration for Persimmon model."""
    
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                self.num_attention_heads, self.num_key_value_heads, self.neuron_config.tp_degree
            )
        # Framework-required attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
    
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "layer_norm_eps",
            "hidden_act",
            "intermediate_size",
            "partial_rotary_factor",
            "qk_layernorm",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return PersimmonNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from pretrained model directory."""
        import json
        import os
        
        neuron_config = kwargs.pop("neuron_config", None)
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace config to our config
        # Persimmon uses num_attention_heads for both Q and KV (no GQA)
        num_attention_heads = hf_config.get("num_attention_heads", 64)
        
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 4096),
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": hf_config.get("num_key_value_heads", num_attention_heads),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 36),
            "vocab_size": hf_config.get("vocab_size", 262144),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 16384),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "hidden_act": hf_config.get("hidden_act", "relu2"),
            "intermediate_size": hf_config.get("intermediate_size", 16384),
            "partial_rotary_factor": hf_config.get("partial_rotary_factor", 0.5),
            "qk_layernorm": hf_config.get("qk_layernorm", True),
            "rope_theta": hf_config.get("rope_theta", 25000.0),
            "pad_token_id": hf_config.get("pad_token_id", None),
            "bos_token_id": hf_config.get("bos_token_id", 1),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
        }
        
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronPersimmonMLP(nn.Module):
    """Persimmon MLP with relu2 activation."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        
        if parallel_state.model_parallel_is_initialized():
            self.dense_h_to_4h = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.dense_4h_to_h = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=get_tp_group(config),
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )
        else:
            self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
    
    def forward(self, x):
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        
        hidden_states = self.dense_h_to_4h(x)
        hidden_states = relu2(hidden_states)
        output = self.dense_4h_to_h(hidden_states)
        
        return output


class NeuronPersimmonAttention(NeuronAttentionBase):
    """
    Persimmon attention with:
    - Fused QKV projection
    - QK LayerNorm
    - Partial rotary embeddings (only applies RoPE to first half of head_dim)
    """
    
    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_ndims = int(head_dim * self.partial_rotary_factor)
        self._head_dim = head_dim
        
        # Persimmon uses same number of heads for Q and KV (no GQA)
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        
        # Create RoPE with rotary_ndims (partial dimension) - NOT full head_dim
        # This matches HuggingFace which computes inv_freq for dim=head_dim*partial_rotary_factor
        rotary_emb = RotaryEmbedding(
            self.rotary_ndims,  # Partial dimension for RoPE (32 for Persimmon)
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 25000.0),
        )
        
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            qkv_bias=True,  # Persimmon uses bias
            o_bias=True,
            o_proj_layer_name="dense",  # Persimmon uses 'dense' for output projection
        )
    
    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """
        Override to apply partial rotary embeddings.
        Persimmon only applies RoPE to the first half of head_dim.
        """
        from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
        
        # Get cos/sin from rotary embedding (already sized for rotary_ndims)
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        
        # Split Q and K into rotary and pass-through parts
        # Q shape: [batch, heads, seq, head_dim]
        q_rot = Q[..., :self.rotary_ndims]
        q_pass = Q[..., self.rotary_ndims:]
        k_rot = K[..., :self.rotary_ndims]
        k_pass = K[..., self.rotary_ndims:]
        
        # Apply RoPE only to the rotary part (cos/sin are already rotary_ndims sized)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_cache, sin_cache)
        
        # Concatenate back
        Q = torch.cat((q_rot, q_pass), dim=-1)
        K = torch.cat((k_rot, k_pass), dim=-1)
        
        return Q, K, cos_cache, sin_cache


class NeuronPersimmonDecoderLayer(nn.Module):
    """Persimmon decoder layer with LayerNorm (not RMSNorm)."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Use the attention class from neuron_config
        attn_cls = config.neuron_config.attn_cls
        if isinstance(attn_cls, str):
            attn_cls = NeuronPersimmonAttention
        
        self.self_attn = attn_cls(
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronPersimmonMLP(config)
        
        # Persimmon uses standard LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        hidden_states = residual + attn_output.hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronPersimmonModel(NeuronBaseModel):
    """Neuron-compatible Persimmon model."""
    
    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.layers = nn.ModuleList([
            NeuronPersimmonDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final LayerNorm - must be named 'norm' for framework compatibility
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class NeuronPersimmonForCausalLM(NeuronBaseForCausalLM):
    """Persimmon causal LM for NeuronX inference."""
    
    _model_cls = NeuronPersimmonModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Persimmon state dict to Neuron format.
        
        Key conversions:
        - model.embed_tokens -> embed_tokens
        - model.layers.X.self_attn.query_key_value -> layers.X.self_attn.qkv_proj (split to q/k/v)
        - model.layers.X.self_attn.dense -> layers.X.self_attn.o_proj
        - model.layers.X.self_attn.q_layernorm -> layers.X.self_attn.q_layernorm
        - model.layers.X.self_attn.k_layernorm -> layers.X.self_attn.k_layernorm
        - model.layers.X.mlp.dense_h_to_4h -> layers.X.mlp.dense_h_to_4h
        - model.layers.X.mlp.dense_4h_to_h -> layers.X.mlp.dense_4h_to_h
        - model.layers.X.input_layernorm -> layers.X.input_layernorm
        - model.layers.X.post_attention_layernorm -> layers.X.post_attention_layernorm
        - model.final_layernorm -> norm
        - lm_head -> lm_head
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        print(f"DEBUG: Converting Persimmon state dict with {len(state_dict)} keys")
        print(f"DEBUG: First 10 keys: {list(state_dict.keys())[:10]}")
        
        for key, value in state_dict.items():
            new_key = key
            
            # Remove "model." prefix
            if new_key.startswith("model."):
                new_key = new_key[6:]
            
            # Handle fused QKV projection - split into separate Q, K, V
            if "query_key_value" in new_key:
                layer_idx = new_key.split(".")[1]
                
                if "weight" in new_key:
                    # Shape: [3 * hidden_size, hidden_size] -> split into Q, K, V
                    # Persimmon stores as interleaved: [Q0, K0, V0, Q1, K1, V1, ...]
                    qkv_weight = value.view(num_heads, 3, head_dim, hidden_size)
                    q_weight = qkv_weight[:, 0, :, :].reshape(hidden_size, hidden_size)
                    k_weight = qkv_weight[:, 1, :, :].reshape(hidden_size, hidden_size)
                    v_weight = qkv_weight[:, 2, :, :].reshape(hidden_size, hidden_size)
                    
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.v_proj.weight"] = v_weight
                elif "bias" in new_key:
                    # Shape: [3 * hidden_size] -> split into Q, K, V
                    qkv_bias = value.view(num_heads, 3, head_dim)
                    q_bias = qkv_bias[:, 0, :].reshape(hidden_size)
                    k_bias = qkv_bias[:, 1, :].reshape(hidden_size)
                    v_bias = qkv_bias[:, 2, :].reshape(hidden_size)
                    
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.q_proj.bias"] = q_bias
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.k_proj.bias"] = k_bias
                    neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.v_proj.bias"] = v_bias
                continue
            
            # Rename dense -> o_proj.o_proj for output projection
            # The model has self_attn.o_proj (GroupQueryAttention_O) which has inner o_proj (RowParallelLinear)
            if "self_attn.dense" in new_key:
                new_key = new_key.replace("self_attn.dense", "self_attn.o_proj.o_proj")
            
            # Rename final_layernorm -> norm for framework compatibility
            if "final_layernorm" in new_key:
                new_key = new_key.replace("final_layernorm", "norm")
            
            neuron_state_dict[new_key] = value
        
        # Add rank utilities for tensor parallelism
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        print(f"DEBUG: Converted state dict has {len(neuron_state_dict)} keys")
        print(f"DEBUG: First 10 converted keys: {list(neuron_state_dict.keys())[:10]}")
        
        return neuron_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied embeddings if configured."""
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        return PersimmonInferenceConfig


__all__ = [
    "NeuronPersimmonForCausalLM",
    "NeuronPersimmonModel",
    "PersimmonInferenceConfig",
    "PersimmonNeuronConfig",
]
