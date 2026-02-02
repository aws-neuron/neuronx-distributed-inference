# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Qwen2.5-VL model for NeuronX Distributed Inference

This implementation focuses on the text model with MRoPE (Multimodal Rotary Position Embeddings).
Vision integration can be added in future iterations.
"""

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Import our MRoPE implementation
# Use absolute imports since this module may be loaded directly
import os
import sys
# Add current directory to path if not already there
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from mrope import Qwen2VLRotaryEmbedding, apply_multimodal_rotary_pos_emb
from config_qwen2vl import Qwen2VLInferenceConfig


def get_rmsnorm_cls():
    """Get the appropriate RMSNorm implementation"""
    # Use CustomRMSNorm for NXD, LlamaRMSNorm for CPU
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class NeuronQwen2VLAttention(NeuronAttentionBase):
    """
    Qwen2.5-VL attention implementation with MRoPE support
    
    Key features:
    - GQA (Grouped Query Attention) with configurable num_key_value_heads
    - MRoPE (Multimodal Rotary Position Embeddings) for 3D position encoding
    - Bias in QKV projections, no bias in output projection
    
    Note: For initial implementation, we use standard RoPE instead of MRoPE
    to simplify integration. MRoPE can be added in a future iteration.
    """
    
    def __init__(self, config: Qwen2VLInferenceConfig, layer_idx: Optional[int] = None):
        # For now, use standard rotary embeddings like Qwen2
        # TODO: Add full MRoPE support in future iteration
        from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
        
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Store layer idx for sliding window attention
        self.layer_idx = layer_idx
        self.config = config
        
        # Determine if this layer uses sliding window attention
        # Qwen2.5-VL has layer_types configuration
        if hasattr(config, 'layer_types') and layer_idx is not None:
            sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        else:
            # Default: no sliding window for initial implementation
            sliding_window = None
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
        )
        
        # Store MRoPE section configuration for future use
        self.mrope_section = config.mrope_section


class NeuronQwen2VLMLP(NeuronLlamaMLP):
    """
    Qwen2.5-VL MLP implementation
    
    Uses SwiGLU activation same as LLaMA, so we can reuse NeuronLlamaMLP
    Formula: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    pass


class NeuronQwen2VLDecoderLayer(nn.Module):
    """
    Qwen2.5-VL decoder layer
    
    Structure:
    - Input LayerNorm
    - Self Attention with MRoPE
    - Residual connection
    - Post-attention LayerNorm
    - MLP
    - Residual connection
    """
    
    def __init__(self, config: Qwen2VLInferenceConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention with MRoPE
        self.self_attn = NeuronQwen2VLAttention(config, layer_idx=layer_idx)
        
        # MLP (reuse LLaMA MLP since it's the same SwiGLU)
        self.mlp = NeuronQwen2VLMLP(config)
        
        # Layer norms
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
        """
        Forward pass for decoder layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices (can be 3D for MRoPE)
            past_key_value: Cached key/value pairs
        
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
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
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        # Return format matching framework expectations
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronQwen2VLTextModel(NeuronBaseModel):
    """
    Qwen2.5-VL text model (decoder-only)
    
    This is the core transformer model that processes text (and eventually multimodal) inputs.
    """
    
    def setup_attr_for_model(self, config: Qwen2VLInferenceConfig):
        """Setup attributes required by the framework"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.num_hidden_layers = config.num_hidden_layers
    
    def init_model(self, config: Qwen2VLInferenceConfig):
        """Initialize model components"""
        # Set padding_idx and vocab_size as attributes
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NeuronQwen2VLDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,
            pad=True,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronQwen2VLForConditionalGeneration(NeuronBaseForCausalLM):
    """
    Qwen2.5-VL model for conditional generation (causal language modeling)
    
    This is the main entry point for the model, handling:
    - Weight loading and conversion
    - Language modeling head
    - Generation interface
    """
    
    _model_cls = NeuronQwen2VLTextModel
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict to handle tied weights.
        
        Qwen2.5-VL ties the embedding and lm_head weights by default.
        """
        # If lm_head.weight is not in the state dict (because of tied weights),
        # copy it from embed_tokens
        if "lm_head.weight" not in state_dict:
            if "embed_tokens.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return Qwen2VLInferenceConfig
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format
        
        Key mappings:
        - model.embed_tokens.weight -> model.embed_tokens.weight
        - model.layers.X.self_attn.q_proj.weight -> model.layers.X.self_attn.qkv_proj.q_proj.weight
        - model.layers.X.self_attn.k_proj.weight -> model.layers.X.self_attn.qkv_proj.k_proj.weight
        - model.layers.X.self_attn.v_proj.weight -> model.layers.X.self_attn.qkv_proj.v_proj.weight
        - model.layers.X.self_attn.o_proj.weight -> model.layers.X.self_attn.o_proj.weight
        - model.layers.X.mlp.gate_proj.weight -> model.layers.X.mlp.gate_proj.weight
        - model.layers.X.mlp.up_proj.weight -> model.layers.X.mlp.up_proj.weight
        - model.layers.X.mlp.down_proj.weight -> model.layers.X.mlp.down_proj.weight
        - model.norm.weight -> model.norm.weight
        - lm_head.weight -> lm_head.weight (if not tied)
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        
        # Map weights from HF format to Neuron format
        for name, param in state_dict.items():
            # Skip visual components for now (text-only model)
            if 'visual' in name:
                continue
            
            # Handle attention QKV projections
            if '.self_attn.q_proj.' in name:
                new_name = name.replace('.self_attn.q_proj.', '.self_attn.qkv_proj.q_proj.')
                neuron_state_dict[new_name] = param.clone()
            elif '.self_attn.k_proj.' in name:
                new_name = name.replace('.self_attn.k_proj.', '.self_attn.qkv_proj.k_proj.')
                neuron_state_dict[new_name] = param.clone()
            elif '.self_attn.v_proj.' in name:
                new_name = name.replace('.self_attn.v_proj.', '.self_attn.qkv_proj.v_proj.')
                neuron_state_dict[new_name] = param.clone()
            else:
                # Copy other weights as-is
                neuron_state_dict[name] = param.clone()
        
        # Add rank utilities for tensor parallel support
        if neuron_config.vocab_parallel:
            neuron_state_dict["model.embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"model.layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        return neuron_state_dict
