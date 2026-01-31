#!/usr/bin/env python3
# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Phi3 model for NeuronxDistributed inference

This implementation is based on the original Phi3 model from:


Key architectural features from the original:
- Combined gate_up_proj in MLP (gate and up projections combined)
- Combined qkv_proj in attention (query, key, value projections combined)
- SiLU activation function
- RoPE (Rotary Position Embeddings) with theta=10000.0
- Sliding window attention (window size = 2047)
- Multi-head attention (not grouped-query attention)
- Residual dropout in decoder layers
"""

import os
import json
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm implementation
    - CustomRMSNorm for NeuronX inference
    - Standard RMSNorm for CPU mode
    """
    if cpu_mode():
        # Use standard RMSNorm for CPU
        class StandardRMSNorm(nn.Module):
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
        
        return StandardRMSNorm
    else:
        return CustomRMSNorm


class Phi3InferenceConfig(InferenceConfig):
    """
    Configuration class for Phi3 model inference
    
    Based on the configuration from:
    
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        
        # Add missing attributes expected by the framework
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        if not hasattr(self, 'tie_word_embeddings'):
            self.tie_word_embeddings = False
        if not hasattr(self, 'hidden_act'):
            self.hidden_act = 'silu'
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads", 
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "intermediate_size",
            "hidden_act",
            "tie_word_embeddings",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments to override configuration
            
        Returns:
            Phi3InferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Expand user home directory if needed
        model_path = os.path.expanduser(model_path)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create configuration with values from config file
        # Based on 
        final_config = {
            "hidden_size": config_dict.get("hidden_size", 3072),
            "num_attention_heads": config_dict.get("num_attention_heads", 32),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 32),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 32),
            "vocab_size": config_dict.get("vocab_size", 32064),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 4096),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-05),
            "intermediate_size": config_dict.get("intermediate_size", 8192),
            "hidden_act": config_dict.get("hidden_act", "silu"),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
            "sliding_window": config_dict.get("sliding_window", 2047),
            "attention_dropout": config_dict.get("attention_dropout", 0.0),
            "resid_pdrop": config_dict.get("resid_pdrop", 0.0),
            "embd_pdrop": config_dict.get("embd_pdrop", 0.0),
            "pad_token_id": config_dict.get("pad_token_id", 32000),
            "bos_token_id": config_dict.get("bos_token_id", 1),
            "eos_token_id": config_dict.get("eos_token_id", 32000),
        }
        
        # Override with any additional kwargs
        final_config.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **final_config)
        return config


class NeuronPhi3MLP(nn.Module):
    """
    Phi3 MLP implementation for NeuronxDistributed
    
    Based on the original Phi3MLP from:
    
    
    Original implementation:
    - gate_up_proj: Linear(hidden_size, 2 * intermediate_size, bias=False)
    - down_proj: Linear(intermediate_size, hidden_size, bias=False)
    - activation_fn: SiLU activation
    - Forward: up_states = gate_up_proj(x); gate, up = chunk(2); up = up * silu(gate); return down_proj(up)
    """
    
    def __init__(self, config: Phi3InferenceConfig):
        super().__init__()
        self.config = config
        
        # Combined gate and up projection - matches original gate_up_proj
        # Original: self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Down projection - matches original down_proj
        # Original: self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function - matches original activation_fn
        # Original: self.activation_fn = ACT2FN[config.hidden_act] (SiLU)
        self.activation_fn = nn.SiLU()
    
    def forward(self, hidden_states):
        """
        Forward pass matching original Phi3MLP implementation
        
        Original forward logic:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)
        """
        # Apply combined gate and up projection
        up_states = self.gate_up_proj(hidden_states)
        
        # Split into gate and up components
        gate, up_states = up_states.chunk(2, dim=-1)
        
        # Apply gated activation: up_states * silu(gate)
        up_states = up_states * self.activation_fn(gate)
        
        # Apply down projection
        output = self.down_proj(up_states)
        
        return output


class NeuronPhi3Attention(NeuronAttentionBase):
    """
    Phi3 Attention implementation for NeuronxDistributed
    
    Based on the original Phi3Attention from:
    
    
    Original implementation:
    - Uses combined qkv_proj: Linear(hidden_size, op_size, bias=False)
    - op_size = num_attention_heads * head_dim + 2 * (num_key_value_heads * head_dim)
    - o_proj: Linear(num_attention_heads * head_dim, hidden_size, bias=False)
    - Applies RoPE to query and key states
    - Uses sliding window attention (sliding_window=2047)
    """
    
    def __init__(self, config: Phi3InferenceConfig):
        # Create rotary embedding - matches original RoPE setup
        rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize base attention with Phi3 parameters
        # Disable sliding window if seq_len is smaller than sliding_window
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window and hasattr(config, 'neuron_config') and config.neuron_config.seq_len < sliding_window:
            sliding_window = None  # Disable sliding window for short sequences
            
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )


class NeuronPhi3DecoderLayer(nn.Module):
    """
    Phi3 Decoder Layer implementation for NeuronxDistributed
    
    Based on the original Phi3DecoderLayer from:
    
    
    Original implementation extends MistralDecoderLayer with:
    - self_attn: Phi3Attention
    - mlp: Phi3MLP
    - resid_attn_dropout: Dropout for attention residual
    - resid_mlp_dropout: Dropout for MLP residual
    - input_layernorm and post_attention_layernorm: RMSNorm layers
    """
    
    def __init__(self, config: Phi3InferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention layer
        self.self_attn = NeuronPhi3Attention(config)
        
        # MLP layer
        self.mlp = NeuronPhi3MLP(config)
        
        # Normalization layers
        RMSNormCls = get_rmsnorm_cls()
        self.input_layernorm = RMSNormCls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormCls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # Residual dropouts - matches original implementation
        self.resid_attn_dropout = nn.Dropout(getattr(config, 'resid_pdrop', 0.0))
        self.resid_mlp_dropout = nn.Dropout(getattr(config, 'resid_pdrop', 0.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass matching original Phi3DecoderLayer implementation
        
        Original forward logic:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(...)
        hidden_states = residual + self.resid_attn_dropout(hidden_states)  # main diff with Llama
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)  # main diff with Llama
        """
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Residual connection with dropout (main difference from Llama)
        hidden_states = residual + self.resid_attn_dropout(hidden_states)
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection with dropout (main difference from Llama)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        
        # Return format matching framework expectations
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)  # None for attention weights
        return outputs


class NeuronPhi3Model(NeuronBaseModel):
    """
    Phi3 Model implementation for NeuronxDistributed
    
    Based on the original Phi3Model structure following the pattern from other models
    in the NeuronxDistributed framework.
    """
    
    def setup_attr_for_model(self, config: Phi3InferenceConfig):
        """Setup attributes required by the framework"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: Phi3InferenceConfig):
        """Initialize model components"""
        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NeuronPhi3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        RMSNormCls = get_rmsnorm_cls()
        self.norm = RMSNormCls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronPhi3ForCausalLM(NeuronBaseForCausalLM):
    """
    Phi3 Causal Language Model for NeuronxDistributed inference
    
    This is the main interface class that follows the NeuronxDistributed framework pattern.
    """
    
    _model_cls = NeuronPhi3Model
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return Phi3InferenceConfig
    
    def checkpoint_loader_fn(self, mmap: bool = False):
        """
        Override checkpoint loader to redirect to original checkpoint directory
        
        This follows the pattern from the Llama3 implementation where the compiled
        model directory doesn't contain the original weights, so we need to redirect
        to the original checkpoint directory for weight loading.
        """
        # Check if this is a compiled model directory
        compiled_model_file = os.path.join(self.model_path, "model.pt")
        if os.path.exists(compiled_model_file):
            # Load weights from original checkpoint directory
            original_checkpoint_path = "./Phi-3-mini-4k-instruct"
            if os.path.exists(original_checkpoint_path):
                # Temporarily redirect to original checkpoint
                original_model_path = self.model_path
                self.model_path = original_checkpoint_path
                try:
                    result = super().checkpoint_loader_fn(mmap=mmap)
                finally:
                    self.model_path = original_model_path
                return result
        
        # Fall back to default behavior
        return super().checkpoint_loader_fn(mmap=mmap)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format
        
        This method converts the original Phi3 parameter names to the format expected
        by the NeuronX framework, including splitting combined projections.
        
        Args:
            state_dict: HuggingFace format state dictionary
            config: Model configuration
            
        Returns:
            dict: NeuronX format state dictionary
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        
        # Convert parameter names from HuggingFace format to NeuronX format
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            if key.startswith('model.'):
                key = key[6:]  # Remove 'model.' prefix
            
            # Handle embeddings
            if key == 'embed_tokens.weight':
                neuron_state_dict['embed_tokens.weight'] = value.clone()
            
            # Handle final norm
            elif key == 'norm.weight':
                neuron_state_dict['norm.weight'] = value.clone()
            
            # Handle language model head
            elif key == 'lm_head.weight':
                neuron_state_dict['lm_head.weight'] = value.clone()
            
            # Handle layer parameters
            elif key.startswith('layers.'):
                parts = key.split('.')
                layer_idx = parts[1]
                
                if 'self_attn.qkv_proj.weight' in key:
                    # Split combined QKV projection into separate Q, K, V projections
                    qkv_weight = value
                    hidden_size = config.hidden_size
                    num_heads = config.num_attention_heads
                    num_kv_heads = config.num_key_value_heads
                    head_dim = hidden_size // num_heads
                    
                    # Calculate sizes
                    q_size = num_heads * head_dim
                    k_size = num_kv_heads * head_dim
                    v_size = num_kv_heads * head_dim
                    
                    # Split the combined weight
                    q_weight = qkv_weight[:q_size, :]
                    k_weight = qkv_weight[q_size:q_size + k_size, :]
                    v_weight = qkv_weight[q_size + k_size:q_size + k_size + v_size, :]
                    
                    # Store split weights
                    neuron_state_dict[f'layers.{layer_idx}.self_attn.qkv_proj.q_proj.weight'] = q_weight.clone()
                    neuron_state_dict[f'layers.{layer_idx}.self_attn.qkv_proj.k_proj.weight'] = k_weight.clone()
                    neuron_state_dict[f'layers.{layer_idx}.self_attn.qkv_proj.v_proj.weight'] = v_weight.clone()
                
                elif 'self_attn.o_proj.weight' in key:
                    neuron_state_dict[f'layers.{layer_idx}.self_attn.o_proj.weight'] = value.clone()
                
                elif 'mlp.gate_up_proj.weight' in key:
                    # Keep combined gate_up projection as-is (matches our MLP implementation)
                    neuron_state_dict[f'layers.{layer_idx}.mlp.gate_up_proj.weight'] = value.clone()
                
                elif 'mlp.down_proj.weight' in key:
                    neuron_state_dict[f'layers.{layer_idx}.mlp.down_proj.weight'] = value.clone()
                
                elif 'input_layernorm.weight' in key:
                    neuron_state_dict[f'layers.{layer_idx}.input_layernorm.weight'] = value.clone()
                
                elif 'post_attention_layernorm.weight' in key:
                    neuron_state_dict[f'layers.{layer_idx}.post_attention_layernorm.weight'] = value.clone()
        
        # Add rank utilities for tensor parallel support
        if hasattr(neuron_config, 'vocab_parallel') and neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        # Add rank information for attention layers
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict
    
    @classmethod
    def from_config(cls, config):
        """
        Create a model from a configuration
        
        Args:
            config: Model configuration
            
        Returns:
            NeuronPhi3ForCausalLM: Model instance
        """
        return cls(config=config)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load a compiled model from a directory
        
        Args:
            model_path: Path to compiled model directory
            **kwargs: Additional arguments
            
        Returns:
            NeuronPhi3ForCausalLM: Loaded model instance
        """
        return cls(model_path=model_path, **kwargs)