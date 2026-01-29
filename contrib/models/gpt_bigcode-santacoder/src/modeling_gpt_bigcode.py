# coding=utf-8
# Copyright 2024 AWS Neuron. All Rights Reserved.
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
NeuronX implementation of GPT-BigCode (SantaCoder) model

This implementation ports GPT-BigCode from HuggingFace to NeuronX Distributed Inference.
Based on the original implementation in transformers/models/gpt_bigcode/modeling_gpt_bigcode.py

Key architectural features:
- Multi-Query Attention (MQA): 1 KV head for all query heads
- LayerNorm (not RMSNorm)
- Absolute position embeddings (not RoPE)
- GELU activation function
- Pre-normalization architecture
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase


##################################################
# Configuration
##################################################

class GPTBigCodeInferenceConfig(InferenceConfig):
    """
    Configuration class for GPT-BigCode model inference.
    
    Maps HuggingFace GPTBigCodeConfig parameters to NeuronX InferenceConfig format.
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters required by the framework"""
        self.num_cores_per_group = 1
        
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",  # Will be 1 for multi_query=True
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GPTBigCodeInferenceConfig":
        """
        Load configuration from a pretrained GPT-BigCode model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments including neuron_config
            
        Returns:
            GPTBigCodeInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read HuggingFace config.json
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace parameters to NeuronX format
        config_dict = {
            # Core architecture parameters
            "hidden_size": hf_config.get("n_embd", 2048),
            "num_hidden_layers": hf_config.get("n_layer", 24),
            "num_attention_heads": hf_config.get("n_head", 16),
            "vocab_size": hf_config.get("vocab_size", 49280),
            "max_position_embeddings": hf_config.get("n_positions", 2048),
            
            # Multi-Query Attention
            "num_key_value_heads": 1 if hf_config.get("multi_query", True) else hf_config.get("n_head", 16),
            
            # MLP intermediate size
            "intermediate_size": hf_config.get("n_inner") if hf_config.get("n_inner") is not None 
                                else 4 * hf_config.get("n_embd", 2048),
            
            # Normalization
            "layer_norm_epsilon": hf_config.get("layer_norm_epsilon", 1e-5),
            
            # Activation function
            "hidden_act": hf_config.get("activation_function", "gelu_pytorch_tanh"),
            
            # Attention configuration
            "scale_attn_weights": hf_config.get("scale_attn_weights", True),
            
            # Standard HuggingFace attributes required by the framework
            "use_cache": True,
            "tie_word_embeddings": False,
            "pad_token_id": hf_config.get("pad_token_id", 0),
            "bos_token_id": hf_config.get("bos_token_id", 49152),
            "eos_token_id": hf_config.get("eos_token_id", 49152),
            "output_attentions": False,
            "output_hidden_states": False,
        }
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # If neuron_config is None, create a minimal dummy config to pass validation
        # It will be replaced by the actual neuron_config later by the inference runner
        if neuron_config is None:
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config
    
    def load_config(self):
        """Load configuration - attributes are set via kwargs in __init__"""
        pass


##################################################
# Custom Embedding with Position
##################################################

class GPTBigCodeEmbedding(nn.Module):
    """
    Combined token and position embeddings for GPT-BigCode.
    
    GPT-BigCode uses learned absolute position embeddings that are added to token embeddings.
    This module wraps both to provide a single embedding layer.
    """
    
    def __init__(self, config: GPTBigCodeInferenceConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Position embeddings (not sharded - relatively small)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining token and position embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len], auto-generated if None
            
        Returns:
            Combined embeddings [batch_size, seq_len, hidden_size]
        """
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Generate position_ids if not provided
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine (GPT-BigCode adds them)
        embeddings = token_embeds + position_embeds
        
        return embeddings


##################################################
# MLP Module
##################################################

class NeuronGPTBigCodeMLP(nn.Module):
    """
    GPT-BigCode MLP module for NeuronX.
    
    Architecture:
    - Linear projection: hidden_size -> intermediate_size (c_fc)
    - GELU activation (gelu_pytorch_tanh variant)
    - Linear projection: intermediate_size -> hidden_size (c_proj)
    - Dropout (not used in inference)
    
    Based on GPTBigCodeMLP in transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    """
    
    def __init__(self, config: GPTBigCodeInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Input projection: hidden_size -> intermediate_size
        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Output projection: intermediate_size -> hidden_size
        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # GELU activation (GPT-BigCode uses gelu_pytorch_tanh variant)
        # In NeuronX, we use standard GELU approximation
        self.act = nn.GELU(approximate='tanh')
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for MLP.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (output_tensor, None) where None is for compatibility with framework expectations
        """
        # Apply input projection
        hidden_states = self.c_fc(hidden_states)
        
        # Apply GELU activation
        hidden_states = self.act(hidden_states)
        
        # Apply output projection
        hidden_states = self.c_proj(hidden_states)
        
        # Return tuple for framework compatibility
        return hidden_states, None


##################################################
# Attention Module
##################################################

class NeuronGPTBigCodeAttention(NeuronAttentionBase):
    """
    GPT-BigCode Multi-Query Attention for NeuronX.
    
    Key features:
    - Multi-Query Attention (MQA): 1 KV head shared across all query heads
    - No rotary position embeddings (uses absolute position embeddings in the model)
    - Attention scaling by 1/sqrt(head_dim) if scale_attn_weights=True
    - Combined QKV projection that splits to (Q, K, V)
    
    Based on GPTBigCodeAttention in transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    """
    
    def __init__(self, config: GPTBigCodeInferenceConfig):
        # GPT-BigCode uses absolute position embeddings, not rotary
        # So we don't initialize rotary_emb
        rotary_emb = None
        
        # Calculate head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Initialize base attention
        # For multi_query=True, num_key_value_heads=1 (single KV head for all queries)
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,  # 1 for MQA
            head_dim=head_dim,
            rotary_emb=rotary_emb,  # No RoPE for GPT-BigCode
            rope_theta=None,
            use_scaled_rope=False,
            qkv_bias=True,  # GPT-BigCode uses bias in QKV projections
            o_bias=True,    # GPT-BigCode uses bias in output projection
        )


##################################################
# Decoder Layer
##################################################

class NeuronGPTBigCodeBlock(nn.Module):
    """
    GPT-BigCode decoder block for NeuronX.
    
    Architecture (pre-normalization):
    1. residual = hidden_states
    2. hidden_states = LayerNorm(hidden_states)
    3. attn_output = Attention(hidden_states)
    4. hidden_states = residual + attn_output
    5. residual = hidden_states
    6. hidden_states = LayerNorm(hidden_states)
    7. mlp_output = MLP(hidden_states)
    8. hidden_states = residual + mlp_output
    
    Based on GPTBigCodeBlock in transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    """
    
    def __init__(self, config: GPTBigCodeInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Pre-attention LayerNorm
        self.ln_1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        
        # Multi-Query Attention
        self.attn = NeuronGPTBigCodeAttention(config)
        
        # Pre-MLP LayerNorm
        self.ln_2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        
        # MLP
        self.mlp = NeuronGPTBigCodeMLP(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for GPT-BigCode decoder block.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs (not used, kept for interface compatibility)
            past_key_value: Cached key-value pairs for fast generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        # Self-attention with pre-normalization
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention
        # NeuronAttentionBase returns (hidden_states, present_key_value, cos_cache, sin_cache)
        # For GPT-BigCode without RoPE, cos_cache and sin_cache will be None
        attn_output, present_key_value, cos_cache, sin_cache = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # MLP with pre-normalization
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        # MLP forward
        mlp_output, _ = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        # Return format expected by NeuronX framework
        # (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


##################################################
# Model
##################################################

class NeuronGPTBigCodeModel(NeuronBaseModel):
    """
    GPT-BigCode model for NeuronX inference.
    
    This is the main model class that follows the NeuronX framework pattern.
    It does NOT implement a forward method - the base class handles that.
    
    Based on GPTBigCodeModel in transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    """
    
    def setup_attr_for_model(self, config: GPTBigCodeInferenceConfig):
        """
        Setup attributes required by the NeuronX framework.
        
        This method is called by the base class during initialization.
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: GPTBigCodeInferenceConfig):
        """
        Initialize model components.
        
        This method is called by the base class to create the model layers.
        """
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        
        # Combined token and position embeddings
        # GPT-BigCode uses absolute position embeddings added to token embeddings
        self.embed_tokens = GPTBigCodeEmbedding(config)
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGPTBigCodeBlock(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final LayerNorm (ln_f in original implementation)
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        
        # Language modeling head (shares weights with token embeddings in original)
        # We create a separate lm_head for clarity, weights will be copied in state dict conversion
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


##################################################
# Causal LM Wrapper
##################################################

class NeuronGPTBigCodeForCausalLM(NeuronBaseForCausalLM):
    """
    GPT-BigCode causal language model wrapper for NeuronX.
    
    This class wraps the NeuronGPTBigCodeModel and provides:
    - State dict conversion from HuggingFace format to NeuronX format
    - Integration with NeuronX generation and sampling
    """
    
    _model_cls = NeuronGPTBigCodeModel
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: GPTBigCodeInferenceConfig) -> dict:
        """
        Convert HuggingFace GPT-BigCode state dict to NeuronX format.
        
        Mapping:
        - transformer.wte.weight -> embed_tokens.weight
        - transformer.wpe.weight -> position_embeddings.weight
        - transformer.h.{i}.ln_1.* -> layers.{i}.ln_1.*
        - transformer.h.{i}.attn.c_attn.* -> layers.{i}.attn.qkv_proj.*
        - transformer.h.{i}.attn.c_proj.* -> layers.{i}.attn.o_proj.*
        - transformer.h.{i}.ln_2.* -> layers.{i}.ln_2.*
        - transformer.h.{i}.mlp.c_fc.* -> layers.{i}.mlp.c_fc.*
        - transformer.h.{i}.mlp.c_proj.* -> layers.{i}.mlp.c_proj.*
        - transformer.ln_f.* -> norm.*
        - lm_head.weight (or reuse wte) -> lm_head.weight
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_state_dict = {}
        
        print("Converting HuggingFace GPT-BigCode weights to NeuronX format...")
        print(f"Original state dict keys (first 10): {list(state_dict.keys())[:10]}")
        
        # Token embeddings
        if "transformer.wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.token_embeddings.weight"] = state_dict["transformer.wte.weight"].clone()
            print("Converted: transformer.wte.weight -> embed_tokens.token_embeddings.weight")
        elif "wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.token_embeddings.weight"] = state_dict["wte.weight"].clone()
            print("Converted: wte.weight -> embed_tokens.token_embeddings.weight")
        
        # Position embeddings
        if "transformer.wpe.weight" in state_dict:
            neuron_state_dict["embed_tokens.position_embeddings.weight"] = state_dict["transformer.wpe.weight"].clone()
            print("Converted: transformer.wpe.weight -> embed_tokens.position_embeddings.weight")
        elif "wpe.weight" in state_dict:
            neuron_state_dict["embed_tokens.position_embeddings.weight"] = state_dict["wpe.weight"].clone()
            print("Converted: wpe.weight -> embed_tokens.position_embeddings.weight")
        
        # Final layer norm
        if "transformer.ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["transformer.ln_f.weight"].clone()
            neuron_state_dict["norm.bias"] = state_dict["transformer.ln_f.bias"].clone()
            print("Converted: transformer.ln_f.* -> norm.*")
        elif "ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["ln_f.weight"].clone()
            neuron_state_dict["norm.bias"] = state_dict["ln_f.bias"].clone()
            print("Converted: ln_f.* -> norm.*")
        
        # Language modeling head (may share weights with wte)
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
            print("Converted: lm_head.weight -> lm_head.weight")
        else:
            # GPT-BigCode ties weights between wte and lm_head
            neuron_state_dict["lm_head.weight"] = neuron_state_dict["embed_tokens.token_embeddings.weight"].clone()
            print("Tied weights: embed_tokens.token_embeddings.weight -> lm_head.weight")
        
        # Decoder layers
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            prefix_hf = f"transformer.h.{i}." if "transformer.h.0.ln_1.weight" in state_dict else f"h.{i}."
            prefix_neuron = f"layers.{i}."
            
            # Layer norms
            for ln_name in ["ln_1", "ln_2"]:
                for param_type in ["weight", "bias"]:
                    key_hf = f"{prefix_hf}{ln_name}.{param_type}"
                    key_neuron = f"{prefix_neuron}{ln_name}.{param_type}"
                    if key_hf in state_dict:
                        neuron_state_dict[key_neuron] = state_dict[key_hf].clone()
            
            # Attention weights
            # c_attn: combined QKV projection -> need to map to qkv_proj in NeuronAttentionBase
            attn_weight_key = f"{prefix_hf}attn.c_attn.weight"
            attn_bias_key = f"{prefix_hf}attn.c_attn.bias"
            
            if attn_weight_key in state_dict:
                # The c_attn weight contains Q, K, V concatenated
                # For multi-query: shape is (hidden_size + 2*kv_dim, hidden_size)
                # We need to split and map to qkv_proj.q_proj, k_proj, v_proj
                qkv_weight = state_dict[attn_weight_key].clone()
                qkv_bias = state_dict[attn_bias_key].clone() if attn_bias_key in state_dict else None
                
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                head_dim = hidden_size // num_heads
                kv_dim = num_kv_heads * head_dim
                
                # Split QKV
                # For multi_query, the split is: (hidden_size, kv_dim, kv_dim)
                q_weight = qkv_weight[:hidden_size, :]
                k_weight = qkv_weight[hidden_size:hidden_size+kv_dim, :]
                v_weight = qkv_weight[hidden_size+kv_dim:, :]
                
                neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.v_proj.weight"] = v_weight
                
                if qkv_bias is not None:
                    q_bias = qkv_bias[:hidden_size]
                    k_bias = qkv_bias[hidden_size:hidden_size+kv_dim]
                    v_bias = qkv_bias[hidden_size+kv_dim:]
                    
                    neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.q_proj.bias"] = q_bias
                    neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.k_proj.bias"] = k_bias
                    neuron_state_dict[f"{prefix_neuron}attn.qkv_proj.v_proj.bias"] = v_bias
            
            # Output projection
            for param_type in ["weight", "bias"]:
                key_hf = f"{prefix_hf}attn.c_proj.{param_type}"
                key_neuron = f"{prefix_neuron}attn.o_proj.{param_type}"
                if key_hf in state_dict:
                    neuron_state_dict[key_neuron] = state_dict[key_hf].clone()
            
            # MLP weights
            for mlp_layer in ["c_fc", "c_proj"]:
                for param_type in ["weight", "bias"]:
                    key_hf = f"{prefix_hf}mlp.{mlp_layer}.{param_type}"
                    key_neuron = f"{prefix_neuron}mlp.{mlp_layer}.{param_type}"
                    if key_hf in state_dict:
                        neuron_state_dict[key_neuron] = state_dict[key_hf].clone()
        
        # Add rank utilities for tensor parallelism
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        
        # Add rank info for attention layers
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        print(f"Conversion complete. NeuronX state dict has {len(neuron_state_dict)} keys")
        
        return neuron_state_dict
