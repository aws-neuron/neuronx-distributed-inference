# coding=utf-8
# Copyright 2024 Falcon authors and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Falcon model for NXD inference
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from torch.nn import LayerNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class FalconInferenceConfig(InferenceConfig):
    """
    Configuration class for Falcon model inference on Neuron hardware.
    
    Falcon-7B architecture specifics:
    - Multi-Query Attention (MQA): 71 query heads, 1 key-value head
    - Parallel attention and MLP (computed in parallel, not sequentially)
    - Standard LayerNorm (not RMSNorm)
    - Standard MLP with GELU activation (not SwiGLU)
    - RoPE position encoding (alibi=False)
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        
        # Falcon-specific defaults - set before calling super
        if not hasattr(self, 'ffn_hidden_size') or self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4
        
        # For MQA: num_kv_heads should be 1 when multi_query=True
        if not hasattr(self, 'multi_query'):
            self.multi_query = False
        
        if not hasattr(self, 'new_decoder_architecture'):
            self.new_decoder_architecture = False
            
        if self.multi_query and not self.new_decoder_architecture:
            self.num_key_value_heads = 1
        elif not hasattr(self, 'num_key_value_heads'):
            self.num_key_value_heads = self.num_attention_heads
        
        # Set default activation if not specified
        if not hasattr(self, 'activation') or self.activation is None:
            self.activation = 'gelu'
        
        # Set defaults for other attributes
        if not hasattr(self, 'alibi'):
            self.alibi = False
        
        if not hasattr(self, 'parallel_attn'):
            self.parallel_attn = True
        
        if not hasattr(self, 'bias'):
            self.bias = False
        
        # Head dimension calculation
        self.head_dim = self.hidden_size // self.num_attention_heads

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "layer_norm_epsilon",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "FalconInferenceConfig":
        """
        Load configuration from a pretrained Falcon model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            FalconInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Expand user path if needed
        model_path = os.path.expanduser(model_path)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Map Falcon configuration to our format
        mapped_config = {
            "hidden_size": config_dict.get("hidden_size", 4544),
            "num_attention_heads": config_dict.get("num_attention_heads", 71),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 32),
            "vocab_size": config_dict.get("vocab_size", 65024),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 2048),
            "layer_norm_epsilon": config_dict.get("layer_norm_epsilon", 1e-5),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "bias": config_dict.get("bias", False),
            "alibi": config_dict.get("alibi", False),
            "multi_query": config_dict.get("multi_query", True),
            "new_decoder_architecture": config_dict.get("new_decoder_architecture", False),
            "parallel_attn": config_dict.get("parallel_attn", True),
            "attention_dropout": config_dict.get("attention_dropout", 0.0),
            "hidden_dropout": config_dict.get("hidden_dropout", 0.0),
            "activation": config_dict.get("activation", "gelu"),
            "ffn_hidden_size": config_dict.get("ffn_hidden_size"),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
            # Add missing attributes for InferenceConfig compatibility
            "output_attentions": config_dict.get("output_attentions", False),
            "output_hidden_states": config_dict.get("output_hidden_states", False),
            "use_return_dict": config_dict.get("use_return_dict", True),
            "pad_token_id": config_dict.get("pad_token_id"),
            "bos_token_id": config_dict.get("bos_token_id", 11),
            "eos_token_id": config_dict.get("eos_token_id", 11),
        }
        
        # Calculate num_key_value_heads based on architecture
        if mapped_config["new_decoder_architecture"]:
            mapped_config["num_key_value_heads"] = config_dict.get("num_kv_heads", 
                                                                   mapped_config["num_attention_heads"])
        elif mapped_config["multi_query"]:
            mapped_config["num_key_value_heads"] = 1  # MQA: single KV head
        else:
            mapped_config["num_key_value_heads"] = mapped_config["num_attention_heads"]  # MHA
        
        # Override with any provided kwargs
        mapped_config.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **mapped_config)
        return config


class NeuronFalconAttention(NeuronAttentionBase):
    """
    Falcon attention implementation for NeuronX.
    
    Supports:
    - Multi-Query Attention (MQA): 71 query heads, 1 KV head
    - RoPE position encoding (when alibi=False)
    - ALiBi position encoding (when alibi=True)
    """
    
    def __init__(self, config: FalconInferenceConfig):
        # Falcon uses RoPE when alibi=False
        rotary_emb = None
        if not getattr(config, "alibi", False):
            rotary_emb = RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000.0),
            )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            qkv_bias=config.bias,
            o_bias=config.bias,
        )


class NeuronFalconMLP(nn.Module):
    """
    Falcon MLP implementation for NeuronX.
    
    Uses standard MLP structure:
    - dense_h_to_4h: hidden_size -> ffn_hidden_size
    - activation: GELU (or other specified activation)
    - dense_4h_to_h: ffn_hidden_size -> hidden_size
    
    Unlike LLaMA, this does NOT use SwiGLU.
    """
    
    def __init__(self, config: FalconInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        
        # Up projection: hidden_size -> ffn_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=config.bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function (GELU by default)
        if config.activation == "gelu":
            self.act_fn = nn.GELU()
        elif config.activation == "relu":
            self.act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation}")
        
        # Down projection: ffn_hidden_size -> hidden_size
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def forward(self, hidden_states):
        """
        Forward pass: x -> dense_h_to_4h -> activation -> dense_4h_to_h
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            tuple: (output_tensor, None) where output_tensor has shape (batch_size, seq_len, hidden_size)
                   Second element is None for compatibility with framework expectations
        """
        # Up projection
        intermediate = self.dense_h_to_4h(hidden_states)
        
        # Activation
        intermediate = self.act_fn(intermediate)
        
        # Down projection
        output = self.dense_4h_to_h(intermediate)
        
        # Return tuple for compatibility with framework
        return output, None


class NeuronFalconDecoderLayer(nn.Module):
    """
    Falcon decoder layer for NeuronX.
    
    Key architectural feature: Parallel attention and MLP computation.
    
    When parallel_attn=True (Falcon-7B default):
        output = residual + attention(ln_attn(x)) + mlp(ln_mlp(x))
    
    When parallel_attn=False:
        output = residual + attention(ln(x))
        output = output + mlp(post_ln(output))
    """
    
    def __init__(self, config: FalconInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention module
        self.self_attn = NeuronFalconAttention(config)
        
        # MLP module
        self.mlp = NeuronFalconMLP(config)
        
        # Layer normalization
        if config.parallel_attn:
            # Parallel architecture: separate layer norms for attention and MLP
            # num_ln_in_parallel_attn determines if we have 1 or 2 layer norms
            if getattr(config, 'num_ln_in_parallel_attn', None) == 2:
                self.ln_attn = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
                self.ln_mlp = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            else:
                # Single shared layer norm for both attention and MLP
                self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            # Sequential architecture: layer norms before each block
            self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for Falcon decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            
        Returns:
            tuple: (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
                   Format matches NeuronX framework expectations
        """
        residual = hidden_states
        
        # Determine which layer norms to use
        if hasattr(self, 'ln_attn') and hasattr(self, 'ln_mlp'):
            # Parallel architecture with 2 layer norms
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        elif hasattr(self, 'input_layernorm'):
            # Shared layer norm or sequential architecture
            attention_layernorm_out = self.input_layernorm(hidden_states)
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                mlp_layernorm_out = None  # Will be computed after attention
        
        # Self attention
        attention_output, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=attention_layernorm_out,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        attn_weights = None  # Not used in NeuronX inference
        
        if self.config.parallel_attn:
            # Parallel architecture: compute MLP in parallel
            mlp_output, _ = self.mlp(mlp_layernorm_out)
            
            # Combine attention and MLP outputs
            hidden_states = residual + attention_output + mlp_output
        else:
            # Sequential architecture
            residual = residual + attention_output
            mlp_layernorm_out = self.post_attention_layernorm(residual)
            mlp_output, _ = self.mlp(mlp_layernorm_out)
            hidden_states = residual + mlp_output
        
        # Return format expected by framework: (hidden_states, present_kv, cos, sin, attn_weights)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        return outputs


class NeuronFalconModel(NeuronBaseModel):
    """
    Falcon base model for NeuronX inference.
    
    This model implements the core Falcon transformer architecture:
    - Token embeddings
    - Stack of decoder layers
    - Final layer normalization
    """
    
    def setup_attr_for_model(self, config: FalconInferenceConfig):
        """Setup attributes required by the NeuronX framework"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: FalconInferenceConfig):
        """Initialize model components"""
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=not config.neuron_config.vocab_parallel,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NeuronFalconDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronFalconForCausalLM(NeuronBaseForCausalLM):
    """
    Falcon model with a causal language modeling head for NeuronX inference.
    
    This is the main model class that should be used for compilation and inference.
    """
    
    _model_cls = NeuronFalconModel
    
    @staticmethod
    def get_config_cls():
        """Return the configuration class"""
        return FalconInferenceConfig
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class"""
        return NeuronConfig
    
    def setup_attr_for_model(self, config: FalconInferenceConfig):
        """Setup attributes for the causal LM model"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
    def init_model(self, config: FalconInferenceConfig):
        """Initialize the base model"""
        # The model includes lm_head internally
        self.model = NeuronFalconModel(config)
        
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Falcon state dict to NeuronX format.
        
        HuggingFace Falcon format:
        - transformer.word_embeddings.weight -> model.embed_tokens.weight
        - transformer.h.{i}.self_attention.query_key_value.weight -> model.layers.{i}.self_attn.qkv_proj.*.weight
        - transformer.h.{i}.self_attention.dense.weight -> model.layers.{i}.self_attn.o_proj.weight
        - transformer.h.{i}.mlp.dense_h_to_4h.weight -> model.layers.{i}.mlp.dense_h_to_4h.weight
        - transformer.h.{i}.mlp.dense_4h_to_h.weight -> model.layers.{i}.mlp.dense_4h_to_h.weight
        - transformer.h.{i}.input_layernorm.weight -> model.layers.{i}.input_layernorm.weight
        - transformer.ln_f.weight -> model.norm.weight
        - lm_head.weight -> lm_head.weight
        
        Args:
            state_dict: HuggingFace format state dictionary
            config: Model configuration
            
        Returns:
            dict: NeuronX format state dictionary with rank utilities
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        
        # Token embeddings
        if "transformer.word_embeddings.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["transformer.word_embeddings.weight"].clone()
        
        # LM head
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
        
        # Final layer norm
        if "transformer.ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["transformer.ln_f.weight"].clone()
        if "transformer.ln_f.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["transformer.ln_f.bias"].clone()
        
        # Process each decoder layer
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            layer_prefix_hf = f"transformer.h.{i}"
            layer_prefix_neuron = f"layers.{i}"  # Changed from model.layers.{i} to layers.{i}
            
            # QKV projection (combined in Falcon)
            qkv_weight_key = f"{layer_prefix_hf}.self_attention.query_key_value.weight"
            if qkv_weight_key in state_dict:
                qkv_weight = state_dict[qkv_weight_key].clone()
                
                # Split QKV based on architecture
                # For Falcon-7B (multi_query=True, new_decoder_architecture=False):
                # qkv shape: (hidden_size + 2*head_dim, hidden_size)
                # = (4544 + 2*64, 4544) = (4672, 4544)
                
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                head_dim = config.head_dim
                
                if config.new_decoder_architecture:
                    # New architecture: interleaved QKV
                    # Split into Q, K, V
                    qkv_size = (num_kv_heads * 2 + num_heads) * head_dim
                    q_size = num_heads * head_dim
                    kv_size = num_kv_heads * head_dim
                    
                    # Extract Q, K, V (this is a simplified version, actual split may be more complex)
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.weight"] = qkv_weight[:q_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.weight"] = qkv_weight[q_size:q_size+kv_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.weight"] = qkv_weight[q_size+kv_size:, :].clone()
                elif config.multi_query:
                    # MQA: Q is hidden_size, K and V are each head_dim
                    q_size = hidden_size
                    kv_size = head_dim
                    
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.weight"] = qkv_weight[:q_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.weight"] = qkv_weight[q_size:q_size+kv_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.weight"] = qkv_weight[q_size+kv_size:, :].clone()
                else:
                    # MHA: Q, K, V are all hidden_size
                    q_size = hidden_size
                    k_size = hidden_size
                    v_size = hidden_size
                    
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.weight"] = qkv_weight[:q_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.weight"] = qkv_weight[q_size:q_size+k_size, :].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.weight"] = qkv_weight[q_size+k_size:, :].clone()
            
            # QKV bias (if present)
            qkv_bias_key = f"{layer_prefix_hf}.self_attention.query_key_value.bias"
            if qkv_bias_key in state_dict:
                qkv_bias = state_dict[qkv_bias_key].clone()
                
                # Split bias similar to weight
                if config.multi_query:
                    q_size = hidden_size
                    kv_size = head_dim
                    
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.bias"] = qkv_bias[:q_size].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.bias"] = qkv_bias[q_size:q_size+kv_size].clone()
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.bias"] = qkv_bias[q_size+kv_size:].clone()
            
            # Output projection
            o_proj_weight_key = f"{layer_prefix_hf}.self_attention.dense.weight"
            if o_proj_weight_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.weight"] = state_dict[o_proj_weight_key].clone()
            
            o_proj_bias_key = f"{layer_prefix_hf}.self_attention.dense.bias"
            if o_proj_bias_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.bias"] = state_dict[o_proj_bias_key].clone()
            
            # MLP weights
            mlp_up_key = f"{layer_prefix_hf}.mlp.dense_h_to_4h.weight"
            if mlp_up_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.dense_h_to_4h.weight"] = state_dict[mlp_up_key].clone()
            
            mlp_up_bias_key = f"{layer_prefix_hf}.mlp.dense_h_to_4h.bias"
            if mlp_up_bias_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.dense_h_to_4h.bias"] = state_dict[mlp_up_bias_key].clone()
            
            mlp_down_key = f"{layer_prefix_hf}.mlp.dense_4h_to_h.weight"
            if mlp_down_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.dense_4h_to_h.weight"] = state_dict[mlp_down_key].clone()
            
            mlp_down_bias_key = f"{layer_prefix_hf}.mlp.dense_4h_to_h.bias"
            if mlp_down_bias_key in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.dense_4h_to_h.bias"] = state_dict[mlp_down_bias_key].clone()
            
            # Layer norms
            # Handle different layer norm configurations
            if config.parallel_attn:
                if getattr(config, 'num_ln_in_parallel_attn', None) == 2:
                    # Separate layer norms
                    ln_attn_weight_key = f"{layer_prefix_hf}.ln_attn.weight"
                    if ln_attn_weight_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.ln_attn.weight"] = state_dict[ln_attn_weight_key].clone()
                    
                    ln_attn_bias_key = f"{layer_prefix_hf}.ln_attn.bias"
                    if ln_attn_bias_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.ln_attn.bias"] = state_dict[ln_attn_bias_key].clone()
                    
                    ln_mlp_weight_key = f"{layer_prefix_hf}.ln_mlp.weight"
                    if ln_mlp_weight_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.ln_mlp.weight"] = state_dict[ln_mlp_weight_key].clone()
                    
                    ln_mlp_bias_key = f"{layer_prefix_hf}.ln_mlp.bias"
                    if ln_mlp_bias_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.ln_mlp.bias"] = state_dict[ln_mlp_bias_key].clone()
                else:
                    # Shared layer norm
                    ln_weight_key = f"{layer_prefix_hf}.input_layernorm.weight"
                    if ln_weight_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.input_layernorm.weight"] = state_dict[ln_weight_key].clone()
                    
                    ln_bias_key = f"{layer_prefix_hf}.input_layernorm.bias"
                    if ln_bias_key in state_dict:
                        neuron_state_dict[f"{layer_prefix_neuron}.input_layernorm.bias"] = state_dict[ln_bias_key].clone()
            else:
                # Sequential architecture
                ln_weight_key = f"{layer_prefix_hf}.input_layernorm.weight"
                if ln_weight_key in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.input_layernorm.weight"] = state_dict[ln_weight_key].clone()
                
                ln_bias_key = f"{layer_prefix_hf}.input_layernorm.bias"
                if ln_bias_key in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.input_layernorm.bias"] = state_dict[ln_bias_key].clone()
                
                post_ln_weight_key = f"{layer_prefix_hf}.post_attention_layernorm.weight"
                if post_ln_weight_key in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.post_attention_layernorm.weight"] = state_dict[post_ln_weight_key].clone()
                
                post_ln_bias_key = f"{layer_prefix_hf}.post_attention_layernorm.bias"
                if post_ln_bias_key in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.post_attention_layernorm.bias"] = state_dict[post_ln_bias_key].clone()
            
            # Add rank utilities for tensor parallelism
            neuron_state_dict[f"{layer_prefix_neuron}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utilities for base model
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict


__all__ = [
    "FalconInferenceConfig",
    "NeuronFalconAttention",
    "NeuronFalconMLP",
    "NeuronFalconDecoderLayer",
    "NeuronFalconModel",
    "NeuronFalconForCausalLM",
]
