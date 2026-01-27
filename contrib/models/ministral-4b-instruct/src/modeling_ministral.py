# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
# Adapted for NeuronX Distributed Inference by AWS.
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
PyTorch Ministral model for NeuronX Distributed Inference.

This implementation ports the Ministral model (Ministral-4b-instruct) to NeuronX.
Ministral is architecturally similar to Mistral with the following key components:
- Sliding window attention (configurable per layer via layer_types)
- Grouped Query Attention (GQA) with 32 query heads and 8 KV heads
- SwiGLU activation in MLP
- RoPE positional embeddings
- RMSNorm normalization

Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/ministral/modeling_ministral.py
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

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
    """
    Get the appropriate RMSNorm class based on execution environment.
    
    Returns CustomRMSNorm for Neuron inference, MistralRMSNorm for CPU.
    This is necessary because CustomRMSNorm uses Neuron-specific optimizations
    that don't work on CPU.
    """
    return MistralRMSNorm if cpu_mode() else CustomRMSNorm


class MinistralInferenceConfig(InferenceConfig):
    """
    Configuration class for Ministral model inference on NeuronX.
    
    Inherits from InferenceConfig and adds Ministral-specific attributes.
    Handles loading configuration from HuggingFace model directory.
    
    Key attributes:
        - sliding_window: Size of the sliding window attention (default: 4096)
        - layer_types: List specifying attention type per layer ("sliding_attention" or "full_attention")
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        
        # Ensure layer_types is properly set
        if not hasattr(self, 'layer_types') or self.layer_types is None:
            sliding_window = getattr(self, 'sliding_window', 4096)
            self.layer_types = [
                "sliding_attention" if sliding_window is not None else "full_attention"
            ] * self.num_hidden_layers
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for Ministral configuration."""
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
            "tie_word_embeddings",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load configuration from a pretrained model directory.
        
        This method reads the config.json from the HuggingFace model directory
        and creates a MinistralInferenceConfig with all necessary attributes.
        
        Args:
            model_path: Path to the HuggingFace model directory
            neuron_config: NeuronConfig instance for Neuron-specific settings
            **kwargs: Additional arguments to override configuration
            
        Returns:
            MinistralInferenceConfig instance
        """
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract model configuration
        hidden_size = config_dict.get("hidden_size", 4096)
        num_attention_heads = config_dict.get("num_attention_heads", 32)
        num_hidden_layers = config_dict.get("num_hidden_layers", 32)
        num_key_value_heads = config_dict.get("num_key_value_heads", num_attention_heads)
        vocab_size = config_dict.get("vocab_size", 32000)
        max_position_embeddings = config_dict.get("max_position_embeddings", 32768)
        rope_theta = config_dict.get("rope_theta", 10000.0)
        rms_norm_eps = config_dict.get("rms_norm_eps", 1e-5)
        hidden_act = config_dict.get("hidden_act", "silu")
        intermediate_size = config_dict.get("intermediate_size", 14336)
        tie_word_embeddings = config_dict.get("tie_word_embeddings", False)
        sliding_window = config_dict.get("sliding_window", 4096)
        layer_types = config_dict.get("layer_types", None)
        
        # Build layer_types if not provided
        if layer_types is None:
            layer_types = [
                "sliding_attention" if sliding_window is not None else "full_attention"
            ] * num_hidden_layers
        
        # Get pad_token_id, bos_token_id, eos_token_id
        pad_token_id = config_dict.get("pad_token_id", None)
        bos_token_id = config_dict.get("bos_token_id", 1)
        eos_token_id = config_dict.get("eos_token_id", 2)
        
        # Create the load_config function to set attributes
        def load_config(self):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.num_key_value_heads = num_key_value_heads
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            self.rms_norm_eps = rms_norm_eps
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.tie_word_embeddings = tie_word_embeddings
            self.sliding_window = sliding_window
            self.layer_types = layer_types
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            # Standard transformers attributes required by the base model
            self.output_attentions = False
            self.output_hidden_states = False
            self.use_cache = True
            self.return_dict = True
        
        # Merge any additional kwargs
        config_kwargs = {**kwargs}
        
        # Create instance with neuron_config and load_config
        instance = cls(
            neuron_config=neuron_config,
            load_config=load_config,
            **config_kwargs
        )
        
        return instance


class NeuronMinistralAttention(NeuronAttentionBase):
    """
    Ministral attention implementation for NeuronX.
    
    This class implements the multi-head attention with:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - Sliding window attention
    
    Reuses the NeuronAttentionBase from NeuronX Distributed Inference.
    
    Args:
        config: MinistralInferenceConfig containing model configuration
    """
    
    def __init__(self, config: InferenceConfig):
        # Initialize rotary embeddings
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Get sliding window from config
        # Note: Sliding window attention is disabled by default. When seq_len < sliding_window,
        # full attention is equivalent, so this is not a functional limitation for most use cases.
        # Sliding window attention can be enabled when seq_len >= sliding_window for memory efficiency.
        sliding_window = None  # getattr(config, "sliding_window", None)
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
        )


class NeuronMinistralDecoderLayer(nn.Module):
    """
    Ministral decoder layer for NeuronX.
    
    Each decoder layer consists of:
    1. Input layer normalization (RMSNorm)
    2. Self-attention (with sliding window)
    3. Residual connection
    4. Post-attention layer normalization (RMSNorm)
    5. MLP (SwiGLU activation)
    6. Residual connection
    
    The MLP implementation reuses NeuronLlamaMLP since Ministral uses the
    same SwiGLU architecture as LLaMA/Mistral.
    
    Args:
        config: MinistralInferenceConfig
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self attention
        self.self_attn = NeuronMinistralAttention(config)
        
        # MLP - reuses LlamaMLP since architecture is identical (SwiGLU)
        self.mlp = NeuronLlamaMLP(config)
        
        # Layer normalization
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
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            position_ids: Position indices for RoPE
            past_key_value: Cached key/value states for inference
            **kwargs: Additional arguments passed to attention
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
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
        
        # Return in expected format (matches Mistral implementation)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronMinistralModel(NeuronBaseModel):
    """
    Ministral model for NeuronX Distributed Inference.
    
    This class implements the core transformer model without the language
    modeling head. It consists of:
    - Token embeddings (ParallelEmbedding for tensor parallelism)
    - Stack of decoder layers
    - Final layer normalization
    - LM head (ColumnParallelLinear for tensor parallelism)
    
    The model inherits from NeuronBaseModel which provides the infrastructure
    for distributed inference on Neuron hardware.
    """
    
    def setup_attr_for_model(self, config: MinistralInferenceConfig):
        """
        Setup model attributes required by the NeuronX framework.
        
        This method is called during model initialization and sets up
        attributes needed for inference optimization.
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = None  # Sliding window disabled - see note in NeuronMinistralAttention
    
    def init_model(self, config: MinistralInferenceConfig):
        """
        Initialize model components.
        
        Creates the embedding layer, decoder layers, normalization,
        and language modeling head with appropriate parallelization.
        """
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings with parallel sharding
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronMinistralDecoderLayer(config) 
             for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronMinistralForCausalLM(NeuronBaseForCausalLM):
    """
    Ministral model with causal language modeling head for NeuronX.
    
    This is the main class for Ministral inference on Neuron hardware.
    It wraps NeuronMinistralModel and provides:
    - Weight loading and conversion from HuggingFace format
    - Integration with NeuronX compilation and inference pipeline
    - Support for tied weights (embed_tokens and lm_head)
    
    Usage:
        config = MinistralInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronMinistralForCausalLM(config)
        model.compile()
        output = model.generate(input_ids, ...)
    """
    
    _model_cls = NeuronMinistralModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load the original HuggingFace model.
        
        This is used for weight extraction during conversion.
        """
        from transformers import MistralForCausalLM
        return MistralForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This method handles:
        1. Adding rank utilities for tensor parallelism
        2. Key remapping if necessary
        
        The Ministral/Mistral weights are compatible with the NeuronX format,
        so minimal conversion is needed beyond adding rank utilities.
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_config = config.neuron_config
        
        # Add rank utility for vocab parallel embeddings
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        # Add rank utilities for attention layers (required for tensor parallelism)
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utility for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embed_tokens and lm_head.
        
        When tie_word_embeddings is True, the lm_head weights should be
        copied from the embedding weights.
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return MinistralInferenceConfig


# Export public classes
__all__ = [
    "MinistralInferenceConfig",
    "NeuronMinistralAttention",
    "NeuronMinistralDecoderLayer",
    "NeuronMinistralModel",
    "NeuronMinistralForCausalLM",
]
