# coding=utf-8
# Copyright 2025 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Mistral3 model for NXD inference

This is a port of Mistral-Small-3.1-24B-Instruct-2503 for NeuronX Distributed Inference.
The implementation focuses on the text-only component of the multimodal Mistral3 model.

Based on the existing Mistral implementation in NeuronxDistributedInference.
Mistral3's text backbone uses the same architecture as standard Mistral but with:
- Larger vocabulary (131072 tokens)
- Higher rope_theta (1000000000.0)
- More layers (40 layers, 24B parameters)
"""

import os
import json
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn

# Import base classes from NeuronxDistributedInference
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try to import MistralRMSNorm from transformers, fallback to CustomRMSNorm
try:
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm
except ImportError:
    MistralRMSNorm = None


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    if cpu_mode() and MistralRMSNorm is not None:
        return MistralRMSNorm
    return CustomRMSNorm


class Mistral3NeuronConfig(NeuronConfig):
    """
    Mistral3-specific NeuronConfig that sets the attention class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronMistral3Attention


class Mistral3InferenceConfig(InferenceConfig):
    """
    Configuration class for Mistral3 inference on NeuronX.
    
    This config handles the text portion of the Mistral3 multimodal model.
    It reads from the nested text_config in the Mistral3 config.json.
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
    
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
            "hidden_act",
            "tie_word_embeddings",
            "intermediate_size",
            "head_dim",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Mistral3InferenceConfig":
        """
        Load configuration from a pretrained Mistral3 model directory.
        
        Mistral3 uses a nested config structure with text_config and vision_config.
        We extract the text_config for text-only inference.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            Mistral3InferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Mistral3 has nested config structure - extract text_config
        if "text_config" in config_dict:
            text_config = config_dict["text_config"]
        else:
            # If no text_config, assume this is already a text config
            text_config = config_dict
        
        # Create config dict with defaults from text config
        inference_config = {
            "hidden_size": text_config.get("hidden_size", 5120),
            "num_attention_heads": text_config.get("num_attention_heads", 32),
            "num_hidden_layers": text_config.get("num_hidden_layers", 40),
            "num_key_value_heads": text_config.get("num_key_value_heads", 8),
            "vocab_size": text_config.get("vocab_size", 131072),
            "max_position_embeddings": text_config.get("max_position_embeddings", 131072),
            "rope_theta": text_config.get("rope_theta", 1000000000.0),
            "rms_norm_eps": text_config.get("rms_norm_eps", 1e-05),
            "hidden_act": text_config.get("hidden_act", "silu"),
            "intermediate_size": text_config.get("intermediate_size", 32768),
            "sliding_window": text_config.get("sliding_window", None),
            "attention_dropout": text_config.get("attention_dropout", 0.0),
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
            "use_cache": text_config.get("use_cache", True),
            # Mistral3 has explicit head_dim (not calculated from hidden_size)
            "head_dim": text_config.get("head_dim", 128),
            # Standard HuggingFace config attributes
            "output_attentions": text_config.get("output_attentions", False),
            "output_hidden_states": text_config.get("output_hidden_states", False),
            "return_dict": text_config.get("return_dict", True),
            # Token IDs - use sensible defaults if not specified
            "pad_token_id": text_config.get("pad_token_id", 0),
            "bos_token_id": text_config.get("bos_token_id", 1),
            "eos_token_id": text_config.get("eos_token_id", 2),
        }
        
        # Override with any provided kwargs
        inference_config.update(kwargs)
        
        # Create config object
        # If neuron_config is None, create a default one for inference
        if neuron_config is None:
            # During inference, neuron_config will be loaded separately by the framework
            # Create a minimal config to pass validation
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
        
        config = cls(neuron_config=neuron_config, **inference_config)
        return config


class NeuronMistral3Attention(NeuronAttentionBase):
    """
    Mistral3 attention implementation for NeuronX.
    
    Uses the same attention mechanism as standard Mistral with:
    - Grouped Query Attention (GQA) with 32 query heads and 8 KV heads
    - Rotary Position Embeddings (RoPE) with very high theta (1B)
    - Optional sliding window attention
    
    Inherits from NeuronAttentionBase which provides:
    - Flash attention computation
    - KV cache management
    - Tensor parallel support
    """
    
    def __init__(self, config: InferenceConfig):
        # Create rotary embeddings with Mistral3's high rope_theta
        # Use explicit head_dim from config instead of calculating it
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize base attention with Mistral3 parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            sliding_window=getattr(config, "sliding_window", None),
        )


class NeuronMistral3DecoderLayer(nn.Module):
    """
    Mistral3 decoder layer for NeuronX.
    
    Architecture:
    - Pre-norm architecture with RMSNorm
    - Self-attention with GQA
    - MLP with SwiGLU activation
    - Residual connections
    
    This matches the standard Mistral decoder layer architecture.
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention with GQA
        self.self_attn = NeuronMistral3Attention(config)
        
        # MLP with SwiGLU activation (same as Llama/Mistral)
        self.mlp = NeuronLlamaMLP(config)
        
        # Layer normalization (RMSNorm)
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
        Forward pass for Mistral3 decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key/value states for generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Pre-norm + Self Attention + Residual
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
        
        # Pre-norm + MLP + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MLP returns tuple, take first element
        hidden_states = residual + hidden_states
        
        # Return format expected by framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronMistral3Model(NeuronBaseModel):
    """
    Mistral3 model for NeuronX (text-only).
    
    This is the base transformer model without the LM head.
    It consists of:
    - Token embeddings
    - Stack of decoder layers
    - Final layer normalization
    - LM head (for causal language modeling)
    
    The model follows the NeuronX pattern:
    - setup_attr_for_model: Set up model attributes
    - init_model: Initialize model components
    - No custom forward method (handled by base class)
    """
    
    def setup_attr_for_model(self, config: Mistral3InferenceConfig):
        """
        Setup attributes required by the NeuronX framework.
        
        This method is called during initialization to set up
        model-specific attributes needed for compilation and inference.
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", None)
    
    def init_model(self, config: Mistral3InferenceConfig):
        """
        Initialize the model components.
        
        This method creates all the model layers:
        - Token embeddings
        - Transformer decoder layers
        - Final layer norm
        - LM head
        """
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings with vocabulary parallelism
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [NeuronMistral3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronMistral3ForCausalLM(NeuronBaseForCausalLM):
    """
    Mistral3 For Causal Language Modeling on NeuronX.
    
    This is the main class for text generation with Mistral3.
    It wraps the base model and provides:
    - Weight loading from HuggingFace checkpoints
    - State dict conversion to NeuronX format
    - Compilation and inference APIs
    
    Usage:
        # Load and compile
        config = Mistral3InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronMistral3ForCausalLM(config)
        model.compile()
        
        # Generate text
        output = model.generate(input_ids, max_length=100)
    """
    
    # Specify the model class to use
    _model_cls = NeuronMistral3Model
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load the HuggingFace model for weight extraction.
        
        Note: This is used for weight loading, not for inference.
        We can't directly use transformers.Mistral3ForConditionalGeneration
        since we only need the text model weights.
        """
        # For Mistral3, we load the full model but only use text weights
        # The base class will handle extracting the relevant weights
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except Exception as e:
            print(f"Warning: Could not load HF model: {e}")
            # Return None to allow manual weight loading
            return None
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This handles:
        1. Extracting text model weights from multimodal checkpoint
        2. Adding rank utilities for tensor parallelism
        3. Handling any weight name mapping if needed
        
        Mistral3 multimodal checkpoint structure:
        - language_model.model.layers.X.self_attn.q_proj.weight -> layers.X.self_attn.qkv_proj.q_proj.weight
        - language_model.model.embed_tokens.weight -> embed_tokens.weight
        - language_model.lm_head.weight -> lm_head.weight
        
        Args:
            state_dict: HuggingFace checkpoint state dict
            config: Model configuration
            
        Returns:
            Converted state dict in NeuronX format
        """
        neuron_config = config.neuron_config
        
        # Handle multimodal checkpoint - extract language_model weights
        converted_state_dict = {}
        for key, value in state_dict.items():
            # Strip multimodal prefixes
            new_key = key
            
            # Remove language_model prefix if present
            if key.startswith("language_model.model."):
                # language_model.model.layers.X -> layers.X
                new_key = key.replace("language_model.model.", "")
            elif key.startswith("language_model."):
                # language_model.lm_head.weight -> lm_head.weight
                new_key = key.replace("language_model.", "")
            elif key.startswith("model.text_model."):
                # Alternative multimodal format
                new_key = key.replace("model.text_model.", "")
            elif key.startswith("text_model."):
                new_key = key.replace("text_model.", "")
            elif key.startswith("model."):
                new_key = key.replace("model.", "")
            
            # Map attention weight names to qkv_proj structure expected by NeuronX
            # HF: layers.X.self_attn.q_proj.weight
            # NeuronX: layers.X.self_attn.qkv_proj.q_proj.weight
            if ".self_attn.q_proj." in new_key:
                new_key = new_key.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.q_proj.")
            elif ".self_attn.k_proj." in new_key:
                new_key = new_key.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.k_proj.")
            elif ".self_attn.v_proj." in new_key:
                new_key = new_key.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.v_proj.")
            
            converted_state_dict[new_key] = value
        
        # Add rank utilities for vocabulary parallelism
        if neuron_config.vocab_parallel:
            converted_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        # Add rank utilities for attention layers (needed for tensor parallelism)
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            converted_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utility for base model
        converted_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return converted_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embeddings and LM head.
        
        If tie_word_embeddings is True, copy embedding weights to LM head.
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class"""
        return Mistral3InferenceConfig
