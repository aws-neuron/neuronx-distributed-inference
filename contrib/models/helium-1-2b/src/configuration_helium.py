# coding=utf-8
# Copyright 2024 The Kyutai and HuggingFace Inc. teams. All rights reserved.
# Ported to NeuronX Distributed Inference
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

"""Helium model configuration for NeuronX Distributed Inference"""

import json
import os
from typing import List, Type

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


class HeliumInferenceConfig(InferenceConfig):
    """
    Configuration class for Helium model inference on NeuronX.
    
    This configuration is based on the Helium architecture which is similar to LLaMA
    with GQA attention, SwiGLU MLP, and RoPE position embeddings.
    
    Key architectural features:
    - Grouped Query Attention (GQA) with configurable query/KV head ratios
    - SwiGLU activation in MLP layers
    - RMSNorm for layer normalization
    - RoPE (Rotary Position Embeddings)
    
    Args:
        vocab_size (int): Size of vocabulary (default: 64000 for helium-1-2b)
        hidden_size (int): Hidden dimension (default: 2048)
        intermediate_size (int): MLP intermediate dimension (default: 8192)
        num_hidden_layers (int): Number of transformer layers (default: 28)
        num_attention_heads (int): Number of query attention heads (default: 16)
        num_key_value_heads (int): Number of key-value heads for GQA (default: 8)
        head_dim (int): Dimension of each attention head (default: 128)
        max_position_embeddings (int): Maximum sequence length (default: 4096)
        rms_norm_eps (float): Epsilon for RMSNorm (default: 1e-8)
        rope_theta (float): Base frequency for RoPE (default: 20000.0)
        attention_bias (bool): Whether to use bias in attention layers (default: False)
        mlp_bias (bool): Whether to use bias in MLP layers (default: False)
        hidden_act (str): Activation function (default: "silu")
        pad_token_id (int): Padding token id (default: 3)
        bos_token_id (int): Beginning of sequence token id (default: 0)
        eos_token_id (int): End of sequence token id (default: 1)
        tie_word_embeddings (bool): Whether to tie embeddings (default: False)
    """
    
    model_type = "helium"
    
    def __init__(
        self,
        vocab_size: int = 64000,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-8,
        rope_theta: float = 20000.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        hidden_act: str = "silu",
        pad_token_id: int = 3,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        neuron_config: NeuronConfig = None,
        **kwargs,
    ):
        """Initialize Helium configuration"""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.hidden_act = hidden_act
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        
        # Add missing attributes expected by the framework
        self.output_attentions = kwargs.get("output_attentions", False)
        self.output_hidden_states = kwargs.get("output_hidden_states", False)
        self.use_cache = kwargs.get("use_cache", True)
        
        # Initialize the base class with neuron_config
        super().__init__(neuron_config=neuron_config, **kwargs)
        
    def add_derived_config(self):
        """Add derived configuration parameters for NeuronX"""
        # Number of cores per group for attention computation
        self.num_cores_per_group = 1
        
    def get_required_attributes(self) -> List[str]:
        """Return list of required attributes for model initialization"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "HeliumInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        This method reads the config.json file from the model directory and
        creates a HeliumInferenceConfig object.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional configuration parameters to override
            
        Returns:
            HeliumInferenceConfig: The loaded configuration
            
        Raises:
            FileNotFoundError: If config.json is not found in model_path
        """
        # Extract neuron_config from kwargs if present
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Expand user path
        model_path = os.path.expanduser(model_path)
        
        # Load config.json
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                f"Please ensure the model directory contains config.json"
            )
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Map HuggingFace config keys to our config keys
        # Most keys are already compatible, but we need to handle special cases
        config_params = {
            "vocab_size": config_dict.get("vocab_size", 64000),
            "hidden_size": config_dict.get("hidden_size", 2048),
            "intermediate_size": config_dict.get("intermediate_size", 8192),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 28),
            "num_attention_heads": config_dict.get("num_attention_heads", 16),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 8),
            "head_dim": config_dict.get("head_dim", 128),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 4096),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-8),
            "rope_theta": config_dict.get("rope_theta", 20000.0),
            "attention_bias": config_dict.get("attention_bias", False),
            "mlp_bias": config_dict.get("mlp_bias", False),
            "hidden_act": config_dict.get("hidden_act", "silu"),
            "pad_token_id": config_dict.get("pad_token_id", 3),
            "bos_token_id": config_dict.get("bos_token_id", 0),
            "eos_token_id": config_dict.get("eos_token_id", 1),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
        }
        
        # Override with any additional kwargs
        config_params.update(kwargs)
        
        # If neuron_config is None and we're loading from a compiled model,
        # we need to create a default one for inference
        if neuron_config is None:
            # Try to load from compiled artifacts if available
            import glob
            compiled_config_path = os.path.join(model_path, "neuron_config.json")
            if os.path.exists(compiled_config_path):
                with open(compiled_config_path, "r") as f:
                    neuron_config_dict = json.load(f)
                    neuron_config = NeuronConfig(**neuron_config_dict)
            else:
                # Create a minimal default config for loading
                print("Warning: Creating default NeuronConfig for inference")
                neuron_config = NeuronConfig(
                    tp_degree=1,
                    batch_size=1,
                    seq_len=128,
                )
        
        # Create and return the config
        config = cls(neuron_config=neuron_config, **config_params)
        
        print(f"Loaded Helium config from {model_path}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Num KV heads: {config.num_key_value_heads} (GQA ratio: {config.num_attention_heads // config.num_key_value_heads}:1)")
        print(f"  - Vocab size: {config.vocab_size}")
        print(f"  - RoPE theta: {config.rope_theta}")
        
        return config
