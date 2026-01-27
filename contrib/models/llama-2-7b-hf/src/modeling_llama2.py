# coding=utf-8
# Copyright 2024 AWS Neuron. All rights reserved.
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
NeuronX implementation of Llama-2-7b-hf for AWS Trainium.

This implementation leverages the existing NeuronLlama infrastructure
from NeuronxDistributedInference and provides a wrapper for Llama-2-7b-hf.

Architecture:
    - Model: Llama-2-7b-hf (32 layers, 4096 hidden size)
    - Attention: Multi-Head Attention (32 heads, no GQA)
    - MLP: SwiGLU activation (gate_proj, up_proj, down_proj)
    - Normalization: RMSNorm (eps=1e-05)
    - Position Encoding: RoPE (theta=10000.0)
    - Vocabulary: 32000 tokens
    - Max Position Embeddings: 4096

Key Differences from Llama-3:
    - Uses Multi-Head Attention (num_key_value_heads = num_attention_heads = 32)
    - No GQA (Grouped Query Attention) like Llama-3
    - rope_theta = 10000.0 (vs 500000.0 for Llama-3)
    - rms_norm_eps = 1e-05 (vs 1e-06 for Llama-3)
"""

import logging
from typing import Type

from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
    LlamaInferenceConfig,
)
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger("Neuron")


class Llama2InferenceConfig(LlamaInferenceConfig):
    """
    Configuration class for Llama-2-7b-hf inference on NeuronX.
    
    Inherits from LlamaInferenceConfig which already handles all required
    Llama architecture parameters. This class is identical to LlamaInferenceConfig
    but provides a distinct class for Llama-2 models.
    
    The parent class automatically loads configuration from HuggingFace's config.json:
        - hidden_size: 4096
        - num_attention_heads: 32
        - num_hidden_layers: 32
        - num_key_value_heads: 32 (MHA, not GQA)
        - vocab_size: 32000
        - intermediate_size: 11008
        - max_position_embeddings: 4096
        - rms_norm_eps: 1e-05
        - rope_theta: 10000.0
        - hidden_act: "silu"
    
    Usage:
        ```python
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        
        # Create config from model path
        config = Llama2InferenceConfig.from_pretrained(
            model_path,
            neuron_config=neuron_config,
        )
        ```
    """
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load configuration from a pretrained model directory.
        
        This method loads the HuggingFace config.json and initializes
        the Llama2InferenceConfig with proper NeuronConfig settings.
        
        Args:
            model_path (str): Path to the model directory containing config.json
            neuron_config (NeuronConfig, optional): Neuron-specific configuration.
                If None, will create a minimal default config (used during inference loading).
            **kwargs: Additional configuration overrides
        
        Returns:
            Llama2InferenceConfig: Initialized configuration object
        
        Example:
            ```python
            # During compilation
            neuron_config = NeuronConfig(tp_degree=2, batch_size=1, seq_len=128)
            config = Llama2InferenceConfig.from_pretrained(
                "/path/to/model",
                neuron_config=neuron_config
            )
            
            # During inference loading (neuron_config loaded separately)
            config = Llama2InferenceConfig.from_pretrained("/path/to/model")
            ```
        """
        # If neuron_config is not provided, create a minimal default
        # This happens during inference when neuron_config is loaded separately
        if neuron_config is None:
            # Create minimal config that will be overridden by loaded neuron_config
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
            logger.debug("Created default neuron_config for config loading")
        
        # Create configuration using load_pretrained_config helper
        # This loads the HuggingFace config.json and maps parameters correctly
        config = cls(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(model_path),
            **kwargs
        )
        return config


class NeuronLlama2ForCausalLM(NeuronLlamaForCausalLM):
    """
    NeuronX implementation of Llama-2-7b-hf for causal language modeling.
    
    This class wraps the existing NeuronLlamaForCausalLM implementation,
    which fully supports the Llama-2 architecture. The only customization
    is using Llama2InferenceConfig for configuration.
    
    The model architecture is identical to the base Llama implementation:
        - Input: token IDs
        - Token Embedding layer (vocab_size=32000)
        - 32 decoder layers, each with:
            * Multi-Head Attention (32 heads, head_dim=128)
            * SwiGLU MLP (intermediate_size=11008)
            * RMSNorm (pre-attention and pre-MLP)
        - Final RMSNorm
        - LM head (vocabulary logits)
    
    Key Features:
        - Tensor Parallelism support (tp_degree)
        - Sequence Parallelism support
        - Flash Attention for efficient computation
        - KV caching for autoregressive generation
        - RoPE position embeddings (theta=10000.0)
        - SwiGLU activation in MLP layers
        - RMSNorm layer normalization
    
    Usage:
        ```python
        from neuronx_distributed_inference.models.config import NeuronConfig
        
        # Create neuron config
        neuron_config = NeuronConfig(
            tp_degree=2,
            batch_size=1,
            seq_len=128,
            torch_dtype=torch.float32,
        )
        
        # Load config and create model
        config = Llama2InferenceConfig.from_pretrained(
            model_path,
            neuron_config=neuron_config,
        )
        model = NeuronLlama2ForCausalLM(model_path, config)
        ```
    """
    
    # Use the same model class as base Llama
    _model_cls = NeuronLlamaModel
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for Llama-2"""
        return Llama2InferenceConfig
    
    # Inherit all other methods from NeuronLlamaForCausalLM:
    # - load_hf_model: Loads HuggingFace LlamaForCausalLM
    # - convert_hf_to_neuron_state_dict: Converts weights to Neuron format
    # - update_state_dict_for_tied_weights: Handles weight tying
    # These work identically for Llama-2


# Export classes
__all__ = [
    "Llama2InferenceConfig",
    "NeuronLlama2ForCausalLM",
]
