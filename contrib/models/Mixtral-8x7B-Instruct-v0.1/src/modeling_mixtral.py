# coding=utf-8
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
""" PyTorch Mixtral-8x7B model for NXD inference - Custom Port"""
import json
import os
from typing import List

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import (
    NeuronMixtralForCausalLM as BaseNeuronMixtralForCausalLM,
)
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import (
    convert_mixtral_to_neuron_state_dict,
)


class MixtralInferenceConfig(InferenceConfig):
    """
    Configuration class for Mixtral-8x7B model inference on NeuronX.
    
    This extends InferenceConfig with Mixtral-specific parameters and adds
    a from_pretrained class method for loading configurations.
    
    Based on: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/mixtral/configuration_mixtral.py
    Reference: NeuronxDistributedInference/src/neuronx_distributed_inference/models/mixtral/modeling_mixtral.py
    """
    
    def get_required_attributes(self) -> List[str]:
        """
        List of required attributes for Mixtral configuration.
        These attributes must be present for the model to function correctly.
        """
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "num_local_experts",
            "num_experts_per_tok",
            "rms_norm_eps",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        """Return the MoE-specific NeuronConfig class"""
        return MoENeuronConfig
    
    def validate_config(self):
        """
        Validates that the config has all required attributes.
        
        Overridden to handle the case where neuron_config is None during
        inference loading (neuron_config is loaded separately).
        """
        # Call parent validation for required attributes
        missing_attributes = [x for x in self.get_required_attributes() if not hasattr(self, x)]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"
        
        # Only validate neuron_config-dependent settings if neuron_config exists
        if self.neuron_config is not None:
            # Call parent's remaining validations that require neuron_config
            # We skip the windowed_context_encoding validation if neuron_config is None
            if hasattr(self.neuron_config, 'windowed_context_encoding_size'):
                wce_size = self.neuron_config.windowed_context_encoding_size
                if wce_size is not None and hasattr(self, "sliding_window") and self.sliding_window is not None:
                    assert wce_size == self.sliding_window, \
                        f"Windowed context encoding size must equal sliding window size. " \
                        f"Got windowed_context_encoding_size = {wce_size}, sliding_window = {self.sliding_window}"
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained Mixtral model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration values
            
        Returns:
            MixtralInferenceConfig: Configuration object
            
        Example:
            config = MixtralInferenceConfig.from_pretrained(
                "/shared/dhwanw/models/Mixtral-8x7B-Instruct-v0.1",
                neuron_config=neuron_config
            )
        """
        # Extract neuron_config from kwargs if provided
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Try to read from a compiled model's neuron_config.json first
        neuron_config_path = os.path.join(model_path, "neuron_config.json")
        if os.path.exists(neuron_config_path):
            # Loading from compiled model
            print(f"📦 Loading from compiled model: {model_path}")
            with open(neuron_config_path, "r") as f:
                saved_config = json.load(f)
            
            # The saved config already has both model config and neuron_config
            # Extract neuron_config if present
            if "neuron_config" in saved_config and neuron_config is None:
                # Neuron config will be loaded separately by the inference framework
                neuron_config = None
            
            # Create config with saved parameters
            config_dict = {k: v for k, v in saved_config.items() if k != "neuron_config"}
            config_dict.update(kwargs)
            
            print(f"✅ Loaded compiled Mixtral configuration")
            return cls(neuron_config=neuron_config, **config_dict)
        
        # Read HuggingFace config.json for original model
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace config to our config format
        config_dict = {
            # Core model dimensions
            "hidden_size": hf_config.get("hidden_size", 4096),
            "num_attention_heads": hf_config.get("num_attention_heads", 32),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 32),
            "num_key_value_heads": hf_config.get("num_key_value_heads", 8),
            "intermediate_size": hf_config.get("intermediate_size", 14336),
            
            # Vocabulary and position
            "vocab_size": hf_config.get("vocab_size", 32000),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 32768),
            
            # Special tokens
            "pad_token_id": hf_config.get("pad_token_id"),
            "bos_token_id": hf_config.get("bos_token_id", 1),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            
            # Normalization and activation
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            
            # Position embeddings
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            
            # MoE specific parameters
            "num_local_experts": hf_config.get("num_local_experts", 8),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 2),
            
            # Sliding window attention (if present)
            "sliding_window": hf_config.get("sliding_window", None),
            
            # Additional parameters
            "attention_dropout": hf_config.get("attention_dropout", 0.0),
            "initializer_range": hf_config.get("initializer_range", 0.02),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
            
            # Inference-specific parameters
            "output_attentions": hf_config.get("output_attentions", False),
            "output_hidden_states": hf_config.get("output_hidden_states", False),
            "use_cache": hf_config.get("use_cache", True),
        }
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        print(f"✅ Loaded Mixtral configuration from {model_path}")
        print(f"   - Hidden size: {config_dict['hidden_size']}")
        print(f"   - Num layers: {config_dict['num_hidden_layers']}")
        print(f"   - Num experts: {config_dict['num_local_experts']}")
        print(f"   - Experts per token: {config_dict['num_experts_per_tok']}")
        print(f"   - Vocab size: {config_dict['vocab_size']}")
        
        # Create and return config object
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronMixtralForCausalLM(BaseNeuronMixtralForCausalLM):
    """
    Mixtral-8x7B Causal Language Model for NeuronX inference.
    
    This class extends the base NeuronMixtralForCausalLM with our custom config
    that includes from_pretrained support.
    
    Architecture:
    - 32 decoder layers
    - Each layer has:
      * Grouped Query Attention (32 Q heads, 8 KV heads)
      * Mixture of 8 Experts with Top-2 routing
      * RMSNorm for normalization
      * Rotary Position Embeddings (RoPE)
    
    Based on: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/mixtral/modeling_mixtral.py
    Reference: NeuronxDistributedInference/src/neuronx_distributed_inference/models/mixtral/modeling_mixtral.py
    """
    
    @classmethod
    def get_config_cls(cls):
        """Return our custom config class with from_pretrained support"""
        return MixtralInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This method handles the conversion of MoE weights from HuggingFace's format
        to the format expected by NeuronX's MoE implementation.
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            dict: Converted state dictionary in NeuronX format
        """
        return convert_mixtral_to_neuron_state_dict(state_dict, config)
