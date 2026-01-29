# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Configuration for Ovis2.5-9B model for NeuronX Distributed Inference.

This configuration wraps the Qwen3 LLM component of the Ovis2.5 multimodal model.
For initial implementation, we only port the text-only LLM component.
Vision components can be added later if needed.
"""

import json
import os
from typing import List, Type

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import (
    Qwen3InferenceConfig,
    Qwen3NeuronConfig,
)


class Ovis2_5_NeuronConfig(Qwen3NeuronConfig):
    """
    NeuronConfig for Ovis2.5 model.
    Inherits from Qwen3NeuronConfig since the LLM backbone is Qwen3.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Ovis2_5_InferenceConfig(Qwen3InferenceConfig):
    """
    InferenceConfig for Ovis2.5 model.
    
    This config extracts the LLM configuration from the Ovis2.5 config.json
    and wraps it as a Qwen3InferenceConfig.
    
    The Ovis2.5 model structure:
    - llm: Qwen3-8B (36 layers, 4096 hidden, 32 heads, 8 KV heads - GQA)
    - visual_tokenizer: Siglip2-NavIT (not ported in initial version)
    - vte: Visual embedding table (not ported in initial version)
    
    For text-only inference, we only need the LLM component.
    """

    def __init__(self, **kwargs):
        # Initialize with Qwen3 config
        super().__init__(**kwargs)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Ovis2_5_NeuronConfig]:
        return Ovis2_5_NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Ovis2_5_InferenceConfig":
        """
        Load configuration from Ovis2.5 model directory.
        
        Extracts the llm_config from the Ovis2.5 config.json and creates
        a Qwen3-compatible configuration.
        
        Args:
            model_path: Path to the Ovis2.5 model directory
            **kwargs: Additional arguments including neuron_config
            
        Returns:
            Ovis2_5_InferenceConfig: Configuration object for NeuronX inference
        """
        import json
        import torch
        
        # Extract neuron_config from kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        
        # If loading from compiled model, try to load neuron_config.json
        if neuron_config is None:
            neuron_config_path = os.path.join(model_path, "neuron_config.json")
            if os.path.exists(neuron_config_path):
                with open(neuron_config_path, "r") as f:
                    saved_config = json.load(f)
                    if "neuron_config" in saved_config:
                        # Load NeuronConfig from saved dict
                        neuron_config = NeuronConfig(**saved_config["neuron_config"])
                        print(f"✓ Loaded neuron_config from {neuron_config_path}")
        
        # Create a default neuron_config if still None (for basic loading)
        if neuron_config is None:
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=512,
                torch_dtype=torch.bfloat16,
            )
            print(f"⚠ Using default neuron_config (no neuron_config provided)")

        # Load Ovis2.5 config.json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            ovis_config = json.load(f)

        # Extract LLM config from Ovis2.5 config
        # The LLM config is nested under "llm_config" key
        if "llm_config" not in ovis_config:
            raise ValueError(
                f"Expected 'llm_config' key in Ovis2.5 config.json, got keys: {list(ovis_config.keys())}"
            )

        llm_config = ovis_config["llm_config"]

        # Create Qwen3-compatible config dict
        config_dict = {
            # Core architecture parameters
            "hidden_size": llm_config.get("hidden_size", 4096),
            "num_attention_heads": llm_config.get("num_attention_heads", 32),
            "num_hidden_layers": llm_config.get("num_hidden_layers", 36),
            "num_key_value_heads": llm_config.get("num_key_value_heads", 8),
            "head_dim": llm_config.get("head_dim", 128),
            # Vocabulary and embedding
            "vocab_size": llm_config.get("vocab_size", 151936),
            "max_position_embeddings": llm_config.get("max_position_embeddings", 40960),
            # Normalization and activation
            "rms_norm_eps": llm_config.get("rms_norm_eps", 1e-6),
            "hidden_act": llm_config.get("hidden_act", "silu"),
            # MLP
            "intermediate_size": llm_config.get("intermediate_size", 12288),
            # RoPE
            "rope_theta": llm_config.get("rope_theta", 1000000),
            # Token IDs
            "bos_token_id": llm_config.get("bos_token_id", 151643),
            "eos_token_id": llm_config.get("eos_token_id", 151645),
            "pad_token_id": llm_config.get("eos_token_id", 151645),  # Use EOS as pad
            # Attention configuration
            "attention_bias": llm_config.get("attention_bias", False),
            "attention_dropout": llm_config.get("attention_dropout", 0.0),
            # Cache configuration
            "use_cache": llm_config.get("use_cache", True),
        }

        # Override with any provided kwargs
        config_dict.update(kwargs)

        # Create and return config
        config = cls(neuron_config=neuron_config, **config_dict)

        return config

    def add_derived_config(self):
        """Add derived configuration parameters"""
        # Call parent implementation
        super().add_derived_config()

        # Add output control attributes if not already present
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

        # Add any Ovis2.5-specific derived config here
        # For now, we just use the Qwen3 defaults

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        # Use Qwen3 required attributes
        return super().get_required_attributes()


__all__ = [
    "Ovis2_5_InferenceConfig",
    "Ovis2_5_NeuronConfig",
]
