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
Qwen2.5-VL configuration for NeuronX Distributed Inference
"""

import json
import os
from typing import List, Type

from neuronx_distributed_inference.models.config import InferenceConfig, MultimodalVisionNeuronConfig


class Qwen2VLVisionConfig:
    """
    Configuration for Qwen2-VL vision encoder
    """
    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        intermediate_size=3420,
        num_heads=16,
        in_chans=3,
        out_hidden_size=2048,
        patch_size=14,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
        window_size=112,
        fullatt_block_indexes=None,
        tokens_per_second=2,
        hidden_act="silu",
        **kwargs
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size  # Alias for compatibility
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.in_channels = in_chans  # Alias
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.tokens_per_second = tokens_per_second
        self.hidden_act = hidden_act


class Qwen2VLNeuronConfig(MultimodalVisionNeuronConfig):
    """
    Neuron-specific configuration for Qwen2.5-VL
    Extends MultimodalVisionNeuronConfig for multimodal support
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Will set attn_cls in the model implementation
        # since we need to define the attention class first


class Qwen2VLInferenceConfig(InferenceConfig):
    """
    Inference configuration for Qwen2.5-VL multimodal model
    
    This configuration handles both text and vision components.
    The text model uses Qwen2-style architecture with MRoPE (Multimodal Rotary Position Embeddings).
    """
    
    def __init__(self, *args, **kwargs):
        # Extract vision_config before calling super().__init__
        vision_config_dict = kwargs.pop("vision_config", None)
        
        super().__init__(*args, **kwargs)
        
        # Initialize vision config
        if vision_config_dict is not None:
            if isinstance(vision_config_dict, dict):
                self.vision_config = Qwen2VLVisionConfig(**vision_config_dict)
            else:
                self.vision_config = vision_config_dict

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        
        # Qwen2-VL attention uses bias for QKV, no bias for output
        self.qkv_bias = True
        self.o_bias = False
        
        # MRoPE-specific settings
        # mrope_section defines how to split the head dimension for 3D rotary embeddings
        # [temporal_dim, height_dim, width_dim]
        if hasattr(self, 'rope_scaling') and self.rope_scaling is not None:
            if 'mrope_section' in self.rope_scaling:
                self.mrope_section = self.rope_scaling['mrope_section']
            else:
                # Default MRoPE sections for Qwen2.5-VL
                self.mrope_section = [16, 24, 24]
        else:
            self.mrope_section = [16, 24, 24]
        
        # HuggingFace compatibility attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for validation"""
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
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen2VLNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Qwen2VLNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional configuration overrides
            
        Returns:
            Qwen2VLInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if present
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Expand user path
        model_path = os.path.expanduser(model_path)
        
        # Load config.json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # If neuron_config is not provided, create a default one
        # This happens when loading a compiled model for inference
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import NeuronConfig
            neuron_config = NeuronConfig(
                tp_degree=2,  # Default from compilation
                batch_size=1,
                seq_len=128,
            )
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        
        return config
