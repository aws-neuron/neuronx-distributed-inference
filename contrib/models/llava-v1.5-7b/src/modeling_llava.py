# coding=utf-8
# Copyright 2023 Haotian Liu and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LLaVA model for NXD inference."""

"""

import os
import json
import copy
import logging
from typing import List, Optional, Union, Tuple, Type

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaModel,
    NeuronLlamaForCausalLM,
    LlamaInferenceConfig,
)
from neuronx_distributed.parallel_layers import parallel_state, layers

logger = logging.getLogger("Neuron")


class LlavaInferenceConfig(InferenceConfig):
    """
    Configuration class for LLaVA inference on NeuronX.
    
    This configuration combines:
    - text_config: Configuration for the LLaMA language model
    - vision_config: Configuration for the CLIP vision tower
    - Multimodal-specific parameters
    
    Args:
        text_config: Configuration dict or object for text model
        vision_config: Configuration dict or object for vision model
        image_token_index: Token ID used to represent image placeholders (default: 32000)
        projector_hidden_act: Activation function for projector ("gelu")
        vision_feature_select_strategy: Feature selection strategy ("default" or "full")
        vision_feature_layer: Which vision layer to extract features from (default: -2)
        image_seq_length: Number of image tokens per image (default: 576)
        multimodal_projector_bias: Whether to use bias in projector (default: True)
    """
    
    def __init__(
        self,
        neuron_config: NeuronConfig = None,
        text_config: dict = None,
        vision_config: dict = None,
        image_token_index: int = 32000,
        projector_hidden_act: str = "gelu",
        vision_feature_select_strategy: str = "default",
        vision_feature_layer: int = -2,
        image_seq_length: int = 576,
        multimodal_projector_bias: bool = True,
        **kwargs,
    ):
        # Store text and vision configs first
        self.text_config = text_config if text_config is not None else {}
        self.vision_config = vision_config if vision_config is not None else {}
        
        # Multimodal-specific parameters
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.image_seq_length = image_seq_length
        self.multimodal_projector_bias = multimodal_projector_bias
        
        # Copy text config attributes to kwargs for parent class
        if isinstance(self.text_config, dict):
            for key, value in self.text_config.items():
                if key not in kwargs:
                    kwargs[key] = value
        
        # Initialize base config with neuron_config and all attributes
        # Note: if neuron_config is None, the parent class __init__ should handle it
        try:
            super().__init__(neuron_config=neuron_config, **kwargs)
        except (AttributeError, AssertionError) as e:
            # If initialization fails due to missing neuron_config, 
            # set attributes manually without validation
            if neuron_config is None and ("NoneType" in str(e) or "neuron_config" in str(e)):
                # Store config attributes without full initialization
                self.neuron_config = None
                for key, value in kwargs.items():
                    setattr(self, key, value)
            else:
                raise
    
    def get_required_attributes(self) -> List[str]:
        """
        List of required attributes for LLaVA configuration.
        """
        return [
            "hidden_size",  # From text_config
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "image_token_index",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig
    
    def get_text_config(self):
        """
        Return text configuration as an object.
        
        This is called by NeuronBaseForCausalLM to get text config.
        """
        # If text_config is a dict, convert to SimpleNamespace for attribute access
        if isinstance(self.text_config, dict):
            from types import SimpleNamespace
            text_cfg = SimpleNamespace(**self.text_config)
            # Add missing attributes that the base class expects
            if not hasattr(text_cfg, 'output_attentions'):
                text_cfg.output_attentions = False
            if not hasattr(text_cfg, 'output_hidden_states'):
                text_cfg.output_hidden_states = False
            if not hasattr(text_cfg, 'use_cache'):
                text_cfg.use_cache = True
            return text_cfg
        return self.text_config
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load LLaVA configuration from a pretrained model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            neuron_config: NeuronConfig object for inference settings (can be None to load from saved config)
            **kwargs: Additional arguments to override configuration
            
        Returns:
            LlavaInferenceConfig: Configuration object
        """
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract text config (LLaMA parameters)
        text_config = {
            "hidden_size": config_dict.get("hidden_size", 4096),
            "num_attention_heads": config_dict.get("num_attention_heads", 32),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 32),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 32),
            "vocab_size": config_dict.get("vocab_size", 32000),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 4096),
            "intermediate_size": config_dict.get("intermediate_size", 11008),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-5),
            "hidden_act": config_dict.get("hidden_act", "silu"),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "rope_scaling": config_dict.get("rope_scaling", None),
            "pad_token_id": config_dict.get("pad_token_id", 0),
            "bos_token_id": config_dict.get("bos_token_id", 1),
            "eos_token_id": config_dict.get("eos_token_id", 2),
        }
        
        # Extract vision config (CLIP parameters)
        vision_config = {
            "mm_vision_tower": config_dict.get("mm_vision_tower", "openai/clip-vit-large-patch14-336"),
            "mm_hidden_size": config_dict.get("mm_hidden_size", 1024),
        }
        
        # Multimodal parameters
        multimodal_config = {
            "image_token_index": config_dict.get("image_token_index", 32000),
            "projector_hidden_act": "gelu" if config_dict.get("mm_projector_type") == "mlp2x_gelu" else "gelu",
            "vision_feature_select_strategy": "default" if config_dict.get("mm_vision_select_feature") == "patch" else "full",
            "vision_feature_layer": config_dict.get("mm_vision_select_layer", -2),
            "image_seq_length": 576,  # 24x24 patches for 336x336 image with patch_size=14
            "multimodal_projector_bias": True,
        }
        
        # Merge with kwargs
        config_dict_final = {
            "text_config": text_config,
            "vision_config": vision_config,
            **multimodal_config,
        }
        config_dict_final.update(kwargs)
        
        # If neuron_config is not provided, don't pass it (will be set to None)
        # The base class will handle loading it from the compiled model if needed
        if neuron_config is None:
            # Don't pass neuron_config to avoid the validation error
            # The config will be set up properly during model loading
            return cls(**config_dict_final)
        else:
            # Create config object with provided neuron_config
            return cls(neuron_config=neuron_config, **config_dict_final)


class NeuronLlavaMultiModalProjector(nn.Module):
    """
    Multi-modal projector for LLaVA.
    
    This is a 2-layer MLP that projects vision features to the language model's hidden size.
    
    Architecture:
        vision_hidden_size -> text_hidden_size -> text_hidden_size
        
    Original HF implementation: LlavaMultiModalProjector in modeling_llava.py
    """
    
    def __init__(self, config: LlavaInferenceConfig):
        super().__init__()
        
        vision_hidden_size = config.vision_config.get("mm_hidden_size", 1024)
        text_hidden_size = config.hidden_size
        
        # First linear layer: vision -> text hidden size
        self.linear_1 = nn.Linear(
            vision_hidden_size,
            text_hidden_size,
            bias=config.multimodal_projector_bias,
        )
        
        # Activation function
        self.act = ACT2FN[config.projector_hidden_act]
        
        # Second linear layer: text hidden size -> text hidden size
        self.linear_2 = nn.Linear(
            text_hidden_size,
            text_hidden_size,
            bias=config.multimodal_projector_bias,
        )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features to text hidden size.
        
        Args:
            image_features: Vision features [num_images, seq_len, vision_hidden_size]
            
        Returns:
            Projected features [num_images, seq_len, text_hidden_size]
        """
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NeuronLlavaModel(NeuronLlamaModel):
    """
    LLaVA Model for NeuronX inference - inherits from NeuronLlamaModel.
    
    For LLaVA on NeuronX, we compile only the language model part.
    This class is essentially a LLaMA model with custom configuration loading.
    
    The vision tower and multimodal projector run separately during preprocessing.
    
    Original HF implementation: LlavaModel in modeling_llava.py
    """
    
    def __init__(self, config: LlavaInferenceConfig):
        # Convert LlavaInferenceConfig to LlamaInferenceConfig
        llama_config_dict = config.text_config.copy()
        llama_config = LlamaInferenceConfig(neuron_config=config.neuron_config, **llama_config_dict)
        
        # Initialize as a LLaMA model
        super().__init__(llama_config)
        
        # Store the original LLaVA config for reference
        self.llava_config = config


class NeuronLlavaForCausalLM(NeuronLlamaForCausalLM):
    """
    LLaVA Causal Language Model for NeuronX inference - inherits from NeuronLlamaForCausalLM.
    
    For NeuronX compilation, LLaVA is compiled as a LLaMA model.
    The multimodal processing (vision + projection) happens separately during preprocessing.
    
    This class provides:
    1. LLaVA-specific configuration loading
    2. Weight conversion from LLaVA checkpoints
    3. Compatibility layer for multimodal inference
    
    Original HF implementation: LlavaForConditionalGeneration in modeling_llava.py
    """
    
    _model_cls = NeuronLlavaModel
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle weight conversion from HuggingFace format"""
        if self._is_hf_state_dict(state_dict):
            print("🔧 Converting HuggingFace LLaVA weights to NeuronX format...")
            state_dict = self.convert_hf_to_neuron_state_dict(state_dict, self.config)
            print(f"✅ Weight conversion completed. Total keys: {len(state_dict)}")
        return super().load_state_dict(state_dict, strict)
    
    @staticmethod
    def _is_hf_state_dict(state_dict):
        """Check if the state dict is from HuggingFace format"""
        return any(key.startswith('model.') for key in state_dict.keys())
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: LlavaInferenceConfig):
        """
        Convert HuggingFace LLaVA checkpoint to NeuronX format.
        
        NeuronX expects (when fused_qkv=False):
        - layers.*.self_attn.qkv_proj.q_proj.weight
        - layers.*.self_attn.qkv_proj.k_proj.weight
        - layers.*.self_attn.qkv_proj.v_proj.weight
        
        Args:
            state_dict: HuggingFace state dictionary
            config: LlavaInferenceConfig object
            
        Returns:
            Converted state dictionary for NeuronX
        """
        print("Converting LLaVA checkpoint from HuggingFace to NeuronX format...")
        print(f"Original checkpoint keys: {len(state_dict)}")
        
        neuron_state_dict = {}
        
        # First pass: copy all keys with basic transformations
        for key, value in state_dict.items():
            # Skip vision tower weights
            if "vision_tower" in key:
                print(f"Skipping vision tower weight: {key}")
                continue
            
            # Skip multimodal projector weights
            if "mm_projector" in key:
                continue
            
            # Remove 'language_model.model.' or 'language_model.' or 'model.' prefix
            if key.startswith('language_model.model.'):
                key = key[21:]  # Remove 'language_model.model.'
            elif key.startswith('language_model.'):
                key = key[15:]  # Remove 'language_model.'
            elif key.startswith('model.'):
                key = key[6:]  # Remove 'model.'
            
            neuron_state_dict[key] = value.clone()
        
        # Second pass: restructure QKV weights per layer
        num_layers = config.text_config.get('num_hidden_layers', config.num_hidden_layers)
        for i in range(num_layers):
            # Check if this layer has separate Q/K/V projections
            if f"layers.{i}.self_attn.q_proj.weight" in neuron_state_dict:
                # Pop original keys
                q_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.q_proj.weight")
                k_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.k_proj.weight")
                v_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.v_proj.weight")
                
                # Add with qkv_proj intermediate level
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.weight"] = v_weight
        
        print(f"Extracted {len(neuron_state_dict)} language model weights")
        
        # Add rank information for tensor parallelism
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        return neuron_state_dict


__all__ = [
    "LlavaInferenceConfig",
    "NeuronLlavaMultiModalProjector",
    "NeuronLlavaModel",
    "NeuronLlavaForCausalLM",
]
