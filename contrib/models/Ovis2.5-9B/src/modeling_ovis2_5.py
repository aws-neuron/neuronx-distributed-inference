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
PyTorch Ovis2.5-9B model for NeuronX Distributed Inference.

This implementation ports the Ovis2.5-9B multimodal model to NeuronX.
For initial implementation, we only port the text-only LLM component (Qwen3-8B).
Vision components can be added later if needed.

Model Architecture:
- LLM: Qwen3-8B (36 layers, 4096 hidden, 32 heads, 8 KV heads - GQA)
- Visual Tokenizer: Siglip2-NavIT (not ported in initial version)
- Visual Embedding Table: 65536 x 4096 (not ported in initial version)

Weight Structure:
- Original Ovis2.5 checkpoint has weights prefixed with "llm.", "visual_tokenizer.", "vte."
- We extract only the "llm." prefixed weights and map them to Qwen3 format
- Weight mapping: "llm.model.*" -> Qwen3 model weights, "llm.lm_head.*" -> Qwen3 lm_head
"""

from typing import Optional

import torch
from transformers import Qwen3ForCausalLM

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import (
    NeuronQwen3ForCausalLM,
    NeuronQwen3Model,
)

try:
    from .configuration_ovis2_5 import Ovis2_5_InferenceConfig
except ImportError:
    from configuration_ovis2_5 import Ovis2_5_InferenceConfig


class NeuronOvis2_5_Model(NeuronQwen3Model):
    """
    Ovis2.5 base model for NeuronX.
    
    This inherits from NeuronQwen3Model since the LLM backbone is Qwen3.
    We reuse all the Qwen3 implementation including:
    - Attention with Q-K normalization
    - MLP with SwiGLU activation
    - RMSNorm layers
    - Rotary position embeddings
    """

    def __init__(self, config: Ovis2_5_InferenceConfig):
        # Initialize as Qwen3 model
        super().__init__(config)


class NeuronOvis2_5_ForCausalLM(NeuronQwen3ForCausalLM):
    """
    Ovis2.5 model for causal language modeling on NeuronX.
    
    This wraps the Qwen3 LLM component of the Ovis2.5 multimodal model.
    For text-only inference, we extract and use only the LLM weights.
    
    Weight Loading:
    - Ovis2.5 checkpoint structure: {"llm.*", "visual_tokenizer.*", "vte.*"}
    - We extract only "llm.*" weights
    - Map "llm.model.*" -> Qwen3 model weights
    - Map "llm.lm_head.*" -> Qwen3 lm_head
    
    Usage:
        config = Ovis2_5_InferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronOvis2_5_ForCausalLM(model_path, config)
        model.compile(output_path)
    """

    _model_cls = NeuronOvis2_5_Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load the HuggingFace model.
        
        Note: Ovis2.5 uses a custom model class, so we need to handle loading differently.
        For weight conversion, we can load the state dict directly or use the Qwen3 loader.
        """
        # For Ovis2.5, we use the Qwen3 loader since we're only using the LLM component
        # The actual weight loading is handled in convert_hf_to_neuron_state_dict
        return Qwen3ForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert Ovis2.5 checkpoint to NeuronX format.
        
        This function:
        1. Extracts LLM weights from Ovis2.5 checkpoint (keys starting with "llm.")
        2. Removes the "llm." prefix
        3. Applies Qwen3 state dict conversion
        
        Weight Mapping:
        - "llm.model.embed_tokens.weight" -> "embed_tokens.weight"
        - "llm.model.layers.{i}.*" -> "layers.{i}.*"
        - "llm.model.norm.weight" -> "norm.weight"
        - "llm.lm_head.weight" -> "lm_head.weight"
        
        Args:
            state_dict: Original Ovis2.5 state dict with "llm.", "visual_tokenizer.", "vte." prefixes
            config: Model configuration
            
        Returns:
            Neuron-compatible state dict for Qwen3 model
        """
        neuron_config = config.neuron_config

        # Step 1: Extract LLM weights and remove "llm." prefix
        llm_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("llm."):
                # Remove "llm." prefix
                new_key = key[4:]  # Skip "llm."
                llm_state_dict[new_key] = value.clone()

        # Debug: Print extracted keys
        print(f"Extracted {len(llm_state_dict)} LLM weights from Ovis2.5 checkpoint")
        if len(llm_state_dict) == 0:
            print("WARNING: No LLM weights found! Available prefixes:")
            prefixes = set([k.split(".")[0] for k in state_dict.keys()])
            print(f"  {prefixes}")

        # Step 2: Apply Qwen3 state dict conversion
        # This handles:
        # - Renaming "q_norm" to "q_layernorm"
        # - Renaming "k_norm" to "k_layernorm"  
        # - Adding rank tensors for tensor parallelism
        neuron_state_dict = {}

        # Add vocab parallel rank if needed
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Process layer weights
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for key, value in llm_state_dict.items():
            # Skip "model." prefix if present
            if key.startswith("model."):
                key = key[6:]  # Remove "model." prefix

            neuron_state_dict[key] = value

        # Add layer-specific conversions
        for i in range(num_layers):
            # Add rank tensors for attention
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # Rename q_norm and k_norm to q_layernorm and k_layernorm
            if f"layers.{i}.self_attn.q_norm.weight" in llm_state_dict:
                neuron_state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                    llm_state_dict[f"layers.{i}.self_attn.q_norm.weight"]
                    .detach()
                    .clone()
                )
            elif f"model.layers.{i}.self_attn.q_norm.weight" in llm_state_dict:
                neuron_state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                    llm_state_dict[f"model.layers.{i}.self_attn.q_norm.weight"]
                    .detach()
                    .clone()
                )

            if f"layers.{i}.self_attn.k_norm.weight" in llm_state_dict:
                neuron_state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                    llm_state_dict[f"layers.{i}.self_attn.k_norm.weight"]
                    .detach()
                    .clone()
                )
            elif f"model.layers.{i}.self_attn.k_norm.weight" in llm_state_dict:
                neuron_state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                    llm_state_dict[f"model.layers.{i}.self_attn.k_norm.weight"]
                    .detach()
                    .clone()
                )

        # Add base model rank tensor
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        print(f"Converted to {len(neuron_state_dict)} Neuron weights")

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied weights.
        
        Qwen3 ties the embedding and lm_head weights.
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the config class for this model"""
        return Ovis2_5_InferenceConfig


__all__ = [
    "NeuronOvis2_5_Model",
    "NeuronOvis2_5_ForCausalLM",
]
