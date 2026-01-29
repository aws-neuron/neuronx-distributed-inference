# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Qwen2.5-VL model for NeuronX Distributed Inference
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class Qwen2_5_VLNeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for Qwen2.5-VL model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronQwen2_5_VLAttention


class Qwen2_5_VLInferenceConfig(InferenceConfig):
    """
    Configuration class for Qwen2.5-VL text model inference on NeuronX
    
    This configuration handles the text component of the Qwen2.5-VL multimodal model.
    The vision component is preprocessed and embedded as part of the input sequence.
    """
    
    def __init__(self, neuron_config=None, **kwargs):
        """
        Initialize configuration
        
        Note: neuron_config can be None during initial loading for inference.
        It will be set later by the inference framework.
        """
        # Store the neuron_config temporarily if it's None
        # The base class will handle validation only if neuron_config is not None
        if neuron_config is not None:
            super().__init__(neuron_config=neuron_config, **kwargs)
        else:
            # Temporarily create a minimal neuron_config to pass validation
            # This will be overwritten by the inference framework
            from neuronx_distributed_inference.models.config import NeuronConfig
            temp_config = NeuronConfig(tp_degree=1, batch_size=1, seq_len=512)
            super().__init__(neuron_config=temp_config, **kwargs)
            # Mark that this needs to be replaced
            self._neuron_config_placeholder = True

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        self.qkv_bias = True  # Qwen2.5-VL uses bias in QKV projections
        self.o_bias = False  # No bias in output projection
        
        # Standard HuggingFace config attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        
        # MRoPE configuration - Qwen2.5-VL uses multi-resolution RoPE
        # with sections for [temporal, height, width] dimensions
        if not hasattr(self, 'mrope_section'):
            # Default mrope_section from config
            self.mrope_section = getattr(self, 'mrope_section', [16, 24, 24])

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
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen2_5_VLNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Qwen2_5_VLNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        This handles two scenarios:
        1. Compilation: Loading from HuggingFace model with neuron_config passed in kwargs
        2. Inference: Loading from compiled artifacts (neuron_config.json exists)
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments including neuron_config for compilation
            
        Returns:
            Qwen2_5_VLInferenceConfig: Configuration object
        """
        # Check if we're loading from compiled artifacts (inference scenario)
        neuron_config_path = os.path.join(model_path, "neuron_config.json")
        
        # Extract neuron_config from kwargs if provided (compilation scenario)
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config.json to get model parameters
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Extract text_config if it exists (for full multimodal config)
        # Otherwise use the config directly (for text-only or compiled config)
        if "text_config" in hf_config:
            text_config = hf_config["text_config"]
        else:
            text_config = hf_config
        
        # Map HuggingFace config to our config
        config_dict = {
            "hidden_size": text_config.get("hidden_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "vocab_size": text_config.get("vocab_size"),
            "max_position_embeddings": text_config.get("max_position_embeddings"),
            "rope_theta": text_config.get("rope_theta", 1000000.0),
            "rms_norm_eps": text_config.get("rms_norm_eps", 1e-6),
            "hidden_act": text_config.get("hidden_act", "silu"),
            "intermediate_size": text_config.get("intermediate_size"),
            "pad_token_id": text_config.get("pad_token_id", 151643),
            "attention_dropout": text_config.get("attention_dropout", 0.0),
            "use_cache": text_config.get("use_cache", True),
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        }
        
        # Handle rope_scaling with mrope_section
        rope_scaling = text_config.get("rope_scaling", {})
        if rope_scaling:
            config_dict["rope_scaling"] = rope_scaling
            # Extract mrope_section if available
            if "mrope_section" in rope_scaling:
                config_dict["mrope_section"] = rope_scaling["mrope_section"]
        
        # Sliding window configuration
        config_dict["use_sliding_window"] = text_config.get("use_sliding_window", False)
        config_dict["sliding_window"] = text_config.get("sliding_window", 32768)
        config_dict["max_window_layers"] = text_config.get("max_window_layers", config_dict["num_hidden_layers"])
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronQwen2_5_VLAttention(NeuronAttentionBase):
    """
    Qwen2.5-VL attention implementation for NeuronX
    
    Key differences from standard attention:
    - Uses bias in QKV projections (q_proj, k_proj, v_proj)
    - No bias in output projection (o_proj)
    - Supports MRoPE (Multi-Resolution Rotary Position Embedding)
    - GQA support (40 attention heads, 8 KV heads for 32B model)
    
    Based on Qwen2_5_VLAttention from modeling_qwen2_5_vl.py
    """

    def __init__(self, config: Qwen2_5_VLInferenceConfig):
        # Create rotary embedding with high base theta for long context
        rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,  # Qwen2.5-VL uses 1000000.0 for long context
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            qkv_bias=config.qkv_bias,  # True for Qwen2.5-VL
            o_bias=config.o_bias,      # False for Qwen2.5-VL
            rotary_emb=rotary_emb,
        )


class NeuronQwen2_5_VLDecoderLayer(nn.Module):
    """
    Qwen2.5-VL decoder layer for NeuronX
    
    Structure:
    1. Input LayerNorm (RMSNorm)
    2. Self-Attention with MRoPE
    3. Residual connection
    4. Post-Attention LayerNorm (RMSNorm)
    5. MLP (SwiGLU activation)
    6. Residual connection
    
    Based on Qwen2_5_VLDecoderLayer from modeling_qwen2_5_vl.py
    """

    def __init__(self, config: Qwen2_5_VLInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention module
        self.self_attn = NeuronQwen2_5_VLAttention(config)
        
        # MLP module - can reuse LlamaMLP as Qwen2.5-VL uses same structure
        # gate_proj, up_proj, down_proj with SwiGLU activation
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
        Forward pass for Qwen2.5-VL decoder layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key-value states
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MLP returns (output, None)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen2_5_VLModel(NeuronBaseModel):
    """
    Qwen2.5-VL text model for NeuronX
    
    This implements the text decoder portion of the Qwen2.5-VL multimodal model.
    For inference, vision inputs are preprocessed and embedded as special tokens
    in the input sequence.
    
    Architecture:
    - Token embeddings (ParallelEmbedding)
    - Stack of decoder layers
    - Final RMSNorm
    - LM head for text generation
    
    Based on Qwen2_5_VLTextModel from modeling_qwen2_5_vl.py
    """

    def setup_attr_for_model(self, config: Qwen2_5_VLInferenceConfig):
        """Setup attributes for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen2_5_VLInferenceConfig):
        """Initialize the model components"""
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronQwen2_5_VLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head for generation
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronQwen2_5_VLForCausalLM(NeuronBaseForCausalLM):
    """
    Qwen2.5-VL causal language model for NeuronX inference
    
    This class wraps the Qwen2.5-VL model for text generation.
    For multimodal inputs, vision tokens should be preprocessed and
    embedded in the input sequence before passing to this model.
    """

    _model_cls = NeuronQwen2_5_VLModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace model"""
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format
        
        Key mappings:
        - model.embed_tokens.weight -> model.embed_tokens.weight
        - model.layers.X.self_attn.q_proj.weight -> model.layers.X.self_attn.qkv_proj.q_proj.weight
        - model.layers.X.self_attn.k_proj.weight -> model.layers.X.self_attn.qkv_proj.k_proj.weight
        - model.layers.X.self_attn.v_proj.weight -> model.layers.X.self_attn.qkv_proj.v_proj.weight
        - model.layers.X.self_attn.o_proj.weight -> model.layers.X.self_attn.o_proj.weight
        - model.layers.X.mlp.gate_proj.weight -> model.layers.X.mlp.gate_proj.weight
        - model.layers.X.mlp.up_proj.weight -> model.layers.X.mlp.up_proj.weight
        - model.layers.X.mlp.down_proj.weight -> model.layers.X.mlp.down_proj.weight
        - model.norm.weight -> model.norm.weight
        - lm_head.weight -> lm_head.weight (if not tied)
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        
        # Map weights from HF format to Neuron format
        for name, param in state_dict.items():
            # Skip visual components for now (text-only model)
            if 'visual' in name or 'vision' in name:
                continue
            
            # Handle attention QKV projections
            if '.self_attn.q_proj.' in name:
                new_name = name.replace('.self_attn.q_proj.', '.self_attn.qkv_proj.q_proj.')
                neuron_state_dict[new_name] = param.clone()
            elif '.self_attn.k_proj.' in name:
                new_name = name.replace('.self_attn.k_proj.', '.self_attn.qkv_proj.k_proj.')
                neuron_state_dict[new_name] = param.clone()
            elif '.self_attn.v_proj.' in name:
                new_name = name.replace('.self_attn.v_proj.', '.self_attn.qkv_proj.v_proj.')
                neuron_state_dict[new_name] = param.clone()
            else:
                # Copy other weights as-is
                neuron_state_dict[name] = param.clone()
        
        # Add rank utilities for tensor parallel support
        if neuron_config.vocab_parallel:
            neuron_state_dict["model.embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"model.layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Update state dict for models with tied embeddings"""
        # Qwen2.5-VL typically doesn't tie weights, but handle it if needed
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class"""
        return Qwen2_5_VLInferenceConfig

    def get_compiler_args(self):
        """
        Get compiler arguments for Neuron compilation
        
        Returns:
            String of compiler flags optimized for Qwen2.5-VL
        """
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        # Add flags for compute-communication overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
