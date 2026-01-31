# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Idefics model for NXD inference."""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm implementation.
    CustomRMSNorm is optimized for NeuronX hardware.
    """
    return CustomRMSNorm


class IdeficsInferenceConfig(InferenceConfig):
    """
    Configuration class for Idefics model inference on NeuronX hardware.
    
    This extends InferenceConfig with Idefics-specific parameters including
    vision configuration, perceiver configuration, and gated cross-attention settings.
    """
    
    def __init__(self, neuron_config: NeuronConfig = None, **kwargs):
        super().__init__(neuron_config=neuron_config, **kwargs)
        
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        
        # Ensure vision_config and perceiver_config are present
        if not hasattr(self, 'vision_config'):
            self.vision_config = {}
        if not hasattr(self, 'perceiver_config'):
            self.perceiver_config = {}
        
        # Add standard HF config attributes if not present
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        if not hasattr(self, 'return_dict'):
            self.return_dict = True
            
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "hidden_act",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "IdeficsInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            IdeficsInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # If neuron_config not provided, try to load from compiled model directory
        if neuron_config is None:
            neuron_config_path = os.path.join(model_path, "neuron_config.json")
            if os.path.exists(neuron_config_path):
                with open(neuron_config_path, "r") as f:
                    neuron_config_dict = json.load(f)
                neuron_config = NeuronConfig(**neuron_config_dict)
            else:
                # Create a default NeuronConfig if not found
                # This is needed for inference when loading from HF model path
                print(f"⚠️  neuron_config.json not found at {neuron_config_path}, creating default NeuronConfig")
                neuron_config = NeuronConfig(
                    tp_degree=1,
                    max_batch_size=1,
                    buckets=[128],
                    torch_dtype="bfloat16",
                )
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        
        # Set _name_or_path for checkpoint loading
        config._name_or_path = model_path
        
        return config


class NeuronIdeficsAttention(NeuronAttentionBase):
    """
    Idefics attention implementation for NeuronX.
    
    This implements both self-attention (with RoPE) and cross-attention (without RoPE).
    Supports optional Q-K layer normalization for improved stability.
    
    Reference: IdeficsAttention in modeling_idefics.py
    """
    
    def __init__(
        self,
        config: IdeficsInferenceConfig,
        is_cross_attention: bool = False,
        qk_layer_norms: bool = False,
    ):
        """
        Initialize Idefics attention module.
        
        Args:
            config: Model configuration
            is_cross_attention: Whether this is a cross-attention layer
            qk_layer_norms: Whether to apply layer norm to queries and keys
        """
        self.is_cross_attention = is_cross_attention
        self.qk_layer_norms = qk_layer_norms
        
        # Only use RoPE for self-attention, not cross-attention
        rotary_emb = None
        if not is_cross_attention:
            rotary_emb = RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=getattr(config, "max_position_embeddings", 2048),
                base=getattr(config, "rope_theta", 10000.0),
            )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Idefics uses MHA, not GQA
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=rotary_emb,
            rope_theta=getattr(config, "rope_theta", 10000.0),
        )
        
        # Q-K layer norms for improved stability
        if self.qk_layer_norms:
            head_dim = config.hidden_size // config.num_attention_heads
            self.q_layer_norm = get_rmsnorm_cls()(
                head_dim,
                eps=config.rms_norm_eps,
            )
            self.k_layer_norm = get_rmsnorm_cls()(
                head_dim,
                eps=config.rms_norm_eps,
            )


class NeuronIdeficsMLP(nn.Module):
    """
    Idefics MLP implementation for NeuronX using SwiGLU activation.
    
    This uses the gated linear unit pattern: down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Reference: IdeficsMLP in modeling_idefics.py
    """
    
    def __init__(self, config: IdeficsInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate projection (for gating activation)
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Up projection (for value)
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Down projection (output)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function (SiLU for SwiGLU)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        """
        Forward pass using SwiGLU activation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # SwiGLU: silu(gate_proj(x)) * up_proj(x)
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        
        # Project back to hidden size
        output = self.down_proj(intermediate)
        
        return output


class NeuronIdeficsDecoderLayer(nn.Module):
    """
    Idefics decoder layer with self-attention and MLP.
    
    This is a standard transformer decoder layer without cross-attention.
    
    Reference: IdeficsDecoderLayer in modeling_idefics.py
    """
    
    def __init__(self, config: IdeficsInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Self-attention
        self.self_attn = NeuronIdeficsAttention(
            config=config,
            is_cross_attention=False,
            qk_layer_norms=False,  # Standard decoder layers don't use Q-K norms
        )
        
        # MLP
        self.mlp = NeuronIdeficsMLP(config)
        
        # Layer norms
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
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return in framework-expected format
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronIdeficsGatedCrossAttentionLayer(nn.Module):
    """
    Idefics gated cross-attention layer for vision-text fusion.
    
    This layer performs cross-attention from text to vision features, with gated
    residual connections controlled by learnable alpha parameters.
    
    NOTE: For initial text-only implementation, this layer will be simplified to
    pass through the input without vision features.
    
    Reference: IdeficsGatedCrossAttentionLayer in modeling_idefics.py
    """
    
    def __init__(self, config: IdeficsInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Cross-attention (from text to vision)
        self.cross_attn = NeuronIdeficsAttention(
            config=config,
            is_cross_attention=True,
            qk_layer_norms=getattr(config, 'qk_layer_norms', False),
        )
        
        # MLP
        self.mlp = NeuronIdeficsMLP(config)
        
        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # Gating parameters
        alpha_type = getattr(config, 'alpha_type', 'float')
        if alpha_type == 'float':
            self.alpha_cross_attn = nn.Parameter(torch.zeros(1))
            self.alpha_dense = nn.Parameter(torch.zeros(1))
        elif alpha_type == 'vector':
            self.alpha_cross_attn = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            self.alpha_dense = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        else:
            raise ValueError(f"Unknown alpha_type: {alpha_type}")
        
        # Gating activations
        self.act_cross_attn = nn.Tanh()
        self.act_dense = nn.Tanh()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for gated cross-attention layer.
        
        For text-only mode (when image_hidden_states is None), this acts as identity.
        
        Args:
            hidden_states: Text hidden states
            attention_mask: Text attention mask
            position_ids: Position indices
            image_hidden_states: Vision features (optional)
            image_attention_mask: Vision attention mask (optional)
            cross_attention_gate: Gate to zero out non-image tokens (optional)
            
        Returns:
            Updated hidden states
        """
        # For text-only mode, just pass through
        # TODO: Implement full cross-attention when adding vision support
        if image_hidden_states is None:
            return hidden_states
        
        # Cross-attention with gated residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # NOTE: Cross-attention would go here, but requires custom implementation
        # For now, we'll use a simplified version that just returns zeros
        # This will be fully implemented when vision support is added
        cross_attn_output = torch.zeros_like(hidden_states)
        
        # Apply gating
        if cross_attention_gate is not None:
            cross_attn_output = cross_attn_output.masked_fill(
                (cross_attention_gate == 0)[:, :, None], 0.0
            )
        
        hidden_states = residual + self.act_cross_attn(self.alpha_cross_attn) * cross_attn_output
        
        # MLP with gated residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states
        
        return hidden_states


class NeuronIdeficsModel(NeuronBaseModel):
    """
    Idefics model implementation for NeuronX hardware.
    
    This is the main model class that combines embeddings, decoder layers,
    cross-attention layers, and final normalization.
    
    NOTE: This initial implementation focuses on text-only inference.
    Vision components (vision_model, perceiver_resampler) are placeholders.
    """
    
    def setup_attr_for_model(self, config: IdeficsInferenceConfig):
        """Setup attributes required by the NeuronX framework."""
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # MHA for Idefics
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
    def init_model(self, config: IdeficsInferenceConfig):
        """Initialize model components."""
        # Embeddings (supports additional vocab for special image tokens)
        vocab_size = config.vocab_size
        additional_vocab_size = getattr(config, 'additional_vocab_size', 0)
        total_vocab_size = vocab_size + additional_vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            total_vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [
                NeuronIdeficsDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        
        # Cross-attention layers (inserted at regular intervals)
        # TODO: Implement full cross-attention with vision features
        # For now, skip cross-attention layers for text-only compilation
        self.cross_layer_interval = getattr(config, 'cross_layer_interval', 4)
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                None  # Placeholder - will implement when adding vision support
                for i in range(num_cross_layers)
            ]
        )
        
        # Final normalization
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # LM head (output projection to vocabulary)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            total_vocab_size,
            bias=False,
            gather_output=not config.neuron_config.vocab_parallel,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Vision components (placeholders for future implementation)
        # TODO: Implement vision_model and perceiver_resampler
        self.vision_model = None
        self.perceiver_resampler = None


class NeuronIdeficsForCausalLM(NeuronBaseForCausalLM):
    """
    Idefics model for causal language modeling on NeuronX hardware.
    
    This wraps the base model and provides the interface for compilation
    and inference.
    """
    
    _model_cls = NeuronIdeficsModel
    
    @classmethod
    def from_config(cls, config: IdeficsInferenceConfig):
        """
        Create a model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            NeuronIdeficsForCausalLM: Model instance
        """
        return cls(config=config)
    
    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        """
        Load the HuggingFace model for weight extraction.
        
        Since Idefics is a custom model, we'll load weights directly from safetensors.
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments
            
        Returns:
            A dummy object that allows weight loading
        """
        from transformers import AutoConfig, AutoModel
        
        # Try to load using AutoModel (which should work with custom models in the transformers repo)
        try:
            model = AutoModel.from_pretrained(model_path, **kwargs)
            return model
        except Exception as e:
            print(f"Warning: Could not load model with AutoModel: {e}")
            print("Loading weights directly from safetensors/pytorch files...")
            
            # Return a simple namespace with state_dict method that loads from files
            class DummyModel:
                def __init__(self, model_path):
                    self.model_path = model_path
                
                def state_dict(self):
                    """Load state dict from safetensors or pytorch files."""
                    from safetensors.torch import load_file
                    import glob
                    
                    state_dict = {}
                    
                    # Try safetensors first
                    safetensors_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))
                    if safetensors_files:
                        print(f"Loading from {len(safetensors_files)} safetensors files...")
                        for file_path in safetensors_files:
                            state_dict.update(load_file(file_path))
                        return state_dict
                    
                    # Fall back to pytorch files
                    pytorch_files = sorted(glob.glob(os.path.join(self.model_path, "pytorch_model*.bin")))
                    if pytorch_files:
                        print(f"Loading from {len(pytorch_files)} pytorch files...")
                        for file_path in pytorch_files:
                            state_dict.update(torch.load(file_path, map_location="cpu"))
                        return state_dict
                    
                    raise FileNotFoundError(f"No model weights found in {self.model_path}")
            
            return DummyModel(model_path)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: IdeficsInferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This maps weight names from the original Idefics format to the NeuronX format.
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary in NeuronX format
        """
        print(f"\n=== Converting HF state dict to Neuron format ===")
        print(f"Input state dict keys (first 5): {list(state_dict.keys())[:5]}")
        
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        
        # Embedding conversion
        # HF format: embed_tokens.weight (main) + embed_tokens.additional_embedding.weight (additional)
        # NOTE: The "model." prefix has already been removed by the framework
        if "embed_tokens.weight" in state_dict:
            main_emb = state_dict["embed_tokens.weight"].clone()
            additional_emb = state_dict.get("embed_tokens.additional_embedding.weight")
            if additional_emb is not None:
                # Concatenate main and additional embeddings
                neuron_state_dict["embed_tokens.weight"] = torch.cat([main_emb, additional_emb], dim=0)
            else:
                neuron_state_dict["embed_tokens.weight"] = main_emb
        
        # Final norm conversion
        if "norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["norm.weight"].clone()
        
        # LM head conversion
        # HF format: lm_head.weight (main) + lm_head.additional_fc.weight (additional)
        if "lm_head.weight" in state_dict:
            main_lm = state_dict["lm_head.weight"].clone()
            additional_lm = state_dict.get("lm_head.additional_fc.weight")
            if additional_lm is not None:
                neuron_state_dict["lm_head.weight"] = torch.cat([main_lm, additional_lm], dim=0)
            else:
                neuron_state_dict["lm_head.weight"] = main_lm
        
        # Decoder layers conversion
        for i in range(config.num_hidden_layers):
            layer_prefix = f"layers.{i}"
            
            # Self-attention Q, K, V, O projections
            # NOTE: Keys are already without "model." prefix
            # Framework expects them directly under self_attn (not self_attn.qkv_proj)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                hf_key = f"{layer_prefix}.self_attn.{proj}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                hf_key = f"{layer_prefix}.mlp.{proj}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Layer norms
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                hf_key = f"{layer_prefix}.{norm}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
        
        # Cross-attention layers conversion
        num_cross_layers = config.num_hidden_layers // getattr(config, 'cross_layer_interval', 4)
        for i in range(num_cross_layers):
            cross_prefix = f"gated_cross_attn_layers.{i}"
            
            # Cross-attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                hf_key = f"{cross_prefix}.cross_attn.{proj}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Cross-attention Q-K layer norms (if present)
            for norm in ["q_layer_norm", "k_layer_norm"]:
                hf_key = f"{cross_prefix}.cross_attn.{norm}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Cross-attention MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                hf_key = f"{cross_prefix}.mlp.{proj}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Cross-attention layer norms
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                hf_key = f"{cross_prefix}.{norm}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Alpha gating parameters
            for alpha in ["alpha_cross_attn", "alpha_dense"]:
                hf_key = f"{cross_prefix}.{alpha}"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
        
        # Add rank utilities for tensor parallelism
        tp_degree = neuron_config.tp_degree
        
        # Add rank for each decoder layer attention
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank for cross-attention layers
        for i in range(num_cross_layers):
            neuron_state_dict[f"gated_cross_attn_layers.{i}.cross_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank for base model (if needed by framework)
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )
        
        print(f"Output state dict keys (first 10): {list(neuron_state_dict.keys())[:10]}")
        print(f"Total keys converted: {len(neuron_state_dict)}")
        print("=== Conversion complete ===\n")
        
        return neuron_state_dict
