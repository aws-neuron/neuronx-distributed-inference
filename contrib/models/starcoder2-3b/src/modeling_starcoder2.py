# coding=utf-8
# Copyright 2024 BigCode and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Starcoder2 model for NXD inference

This implementation is based on transformers/models/starcoder2/modeling_starcoder2.py
and adapted for AWS Neuron/Trainium using the NeuronxDistributedInference framework.

Key differences from HuggingFace implementation:
- Uses NeuronAttentionBase for attention
- Uses NeuronBaseModel for model structure
- Uses parallel layers (ColumnParallelLinear, RowParallelLinear, ParallelEmbedding)
- Supports tensor parallelism and distributed inference
- No custom forward() method in main model class (framework handles it)
"""

from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from torch.nn import functional as F

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class Starcoder2NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for Starcoder2
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronStarcoder2Attention


class Starcoder2InferenceConfig(InferenceConfig):
    """
    Configuration class for Starcoder2 inference on Neuron.

    This configuration extends InferenceConfig with Starcoder2-specific parameters.
    """

    def add_derived_config(self):
        """Add derived configuration parameters required by the framework"""
        self.num_cores_per_group = 1
        # head_dim is optional in config, calculate if not present
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Add framework-required attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for Starcoder2 configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "hidden_act",
            "norm_epsilon",
            "use_bias",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Starcoder2NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from HuggingFace format

        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional parameters to override config values

        Returns:
            Starcoder2InferenceConfig instance
        """
        import json
        import os

        # Extract neuron_config from kwargs if present
        neuron_config = kwargs.pop("neuron_config", None)

        # Read config.json from model directory
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as f:
            hf_config = json.load(f)

        # Extract required parameters
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 3072),
            "num_attention_heads": hf_config.get("num_attention_heads", 24),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 30),
            "num_key_value_heads": hf_config.get("num_key_value_heads", 2),
            "vocab_size": hf_config.get("vocab_size", 49152),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 4096),
            "intermediate_size": hf_config.get("intermediate_size", 12288),
            "hidden_act": hf_config.get("hidden_act", "gelu_pytorch_tanh"),
            "norm_epsilon": hf_config.get("norm_epsilon", 1e-5),
            "use_bias": hf_config.get("use_bias", True),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "sliding_window": None,  # Disabled sliding window to avoid compilation issues
            "pad_token_id": hf_config.get("pad_token_id", None),
            "bos_token_id": hf_config.get("bos_token_id", 50256),
            "eos_token_id": hf_config.get("eos_token_id", 50256),
            # Starcoder2 ALWAYS ties embeddings (no separate lm_head in checkpoint)
            "tie_word_embeddings": True,
        }

        # Calculate head_dim if not present
        if "head_dim" not in hf_config:
            config_dict["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]
        else:
            config_dict["head_dim"] = hf_config["head_dim"]

        # Apply overrides from kwargs
        config_dict.update(kwargs)

        # Create and return config
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronStarcoder2Attention(NeuronAttentionBase):
    """
    Starcoder2 attention implementation for NeuronX

    Based on transformers Starcoder2Attention but adapted for Neuron hardware.
    Uses NeuronAttentionBase which provides:
    - Optimized attention computation with flash attention
    - KV cache management
    - RoPE integration
    - Tensor parallelism support
    """

    def __init__(self, config: Starcoder2InferenceConfig):
        # Create rotary embedding
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
        )

        # Initialize base attention with Starcoder2-specific parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.use_bias,  # Starcoder2 uses bias
            o_bias=config.use_bias,
            sliding_window=getattr(config, 'sliding_window', None),
        )


class NeuronStarcoder2MLP(nn.Module):
    """
    Starcoder2 MLP implementation for NeuronX

    Starcoder2 uses a simple 2-layer MLP structure:
    - c_fc: hidden_size -> intermediate_size
    - activation: GELU
    - c_proj: intermediate_size -> hidden_size

    This is different from LLaMA's SwiGLU MLP structure.
    """

    def __init__(self, config: Starcoder2InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # First projection (input to intermediate)
        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # Activation function - Starcoder2 uses GELU
        # Note: hidden_act is "gelu_pytorch_tanh" in config
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act = lambda x: F.gelu(x, approximate="tanh")
        else:
            self.act = F.gelu

        # Second projection (intermediate to output)
        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        self.residual_dropout = getattr(config, 'residual_dropout', 0.0)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for MLP

        Args:
            hidden_states: Input tensor

        Returns:
            Tuple of (output_tensor, None) - None for compatibility with framework
        """
        # c_fc projection
        hidden_states = self.c_fc(hidden_states)

        # GELU activation
        hidden_states = self.act(hidden_states)

        # c_proj projection
        hidden_states = self.c_proj(hidden_states)

        # Apply dropout if in training mode
        if self.training and self.residual_dropout > 0.0:
            hidden_states = F.dropout(hidden_states, p=self.residual_dropout, training=self.training)

        # Return tuple for compatibility with framework expectations
        return hidden_states, None


class NeuronStarcoder2DecoderLayer(nn.Module):
    """
    Starcoder2 decoder layer implementation for NeuronX

    Each decoder layer consists of:
    1. Input LayerNorm
    2. Self-attention
    3. Residual connection
    4. Post-attention LayerNorm
    5. MLP
    6. Residual connection
    """

    def __init__(self, config: Starcoder2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention
        self.self_attn = NeuronStarcoder2Attention(config)

        # MLP
        self.mlp = NeuronStarcoder2MLP(config)

        # LayerNorm layers (Starcoder2 uses LayerNorm, not RMSNorm)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
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
        Forward pass for decoder layer

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            **kwargs: Additional arguments

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention returns 4 values: (output, kv_cache, cos, sin)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MLP returns (output, None)

        # Residual connection
        hidden_states = residual + hidden_states

        # Return 5 values as expected by framework:
        # (hidden_states, kv_cache, cos, sin, attn_weights)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronStarcoder2Model(NeuronBaseModel):
    """
    Starcoder2 base model for NeuronX

    This is the main transformer model without the language modeling head.
    Framework pattern:
    - No custom forward() method (base class handles it)
    - setup_attr_for_model() sets required attributes
    - init_model() initializes model components
    """

    def setup_attr_for_model(self, config: Starcoder2InferenceConfig):
        """
        Setup attributes required by the framework

        This method is called during initialization to set up model attributes
        needed by the distributed inference framework.
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", None)

    def init_model(self, config: Starcoder2InferenceConfig):
        """
        Initialize model components

        This method creates all the model components including embeddings,
        decoder layers, normalization, and language modeling head.
        """
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
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronStarcoder2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm (Starcoder2 uses LayerNorm, not RMSNorm)
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )

        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # Starcoder2 doesn't use bias in lm_head
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronStarcoder2ForCausalLM(NeuronBaseForCausalLM):
    """
    Starcoder2 Causal Language Model for NeuronX

    This class wraps the base Starcoder2 model and provides causal language
    modeling functionality compatible with HuggingFace's Starcoder2ForCausalLM.
    """

    _model_cls = NeuronStarcoder2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load HuggingFace Starcoder2 model

        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments for model loading

        Returns:
            Loaded HuggingFace model
        """
        from transformers import Starcoder2ForCausalLM
        return Starcoder2ForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format

        This function converts weight names and adds necessary metadata for
        distributed inference on Neuron hardware.

        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration

        Returns:
            Converted state dictionary for Neuron
        """
        neuron_config = config.neuron_config

        # Add rank utilities for vocabulary parallelism
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Add rank utilities for attention tensor parallelism
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank utilities for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied embeddings and lm_head weights

        Starcoder2 uses tied weights between embeddings and lm_head.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return Starcoder2InferenceConfig
