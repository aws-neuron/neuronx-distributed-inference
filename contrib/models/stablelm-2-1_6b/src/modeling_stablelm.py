# coding=utf-8
# Copyright 2024 Stability AI and the HuggingFace Inc. team. All rights reserved.
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
PyTorch StableLM model for NeuronX Distributed Inference.

This is a port of the HuggingFace StableLM model to the NeuronX Distributed Inference framework.
Based on the original implementation in transformers/models/stablelm/modeling_stablelm.py
"""

import os
import json
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
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


# =============================================================================
# HuggingFace-compatible Partial Rotary Embedding Implementation
# =============================================================================
# StableLM uses partial_rotary_factor=0.25 (only 25% of head_dim is rotated)
# The HF implementation has specific cos/sin cache format and indexing that
# differs from NxDI's standard implementation.


def rotate_half_hf(x):
    """
    Rotates half the hidden dims of the input - HuggingFace style.

    This matches the HuggingFace implementation:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors - HuggingFace style.

    This matches the HuggingFace implementation. The cos/sin tensors are already
    computed for the correct positions, so we just need to unsqueeze and apply.

    Args:
        q: Query tensor [batch, num_heads, seq_len, rotary_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, rotary_dim]
        cos: Cosine values [batch, seq_len, rotary_dim] - already position-specific
        sin: Sine values [batch, seq_len, rotary_dim] - already position-specific
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    # Unsqueeze for broadcasting to heads dimension
    # cos/sin shape: [batch, seq_len, rotary_dim] -> [batch, 1, seq_len, rotary_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotary embedding: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half_hf(q) * sin)
    k_embed = (k * cos) + (rotate_half_hf(k) * sin)
    return q_embed, k_embed


class StableLmPartialRotaryEmbedding(nn.Module):
    """
    StableLM Partial Rotary Embedding - HuggingFace compatible.

    This implements the exact cos/sin computation used by HuggingFace:
    - Computes position-specific cos/sin using position_ids
    - Uses torch.cat((freqs, freqs), dim=-1) for frequency duplication
    
    Key difference from NxDI's standard RotaryEmbedding:
    - Only rotates a fraction of head_dim (partial_rotary_factor)
    - The dim parameter is rotary_ndims = head_dim * partial_rotary_factor
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # This is the rotary dimension (partial_rotary_factor * head_dim)
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        # inv_freq shape: [dim // 2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Compute position-specific cos/sin values.

        Args:
            x: Input tensor (used to determine device and dtype)
            position_ids: Position indices [batch, seq_len]

        Returns:
            Tuple of (cos, sin) tensors of shape [batch, seq_len, dim]
        """
        # Ensure inv_freq is on the right device
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        
        # Expand inv_freq for batch matmul
        # inv_freq: [dim // 2] -> [batch, dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies: [batch, dim // 2, 1] @ [batch, 1, seq_len] -> [batch, dim // 2, seq_len]
        # Then transpose to [batch, seq_len, dim // 2]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        
        # HuggingFace duplicates the frequencies: [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Compute cos and sin
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def get_layernorm_cls():
    """
    Get the appropriate LayerNorm class.
    StableLM uses standard LayerNorm, not RMSNorm.
    """
    # For now, use PyTorch's LayerNorm
    # CustomRMSNorm only works on Neuron hardware, not for LayerNorm
    return nn.LayerNorm


class StableLmNeuronConfig(NeuronConfig):
    """NeuronConfig for StableLM model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the attention class
        self.attn_cls = NeuronStableLmAttention


class StableLmInferenceConfig(InferenceConfig):
    """
    Inference configuration for StableLM model.
    
    This configuration class handles StableLM-specific parameters and provides
    the interface between HuggingFace config format and NeuronX format.
    """

    def load_config(self):
        """
        Load configuration from HuggingFace config.json.
        
        This method is called during __init__ to load model-specific parameters.
        """
        # These attributes should already be set from kwargs passed to __init__
        # The framework will pass them from the HF config.json
        pass

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        
        # StableLM uses QKV bias by default
        self.qkv_bias = getattr(self, "use_qkv_bias", True)
        self.o_bias = False  # Output projection has no bias
        
        # Partial rotary factor - only apply RoPE to a fraction of head dimensions
        self.partial_rotary_factor = getattr(self, "partial_rotary_factor", 0.25)
        
        # Q-K layer normalization per head (optional feature)
        self.qk_layernorm = getattr(self, "qk_layernorm", False)
        
        # Parallel residual connections (optional feature)
        self.use_parallel_residual = getattr(self, "use_parallel_residual", False)
        
        # Dropout (usually 0 for inference)
        self.hidden_dropout = getattr(self, "hidden_dropout", 0.0)
        self.attention_dropout = getattr(self, "attention_dropout", 0.0)
        
        # Pad token id (StableLM doesn't use one typically)
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = None
        
        # Output flags for compatibility with base model
        self.output_attentions = getattr(self, "output_attentions", False)
        self.output_hidden_states = getattr(self, "output_hidden_states", False)
        self.return_dict = getattr(self, "return_dict", True)
        self.use_cache = getattr(self, "use_cache", True)

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
            "layer_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[StableLmNeuronConfig]:
        """Return the NeuronConfig class to use."""
        return StableLmNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Create config from a pretrained model directory.
        
        This loads the HuggingFace config.json and creates a StableLmInferenceConfig.
        
        Args:
            model_path: Path to the model directory containing config.json
            neuron_config: NeuronConfig instance (optional, can be None during inference loading)
            **kwargs: Additional config overrides
            
        Returns:
            StableLmInferenceConfig instance
        """
        # Load HuggingFace config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Create config dict from HF config
        config_dict = {
            "hidden_size": hf_config.get("hidden_size"),
            "num_attention_heads": hf_config.get("num_attention_heads"),
            "num_hidden_layers": hf_config.get("num_hidden_layers"),
            "num_key_value_heads": hf_config.get("num_key_value_heads"),
            "vocab_size": hf_config.get("vocab_size"),
            "max_position_embeddings": hf_config.get("max_position_embeddings"),
            "rope_theta": hf_config.get("rope_theta", 10000),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "intermediate_size": hf_config.get("intermediate_size"),
            "use_qkv_bias": hf_config.get("use_qkv_bias", True),
            "partial_rotary_factor": hf_config.get("partial_rotary_factor", 0.25),
            "qk_layernorm": hf_config.get("qk_layernorm", False),
            "use_parallel_residual": hf_config.get("use_parallel_residual", False),
            "hidden_dropout": hf_config.get("hidden_dropout", 0.0),
            "attention_dropout": hf_config.get("attention_dropout", 0.0),
            "bos_token_id": hf_config.get("bos_token_id"),
            "eos_token_id": hf_config.get("eos_token_id"),
            "pad_token_id": hf_config.get("pad_token_id"),
        }
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # If neuron_config is None, create a default one
        # This happens during inference when loading the compiled model
        if neuron_config is None:
            # Create a minimal neuron config - it will be loaded from saved config later
            neuron_config = cls.get_neuron_config_cls()()
        
        # Create and return config
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronStableLmAttention(NeuronAttentionBase):
    """
    StableLM attention module for NeuronX.
    
    Key features:
    - Partial rotary embeddings (only applies RoPE to a fraction of head dimensions)
    - Optional Q-K layer normalization per head
    - QKV bias support
    
    Based on: transformers/models/stablelm/modeling_stablelm.py:StableLmAttention
    """

    def __init__(self, config: StableLmInferenceConfig, layer_idx: Optional[int] = None):
        self.layer_idx = layer_idx
        self.partial_rotary_factor = config.partial_rotary_factor
        self.qk_layernorm = config.qk_layernorm

        # Calculate rotary dimensions - only a fraction of head_dim is rotated
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_ndims = int(head_dim * self.partial_rotary_factor)

        # Create HuggingFace-compatible rotary embedding for partial rotation
        # This uses the exact same cos/sin cache format as HuggingFace:
        # - torch.cat((freqs, freqs), dim=-1) for frequency duplication
        # - position_ids indexing for cos/sin lookup
        rotary_emb = StableLmPartialRotaryEmbedding(
            self.rotary_ndims,  # Only rotate partial dimensions
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Initialize base attention
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
        )

        # Store for use in forward pass
        self.head_dim = head_dim

        # Optional Q-K layer normalization per head
        # Note: This is a complex feature that may need custom implementation
        # For now, we'll skip it and add a warning if it's enabled
        if self.qk_layernorm:
            print("WARNING: Q-K layernorm per head is not fully supported yet. "
                  "This feature will be skipped in the implementation.")
            # TODO: Implement StableLmLayerNormPerHead equivalent if needed
            # self.q_layernorm = StableLmLayerNormPerHead(...)
            # self.k_layernorm = StableLmLayerNormPerHead(...)

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """
        Override to handle partial rotary embeddings with HuggingFace-compatible behavior.

        StableLM uses partial rotary where only a fraction (partial_rotary_factor) of
        head dimensions are rotated, while the rest pass through unchanged.

        Key differences from NxDI standard implementation:
        1. Uses HuggingFace-style rotate_half: torch.cat((-x2, x1), dim=-1)
        2. Uses HuggingFace-style cos/sin: torch.cat((freqs, freqs), dim=-1)
        3. Computes position-specific cos/sin using position_ids (not cache indexing)
        """
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            # Generate position-specific cos/sin using HuggingFace-compatible rotary embedding
            # This computes cos/sin dynamically from position_ids, not from a cache
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            # Split Q and K into rotary and pass-through portions
            Q_rot = Q[..., : self.rotary_ndims]
            Q_pass = Q[..., self.rotary_ndims :]

            K_rot = K[..., : self.rotary_ndims]
            K_pass = K[..., self.rotary_ndims :]

            # Apply rotary embeddings using HuggingFace-compatible function
            # cos_cache/sin_cache are already position-specific [batch, seq_len, rotary_dim]
            Q_rot, K_rot = apply_rotary_pos_emb_hf(Q_rot, K_rot, cos_cache, sin_cache)

            # Concatenate rotated and pass-through portions
            Q = torch.cat((Q_rot, Q_pass), dim=-1)
            K = torch.cat((K_rot, K_pass), dim=-1)

        elif use_polar_compatible_rope:
            # Polar compatible RoPE not used with partial rotary for StableLM
            raise NotImplementedError("Polar compatible RoPE not supported with partial rotary embeddings")

        return Q, K, cos_cache, sin_cache


class NeuronStableLmMLP(nn.Module):
    """
    StableLM MLP module for NeuronX.
    
    Uses standard GLU (Gated Linear Unit) architecture with:
    - gate_proj: Projects to intermediate size
    - up_proj: Projects to intermediate size
    - down_proj: Projects back to hidden size
    - Activation: SiLU (Swish)
    
    Based on: transformers/models/stablelm/modeling_stablelm.py:StableLmMLP
    """

    def __init__(self, config: StableLmInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate projection (for gating mechanism)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Up projection (for main pathway)
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Down projection (back to hidden size)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function (SiLU)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """
        Forward pass: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        
        This is the standard GLU/SwiGLU pattern used in modern LLMs.
        """
        # Apply gating: gate and up projections
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        
        # Element-wise multiplication
        intermediate_output = gate_output * up_output
        
        # Project back down to hidden size
        output = self.down_proj(intermediate_output)
        
        # Return tuple for compatibility with framework
        return output, None


class NeuronStableLmDecoderLayer(nn.Module):
    """
    StableLM decoder layer for NeuronX.
    
    Supports two residual connection patterns:
    1. Standard (use_parallel_residual=False):
       x = x + attn(ln1(x))
       x = x + mlp(ln2(x))
    
    2. Parallel (use_parallel_residual=True):
       x = x + attn(ln1(x)) + mlp(ln1(x))
    
    Based on: transformers/models/stablelm/modeling_stablelm.py:StableLmDecoderLayer
    """

    def __init__(self, config: StableLmInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_parallel_residual = config.use_parallel_residual
        
        # Self-attention
        self.self_attn = NeuronStableLmAttention(config, layer_idx=layer_idx)
        
        # MLP
        self.mlp = NeuronStableLmMLP(config)
        
        # Pre-attention layer normalization
        self.input_layernorm = get_layernorm_cls()(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Post-attention layer normalization (only for non-parallel residual)
        self.post_attention_layernorm = None
        if not self.use_parallel_residual:
            self.post_attention_layernorm = get_layernorm_cls()(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )
        
        # Dropout (usually 0 for inference)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        residual = hidden_states
        
        # Pre-attention normalization
        normalized_hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        if self.use_parallel_residual:
            # Parallel residual: x = x + attn(ln1(x)) + mlp(ln1(x))
            # Both attention and MLP use the same normalized input
            mlp_output = self.mlp(normalized_hidden_states)[0]
            mlp_output = self.dropout(mlp_output)
            
            # Combine both paths with residual
            hidden_states = residual + attn_output + mlp_output
        else:
            # Standard residual: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
            residual = residual + attn_output
            
            # Post-attention normalization and MLP
            hidden_states = self.post_attention_layernorm(residual)
            mlp_output = self.mlp(hidden_states)[0]
            mlp_output = self.dropout(mlp_output)
            
            hidden_states = residual + mlp_output

        # Return in the format expected by the framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


class NeuronStableLmModel(NeuronBaseModel):
    """
    StableLM model for NeuronX inference.
    
    Architecture:
    - Token embeddings
    - Stack of decoder layers
    - Final layer normalization
    - LM head for next token prediction
    
    Based on: transformers/models/stablelm/modeling_stablelm.py:StableLmModel
    """

    def setup_attr_for_model(self, config: StableLmInferenceConfig):
        """Setup attributes required by the framework."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: StableLmInferenceConfig):
        """Initialize model components."""
        self.padding_idx = None  # StableLM doesn't use padding_idx for embeddings
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronStableLmDecoderLayer(config, layer_idx=i) 
             for i in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = get_layernorm_cls()(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # LM head (output projection to vocabulary)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronStableLmForCausalLM(NeuronBaseForCausalLM):
    """
    StableLM for causal language modeling on NeuronX.
    
    This class provides the main interface for:
    - Loading HuggingFace checkpoints
    - Converting weights to NeuronX format
    - Compiling for Neuron hardware
    - Running inference
    
    Based on: transformers/models/stablelm/modeling_stablelm.py:StableLmForCausalLM
    """
    
    _model_cls = NeuronStableLmModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load the HuggingFace model for weight extraction.
        
        Args:
            model_path: Path to the HuggingFace model
            **kwargs: Additional arguments
            
        Returns:
            HuggingFace model instance
        """
        # Import here to avoid requiring transformers at module level
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except Exception as e:
            print(f"Warning: Could not load HuggingFace model: {e}")
            print("This is expected during compilation from scratch.")
            return None

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This function handles:
        - Adding rank utilities for tensor parallelism
        - Any necessary weight name mappings
        - Weight format conversions
        
        Args:
            state_dict: HuggingFace format state dictionary
            config: Model configuration
            
        Returns:
            NeuronX format state dictionary
        """
        neuron_config = config.neuron_config

        # Add rank utilities for vocab parallelism
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Add rank utilities for attention layers
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Handle fused QKV if enabled
        if neuron_config.fused_qkv:
            from neuronx_distributed_inference.models.model_base import convert_state_dict_to_fused_qkv
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        # Add rank utilities for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied weights.
        
        StableLM has tie_word_embeddings=False by default, so lm_head and
        embed_tokens are separate. This function handles cases where they
        might be tied.
        """
        # Check if weights should be tied (usually not for StableLM)
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class."""
        return StableLmInferenceConfig

    def get_compiler_args(self):
        """
        Get compiler arguments for NeuronX compilation.
        
        These arguments control optimization and compilation behavior.
        """
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        
        # Add flags for compute-communication overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        
        # Add HLO verification
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        
        return compiler_args
