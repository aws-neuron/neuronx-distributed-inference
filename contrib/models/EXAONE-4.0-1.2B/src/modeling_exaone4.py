# coding=utf-8
# Copyright 2025 The LG AI Research and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch EXAONE-4.0 model for NeuronX Distributed Inference."""

import logging
import math
from typing import List, Optional, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> torch RMSNorm implementation (CustomRMSNorm does not work on CPU)
    """
    if cpu_mode():
        # Simple RMSNorm implementation for CPU
        class RMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)
        
        return RMSNorm
    else:
        return CustomRMSNorm


class Exaone4InferenceConfig(InferenceConfig):
    """
    Configuration class for EXAONE-4.0 model inference on NeuronX.
    
    This configuration extends InferenceConfig to support EXAONE-4.0 specific parameters.
    
    Key EXAONE-4.0 specific features:
    - Tied word embeddings (tie_word_embeddings=True)
    - Post-attention and post-feedforward layer normalization
    - Llama3 RoPE scaling for long context (up to 65536)
    - Grouped Query Attention (GQA) with 32 attention heads and 8 KV heads
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # EXAONE-4.0 uses standard full attention for all layers

        # Add required config attributes that may be missing
        # These are needed by NeuronBaseForCausalLM but not always in HF config
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_path: Path to the HuggingFace model directory
            neuron_config: NeuronConfig instance for hardware configuration (optional for loading)
            **kwargs: Additional configuration overrides
            
        Returns:
            Exaone4InferenceConfig: Configuration object
        """
        # If neuron_config is not provided, create a minimal one for loading purposes
        if neuron_config is None:
            # Create a basic NeuronConfig for configuration loading
            # This will be replaced with the actual compiled config during model loading
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
        
        # Use the load_pretrained_config helper from hf_adapter
        config = cls(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(model_path),
            **kwargs
        )
        return config


class NeuronExaone4Attention(NeuronAttentionBase):
    """
    EXAONE-4.0 attention implementation for NeuronX.
    
    Based on HuggingFace transformers/models/exaone4/modeling_exaone4.py Exaone4Attention.
    
    Key features:
    - Grouped Query Attention (GQA) with configurable num_key_value_heads
    - RoPE (Rotary Position Embeddings) with llama3 scaling
    - No bias in projections (q_proj, k_proj, v_proj, o_proj)
    """
    
    def __init__(self, config: Exaone4InferenceConfig, layer_idx: int):
        """
        Initialize EXAONE-4.0 attention module.
        
        Args:
            config: Model configuration
            layer_idx: Layer index for this attention module
        """
        # EXAONE-4.0 uses RoPE with llama3 scaling
        rotary_emb = self.get_rope(config)
        
        # Create Q-K normalization layers (EXAONE-4.0 specific)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        q_layernorm = get_rmsnorm_cls()(head_dim, eps=config.rms_norm_eps)
        k_layernorm = get_rmsnorm_cls()(head_dim, eps=config.rms_norm_eps)
        
        # Initialize base attention with GQA support
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            # EXAONE-4.0 specific: Q-K normalization
            use_qk_norm=True,
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
            # Other EXAONE-4.0 parameters
            clip_qkv=None,  # EXAONE-4.0 doesn't use QKV clipping
            qkv_bias=False,  # EXAONE-4.0 doesn't use bias in QKV projections
            o_bias=False,  # EXAONE-4.0 doesn't use bias in output projection
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=getattr(config, "sliding_window", None),
        )
    
    def get_rope(self, config: Exaone4InferenceConfig):
        """
        Create RoPE embeddings for EXAONE-4.0.
        
        EXAONE-4.0 uses Llama3-style RoPE scaling for long context support.
        """
        rope_config = getattr(config, "rope_scaling", None)
        
        if rope_config and rope_config.get("rope_type") == "llama3":
            # Import Llama3 RoPE implementation
            from neuronx_distributed_inference.models.llama.modeling_llama import Llama3RotaryEmbedding
            
            rotary_emb = Llama3RotaryEmbedding(
                dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                factor=rope_config.get("factor", 8.0),
                low_freq_factor=rope_config.get("low_freq_factor", 1.0),
                high_freq_factor=rope_config.get("high_freq_factor", 4.0),
                original_max_position_embeddings=rope_config.get("original_max_position_embeddings", 8192),
            )
        else:
            # Standard RoPE
            rotary_emb = RotaryEmbedding(
                dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        
        return rotary_emb


class NeuronExaone4MLP(nn.Module):
    """
    EXAONE-4.0 MLP implementation for NeuronX.
    
    Based on HuggingFace transformers/models/exaone4/modeling_exaone4.py Exaone4MLP.
    
    Architecture:
    - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
    - up_proj: Linear(hidden_size, intermediate_size, bias=False)
    - down_proj: Linear(intermediate_size, hidden_size, bias=False)
    - Activation: SwiGLU (silu(gate_proj(x)) * up_proj(x))
    
    This follows the standard LLaMA-style MLP with SwiGLU activation.
    """
    
    def __init__(self, config: Exaone4InferenceConfig):
        """
        Initialize EXAONE-4.0 MLP module.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # EXAONE-4.0 uses silu activation (which is the same as swish)
        self.act_fn = ACT2FN[config.hidden_act]
        
        if parallel_state.model_parallel_is_initialized():
            # Column parallel for gate and up projections (split along intermediate_size)
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
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
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            # Row parallel for down projection (input is split, gather output)
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            # CPU mode - use standard linear layers
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        """
        Forward pass for EXAONE-4.0 MLP.
        
        Implements: down_proj(silu(gate_proj(x)) * up_proj(x))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # SwiGLU activation: silu(gate) * up
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate_output = gate_output * up_output
        
        # Apply down projection
        output = self.down_proj(intermediate_output)
        
        return output


class NeuronExaone4DecoderLayer(nn.Module):
    """
    EXAONE-4.0 decoder layer implementation for NeuronX.
    
    Based on HuggingFace transformers/models/exaone4/modeling_exaone4.py Exaone4DecoderLayer.
    
    Key architectural feature of EXAONE-4.0:
    - TWO layer normalizations per layer (unique to EXAONE):
      1. post_attention_layernorm: Applied after attention
      2. post_feedforward_layernorm: Applied after MLP
    
    This is different from standard LLaMA which uses:
    - input_layernorm: Before attention
    - post_attention_layernorm: Before MLP
    """
    
    def __init__(self, config: Exaone4InferenceConfig, layer_idx: int):
        """
        Initialize EXAONE-4.0 decoder layer.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention module
        self.self_attn = NeuronExaone4Attention(config, layer_idx)
        
        # MLP module
        self.mlp = NeuronExaone4MLP(config)
        
        # EXAONE-4.0 specific: TWO post-layer normalizations
        # Note: These are applied AFTER the residual connection
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.post_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Forward pass for EXAONE-4.0 decoder layer.

        Architecture (EXAONE-4.0 specific ordering - layer norm BEFORE residual):
        1. residual = hidden
        2. hidden = self_attn(hidden)
        3. hidden = post_attention_layernorm(hidden)
        4. hidden = residual + hidden  <- residual AFTER layernorm
        5. residual = hidden
        6. hidden = mlp(hidden)
        7. hidden = post_feedforward_layernorm(hidden)
        8. hidden = residual + hidden  <- residual AFTER layernorm

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value tensors

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache)
        """
        residual = hidden_states

        # Self attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Unpack attention output - NeuronAttentionBase returns NeuronAttentionBaseOutput
        # which can be unpacked like a tuple or accessed by attributes
        hidden_states = attn_output.hidden_states
        present_key_value = attn_output.present_key_value
        cos_cache = attn_output.cos_cache
        sin_cache = attn_output.sin_cache

        # Post-attention layer norm (EXAONE-4.0 specific: norm BEFORE residual add)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Residual connection (AFTER layernorm)
        hidden_states = residual + hidden_states

        # MLP with layer norm before residual
        residual = hidden_states
        mlp_output = self.mlp(hidden_states)

        # Post-feedforward layer norm (EXAONE-4.0 specific: norm BEFORE residual add)
        mlp_output = self.post_feedforward_layernorm(mlp_output)

        # Residual connection (AFTER layernorm)
        hidden_states = residual + mlp_output
        
        # Return format expected by framework: (hidden_states, kv_cache, cos_cache, sin_cache, residual)
        # EXAONE-4.0 doesn't use fused residual operations, so residual is None
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronExaone4Model(NeuronBaseModel):
    """
    EXAONE-4.0 base model for NeuronX inference.
    
    Based on HuggingFace transformers/models/exaone4/modeling_exaone4.py Exaone4Model.
    
    Architecture:
    - Token embeddings (with padding)
    - 30 decoder layers (for 1.2B model)
    - Final RMSNorm
    - LM head for generation
    
    Key feature: Tied embeddings (embed_tokens and lm_head share weights)
    """
    
    def setup_attr_for_model(self, config: Exaone4InferenceConfig):
        """
        Setup attributes required by the NeuronX framework.
        
        Args:
            config: Model configuration
        """
        # Required by framework for inference optimization
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
        # EXAONE-4.0 doesn't use sliding window attention
        self.sliding_window = None
        
    def init_model(self, config: Exaone4InferenceConfig):
        """
        Initialize model components.
        
        Args:
            config: Model configuration
        """
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        if parallel_state.model_parallel_is_initialized():
            # Token embeddings
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
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )
            
            # LM head (for generation)
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
            # CPU mode
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronExaone4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class NeuronExaone4ForCausalLM(NeuronBaseForCausalLM):
    """
    EXAONE-4.0 Causal Language Model for NeuronX inference.
    
    This class extends NeuronBaseForCausalLM and provides EXAONE-4.0 specific
    functionality including weight conversion and tied embeddings support.
    
    Key features:
    - Tied word embeddings (embed_tokens and lm_head share weights)
    - HuggingFace checkpoint conversion to Neuron format
    - Tensor parallelism support
    """
    
    _model_cls = NeuronExaone4Model
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace EXAONE-4.0 checkpoint to Neuron format.
        
        This function handles:
        1. Removing "model." prefix from keys
        2. Adding rank utilities for tensor parallelism
        3. Preserving all weight mappings (EXAONE-4.0 uses same names as HF after prefix removal)
        
        Args:
            state_dict: HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for Neuron
        """
        neuron_config = config.neuron_config
        
        # Remove "model." prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        state_dict = new_state_dict
        
        # Rename keys from HF format to Neuron format
        # The NeuronAttentionBase uses GroupQueryAttention_QKV and GroupQueryAttention_O
        # which have different weight key structures:
        # HF: layers.{i}.self_attn.{q,k,v}_proj.weight -> Neuron: layers.{i}.self_attn.qkv_proj.{q,k,v}_proj.weight
        # HF: layers.{i}.self_attn.o_proj.weight -> Neuron: layers.{i}.self_attn.o_proj.o_proj.weight
        # HF: layers.{i}.self_attn.{q,k}_norm.weight -> Neuron: layers.{i}.self_attn.{q,k}_layernorm.weight
        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle Q-K norm rename
            if ".self_attn.q_norm." in key:
                new_key = key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
                new_state_dict[new_key] = value
            elif ".self_attn.k_norm." in key:
                new_key = key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")
                new_state_dict[new_key] = value
            # Handle QKV projection rename - add qkv_proj. prefix
            elif ".self_attn.q_proj." in key:
                new_key = key.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.q_proj.")
                new_state_dict[new_key] = value
            elif ".self_attn.k_proj." in key:
                new_key = key.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.k_proj.")
                new_state_dict[new_key] = value
            elif ".self_attn.v_proj." in key:
                new_key = key.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.v_proj.")
                new_state_dict[new_key] = value
            # Handle O projection rename - add extra o_proj. prefix
            elif ".self_attn.o_proj." in key:
                new_key = key.replace(".self_attn.o_proj.", ".self_attn.o_proj.o_proj.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict
        
        # Add rank utilities for attention layers (required for tensor parallelism)
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utility for vocabulary parallel mode
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        # Add rank utility for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        # Weight mapping summary after transformations:
        # HF -> Neuron:
        # - model.embed_tokens.weight -> embed_tokens.weight
        # - model.layers.{i}.self_attn.q_proj.weight -> layers.{i}.self_attn.qkv_proj.q_proj.weight
        # - model.layers.{i}.self_attn.k_proj.weight -> layers.{i}.self_attn.qkv_proj.k_proj.weight
        # - model.layers.{i}.self_attn.v_proj.weight -> layers.{i}.self_attn.qkv_proj.v_proj.weight
        # - model.layers.{i}.self_attn.o_proj.weight -> layers.{i}.self_attn.o_proj.o_proj.weight
        # - model.layers.{i}.self_attn.{q,k}_norm.weight -> layers.{i}.self_attn.{q,k}_layernorm.weight
        # - model.layers.{i}.mlp.{gate,up,down}_proj.weight -> layers.{i}.mlp.{gate,up,down}_proj.weight
        # - model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.post_attention_layernorm.weight
        # - model.layers.{i}.post_feedforward_layernorm.weight -> layers.{i}.post_feedforward_layernorm.weight
        # - model.norm.weight -> norm.weight
        # - lm_head.weight -> lm_head.weight (tied to embed_tokens.weight)
        
        return state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied word embeddings for EXAONE-4.0.
        
        EXAONE-4.0 sets tie_word_embeddings=True, meaning the lm_head
        shares weights with embed_tokens.
        
        Args:
            state_dict: Model state dictionary
        """
        # Tie lm_head to embed_tokens
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for EXAONE-4.0."""
        return Exaone4InferenceConfig
