# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch OPT model for NXD inference."""

"""
Key architectural features of OPT:
- Decoder-only causal language model (like GPT)
- Learned positional embeddings (not RoPE)
- Standard Multi-Head Attention (not GQA)
- LayerNorm (not RMSNorm)
- ReLU activation in MLP (not SwiGLU)
- Pre-norm architecture (LayerNorm before attention and MLP)
- Optional word embedding projection dimension different from hidden size
"""

import os
import json
import math
from typing import Optional, Tuple, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class OPTInferenceConfig(InferenceConfig):
    """
    Configuration class for OPT model inference on NeuronX.
    
    This configuration extends InferenceConfig to support OPT-specific parameters
    and maps HuggingFace OPT configuration to the NeuronX framework.
    
    Key OPT-specific parameters:
    - ffn_dim: Intermediate size in MLP (called ffn_dim in OPT)
    - activation_function: Activation function (typically "relu" for OPT)
    - do_layer_norm_before: Pre-norm architecture flag
    - word_embed_proj_dim: Embedding dimension (can differ from hidden_size)
    - enable_bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        neuron_config: Optional[NeuronConfig] = None,
        vocab_size: int = 50272,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        ffn_dim: int = 3072,
        max_position_embeddings: int = 2048,
        activation_function: str = "relu",
        do_layer_norm_before: bool = True,
        word_embed_proj_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        enable_bias: bool = True,
        layer_norm_elementwise_affine: bool = True,
        layerdrop: float = 0.0,
        init_std: float = 0.02,
        pad_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 2,
        _remove_final_layer_norm: bool = False,
        **kwargs
    ):
        # OPT uses standard MHA (not GQA), so num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        
        # Set word_embed_proj_dim to hidden_size if not specified
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        
        # OPT-specific parameters
        self.ffn_dim = ffn_dim  # intermediate_size in other models
        self.activation_function = activation_function
        self.do_layer_norm_before = do_layer_norm_before
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        self.layerdrop = layerdrop
        self.init_std = init_std
        self._remove_final_layer_norm = _remove_final_layer_norm
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        # Additional attributes expected by the framework
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_cache = True
        
        # Call parent constructor
        super().__init__(
            neuron_config=neuron_config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
    def add_derived_config(self):
        """Add derived configuration parameters required by the framework."""
        self.num_cores_per_group = 1
        
        # OPT uses learned positional embeddings, not RoPE
        self.position_embedding_type = "learned"
        
        # Set intermediate_size to ffn_dim for compatibility
        self.intermediate_size = self.ffn_dim
        
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads", 
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "ffn_dim",
            "word_embed_proj_dim",
        ]
    
    def validate_config(self):
        """Override to handle None neuron_config during inference loading."""
        if self.neuron_config is None:
            # Skip validation when neuron_config is None (happens during inference loading)
            return
        # Call parent validation
        super().validate_config()
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "OPTInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            OPTInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs (may be None for inference loading)
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read HuggingFace config.json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace OPT config to our config
        config_dict = {
            "vocab_size": hf_config.get("vocab_size", 50272),
            "hidden_size": hf_config.get("hidden_size", 768),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 12),
            "num_attention_heads": hf_config.get("num_attention_heads", 12),
            "ffn_dim": hf_config.get("ffn_dim", 3072),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 2048),
            "activation_function": hf_config.get("activation_function", "relu"),
            "do_layer_norm_before": hf_config.get("do_layer_norm_before", True),
            "word_embed_proj_dim": hf_config.get("word_embed_proj_dim"),
            "dropout": hf_config.get("dropout", 0.1),
            "attention_dropout": hf_config.get("attention_dropout", 0.0),
            "enable_bias": hf_config.get("enable_bias", True),
            "layer_norm_elementwise_affine": hf_config.get("layer_norm_elementwise_affine", True),
            "layerdrop": hf_config.get("layerdrop", 0.0),
            "init_std": hf_config.get("init_std", 0.02),
            "pad_token_id": hf_config.get("pad_token_id", 1),
            "bos_token_id": hf_config.get("bos_token_id", 2),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            "_remove_final_layer_norm": hf_config.get("_remove_final_layer_norm", False),
        }
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        # Create and return config (neuron_config may be None for inference)
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class OPTLearnedPositionalEmbedding(nn.Module):
    """
    OPT-specific learned positional embeddings.
    
    OPT uses learned positional embeddings with an offset of 2 to accommodate
    padding. This is different from RoPE or absolute positional embeddings
    used in other models.
    
    Reference: OPTLearnedPositionalEmbedding in modeling_opt.py
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 1):
        super().__init__()
        # OPT offsets embedding ids by 2
        self.offset = 2
        self.padding_idx = padding_idx
        
        # Use ParallelEmbedding for tensor parallelism support
        self.embedding = ParallelEmbedding(
            num_embeddings + self.offset,
            embedding_dim,
            padding_idx=None,  # We handle padding manually
        )
    
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for learned positional embeddings.
        
        Args:
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values_length: Length of past key values for generation
            position_ids: Optional explicit position ids
            
        Returns:
            Positional embeddings [batch_size, seq_len, embedding_dim]
        """
        if position_ids is None:
            # Calculate position_ids from attention_mask
            # Position ids are cumulative sum of attention mask
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            
            # Cut positions if past_key_values_length > 0
            if past_key_values_length > 0:
                position_ids = position_ids[:, past_key_values_length:]
        
        # Add offset to position_ids
        position_ids = position_ids + self.offset
        
        # Get embeddings
        return self.embedding(position_ids)


class NeuronOPTAttention(NeuronAttentionBase):
    """
    OPT attention implementation for NeuronX.
    
    OPT uses standard Multi-Head Attention (MHA), not Grouped Query Attention (GQA).
    Key differences from models like LLaMA:
    - No rotary position embeddings (uses learned positional embeddings)
    - num_key_value_heads = num_attention_heads (standard MHA)
    - Has bias terms in projections (configurable)
    - Scaling applied to query before attention computation
    
    Reference: OPTAttention in modeling_opt.py
    """
    
    def __init__(self, config: OPTInferenceConfig, layer_idx: Optional[int] = None):
        self.config = config
        self.layer_idx = layer_idx
        
        # Calculate head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        
        # OPT does not use rotary embeddings
        rotary_emb = None
        
        # Initialize base attention
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Standard MHA
            head_dim=head_dim,
            rotary_emb=rotary_emb,  # No RoPE for OPT
            qkv_bias=config.enable_bias,  # OPT supports bias in QKV projections
            o_bias=config.enable_bias,  # OPT supports bias in output projection
        )


class NeuronOPTMLP(nn.Module):
    """
    OPT MLP (Feed-Forward Network) implementation for NeuronX.
    
    OPT uses a standard 2-layer feedforward network with ReLU activation,
    unlike LLaMA which uses SwiGLU. The structure is:
    - fc1: Linear(hidden_size, ffn_dim) with bias
    - activation: ReLU
    - fc2: Linear(ffn_dim, hidden_size) with bias
    
    Reference: OPTDecoderLayer.fc1, fc2 in modeling_opt.py
    """
    
    def __init__(self, config: OPTInferenceConfig):
        super().__init__()
        self.config = config
        
        # Input projection (hidden_size -> ffn_dim)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_dim,
            bias=config.enable_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function (ReLU for OPT)
        if config.activation_function == "relu":
            self.act_fn = nn.ReLU()
        elif config.activation_function == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_function}")
        
        # Output projection (ffn_dim -> hidden_size)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            config.hidden_size,
            bias=config.enable_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # fc1: hidden_size -> ffn_dim
        hidden_states = self.fc1(hidden_states)
        
        # activation
        hidden_states = self.act_fn(hidden_states)
        
        # fc2: ffn_dim -> hidden_size
        hidden_states = self.fc2(hidden_states)[0]  # RowParallelLinear returns tuple
        
        return hidden_states


class NeuronOPTDecoderLayer(nn.Module):
    """
    OPT decoder layer implementation for NeuronX.
    
    OPT uses a pre-norm architecture where LayerNorm is applied before
    self-attention and before the MLP. This is controlled by the
    do_layer_norm_before flag (True for most OPT models).
    
    Layer structure (pre-norm):
    1. LayerNorm -> Self-Attention -> Dropout -> Residual
    2. LayerNorm -> MLP -> Dropout -> Residual
    
    Reference: OPTDecoderLayer in modeling_opt.py
    """
    
    def __init__(self, config: OPTInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = NeuronOPTAttention(config, layer_idx=layer_idx)
        
        # Self-attention LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=1e-5,  # Default PyTorch LayerNorm eps
            elementwise_affine=config.layer_norm_elementwise_affine,
        )
        
        # MLP
        self.mlp = NeuronOPTMLP(config)
        
        # MLP LayerNorm
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=1e-5,
            elementwise_affine=config.layer_norm_elementwise_affine,
        )
        
        # Dropout
        self.dropout = config.dropout
        self.do_layer_norm_before = config.do_layer_norm_before
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of decoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs (not used for OPT, uses learned embeddings)
            past_key_value: Cached key-value states
            
        Returns:
            Tuple of (hidden_states, past_key_value)
        """
        residual = hidden_states
        
        # Self-Attention with pre-norm
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self-attention
        # NeuronAttentionBase returns NeuronAttentionBaseOutput with multiple fields
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states
        present_key_value = attn_output.present_key_value
        
        # Dropout and residual
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        # Post-norm (for models with do_layer_norm_before=False)
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # MLP
        residual = hidden_states
        
        # MLP with pre-norm
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # MLP forward
        hidden_states = self.mlp(hidden_states)
        
        # Dropout and residual
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        # Post-norm (for models with do_layer_norm_before=False)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # Return format must match base class expectations
        # (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        # OPT doesn't use RoPE, so cos_cache and sin_cache are None
        # OPT doesn't expose residual from attention, so residual is None
        return hidden_states, present_key_value, None, None, None


class NeuronOPTModel(NeuronBaseModel):
    """
    OPT model implementation for NeuronX.
    
    The OPT model consists of:
    - Token embeddings (possibly with projection from word_embed_proj_dim to hidden_size)
    - Learned positional embeddings
    - Stack of decoder layers
    - Final LayerNorm (if do_layer_norm_before and not _remove_final_layer_norm)
    - Optional projection from hidden_size to word_embed_proj_dim
    
    Reference: OPTDecoder and OPTModel in modeling_opt.py
    """
    
    def setup_attr_for_model(self, config: OPTInferenceConfig):
        """Setup attributes for model initialization."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
    def init_model(self, config: OPTInferenceConfig):
        """Initialize the OPT model."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings - no wrapper, let base class call it
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            padding_idx=config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Positional embeddings (learned, OPT-specific with offset)
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        
        # Optional projection from word_embed_proj_dim to hidden_size
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
            )
        else:
            self.project_in = None
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NeuronOPTDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final LayerNorm - base class expects it to be called 'norm'
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.norm = nn.LayerNorm(
                config.hidden_size,
                eps=1e-5,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.norm = None
        
        # Optional projection from hidden_size to word_embed_proj_dim
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
            )
        else:
            self.project_out = None
        
        # Language modeling head (at model level, not ForCausalLM level)
        # Note: In HuggingFace OPT, lm_head is tied to embed_tokens
        self.lm_head = ColumnParallelLinear(
            config.word_embed_proj_dim,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def process_sequence_parallel_hidden_states(
        self,
        inputs_embeds: torch.FloatTensor,
        seq_length: int,
        active_block_table: torch.IntTensor = None,
    ) -> torch.Tensor:
        """
        Override to add OPT's learned positional embeddings before sequence parallel processing.
        
        OPT uses learned positional embeddings that need to be added to token embeddings,
        unlike RoPE which is applied during attention.
        """
        # First, add positional embeddings if we haven't already
        # Create a simple attention mask for positional embedding calculation
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)
        
        # Get positional embeddings
        # Note: past_key_values_length would be computed in the base class forward
        # For now, we assume 0 for simplicity
        pos_embeds = self.embed_positions(
            attention_mask,
            past_key_values_length=0,
            position_ids=None,
        )
        
        # Project token embeddings if needed (OPT-specific)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        
        # Add positional embeddings
        inputs_embeds = inputs_embeds + pos_embeds
        
        # Now call parent's sequence parallel processing
        return super().process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, active_block_table
        )


class NeuronOPTForCausalLM(NeuronBaseForCausalLM):
    """
    OPT model for causal language modeling on NeuronX.
    
    This is the top-level model class that includes the OPT model and
    the language modeling head.
    
    Reference: OPTForCausalLM in modeling_opt.py
    """
    
    _model_cls = NeuronOPTModel
    
    @classmethod  
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return OPTInferenceConfig
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: OPTInferenceConfig) -> dict:
        """
        Convert HuggingFace OPT checkpoint to NeuronX format.
        
        This method maps the weight names from the HuggingFace format to the
        NeuronX format and handles tensor parallelism setup.
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        print(f"\n🔍 DEBUG: convert_hf_to_neuron_state_dict called!")
        print(f"🔍 DEBUG: state_dict has {len(state_dict)} keys")
        print(f"🔍 DEBUG: First 5 keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"   {i+1}. {key}")
        
        neuron_state_dict = {}
        
        # Token embeddings
        # HF: model.decoder.embed_tokens.weight -> Neuron: embed_tokens.weight
        if "model.decoder.embed_tokens.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["model.decoder.embed_tokens.weight"].clone()
        
        # Positional embeddings
        # HF: model.decoder.embed_positions.weight -> Neuron: embed_positions.embedding.weight
        if "model.decoder.embed_positions.weight" in state_dict:
            neuron_state_dict["embed_positions.embedding.weight"] = state_dict["model.decoder.embed_positions.weight"].clone()
        
        # Optional projection layers
        if "model.decoder.project_in.weight" in state_dict:
            neuron_state_dict["project_in.weight"] = state_dict["model.decoder.project_in.weight"].clone()
        
        if "model.decoder.project_out.weight" in state_dict:
            neuron_state_dict["project_out.weight"] = state_dict["model.decoder.project_out.weight"].clone()
        
        # Final LayerNorm (now called 'norm')
        if "model.decoder.final_layer_norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["model.decoder.final_layer_norm.weight"].clone()
        if "model.decoder.final_layer_norm.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["model.decoder.final_layer_norm.bias"].clone()
        
        # Decoder layers
        for i in range(config.num_hidden_layers):
            layer_prefix_hf = f"model.decoder.layers.{i}"
            layer_prefix_neuron = f"layers.{i}"
            
            # Self-attention LayerNorm
            if f"{layer_prefix_hf}.self_attn_layer_norm.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn_layer_norm.weight"] = state_dict[f"{layer_prefix_hf}.self_attn_layer_norm.weight"].clone()
            if f"{layer_prefix_hf}.self_attn_layer_norm.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn_layer_norm.bias"] = state_dict[f"{layer_prefix_hf}.self_attn_layer_norm.bias"].clone()
            
            # MLP LayerNorm
            if f"{layer_prefix_hf}.final_layer_norm.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.final_layer_norm.weight"] = state_dict[f"{layer_prefix_hf}.final_layer_norm.weight"].clone()
            if f"{layer_prefix_hf}.final_layer_norm.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.final_layer_norm.bias"] = state_dict[f"{layer_prefix_hf}.final_layer_norm.bias"].clone()
            
            # Attention Q, K, V projections
            if f"{layer_prefix_hf}.self_attn.q_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.weight"] = state_dict[f"{layer_prefix_hf}.self_attn.q_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.k_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.weight"] = state_dict[f"{layer_prefix_hf}.self_attn.k_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.v_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.weight"] = state_dict[f"{layer_prefix_hf}.self_attn.v_proj.weight"].clone()
            
            # Attention bias terms (if enabled)
            if config.enable_bias:
                if f"{layer_prefix_hf}.self_attn.q_proj.bias" in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.bias"] = state_dict[f"{layer_prefix_hf}.self_attn.q_proj.bias"].clone()
                if f"{layer_prefix_hf}.self_attn.k_proj.bias" in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.bias"] = state_dict[f"{layer_prefix_hf}.self_attn.k_proj.bias"].clone()
                if f"{layer_prefix_hf}.self_attn.v_proj.bias" in state_dict:
                    neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.bias"] = state_dict[f"{layer_prefix_hf}.self_attn.v_proj.bias"].clone()
            
            # Attention output projection
            if f"{layer_prefix_hf}.self_attn.out_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.weight"] = state_dict[f"{layer_prefix_hf}.self_attn.out_proj.weight"].clone()
            if config.enable_bias and f"{layer_prefix_hf}.self_attn.out_proj.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.bias"] = state_dict[f"{layer_prefix_hf}.self_attn.out_proj.bias"].clone()
            
            # MLP fc1
            if f"{layer_prefix_hf}.fc1.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc1.weight"] = state_dict[f"{layer_prefix_hf}.fc1.weight"].clone()
            if config.enable_bias and f"{layer_prefix_hf}.fc1.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc1.bias"] = state_dict[f"{layer_prefix_hf}.fc1.bias"].clone()
            
            # MLP fc2
            if f"{layer_prefix_hf}.fc2.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc2.weight"] = state_dict[f"{layer_prefix_hf}.fc2.weight"].clone()
            if config.enable_bias and f"{layer_prefix_hf}.fc2.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc2.bias"] = state_dict[f"{layer_prefix_hf}.fc2.bias"].clone()
        
        # LM head
        # In HuggingFace OPT, lm_head.weight is tied to embed_tokens.weight
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
        elif "model.decoder.embed_tokens.weight" in state_dict:
            # If lm_head is not present, use embed_tokens (weight tying)
            neuron_state_dict["lm_head.weight"] = state_dict["model.decoder.embed_tokens.weight"].clone()
        
        # Add rank tensors for tensor parallelism
        tp_degree = config.neuron_config.tp_degree
        rank_tensor = torch.arange(0, tp_degree, dtype=torch.int32)
        
        # Add rank for each attention layer (no "model." prefix)
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = rank_tensor.clone()
        
        # Add rank for base model (no "model." prefix)
        neuron_state_dict["rank_util.rank"] = rank_tensor.clone()
        
        return neuron_state_dict


# Export all classes
__all__ = [
    "OPTInferenceConfig",
    "OPTLearnedPositionalEmbedding",
    "NeuronOPTAttention",
    "NeuronOPTMLP",
    "NeuronOPTDecoderLayer",
    "NeuronOPTModel",
    "NeuronOPTForCausalLM",
]
