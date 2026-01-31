# coding=utf-8
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science. All rights reserved.
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
"""NeuronX Distributed BioGPT model for inference."""

import json
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)


class BioGptNeuronConfig(NeuronConfig):
    """
    Neuron configuration for BioGPT model
    """
    pass


class BioGptInferenceConfig(InferenceConfig):
    """
    Configuration class for BioGPT inference on Neuron
    
    Maps from HuggingFace BioGPT configuration to NeuronX Distributed Inference format
    """

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        # BioGPT uses standard LayerNorm, not RMSNorm
        self.use_rms_norm = False
        # BioGPT has bias in all linear layers
        self.qkv_bias = True
        self.o_bias = True
        self.mlp_bias = True
        # BioGPT uses learned absolute positional embeddings
        self.use_absolute_position_embeddings = True
        # BioGPT scales embeddings by sqrt(hidden_size)
        self.scale_embedding = getattr(self, 'scale_embedding', True)
        # Default output settings
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "hidden_act",
            "layer_norm_eps",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return BioGptNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "BioGptInferenceConfig":
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory (HuggingFace format)
            **kwargs: Additional arguments to override configuration
            
        Returns:
            BioGptInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        # If not provided, create a minimal default (for inference loading)
        neuron_config = kwargs.pop("neuron_config", None)
        if neuron_config is None:
            # Create a minimal default neuron config for inference
            neuron_config = BioGptNeuronConfig(
                tp_degree=1,
                batch_size=1,
                max_length=1024,  # Total sequence length (context + generation)
                max_context_length=512,
                max_new_tokens=512,
            )
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Map HuggingFace BioGPT config to our format
        # BioGPT config parameters map directly, no renaming needed
        inference_config = {
            "hidden_size": config_dict.get("hidden_size", 1024),
            "num_attention_heads": config_dict.get("num_attention_heads", 16),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 24),
            "vocab_size": config_dict.get("vocab_size", 42384),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 1024),
            "intermediate_size": config_dict.get("intermediate_size", 4096),
            "hidden_act": config_dict.get("hidden_act", "gelu"),
            "layer_norm_eps": config_dict.get("layer_norm_eps", 1e-12),
            "pad_token_id": config_dict.get("pad_token_id", 1),
            "bos_token_id": config_dict.get("bos_token_id", 0),
            "eos_token_id": config_dict.get("eos_token_id", 2),
            "scale_embedding": config_dict.get("scale_embedding", True),
            "hidden_dropout_prob": config_dict.get("hidden_dropout_prob", 0.1),
            "attention_probs_dropout_prob": config_dict.get("attention_probs_dropout_prob", 0.1),
        }
        
        # BioGPT does not have separate num_key_value_heads (standard MHA)
        inference_config["num_key_value_heads"] = inference_config["num_attention_heads"]
        
        # Override with remaining kwargs
        inference_config.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **inference_config)
        return config


class NeuronBioGptAttention(NeuronAttentionBase):
    """
    BioGPT attention implementation for NeuronX
    
    Key features:
    - Standard multi-head attention (no GQA)
    - No rotary position embeddings (uses learned absolute positions)
    - Has bias terms in all projections
    - Scaling by head_dim ** -0.5
    
    Class: BioGptAttention
    """

    def __init__(self, config: BioGptInferenceConfig):
        # BioGPT uses standard attention without rotary embeddings
        # Positional information comes from learned absolute embeddings
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Standard MHA
            head_dim=config.hidden_size // config.num_attention_heads,
            qkv_bias=True,  # BioGPT has bias in QKV projections
            o_bias=True,    # BioGPT has bias in output projection
            rotary_emb=None,  # No RoPE, uses absolute positional embeddings
        )


class NeuronBioGptMLP(nn.Module):
    """
    BioGPT MLP implementation for NeuronX
    
    Key features:
    - Standard feed-forward network (not SwiGLU)
    - fc1: hidden_size -> intermediate_size with bias
    - activation: GELU
    - fc2: intermediate_size -> hidden_size with bias
    
    Class: BioGptDecoderLayer (fc1, fc2 components)
    """

    def __init__(self, config: BioGptInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # fc1: expand from hidden_size to intermediate_size
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,  # BioGPT has bias
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # fc2: project back from intermediate_size to hidden_size
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,  # BioGPT has bias
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Activation function (GELU for BioGPT)
        if config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")

    def forward(self, hidden_states):
        """
        Forward pass for BioGPT MLP
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Tuple of (output_tensor, None) for compatibility with framework
        """
        # Expand to intermediate size
        hidden_states = self.fc1(hidden_states)
        
        # Apply activation
        hidden_states = self.act_fn(hidden_states)
        
        # Project back to hidden size
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states, None  # Return None as second output for compatibility


class NeuronBioGptDecoderLayer(nn.Module):
    """
    BioGPT decoder layer implementation for NeuronX
    
    Architecture (pre-normalization):
    1. LayerNorm -> Self-Attention -> Dropout -> Residual
    2. LayerNorm -> MLP -> Dropout -> Residual
    
    Class: BioGptDecoderLayer
    """

    def __init__(self, config: BioGptInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = NeuronBioGptAttention(config)
        
        # MLP
        self.mlp = NeuronBioGptMLP(config)
        
        # Layer norms (BioGPT uses LayerNorm, not RMSNorm)
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Dropout is handled in training mode; for inference we don't need it
        # but we keep the config for compatibility
        self.dropout = config.hidden_dropout_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for BioGPT decoder layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs (not used for attention, BioGPT uses absolute embeddings)
            past_key_value: Cached key/value tensors for generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Pre-norm architecture: normalize before attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self-attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Residual connection (dropout disabled for inference)
        hidden_states = residual + hidden_states
        
        # Pre-norm architecture: normalize before MLP
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        # MLP
        hidden_states, _ = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Return format compatible with NeuronX framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


class ScaledEmbedding(nn.Module):
    """
    Wrapper around ParallelEmbedding that applies scaling
    """
    def __init__(self, embedding, scale):
        super().__init__()
        self.embedding = embedding
        self.scale = scale
    
    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        return embeds * self.scale


class BioGptPositionalEmbedding(nn.Module):
    """
    BioGPT positional embedding with offset
    """
    def __init__(self, embedding, offset=2):
        super().__init__()
        self.embedding = embedding
        self.offset = offset
    
    def forward(self, position_ids):
        # Add offset to position_ids
        return self.embedding(position_ids + self.offset)


class NeuronBioGptModel(NeuronBaseModel):
    """
    BioGPT base model for NeuronX inference
    
    Class: BioGptModel
    """

    def setup_attr_for_model(self, config: BioGptInferenceConfig):
        """Setup attributes for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # Standard MHA for BioGPT
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: BioGptInferenceConfig):
        """Initialize the BioGPT model"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.scale_embedding = config.scale_embedding
        
        # Embedding scaling factor and position offset
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        position_offset = 2  # BioGPT uses an offset of 2 for positional embeddings
        
        # Token embeddings (BioGPT uses scaled embeddings)
        base_embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        # Wrap with scaling
        self.embed_tokens = ScaledEmbedding(base_embed_tokens, embed_scale)
        
        # Learned positional embeddings (absolute positions, not RoPE)
        # BioGPT uses an offset of 2 for positional embeddings, so actual size is max_position_embeddings + 2
        base_embed_positions = ParallelEmbedding(
            config.max_position_embeddings + position_offset,
            config.hidden_size,
            padding_idx=None,  # No padding for position embeddings
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        # Wrap with offset
        self.embed_positions = BioGptPositionalEmbedding(base_embed_positions, position_offset)
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronBioGptDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer norm (named 'norm' for base class compatibility)
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # BioGPT lm_head has no bias
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        """
        Override base model's get_model_output to add absolute positional embeddings.
        BioGPT uses learned absolute positional embeddings, unlike models with RoPE.
        """
        # Get basic past_key_values_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]
        
        # Get embeddings (scaling is handled by ScaledEmbedding wrapper)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Generate position_ids for positional embeddings
        if position_ids is None:
            device = inputs_embeds.device
            # Simple sequential position_ids
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)
        
        # Add positional embeddings (offset is handled by BioGptPositionalEmbedding wrapper)
        position_embeddings = self.embed_positions(position_ids)
        inputs_embeds = inputs_embeds + position_embeddings
        
        # Call parent's get_model_output with modified inputs_embeds
        # We pass inputs_embeds so the parent won't call embed_tokens again
        return super().get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,  # Pass our modified embeddings
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_ids,
            update_cache=update_cache,
            is_for_context_encoding=is_for_context_encoding,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            local_attn_mask=local_attn_mask,
            windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
            **kwargs,
        )


class NeuronBioGptForCausalLM(NeuronBaseForCausalLM):
    """
    BioGPT for causal language modeling on NeuronX
    
    This class can be used as BioGptForCausalLM
    
    Class: BioGptForCausalLM
    """

    _model_cls = NeuronBioGptModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """
        Convert HuggingFace BioGPT weights to NeuronX format
        
        Args:
            state_dict: HuggingFace state dictionary
            config: BioGptInferenceConfig
            
        Returns:
            Dictionary with NeuronX-compatible weights
            
        Reference: HuggingFace BioGPT weight names in
        """
        neuron_state_dict = {}
        
        # Token embeddings
        # HF: biogpt.embed_tokens.weight -> Neuron: embed_tokens.embedding.weight (wrapped)
        if "biogpt.embed_tokens.weight" in state_dict:
            neuron_state_dict["embed_tokens.embedding.weight"] = state_dict["biogpt.embed_tokens.weight"].clone()
        
        # Positional embeddings
        # HF: biogpt.embed_positions.weight -> Neuron: embed_positions.embedding.weight (wrapped)
        if "biogpt.embed_positions.weight" in state_dict:
            neuron_state_dict["embed_positions.embedding.weight"] = state_dict["biogpt.embed_positions.weight"].clone()
        
        # Final layer norm
        # HF: biogpt.layer_norm.weight/bias -> Neuron: norm.weight/bias
        if "biogpt.layer_norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["biogpt.layer_norm.weight"].clone()
        if "biogpt.layer_norm.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["biogpt.layer_norm.bias"].clone()
        
        # Language modeling head
        # HF: output_projection.weight -> Neuron: lm_head.weight
        if "output_projection.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["output_projection.weight"].clone()
        
        # Process each decoder layer
        for i in range(config.num_hidden_layers):
            layer_prefix_hf = f"biogpt.layers.{i}"
            layer_prefix_neuron = f"layers.{i}"
            
            # Self-attention layer norm
            # HF: biogpt.layers.{i}.self_attn_layer_norm.weight/bias
            # Neuron: layers.{i}.self_attn_layer_norm.weight/bias
            if f"{layer_prefix_hf}.self_attn_layer_norm.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn_layer_norm.weight"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn_layer_norm.weight"].clone()
            if f"{layer_prefix_hf}.self_attn_layer_norm.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn_layer_norm.bias"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn_layer_norm.bias"].clone()
            
            # Attention Q projection
            # HF: biogpt.layers.{i}.self_attn.q_proj.weight/bias
            # Neuron: layers.{i}.self_attn.qkv_proj.q_proj.weight/bias
            if f"{layer_prefix_hf}.self_attn.q_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.weight"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.q_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.q_proj.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.q_proj.bias"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.q_proj.bias"].clone()
            
            # Attention K projection
            # HF: biogpt.layers.{i}.self_attn.k_proj.weight/bias
            # Neuron: layers.{i}.self_attn.qkv_proj.k_proj.weight/bias
            if f"{layer_prefix_hf}.self_attn.k_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.weight"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.k_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.k_proj.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.k_proj.bias"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.k_proj.bias"].clone()
            
            # Attention V projection
            # HF: biogpt.layers.{i}.self_attn.v_proj.weight/bias
            # Neuron: layers.{i}.self_attn.qkv_proj.v_proj.weight/bias
            if f"{layer_prefix_hf}.self_attn.v_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.weight"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.v_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.v_proj.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.qkv_proj.v_proj.bias"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.v_proj.bias"].clone()
            
            # Attention output projection
            # HF: biogpt.layers.{i}.self_attn.out_proj.weight/bias
            # Neuron: layers.{i}.self_attn.o_proj.o_proj.weight/bias (double o_proj due to NeuronAttentionBase)
            if f"{layer_prefix_hf}.self_attn.out_proj.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.o_proj.weight"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.out_proj.weight"].clone()
            if f"{layer_prefix_hf}.self_attn.out_proj.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.self_attn.o_proj.o_proj.bias"] = \
                    state_dict[f"{layer_prefix_hf}.self_attn.out_proj.bias"].clone()
            
            # Final layer norm (before MLP)
            # HF: biogpt.layers.{i}.final_layer_norm.weight/bias
            # Neuron: layers.{i}.final_layer_norm.weight/bias
            if f"{layer_prefix_hf}.final_layer_norm.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.final_layer_norm.weight"] = \
                    state_dict[f"{layer_prefix_hf}.final_layer_norm.weight"].clone()
            if f"{layer_prefix_hf}.final_layer_norm.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.final_layer_norm.bias"] = \
                    state_dict[f"{layer_prefix_hf}.final_layer_norm.bias"].clone()
            
            # MLP fc1 (input projection)
            # HF: biogpt.layers.{i}.fc1.weight/bias
            # Neuron: layers.{i}.mlp.fc1.weight/bias
            if f"{layer_prefix_hf}.fc1.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc1.weight"] = \
                    state_dict[f"{layer_prefix_hf}.fc1.weight"].clone()
            if f"{layer_prefix_hf}.fc1.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc1.bias"] = \
                    state_dict[f"{layer_prefix_hf}.fc1.bias"].clone()
            
            # MLP fc2 (output projection)
            # HF: biogpt.layers.{i}.fc2.weight/bias
            # Neuron: layers.{i}.mlp.fc2.weight/bias
            if f"{layer_prefix_hf}.fc2.weight" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc2.weight"] = \
                    state_dict[f"{layer_prefix_hf}.fc2.weight"].clone()
            if f"{layer_prefix_hf}.fc2.bias" in state_dict:
                neuron_state_dict[f"{layer_prefix_neuron}.mlp.fc2.bias"] = \
                    state_dict[f"{layer_prefix_hf}.fc2.bias"].clone()
        
        # Add rank information for tensor parallelism
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        
        # Add rank tensors for attention layers
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = \
                torch.arange(0, tp_degree, dtype=torch.int32)
        
        # Add rank tensor for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        print(f"Converted {len(state_dict)} HuggingFace parameters to {len(neuron_state_dict)} NeuronX parameters")
        
        return neuron_state_dict
