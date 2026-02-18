# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Phi model for NXD inference

This implementation ports the Phi-1_5 model architecture to NeuronX Distributed Inference.
Reference implementation: transformers/models/phi/modeling_phi.py

Key architectural features of Phi-1_5:
- Decoder-only transformer with 24 layers
- Multi-head attention (32 heads, no GQA)
- Partial rotary position embeddings (50% of head dimensions)
- GELU activation in MLP (not SwiGLU)
- LayerNorm (not RMSNorm like LLaMA)
- Bias in all linear layers
- Embedding and residual dropout
"""

from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class PhiNeuronConfig(NeuronConfig):
    """
    NeuronConfig for Phi model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronPhiAttention


class PhiInferenceConfig(InferenceConfig):
    """
    Configuration class for Phi model inference on NeuronX
    
    This configuration handles the unique features of Phi models:
    - Partial rotary embeddings (partial_rotary_factor)
    - LayerNorm instead of RMSNorm
    - GELU activation
    - Bias in all linear layers
    """

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        # Phi-specific: All linear layers have bias
        self.qkv_bias = True
        self.o_bias = True
        
        # Phi uses partial rotary embeddings (default 0.5 = 50% of dimensions)
        if not hasattr(self, 'partial_rotary_factor'):
            self.partial_rotary_factor = 0.5
        
        # Phi uses standard LayerNorm (not RMSNorm)
        if not hasattr(self, 'layer_norm_eps'):
            self.layer_norm_eps = 1e-5
            
        # Phi uses GELU activation
        if not hasattr(self, 'hidden_act'):
            self.hidden_act = 'gelu_new'
        
        # Dropout configurations
        if not hasattr(self, 'embd_pdrop'):
            self.embd_pdrop = 0.0
        if not hasattr(self, 'resid_pdrop'):
            self.resid_pdrop = 0.0
        if not hasattr(self, 'attention_dropout'):
            self.attention_dropout = 0.0
            
        # Optional Q-K layernorm (not used in phi-1_5 but supported in architecture)
        if not hasattr(self, 'qk_layernorm'):
            self.qk_layernorm = False
        
        # Output configuration flags (for HF compatibility)
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
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
            "rope_theta",
            "layer_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[PhiNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return PhiNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments including neuron_config
            
        Returns:
            PhiInferenceConfig: Configuration object
        """
        import json
        import os
        
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Create config dict from HF format
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 2048),
            "num_attention_heads": hf_config.get("num_attention_heads", 32),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 24),
            "vocab_size": hf_config.get("vocab_size", 51200),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 2048),
            "intermediate_size": hf_config.get("intermediate_size", 8192),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "hidden_act": hf_config.get("hidden_act", "gelu_new"),
            "partial_rotary_factor": hf_config.get("partial_rotary_factor", 0.5),
            "qk_layernorm": hf_config.get("qk_layernorm", False),
            "embd_pdrop": hf_config.get("embd_pdrop", 0.0),
            "resid_pdrop": hf_config.get("resid_pdrop", 0.0),
            "attention_dropout": hf_config.get("attention_dropout", 0.0),
            "pad_token_id": hf_config.get("pad_token_id", None),
        }
        
        # Handle num_key_value_heads (if None, will default to num_attention_heads)
        if "num_key_value_heads" in hf_config and hf_config["num_key_value_heads"] is not None:
            config_dict["num_key_value_heads"] = hf_config["num_key_value_heads"]
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronPhiAttention(NeuronAttentionBase):
    """
    Phi attention implementation for NeuronX
    
    Key differences from LLaMA attention:
    - Uses partial rotary embeddings (only rotary_ndims dimensions)
    - All projections have bias=True
    - Optional Q-K layernorm
    - Multi-head attention (not GQA) - num_key_value_heads = num_attention_heads
    
    Reference: transformers/models/phi/modeling_phi.py::PhiAttention
    """

    def __init__(self, config: PhiInferenceConfig):
        # Calculate dimensions for partial rotary embeddings
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)
        
        # Create rotary embedding only for the rotary dimensions
        rotary_emb = RotaryEmbedding(
            self.rotary_ndims,  # Only partial dimensions use RoPE
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Phi uses MHA (not GQA), so num_key_value_heads = num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', None)
        if num_key_value_heads is None:
            num_key_value_heads = config.num_attention_heads

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=self.head_dim,
            qkv_bias=config.qkv_bias,  # Phi uses bias in QKV projections
            o_bias=config.o_bias,      # Phi uses bias in output projection
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta,
        )
        
        # Store config for partial rotary
        self.partial_rotary_factor = config.partial_rotary_factor
        self.attention_dropout_prob = config.attention_dropout
        
        # Optional Q-K layernorm (not used in phi-1_5 but supported)
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            # Note: Q-K layernorm in Phi is applied per-head after projection
            # Overriding the base class q_layernorm and k_layernorm
            self.q_layernorm = nn.LayerNorm(
                self.head_dim,
                eps=config.layer_norm_eps,
                elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                self.head_dim,
                eps=config.layer_norm_eps,
                elementwise_affine=True
            )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """
        Override base class method to implement partial rotary embeddings
        
        Phi applies rotary embeddings only to the first rotary_ndims dimensions
        of Q and K, leaving the remaining dimensions as pass-through.
        
        Args:
            Q: Query tensor [batch, num_heads, seq_len, head_dim]
            K: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            V: Value tensor (used for shape inference)
            position_ids: Position IDs for RoPE
            cos_cache: Precomputed cos cache (optional)
            sin_cache: Precomputed sin cache (optional)
            use_polar_compatible_rope: Whether to use polar-compatible RoPE
            
        Returns:
            Q, K, cos_cache, sin_cache with partial rotary embeddings applied
        """
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            # Compute cos/sin if not cached
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            
            # Split Q and K into rotary and pass-through parts
            # Q: [batch, num_heads, seq_len, head_dim]
            Q_rot = Q[..., :self.rotary_ndims]
            Q_pass = Q[..., self.rotary_ndims:]
            K_rot = K[..., :self.rotary_ndims]
            K_pass = K[..., self.rotary_ndims:]
            
            # Apply rotary embeddings only to rotary part
            from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
            Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)
            
            # Concatenate back
            Q = torch.cat([Q_rot, Q_pass], dim=-1)
            K = torch.cat([K_rot, K_pass], dim=-1)
        
        elif use_polar_compatible_rope:
            # For polar-compatible RoPE, we still need partial application
            # This is a more complex case - for now, fall back to standard implementation
            # TODO: Implement partial polar-compatible RoPE if needed
            raise NotImplementedError("Polar-compatible RoPE with partial rotary is not yet implemented")
        
        return Q, K, cos_cache, sin_cache


class NeuronPhiMLP(nn.Module):
    """
    Phi MLP implementation for NeuronX
    
    Key differences from LLaMA MLP:
    - Uses simple 2-layer MLP (not SwiGLU)
    - Uses GELU activation (not SiLU)
    - Has bias in both projections
    - fc1: hidden_size -> intermediate_size
    - activation: GELU
    - fc2: intermediate_size -> hidden_size
    
    Reference: transformers/models/phi/modeling_phi.py::PhiMLP
    """

    def __init__(self, config: PhiInferenceConfig):
        super().__init__()
        self.config = config
        
        # fc1: up projection with GELU activation
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,  # Phi uses bias
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # GELU activation (new variant)
        self.activation_fn = nn.GELU(approximate='tanh')  # gelu_new uses tanh approximation
        
        # fc2: down projection
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,  # Phi uses bias
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through MLP
        
        Returns:
            Tuple of (hidden_states, None) for compatibility with framework
        """
        # Up projection
        hidden_states = self.fc1(hidden_states)
        
        # GELU activation
        hidden_states = self.activation_fn(hidden_states)
        
        # Down projection
        hidden_states = self.fc2(hidden_states)
        
        # Return tuple for compatibility
        return hidden_states, None


class NeuronPhiDecoderLayer(nn.Module):
    """
    Phi decoder layer for NeuronX
    
    Architecture:
    - Pre-norm with LayerNorm (not RMSNorm)
    - Self-attention with partial RoPE
    - MLP with GELU activation
    - Residual dropout (applied to both attention and MLP outputs)
    - Parallel attention and MLP computation (both use same normalized input)
    
    Reference: transformers/models/phi/modeling_phi.py::PhiDecoderLayer
    """

    def __init__(self, config: PhiInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = NeuronPhiAttention(config)
        
        # MLP
        self.mlp = NeuronPhiMLP(config)
        
        # Pre-norm LayerNorm (not RMSNorm like LLaMA)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Residual dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass through decoder layer
        
        Phi uses a unique architecture where:
        1. Apply LayerNorm once to input
        2. Pass normalized input to both attention and MLP (in parallel)
        3. Add dropout to both outputs
        4. Add both outputs to the original residual
        
        This is different from LLaMA which uses:
        - residual + attention(norm(x))
        - residual + mlp(norm(x))
        """
        residual = hidden_states
        
        # Apply pre-norm (shared by attention and MLP)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_output, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        attn_output = self.resid_dropout(attn_output)
        
        # MLP (uses same normalized input)
        mlp_output = self.mlp(hidden_states)[0]
        mlp_output = self.resid_dropout(mlp_output)
        
        # Combine: residual + attention_output + mlp_output
        hidden_states = attn_output + mlp_output + residual
        
        # Return in framework format
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


class NeuronPhiModel(NeuronBaseModel):
    """
    Phi model for NeuronX inference
    
    This is the main model class that inherits from NeuronBaseModel.
    It implements the required methods for the NeuronX framework:
    - setup_attr_for_model: Set up model attributes
    - init_model: Initialize model components
    
    Reference: transformers/models/phi/modeling_phi.py::PhiModel
    """

    def setup_attr_for_model(self, config: PhiInferenceConfig):
        """Setup attributes required by the framework"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: PhiInferenceConfig):
        """Initialize model components"""
        # Embedding layer
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Embedding dropout (unique to Phi)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronPhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final LayerNorm (not RMSNorm)
        # Note: The base class expects this to be named 'norm'
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # LM head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,  # Phi uses bias in lm_head
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronPhiForCausalLM(NeuronBaseForCausalLM):
    """
    Phi model for causal language modeling on NeuronX
    
    This class wraps the NeuronPhiModel and provides:
    - Model loading from HuggingFace checkpoints
    - State dict conversion from HF to Neuron format
    - Compiler arguments for NeuronX compilation
    
    Reference: transformers/models/phi/modeling_phi.py::PhiForCausalLM
    """

    _model_cls = NeuronPhiModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace model for weight extraction"""
        from transformers import PhiForCausalLM
        return PhiForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format
        
        HuggingFace Phi weight names:
        - model.embed_tokens.weight
        - model.layers.{i}.self_attn.q_proj.weight/bias
        - model.layers.{i}.self_attn.k_proj.weight/bias
        - model.layers.{i}.self_attn.v_proj.weight/bias
        - model.layers.{i}.self_attn.dense.weight/bias  (output projection)
        - model.layers.{i}.mlp.fc1.weight/bias
        - model.layers.{i}.mlp.fc2.weight/bias
        - model.layers.{i}.input_layernorm.weight/bias
        - model.final_layernorm.weight/bias
        - lm_head.weight/bias
        
        Neuron format (NeuronAttentionBase expects):
        - embed_tokens.weight
        - layers.{i}.self_attn.qkv_proj.q_proj.weight/bias
        - layers.{i}.self_attn.qkv_proj.k_proj.weight/bias
        - layers.{i}.self_attn.qkv_proj.v_proj.weight/bias
        - layers.{i}.self_attn.o_proj.o_proj.weight/bias
        - layers.{i}.mlp.fc1.weight/bias
        - layers.{i}.mlp.fc2.weight/bias
        - layers.{i}.input_layernorm.weight/bias
        - norm.weight/bias
        - lm_head.weight/bias
        """
        neuron_config = config.neuron_config
        
        # Convert HF naming to Neuron naming
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            
            # Remove 'model.' prefix if present
            if new_key.startswith('model.'):
                new_key = new_key[6:]  # Remove 'model.'
            
            # Rename attention projections to match NeuronAttentionBase format
            # q_proj -> qkv_proj.q_proj
            if '.self_attn.q_proj.' in new_key:
                new_key = new_key.replace('.self_attn.q_proj.', '.self_attn.qkv_proj.q_proj.')
            elif '.self_attn.k_proj.' in new_key:
                new_key = new_key.replace('.self_attn.k_proj.', '.self_attn.qkv_proj.k_proj.')
            elif '.self_attn.v_proj.' in new_key:
                new_key = new_key.replace('.self_attn.v_proj.', '.self_attn.qkv_proj.v_proj.')
            # dense -> o_proj.o_proj
            elif '.self_attn.dense.' in new_key:
                new_key = new_key.replace('.self_attn.dense.', '.self_attn.o_proj.o_proj.')
            
            # Rename final layernorm: final_layernorm -> norm
            if new_key.startswith('final_layernorm.'):
                new_key = new_key.replace('final_layernorm.', 'norm.')
            
            new_state_dict[new_key] = value
        
        state_dict = new_state_dict
        
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
        Update state dict for tied embeddings
        
        Phi-1_5 does not tie embeddings by default (tie_word_embeddings=False),
        but this method is here for compatibility if needed.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class"""
        return PhiInferenceConfig

    def get_compiler_args(self):
        """
        Get compiler arguments for NeuronX compilation
        
        Uses similar flags to Qwen2 as they have similar architectures
        """
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
