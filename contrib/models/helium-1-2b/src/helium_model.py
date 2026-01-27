# coding=utf-8
# Copyright 2024 The Kyutai and HuggingFace Inc. teams. All rights reserved.
# Ported to NeuronX Distributed Inference
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied,
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helium model for NeuronX Distributed Inference

This is a port of the Helium model architecture to run on AWS Neuron hardware.
The architecture is similar to LLaMA with:
- Grouped Query Attention (GQA)
- SwiGLU activation in MLP
- RMSNorm for layer normalization  
- RoPE (Rotary Position Embeddings)

Original implementation reference:
/shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/helium/
"""

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group

# Import the configuration
from helium_config import HeliumInferenceConfig


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm class based on execution mode.
    
    Returns CustomRMSNorm for Neuron hardware, standard RMSNorm for CPU.
    This follows the pattern used in the LLaMA implementation.
    """
    if cpu_mode():
        # For CPU mode, use a simple implementation
        class SimpleRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
        
        return SimpleRMSNorm
    else:
        # For Neuron hardware, use optimized CustomRMSNorm
        return CustomRMSNorm


class NeuronHeliumMLP(nn.Module):
    """
    Helium MLP layer with SwiGLU activation.
    
    This follows the same architecture as the original Helium MLP:
    - gate_proj: Projects hidden_size -> intermediate_size
    - up_proj: Projects hidden_size -> intermediate_size  
    - down_proj: Projects intermediate_size -> hidden_size
    - Activation: SiLU (Swish)
    - Pattern: down_proj(act_fn(gate_proj(x)) * up_proj(x))
    
    Reference: HeliumMLP in modeling_helium.py
    """
    
    def __init__(self, config: HeliumInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate and up projections use ColumnParallelLinear for tensor parallelism
        # These project from hidden_size to intermediate_size
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # Down projection uses RowParallelLinear
        # Input is parallel (from gate/up), output is gathered
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # SiLU activation (also known as Swish)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        """
        Forward pass for SwiGLU MLP.
        
        Implements: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            tuple: (output, None) - None for compatibility with framework expectations
        """
        # Apply gate projection and activation
        gate_output = self.act_fn(self.gate_proj(x))
        
        # Apply up projection
        up_output = self.up_proj(x)
        
        # Element-wise multiplication (SwiGLU)
        intermediate_output = gate_output * up_output
        
        # Apply down projection
        output = self.down_proj(intermediate_output)
        
        # Return tuple for compatibility with framework
        return output, None


class NeuronHeliumAttention(NeuronAttentionBase):
    """
    Helium attention layer with Grouped Query Attention (GQA) and RoPE.
    
    This extends NeuronAttentionBase to provide GQA support where:
    - Query heads: num_attention_heads (e.g., 16)
    - Key-Value heads: num_key_value_heads (e.g., 8)
    - GQA ratio: num_attention_heads / num_key_value_heads (e.g., 2:1)
    
    Features:
    - Rotary Position Embeddings (RoPE)
    - Optional bias in projections (controlled by attention_bias)
    - Tensor parallelism support
    
    Reference: HeliumAttention in modeling_helium.py
    """
    
    def __init__(self, config: HeliumInferenceConfig):
        # Create RoPE embeddings
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize the base attention class with all required parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.attention_bias,
            o_bias=False,  # Helium uses bias=False for o_proj
            rms_norm_eps=config.rms_norm_eps,
        )


class NeuronHeliumDecoderLayer(nn.Module):
    """
    Helium decoder layer combining attention and MLP with residual connections.
    
    Architecture:
    1. Input -> LayerNorm -> Attention -> Residual Add
    2. -> LayerNorm -> MLP -> Residual Add -> Output
    
    This follows the standard transformer decoder architecture used in Helium.
    
    Reference: HeliumDecoderLayer in modeling_helium.py
    """
    
    def __init__(self, config: HeliumInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention layer
        self.self_attn = NeuronHeliumAttention(config)
        
        # MLP layer  
        self.mlp = NeuronHeliumMLP(config)
        
        # Layer normalization (RMSNorm)
        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        """
        Forward pass for decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key-value pairs
            **kwargs: Additional arguments
            
        Returns:
            tuple: (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        # Save residual
        residual = hidden_states
        
        # Pre-attention layer norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        # NeuronAttentionBase returns (hidden_states, present_key_value, cos_cache, sin_cache)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Save residual again
        residual = hidden_states
        
        # Pre-MLP layer norm
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states, _ = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Return format consistent with framework expectations
        # (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


class NeuronHeliumModel(NeuronBaseModel):
    """
    Helium transformer model without the language modeling head.
    
    This is the core transformer model that processes input token IDs through:
    1. Token embeddings
    2. Multiple decoder layers
    3. Final layer normalization
    
    Reference: HeliumModel in modeling_helium.py
    """
    
    def setup_attr_for_model(self, config: HeliumInferenceConfig):
        """
        Setup attributes required by the NeuronBaseModel framework.
        
        This method is called during initialization and sets up all the
        attributes needed for distributed training and inference optimization.
        """
        # Required for inference optimization
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: HeliumInferenceConfig):
        """
        Initialize the model components.
        
        This method creates all the model layers:
        - Token embeddings
        - Transformer decoder layers
        - Final layer normalization
        - Language model head (lm_head)
        """
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Initialize token embeddings
        if parallel_state.model_parallel_is_initialized():
            # Use ParallelEmbedding for distributed training
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            # Language model head for token prediction
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
            # Standard embeddings for non-distributed mode
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            NeuronHeliumDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)


class NeuronHeliumForCausalLM(NeuronBaseForCausalLM):
    """
    Helium model for causal language modeling.
    
    This is the main model class that wraps NeuronHeliumModel and provides
    the interface for:
    - Model compilation
    - Weight loading
    - Inference
    
    It follows the NeuronxDistributed framework patterns for model deployment.
    
    Reference: HeliumForCausalLM in modeling_helium.py
    """
    
    # Specify the model class to use
    _model_cls = NeuronHeliumModel
    
    @staticmethod
    def get_config_cls():
        """Return the configuration class for this model"""
        return HeliumInferenceConfig
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        This method handles the conversion of weight names and formats from
        the HuggingFace checkpoint format to the NeuronX format expected by
        our model implementation.
        
        Key conversions:
        - Adds rank utilities for tensor parallelism
        - Maps weight names between formats
        
        Args:
            state_dict: HuggingFace format state dictionary
            config: Model configuration
            
        Returns:
            dict: NeuronX format state dictionary
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        # Add rank utilities for tensor parallelism support
        # This is required by the attention mechanism
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        print(f"Converted HuggingFace state dict to NeuronX format")
        print(f"  - Added rank utilities for {num_layers} layers")
        print(f"  - TP degree: {tp_degree}")
        
        return state_dict
