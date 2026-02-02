#!/usr/bin/env python3
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
"""PyTorch Phi-3.5-MoE model for NXD inference."""

import math
import warnings
import gc
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# NeuronX distributed imports
from neuronx_distributed.parallel_layers import layers
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers.utils import divide
from neuronx_distributed.parallel_layers import parallel_state

# NeuronX distributed inference imports
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig, MoENeuronConfig, InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

# Transformers imports
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


def convert_phi35_moe_hf_to_neuron_state_dict(state_dict: dict, config):
    """
    Convert Phi-3.5-MoE HuggingFace state dict to NeuronX distributed inference format.
    
    CORRECTED VERSION: This function properly handles all weight mappings based on successful fixes.
    NOTE: The base class may have already removed the 'model.' prefix before calling this function.
    """
    print("🚨 CORRECTED WEIGHT CONVERSION FUNCTION CALLED! 🚨")
    print(f"📊 State dict has {len(state_dict)} keys")
    
    # Check if this is already a converted state dict
    if any(key.startswith('layers.') and 'qkv_proj' in key for key in state_dict.keys()):
        print("🔄 State dict already converted, returning as-is")
        return state_dict
    
    # Detect if model. prefix has been removed by base class
    has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
    print(f"📋 Has 'model.' prefix: {has_model_prefix}")
    
    print(f"🔄 Converting weights for {config.num_hidden_layers} layers with {config.num_local_experts} experts each")
    
    neuron_state_dict = {}
    target_dtype = torch.bfloat16
    attention_weights_converted = 0
    
    # 1. Direct mappings - handle both with and without model. prefix
    if has_model_prefix:
        direct_mappings = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight": "lm_head.weight", 
            "model.norm.weight": "norm.weight",
            "model.norm.bias": "norm.bias",
        }
    else:
        # Base class already removed model. prefix
        direct_mappings = {
            "embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight": "lm_head.weight", 
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
        }
    
    for hf_key, neuron_key in direct_mappings.items():
        if hf_key in state_dict:
            weight = state_dict[hf_key].to(target_dtype)
            neuron_state_dict[neuron_key] = weight
            print(f"   ✅ Mapped {hf_key} → {neuron_key}")
    
    # 2. Layer-by-layer conversion with FIXED mappings
    num_layers = config.num_hidden_layers
    
    for layer_idx in range(num_layers):
        # Handle both with and without model. prefix
        if has_model_prefix:
            layer_prefix_hf = f"model.layers.{layer_idx}"
        else:
            layer_prefix_hf = f"layers.{layer_idx}"
        layer_prefix_neuron = f"layers.{layer_idx}"
        
        # Layer norms (should be identical)
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            for param_type in ["weight", "bias"]:
                hf_key = f"{layer_prefix_hf}.{norm_name}.{param_type}"
                neuron_key = f"{layer_prefix_neuron}.{norm_name}.{param_type}"
                if hf_key in state_dict:
                    weight = state_dict[hf_key].to(target_dtype)
                    neuron_state_dict[neuron_key] = weight
        
        # FIXED: Attention weights mapping - GQA module expects qkv_proj structure
        attention_mappings = {
            "q_proj": "qkv_proj.q_proj",
            "k_proj": "qkv_proj.k_proj", 
            "v_proj": "qkv_proj.v_proj",
            "o_proj": "o_proj"
        }
        
        for hf_proj, neuron_proj in attention_mappings.items():
            for param_type in ["weight", "bias"]:
                hf_key = f"{layer_prefix_hf}.self_attn.{hf_proj}.{param_type}"
                neuron_key = f"{layer_prefix_neuron}.self_attn.{neuron_proj}.{param_type}"
                if hf_key in state_dict:
                    weight = state_dict[hf_key].to(target_dtype)
                    neuron_state_dict[neuron_key] = weight
                    attention_weights_converted += 1
        
        # MoE router weights - KEEP IN FLOAT32 for precision in expert selection
        # Router dtype is controlled by RouterConfig (default float32), so keep weights in float32
        hf_router_key = f"{layer_prefix_hf}.block_sparse_moe.gate.weight"
        neuron_router_key = f"{layer_prefix_neuron}.block_sparse_moe.router.linear_router.weight"
        if hf_router_key in state_dict:
            # Use float32 for router to match RouterConfig.dtype default
            weight = state_dict[hf_router_key].to(torch.float32)
            neuron_state_dict[neuron_router_key] = weight
        
        # CRITICAL FIX: MoE expert weights transformation
        print(f"🔄 Transforming MoE expert weights for layer {layer_idx}")
        num_experts = config.num_local_experts
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
        
        # Collect all expert weights
        expert_gate_weights = []
        expert_up_weights = []
        expert_down_weights = []
        
        for expert_idx in range(num_experts):
            w1_key = f"{layer_prefix_hf}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            w2_key = f"{layer_prefix_hf}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            w3_key = f"{layer_prefix_hf}.block_sparse_moe.experts.{expert_idx}.w3.weight"
            
            if all(key in state_dict for key in [w1_key, w2_key, w3_key]):
                w1_weight = state_dict[w1_key].to(target_dtype)
                w2_weight = state_dict[w2_key].to(target_dtype)
                w3_weight = state_dict[w3_key].to(target_dtype)
                
                expert_gate_weights.append(w1_weight)
                expert_down_weights.append(w2_weight)
                expert_up_weights.append(w3_weight)
        
        if len(expert_gate_weights) == num_experts:
            # Create gate_up_proj tensor
            gate_up_list = []
            for gate_w, up_w in zip(expert_gate_weights, expert_up_weights):
                gate_up_concat = torch.cat([gate_w, up_w], dim=0)
                gate_up_transposed = gate_up_concat.transpose(0, 1)
                gate_up_list.append(gate_up_transposed)
            
            stacked_gate_up = torch.stack(gate_up_list, dim=0)
            neuron_gate_up_key = f"{layer_prefix_neuron}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"
            neuron_state_dict[neuron_gate_up_key] = stacked_gate_up
            
            # Create down_proj tensor
            down_list = []
            for down_w in expert_down_weights:
                down_transposed = down_w.transpose(0, 1)
                down_list.append(down_transposed)
            
            stacked_down = torch.stack(down_list, dim=0)
            neuron_down_key = f"{layer_prefix_neuron}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"
            neuron_state_dict[neuron_down_key] = stacked_down
        
    
    print(f"✅ Converted {attention_weights_converted} attention weights")
    print(f"✅ Converted {len(neuron_state_dict)} total weights")
    
    return neuron_state_dict
    


class PhiMoeInferenceConfig(InferenceConfig):
    """Configuration class for Phi-3.5-MoE model"""
    
    def __init__(self, neuron_config=None, **kwargs):
        # If neuron_config is not provided, create a default MoENeuronConfig
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import MoENeuronConfig
            neuron_config = MoENeuronConfig(
                tp_degree=kwargs.get('tp_degree', 1),
                ep_degree=kwargs.get('ep_degree', 1),
                batch_size=kwargs.get('batch_size', 1),
                max_context_length=kwargs.get('max_context_length', 128),
                seq_len=kwargs.get('seq_len', 256),
                on_cpu=kwargs.get('on_cpu', True),
            )
        
        # Call parent InferenceConfig __init__ with neuron_config
        super().__init__(neuron_config=neuron_config, **kwargs)
        
        # Set model-specific attributes with defaults
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 32000)
        self.model_type = kwargs.get('model_type', 'phimoe')
        self.architectures = kwargs.get("architectures", ["PhiMoEForCausalLM"])
        self.auto_map = kwargs.get("auto_map", {})
        self.transformers_version = kwargs.get("transformers_version", "4.37.0")
        
        # Model architecture parameters - required by the model
        self.vocab_size = kwargs.get('vocab_size', 32064)
        self.hidden_size = kwargs.get('hidden_size', 3584)
        self.intermediate_size = kwargs.get('intermediate_size', 14336)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 32)
        self.num_attention_heads = kwargs.get('num_attention_heads', 32)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 8)
        self.num_local_experts = kwargs.get('num_local_experts', 16)
        self.num_experts_per_tok = kwargs.get('num_experts_per_tok', 2)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 131072)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-5)
        self.attention_bias = kwargs.get('attention_bias', True)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        self.rope_scaling = kwargs.get('rope_scaling', None)
        self.hidden_act = kwargs.get('hidden_act', 'silu')
        self.torch_dtype = kwargs.get('torch_dtype', 'bfloat16')
        self.rope_scaling = kwargs.get('rope_scaling', None)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-05)
        self.hidden_act = kwargs.get('hidden_act', 'silu')
        self.tie_word_embeddings = kwargs.get('tie_word_embeddings', False)
        self.use_cache = kwargs.get('use_cache', True)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        
        # Add missing attributes required by model_base
        self.output_attentions = kwargs.get('output_attentions', False)
        self.output_hidden_states = kwargs.get('output_hidden_states', False)
        self.use_return_dict = kwargs.get('use_return_dict', True)
        
        # Expose torch_dtype from neuron_config for compatibility
        if hasattr(self, 'neuron_config') and hasattr(self.neuron_config, 'torch_dtype'):
            self.torch_dtype = self.neuron_config.torch_dtype
        
        # Phi-3.5-MoE has no shared experts (like Qwen3)
        self.n_shared_experts = 0
        
        # Set MoE-specific neuron config parameters
        if hasattr(self, 'neuron_config'):
            # Set GLU MLP configuration for Phi-3.5-MoE
            if not hasattr(self.neuron_config, 'glu_type'):
                self.neuron_config.glu_type = 'swiglu'
            if not hasattr(self.neuron_config, 'glu_mlp'):
                self.neuron_config.glu_mlp = True
    
    def save(self, save_directory):
        """Save the configuration to a directory."""
        import os
        import json
        import torch
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        config_file = os.path.join(save_directory, "config.json")
        
        # Convert config to dictionary
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'neuron_config':
                # Handle torch.dtype serialization
                if isinstance(value, torch.dtype):
                    config_dict[key] = str(value).replace('torch.', '')
                elif hasattr(value, '__name__'):
                    config_dict[key] = str(value)
                elif value is None:
                    config_dict[key] = None
                else:
                    try:
                        # Test if value is JSON serializable
                        json.dumps(value)
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        config_dict[key] = str(value)
        
        # Save to JSON file
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
    
    def to_dict(self):
        """Convert config to dictionary."""
        import torch
        import json
        
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'neuron_config':
                # Handle torch.dtype serialization
                if isinstance(value, torch.dtype):
                    config_dict[key] = str(value).replace('torch.', '')
                elif hasattr(value, '__name__'):
                    config_dict[key] = str(value)
                elif value is None:
                    config_dict[key] = None
                else:
                    try:
                        # Test if value is JSON serializable
                        json.dumps(value)
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        config_dict[key] = str(value)
        return config_dict
        
        # Required for inference
        self.fused_spec_config = kwargs.get('fused_spec_config', None)
    
    def get_text_config(self):
        """Return text configuration for compatibility"""
        return self
    
    @classmethod
    def from_pretrained(cls, model_path, neuron_config=None, **kwargs):
        """Load configuration from pretrained model"""
        from transformers import AutoConfig
        
        # Load HuggingFace config
        hf_config = AutoConfig.from_pretrained(model_path)
        
        # Convert to our config format
        config_dict = {
            'vocab_size': hf_config.vocab_size,
            'hidden_size': hf_config.hidden_size,
            'intermediate_size': hf_config.intermediate_size,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'num_attention_heads': hf_config.num_attention_heads,
            'num_key_value_heads': hf_config.num_key_value_heads,
            'max_position_embeddings': hf_config.max_position_embeddings,
            'num_local_experts': hf_config.num_local_experts,
            'num_experts_per_tok': hf_config.num_experts_per_tok,
            'router_aux_loss_coef': getattr(hf_config, 'router_aux_loss_coef', 0.0),
            'router_jitter_noise': getattr(hf_config, 'router_jitter_noise', 0.01),
            'input_jitter_noise': getattr(hf_config, 'input_jitter_noise', 0.01),
            'attention_dropout': getattr(hf_config, 'attention_dropout', 0.0),
            'hidden_dropout': getattr(hf_config, 'hidden_dropout', 0.0),
            'attention_bias': getattr(hf_config, 'attention_bias', True),
            'hidden_act': hf_config.hidden_act,
            'initializer_range': hf_config.initializer_range,
            'rms_norm_eps': hf_config.rms_norm_eps,
            'rope_theta': hf_config.rope_theta,
            'rope_scaling': getattr(hf_config, 'rope_scaling', None),
            'tie_word_embeddings': hf_config.tie_word_embeddings,
            'use_cache': hf_config.use_cache,
            'bos_token_id': hf_config.bos_token_id,
            'eos_token_id': hf_config.eos_token_id,
            'torch_dtype': getattr(hf_config, 'torch_dtype', torch.bfloat16),
            'num_cores_per_group': 1
        }
        
        # Add neuron-specific parameters if provided
        if neuron_config is not None:
            if isinstance(neuron_config, dict):
                config_dict.update(neuron_config)
            else:
                # If it's a config object, extract relevant attributes
                for attr in ['tp_degree', 'max_batch_size', 'seq_len', 'buckets']:
                    if hasattr(neuron_config, attr):
                        config_dict[attr] = getattr(neuron_config, attr)
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        # Create neuron_config if not provided
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import NeuronConfig
            neuron_config = NeuronConfig()
        
        # Remove neuron_config from config_dict if it exists to avoid duplicate argument
        config_dict.pop('neuron_config', None)
        
        # Pass neuron_config as keyword argument for MoENeuronConfig
        return cls(neuron_config=neuron_config, **config_dict)

class PhiMoELayerNorm(nn.Module):
    """LayerNorm for Phi-3.5-MoE to match HuggingFace architecture"""
    
    def __init__(self, hidden_size, eps=1e-6, dtype=None):
        super().__init__()
        # Use the specified dtype or default to bfloat16 for memory efficiency
        if dtype is None:
            dtype = torch.bfloat16
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = ((hidden_states - mean) ** 2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states + self.bias
        return hidden_states.to(input_dtype)



class PhiMoEAttention(NeuronAttentionBase):
    """Multi-Head Attention for Phi-3.5-MoE with NeuronX optimization"""
    
    def __init__(self, config: PhiMoeInferenceConfig, layer_idx: Optional[int] = None):
        # Create rotary embedding
        rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=rotary_emb,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
        )
        
        self.layer_idx = layer_idx





class PhiMoEDecoderLayer(nn.Module):
    """Decoder layer for Phi-3.5-MoE"""
    
    def __init__(self, config: PhiMoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = PhiMoEAttention(config, layer_idx=layer_idx)
        
        # Use the new MoE v2 module (same as Qwen3 MoE)
        self.block_sparse_moe = initialize_moe_module(config=config)
        
        # Ensure MoE is in the correct dtype
        target_dtype = getattr(config, 'torch_dtype', torch.bfloat16)
        self.block_sparse_moe = self.block_sparse_moe.to(target_dtype)
        
        self.input_layernorm = PhiMoELayerNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)
        self.post_attention_layernorm = PhiMoELayerNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# Configuration class for NeuronX inference

# NeuronX model wrapper
class PhiMoEModel(NeuronBaseModel):
    """NeuronX Phi-3.5-MoE base model for tracing"""
    
    def setup_attr_for_model(self, config: PhiMoeInferenceConfig):
        """Setup attributes for the model"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: PhiMoeInferenceConfig):
        """Initialize the model components"""
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList([
            PhiMoEDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = PhiMoELayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=getattr(config, 'lm_head_bias', False),  # FIXED: Use config parameter with fallback
        )


class PhiMoEForCausalLM(NeuronBaseForCausalLM):
    """NeuronX wrapper for Phi-3.5-MoE Causal Language Model"""
    
    _model_cls = PhiMoEModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace model (not used in our case)"""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return PhiMoEInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: PhiMoeInferenceConfig) -> dict:
        """Convert HuggingFace state dict to NeuronX format"""
        return convert_phi35_moe_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        """Get compiler arguments for Phi-3.5-MoE"""
        compiler_args = "--model-type=transformer -O1"
        # MoE-specific optimizations
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma --max-local-tensor-tile-size-in-bytes=4096'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=false'"
        compiler_args += " --verbose=35 --enable-internal-neff-wrapper"
        return compiler_args

# Aliases for compatibility with test scripts
NeuronPhiMoEForCausalLM = PhiMoEForCausalLM
PhiMoEConfig = PhiMoeInferenceConfig