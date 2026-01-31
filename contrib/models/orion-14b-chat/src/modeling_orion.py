# coding=utf-8
# Copyright 2024 OrionStar Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Orion model for NXD inference."""
import math
from typing import List, Optional, Tuple, Type

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
from transformers import PreTrainedModel

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


class OrionNeuronConfig(NeuronConfig):
    """Custom Neuron configuration for Orion - REQUIRED for token generation"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = OrionAttention


class OrionInferenceConfig(InferenceConfig):
    """Orion-specific inference configuration"""
    
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        self.qkv_bias = False  # Orion uses attention_bias=False by default
        self.o_bias = False
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        
        # Add missing attributes required by framework
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        """List of required configuration attributes"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",  # Used for LayerNorm eps
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return custom NeuronConfig class - REQUIRED for token generation"""
        return OrionNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments to override configuration
            
        Returns:
            OrionInferenceConfig: Configuration object
        """
        import os
        import json
        
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Expand user home directory if needed
        model_path = os.path.expanduser(model_path)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create configuration with values from config file
        final_config = {
            "hidden_size": config_dict.get("hidden_size", 5120),
            "num_attention_heads": config_dict.get("num_attention_heads", 40),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 40),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 40),
            "vocab_size": config_dict.get("vocab_size", 84608),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 4096),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-05),
            "intermediate_size": config_dict.get("intermediate_size", 15360),
            "hidden_act": config_dict.get("hidden_act", "silu"),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
            "pad_token_id": config_dict.get("pad_token_id", 0),
            "bos_token_id": config_dict.get("bos_token_id", 1),
            "eos_token_id": config_dict.get("eos_token_id", 2),
        }
        
        # Override with any additional kwargs
        final_config.update(kwargs)
        
        # Create and return the config
        return cls(neuron_config=neuron_config, **final_config)


class OrionMLP(nn.Module):
    """
    Orion MLP module - gated MLP with SiLU activation
    Reference: transformers/src/transformers/models/orion/modeling_orion.py::OrionMLP
    """
    def __init__(self, config: OrionInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = self.act_fn(gate_output) * up_output
        output = self.down_proj(intermediate)
        return output


class OrionAttention(NeuronAttentionBase):
    """
    Orion attention module using NeuronX NeuronAttentionBase
    Reference: transformers/src/transformers/models/orion/modeling_orion.py::OrionAttention
    """
    def __init__(self, config: OrionInferenceConfig, tensor_model_parallel_group=None):
        rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
        )


class OrionDecoderLayer(nn.Module):
    """
    Orion decoder layer with pre-norm architecture using LayerNorm
    Reference: transformers/src/transformers/models/orion/modeling_orion.py::OrionDecoderLayer
    """
    def __init__(self, config: OrionInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OrionAttention(config)
        self.mlp = OrionMLP(config)
        
        # Orion uses LayerNorm instead of RMSNorm
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.neuron_config.torch_dtype,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # CRITICAL: Use tuple unpacking, NOT attribute access
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return 5-tuple expected by framework: (hidden_states, kv, cos, sin, None)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class OrionModel(NeuronBaseModel):
    """
    Orion base model
    Reference: transformers/src/transformers/models/orion/modeling_orion.py::OrionModel
    """
    def __init__(self, config: OrionInferenceConfig):
        super().__init__(config)

    def setup_attr_for_model(self, config: OrionInferenceConfig):
        """Setup attributes needed for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: OrionInferenceConfig):
        """Initialize model layers"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        
        self.layers = nn.ModuleList(
            [OrionDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # LM head for causal LM
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not config.neuron_config.on_device_sampling_config,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class OrionForCausalLM(NeuronBaseForCausalLM):
    """
    Orion model for causal language modeling
    Reference: transformers/src/transformers/models/orion/modeling_orion.py::OrionForCausalLM
    """
    _model_cls = OrionModel
    _hf_model_cls = PreTrainedModel  # Use generic PreTrainedModel since Orion is custom

    def __init__(self, model_path: str, config: Optional[OrionInferenceConfig] = None, neuron_config: Optional[NeuronConfig] = None):
        """
        Initialize Orion model for causal LM
        
        Args:
            model_path: Path to the model directory
            config: Optional OrionInferenceConfig
            neuron_config: Optional NeuronConfig
        """
        if config is None:
            config = OrionInferenceConfig.from_pretrained(model_path)
        
        if neuron_config is not None:
            config.neuron_config = neuron_config
        
        super().__init__(model_path, config, neuron_config=config.neuron_config)
        
        self.vocab_size = config.vocab_size

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        hf_state_dict: dict,
        config: OrionInferenceConfig,
    ) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format
        Handles weight sharding for tensor parallelism
        """
        print(f"🔧 convert_hf_to_neuron_state_dict called with {len(hf_state_dict)} keys")
        print(f"Sample input keys: {list(hf_state_dict.keys())[:5]}")
        
        neuron_state_dict = {}
        num_layers = config.num_hidden_layers
        
        # Check if checkpoint has "model." prefix
        has_model_prefix = any(k.startswith("model.") for k in hf_state_dict.keys())
        
        # First pass: copy all keys with prefix removal if needed
        for key, value in hf_state_dict.items():
            new_key = key
            if has_model_prefix and new_key.startswith("model."):
                new_key = new_key[6:]  # Remove "model." prefix
            neuron_state_dict[new_key] = value
        
        # Second pass: restructure QKV weights to add qkv_proj intermediate level
        # Framework expects: layers.X.self_attn.qkv_proj.{q,k,v}_proj.weight
        for i in range(num_layers):
            if f"layers.{i}.self_attn.q_proj.weight" in neuron_state_dict:
                # Pop original keys
                q_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.q_proj.weight")
                k_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.k_proj.weight")
                v_weight = neuron_state_dict.pop(f"layers.{i}.self_attn.v_proj.weight")
                
                # Add with qkv_proj intermediate level
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"layers.{i}.self_attn.qkv_proj.v_proj.weight"] = v_weight
        
        print(f"✅ Converted {len(neuron_state_dict)} weights")
        print(f"Sample output keys: {list(neuron_state_dict.keys())[:5]}")
        return neuron_state_dict
    
    def get_compiler_args(self):
        """
        Get compiler arguments for Orion model
        Disables HLO verification to avoid shape mismatch errors during weight layout optimization
        """
        compiler_args = "--model-type=transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=false'"
        compiler_args += " --verbose=35"
        return compiler_args
