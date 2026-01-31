# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
PyTorch HunYuanDenseV1 model for NeuronX inference.

This is a port of the HuggingFace HunYuanDenseV1ForCausalLM model to run on AWS Trainium/Inferentia
using the neuronx_distributed_inference framework.

Key architectural features of HunYuanDenseV1:
- Dense transformer decoder (not MoE)
- Grouped Query Attention (GQA) with configurable num_key_value_heads
- SwiGLU MLP activation (gate_proj, up_proj, down_proj)
- RMSNorm for layer normalization
- RoPE (Rotary Position Embeddings) with optional dynamic scaling
- Query and Key layer normalization after projection (unique to HunYuan)

Reference: transformers/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py
"""
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm implementation.
    
    Uses CustomRMSNorm for NeuronX hardware, falls back to a simple
    RMSNorm implementation for CPU mode.
    """
    if cpu_mode():
        # Simple RMSNorm for CPU mode
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
                return self.weight * hidden_states.to(input_dtype)
        return SimpleRMSNorm
    return CustomRMSNorm


class HunYuanDenseV1NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for HunYuanDenseV1.
    
    CRITICAL: This class is REQUIRED for token generation to work.
    Without it, token generation HLO tracing fails with tensor shape mismatches.
    
    The attn_cls attribute tells the framework which attention class to use
    during token generation tracing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronHunYuanDenseV1Attention


class HunYuanDenseV1InferenceConfig(InferenceConfig):
    """
    Configuration class for HunYuanDenseV1 inference on Neuron.
    
    This class handles loading the HuggingFace config and adding
    derived attributes required by the NeuronX framework.
    """

    def add_derived_config(self):
        """
        Add derived configuration parameters required by the framework.
        
        CRITICAL: This method is called during initialization and MUST set
        all framework-required attributes.
        """
        # REQUIRED: Framework uses this for attention computation distribution
        self.num_cores_per_group = 1
        
        # Calculate head_dim if not present in HF config
        if not hasattr(self, 'head_dim') or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Handle rope_theta from rope_scaling or direct attribute
        if not hasattr(self, 'rope_theta') or self.rope_theta is None:
            if hasattr(self, 'rope_scaling') and self.rope_scaling:
                self.rope_theta = 10000.0  # Default base
            else:
                self.rope_theta = 10000.0
        
        # Handle use_qk_norm flag for query/key normalization
        if not hasattr(self, 'use_qk_norm'):
            self.use_qk_norm = True  # HunYuan uses QK norm by default
        
        # REQUIRED: Framework expects all 4 of these attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        """
        List of required attributes from HuggingFace config.json.
        
        These attributes MUST be present in the HF config or provided during initialization.
        """
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """
        Return the NeuronConfig class to use.
        
        CRITICAL: MUST return custom NeuronConfig class, NOT base NeuronConfig.
        Returning base NeuronConfig will cause token generation to fail.
        """
        return HunYuanDenseV1NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from HuggingFace model directory.
        
        Args:
            model_path: Path to HuggingFace model directory
            **kwargs: Additional config overrides including neuron_config
        """
        import json
        import os
        
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        def load_config_fn(config_instance):
            """Callback to load config attributes"""
            for key, value in config_dict.items():
                if not key.startswith("_"):
                    setattr(config_instance, key, value)
            for key, value in kwargs.items():
                setattr(config_instance, key, value)

        # CRITICAL: Create default NeuronConfig if none provided
        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()

        return cls(neuron_config=neuron_config, load_config=load_config_fn)


class NeuronHunYuanDenseV1Attention(NeuronAttentionBase):
    """
    HunYuanDenseV1 attention implementation for NeuronX.
    
    Key differences from standard Llama attention:
    - Query and Key layer normalization after projection (query_layernorm, key_layernorm)
    
    Reference: HunYuanDenseV1Attention in modeling_hunyuan_v1_dense.py
    """

    def __init__(self, config: HunYuanDenseV1InferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # HunYuanDenseV1 uses DynamicNTKAlpha RoPE scaling
        # When rope_scaling has type="dynamic" and alpha is set:
        # base = rope_theta * alpha ** (head_dim / (head_dim - 2))
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        rope_scaling = getattr(config, 'rope_scaling', None)
        
        if rope_scaling and rope_scaling.get('type') == 'dynamic' and rope_scaling.get('alpha'):
            alpha = rope_scaling['alpha']
            # DynamicNTKAlpha formula from HunYuanDenseV1RotaryEmbedding
            rope_base = rope_theta * (alpha ** (head_dim / (head_dim - 2)))
        else:
            rope_base = rope_theta
        
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_base,
        )

        # HunYuanDenseV1 uses query and key layer normalization
        # Reference: self.query_layernorm and self.key_layernorm in HunYuanDenseV1Attention
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, 'attention_bias', False),
            o_bias=getattr(config, 'attention_bias', False),
            # HunYuanDenseV1 specific: query and key layer normalization
            q_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
        )


class NeuronHunYuanDenseV1DecoderLayer(nn.Module):
    """
    HunYuanDenseV1 decoder layer implementation for NeuronX.
    
    Architecture:
    - Pre-normalization (input_layernorm before attention)
    - Self-attention with query/key layer normalization
    - Post-attention normalization (post_attention_layernorm before MLP)
    - SwiGLU MLP (reuses NeuronLlamaMLP which implements SwiGLU)
    
    Reference: HunYuanDenseV1DecoderLayer in modeling_hunyuan_v1_dense.py
    """

    def __init__(self, config: HunYuanDenseV1InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronHunYuanDenseV1Attention(config)
        # Reuse NeuronLlamaMLP which implements SwiGLU (gate_proj, up_proj, down_proj)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
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
        Forward pass for decoder layer.
        
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - CRITICAL: Use tuple unpacking, not attribute access
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        # Return 5-tuple expected by framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronHunYuanDenseV1Model(NeuronBaseModel):
    """
    HunYuanDenseV1 base model for NeuronX.
    
    IMPORTANT: Inherits from NeuronBaseModel, NOT NeuronBaseForCausalLM.
    The CausalLM wrapper comes later.
    
    Reference: HunYuanDenseV1Model in modeling_hunyuan_v1_dense.py
    """

    def setup_attr_for_model(self, config: HunYuanDenseV1InferenceConfig):
        """
        Setup attributes required by the framework.
        Called BEFORE init_model() to set up instance attributes.
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: HunYuanDenseV1InferenceConfig):
        """
        Initialize model components.
        Called AFTER setup_attr_for_model() to create layers.
        """
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # Decoder layers - no layer_idx needed
        self.layers = nn.ModuleList(
            [NeuronHunYuanDenseV1DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head - CRITICAL: lm_head belongs in base model
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronHunYuanDenseV1ForCausalLM(NeuronBaseForCausalLM):
    """
    HunYuanDenseV1 Causal Language Model wrapper for NeuronX.
    
    This is the top-level class that wraps the base model and provides
    weight loading and conversion utilities.
    
    Reference: HunYuanDenseV1ForCausalLM in modeling_hunyuan_v1_dense.py
    """

    _model_cls = NeuronHunYuanDenseV1Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load HuggingFace model for weight extraction.
        """
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format.
        
        CRITICAL: Must add rank utilities for tensor parallelism.
        CRITICAL: Must rename query_layernorm/key_layernorm to q_layernorm/k_layernorm.
        """
        neuron_config = config.neuron_config

        # Add rank utilities for vocabulary parallelism
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        # Add rank utilities for attention layers (tensor parallelism)
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
            
            # HunYuanDenseV1 uses query_layernorm and key_layernorm
            # Map to q_layernorm and k_layernorm expected by NeuronAttentionBase
            # CRITICAL: Must rename, not just copy, to avoid redundant keys warning
            query_ln_key = f"layers.{i}.self_attn.query_layernorm.weight"
            key_ln_key = f"layers.{i}.self_attn.key_layernorm.weight"
            q_ln_key = f"layers.{i}.self_attn.q_layernorm.weight"
            k_ln_key = f"layers.{i}.self_attn.k_layernorm.weight"
            
            if query_ln_key in state_dict:
                state_dict[q_ln_key] = state_dict.pop(query_ln_key)
            if key_ln_key in state_dict:
                state_dict[k_ln_key] = state_dict.pop(key_ln_key)

        # Add rank utilities for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied embeddings and lm_head weights.
        
        CRITICAL: HunYuanDenseV1 ties embed_tokens and lm_head weights.
        HuggingFace only saves one copy, but Neuron expects both keys.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return HunYuanDenseV1InferenceConfig


__all__ = [
    "HunYuanDenseV1NeuronConfig",
    "HunYuanDenseV1InferenceConfig",
    "NeuronHunYuanDenseV1Attention",
    "NeuronHunYuanDenseV1DecoderLayer",
    "NeuronHunYuanDenseV1Model",
    "NeuronHunYuanDenseV1ForCausalLM",
]
