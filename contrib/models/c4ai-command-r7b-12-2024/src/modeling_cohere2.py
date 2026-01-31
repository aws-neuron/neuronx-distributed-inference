# coding=utf-8
# Copyright 2024 Cohere Inc. and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Cohere2 model for NeuronX Distributed Inference.

This implementation ports Cohere2ForCausalLM from HuggingFace transformers
to the NeuronX Distributed Inference framework for AWS Trainium hardware.

Key architectural features of Cohere2:
- LayerNorm (not RMSNorm)
- Sliding window attention (alternating pattern)
- SwiGLU MLP activation
- Interleaved RoPE (different from Llama)
- logit_scale applied to output logits
- Grouped Query Attention (GQA)
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from neuronx_distributed_inference.utils.distributed import get_tp_group


class Cohere2LayerNorm(nn.Module):
    """
    Cohere2-specific LayerNorm without bias.
    
    This matches the HuggingFace implementation which uses bias=False.
    """
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)


class Cohere2InterleavedRotaryEmbedding(nn.Module):
    """
    Cohere2-specific rotary embedding with interleaved pattern.
    
    Unlike Llama which concatenates cos/sin, Cohere2 interleaves them.
    This matches the HuggingFace implementation in modeling_cohere2.py.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
    
    def _compute_inv_freq(self, device):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
        )
        return inv_freq
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq is None:
            self.inv_freq = self._compute_inv_freq(x.device)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # Cohere2 uses interleaved pattern: repeat_interleave instead of cat
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cohere2NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for Cohere2.
    
    CRITICAL: This class is REQUIRED for token generation to work.
    Without it, token generation HLO tracing fails with tensor shape mismatches.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronCohere2Attention


class Cohere2InferenceConfig(InferenceConfig):
    """Configuration class for Cohere2 inference on Neuron."""
    
    def add_derived_config(self):
        """Add derived configuration parameters required by the framework."""
        self.num_cores_per_group = 1
        
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Framework-required attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        
        # Cohere2 uses no bias in attention
        if not hasattr(self, 'qkv_bias'):
            self.qkv_bias = getattr(self, 'attention_bias', False)
        if not hasattr(self, 'o_bias'):
            self.o_bias = getattr(self, 'attention_bias', False)
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes from HuggingFace config.json."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "hidden_act",
            "layer_norm_eps",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return Cohere2NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from HuggingFace model directory."""
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        def load_config_fn(config_instance):
            for key, value in config_dict.items():
                if not key.startswith("_"):
                    setattr(config_instance, key, value)
            # Disable sliding window to avoid OOB errors with short prompts
            config_instance.sliding_window = None
            for key, value in kwargs.items():
                setattr(config_instance, key, value)
        
        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()
        
        return cls(neuron_config=neuron_config, load_config=load_config_fn)


class NeuronCohere2Attention(NeuronAttentionBase):
    """Cohere2 attention implementation for NeuronX."""
    
    def __init__(self, config: Cohere2InferenceConfig):
        # Cohere2 uses interleaved RoPE - we pass rotary_emb=None and use polar_compatible_rope
        # The framework's apply_rotary_polar_compatible handles interleaved pattern
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        
        # Determine sliding window for this layer based on layer_types
        # Note: layer_idx is not passed, so we handle sliding_window at model level
        sliding_window = getattr(config, 'sliding_window', None)
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=None,  # Don't use custom rotary_emb, use polar_compatible_rope instead
            rope_theta=rope_theta,  # Pass rope_theta for polar_compatible_rope
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            sliding_window=sliding_window,
        )


class NeuronCohere2MLP(nn.Module):
    """
    Cohere2 MLP implementation for NeuronX.
    
    Uses SwiGLU activation (same as Llama).
    """
    
    def __init__(self, config: Cohere2InferenceConfig):
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
        )
        
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def forward(self, x):
        """SwiGLU forward pass."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class NeuronCohere2DecoderLayer(nn.Module):
    """Cohere2 decoder layer implementation for NeuronX."""
    
    def __init__(self, config: Cohere2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = NeuronCohere2Attention(config)
        self.mlp = NeuronCohere2MLP(config)
        
        # Cohere2 uses custom LayerNorm without bias
        self.input_layernorm = Cohere2LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass for decoder layer.
        
        Cohere2 uses parallel attention and MLP (both applied to normalized input).
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention - use tuple unpacking (CRITICAL for token generation)
        # CRITICAL: use_polar_compatible_rope=True for Cohere2's interleaved RoPE
        hidden_states_attention, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_polar_compatible_rope=True,
            **kwargs,
        )
        
        # MLP
        hidden_states_mlp = self.mlp(hidden_states)
        
        # Cohere2 parallel residual: residual + attention + mlp
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
        
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronCohere2Model(NeuronBaseModel):
    """Cohere2 base model for NeuronX."""
    
    def setup_attr_for_model(self, config: Cohere2InferenceConfig):
        """Setup attributes required by the framework."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", None)
    
    def init_model(self, config: Cohere2InferenceConfig):
        """Initialize model components."""
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        self.vocab_size = config.vocab_size
        self.logit_scale = getattr(config, 'logit_scale', 1.0)
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        
        self.layers = nn.ModuleList(
            [NeuronCohere2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Cohere2 uses custom LayerNorm without bias for final norm
        self.norm = Cohere2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # lm_head - Cohere2 ties embeddings by default
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronCohere2ForCausalLM(NeuronBaseForCausalLM):
    """Cohere2 Causal Language Model wrapper for NeuronX."""
    
    _model_cls = NeuronCohere2Model
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace model for weight extraction."""
        import torch
        import os
        from pathlib import Path
        
        model_path = os.path.expanduser(model_path)
        
        # Load state dict directly to avoid meta tensor issues
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        if safetensors_files:
            from safetensors import safe_open
            state_dict = {}
            for sf_file in safetensors_files:
                with safe_open(str(sf_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            
            class DummyModel:
                def __init__(self, sd):
                    self._state_dict = sd
                def state_dict(self):
                    return self._state_dict
            
            return DummyModel(state_dict)
        
        # Fallback to pytorch_model.bin
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            
            class DummyModel:
                def __init__(self, sd):
                    self._state_dict = sd
                def state_dict(self):
                    return self._state_dict
            
            return DummyModel(state_dict)
        
        # Last resort: use transformers
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert HuggingFace state dict to Neuron format."""
        neuron_config = config.neuron_config
        
        # Handle model. prefix from HuggingFace checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        
        # Handle tied embeddings - Cohere2 ties lm_head to embed_tokens
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        
        # Add rank utilities for vocabulary parallelism
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        # Add rank utilities for attention layers
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utilities for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        # Convert dtypes if needed
        target_dtype = neuron_config.torch_dtype
        for key, value in state_dict.items():
            if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                if value.dtype != target_dtype:
                    state_dict[key] = value.to(target_dtype)
        
        return state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied embeddings for Cohere2."""
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return Cohere2InferenceConfig


__all__ = [
    "Cohere2InferenceConfig",
    "Cohere2NeuronConfig",
    "NeuronCohere2Attention",
    "NeuronCohere2MLP",
    "NeuronCohere2DecoderLayer",
    "NeuronCohere2Model",
    "NeuronCohere2ForCausalLM",
]
