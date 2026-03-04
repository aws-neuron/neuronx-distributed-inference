# coding=utf-8
# Copyright 2025 The GLM4 & ZhipuAI team and the HuggingFace Inc. team. All rights reserved.
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
PyTorch GLM-4 model for NeuronX Distributed Inference

This implementation ports the GLM-4 architecture from:

Key architectural features:
- Grouped Query Attention (GQA) with 32 Q heads and 2 KV heads
- Attention projections have bias (attention_bias=True)
- 2 RMSNorm layers per decoder layer (model_type="glm", not "glm4")
- Fused gate_up_proj in MLP that is split into gate_proj and up_proj  
- Custom RoPE with partial_rotary_factor=0.5 - only half of head_dim gets rotary
- INTERLEAVED RoPE pattern: rotate_half uses x[..., 0::2] and x[..., 1::2]
- SiLU activation in MLP
"""

import math
import gc
import os
import json
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm.
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class Glm4RotaryEmbedding(nn.Module):
    """
    GLM-4 Rotary Position Embedding.

    GLM-4-9b-chat-hf uses partial_rotary_factor=0.5 (half of head_dim=128, so rotary_dim=64).
    Only the first 64 dimensions of Q and K get rotary embeddings applied.
    The remaining 64 dimensions pass through unchanged.

    This model also uses an INTERLEAVED RoPE pattern where rotate_half operates on
    alternating elements (x[..., 0::2] and x[..., 1::2]) rather than splitting in half.

    Reference: transformers/src/transformers/models/glm/modeling_glm.py
    Reference: transformers/src/transformers/models/glm/configuration_glm.py (partial_rotary_factor=0.5)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        partial_rotary_factor: float = 0.5,  # GLM-4 uses 0.5 by default
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate the rotary dimension (full head_dim for GLM-4, no partial factor)
        self.rotary_dim = int(dim * partial_rotary_factor)

        # Compute inverse frequencies for the full rotary dimension
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            position_ids: Position IDs of shape (batch, seq_len)
            
        Returns:
            cos, sin: Rotary embeddings of shape (batch, seq_len, rotary_dim)
        """
        # Expand inv_freq for batch processing
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # GLM-4 concatenates freqs instead of interleaving
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    GLM-4 rotates half the hidden dims using interleaved pattern.
    
    Reference: modeling_glm4.py - rotate_half function
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_glm4_rotary_pos_emb(q, k, cos, sin, partial_rotary_factor=0.5):
    """
    Apply GLM-4's rotary position embedding to query and key tensors.
    
    GLM-4 applies rotary embeddings only to the first `rotary_dim` dimensions
    (controlled by partial_rotary_factor, typically 0.5).
    
    Reference: modeling_glm4.py - apply_rotary_pos_emb function
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, kv_heads, seq_len, head_dim)
        cos: Cosine of rotary embeddings (batch, seq_len, rotary_dim)
        sin: Sine of rotary embeddings (batch, seq_len, rotary_dim)
        partial_rotary_factor: Fraction of head_dim to apply rotary to
        
    Returns:
        Rotated query and key tensors
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, rotary_dim)
    sin = sin.unsqueeze(1)  # (batch, 1, seq_len, rotary_dim)
    
    # GLM-4 uses interleaved cos/sin
    rotary_dim = cos.shape[-1]
    cos = cos[..., :rotary_dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., :rotary_dim // 2].repeat_interleave(2, dim=-1)
    
    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    # Apply rotary embeddings
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate rotary and pass-through parts
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    
    return q_embed, k_embed


class Glm4NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for GLM-4 model.
    Extends base NeuronConfig with GLM-4 specific attention class.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = "NeuronGlm4Attention"


class Glm4InferenceConfig(InferenceConfig):
    """
    Configuration class for GLM-4 model inference on Neuron.
    
    Key GLM-4 specific attributes:
    - attention_bias: True (QKV projections have bias)
    - partial_rotary_factor: 0.5 (only half of head_dim gets rotary)
    - 4 layer norms per decoder layer
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters for GLM-4."""
        self.num_cores_per_group = 1
        # GLM-4 uses bias in attention projections
        self.qkv_bias = getattr(self, 'attention_bias', True)
        self.o_bias = False  # Output projection has no bias
        # Partial rotary factor
        self.partial_rotary_factor = getattr(self, 'partial_rotary_factor', 0.5)
        
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for GLM-4 configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]
        
    @classmethod
    def get_neuron_config_cls(cls) -> Type[Glm4NeuronConfig]:
        return Glm4NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Glm4InferenceConfig":
        """
        Load configuration from a pretrained GLM-4 model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            Glm4InferenceConfig: Configuration object for GLM-4
        """
        # Handle tilde expansion
        model_path = os.path.expanduser(model_path)
        
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            params = json.load(f)
        
        # Create config dict from GLM-4 config.json
        # Map HuggingFace config names to our expected names
        config_dict = {
            "hidden_size": params.get("hidden_size", 4096),
            "num_attention_heads": params.get("num_attention_heads", 32),
            "num_hidden_layers": params.get("num_hidden_layers", 40),
            "num_key_value_heads": params.get("num_key_value_heads", 2),
            "vocab_size": params.get("vocab_size", 151552),
            "max_position_embeddings": params.get("max_position_embeddings", 131072),
            "rope_theta": params.get("rope_theta", 10000.0),
            "rms_norm_eps": params.get("rms_norm_eps", 1.5625e-07),
            "hidden_act": params.get("hidden_act", "silu"),
            "intermediate_size": params.get("intermediate_size", 13696),
            "head_dim": params.get("head_dim", 128),
            "attention_bias": params.get("attention_bias", True),
            "partial_rotary_factor": params.get("partial_rotary_factor", 0.5),
            "pad_token_id": params.get("pad_token_id", 151329),
            "tie_word_embeddings": params.get("tie_word_embeddings", False),
            # Standard HuggingFace config attributes needed by the framework
            "output_attentions": False,
            "output_hidden_states": False,
            "use_cache": True,
        }
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronGlm4Attention(NeuronAttentionBase):
    """
    GLM-4 Attention implementation for NeuronX.
    
    Key differences from standard LLaMA attention:
    1. Attention projections (Q, K, V) have bias (attention_bias=True)
    2. Uses custom rotary embeddings with partial_rotary_factor
    3. GQA with 32 Q heads and 2 KV heads
    
    """
    
    def __init__(self, config: Glm4InferenceConfig):
        # Create GLM-4 specific rotary embedding
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        partial_rotary_factor = getattr(config, 'partial_rotary_factor', 0.5)
        
        rotary_emb = Glm4RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            qkv_bias=getattr(config, 'qkv_bias', True),  # GLM-4 has attention bias
            o_bias=getattr(config, 'o_bias', False),
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
        )
        
        self.partial_rotary_factor = partial_rotary_factor
        
    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope=False):
        """
        Override to use GLM-4's custom rotary embedding application.
        """
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_glm4_rotary_pos_emb(Q, K, cos_cache, sin_cache, self.partial_rotary_factor)
        return Q, K, cos_cache, sin_cache


class NeuronGlm4MLP(nn.Module):
    """
    GLM-4 MLP implementation for NeuronX.
    
    Key differences from standard LLaMA MLP:
    - Original GLM-4 uses fused gate_up_proj (single linear -> 2 * intermediate_size)
    - For NeuronX, we split this into separate gate_proj and up_proj for parallelization
    - Uses SwiGLU activation (silu(gate) * up)
    
    Reference: modeling_glm4.py - Glm4MLP class
    """
    
    def __init__(self, config: Glm4InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]  # silu
        
        if parallel_state.model_parallel_is_initialized():
            # Split gate and up projections for tensor parallelism
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            
    def forward(self, x):
        """
        Forward pass implementing SwiGLU activation.
        
        Original GLM-4: up_states = gate_up_proj(x); gate, up = chunk(up_states); out = down(up * silu(gate))
        Our implementation: out = down(up_proj(x) * silu(gate_proj(x)))
        """
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        return self.down_proj(gate_output * up_output), None  # Return None for compatibility


class NeuronGlm4DecoderLayer(nn.Module):
    """
    GLM-4 Decoder Layer implementation for NeuronX.
    
    The GLM-4-9b-chat-hf model uses the GLM architecture (model_type="glm"), which has
    only 2 RMSNorm layers per decoder layer:
    - input_layernorm: Before attention
    - post_attention_layernorm: After first residual add, before MLP
    
    Note: The HuggingFace GLM4 code (model_type="glm4") shows 4 RMSNorm layers, but
    GLM-4-9b-chat-hf actually uses model_type="glm" which loads GlmForCausalLM from
    transformers.models.glm.modeling_glm - this architecture has only 2 norms.
    
    Reference: transformers/src/transformers/models/glm/modeling_glm.py - GlmDecoderLayer class
    """
    
    def __init__(self, config: Glm4InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = NeuronGlm4Attention(config)
        
        # MLP
        self.mlp = NeuronGlm4MLP(config)
        
        # 2 Layer norms (matching the actual checkpoint structure)
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
        Forward pass following standard pre-LN transformer pattern:
        
        1. residual = hidden_states
        2. hidden_states = input_layernorm(hidden_states)
        3. hidden_states = self_attn(hidden_states)
        4. hidden_states = residual + hidden_states
        
        5. residual = hidden_states
        6. hidden_states = post_attention_layernorm(hidden_states)
        7. hidden_states = mlp(hidden_states)
        8. hidden_states = residual + hidden_states
        """
        # First residual block (attention)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        hidden_states = attn_output.hidden_states
        present_key_value = attn_output.present_key_value
        cos_cache = attn_output.cos_cache
        sin_cache = attn_output.sin_cache
        
        # Residual add
        hidden_states = residual + hidden_states
        
        # Second residual block (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronGlm4Model(NeuronBaseModel):
    """
    GLM-4 base model implementation for NeuronX.
    
    This is the main transformer model without the language modeling head.
    """
    
    def setup_attr_for_model(self, config: Glm4InferenceConfig):
        """Setup attributes required for model initialization."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
    def init_model(self, config: Glm4InferenceConfig):
        """Initialize model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            tensor_model_parallel_group=get_tp_group(config),
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGlm4DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            tensor_model_parallel_group=get_tp_group(config),
        )


class NeuronGlm4ForCausalLM(NeuronBaseForCausalLM):
    """
    GLM-4 for Causal Language Modeling on NeuronX.
    
    This is the main entry point for inference, extending NeuronBaseForCausalLM
    with GLM-4 specific weight conversion and configuration.
    """
    
    _model_cls = NeuronGlm4Model
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace GLM-4 model for weight extraction."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace GLM-4 state dict to NeuronX format.
        
        Key transformations:
        1. Remove 'model.' prefix from keys
        2. Split fused gate_up_proj into separate gate_proj and up_proj weights
        3. Add rank utilities for tensor parallelism
        
        HuggingFace GLM-4 weight structure:
        - model.embed_tokens.weight
        - model.layers.{i}.input_layernorm.weight
        - model.layers.{i}.self_attn.q_proj.weight/bias
        - model.layers.{i}.self_attn.k_proj.weight/bias
        - model.layers.{i}.self_attn.v_proj.weight/bias
        - model.layers.{i}.self_attn.o_proj.weight
        - model.layers.{i}.post_self_attn_layernorm.weight
        - model.layers.{i}.post_attention_layernorm.weight
        - model.layers.{i}.mlp.gate_up_proj.weight  -> Split to gate_proj + up_proj
        - model.layers.{i}.mlp.down_proj.weight
        - model.layers.{i}.post_mlp_layernorm.weight
        - model.norm.weight
        - lm_head.weight
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        intermediate_size = config.intermediate_size
        
        for key, value in state_dict.items():
            # Remove 'model.' prefix
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            
            # Handle MLP gate_up_proj splitting
            if "mlp.gate_up_proj.weight" in new_key:
                # gate_up_proj is [2 * intermediate_size, hidden_size]
                # Split into gate_proj [intermediate_size, hidden_size] and up_proj [intermediate_size, hidden_size]
                gate_proj_weight = value[:intermediate_size, :].clone()
                up_proj_weight = value[intermediate_size:, :].clone()

                # Create new keys
                layer_prefix = new_key.replace("mlp.gate_up_proj.weight", "")
                neuron_state_dict[f"{layer_prefix}mlp.gate_proj.weight"] = gate_proj_weight
                neuron_state_dict[f"{layer_prefix}mlp.up_proj.weight"] = up_proj_weight
            # Handle attention projection weights - add qkv_proj prefix for NeuronAttentionBase
            elif "self_attn.q_proj." in new_key:
                new_key = new_key.replace("self_attn.q_proj.", "self_attn.qkv_proj.q_proj.")
                neuron_state_dict[new_key] = value.clone()
            elif "self_attn.k_proj." in new_key:
                new_key = new_key.replace("self_attn.k_proj.", "self_attn.qkv_proj.k_proj.")
                neuron_state_dict[new_key] = value.clone()
            elif "self_attn.v_proj." in new_key:
                new_key = new_key.replace("self_attn.v_proj.", "self_attn.qkv_proj.v_proj.")
                neuron_state_dict[new_key] = value.clone()
            # Handle output projection weight - add nested o_proj prefix for GroupQueryAttention_O
            elif "self_attn.o_proj." in new_key:
                new_key = new_key.replace("self_attn.o_proj.", "self_attn.o_proj.o_proj.")
                neuron_state_dict[new_key] = value.clone()
            else:
                neuron_state_dict[new_key] = value.clone()
        
        # Add rank utilities for tensor parallelism
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utility for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        gc.collect()
        return neuron_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embed_tokens and lm_head."""
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the config class for GLM-4."""
        return Glm4InferenceConfig
    
    def get_compiler_args(self):
        """Return compiler arguments optimized for GLM-4."""
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        return compiler_args


# Module exports
__all__ = [
    "Glm4InferenceConfig",
    "Glm4NeuronConfig",
    "NeuronGlm4Attention",
    "NeuronGlm4MLP",
    "NeuronGlm4DecoderLayer",
    "NeuronGlm4Model",
    "NeuronGlm4ForCausalLM",
]
