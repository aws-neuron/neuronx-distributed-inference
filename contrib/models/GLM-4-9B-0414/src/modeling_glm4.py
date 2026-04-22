# coding=utf-8
# Copyright 2025 The GLM4 & ZhipuAI team and NeuronX Distributed Inference port.
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
PyTorch GLM-4-9B-0414 model for NeuronX Distributed Inference (model_type="glm4").

Key architectural features vs glm-4-9b-chat-hf (model_type="glm"):
- 4 RMSNorm layers per decoder layer (input, post_self_attn, post_attention, post_mlp)
- Same GQA (32 Q heads, 2 KV heads), partial RoPE (factor=0.5), SiLU MLP
- Fused gate_up_proj in HF weights, split into gate_proj + up_proj for TP
"""

import gc
import os
import json
from typing import List, Optional, Tuple, Type

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
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
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
    GLM-4 rotary embedding with partial_rotary_factor and interleaved cos/sin pattern.
    Only a portion of head_dim (default 0.5) gets rotary embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        partial_rotary_factor: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor
        
        # Calculate the rotary dimension (half of head_dim for GLM-4)
        self.rotary_dim = int(dim * partial_rotary_factor)
        
        # Compute inverse frequencies
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
    """Interleaved rotation: stack(-x[..., 1::2], x[..., 0::2])."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_glm4_rotary_pos_emb(q, k, cos, sin, partial_rotary_factor=0.5):
    """Apply partial rotary embeddings to Q and K (first rotary_dim dims only)."""
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
        self.fused_qkv = True  # Use fused QKV for simpler weight handling


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
            "num_key_value_heads": params.get("multi_query_group_num", params.get("num_key_value_heads", 2)),
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
    """GLM-4 attention with QKV bias and partial rotary embeddings."""
    
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
    """SwiGLU MLP with separate gate/up projections for tensor parallelism."""
    
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
    GLM-4 decoder layer with 4 RMSNorm layers per block.

    Unlike glm-4-9b-chat-hf (model_type="glm", 2 norms), GLM-4-9B-0414
    (model_type="glm4") has post_self_attn_layernorm and post_mlp_layernorm.
    """
    
    def __init__(self, config: Glm4InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = NeuronGlm4Attention(config)
        
        # MLP
        self.mlp = NeuronGlm4MLP(config)
        
        # 4 Layer norms (matching HF GLM-4 architecture)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_self_attn_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_mlp_layernorm = get_rmsnorm_cls()(
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
        Forward pass following GLM-4's architecture with 4 layer norms:
        
        1. hidden_states = input_layernorm(hidden_states)
        2. hidden_states = self_attn(hidden_states)
        3. hidden_states = post_self_attn_layernorm(hidden_states)
        4. hidden_states = residual + hidden_states
        
        5. hidden_states = post_attention_layernorm(hidden_states)
        6. hidden_states = mlp(hidden_states)
        7. hidden_states = post_mlp_layernorm(hidden_states)
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
        
        # Post-attention layernorm and residual add
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        # Second residual block (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_mlp_layernorm(hidden_states)
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
        
        GLM-4-9B-0414 has separate Q/K/V projections that need to be fused into Wqkv.
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        head_dim = hidden_size // num_attention_heads
        
        # Collect Q/K/V weights per layer for fusion
        qkv_weights = {}
        qkv_biases = {}
        
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            
            # Collect separate Q/K/V for fusion
            if "self_attn.q_proj.weight" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_weights.setdefault(layer_idx, {})["q"] = value
            elif "self_attn.k_proj.weight" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_weights.setdefault(layer_idx, {})["k"] = value
            elif "self_attn.v_proj.weight" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_weights.setdefault(layer_idx, {})["v"] = value
            elif "self_attn.q_proj.bias" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_biases.setdefault(layer_idx, {})["q"] = value
            elif "self_attn.k_proj.bias" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_biases.setdefault(layer_idx, {})["k"] = value
            elif "self_attn.v_proj.bias" in new_key:
                layer_idx = new_key.split(".")[1]
                qkv_biases.setdefault(layer_idx, {})["v"] = value
            elif "self_attn.o_proj.weight" in new_key:
                neuron_state_dict[new_key] = value.clone()
            elif "mlp.gate_up_proj.weight" in new_key:
                # Split fused gate_up_proj into separate gate_proj and up_proj
                gate_proj_weight = value[:intermediate_size, :].clone()
                up_proj_weight = value[intermediate_size:, :].clone()
                layer_prefix = new_key.replace("mlp.gate_up_proj.weight", "")
                neuron_state_dict[f"{layer_prefix}mlp.gate_proj.weight"] = gate_proj_weight
                neuron_state_dict[f"{layer_prefix}mlp.up_proj.weight"] = up_proj_weight
            elif "mlp.down_proj.weight" in new_key:
                neuron_state_dict[new_key] = value.clone()
            elif "embed_tokens.weight" in new_key:
                neuron_state_dict["embed_tokens.weight"] = value.clone()
            elif "lm_head.weight" in new_key:
                neuron_state_dict["lm_head.weight"] = value.clone()
            elif "norm.weight" in new_key and "layers" not in new_key:
                neuron_state_dict["norm.weight"] = value.clone()
            elif "input_layernorm" in new_key or "post_attention_layernorm" in new_key or "post_self_attn_layernorm" in new_key or "post_mlp_layernorm" in new_key:
                neuron_state_dict[new_key] = value.clone()
        
        # Fuse Q/K/V weights into Wqkv format: [Q, K, V] concatenated
        for layer_idx, qkv in qkv_weights.items():
            q_weight = qkv["q"]  # [num_heads * head_dim, hidden_size]
            k_weight = qkv["k"]  # [num_kv_heads * head_dim, hidden_size]
            v_weight = qkv["v"]  # [num_kv_heads * head_dim, hidden_size]
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            neuron_state_dict[f"layers.{layer_idx}.self_attn.Wqkv.weight"] = fused_weight
        
        for layer_idx, qkv in qkv_biases.items():
            q_bias = qkv["q"]
            k_bias = qkv["k"]
            v_bias = qkv["v"]
            fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            neuron_state_dict[f"layers.{layer_idx}.self_attn.Wqkv.bias"] = fused_bias
        
        # Add rank utilities
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
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
