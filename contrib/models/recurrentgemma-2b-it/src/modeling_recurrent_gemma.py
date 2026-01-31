# coding=utf-8
# Copyright 2024 Google Inc. and the HuggingFace Inc. team. All rights reserved.
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
NeuronX RecurrentGemma model implementation.

RecurrentGemma is a hybrid architecture combining:
1. Recurrent blocks (RG-LRU - Real-Gated Linear Recurrent Unit)
2. Attention blocks (SDPA attention with GQA)

The model alternates between these block types according to a pattern
defined in config.block_types (default: ['recurrent', 'recurrent', 'attention']).

Key differences from standard transformers:
- Recurrent blocks use RG-LRU instead of attention (similar to Mamba)
- Partial rotary embeddings (only 50% of head_dim gets RoPE)
- Logits soft-capping at 30.0
- RMSNorm uses (1 + weight) scaling

Reference:
- Original HuggingFace implementation in:
"""

import json
import math
import os
from typing import List, Optional, Type, Tuple, Union

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
from transformers import AutoModelForCausalLM

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.utils.distributed import get_tp_group


# ==============================================================================
# Configuration
# ==============================================================================

class RecurrentGemmaInferenceConfig(InferenceConfig):
    """
    Configuration for RecurrentGemma inference on NeuronX.
    
    Extends InferenceConfig with RecurrentGemma-specific parameters.
    """
    
    def __init__(self, **kwargs):
        # RecurrentGemma-specific parameters
        self.lru_width = kwargs.pop("lru_width", None)
        self.attention_window_size = kwargs.pop("attention_window_size", 2048)
        self.conv1d_width = kwargs.pop("conv1d_width", 4)
        self.logits_soft_cap = kwargs.pop("logits_soft_cap", 30.0)
        self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.5)
        self.block_types = kwargs.pop("block_types", ["recurrent", "recurrent", "attention"])
        self.hidden_activation = kwargs.pop("hidden_activation", "gelu_pytorch_tanh")
        self.w_init_variance_scale = kwargs.pop("w_init_variance_scale", 0.01)
        self.final_w_init_variance_scale = kwargs.pop("final_w_init_variance_scale", None)
        self.embeddings_scale_by_sqrt_dim = kwargs.pop("embeddings_scale_by_sqrt_dim", True)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        
        # HuggingFace-style config attributes expected by the framework
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.use_cache = kwargs.pop("use_cache", True)
        
        super().__init__(**kwargs)
        
        # Set default lru_width if not provided
        if self.lru_width is None:
            self.lru_width = self.hidden_size
            
        # Compute final_w_init_variance_scale if not provided
        if self.final_w_init_variance_scale is None:
            self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # Generate the layer block type pattern
        self.layers_block_type = (self.block_types * 100)[:self.num_hidden_layers]
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "rms_norm_eps",
            "intermediate_size",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "RecurrentGemmaInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional arguments to override configuration
            
        Returns:
            RecurrentGemmaInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace config to NeuronX config
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 2560),
            "num_attention_heads": hf_config.get("num_attention_heads", 10),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 26),
            "num_key_value_heads": hf_config.get("num_key_value_heads", 1),
            "vocab_size": hf_config.get("vocab_size", 256000),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "intermediate_size": hf_config.get("intermediate_size", 15360),
            "pad_token_id": hf_config.get("pad_token_id", 0),
            "bos_token_id": hf_config.get("bos_token_id", 2),
            "eos_token_id": hf_config.get("eos_token_id", 1),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "max_position_embeddings": hf_config.get("attention_window_size", 2048),
            # RecurrentGemma-specific
            "lru_width": hf_config.get("lru_width", hf_config.get("hidden_size", 2560)),
            "attention_window_size": hf_config.get("attention_window_size", 2048),
            "conv1d_width": hf_config.get("conv1d_width", 4),
            "logits_soft_cap": hf_config.get("logits_soft_cap", 30.0),
            "partial_rotary_factor": hf_config.get("partial_rotary_factor", 0.5),
            "block_types": hf_config.get("_block_types", ["recurrent", "recurrent", "attention"]),
            "hidden_activation": hf_config.get("hidden_activation", "gelu_pytorch_tanh"),
            "w_init_variance_scale": hf_config.get("w_init_variance_scale", 0.01),
            "final_w_init_variance_scale": hf_config.get("final_w_init_variance_scale"),
            "embeddings_scale_by_sqrt_dim": hf_config.get("embeddings_scale_by_sqrt_dim", True),
            "attention_bias": hf_config.get("attention_bias", False),
            "attention_dropout": hf_config.get("attention_dropout", 0.0),
            "head_dim": hf_config.get("head_dim", hf_config.get("hidden_size", 2560) // hf_config.get("num_attention_heads", 10)),
            # RecurrentGemma uses tied weights (lm_head shares embed_tokens)
            "tie_word_embeddings": True,
        }
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


# ==============================================================================
# Normalization
# ==============================================================================

class RecurrentGemmaRMSNorm(nn.Module):
    """
    RecurrentGemma RMSNorm implementation.
    
    Differs from standard RMSNorm by using (1 + weight) scaling:
    output = norm(x) * (1.0 + weight)
    
    Reference: modeling_recurrent_gemma.py RecurrentGemmaRMSNorm
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize weight to zeros (so effective scaling is 1.0)
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        # RecurrentGemma uses (x * w).to(float16) pattern, not x.to(float16) * w
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm class based on execution mode.
    
    For NeuronX inference, we use the standard RecurrentGemmaRMSNorm for now
    since it has the (1 + weight) scaling that differs from CustomRMSNorm.
    """
    # TODO: Consider implementing a Neuron-optimized version if needed
    return RecurrentGemmaRMSNorm


# ==============================================================================
# Activation Functions
# ==============================================================================

def gelu_pytorch_tanh(x):
    """
    GELU activation with tanh approximation (matches PyTorch's gelu with approximate='tanh').
    """
    return F.gelu(x, approximate='tanh')


ACT2FN = {
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "gelu": F.gelu,
    "silu": F.silu,
    "relu": F.relu,
}


# ==============================================================================
# Rotary Embeddings
# ==============================================================================

class RecurrentGemmaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for RecurrentGemma.
    
    RecurrentGemma uses partial rotary embeddings - only a fraction of the head_dim
    gets RoPE applied (controlled by partial_rotary_factor).
    """
    
    def __init__(self, dim: int, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute rotary embeddings for given positions.
        
        Args:
            x: Input tensor [batch, num_heads, seq_len, head_size]
            position_ids: Position indices [batch, seq_len]
            
        Returns:
            cos, sin: Rotary embedding tensors
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA.
    
    hidden_states: [batch, num_key_value_heads, seqlen, head_dim]
    -> [batch, num_attention_heads, seqlen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ==============================================================================
# MLP
# ==============================================================================

class RecurrentGemmaMLP(nn.Module):
    """
    RecurrentGemma MLP layer.
    
    Uses gated activation: gate_proj -> activation -> multiply with up_proj -> down_proj
    Note: Uses intermediate_size // 2 for gate_proj and up_proj dimensions.
    All linear layers have bias=True.
    """
    
    def __init__(self, config: RecurrentGemmaInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # RecurrentGemma MLP uses intermediate_size // 2
        self.intermediate_size = config.intermediate_size // 2
        self.act_fn = ACT2FN[config.hidden_activation]
        
        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


# ==============================================================================
# Attention
# ==============================================================================

class RecurrentGemmaSdpaAttention(nn.Module):
    """
    RecurrentGemma SDPA Attention with partial rotary embeddings.
    
    Key features:
    - Grouped Query Attention (GQA) support
    - Partial rotary embeddings (only partial_rotary_factor of head_dim gets RoPE)
    - Uses scaled dot product attention
    
    Note: For token generation with KV cache, we store the full key/value states
    without rotary embeddings applied, then apply RoPE to the rotary portion during attention.
    However, for simplicity in this initial port, we handle context encoding only
    (the attention layers are only 1/3 of the layers anyway).
    """
    
    def __init__(self, config: RecurrentGemmaInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.partial_rotary_factor = config.partial_rotary_factor
        self.attention_dropout = config.attention_dropout
        
        # Compute rotary embedding dimension
        rotary_dim = int(self.partial_rotary_factor * self.head_dim)
        
        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
                bias=config.attention_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            # Adjust head counts for TP
            tp_degree = config.neuron_config.tp_degree
            self.num_attention_heads_per_partition = self.num_attention_heads // tp_degree
            self.num_key_value_heads_per_partition = max(1, self.num_key_value_heads // tp_degree)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
            self.num_attention_heads_per_partition = self.num_attention_heads
            self.num_key_value_heads_per_partition = self.num_key_value_heads
        
        self.rotary_emb = RecurrentGemmaRotaryEmbedding(
            rotary_dim,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the attention layer.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            cache_position: Optional cache position indices
            use_cache: Whether to use KV cache
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention: [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads_per_partition, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
        
        # Compute rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # Apply partial rotary embeddings
        # Split into rotary and non-rotary parts
        rotary_dim = int(self.partial_rotary_factor * self.head_dim)
        query_rot = query_states[..., :rotary_dim]
        query_pass = query_states[..., rotary_dim:]
        key_rot = key_states[..., :rotary_dim]
        key_pass = key_states[..., rotary_dim:]
        
        # Apply RoPE to rotary portion
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        
        # Concatenate back
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        
        # Repeat KV for GQA
        num_kv_groups = self.num_attention_heads_per_partition // self.num_key_value_heads_per_partition
        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)
        
        # Attention computation using scaled_dot_product_attention
        # Create causal mask if needed
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        
        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.head_dim ** -0.5,
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# ==============================================================================
# RG-LRU (Real-Gated Linear Recurrent Unit)
# ==============================================================================

class RecurrentGemmaRglru(nn.Module):
    """
    Real-Gated Linear Recurrent Unit (RG-LRU) layer.
    
    This is the core recurrent component of RecurrentGemma.
    It implements a gated linear recurrence with:
    - Input gate and recurrent gate (both with learned parameters)
    - Diagonal recurrence matrix (learned)
    
    For autoregressive generation, the recurrent_states must persist across forward calls.
    
    Reference: modeling_recurrent_gemma.py RecurrentGemmaRglru
    """
    
    def __init__(self, config: RecurrentGemmaInferenceConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.lru_width = config.lru_width
        self.block_width = config.lru_width // self.num_attention_heads
        
        # Recurrent parameter (diagonal of the recurrence matrix)
        self.recurrent_param = nn.Parameter(torch.empty(config.lru_width))
        
        # Input gate parameters
        self.input_gate_weight = nn.Parameter(
            torch.empty(self.num_attention_heads, self.block_width, self.block_width)
        )
        self.input_gate_bias = nn.Parameter(
            torch.empty(self.num_attention_heads, self.block_width)
        )
        
        # Recurrent gate parameters
        self.recurrent_gate_weight = nn.Parameter(
            torch.empty(self.num_attention_heads, self.block_width, self.block_width)
        )
        self.recurrent_gate_bias = nn.Parameter(
            torch.empty(self.num_attention_heads, self.block_width)
        )
        
        # Recurrent states (will be set externally for state persistence)
        self.recurrent_states = None
    
    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the RG-LRU.
        
        Args:
            activations: [batch, seq_len, lru_width]
            position_ids: [batch, seq_len]
            
        Returns:
            [batch, seq_len, lru_width]
        """
        batch_size, seq_len, lru_width = activations.shape
        
        # Reset indicator - when position is 0, reset the state
        reset = position_ids[:, :, None] == 0
        
        # Reshape for batch matrix multiplication
        # [batch * seq_len, num_heads, block_width]
        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.permute(1, 0, 2)  # [num_heads, batch * seq_len, block_width]
        
        # Compute input gate: sigmoid(W_input @ x + b_input)
        res = torch.baddbmm(self.input_gate_bias[:, None, :], reshape_act, self.input_gate_weight)
        input_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))
        
        # Compute recurrent gate: sigmoid(W_recurrent @ x + b_recurrent)
        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))
        
        # Compute the parameter `A` of the recurrence
        # log_recurrent_gate = -8.0 * recurrent_gate * softplus(recurrent_param)
        log_recurrent_gate = -8.0 * recurrent_gate * F.softplus(self.recurrent_param)
        recurrent_gate_a = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)
        
        # Gate the input
        gated_inputs = activations * input_gate
        
        # Apply gamma normalization
        # multiplier = sqrt(1 - a^2) for gamma normalization
        multiplier = torch.sqrt(torch.clamp(1 - a_square, min=1e-10))
        multiplier = reset.float() + (~reset).float() * multiplier
        normalized_x = gated_inputs * multiplier.type(activations.dtype)
        
        # Run the RNN scan
        hidden_states, recurrent_states = self._rnn_scan(
            hidden_states=normalized_x,
            recurrent_gate=recurrent_gate_a,
            reset=reset,
            recurrent_states=self.recurrent_states,
        )
        
        # Store states for next forward pass
        self.recurrent_states = recurrent_states
        
        return hidden_states
    
    def _rnn_scan(
        self,
        hidden_states: torch.Tensor,
        recurrent_gate: torch.Tensor,
        reset: torch.Tensor,
        recurrent_states: Optional[torch.Tensor],
        acc_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the recurrence of the linear RNN.
        
        Args:
            hidden_states: [batch, seq_len, lru_width] - input sequence
            recurrent_gate: [batch, seq_len, lru_width] - diagonal of recurrence matrix A
            reset: [batch, seq_len, 1] - document boundary indicators
            recurrent_states: [batch, lru_width] - initial hidden state
            acc_dtype: Accumulation dtype
            
        Returns:
            output: [batch, seq_len, lru_width]
            final_state: [batch, lru_width]
        """
        # Multiply recurrent_gate by ~reset to reset state at document boundaries
        recurrent_gate = recurrent_gate * (~reset).float()
        
        if hidden_states.shape[1] == 1:
            # Token generation mode (seq_len == 1)
            if recurrent_states is None:
                return hidden_states, hidden_states[:, 0].type(acc_dtype)
            else:
                contextualized_states = recurrent_gate.type(acc_dtype) * recurrent_states[:, None].to(recurrent_gate.device)
                contextualized_states = contextualized_states + hidden_states.type(acc_dtype)
                return contextualized_states.type(hidden_states.dtype), contextualized_states[:, -1]
        else:
            # Context encoding mode (seq_len > 1)
            if recurrent_states is None:
                recurrent_states = torch.zeros(hidden_states[:, 0].shape, dtype=acc_dtype, device=hidden_states.device)
            
            contextualized_states = torch.zeros_like(hidden_states)
            for t in range(hidden_states.shape[1]):
                recurrent_states = recurrent_gate[:, t].type(acc_dtype) * recurrent_states
                recurrent_states = recurrent_states + hidden_states[:, t].type(acc_dtype)
                contextualized_states[:, t] = recurrent_states.type(hidden_states.dtype)
            
            return contextualized_states, recurrent_states


# ==============================================================================
# Recurrent Block
# ==============================================================================

class RecurrentGemmaRecurrentBlock(nn.Module):
    """
    RecurrentGemma Recurrent Block (Griffin/Hawk style).
    
    Architecture:
    1. Linear Y: hidden_size -> lru_width (with activation)
    2. Linear X: hidden_size -> lru_width
    3. Conv1D: lru_width (depthwise, causal)
    4. RG-LRU: recurrent processing
    5. Output: (rg_lru_output * y_branch), then linear_out
    
    Reference: modeling_recurrent_gemma.py RecurrentGemmaRecurrentBlock
    """
    
    def __init__(self, config: RecurrentGemmaInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.lru_width = config.lru_width
        self.hidden_size = config.hidden_size
        self.conv1d_width = config.conv1d_width
        self.act_fn = ACT2FN[config.hidden_activation]
        
        # RecurrentGemma recurrent block linear layers have bias=True
        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.linear_y = ColumnParallelLinear(
                config.hidden_size,
                config.lru_width,
                bias=True,  # Has bias
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.linear_x = ColumnParallelLinear(
                config.hidden_size,
                config.lru_width,
                bias=True,  # Has bias
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.linear_out = ColumnParallelLinear(
                config.lru_width,
                config.hidden_size,
                bias=True,  # Has bias
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.linear_y = nn.Linear(config.hidden_size, config.lru_width, bias=True)
            self.linear_x = nn.Linear(config.hidden_size, config.lru_width, bias=True)
            self.linear_out = nn.Linear(config.lru_width, config.hidden_size, bias=True)
        
        # Depthwise Conv1D
        self.conv_1d = nn.Conv1d(
            config.lru_width,
            config.lru_width,
            kernel_size=config.conv1d_width,
            groups=config.lru_width,
            padding=config.conv1d_width - 1,  # Causal padding
        )
        
        # RG-LRU
        self.rg_lru = RecurrentGemmaRglru(config)
        
        # Conv1D state for token generation
        self.conv1d_state = None
    
    def forward(
        self,
        input_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the recurrent block.
        
        Args:
            input_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            attention_mask: Attention mask (not used for recurrent)
            cache_position: Cache position indices
            use_cache: Whether to use/update state cache
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = input_states.shape
        
        # Y branch: linear + activation
        y_branch = self.linear_y(input_states)
        y_branch = self.act_fn(y_branch)
        
        # X branch: linear
        x_branch = self.linear_x(input_states)
        x_branch = x_branch.transpose(1, 2)  # [batch, lru_width, seq_len]
        
        # Apply Conv1D - always use the full convolution approach for tracing
        # The conditional decoding path causes issues with XLA tracing
        # For context encoding (seq_len > 1): standard convolution
        # For token generation (seq_len == 1): we initialize and update state
        
        if seq_len > 1:
            # Context encoding / prefill mode
            # Apply causal convolution
            x_branch = self.conv_1d(x_branch)[..., :seq_len]
            # Initialize state for future token generation
            if use_cache:
                self.conv1d_state = F.pad(x_branch, (self.conv1d_width - seq_len - 1, 0))
        else:
            # Token generation mode - but handle None state for tracing
            if self.conv1d_state is None:
                # Initialize conv1d state with zeros for tracing
                self.conv1d_state = torch.zeros(
                    batch_size, self.lru_width, self.conv1d_width - 1,
                    device=x_branch.device, dtype=x_branch.dtype
                )
            conv_state = torch.cat((self.conv1d_state, x_branch), -1)
            x_branch = torch.sum(conv_state * self.conv_1d.weight[:, 0, :], dim=-1) + self.conv_1d.bias
            x_branch = x_branch.unsqueeze(-1)
            self.conv1d_state = conv_state[:, :, 1:]
        
        x_branch = x_branch.transpose(1, 2)  # [batch, seq_len, lru_width]
        
        # Apply RG-LRU
        x_branch = self.rg_lru(x_branch, position_ids)
        
        # Combine branches and output
        hidden_states = x_branch * y_branch
        hidden_states = self.linear_out(hidden_states)
        
        return hidden_states
    
    def _setup_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Initialize cache states for token generation.
        """
        # Recurrent states are always computed in full precision
        self.rg_lru.recurrent_states = torch.zeros(
            (batch_size, self.lru_width), device=device, dtype=torch.float32
        )
        self.conv1d_state = torch.zeros(
            (batch_size, self.lru_width, self.conv1d_width - 1), device=device, dtype=dtype
        )


# ==============================================================================
# Temporal Block Mapping
# ==============================================================================

TEMPORAL_BLOCK_CLASSES = {
    "recurrent": RecurrentGemmaRecurrentBlock,
    "attention": RecurrentGemmaSdpaAttention,
}


# ==============================================================================
# Decoder Layer
# ==============================================================================

class RecurrentGemmaDecoderLayer(nn.Module):
    """
    RecurrentGemma Decoder Layer.
    
    Each layer consists of:
    1. temporal_pre_norm (RMSNorm)
    2. temporal_block (either RecurrentBlock or Attention)
    3. channel_pre_norm (RMSNorm)
    4. mlp_block
    
    Reference: modeling_recurrent_gemma.py RecurrentGemmaDecoderLayer
    """
    
    def __init__(self, config: RecurrentGemmaInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Determine block type for this layer
        block_type = config.layers_block_type[layer_idx]
        self.is_attention_layer = (block_type == "attention")
        
        # Normalization layers
        self.temporal_pre_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.channel_pre_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Temporal block (attention or recurrent)
        self.temporal_block = TEMPORAL_BLOCK_CLASSES[block_type](config, layer_idx)
        
        # MLP
        self.mlp_block = RecurrentGemmaMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,  # Accept additional arguments from framework
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: [batch, seq_len]
            past_key_value: Past KV cache (for attention layers)
            adapter_ids: Adapter IDs for LoRA
            rotary_position_ids: Position IDs for rotary embeddings
            residual: Residual from previous layer
            **kwargs: Additional arguments from framework (seq_ids, etc.)
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        raw_activations = hidden_states
        
        # Extract cache_position from kwargs if available
        cache_position = kwargs.get('cache_position', None)
        if cache_position is None:
            cache_position = position_ids[0] if position_ids is not None else torch.arange(hidden_states.shape[1], device=hidden_states.device)
        
        # Check if this is for context encoding or token generation
        is_for_context_encoding = kwargs.get('is_for_context_encoding', hidden_states.shape[1] > 1)
        
        # First normalize
        inputs_normalized = self.temporal_pre_norm(raw_activations)
        
        # Temporal block (attention or recurrent)
        if self.is_attention_layer:
            # For attention layers, we need position_ids
            temporal_output = self.temporal_block(
                inputs_normalized,
                position_ids if position_ids is not None else torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1),
                attention_mask,
                cache_position=cache_position,
                use_cache=not is_for_context_encoding,
            )
        else:
            # For recurrent layers
            temporal_output = self.temporal_block(
                inputs_normalized,
                position_ids if position_ids is not None else torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1),
                attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
        
        # First residual
        residual_out = temporal_output + raw_activations
        
        # Second normalize
        mlp_input = self.channel_pre_norm(residual_out)
        
        # MLP
        mlp_output = self.mlp_block(mlp_input)
        
        # Second residual
        hidden_states_out = mlp_output + residual_out
        
        # Return format expected by NeuronBaseModel.get_model_output:
        # (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        # For RecurrentGemma, we return dummy KV cache tensors for compatibility
        # The actual state is managed internally by the recurrent blocks
        batch_size = hidden_states_out.shape[0]
        device = hidden_states_out.device
        dtype = hidden_states_out.dtype
        
        # Create dummy KV cache tensors with correct shape
        # Shape should be [batch, num_kv_heads, seq_len, head_dim]
        seq_len = hidden_states_out.shape[1]
        dummy_k = torch.zeros(batch_size, self.config.num_key_value_heads, seq_len, 
                              self.config.hidden_size // self.config.num_attention_heads,
                              device=device, dtype=dtype)
        dummy_v = torch.zeros_like(dummy_k)
        
        return (hidden_states_out, (dummy_k, dummy_v), None, None, None)
    
    def _setup_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Setup cache for token generation."""
        if not self.is_attention_layer:
            self.temporal_block._setup_cache(batch_size, device, dtype)


# ==============================================================================
# Main Model
# ==============================================================================

class NeuronRecurrentGemmaModel(NeuronBaseModel):
    """
    NeuronX RecurrentGemma Model.
    
    This is the base transformer model without the language modeling head.
    """
    
    def setup_attr_for_model(self, config: RecurrentGemmaInferenceConfig):
        """Setup model attributes required by NeuronBaseModel."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: RecurrentGemmaInferenceConfig):
        """Initialize the model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings_scale_by_sqrt_dim = config.embeddings_scale_by_sqrt_dim
        
        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            RecurrentGemmaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm - MUST be named 'norm' to match NeuronBaseModel expectation
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Logits soft cap
        self.logits_soft_cap = config.logits_soft_cap


class NeuronRecurrentGemmaForCausalLM(NeuronBaseForCausalLM):
    """
    NeuronX RecurrentGemma for Causal Language Modeling.
    
    This class wraps NeuronRecurrentGemmaModel and adds the language modeling head.
    """
    
    _model_cls = NeuronRecurrentGemmaModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace model for weight extraction."""
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: RecurrentGemmaInferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.
        
        IMPORTANT: The framework strips 'model.' prefix before calling this method,
        so we receive keys like 'embed_tokens.weight' not 'model.embed_tokens.weight'.
        
        Also handles tied weights (lm_head = embed_tokens).
        """
        neuron_state_dict = {}
        
        # Embeddings - framework already stripped 'model.' prefix
        # RecurrentGemma scales embeddings by sqrt(hidden_size)
        if "embed_tokens.weight" in state_dict:
            embed_weight = state_dict["embed_tokens.weight"].clone()
            if config.embeddings_scale_by_sqrt_dim:
                normalizer = config.hidden_size ** 0.5
                embed_weight = embed_weight * normalizer
            neuron_state_dict["embed_tokens.weight"] = embed_weight
        
        # Final norm - RecurrentGemma uses 'final_norm', we need 'norm'
        if "final_norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["final_norm.weight"].clone()
        
        # LM head - tied to embeddings in RecurrentGemma
        # IMPORTANT: lm_head should NOT be scaled (only embeddings are scaled)
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
        elif "embed_tokens.weight" in state_dict:
            # Tie weights - use UNSCALED embeddings for lm_head
            neuron_state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
            # Tie weights if lm_head not present
            neuron_state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        
        # Decoder layers - keys already have 'model.' stripped
        for i in range(config.num_hidden_layers):
            prefix = f"layers.{i}"
            
            # Layer norms
            for norm_name in ["temporal_pre_norm", "channel_pre_norm"]:
                hf_key = f"{prefix}.{norm_name}.weight"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # MLP
            for mlp_layer in ["gate_proj", "up_proj", "down_proj"]:
                for param in ["weight", "bias"]:
                    hf_key = f"{prefix}.mlp_block.{mlp_layer}.{param}"
                    if hf_key in state_dict:
                        neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            # Determine block type
            block_type = config.layers_block_type[i]
            
            if block_type == "attention":
                # Attention layers - q, k, v have no bias, o has bias
                for attn_layer in ["q_proj", "k_proj", "v_proj"]:
                    hf_key = f"{prefix}.temporal_block.{attn_layer}.weight"
                    if hf_key in state_dict:
                        neuron_state_dict[hf_key] = state_dict[hf_key].clone()
                
                # o_proj has both weight and bias
                for param in ["weight", "bias"]:
                    hf_key = f"{prefix}.temporal_block.o_proj.{param}"
                    if hf_key in state_dict:
                        neuron_state_dict[hf_key] = state_dict[hf_key].clone()
            
            else:
                # Recurrent layers
                # Linear projections (all have bias)
                for linear_name in ["linear_y", "linear_x", "linear_out"]:
                    for param in ["weight", "bias"]:
                        hf_key = f"{prefix}.temporal_block.{linear_name}.{param}"
                        if hf_key in state_dict:
                            neuron_state_dict[hf_key] = state_dict[hf_key].clone()
                
                # Conv1d
                for param in ["weight", "bias"]:
                    hf_key = f"{prefix}.temporal_block.conv_1d.{param}"
                    if hf_key in state_dict:
                        neuron_state_dict[hf_key] = state_dict[hf_key].clone()
                
                # RG-LRU parameters
                hf_key = f"{prefix}.temporal_block.rg_lru.recurrent_param"
                if hf_key in state_dict:
                    neuron_state_dict[hf_key] = state_dict[hf_key].clone()
                
                for gate_name in ["input_gate", "recurrent_gate"]:
                    for param in ["weight", "bias"]:
                        hf_key = f"{prefix}.temporal_block.rg_lru.{gate_name}_{param}"
                        if hf_key in state_dict:
                            neuron_state_dict[hf_key] = state_dict[hf_key].clone()
        
        # Add rank utility for TP
        tp_degree = config.neuron_config.tp_degree
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights - lm_head is tied to embed_tokens."""
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the config class to use."""
        return RecurrentGemmaInferenceConfig


# Export all public classes
__all__ = [
    "RecurrentGemmaInferenceConfig",
    "NeuronRecurrentGemmaModel",
    "NeuronRecurrentGemmaForCausalLM",
    "RecurrentGemmaRMSNorm",
    "RecurrentGemmaMLP",
    "RecurrentGemmaSdpaAttention",
    "RecurrentGemmaRecurrentBlock",
    "RecurrentGemmaRglru",
    "RecurrentGemmaDecoderLayer",
]
