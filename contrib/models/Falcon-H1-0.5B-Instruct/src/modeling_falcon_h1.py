# coding=utf-8
# Copyright 2025 Technology Innovation Institute and the HuggingFace Inc. team. All rights reserved.
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
Falcon-H1 model implementation for NeuronX Distributed Inference.

This is a hybrid Mamba2 + Attention architecture with MLP.
Based on the transformers implementation at:

"""

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


# ==============================================================================
# Configuration Class
# ==============================================================================

class FalconH1InferenceConfig(InferenceConfig):
    """
    Inference configuration for Falcon-H1 model.
    Maps HuggingFace config attributes to NeuronX config.
    """
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load configuration from a HuggingFace model path.
        
        Args:
            model_path: Path to the HuggingFace model directory containing config.json
            neuron_config: NeuronConfig instance for NeuronX settings
            **kwargs: Additional configuration overrides
            
        Returns:
            FalconH1InferenceConfig instance
        """
        import json
        import os
        
        # Load HuggingFace config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Create a custom load_config function to populate attributes from HF config
        def load_hf_config(self):
            # Core model parameters
            self.hidden_size = hf_config.get("hidden_size", 1024)
            self.intermediate_size = hf_config.get("intermediate_size", 2048)
            self.num_hidden_layers = hf_config.get("num_hidden_layers", 36)
            self.num_attention_heads = hf_config.get("num_attention_heads", 8)
            self.num_key_value_heads = hf_config.get("num_key_value_heads", 2)
            self.vocab_size = hf_config.get("vocab_size", 32784)
            self.max_position_embeddings = hf_config.get("max_position_embeddings", 16384)
            self.hidden_act = hf_config.get("hidden_act", "silu")
            self.rms_norm_eps = hf_config.get("rms_norm_eps", 1e-5)
            self.rope_theta = hf_config.get("rope_theta", 100000000000.0)
            self.rope_scaling = hf_config.get("rope_scaling", None)
            
            # Token IDs
            self.pad_token_id = hf_config.get("pad_token_id", 0)
            self.bos_token_id = hf_config.get("bos_token_id", 1)
            self.eos_token_id = hf_config.get("eos_token_id", 11)
            
            # Head dimension
            self.head_dim = hf_config.get("head_dim", self.hidden_size // self.num_attention_heads)
            
            # Attention settings
            self.attention_bias = hf_config.get("attention_bias", False)
            self.attention_dropout = hf_config.get("attention_dropout", 0.0)
            self.mlp_bias = hf_config.get("mlp_bias", False)
            self.projectors_bias = hf_config.get("projectors_bias", False)
            
            # Mamba-specific configurations
            self.mamba_d_ssm = hf_config.get("mamba_d_ssm", 1536)
            self.mamba_n_heads = hf_config.get("mamba_n_heads", 24)
            self.mamba_d_head = hf_config.get("mamba_d_head", 64)
            self.mamba_n_groups = hf_config.get("mamba_n_groups", 1)
            self.mamba_d_state = hf_config.get("mamba_d_state", 128)
            self.mamba_d_conv = hf_config.get("mamba_d_conv", 4)
            self.mamba_expand = hf_config.get("mamba_expand", 2)
            self.mamba_chunk_size = hf_config.get("mamba_chunk_size", 128)
            self.mamba_conv_bias = hf_config.get("mamba_conv_bias", True)
            self.mamba_proj_bias = hf_config.get("mamba_proj_bias", False)
            self.mamba_norm_before_gate = hf_config.get("mamba_norm_before_gate", False)
            self.mamba_rms_norm = hf_config.get("mamba_rms_norm", False)
            
            # MuP multipliers
            self.embedding_multiplier = hf_config.get("embedding_multiplier", 1.0)
            self.lm_head_multiplier = hf_config.get("lm_head_multiplier", 1.0)
            self.mlp_multipliers = hf_config.get("mlp_multipliers", [1.0, 1.0])
            self.attention_in_multiplier = hf_config.get("attention_in_multiplier", 1.0)
            self.attention_out_multiplier = hf_config.get("attention_out_multiplier", 1.0)
            self.key_multiplier = hf_config.get("key_multiplier", 1.0)
            self.ssm_multipliers = hf_config.get("ssm_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0])
            self.ssm_in_multiplier = hf_config.get("ssm_in_multiplier", 1.0)
            self.ssm_out_multiplier = hf_config.get("ssm_out_multiplier", 1.0)
            
            # Output settings required by base model
            self.output_attentions = hf_config.get("output_attentions", False)
            self.output_hidden_states = hf_config.get("output_hidden_states", False)
            self.use_cache = hf_config.get("use_cache", True)
            self.return_dict = hf_config.get("return_dict", True)
            self.tie_word_embeddings = hf_config.get("tie_word_embeddings", False)
        
        # Create the config instance
        if neuron_config is None:
            neuron_config = NeuronConfig()
            
        return cls(
            neuron_config=neuron_config,
            load_config=load_hf_config,
            **kwargs
        )
    
    def add_derived_config(self):
        """Add derived configuration attributes."""
        self.num_cores_per_group = 1
        
        # Mamba-specific configurations from HF config (these should already be set from from_pretrained)
        self.mamba_d_ssm = getattr(self, 'mamba_d_ssm', 1536)
        self.mamba_n_heads = getattr(self, 'mamba_n_heads', 24)
        self.mamba_d_head = getattr(self, 'mamba_d_head', 64)
        self.mamba_n_groups = getattr(self, 'mamba_n_groups', 1)
        self.mamba_d_state = getattr(self, 'mamba_d_state', 128)
        self.mamba_d_conv = getattr(self, 'mamba_d_conv', 4)
        self.mamba_chunk_size = getattr(self, 'mamba_chunk_size', 128)
        self.mamba_conv_bias = getattr(self, 'mamba_conv_bias', True)
        self.mamba_proj_bias = getattr(self, 'mamba_proj_bias', False)
        self.mamba_norm_before_gate = getattr(self, 'mamba_norm_before_gate', False)
        self.mamba_rms_norm = getattr(self, 'mamba_rms_norm', False)
        
        # Mamba intermediate size calculation
        mamba_expand = getattr(self, 'mamba_expand', 2)
        if self.mamba_d_ssm is None:
            self.mamba_intermediate_size = mamba_expand * self.hidden_size
        else:
            self.mamba_intermediate_size = self.mamba_d_ssm
            
        # MuP multipliers (ensure they're set even if not from HF config)
        self.embedding_multiplier = getattr(self, 'embedding_multiplier', 1.0)
        self.lm_head_multiplier = getattr(self, 'lm_head_multiplier', 1.0)
        self.mlp_multipliers = getattr(self, 'mlp_multipliers', [1.0, 1.0])
        self.attention_in_multiplier = getattr(self, 'attention_in_multiplier', 1.0)
        self.attention_out_multiplier = getattr(self, 'attention_out_multiplier', 1.0)
        self.key_multiplier = getattr(self, 'key_multiplier', 1.0)
        self.ssm_multipliers = getattr(self, 'ssm_multipliers', [1.0, 1.0, 1.0, 1.0, 1.0])
        self.ssm_in_multiplier = getattr(self, 'ssm_in_multiplier', 1.0)
        self.ssm_out_multiplier = getattr(self, 'ssm_out_multiplier', 1.0)

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ==============================================================================
# RMSNorm Classes
# ==============================================================================

class FalconH1RMSNorm(nn.Module):
    """
    RMSNorm implementation for Falcon-H1.
    Equivalent to T5LayerNorm.
    """
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


class FalconH1RMSNormGated(nn.Module):
    """
    Gated RMSNorm used in Mamba mixer.
    """
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        
        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        
        if len(hidden_states.shape) == 3:
            batch_size, seq_len, dim = hidden_states.shape
        else:
            batch_size, dim = hidden_states.shape
            seq_len = 1
            
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.view(batch_size, seq_len, self.n_groups, int(dim // self.n_groups))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.view(self.n_groups, int(dim // self.n_groups)) * hidden_states
        hidden_states = hidden_states.view(batch_size, seq_len, dim)
        
        if seq_len == 1:
            hidden_states = hidden_states.squeeze(1)
            
        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
            
        return hidden_states.to(input_dtype)


# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_mup_vector(config):
    """
    Computes the MuP vector based on model configuration.
    This applies different MuP multipliers for each dimension.
    """
    intermediate_size = config.mamba_intermediate_size
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    zxbcdt_multipliers = config.ssm_multipliers

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    mup_vector = torch.ones(1, 1, vector_shape)

    # Apply multipliers to different sections
    mup_vector[:, :, :intermediate_size] *= zxbcdt_multipliers[0]
    mup_vector[:, :, intermediate_size:2*intermediate_size] *= zxbcdt_multipliers[1]
    mup_vector[:, :, 2*intermediate_size:2*intermediate_size+groups_time_state_size] *= zxbcdt_multipliers[2]
    mup_vector[:, :, 2*intermediate_size+groups_time_state_size:2*intermediate_size+2*groups_time_state_size] *= zxbcdt_multipliers[3]
    mup_vector[:, :, 2*intermediate_size+2*groups_time_state_size:] *= zxbcdt_multipliers[4]

    return mup_vector


def pad_tensor_by_size(input_tensor, pad_size):
    """Padding x tensor with `pad_size` on the seq_len dim (dim=1)"""
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """Padding and splitting into chunk sequences."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def segment_sum(input_tensor):
    """Stable segment sum calculation using cumulative sums and masking."""
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    # Use -1e9 instead of -inf for numerical stability (Neuron XLA bug)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -1e9)
    return tensor_segsum


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Tunes out the hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


# ==============================================================================
# RoPE Implementation
# ==============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# MLP Component
# ==============================================================================

class FalconH1MLP(nn.Module):
    """
    Falcon-H1 MLP layer with SwiGLU activation.
    Supports MuP multipliers for gate and down projections.
    """
    def __init__(self, config: FalconH1InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        mlp_bias = getattr(config, 'mlp_bias', False)
        
        # MuP multipliers
        self.gate_multiplier = config.mlp_multipliers[0]
        self.down_multiplier = config.mlp_multipliers[1]
        
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def forward(self, x):
        # SwiGLU with MuP multipliers
        y = self.up_proj(x) * F.silu(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


# ==============================================================================
# Attention Component
# ==============================================================================

class FalconH1Attention(NeuronAttentionBase):
    """
    Falcon-H1 Attention layer.
    Multi-headed attention with RoPE and key multiplier.
    """
    
    def __init__(self, config: FalconH1InferenceConfig, layer_idx: int):
        self.key_multiplier = config.key_multiplier
        
        super().__init__(
            config=config,
            tensor_model_parallel_group=get_tp_group(config),
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self._get_rope(config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.layer_idx = layer_idx

    def _get_rope(self, config):
        """Get RoPE embedding."""
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )


# ==============================================================================
# Mamba Mixer Component (SSM)
# ==============================================================================

class FalconH1Mixer(nn.Module):
    """
    Falcon-H1 Mamba2 Mixer implementing Selective State Space Model.
    
    This replaces CUDA kernels (mamba_ssm, causal_conv1d) with pure PyTorch
    for NeuronX compatibility.
    
    Key differences from HF implementation:
    - Uses explicit slicing instead of split() (Neuron XLA bug workaround)
    - Pure PyTorch SSM computation without CUDA kernels
    - Supports both context encoding and token generation modes
    """
    
    def __init__(self, config: FalconH1InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.intermediate_size = config.mamba_intermediate_size
        
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.mamba_rms_norm = config.mamba_rms_norm
        self.mamba_norm_before_gate = config.mamba_norm_before_gate
        
        # Time step limits
        self.time_step_limit = (0.0, float("inf"))
        
        # Groups time state size
        self.groups_time_state_size = self.n_groups * self.ssm_state_size
        
        # Conv dimension
        self.conv_dim = self.intermediate_size + 2 * self.groups_time_state_size
        
        # MuP multipliers
        self.ssm_in_multiplier = config.ssm_in_multiplier
        
        # Projection size for in_proj
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        
        # Create layers
        if parallel_state.model_parallel_is_initialized():
            # For parallel execution, we use ColumnParallelLinear
            # Note: Using ColumnParallelLinear with gather_output=True to avoid
            # the RowParallelLinear scatter bug mentioned in Mamba porting guide
            self.in_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=self.use_bias,
                gather_output=True,  # Gather output to avoid scatter bug
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.out_proj = ColumnParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=getattr(config, 'projectors_bias', False),
                gather_output=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=self.use_bias)
            self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, 
                                      bias=getattr(config, 'projectors_bias', False))
        
        # Depthwise Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        
        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        
        # Gated RMSNorm (optional)
        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=config.rms_norm_eps,
                n_groups=self.n_groups,
                norm_before_gate=self.mamba_norm_before_gate,
            )
        
        # MuP vector - will be set externally if needed
        # Don't register here to avoid double registration
        self.mup_vector = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for Mamba mixer using pure PyTorch (no CUDA kernels).
        
        This implementation uses a vectorized scan approach that's compatible with
        Neuron XLA compilation.
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        
        # Apply mask to padding states
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        
        # Apply SSM input multiplier
        hidden_states = hidden_states * self.ssm_in_multiplier
        
        # Project input
        projected_states = self.in_proj(hidden_states)
        
        # Apply MuP vector if available
        if self.mup_vector is not None:
            projected_states = projected_states * self.mup_vector.to(projected_states.device)
        
        # IMPORTANT: Use explicit slicing instead of split() - Neuron XLA bug workaround
        gate = projected_states[..., :self.intermediate_size]
        hidden_states_B_C = projected_states[..., self.intermediate_size:self.intermediate_size + self.conv_dim]
        dt = projected_states[..., -self.num_heads:]
        
        # Conv1d
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)  # [B, conv_dim, L]
        # Ensure dtype matches conv1d weights
        conv_dtype = self.conv1d.weight.dtype
        hidden_states_B_C = hidden_states_B_C.to(conv_dtype)
        hidden_states_B_C = self.conv1d(hidden_states_B_C)[..., :seq_len]
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)  # [B, L, conv_dim]
        hidden_states_B_C = F.silu(hidden_states_B_C)
        
        # Apply attention mask after conv
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        
        # Split conv output using explicit slicing (NOT split())
        hidden_states_ssm = hidden_states_B_C[..., :self.intermediate_size]
        B = hidden_states_B_C[..., self.intermediate_size:self.intermediate_size + self.groups_time_state_size]
        C = hidden_states_B_C[..., -self.groups_time_state_size:]
        
        # SSM computation with vectorized approach
        A = -torch.exp(self.A_log.float())  # [num_heads]
        
        # Time step (dt) processing
        dt = F.softplus(dt + self.dt_bias)  # [B, L, num_heads]
        dt = torch.clamp(dt, self.time_step_limit[0], 1e6)  # Avoid inf
        
        # Reshape for SSM
        # x: [B, L, num_heads, head_dim]
        hidden_states_ssm = hidden_states_ssm.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()
        # B, C: [B, L, n_groups, state_size] -> [B, L, num_heads, state_size]
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        
        # Compute discretized A and B for all timesteps
        # dt: [B, L, num_heads]
        # A: [num_heads]
        # dA: [B, L, num_heads]
        dA_log = dt * A.view(1, 1, -1)  # [B, L, num_heads]
        dA = torch.exp(dA_log)  # [B, L, num_heads]
        
        # Discretized B: dB = dt * B
        # dt: [B, L, num_heads] -> [B, L, num_heads, 1]
        # B: [B, L, num_heads, state_size]
        dB = dt.unsqueeze(-1) * B  # [B, L, num_heads, state_size]
        
        # Compute dBx: input multiplied by discretized B
        # x: [B, L, num_heads, head_dim]
        # dB: [B, L, num_heads, state_size]
        # dBx: [B, L, num_heads, head_dim, state_size]
        dBx = dB.unsqueeze(3) * hidden_states_ssm.unsqueeze(-1)  # [B, L, num_heads, head_dim, state_size]
        
        # Compute cumulative products of dA for each position
        # For computing state at position t:
        # state[t] = sum_{i=0}^{t} (prod_{j=i+1}^{t} dA[j]) * dBx[i]
        
        # Create a lower triangular mask for causal attention
        # We'll use log-space for numerical stability
        # log_dA_cumsum[t] = sum_{j=0}^{t} log(dA[j])
        log_dA_cumsum = torch.cumsum(dA_log, dim=1)  # [B, L, num_heads]
        
        # For state computation, we need:
        # weights[i, t] = exp(log_dA_cumsum[t] - log_dA_cumsum[i]) for i < t
        # This gives prod_{j=i+1}^{t} dA[j]
        
        # Create indices for broadcasting
        # log_dA_cumsum[:, :, :, None] - log_dA_cumsum[:, None, :, :] gives [B, L, num_heads, L]
        # But we need [B, L_t, L_i, num_heads] or similar
        
        # Use einsum-like operations for state computation
        # For simplicity, let's compute state using a different approach:
        # state[t] = dA[t] * state[t-1] + dBx[t]
        
        # We can express this as a parallel scan:
        # Using associative scan: (a1, b1) o (a2, b2) = (a1*a2, a2*b1 + b2)
        # state[t] = a[t] * state[t-1] + b[t]
        # where a[t] = dA[t], b[t] = dBx[t]
        
        # For XLA compatibility, compute using matrix operations
        # Build the state transition matrix
        # state = sum_{i=0}^{L-1} W[i] @ dBx[i] where W[i] is cumulative dA from i to t
        
        # Compute the weights matrix using outer subtraction of cumsum
        # weights[b, t, i, h] = exp(log_dA_cumsum[b, t, h] - log_dA_cumsum[b, i, h]) if i <= t else 0
        # log_dA_cumsum: [B, L, num_heads]
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=dA.device, dtype=dA.dtype))
        
        # Compute log differences for all pairs
        # We want: log_dA_cumsum[:, t, :] - log_dA_cumsum[:, i, :] for all t, i
        # Shape: [B, L_t, L_i, num_heads]
        log_diff = log_dA_cumsum.unsqueeze(2) - log_dA_cumsum.unsqueeze(1)  # [B, L, L, num_heads]
        
        # Apply causal mask in log space by setting future positions to -inf
        log_diff = log_diff.masked_fill(causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9)
        
        # Exponentiate to get actual weights
        weights = torch.exp(log_diff)  # [B, L_t, L_i, num_heads]
        
        # Now compute states for all positions
        # state[t] = sum_{i=0}^{t} weights[t, i] * dBx[i]
        # weights: [B, L_t, L_i, num_heads]
        # dBx: [B, L, num_heads, head_dim, state_size]
        
        # Reshape for einsum: weights[b, t, i, h] @ dBx[b, i, h, d, s]
        # Result: state[b, t, h, d, s]
        states = torch.einsum('btih,bihds->bthds', weights, dBx)  # [B, L, num_heads, head_dim, state_size]
        
        # Compute output: y = C @ state + D * x
        # C: [B, L, num_heads, state_size]
        # states: [B, L, num_heads, head_dim, state_size]
        # y: [B, L, num_heads, head_dim]
        y = torch.einsum('blhs,blhds->blhd', C, states)  # [B, L, num_heads, head_dim]
        
        # Add D skip connection
        # D: [num_heads]
        # x: [B, L, num_heads, head_dim]
        y = y + self.D.view(1, 1, -1, 1) * hidden_states_ssm  # [B, L, num_heads, head_dim]
        
        # Reshape output
        y = y.reshape(batch_size, seq_len, -1)
        
        # Apply gated normalization or simple gating
        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * F.silu(gate)
        
        # Output projection
        contextualized_states = self.out_proj(scan_output.to(dtype))
        
        return contextualized_states


# ==============================================================================
# Decoder Layer
# ==============================================================================

class FalconH1DecoderLayer(nn.Module):
    """
    Falcon-H1 Decoder Layer combining Mamba mixer, Self-Attention, and MLP.
    Each layer processes: norm -> (mamba || attention) -> add residual -> norm -> mlp -> add residual
    """
    
    def __init__(self, config: FalconH1InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Multipliers
        self.attention_in_multiplier = config.attention_in_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier
        
        # Components
        self.input_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Mamba mixer
        self.mamba = FalconH1Mixer(config, layer_idx)
        
        # Self-attention
        self.self_attn = FalconH1Attention(config, layer_idx)
        
        # MLP
        self.feed_forward = FalconH1MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        Forward pass combining Mamba, Attention, and MLP.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Mamba branch
        mamba_hidden_states = self.mamba(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier
        
        # Attention branch
        attn_output = self.self_attn(
            hidden_states=hidden_states * self.attention_in_multiplier,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        attention_hidden_states = attn_output.hidden_states * self.attn_out_multiplier
        
        # Combine Mamba and Attention outputs
        hidden_states = mamba_hidden_states + attention_hidden_states
        
        # First residual connection
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)
        return outputs


# ==============================================================================
# Wrapper Classes for Embedding and LMHead with Multipliers
# ==============================================================================

class ScaledEmbedding(nn.Module):
    """Embedding wrapper that applies a multiplier to the output."""
    def __init__(self, embedding, multiplier):
        super().__init__()
        self.embedding = embedding
        self.multiplier = multiplier
        # Forward all attributes from the embedding
        self.weight = embedding.weight if hasattr(embedding, 'weight') else None
        
    def forward(self, input_ids):
        return self.embedding(input_ids) * self.multiplier


class ScaledLinear(nn.Module):
    """Linear wrapper that applies a multiplier to the output."""
    def __init__(self, linear, multiplier):
        super().__init__()
        self.linear = linear
        self.multiplier = multiplier
        # Forward all attributes from the linear
        self.weight = linear.weight if hasattr(linear, 'weight') else None
        self.bias = linear.bias if hasattr(linear, 'bias') else None
        # For ColumnParallelLinear
        if hasattr(linear, 'gather_output'):
            self.gather_output = linear.gather_output
        if hasattr(linear, 'tensor_parallel_group'):
            self.tensor_parallel_group = linear.tensor_parallel_group
        if hasattr(linear, 'pad_size'):
            self.pad_size = linear.pad_size
        
    def forward(self, hidden_states):
        return self.linear(hidden_states) * self.multiplier


# ==============================================================================
# Model
# ==============================================================================

class NeuronFalconH1Model(NeuronBaseModel):
    """
    NeuronX implementation of Falcon-H1 Model.
    """
    
    def setup_attr_for_model(self, config: FalconH1InferenceConfig):
        """Set up model attributes."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
        # Falcon-H1 specific
        self.embedding_multiplier = config.embedding_multiplier
        self.lm_head_multiplier = config.lm_head_multiplier

    def init_model(self, config: FalconH1InferenceConfig):
        """Initialize model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        if parallel_state.model_parallel_is_initialized():
            base_embed = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            base_lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            base_embed = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            base_lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        
        # Wrap embeddings and lm_head with scalers for MuP
        self.embed_tokens = ScaledEmbedding(base_embed, self.embedding_multiplier)
        self.lm_head = ScaledLinear(base_lm_head, self.lm_head_multiplier)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            FalconH1DecoderLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Compute and register MuP vectors for each layer
        mup_vector = compute_mup_vector(config)
        for layer in self.layers:
            # Only set mup_vector as attribute, don't use register_buffer
            # to avoid conflicts with existing attribute
            layer.mamba.mup_vector = mup_vector.clone()


# ==============================================================================
# CausalLM Wrapper
# ==============================================================================

class NeuronFalconH1ForCausalLM(NeuronBaseForCausalLM):
    """
    NeuronX implementation of Falcon-H1 for Causal Language Modeling.
    """
    
    _model_cls = NeuronFalconH1Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace model."""
        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: FalconH1InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to NeuronX format.

        Note: The base class already removes the "model." prefix, so keys arrive as:
        - layers.X.feed_forward.* -> stays as layers.X.feed_forward.*
        - layers.X.mamba.* -> stays as layers.X.mamba.*
        - layers.X.self_attn.* -> needs mapping to NeuronAttentionBase structure
        - final_layernorm.weight -> needs to map to norm.weight
        - embed_tokens.weight -> needs to map to embed_tokens.embedding.weight
        - lm_head.weight -> needs to map to lm_head.linear.weight

        CRITICAL: NeuronAttentionBase expects attention weights in specific format:
        - HF: self_attn.q_proj.weight -> Neuron: self_attn.qkv_proj.q_proj.weight
        - HF: self_attn.k_proj.weight -> Neuron: self_attn.qkv_proj.k_proj.weight
        - HF: self_attn.v_proj.weight -> Neuron: self_attn.qkv_proj.v_proj.weight
        - HF: self_attn.o_proj.weight -> Neuron: self_attn.o_proj.o_proj.weight
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        num_layers = config.num_hidden_layers

        for key, value in state_dict.items():
            new_key = key

            # Map final_layernorm to norm
            if "final_layernorm" in new_key:
                new_key = new_key.replace("final_layernorm", "norm")

            # Map embed_tokens.weight to embed_tokens.embedding.weight (for ScaledEmbedding wrapper)
            if new_key == "embed_tokens.weight":
                new_key = "embed_tokens.embedding.weight"

            # Map lm_head.weight to lm_head.linear.weight (for ScaledLinear wrapper)
            if new_key == "lm_head.weight":
                new_key = "lm_head.linear.weight"

            # CRITICAL FIX: Map attention projection weights to NeuronAttentionBase structure
            # NeuronAttentionBase creates internal qkv_proj and o_proj modules
            # The HF weights must be mapped to these internal names
            # NOTE: Use exact pattern matching (endswith) to avoid double-application
            if new_key.endswith(".self_attn.q_proj.weight"):
                new_key = new_key.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.q_proj.weight")
            elif new_key.endswith(".self_attn.k_proj.weight"):
                new_key = new_key.replace(".self_attn.k_proj.weight", ".self_attn.qkv_proj.k_proj.weight")
            elif new_key.endswith(".self_attn.v_proj.weight"):
                new_key = new_key.replace(".self_attn.v_proj.weight", ".self_attn.qkv_proj.v_proj.weight")
            elif new_key.endswith(".self_attn.o_proj.weight"):
                new_key = new_key.replace(".self_attn.o_proj.weight", ".self_attn.o_proj.o_proj.weight")
            # Also handle biases if present
            elif new_key.endswith(".self_attn.q_proj.bias"):
                new_key = new_key.replace(".self_attn.q_proj.bias", ".self_attn.qkv_proj.q_proj.bias")
            elif new_key.endswith(".self_attn.k_proj.bias"):
                new_key = new_key.replace(".self_attn.k_proj.bias", ".self_attn.qkv_proj.k_proj.bias")
            elif new_key.endswith(".self_attn.v_proj.bias"):
                new_key = new_key.replace(".self_attn.v_proj.bias", ".self_attn.qkv_proj.v_proj.bias")
            elif new_key.endswith(".self_attn.o_proj.bias"):
                new_key = new_key.replace(".self_attn.o_proj.bias", ".self_attn.o_proj.o_proj.bias")

            neuron_state_dict[new_key] = value

        # Add rank tensors for attention layers
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank tensor for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embedding and lm_head."""
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class."""
        return FalconH1InferenceConfig


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    "FalconH1InferenceConfig",
    "NeuronFalconH1Model",
    "NeuronFalconH1ForCausalLM",
    "FalconH1MLP",
    "FalconH1Attention",
    "FalconH1Mixer",
    "FalconH1DecoderLayer",
    "FalconH1RMSNorm",
    "FalconH1RMSNormGated",
]
