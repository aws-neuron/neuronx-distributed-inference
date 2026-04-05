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
"""PyTorch Phi-3 model for NXD inference."""

import json
import logging
import math
import os
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

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """
    LongRoPE Scaled Rotary Position Embedding for Phi-3.5.
    
    This implements the LongRoPE scaling mechanism that allows Phi-3.5 to handle
    extended context lengths (up to 128k tokens) by applying position-dependent
    scaling factors to the rotary embedding.
    
    Key features:
    - Uses short_factor for sequences <= original_max_position_embeddings (4096)
    - Uses long_factor for longer sequences
    - Applies a scaling factor based on context length ratio
    
    Reference: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        original_max_position_embeddings: int = 4096,
        short_factor: list = None,
        long_factor: list = None,
        device=None,
    ):
        """
        Initialize LongRoPE rotary embedding.
        
        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length (131072 for Phi-3.5)
            base: RoPE theta base (10000.0)
            original_max_position_embeddings: Original context length (4096)
            short_factor: Scaling factors for short sequences (list of floats)
            long_factor: Scaling factors for long sequences (list of floats)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        
        # Store scaling factors
        self.short_factor = short_factor if short_factor is not None else [1.0] * (dim // 2)
        self.long_factor = long_factor if long_factor is not None else [1.0] * (dim // 2)
        
        # Register buffers for inv_freq (will be computed dynamically)
        self.register_buffer("inv_freq", None, persistent=False)
        
        logger.info(f"Phi3LongRoPEScaledRotaryEmbedding: dim={dim}, base={base}, "
                   f"max_pos={max_position_embeddings}, "
                   f"original_max_pos={original_max_position_embeddings}")
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings with LongRoPE scaling.
        
        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            position_ids: Position indices [batch, seq_len]
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        # Determine sequence length from position_ids
        seq_len = position_ids.max().item() + 1 if position_ids.numel() > 0 else 1
        
        # Choose scaling factors based on sequence length
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)
        
        # Compute inverse frequencies with scaling
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        inv_freq = 1.0 / (ext_factors * self.base ** inv_freq_shape)
        
        # Expand for batch computation
        # inv_freq: [dim/2] -> [batch, dim/2, 1]
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies: [batch, dim/2, seq_len] -> [batch, seq_len, dim/2]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        
        # Concatenate for full dimension: [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Compute scaling factor for long contexts
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        
        # Apply scaling and convert to target dtype
        cos = (emb.cos() * scaling_factor).to(dtype=x.dtype)
        sin = (emb.sin() * scaling_factor).to(dtype=x.dtype)
        
        return cos, sin


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm implementation based on execution mode.
    CustomRMSNorm for NXD, standard implementation for CPU.
    """
    # Import here to avoid circular dependencies
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class Phi3InferenceConfig(InferenceConfig):
    """
    Configuration class for Phi-3 inference on Neuron.
    
    Phi-3 uses a similar architecture to LLaMA with some key differences:
    - Fused QKV projection (single linear layer)
    - Fused gate_up projection in MLP
    - LongRoPE scaling support
    - Partial rotary factor
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        
        # Only proceed if neuron_config is available
        if self.neuron_config is None:
            return
            
        # Handle partial rotary factor
        if not hasattr(self, 'partial_rotary_factor'):
            self.partial_rotary_factor = 1.0
            
        # Calculate rotary dimensions considering partial factor
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_dim * self.partial_rotary_factor)
        
        # Handle rope_scaling for LongRoPE
        if hasattr(self, 'rope_scaling') and self.rope_scaling is not None:
            if 'type' in self.rope_scaling and self.rope_scaling['type'] == 'longrope':
                logger.info("LongRoPE scaling detected in configuration")
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
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
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    def validate_config(self):
        """Validate configuration, handling None neuron_config."""
        # If neuron_config is None, skip validation - it will be set later
        if self.neuron_config is None:
            return
        # Otherwise, call parent validation
        super().validate_config()
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        """
        Load configuration from a pretrained model path.
        
        This method loads the HuggingFace config and initializes the Neuron config.
        
        Args:
            model_path: Path to the HuggingFace model directory
            neuron_config: NeuronConfig instance (optional, will load if not provided)
            **kwargs: Additional arguments to override config values
            
        Returns:
            Phi3InferenceConfig instance
        """
        from transformers import AutoConfig
        import json
        
        # If neuron_config is not provided, try to load it from model_path
        if neuron_config is None:
            config_json_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_json_path):
                with open(config_json_path, 'r') as f:
                    saved_config = json.load(f)
                    if 'neuron_config' in saved_config:
                        neuron_config = NeuronConfig(**saved_config['neuron_config'])
        
        # Load HuggingFace config
        hf_config = AutoConfig.from_pretrained(model_path)
        config_dict = hf_config.to_dict()
        
        # Create load_config function to load attributes from HF config
        def load_config_fn(self):
            for key, value in config_dict.items():
                if not hasattr(self, key):
                    setattr(self, key, value)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config instance
        return cls(neuron_config=neuron_config, load_config=load_config_fn, **config_dict)


class NeuronPhi3MLP(nn.Module):
    """
    Phi-3 MLP implementation for NeuronX.
    
    Key difference from LLaMA: Phi-3 uses a fused gate_up_proj layer
    (single linear layer that outputs 2 * intermediate_size), which is then
    split into gate and up components.
    
    Original Phi-3 structure:
        gate_up_proj: Linear(hidden_size, 2 * intermediate_size, bias=False)
        down_proj: Linear(intermediate_size, hidden_size, bias=False)
        activation: SiLU(gate) * up
    
    For Neuron, we keep separate gate_proj and up_proj for compatibility
    with tensor parallelism, but load weights from the fused checkpoint.
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Use separate gate and up projections for tensor parallelism
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states):
        """
        Forward pass using SwiGLU activation.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        # Gate and up projections
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        
        # SwiGLU activation: silu(gate) * up
        intermediate_output = self.act_fn(gate_output) * up_output
        
        # Down projection
        output = self.down_proj(intermediate_output)
        
        return output


class NeuronPhi3Attention(NeuronAttentionBase):
    """
    Phi-3 attention implementation for NeuronX.
    
    Key difference from LLaMA: Phi-3 uses a fused QKV projection
    (single linear layer) instead of separate Q, K, V projections.
    
    The fused weights are split during state dict conversion.
    NeuronAttentionBase handles creating the separate q_proj, k_proj, v_proj
    through GroupQueryAttention_QKV.
    
    Phi-3.5 uses LongRoPE scaling for extended context support (128k tokens).
    """
    
    def __init__(self, config: Phi3InferenceConfig, layer_idx: Optional[int] = None):
        """
        Initialize Phi-3 attention.
        
        Args:
            config: Model configuration
            layer_idx: Layer index for caching
        """
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Check if LongRoPE scaling is configured
        rope_scaling = getattr(config, 'rope_scaling', None)
        
        if rope_scaling is not None and rope_scaling.get('type') == 'longrope':
            # Use LongRoPE for Phi-3.5
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
                base=getattr(config, "rope_theta", 10000.0),
                original_max_position_embeddings=getattr(config, "original_max_position_embeddings", 4096),
                short_factor=rope_scaling.get('short_factor'),
                long_factor=rope_scaling.get('long_factor'),
            )
            logger.info(f"Using Phi3LongRoPEScaledRotaryEmbedding for layer {layer_idx}")
        else:
            # Fall back to standard RotaryEmbedding for non-LongRoPE models
            partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
            rotary_ndims = int(head_dim * partial_rotary_factor)
            rotary_emb = RotaryEmbedding(
                rotary_ndims,
                max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
                base=getattr(config, "rope_theta", 10000.0),
            )
        
        # Initialize base attention
        # NeuronAttentionBase will create qkv_proj and o_proj internally
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            qkv_bias=False,  # Phi-3 doesn't use bias
            o_bias=False,
        )


class NeuronPhi3DecoderLayer(nn.Module):
    """
    Phi-3 decoder layer implementation for NeuronX.
    
    Structure:
        1. input_layernorm (RMSNorm)
        2. self_attn (NeuronPhi3Attention)
        3. Residual connection
        4. post_attention_layernorm (RMSNorm)
        5. mlp (NeuronPhi3MLP)
        6. Residual connection
    """
    
    def __init__(self, config: Phi3InferenceConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Attention
        self.self_attn = NeuronPhi3Attention(config, layer_idx=layer_idx)
        
        # MLP
        self.mlp = NeuronPhi3MLP(config)
        
        # Layer normalization
        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = rmsnorm_cls(
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
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value tensors
            
        Returns:
            Tuple of (output tensor, updated cache)
        """
        residual = hidden_states
        
        # Pre-attention normalization
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
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return tuple (hidden_states, kv_cache, cos_cache, sin_cache, residual)
        # Set residual to None as we've already added it
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronPhi3Model(NeuronBaseModel):
    """
    Phi-3 base model for NeuronX.
    
    This is the main transformer model including the language modeling head.
    """
    
    def setup_attr_for_model(self, config: Phi3InferenceConfig):
        """Setup attributes for model initialization."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: Phi3InferenceConfig):
        """Initialize the model components."""
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            
            # Language modeling head
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
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NeuronPhi3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class NeuronPhi3ForCausalLM(NeuronBaseForCausalLM):
    """
    Phi-3 causal language model for NeuronX inference.
    
    This class extends NeuronBaseForCausalLM and provides Phi-3 specific
    implementations for weight conversion and model loading.
    """
    
    _model_cls = NeuronPhi3Model
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Phi-3 checkpoint to Neuron format.
        
        Key conversions needed:
        1. Strip 'model.' prefix from HuggingFace keys
        2. Unfuse QKV projection weights and add qkv_proj wrapper prefix
        3. Unfuse gate_up MLP projection weights
        4. Map o_proj to o_proj.o_proj for NeuronAttentionBase
        5. Add rank tensors for tensor parallelism
        
        HuggingFace Phi-3 format:
            - model.layers.X.self_attn.qkv_proj.weight: [total_size, hidden_size]
              where total_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
            - model.layers.X.self_attn.o_proj.weight
            - model.layers.X.mlp.gate_up_proj.weight: [2 * intermediate_size, hidden_size]
        
        Neuron format (NeuronAttentionBase expects):
            - layers.X.self_attn.qkv_proj.q_proj.weight
            - layers.X.self_attn.qkv_proj.k_proj.weight
            - layers.X.self_attn.qkv_proj.v_proj.weight
            - layers.X.self_attn.o_proj.o_proj.weight
            - layers.X.mlp.gate_proj.weight
            - layers.X.mlp.up_proj.weight
        
        Args:
            state_dict: Original HuggingFace state dict
            config: Model configuration
            
        Returns:
            Converted state dict for Neuron
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        intermediate_size = config.intermediate_size
        
        # Process each key in the original state dict
        for key, value in state_dict.items():
            # First, strip 'model.' prefix if present
            working_key = key
            if working_key.startswith('model.'):
                working_key = working_key[6:]  # Remove 'model.' prefix
            
            # Handle fused QKV projection
            if '.self_attn.qkv_proj.weight' in working_key:
                # Extract layer index from the key (now without 'model.' prefix)
                # Format: layers.X.self_attn.qkv_proj.weight
                layer_idx = int(working_key.split('.')[1])
                
                # Split the fused QKV weight
                # Shape: [num_heads * head_dim + 2 * num_kv_heads * head_dim, hidden_size]
                q_size = num_heads * head_dim
                k_size = num_kv_heads * head_dim
                v_size = num_kv_heads * head_dim
                
                q_weight = value[:q_size, :]
                k_weight = value[q_size:q_size + k_size, :]
                v_weight = value[q_size + k_size:q_size + k_size + v_size, :]
                
                # Store split weights with qkv_proj wrapper prefix for NeuronAttentionBase
                neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.q_proj.weight"] = q_weight
                neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.k_proj.weight"] = k_weight
                neuron_state_dict[f"layers.{layer_idx}.self_attn.qkv_proj.v_proj.weight"] = v_weight
                
            # Handle fused gate_up projection
            elif '.mlp.gate_up_proj.weight' in working_key:
                # Extract layer index
                layer_idx = int(working_key.split('.')[1])
                
                # Split the fused gate_up weight
                # Shape: [2 * intermediate_size, hidden_size]
                gate_weight = value[:intermediate_size, :]
                up_weight = value[intermediate_size:, :]
                
                # Store split weights
                neuron_state_dict[f"layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight
                neuron_state_dict[f"layers.{layer_idx}.mlp.up_proj.weight"] = up_weight
                
            # Handle o_proj - preshard_hook will add the o_proj.o_proj wrapper
            # So we just need to provide layers.X.self_attn.o_proj.weight
            elif '.self_attn.o_proj.' in working_key:
                # Just copy as-is (already stripped 'model.' prefix)
                neuron_state_dict[working_key] = value
                
            # Copy other weights directly (already stripped 'model.' prefix)
            elif 'qkv_proj' not in working_key and 'gate_up_proj' not in working_key:
                neuron_state_dict[working_key] = value
        
        # Add rank tensors for tensor parallelism
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank tensor for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        logger.info(f"Converted {len(state_dict)} HF weights to {len(neuron_state_dict)} Neuron weights")
        
        return neuron_state_dict
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return Phi3InferenceConfig


# Export public API
__all__ = [
    "Phi3InferenceConfig",
    "NeuronPhi3Model",
    "NeuronPhi3ForCausalLM",
    "NeuronPhi3MLP",
    "NeuronPhi3Attention",
    "NeuronPhi3DecoderLayer",
]
