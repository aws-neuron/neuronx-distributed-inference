# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team and the Swiss AI Initiative. All rights reserved.
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
PyTorch Apertus model for NXD inference
Adapted from transformers implementation at:
/shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/apertus/
"""

import os
import json
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.models.llama.modeling_llama import Llama3RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> torch RMSNorm (CustomRMSNorm does not work on CPU)
    """
    if cpu_mode():
        # Fallback RMSNorm implementation for CPU
        class ApertusRMSNorm(nn.Module):
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
        return ApertusRMSNorm
    else:
        return CustomRMSNorm


class XIELUActivation(nn.Module):
    """
    XieLU activation function for Neuron inference
    Based on transformers.activations.XIELUActivation but adapted for Neuron
    Uses Python implementation (CUDA version not compatible with Neuron)
    
    From: https://arxiv.org/abs/2411.13010
    """
    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_p_init, dtype=dtype))).unsqueeze(0)
        )
        self.alpha_n = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_n_init - beta, dtype=dtype))).unsqueeze(0)
        )
        self.beta = beta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, torch.tensor(self.eps, device=x.device))) - x) * alpha_n + self.beta * x,
        )


class ApertusNeuronConfig(NeuronConfig):
    """Neuron-specific configuration for Apertus model"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronApertusAttention


class ApertusInferenceConfig(InferenceConfig):
    """
    Configuration class for Apertus model inference on Neuron
    
    Inherits from InferenceConfig and adds Apertus-specific parameters
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        # Add head_dim if not present
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads
        # Add standard HuggingFace config attributes if not present
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[ApertusNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return ApertusNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            ApertusInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract relevant parameters with defaults
        model_config = {
            "vocab_size": config_dict.get("vocab_size", 131072),
            "hidden_size": config_dict.get("hidden_size", 4096),
            "intermediate_size": config_dict.get("intermediate_size", 21504),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 32),
            "num_attention_heads": config_dict.get("num_attention_heads", 32),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 8),
            "hidden_act": config_dict.get("hidden_act", "xielu"),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 65536),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-5),
            "rope_theta": config_dict.get("rope_theta", 12000000.0),
            "rope_scaling": config_dict.get("rope_scaling", None),
            "attention_bias": config_dict.get("attention_bias", False),
            "attention_dropout": config_dict.get("attention_dropout", 0.0),
            "pad_token_id": config_dict.get("pad_token_id", 3),
            "bos_token_id": config_dict.get("bos_token_id", 1),
            "eos_token_id": config_dict.get("eos_token_id", 68),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
            "qk_norm": config_dict.get("qk_norm", True),
        }
        
        # Override with any additional kwargs
        model_config.update(kwargs)
        
        # If neuron_config is None, create a default one for inference loading
        # This will be replaced by the actual neuron_config from compiled artifacts
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import NeuronConfig
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
        
        # Create config object
        config = cls(neuron_config=neuron_config, **model_config)
        return config


class NeuronApertusAttention(NeuronAttentionBase):
    """
    Apertus attention implementation for NeuronX
    
    Key features:
    - Grouped Query Attention (GQA) with 32 query heads and 8 KV heads
    - Q-K normalization: RMSNorm applied to query and key after projection
    - RoPE (Rotary Position Embeddings) with LLaMA3 scaling
    - No bias in projections (attention_bias=False)
    
    Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/apertus/modeling_apertus.py
    """
    
    def __init__(self, config: ApertusInferenceConfig):
        # Calculate head dimension
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # Initialize rotary embeddings
        # Apertus uses LLaMA3-style RoPE scaling with very high base (12M)
        rope_scaling = getattr(config, "rope_scaling", None)
        
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            # Use Llama3RotaryEmbedding for LLaMA3-style scaling
            rotary_emb = Llama3RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )
        else:
            # Use standard RotaryEmbedding
            rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        
        # Initialize attention with Q-K normalization
        # q_layernorm and k_layernorm are applied after projection but before RoPE
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            q_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
        )


class NeuronApertusMLP(nn.Module):
    """
    Apertus MLP implementation for NeuronX
    
    Key differences from LLaMA:
    - Uses XieLU activation instead of SwiGLU
    - Simple structure: up_proj -> xielu -> down_proj
    - No gate_proj (unlike LLaMA which has gate_proj + up_proj)
    - No bias in projections (mlp_bias=False)
    
    Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/apertus/modeling_apertus.py
    Class: ApertusMLP
    """
    
    def __init__(self, config: ApertusInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Apertus uses simple MLP with XieLU activation
        # up_proj: hidden_size -> intermediate_size
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        
        # XieLU activation function
        self.act_fn = XIELUActivation(dtype=config.neuron_config.torch_dtype)
        
        # down_proj: intermediate_size -> hidden_size
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
    
    def forward(self, x):
        """
        Forward pass: down_proj(xielu(up_proj(x)))
        
        Returns:
            Tuple[torch.Tensor, None]: Output tensor and None for compatibility
        """
        # Project to intermediate size
        intermediate = self.up_proj(x)
        
        # Apply XieLU activation
        activated = self.act_fn(intermediate)
        
        # Project back to hidden size
        output = self.down_proj(activated)
        
        # Return tuple for compatibility with NXD framework
        return output, None


class NeuronApertusDecoderLayer(nn.Module):
    """
    Apertus decoder layer for NeuronX
    
    Architecture (pre-norm):
    1. residual = hidden_states
    2. hidden_states = attention_layernorm(hidden_states)
    3. hidden_states = self_attn(hidden_states)
    4. hidden_states = residual + hidden_states
    5. residual = hidden_states
    6. hidden_states = feedforward_layernorm(hidden_states)
    7. hidden_states = mlp(hidden_states)
    8. hidden_states = residual + hidden_states
    
    Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/apertus/modeling_apertus.py
    Class: ApertusDecoderLayer
    """
    
    def __init__(self, config: ApertusInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention block
        self.self_attn = NeuronApertusAttention(config)
        
        # MLP block
        self.mlp = NeuronApertusMLP(config)
        
        # Layer normalization (pre-norm architecture)
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
        Forward pass through decoder layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key-value pairs
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing:
                - hidden_states: Output tensor
                - present_key_value: Updated KV cache
                - cos_cache: Cosine cache for RoPE
                - sin_cache: Sine cache for RoPE
                - None: Placeholder for compatibility
        """
        # Self Attention block with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # MLP block with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronApertusModel(NeuronBaseModel):
    """
    Apertus model for NeuronX inference
    
    This is the main model class that contains:
    - Token embeddings
    - Stack of decoder layers
    - Final layer normalization
    - LM head for next-token prediction
    
    Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/apertus/modeling_apertus.py
    Class: ApertusModel
    """
    
    def setup_attr_for_model(self, config: ApertusInferenceConfig):
        """Setup attributes required by NeuronBaseModel"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: ApertusInferenceConfig):
        """Initialize model components"""
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
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronApertusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (output projection to vocabulary)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronApertusForCausalLM(NeuronBaseForCausalLM):
    """
    Apertus model for causal language modeling on NeuronX
    
    This is the main entry point for using the Apertus model.
    It wraps NeuronApertusModel and provides:
    - Model loading from HuggingFace checkpoints
    - Weight conversion from HF format to Neuron format
    - Compilation and inference interfaces
    
    Usage:
        config = ApertusInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronApertusForCausalLM.from_config(config)
        model.load_weights(checkpoint_path)
        model.compile()
        outputs = model.generate(...)
    """
    
    _model_cls = NeuronApertusModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load HuggingFace model (not used for Neuron inference, but kept for compatibility)
        """
        # Note: We don't actually load the HF model for Neuron inference
        # This is just for reference/compatibility
        print(f"Loading HF model from {model_path} (reference only)")
        return None
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format
        
        This function maps weight names from HuggingFace format to NeuronX format
        and adds necessary metadata for tensor parallelism.
        
        HF Format -> Neuron Format:
        - model.embed_tokens.weight -> embed_tokens.weight
        - model.layers.{i}.self_attn.q_proj.weight -> layers.{i}.self_attn.qkv_proj.q_proj.weight
        - model.layers.{i}.self_attn.q_norm.weight -> layers.{i}.self_attn.q_layernorm.weight
        - model.layers.{i}.self_attn.k_norm.weight -> layers.{i}.self_attn.k_layernorm.weight
        - model.layers.{i}.input_layernorm.weight -> layers.{i}.input_layernorm.weight
        - model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.post_attention_layernorm.weight
        - model.layers.{i}.mlp.up_proj.weight -> layers.{i}.mlp.up_proj.weight
        - model.layers.{i}.mlp.down_proj.weight -> layers.{i}.mlp.down_proj.weight
        - model.norm.weight -> norm.weight
        - lm_head.weight -> lm_head.weight
        
        Args:
            state_dict: HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            dict: Neuron-format state dictionary
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}
        
        # Handle vocabulary parallel sharding
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        # Process each layer
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        for key, value in state_dict.items():
            new_key = key
            
            # Remove 'model.' prefix if present
            if new_key.startswith("model."):
                new_key = new_key[6:]  # Remove "model."
            
            # Rename q_norm and k_norm to q_layernorm and k_layernorm
            if ".q_norm." in new_key:
                new_key = new_key.replace(".q_norm.", ".q_layernorm.")
            if ".k_norm." in new_key:
                new_key = new_key.replace(".k_norm.", ".k_layernorm.")
            
            # Rename attention_layernorm to input_layernorm
            if ".attention_layernorm." in new_key:
                new_key = new_key.replace(".attention_layernorm.", ".input_layernorm.")
            
            # Rename feedforward_layernorm to post_attention_layernorm
            if ".feedforward_layernorm." in new_key:
                new_key = new_key.replace(".feedforward_layernorm.", ".post_attention_layernorm.")
            
            # Copy the weight
            neuron_state_dict[new_key] = value.detach().clone()
        
        # Add rank information for tensor parallelism
        for i in range(num_layers):
            # Rank information for attention layers
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        print(f"Converted {len(state_dict)} HF weights to {len(neuron_state_dict)} Neuron weights")
        return neuron_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embedding and LM head
        
        Note: Apertus uses tie_word_embeddings=False by default,
        so this is typically not needed, but kept for compatibility.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class"""
        return ApertusInferenceConfig
