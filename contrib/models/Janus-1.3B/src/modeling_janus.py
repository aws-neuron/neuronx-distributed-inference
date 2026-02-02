# coding=utf-8
# Copyright 2023 DeepSeek and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Janus model for NeuronX Distributed Inference.

Janus is a multimodal model that combines:
- Vision encoder (SigLIP-based) for image understanding
- Language model (LLaMA-based) for text generation
- VQVAE for image generation

This implementation focuses on text-only inference using the language model component.
"""

import json
import logging
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
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


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm.
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    from neuronx_distributed.utils import cpu_mode
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class JanusInferenceConfig(InferenceConfig):
    """
    Configuration class for Janus model inference on NeuronX.
    
    Janus-1.3B has a LLaMA-based language model with these specs:
    - hidden_size: 2048
    - num_hidden_layers: 24
    - num_attention_heads: 16
    - num_key_value_heads: 16 (same as num_attention_heads, no GQA)
    - intermediate_size: 5632
    - vocab_size: 102400
    - max_position_embeddings: 16384
    - rope_theta: 500000.0
    - rms_norm_eps: 1e-5 (default)
    """
    
    def __init__(self, **kwargs):
        # Extract neuron_config before calling super().__init__
        # If neuron_config is None, create a default one (for inference loading)
        neuron_config = kwargs.get('neuron_config', None)
        if neuron_config is None:
            # During inference, this will be set later after loading from compiled artifacts
            # For now, create a minimal neuron_config to satisfy validation
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
            )
            kwargs['neuron_config'] = neuron_config
        
        super().__init__(**kwargs)
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        
        # Calculate intermediate_size if not provided
        if not hasattr(self, 'intermediate_size') or self.intermediate_size is None:
            # Default for LLaMA-style models is roughly 2.7 * hidden_size
            # For Janus-1.3B, it's explicitly 5632
            self.intermediate_size = getattr(self, 'intermediate_size', int(2.7 * self.hidden_size))
        
        # Set default RMSNorm epsilon if not provided
        if not hasattr(self, 'rms_norm_eps') or self.rms_norm_eps is None:
            self.rms_norm_eps = 1e-5
        
        # Set default RoPE theta if not provided
        if not hasattr(self, 'rope_theta') or self.rope_theta is None:
            self.rope_theta = 500000.0
        
        # Set default hidden activation if not provided
        if not hasattr(self, 'hidden_act') or self.hidden_act is None:
            self.hidden_act = 'silu'
        
        # Janus uses image token ID for multimodal (default 100581)
        if not hasattr(self, 'image_token_id'):
            self.image_token_id = 100581
        
        # Add standard HuggingFace config attributes that the base class expects
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        if not hasattr(self, 'return_dict'):
            self.return_dict = True
        if not hasattr(self, 'tie_word_embeddings'):
            self.tie_word_embeddings = False
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "JanusInferenceConfig":
        """
        Load configuration from a pretrained Janus model directory.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            JanusInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            params = json.load(f)
        
        # Extract language model config from the nested structure
        if "language_config" in params:
            language_config = params["language_config"]
        else:
            # If no language_config, assume the params are directly for language model
            language_config = params
        
        # Create config dict with defaults from config file
        config_dict = {
            "hidden_size": language_config.get("hidden_size", 2048),
            "num_hidden_layers": language_config.get("num_hidden_layers", 24),
            "num_attention_heads": language_config.get("num_attention_heads", 16),
            "num_key_value_heads": language_config.get("num_key_value_heads", 16),
            "intermediate_size": language_config.get("intermediate_size", 5632),
            "vocab_size": language_config.get("vocab_size", 102400),
            "max_position_embeddings": language_config.get("max_position_embeddings", 16384),
            "rope_theta": language_config.get("rope_theta", 500000.0),
            "rms_norm_eps": language_config.get("rms_norm_eps", 1e-5),
            "hidden_act": language_config.get("hidden_act", "silu"),
            "pad_token_id": language_config.get("pad_token_id", 0),
        }
        
        # Handle torch_dtype
        torch_dtype_str = language_config.get("torch_dtype", "bfloat16")
        if torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        
        # Store vision and vq configs for potential future use
        if "vision_config" in params:
            config_dict["vision_config"] = params["vision_config"]
        if "vq_config" in params:
            config_dict["vq_config"] = params["vq_config"]
        if "image_token_id" in params:
            config_dict["image_token_id"] = params["image_token_id"]
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronJanusAttention(NeuronAttentionBase):
    """
    Janus attention implementation for NeuronX.
    
    Janus uses standard multi-head attention (not GQA) with:
    - RoPE (Rotary Position Embeddings)
    - No sliding window
    - Same number of heads for Q, K, V
    
    Reference: JanusVisionAttention and language model attention in modeling_janus.py
    """
    
    def __init__(self, config: JanusInferenceConfig):
        # Initialize rotary embeddings
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize base attention
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )


class NeuronJanusMLP(nn.Module):
    """
    Janus MLP implementation for NeuronX.
    
    Uses SwiGLU activation: down_proj(silu(gate_proj(x)) * up_proj(x))
    This is the same as LLaMA MLP.
    
    Reference: JanusVisionMLP in modeling_janus.py (but language model uses standard LLaMA MLP)
    """
    
    def __init__(self, config: JanusInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Parallel linear layers
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            # CPU fallback
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        # SwiGLU: gate * act(up)
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output


class NeuronJanusDecoderLayer(nn.Module):
    """
    Janus decoder layer for NeuronX.
    
    Structure:
    1. Input layer norm
    2. Self attention
    3. Residual connection
    4. Post-attention layer norm
    5. MLP
    6. Residual connection
    
    Reference: JanusEncoderLayer in modeling_janus.py (vision encoder structure)
    and language model structure (similar to LLaMA decoder)
    """
    
    def __init__(self, config: JanusInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = NeuronJanusAttention(config)
        
        # MLP
        self.mlp = NeuronJanusMLP(config)
        
        # Layer norms
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Forward pass for the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key-value pairs for generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        residual = hidden_states
        
        # Self attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = attn_output.hidden_states
        
        # First residual connection
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        # Return same format as LLaMA decoder layer:
        # (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        outputs = (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,  # residual (not used for fusion in basic implementation)
        )
        
        return outputs


class NeuronJanusModel(NeuronBaseModel):
    """
    Janus base model for NeuronX.
    
    This implements the language model component of Janus.
    Vision and VQVAE components are not included in this text-only version.
    
    Reference: JanusModel in modeling_janus.py
    """
    
    def setup_attr_for_model(self, config: JanusInferenceConfig):
        """Setup attributes for model initialization."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: JanusInferenceConfig):
        """Initialize the model layers."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embeddings
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )
            
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
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronJanusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value


class NeuronJanusForCausalLM(NeuronBaseForCausalLM):
    """
    Janus model for causal language modeling on NeuronX.
    
    This is the main model class for text generation.
    
    Reference: JanusForConditionalGeneration in modeling_janus.py
    """
    
    _model_cls = NeuronJanusModel
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model."""
        return JanusInferenceConfig
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: JanusInferenceConfig) -> dict:
        """
        Convert HuggingFace Janus weights to NeuronX format.
        
        The HuggingFace Janus model has this structure:
        - language_model.model.* (the actual LLaMA-like model)
        - language_model.lm_head.weight (language modeling head)
        - vision_model.* (vision encoder - skip for text-only)
        - gen_vision_model.* (VQVAE - skip for text-only)
        - aligner.* (vision aligner - skip for text-only)
        
        We need to map:
        - language_model.model.* -> (empty, remove this prefix)
        - language_model.lm_head.weight -> lm_head.weight
        - layers.X.self_attn.{q,k,v}_proj -> layers.X.self_attn.qkv_proj.{q,k,v}_proj
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_state_dict = {}
        
        logger.info("Converting HuggingFace Janus weights to NeuronX format")
        logger.info(f"Total original state dict keys: {len(state_dict)}")
        
        for key, value in state_dict.items():
            # Skip non-language model weights
            if not key.startswith("language_model."):
                logger.debug(f"Skipping non-language-model weight: {key}")
                continue
            
            # Remove "language_model." prefix
            new_key = key.replace("language_model.", "")
            
            # Map model.* to (empty) - remove "model." prefix
            new_key = new_key.replace("model.", "")
            
            # Map attention projections from separate q/k/v to qkv_proj structure
            # e.g., layers.0.self_attn.q_proj.weight -> layers.0.self_attn.qkv_proj.q_proj.weight
            if ".self_attn.q_proj." in new_key:
                new_key = new_key.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.q_proj.")
            elif ".self_attn.k_proj." in new_key:
                new_key = new_key.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.k_proj.")
            elif ".self_attn.v_proj." in new_key:
                new_key = new_key.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.v_proj.")
            
            # Keep the weight
            neuron_state_dict[new_key] = value.clone()
        
        # Add rank information for tensor parallelism (required by NeuronX)
        tp_degree = config.neuron_config.tp_degree
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        logger.info(f"Converted {len(neuron_state_dict)} parameters")
        logger.info(f"Sample converted keys: {list(neuron_state_dict.keys())[:10]}")
        
        return neuron_state_dict


# Export classes
__all__ = [
    "JanusInferenceConfig",
    "NeuronJanusAttention",
    "NeuronJanusMLP",
    "NeuronJanusDecoderLayer",
    "NeuronJanusModel",
    "NeuronJanusForCausalLM",
]
