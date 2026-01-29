# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Qwen3-VL model for NXD inference
This implementation focuses on the text-only model (Qwen3VLTextModel) and defers
vision components for future implementation.
"""
import os
import json
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

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
    Get appropriate RMSNorm implementation based on execution mode
    - CPU mode: Use HF RMSNorm
    - Neuron mode: Use CustomRMSNorm (optimized for Neuron hardware)
    """
    return Qwen3RMSNorm if cpu_mode() else CustomRMSNorm


class Qwen3VLNeuronConfig(NeuronConfig):
    """
    Extended NeuronConfig for Qwen3-VL with custom attention class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronQwen3VLAttention


class Qwen3VLInferenceConfig(InferenceConfig):
    """
    Configuration class for Qwen3-VL inference on Neuron hardware
    
    This config handles the text model portion of Qwen3-VL, which uses:
    - Multi-dimensional RoPE (MRoPE) for temporal, height, width positions
    - Q-K normalization (RMSNorm on query and key after projection)
    - Grouped Query Attention (GQA) with 32 attention heads and 8 KV heads
    - SwiGLU MLP activation
    """

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        
        # Handle MRoPE configuration
        if hasattr(self, 'rope_scaling') and self.rope_scaling is not None:
            self.mrope_section = self.rope_scaling.get('mrope_section', [24, 20, 20])
            self.mrope_interleaved = self.rope_scaling.get('mrope_interleaved', True)
        else:
            self.mrope_section = [24, 20, 20]
            self.mrope_interleaved = True

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen3VLNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Qwen3VLNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Qwen3VLInferenceConfig":
        """
        Load configuration from a pretrained Qwen3-VL model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration
            
        Returns:
            Qwen3VLInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # If neuron_config is not provided, try to create a minimal one for validation
        # This handles the case where from_pretrained is called during inference loading
        if neuron_config is None:
            # Check if neuron_config.json exists in model_path
            neuron_config_path = os.path.join(model_path, "neuron_config.json")
            if os.path.exists(neuron_config_path):
                neuron_config = NeuronConfig.from_json(neuron_config_path)
            else:
                # Create a minimal neuron_config for validation during inference
                # The actual neuron_config will be loaded separately by the inference framework
                neuron_config = NeuronConfig(
                    tp_degree=1,
                    max_batch_size=1,
                    seq_len=512,
                    torch_dtype=torch.bfloat16,
                )
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract text_config from Qwen3-VL config
        if "text_config" in config_dict:
            text_config = config_dict["text_config"]
        else:
            text_config = config_dict
        
        # Create config dict with text model parameters
        inference_config = {
            "hidden_size": text_config.get("hidden_size", 4096),
            "num_attention_heads": text_config.get("num_attention_heads", 32),
            "num_hidden_layers": text_config.get("num_hidden_layers", 36),
            "num_key_value_heads": text_config.get("num_key_value_heads", 8),
            "head_dim": text_config.get("head_dim", 128),
            "vocab_size": text_config.get("vocab_size", 151936),
            "max_position_embeddings": text_config.get("max_position_embeddings", 262144),
            "rope_theta": text_config.get("rope_theta", 5000000.0),
            "rms_norm_eps": text_config.get("rms_norm_eps", 1e-6),
            "hidden_act": text_config.get("hidden_act", "silu"),
            "intermediate_size": text_config.get("intermediate_size", 12288),
            "attention_bias": text_config.get("attention_bias", False),
            "attention_dropout": text_config.get("attention_dropout", 0.0),
            "rope_scaling": text_config.get("rope_scaling", None),
            "pad_token_id": text_config.get("pad_token_id", 0),
            "bos_token_id": text_config.get("bos_token_id", 151643),
            "eos_token_id": text_config.get("eos_token_id", 151645),
            # Additional attributes for compatibility
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True,
        }
        
        # Override with remaining kwargs
        inference_config.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **inference_config)
        return config


class Qwen3VLRotaryEmbedding(nn.Module):
    """
    Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-VL
    
    Unlike standard RoPE, MRoPE handles 3D position information:
    - Temporal dimension (for video/sequence)
    - Height dimension (for 2D spatial layout)
    - Width dimension (for 2D spatial layout)
    
    The position IDs have shape (3, batch_size, seq_len) instead of (batch_size, seq_len).
    """
    
    def __init__(self, config: Qwen3VLInferenceConfig):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        
        # MRoPE specific configuration
        self.mrope_section = getattr(config, 'mrope_section', [24, 20, 20])
        self.mrope_interleaved = getattr(config, 'mrope_interleaved', True)
        
        # Attention scaling (default to 1.0 for standard rope)
        # In HF, this comes from rope_init_fn, but we use default here
        self.attention_scaling = 1.0
        
        # Initialize inverse frequencies for RoPE
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        Apply interleaved MRoPE to 3D rotary embeddings.
        
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        
        Args:
            freqs: (3, bs, seq_len, head_dim // 2) - frequencies for T, H, W
            mrope_section: (3,) - sections for temporal, height, width
            
        Returns:
            freqs_t: (bs, seq_len, head_dim // 2) - interleaved frequencies
        """
        freqs_t = freqs[0].clone()  # Start with temporal frequencies
        
        # Interleave height and width frequencies into temporal
        for dim, offset in enumerate((1, 2), start=1):  # H, W dimensions
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        
        return freqs_t
    
    def forward(self, x, position_ids):
        """
        Forward pass for MRoPE
        
        Args:
            x: Input tensor for determining device and dtype
            position_ids: Position IDs with shape (3, batch_size, seq_len) or (batch_size, seq_len)
                         If 2D, it will be expanded to 3D for T, H, W dimensions
        
        Returns:
            cos: Cosine embeddings (batch_size, seq_len, head_dim // 2)
            sin: Sine embeddings (batch_size, seq_len, head_dim // 2)
        """
        # Expand position_ids to 3D if needed (T, H, W dimensions)
        if position_ids.ndim == 2:
            # Shape: (batch_size, seq_len) -> (3, batch_size, seq_len)
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Expand inv_freq to match position_ids shape
        # Shape: (head_dim // 2) -> (3, batch_size, head_dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        
        # Expand position_ids for matmul
        # Shape: (3, batch_size, seq_len) -> (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()
        
        # Compute frequencies for each dimension
        # Shape: (3, batch_size, head_dim // 2, seq_len)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            
            # Apply interleaved MRoPE if configured
            if self.mrope_interleaved:
                freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            else:
                # Use only temporal frequencies if not interleaved
                freqs = freqs[0]
            
            # Create cos/sin embeddings
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronQwen3VLAttention(NeuronAttentionBase):
    """
    Qwen3-VL Attention module with Q-K normalization and MRoPE support
    
    Key features:
    - Grouped Query Attention (GQA) with 32 Q heads and 8 KV heads
    - RMSNorm applied to query and key after projection (on head_dim)
    - Multi-dimensional RoPE (MRoPE) for 3D position encoding
    - No bias in attention projections (attention_bias=False)
    """

    def __init__(self, config: Qwen3VLInferenceConfig):
        # Use custom MRoPE instead of standard RoPE
        rotary_emb = Qwen3VLRotaryEmbedding(config)
        
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # Initialize with Q-K normalization
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            q_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(hidden_size=head_dim, eps=config.rms_norm_eps),
        )


class NeuronQwen3VLDecoderLayer(nn.Module):
    """
    Qwen3-VL decoder layer with pre-normalization
    
    Architecture:
    1. Input LayerNorm
    2. Self-Attention with Q-K normalization
    3. Residual connection
    4. Post-attention LayerNorm
    5. MLP (SwiGLU)
    6. Residual connection
    """

    def __init__(self, config: Qwen3VLInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention module with Q-K normalization
        self.self_attn = NeuronQwen3VLAttention(config)
        
        # MLP module (reuse LlamaMLP which implements SwiGLU)
        self.mlp = NeuronLlamaMLP(config)
        
        # Layer normalization
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
        Forward pass for decoder layer
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs (can be 2D or 3D for MRoPE)
            past_key_value: Cached key-value pairs
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, router_logits)
        """
        # Pre-attention normalization and self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-MLP normalization and MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MLP returns (output, None)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen3VLModel(NeuronBaseModel):
    """
    Qwen3-VL base model for text generation
    
    This model implements the text decoder portion of Qwen3-VL, which can be used
    for language modeling tasks. Vision components are not included in this
    initial implementation.
    
    Architecture:
    - Token embeddings (ParallelEmbedding with vocab sharding)
    - Stack of Qwen3VL decoder layers (36 layers for 8B model)
    - Final RMSNorm
    - Language modeling head (shared with embeddings if tie_word_embeddings=True)
    """

    def setup_attr_for_model(self, config: Qwen3VLInferenceConfig):
        """Setup attributes for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen3VLInferenceConfig):
        """Initialize model components"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings with vocabulary sharding for tensor parallelism
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
            [NeuronQwen3VLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronQwen3VLForCausalLM(NeuronBaseForCausalLM):
    """
    Qwen3-VL model for causal language modeling
    
    This class provides the complete interface for text generation, including:
    - Model loading from HuggingFace checkpoints
    - Weight conversion to Neuron format
    - Compilation for Neuron hardware
    - Inference and generation
    
    Usage:
        config = Qwen3VLInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronQwen3VLForCausalLM.from_config(config)
        model.load_weights(model_path)
        model.compile()
        outputs = model.generate(input_ids)
    """

    _model_cls = NeuronQwen3VLModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load HuggingFace model weights
        
        Note: Qwen3-VL uses a different model class (Qwen3VLForConditionalGeneration)
        but we can load the text model weights directly from safetensors.
        """
        # We load weights directly from safetensors instead of using HF model class
        # since Qwen3VLForConditionalGeneration includes vision components we don't need
        return None  # Will load weights directly in convert_hf_to_neuron_state_dict

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Qwen3-VL weights to Neuron format
        
        Key mappings:
        - model.language_model.embed_tokens.weight -> embed_tokens.weight
        - model.language_model.layers.{i}.* -> layers.{i}.*
        - model.language_model.norm.weight -> norm.weight
        - lm_head.weight (already at root level)
        - self_attn.q_norm.weight -> self_attn.q_layernorm.weight
        - self_attn.k_norm.weight -> self_attn.k_layernorm.weight
        
        Note: q_proj, k_proj, v_proj are NOT renamed to qkv_proj.* here because
        the preshard_hook in GQA handles that mapping automatically.
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for Neuron model
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}
        
        # Debug: check original state_dict
        print(f"📥 Original state_dict has {len(state_dict)} keys")
        layer_0_orig_keys = [k for k in state_dict.keys() if "layers.0" in k or "layer.0" in k]
        print(f"   Layer 0 keys in original ({len(layer_0_orig_keys)}):")
        for key in sorted(layer_0_orig_keys)[:10]:
            print(f"     - {key}")
        
        # Add rank information for tensor parallelism
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        # Process each key in the original state dict
        for key, value in state_dict.items():
            # Skip vision-related weights
            if "visual" in key or "vision" in key:
                continue
            
            # Map language_model.* to root level
            # Qwen3-VL uses "language_model." prefix (not "model.language_model.")
            new_key = key
            if key.startswith("language_model."):
                new_key = key.replace("language_model.", "")
            elif key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "")
            elif key.startswith("model."):
                # In case of other model.* patterns
                new_key = key.replace("model.", "")
            
            # Rename q_norm and k_norm to q_layernorm and k_layernorm
            # This is specific to Qwen3-VL's Q-K normalization feature
            if "self_attn.q_norm.weight" in new_key:
                new_key = new_key.replace("self_attn.q_norm.weight", "self_attn.q_layernorm.weight")
            elif "self_attn.k_norm.weight" in new_key:
                new_key = new_key.replace("self_attn.k_norm.weight", "self_attn.k_layernorm.weight")
            
            neuron_state_dict[new_key] = value.detach().clone()
        
        # Add rank information for attention layers
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        # Debug: print layer 0 attention keys
        layer_0_attn_keys = [k for k in neuron_state_dict.keys() if k.startswith("layers.0.self_attn")]
        print(f"✅ Converted {len(neuron_state_dict)} weights to Neuron format")
        print(f"   Layer 0 attention keys ({len(layer_0_attn_keys)}):")
        for key in sorted(layer_0_attn_keys):
            print(f"     - {key}")
        
        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embeddings and lm_head
        
        In Qwen3-VL, tie_word_embeddings is typically False, but we support both cases.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return Qwen3VLInferenceConfig
