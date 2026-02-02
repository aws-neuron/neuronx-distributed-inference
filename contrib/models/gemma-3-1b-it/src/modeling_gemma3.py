# coding=utf-8
# Copyright 2025 Google Inc. and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Gemma3 model for NeuronX Distributed Inference

This implementation ports Google's Gemma3 model to NeuronX hardware.
Key architectural features:
- Q-K normalization (similar to Qwen3)
- Scaled embeddings (embed * sqrt(hidden_size))
- Dual RoPE implementations (global and local for sliding window)
- Four normalization layers per block
- Alternating sliding window attention pattern
- MQA (num_kv_heads=1)
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

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
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


# ====================================================================================
# Configuration Classes
# ====================================================================================


class Gemma3NeuronConfig(NeuronConfig):
    """
    NeuronConfig for Gemma3 model
    Specifies the attention class to use for Gemma3
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use Gemma3-specific attention class
        self.attn_cls = NeuronGemma3Attention


class Gemma3InferenceConfig(InferenceConfig):
    """
    Configuration class for Gemma3 model inference on NeuronX
    
    Inherits from InferenceConfig and adds Gemma3-specific parameters.
    This class handles loading configuration from HuggingFace format.
    """

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1

        # Add required attributes for HF compatibility
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True

        # Add Gemma3-specific parameters with defaults
        if not hasattr(self, "query_pre_attn_scalar"):
            self.query_pre_attn_scalar = 256

        # NOTE: Disabling sliding window for now as the NKI kernel doesn't support head_dim > 128
        # Gemma3 uses head_dim=256 which exceeds this limit
        # TODO: Re-enable when kernel support is added or use alternative implementation
        if not hasattr(self, "sliding_window"):
            self.sliding_window = None  # Disabled for now

        if not hasattr(self, "sliding_window_pattern"):
            self.sliding_window_pattern = 6

        if not hasattr(self, "rope_local_base_freq"):
            self.rope_local_base_freq = 10000

        if not hasattr(self, "attn_logit_softcapping"):
            self.attn_logit_softcapping = None

        if not hasattr(self, "final_logit_softcapping"):
            self.final_logit_softcapping = None

        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        if not hasattr(self, "attention_dropout"):
            self.attention_dropout = 0.0

        # Generate layer_types based on sliding_window_pattern
        # NOTE: Currently all layers use global attention due to head_dim limitation
        if not hasattr(self, "layer_types"):
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                # Disabled sliding window due to head_dim > 128 limitation
                self.layer_types.append("global_attention")

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
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma3NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Gemma3NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Gemma3InferenceConfig":
        """
        Load configuration from a pretrained model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments (including neuron_config)
            
        Returns:
            Gemma3InferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config.json from the model directory
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Override with remaining kwargs
        config_dict.update(kwargs)
        
        # Add required attributes that might not be in HF config
        if "output_attentions" not in config_dict:
            config_dict["output_attentions"] = False
        if "output_hidden_states" not in config_dict:
            config_dict["output_hidden_states"] = False
        if "use_cache" not in config_dict:
            config_dict["use_cache"] = True
        # Gemma3 defaults to tied embeddings
        if "tie_word_embeddings" not in config_dict:
            config_dict["tie_word_embeddings"] = True
        
        # If neuron_config is None, create a default one for validation
        # The actual neuron_config will be loaded from the compiled model during inference
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import NeuronConfig
            neuron_config = NeuronConfig()
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


# ====================================================================================
# Model Components
# ====================================================================================


class Gemma3RMSNorm(nn.Module):
    """
    Gemma3-specific RMSNorm implementation
    
    Key difference from standard RMSNorm:
    - Uses (1.0 + weight) instead of just weight for scaling
    - This is specific to Gemma3 architecture
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3RMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize weight to zeros (Gemma3-specific)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """Root mean square normalization"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Gemma3-specific: use (1.0 + weight) for scaling
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm implementation based on execution mode
    
    Returns:
        Gemma3RMSNorm for CPU mode (CustomRMSNorm doesn't work on CPU)
        CustomRMSNorm for NeuronX mode (optimized for Neuron hardware)
    """
    # For Gemma3, we need to use the custom Gemma3RMSNorm which has
    # the specific (1.0 + weight) scaling. However, CustomRMSNorm doesn't
    # support this yet, so we'll use Gemma3RMSNorm everywhere for now.
    return Gemma3RMSNorm


class Gemma3ScaledEmbedding(nn.Module):
    """
    Gemma3-specific scaled embeddings
    
    Embeddings are multiplied by sqrt(hidden_size) as per Gemma3 architecture.
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3TextScaledWordEmbedding
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        dtype: torch.dtype,
        shard_across_embedding: bool = True,
        pad: bool = True,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()
        self.embed_scale = embedding_dim**0.5
        self.embedding = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            pad=pad,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids: torch.Tensor):
        # Get embeddings and scale by sqrt(hidden_size)
        embeds = self.embedding(input_ids)
        return embeds * self.embed_scale


class NeuronGemma3Attention(NeuronAttentionBase):
    """
    Gemma3 attention mechanism with Q-K normalization
    
    Key features:
    - Q-K normalization after projection (similar to Qwen3)
    - Support for both global and local (sliding window) attention
    - query_pre_attn_scalar for attention score scaling
    - Optional attention logit softcapping
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3Attention
    """

    def __init__(self, config: Gemma3InferenceConfig, is_sliding: bool = False):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Determine which RoPE to use based on attention type
        # Sliding window uses local RoPE with smaller base frequency
        if is_sliding:
            rope_theta = config.rope_local_base_freq
        else:
            rope_theta = config.rope_theta

        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # Determine sliding window size
        sliding_window = config.sliding_window if is_sliding else None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
            # Q-K normalization (like Qwen3)
            q_layernorm=get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(dim=head_dim, eps=config.rms_norm_eps),
        )

        # Store Gemma3-specific parameters
        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.attn_logit_softcapping = config.attn_logit_softcapping


class NeuronGemma3MLP(nn.Module):
    """
    Gemma3 MLP (feed-forward network)
    
    Architecture: gate_proj, up_proj, down_proj with GELU activation
    Similar to LLaMA but uses gelu_pytorch_tanh instead of SiLU
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3MLP
    """

    def __init__(self, config: Gemma3InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate and up projections (column parallel)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        # GELU activation (gelu_pytorch_tanh approximation)
        # This is GELU with tanh approximation as used in Gemma3
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        # Gemma3 MLP: down_proj(act(gate_proj(x)) * up_proj(x))
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_output = self.down_proj(gate_output * up_output)
        return down_output, None  # Return None for compatibility


class NeuronGemma3DecoderLayer(nn.Module):
    """
    Gemma3 decoder layer
    
    Key architectural features:
    - Four normalization layers: input, post_attention, pre_feedforward, post_feedforward
    - Pre-norm architecture with residual connections
    - Support for both global and sliding window attention
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3DecoderLayer
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine if this layer uses sliding window attention
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        # Attention and MLP
        self.self_attn = NeuronGemma3Attention(config, is_sliding=is_sliding)
        self.mlp = NeuronGemma3MLP(config)

        # Four normalization layers (Gemma3-specific)
        self.input_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: attention mask tensor
            position_ids: position indices tensor
            past_key_value: cached key-value pairs for efficient generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Attention block with pre and post normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


# ====================================================================================
# Model Classes
# ====================================================================================


class NeuronGemma3Model(NeuronBaseModel):
    """
    Gemma3 base model for NeuronX inference
    
    This is the main transformer model without the language modeling head.
    Includes embeddings, decoder layers, and final normalization.
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3TextModel
    """

    def setup_attr_for_model(self, config: Gemma3InferenceConfig):
        """Setup attributes for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Gemma3InferenceConfig):
        """Initialize the model components"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Scaled embeddings (Gemma3-specific)
        self.embed_tokens = Gemma3ScaledEmbedding(
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
            [NeuronGemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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


class NeuronGemma3ForCausalLM(NeuronBaseForCausalLM):
    """
    Gemma3 model for causal language modeling on NeuronX
    
    This class wraps NeuronGemma3Model and provides the interface for
    compilation, inference, and weight loading.
    
    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3ForCausalLM
    """

    _model_cls = NeuronGemma3Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """
        Load the HuggingFace Gemma3 model
        
        Note: We import here to avoid dependency issues
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Gemma3 state dict to NeuronX format
        
        Key mappings:
        - embed_tokens.weight -> embed_tokens.embedding.weight
        - layers.*.self_attn.q_norm -> layers.*.self_attn.q_layernorm
        - layers.*.self_attn.k_norm -> layers.*.self_attn.k_layernorm
        - norm.weight -> norm.weight
        - lm_head.weight -> lm_head.weight
        
        Note: The input state_dict already has the "model." prefix stripped by the framework.
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}

        # Handle embeddings with scaling
        if "embed_tokens.weight" in state_dict:
            neuron_state_dict["embed_tokens.embedding.weight"] = (
                state_dict["embed_tokens.weight"].detach().clone()
            )

        # Handle final norm
        if "norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["norm.weight"].detach().clone()

        # Handle lm_head
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].detach().clone()

        # Handle decoder layers
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for i in range(num_layers):
            prefix = f"layers.{i}"  # No "model." prefix needed

            # Attention weights (Q, K, V projections)
            # NOTE: Do NOT rename to qkv_proj.q_proj - the preshard_hook will handle that!
            # Just copy the keys as-is
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"{prefix}.self_attn.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            # Q-K normalization weights (Gemma3-specific)
            if f"{prefix}.self_attn.q_norm.weight" in state_dict:
                neuron_state_dict[f"{prefix}.self_attn.q_layernorm.weight"] = (
                    state_dict[f"{prefix}.self_attn.q_norm.weight"].detach().clone()
                )

            if f"{prefix}.self_attn.k_norm.weight" in state_dict:
                neuron_state_dict[f"{prefix}.self_attn.k_layernorm.weight"] = (
                    state_dict[f"{prefix}.self_attn.k_norm.weight"].detach().clone()
                )

            # MLP weights
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            # Layer normalization weights (four norms per layer)
            for norm_name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]:
                key = f"{prefix}.{norm_name}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key].detach().clone()

            # Add rank information for tensor parallelism in attention
            neuron_state_dict[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank information for vocabulary parallelism
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.embedding.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embeddings and lm_head
        
        In Gemma3, embeddings are tied by default (tie_word_embeddings=True in config)
        Note: The embedding is nested as embed_tokens.embedding.weight due to scaling wrapper
        """
        # Check both possible key locations for embedding weights
        if "embed_tokens.embedding.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.embedding.weight"].clone()
        elif "embed_tokens.weight" in state_dict:
            # Fallback if the embedding hasn't been wrapped yet
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class"""
        return Gemma3InferenceConfig
