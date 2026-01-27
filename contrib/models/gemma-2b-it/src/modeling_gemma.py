# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
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
PyTorch Gemma model for NXD inference

Ported from HuggingFace transformers:
/shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/gemma/modeling_gemma.py

Key architectural features:
- Multi-Query Attention (MQA) with 1 KV head
- RoPE position embeddings
- Unique GemmaRMSNorm: output * (1.0 + weight) instead of output * weight
- GELU activation in MLP (gate_proj(x) * gelu(up_proj(x)))
- Embedding normalization: hidden_states * sqrt(hidden_size)
"""

from typing import List, Optional, Tuple, Type

import torch
import gc
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers import GemmaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Get the appropriate RMSNorm class based on execution mode.
    - If infer on NXD -> CustomRMSNorm
    - If infer on CPU -> HF LlamaRMSNorm (CustomRMSNorm does not work on CPU)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class GemmaRMSNorm(nn.Module):
    """
    Gemma-specific RMSNorm implementation.
    
    Unlike standard RMSNorm which does: output * weight
    Gemma does: output * (1.0 + weight)
    
    Reference: HF GemmaRMSNorm in modeling_gemma.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Gemma-specific: multiply by (1.0 + weight) instead of just weight
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaNeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for Gemma.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemmaAttention


class GemmaInferenceConfig(InferenceConfig):
    """
    Configuration class for Gemma model inference.
    
    Inherits from InferenceConfig and adds Gemma-specific parameters.
    """

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # Gemma does not use bias in attention projections
        self.qkv_bias = False
        self.o_bias = False

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
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
            "head_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[GemmaNeuronConfig]:
        """Return the NeuronConfig class to use."""
        return GemmaNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: Optional[NeuronConfig] = None, **kwargs) -> "GemmaInferenceConfig":
        """
        Load configuration from a pretrained Gemma model.
        
        Args:
            model_path: Path to the HuggingFace model directory
            neuron_config: NeuronConfig instance for compilation settings (can be None for inference)
            **kwargs: Additional configuration overrides
            
        Returns:
            GemmaInferenceConfig instance
        """
        import json
        import os
        
        # Load HuggingFace config.json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace config to our config format
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 2048),
            "intermediate_size": hf_config.get("intermediate_size", 16384),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 18),
            "num_attention_heads": hf_config.get("num_attention_heads", 8),
            "num_key_value_heads": hf_config.get("num_key_value_heads", 1),
            "head_dim": hf_config.get("head_dim", 256),
            "vocab_size": hf_config.get("vocab_size", 256000),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 8192),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "pad_token_id": hf_config.get("pad_token_id", 0),
            "bos_token_id": hf_config.get("bos_token_id", 2),
            "eos_token_id": hf_config.get("eos_token_id", 1),
        }
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        # Add common HuggingFace config attributes that may be expected
        if "output_attentions" not in config_dict:
            config_dict["output_attentions"] = False
        if "output_hidden_states" not in config_dict:
            config_dict["output_hidden_states"] = False
        if "return_dict" not in config_dict:
            config_dict["return_dict"] = True
        
        # If neuron_config is not provided, we need to create a minimal one or skip validation
        # During inference loading, the neuron_config will be loaded separately
        if neuron_config is None:
            # For inference, load config without full validation
            # The neuron_config will be loaded from saved artifacts
            config = cls.__new__(cls)
            config.neuron_config = None
            config.fused_spec_config = None
            config.metadata = None
            for key, value in config_dict.items():
                setattr(config, key, value)
            # Skip add_derived_config and validate_config when neuron_config is None
            return config
        
        # Create config instance with full initialization
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronGemmaAttention(NeuronAttentionBase):
    """
    Gemma attention implementation for NeuronX.
    
    Features:
    - Multi-Query Attention (MQA): 8 query heads, 1 key-value head
    - RoPE position embeddings
    - No bias in projections
    
    Reference: GemmaAttention in modeling_gemma.py
    """

    def __init__(self, config: GemmaInferenceConfig):
        rotary_emb = RotaryEmbedding(
            config.head_dim,  # Use head_dim directly
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=False,  # Gemma does not use bias
            o_bias=False,
            rotary_emb=rotary_emb,
        )


class NeuronGemmaMLP(nn.Module):
    """
    Gemma MLP implementation for NeuronX.
    
    Architecture: gelu(gate_proj(x)) * up_proj(x) -> down_proj
    
    Unlike LLaMA which uses SwiGLU (silu(gate) * up),
    Gemma uses: gelu(gate) * up
    
    Reference: GemmaMLP in modeling_gemma.py
    """

    def __init__(self, config: GemmaInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        # Gemma uses GELU activation (tanh approximation)
        # Config specifies "gelu" which maps to tanh approximation in HF
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        # Gemma-specific: gelu(gate) * up
        # Different from LLaMA's: silu(gate) * up
        # Reference: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # Apply GELU to gate, then multiply with up
        intermediate_output = self.act_fn(gate_output) * up_output
        
        # Apply down projection
        output = self.down_proj(intermediate_output)
        
        return output, None  # Return None as second output for compatibility


class NeuronGemmaDecoderLayer(nn.Module):
    """
    Gemma decoder layer implementation for NeuronX.
    
    Structure:
    1. Input LayerNorm
    2. Self Attention
    3. Residual connection
    4. Post-attention LayerNorm
    5. MLP
    6. Residual connection
    
    Reference: GemmaDecoderLayer in modeling_gemma.py
    """

    def __init__(self, config: GemmaInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronGemmaAttention(config)
        self.mlp = NeuronGemmaMLP(config)
        
        # Use Gemma-specific RMSNorm
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
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
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key-value pairs for autoregressive generation
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
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

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class GemmaNormalizedEmbedding(ParallelEmbedding):
    """
    Gemma-specific embedding that applies normalization after embedding lookup.
    
    Gemma normalizes embeddings by sqrt(hidden_size) after embedding lookup.
    Reference: GemmaModel.forward in modeling_gemma.py
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, dtype, shard_across_embedding, pad, hidden_size):
        super().__init__(num_embeddings, embedding_dim, padding_idx, dtype=dtype, 
                        shard_across_embedding=shard_across_embedding, pad=pad)
        self.normalizer = hidden_size ** 0.5
    
    def forward(self, input_ids, **kwargs):
        """Forward pass with Gemma normalization."""
        embeddings = super().forward(input_ids, **kwargs)
        # Apply Gemma normalization: multiply by sqrt(hidden_size)
        return embeddings * self.normalizer


class NeuronGemmaModel(NeuronBaseModel):
    """
    Gemma model implementation for NeuronX.
    
    Key features:
    - Embedding normalization by sqrt(hidden_size)
    - Multi-Query Attention with 1 KV head
    - Gemma-specific RMSNorm
    - GELU activation in MLP
    
    Reference: GemmaModel in modeling_gemma.py
    """

    def setup_attr_for_model(self, config: GemmaInferenceConfig):
        """Setup attributes for model initialization."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
        # Gemma-specific: normalizer for embeddings
        self.normalizer = self.hidden_size ** 0.5

    def init_model(self, config: GemmaInferenceConfig):
        """Initialize the model layers."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Create Gemma-specific embedding with normalization
        self.embed_tokens = GemmaNormalizedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            hidden_size=config.hidden_size,
        )
        
        self.layers = nn.ModuleList(
            [NeuronGemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Use Gemma-specific RMSNorm for final layer
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronGemmaForCausalLM(NeuronBaseForCausalLM):
    """
    Gemma causal language model for inference.
    
    This class can be used as a drop-in replacement for GemmaForCausalLM.
    """

    _model_cls = NeuronGemmaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace Gemma model for weight conversion."""
        return GemmaForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Gemma weights to NeuronX format.
        
        Weight name mappings:
        - model.embed_tokens.weight -> embed_tokens.weight (GemmaNormalizedEmbedding inherits from ParallelEmbedding)
        - model.layers.{i}.* -> layers.{i}.*
        - model.norm.weight -> norm.weight
        - lm_head.weight -> lm_head.weight (tied to embed_tokens)
        
        Args:
            state_dict: HuggingFace model state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}

        # Handle vocab parallel for embeddings
        if neuron_config.vocab_parallel:
            state_dict["model.embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Convert model weights (remove "model." prefix)
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
                neuron_state_dict[new_key] = value.clone()
            else:
                # Keep lm_head.weight as is (if it exists in HF checkpoint)
                neuron_state_dict[key] = value.clone()

        # Gemma ties embed_tokens and lm_head - explicitly create lm_head.weight
        # if it doesn't exist (which is the case for Gemma)
        if "lm_head.weight" not in neuron_state_dict and "embed_tokens.weight" in neuron_state_dict:
            neuron_state_dict["lm_head.weight"] = neuron_state_dict["embed_tokens.weight"].clone()
            print("✅ Tied lm_head.weight to embed_tokens.weight")

        # Add rank information for tensor parallelism in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Handle fused QKV if enabled
        if neuron_config.fused_qkv:
            neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied embeddings and LM head.
        
        In Gemma, embeddings and LM head can share weights.
        GemmaNormalizedEmbedding inherits from ParallelEmbedding, so weight is at embed_tokens.weight
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class."""
        return GemmaInferenceConfig

    def get_compiler_args(self):
        """
        Get compiler arguments for Gemma model compilation.
        
        Returns optimized compiler settings for Neuron.
        """
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        # Add flags for compute-communication overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args


def _helper_concat_and_delete_qkv(gemma_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fused QKV.
    
    Args:
        gemma_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight', 'bias', or 'scale')
    """
    qkv_parts = []
    keys_to_delete = []
    
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        key = f"layers.{layer_num}.self_attn.{proj}.{attr}"
        if key in gemma_state_dict:
            qkv_parts.append(gemma_state_dict[key])
            keys_to_delete.append(key)
    
    if qkv_parts:
        gemma_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(qkv_parts)
        for key in keys_to_delete:
            del gemma_state_dict[key]


def convert_state_dict_to_fused_qkv(gemma_state_dict, cfg: InferenceConfig):
    """
    Convert separate QKV weights to fused QKV format.
    
    This function concatenates the q, k, v projection weights into a single
    Wqkv weight for more efficient computation.
    
    Args:
        gemma_state_dict: State dictionary with separate QKV weights
        cfg: Model configuration
        
    Returns:
        Updated state dictionary with fused QKV weights
    """
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for layer_idx in range(cfg.num_hidden_layers):
        if f"layers.{layer_idx}.self_attn" not in mods_to_not_conv:
            # Concatenate weight
            _helper_concat_and_delete_qkv(gemma_state_dict, layer_idx, "weight")
            
            # Note: Gemma does not use bias in attention, so we skip bias concatenation

    gc.collect()
    return gemma_state_dict
