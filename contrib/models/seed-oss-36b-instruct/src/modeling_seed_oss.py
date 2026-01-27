# coding=utf-8
# Copyright 2025 Bytedance-Seed Ltd and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Seed-OSS model for NXD inference
Based on /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/seed_oss/modeling_seed_oss.py
"""
from typing import List, Optional, Tuple, Type

import torch
import gc
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class SeedOssNeuronConfig(NeuronConfig):
    """
    NeuronConfig for Seed-OSS model with attention class specification
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronSeedOssAttention


class SeedOssInferenceConfig(InferenceConfig):
    """
    Configuration class for Seed-OSS model inference
    
    Based on Seed-OSS configuration from:
    /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/seed_oss/configuration_seed_oss.py
    
    Key features:
    - attention_bias: True (Q/K/V projections use bias)
    - attention_out_bias: False (output projection has no bias)
    - mlp_bias: False (MLP layers have no bias)
    - attention_dropout: 0.1 (dropout in attention - not used during inference)
    - residual_dropout: 0.1 (dropout in residual connections - not used during inference)
    - rope_theta: 10000000.0 (very large for long context support)
    - head_dim: 128 (explicit head dimension)
    """

    def add_derived_config(self):
        """Add derived configuration parameters specific to Seed-OSS"""
        self.num_cores_per_group = 1
        
        # Seed-OSS specific attention configuration
        self.qkv_bias = getattr(self, "attention_bias", True)
        self.o_bias = getattr(self, "attention_out_bias", False)
        
        # MLP configuration
        self.mlp_bias = getattr(self, "mlp_bias", False)
        
        # Dropout values (not used during inference, but needed for compatibility)
        self.attention_dropout = getattr(self, "attention_dropout", 0.1)
        self.residual_dropout = getattr(self, "residual_dropout", 0.1)
        
        # Ensure head_dim is set
        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Add standard transformer config attributes
        self.output_attentions = getattr(self, "output_attentions", False)
        self.output_hidden_states = getattr(self, "output_hidden_states", False)
        self.return_dict = getattr(self, "return_dict", True)

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for Seed-OSS configuration"""
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
    def get_neuron_config_cls(cls) -> Type[SeedOssNeuronConfig]:
        """Return the NeuronConfig class to use for Seed-OSS"""
        return SeedOssNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from a pretrained Seed-OSS model directory
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional configuration parameters to override
            
        Returns:
            SeedOssInferenceConfig: Configuration object
        """
        import json
        import os
        
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config.json from model directory
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        # If neuron_config is None, create a dummy one to pass validation
        # (it will be replaced later by the inference runner)
        if neuron_config is None:
            from neuronx_distributed_inference.models.config import NeuronConfig
            import torch
            neuron_config = NeuronConfig(
                tp_degree=1,
                batch_size=1,
                seq_len=128,
                torch_dtype=torch.bfloat16,
            )
        
        # Create and return config object
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronSeedOssAttention(NeuronAttentionBase):
    """
    Seed-OSS attention implementation for NeuronX
    
    Based on SeedOssAttention from:
    /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/seed_oss/modeling_seed_oss.py
    
    Key differences from standard attention:
    - Uses bias in Q/K/V projections (attention_bias=True)
    - No bias in output projection (attention_out_bias=False)
    - Uses GQA with 80 query heads and 8 KV heads
    - Very large rope_theta (10M) for long context
    """

    def __init__(self, config: SeedOssInferenceConfig):
        # Create rotary embeddings with Seed-OSS specific parameters
        rotary_emb = RotaryEmbedding(
            config.head_dim,  # Use explicit head_dim
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,  # Very large theta: 10000000.0
        )

        # Initialize base attention with Seed-OSS specific parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,  # Explicit head_dim=128
            qkv_bias=config.qkv_bias,  # True for Seed-OSS
            o_bias=config.o_bias,      # False for Seed-OSS
            rotary_emb=rotary_emb,
        )


class NeuronSeedOssDecoderLayer(nn.Module):
    """
    Seed-OSS decoder layer implementation
    
    Based on SeedOssDecoderLayer from:
    /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/seed_oss/modeling_seed_oss.py
    
    Structure:
    - Input LayerNorm (RMSNorm)
    - Self Attention (with residual connection)
    - Post-Attention LayerNorm (RMSNorm)
    - MLP (with residual connection)
    
    Note: Original implementation has attention_dropout and residual_dropout,
    but these are not used during inference.
    """

    def __init__(self, config: SeedOssInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention layer
        self.self_attn = NeuronSeedOssAttention(config)
        
        # MLP layer - reuse LlamaMLP (same SwiGLU structure with configurable bias)
        self.mlp = NeuronLlamaMLP(config)
        
        # Layer normalization layers
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
        Forward pass for Seed-OSS decoder layer
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs for positional encoding
            past_key_value: Cached key-value pairs for efficient generation
            
        Returns:
            Tuple containing:
            - hidden_states: Output tensor
            - present_key_value: Updated key-value cache
            - cos_cache: Cosine cache for RoPE
            - sin_cache: Sine cache for RoPE
            - None: Placeholder for compatibility
        """
        # Pre-attention normalization
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
        
        # Residual connection (dropout not applied during inference)
        hidden_states = residual + hidden_states

        # Pre-MLP normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)[0]
        
        # Residual connection (dropout not applied during inference)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronSeedOssModel(NeuronBaseModel):
    """
    Seed-OSS model implementation for NeuronX
    
    Based on SeedOssModel from:
    /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/seed_oss/modeling_seed_oss.py
    
    Architecture:
    - Token embeddings (vocab_size=155136, hidden_size=5120)
    - 64 decoder layers
    - Final normalization (RMSNorm)
    - LM head for token generation
    """

    def setup_attr_for_model(self, config: SeedOssInferenceConfig):
        """Setup attributes required for model initialization"""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: SeedOssInferenceConfig):
        """Initialize the Seed-OSS model components"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings with vocabulary parallelism
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )
        
        # Decoder layers (64 layers for 36B model)
        self.layers = nn.ModuleList(
            [NeuronSeedOssDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head for token generation
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # Seed-OSS does not use bias in lm_head
            pad=True,
            gather_output=not self.on_device_sampling,
        )


class NeuronSeedOssForCausalLM(NeuronBaseForCausalLM):
    """
    Seed-OSS causal language model for NeuronX inference
    
    This class provides the main interface for:
    - Loading HuggingFace checkpoints
    - Converting weights to NeuronX format
    - Compilation and inference
    """

    _model_cls = NeuronSeedOssModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace Seed-OSS model for weight extraction"""
        # Import dynamically to avoid dependencies
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Seed-OSS weights to NeuronX format
        
        Weight mapping:
        HF Format -> NeuronX Format
        - model.embed_tokens.weight -> embed_tokens.weight
        - model.layers.{i}.* -> layers.{i}.*
        - model.norm.weight -> norm.weight
        - lm_head.weight -> lm_head.weight
        
        For attention layers:
        - self_attn.q_proj.* -> self_attn.q_proj.*
        - self_attn.k_proj.* -> self_attn.k_proj.*
        - self_attn.v_proj.* -> self_attn.v_proj.*
        - self_attn.o_proj.* -> self_attn.o_proj.*
        
        For MLP layers:
        - mlp.gate_proj.* -> mlp.gate_proj.*
        - mlp.up_proj.* -> mlp.up_proj.*
        - mlp.down_proj.* -> mlp.down_proj.*
        """
        neuron_config = config.neuron_config
        neuron_state_dict = {}
        
        # Process each key in the state dict
        for key, value in state_dict.items():
            new_key = key
            
            # Remove 'model.' prefix if present (HF format)
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model."
            
            # Copy the weight
            neuron_state_dict[new_key] = value.clone()
        
        # Add rank information for tensor parallelism in embeddings
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Add rank information for attention in each layer
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Handle fused QKV if enabled
        if neuron_config.fused_qkv:
            neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Update state dict for tied embeddings
        
        Note: Seed-OSS has tie_word_embeddings=False, so this may not be needed,
        but we provide it for compatibility.
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for Seed-OSS"""
        return SeedOssInferenceConfig

    def get_compiler_args(self):
        """
        Get compiler arguments for Seed-OSS model compilation
        
        Based on Qwen2 compiler args with optimizations for:
        - Mixed precision accumulation
        - Saturate infinity handling
        - Compute-overlap optimizations
        """
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"
        
        # Add flags for compute-communication overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
        
        # Add HLO verification
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        
        return compiler_args


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fused QKV (weight or bias).
    
    Args:
        state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'bias')
    """
    # Concatenate Q, K, V weights/biases
    qkv_components = []
    for proj in ["q_proj", "k_proj", "v_proj"]:
        key = f"layers.{layer_num}.self_attn.{proj}.{attr}"
        if key in state_dict:
            qkv_components.append(state_dict[key])
    
    if qkv_components:
        # Create fused QKV
        state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(qkv_components)
        
        # Delete individual Q, K, V weights/biases
        for proj in ["q_proj", "k_proj", "v_proj"]:
            key = f"layers.{layer_num}.self_attn.{proj}.{attr}"
            if key in state_dict:
                del state_dict[key]


def convert_state_dict_to_fused_qkv(state_dict, cfg: InferenceConfig):
    """
    Convert state dict to fused QKV format
    
    This function concatenates the Q, K, V weights and biases into a single Wqkv tensor
    for more efficient computation with fused QKV kernels.
    
    Args:
        state_dict: State dictionary to convert
        cfg: Model configuration
        
    Returns:
        Updated state dictionary with fused QKV weights
    """
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for layer_idx in range(cfg.num_hidden_layers):
        if f"layers.{layer_idx}.self_attn" not in mods_to_not_conv:
            # Fuse weights
            _helper_concat_and_delete_qkv(state_dict, layer_idx, "weight")
            
            # Fuse biases (Seed-OSS has attention_bias=True)
            _helper_concat_and_delete_qkv(state_dict, layer_idx, "bias")
            
            # Handle quantization scales if present
            if (cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized):
                if f"layers.{layer_idx}.self_attn.q_proj.scale" in state_dict:
                    _helper_concat_and_delete_qkv(state_dict, layer_idx, "scale")

    gc.collect()
    return state_dict
