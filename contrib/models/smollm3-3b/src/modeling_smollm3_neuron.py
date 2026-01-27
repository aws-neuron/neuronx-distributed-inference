"""
SmolLM3 model implementation for NeuronX Distributed Inference

This implementation is based on:
- Original SmolLM3 from transformers: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/smollm3/
- NeuronX LLaMA implementation patterns from NeuronxDistributedInference

Key architectural features of SmolLM3:
1. LLaMA-like architecture with GQA (4 KV heads, 16 Q heads)
2. SwiGLU activation in MLP
3. RMSNorm for layer normalization
4. NoPE layers - Every 4th layer does NOT use RoPE (unique to SmolLM3!)
5. Tied embeddings between input and output
6. No bias in attention or MLP layers
"""

import json
import logging
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group

# Import RMSNorm from transformers for CPU mode
try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as SmolLM3RMSNorm
except ImportError:
    # Fallback if transformers not available
    SmolLM3RMSNorm = None

logger = logging.getLogger(__name__)

# Activation function mapping
ACT2FN = {
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_rmsnorm_cls():
    """
    Get appropriate RMSNorm implementation
    - NXD/Neuron: CustomRMSNorm (optimized)
    - CPU: SmolLM3RMSNorm (from transformers)
    """
    return SmolLM3RMSNorm if cpu_mode() else CustomRMSNorm


def get_tp_group(config: InferenceConfig):
    """Get tensor parallel group based on configuration"""
    # For now, return None to use default group
    # This can be customized if needed
    return None


class SmolLM3InferenceConfig(InferenceConfig):
    """
    Configuration class for SmolLM3 model inference on NeuronX
    
    Extends InferenceConfig with SmolLM3-specific parameters including
    NoPE (No Position Embedding) layer configuration.
    """
    
    # Set default values for HF-compatible attributes
    output_attentions = False
    output_hidden_states = False
    use_cache = True

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        # Check if neuron_config exists and flash_decoding_enabled
        if hasattr(self, 'neuron_config') and self.neuron_config and getattr(self.neuron_config, 'flash_decoding_enabled', False):
            num_attn_heads = self.num_attention_heads
            num_kv_heads = self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration"""
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
            # SmolLM3-specific attributes
            "no_rope_layers",
            "no_rope_layer_interval",
            "layer_types",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use"""
        return NeuronConfig

    @classmethod  
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from HuggingFace model directory
        
        This method reads config.json and creates a SmolLM3InferenceConfig.
        During inference, neuron_config will be set later by the framework.
        """
        import json
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Extract neuron_config if passed in kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        hf_config.update(kwargs)
        
        # Pass neuron_config (may be None initially)
        return cls(neuron_config=neuron_config, **hf_config)
    
    def validate_config(self):
        """
        Validate configuration - override to handle None neuron_config gracefully
        """
        # Only validate if neuron_config is set
        if self.neuron_config is not None:
            super().validate_config()
        # Otherwise skip validation (will be validated after neuron_config is set)


class NeuronSmolLM3MLP(nn.Module):
    """
    SmolLM3 MLP implementation for NeuronX
    
    Uses SwiGLU activation: down_proj(silu(gate_proj(x)) * up_proj(x))
    This is identical to LLaMA MLP architecture.
    """

    def __init__(self, config: SmolLM3InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.fused_rmsnorm_skip_gamma = self.config.neuron_config.fused_rmsnorm_skip_gamma
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound
        self.logical_nc_config = self.neuron_config.logical_nc_config
        self.activation_quantization_type = self.neuron_config.activation_quantization_type
        mlp_bias = getattr(config, "mlp_bias", False)

        if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float("inf"):
            logging.warning(
                "quantize_clamp_bound not specified. Using default 1200 for SmolLM3 quantized kernels."
            )
            self.quantize_clamp_bound = 1200.0

        if parallel_state.model_parallel_is_initialized():
            if self.neuron_config.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
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
                bias=mlp_bias,
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
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def forward(self, hidden_states):
        """
        Forward pass of MLP with SwiGLU activation
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tuple of (output, None) - None for compatibility with other modules
        """
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        
        # Apply activation to gate and multiply with up
        intermediate = self.act_fn(gate_output) * up_output
        
        # Project back down
        output = self.down_proj(intermediate)
        
        return output, None


class NeuronSmolLM3Attention(NeuronAttentionBase):
    """
    SmolLM3 attention implementation for NeuronX
    
    Key features:
    - GQA with 4 KV heads, 16 Q heads
    - Conditional RoPE based on layer index (NoPE layers)
    - No bias in projections
    - Based on NeuronAttentionBase for flash attention support
    """

    def __init__(self, config: SmolLM3InferenceConfig, layer_idx: int):
        """
        Initialize SmolLM3 attention layer
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer (used for NoPE determination)
        """
        self.layer_idx = layer_idx
        self.config = config
        
        # Check if this layer uses RoPE (NoPE layers have 0 in no_rope_layers)
        self.use_rope = config.no_rope_layers[layer_idx] if config.no_rope_layers else True
        
        # Create RoPE embeddings only if this layer uses them
        rotary_emb = None
        if self.use_rope:
            head_dim = config.hidden_size // config.num_attention_heads
            rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
            logger.debug(f"Layer {layer_idx}: RoPE enabled with theta={config.rope_theta}")
        else:
            logger.debug(f"Layer {layer_idx}: NoPE layer (no RoPE)")
        
        # Check for sliding window attention
        sliding_window = None
        if config.use_sliding_window and config.sliding_window is not None:
            if config.layer_types and config.layer_types[layer_idx] == "sliding_attention":
                sliding_window = config.sliding_window
                logger.debug(f"Layer {layer_idx}: Sliding window attention enabled (window={sliding_window})")
        
        # Initialize base attention module
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta,
            use_scaled_rope=False,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
        )


class NeuronSmolLM3DecoderLayer(nn.Module):
    """
    SmolLM3 decoder layer implementation
    
    Architecture:
    - Pre-norm with RMSNorm
    - Self-attention with residual connection
    - MLP with residual connection
    """

    def __init__(self, config: SmolLM3InferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Get appropriate RMSNorm implementation
        rms_norm_cls = get_rmsnorm_cls()
        
        # Attention and normalization
        self.self_attn = NeuronSmolLM3Attention(config, layer_idx)
        self.input_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP and normalization
        self.mlp = NeuronSmolLM3MLP(config)
        self.post_attention_layernorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        **kwargs,
    ):
        """
        Forward pass of decoder layer
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value pairs
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        # Attention returns NeuronAttentionBaseOutput with hidden_states and present_key_value
        hidden_states = attn_output.hidden_states
        present_key_value = attn_output.present_key_value
        cos_cache = attn_output.cos_cache
        sin_cache = attn_output.sin_cache
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return format expected by NeuronBaseModel
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs


class NeuronSmolLM3Model(NeuronBaseModel):
    """
    SmolLM3 base model implementation for NeuronX
    
    This is the core transformer model without the language modeling head.
    """

    def setup_attr_for_model(self, config: SmolLM3InferenceConfig):
        """Setup attributes needed for model initialization"""
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", None)

    def init_model(self, config: SmolLM3InferenceConfig):
        """Initialize model layers and components"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Get appropriate RMSNorm implementation
        rms_norm_cls = get_rmsnorm_cls()
        
        # Token embeddings and LM head
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                tensor_model_parallel_group=get_tp_group(config),
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
                padding_idx=self.padding_idx,
            )
            
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        
        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronSmolLM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)


class NeuronSmolLM3ForCausalLM(NeuronBaseForCausalLM):
    """
    SmolLM3 model with language modeling head for causal LM
    
    This wraps the base model and adds the output projection for text generation.
    SmolLM3 uses tied embeddings, so lm_head shares weights with embed_tokens.
    """

    _model_cls = NeuronSmolLM3Model

    @classmethod
    def from_config(cls, config: SmolLM3InferenceConfig):
        """
        Create model from configuration
        
        Args:
            config: Model configuration
            
        Returns:
            NeuronSmolLM3ForCausalLM instance
        """
        return cls(config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied embeddings for SmolLM3
        
        SmolLM3 ties the input embeddings with the output lm_head weights.
        This method ensures lm_head.weight is set to embed_tokens.weight.
        
        Args:
            state_dict: Model state dictionary to update
        """
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        elif "lm_head.weight" in state_dict and "embed_tokens.weight" in state_dict:
            # Both exist, use embed_tokens for tied weights
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return SmolLM3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config: SmolLM3InferenceConfig):
        """
        Convert HuggingFace state dict to NeuronX format
        
        Weight name mapping:
        HF Format                          -> NeuronX Format
        ---------------------------------------------
        model.embed_tokens.weight          -> model.embed_tokens.weight
        model.layers.N.self_attn.q_proj    -> model.layers.N.self_attn.qkv_proj.q_proj
        model.layers.N.self_attn.k_proj    -> model.layers.N.self_attn.qkv_proj.k_proj
        model.layers.N.self_attn.v_proj    -> model.layers.N.self_attn.qkv_proj.v_proj
        model.layers.N.self_attn.o_proj    -> model.layers.N.self_attn.o_proj
        model.layers.N.mlp.gate_proj       -> model.layers.N.mlp.gate_proj
        model.layers.N.mlp.up_proj         -> model.layers.N.mlp.up_proj
        model.layers.N.mlp.down_proj       -> model.layers.N.mlp.down_proj
        model.layers.N.input_layernorm     -> model.layers.N.input_layernorm
        model.layers.N.post_attention_layernorm -> model.layers.N.post_attention_layernorm
        model.norm.weight                  -> model.norm.weight
        lm_head.weight                     -> lm_head.weight (or tied to embed_tokens)
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_state_dict = {}
        
        print(f"Converting HF checkpoint to NeuronX format...")
        print(f"Total keys in HF checkpoint: {len(state_dict)}")
        
        # Handle tied embeddings
        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            print("Using tied embeddings: lm_head will share weights with embed_tokens")
        
        for key, value in state_dict.items():
            new_key = key
            
            # Convert attention projection keys
            if ".self_attn.q_proj" in key:
                new_key = key.replace(".self_attn.q_proj", ".self_attn.qkv_proj.q_proj")
            elif ".self_attn.k_proj" in key:
                new_key = key.replace(".self_attn.k_proj", ".self_attn.qkv_proj.k_proj")
            elif ".self_attn.v_proj" in key:
                new_key = key.replace(".self_attn.v_proj", ".self_attn.qkv_proj.v_proj")
            
            # Copy weight
            neuron_state_dict[new_key] = value.clone()
            
            if new_key != key:
                logger.debug(f"Mapped: {key} -> {new_key}")
        
        # Handle tied embeddings if lm_head.weight not in checkpoint
        if config.tie_word_embeddings and "lm_head.weight" not in neuron_state_dict:
            if "model.embed_tokens.weight" in neuron_state_dict:
                neuron_state_dict["lm_head.weight"] = neuron_state_dict["model.embed_tokens.weight"]
                print("Tied lm_head.weight to model.embed_tokens.weight")
        
        print(f"Total keys in NeuronX checkpoint: {len(neuron_state_dict)}")
        
        return neuron_state_dict


# Export classes
__all__ = [
    "SmolLM3InferenceConfig",
    "NeuronSmolLM3Model",
    "NeuronSmolLM3ForCausalLM",
    "NeuronSmolLM3Attention",
    "NeuronSmolLM3MLP",
    "NeuronSmolLM3DecoderLayer",
]
