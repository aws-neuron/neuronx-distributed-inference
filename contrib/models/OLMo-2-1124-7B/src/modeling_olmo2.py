# coding=utf-8
# Copyright 2024 Allen AI and NeuronX Port
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
PyTorch OLMo2 model for NXD inference.

This module ports the OLMo-2-1124-7B model to NeuronX Distributed Inference.
Key architectural differences from LLaMA:
1. Post-layer normalization (RMSNorm after attention and MLP, not before)
2. Q-K normalization (RMSNorm on Q and K projections before RoPE)

Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/olmo2/modeling_olmo2.py
"""

import os
import json
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP, get_rmsnorm_cls
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


# ============================================================================
# Custom RMSNorm with TP Sharding Support
# ============================================================================

from neuronx_distributed.parallel_layers.layers import BaseParallelLinear
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes


class ShardedRMSNorm(BaseParallelLinear):
    """
    RMSNorm that supports tensor parallel sharding with correct variance computation.
    
    This is needed for OLMo2's Q-K normalization where the norm is applied
    BEFORE reshaping to heads. Since Q/K projections are sharded across TP,
    the norm weights must also be sharded.
    
    CRITICAL: The variance must be computed over the FULL dimension (4096),
    not the sharded dimension (512). This requires an all-reduce across TP ranks
    to sum the squared values before computing the mean.
    
    By inheriting from BaseParallelLinear, this module is recognized by the
    framework's shard_children function and will have its weights properly
    sharded across TP ranks.
    """
    
    def __init__(self, hidden_size: int, full_hidden_size: int, eps: float = 1e-6, tp_degree: int = 1):
        super().__init__(device=None)
        self.hidden_size = hidden_size  # Sharded size (per-rank)
        self.full_hidden_size = full_hidden_size  # Full size (before sharding)
        self.eps = eps
        self.tp_degree = tp_degree
        
        # Create weight with SHARDED size - this is what the forward pass uses
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Mark the weight for tensor parallel sharding
        # This tells shard_children how to shard the checkpoint weight
        # The checkpoint has full_hidden_size, and we want to shard it into tp_degree parts
        set_tensor_model_parallel_attributes(
            tensor=self.weight,
            is_parallel=True,
            dim=0,  # Shard along dimension 0
            stride=1,  # Contiguous sharding
            num_partitions=tp_degree,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm with correct variance computation across TP ranks.
        
        The variance must be computed over the FULL dimension, not the sharded dimension.
        This is done by:
        1. Computing sum of squares locally (over sharded dimension)
        2. All-reduce to get global sum of squares
        3. Divide by full dimension size to get variance
        4. Apply normalization with the correct variance
        """
        from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
        
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute local sum of squares (not mean yet!)
        local_sum_sq = x.pow(2).sum(-1, keepdim=True)
        
        # All-reduce to get global sum of squares across all TP ranks
        # This is needed because variance should be computed over the FULL dimension
        # Use reduce_from_tensor_model_parallel_region which is the standard NeuronX way
        if self.tp_degree > 1:
            global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)
        else:
            global_sum_sq = local_sum_sq
        
        # Compute variance as mean of squares over FULL dimension
        variance = global_sum_sq / self.full_hidden_size
        
        # Apply RMSNorm: x * rsqrt(variance + eps) * weight
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ============================================================================
# Configuration Classes
# ============================================================================

class Olmo2NeuronConfig(NeuronConfig):
    """
    NeuronConfig subclass for OLMo2 model.
    
    Sets up the attention class to use NeuronOlmo2Attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronOlmo2Attention


class Olmo2InferenceConfig(InferenceConfig):
    """
    InferenceConfig for OLMo2 model.
    
    Configuration class to store the configuration of an OLMo2 model for NeuronX inference.
    This class handles loading configuration from HuggingFace format and setting up
    the required attributes for inference.
    
    Reference: /shared/dhwanw/agent_friday_test/example/transformers/src/transformers/models/olmo2/configuration_olmo2.py
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
    
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
    def get_neuron_config_cls(cls) -> Type[Olmo2NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return Olmo2NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Olmo2InferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        This method reads the config.json file from the HuggingFace model directory
        and creates an Olmo2InferenceConfig object with the appropriate parameters.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration, including neuron_config
            
        Returns:
            Olmo2InferenceConfig: Configuration object for the model
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read the HuggingFace config.json file
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)
        
        # Map HuggingFace config to our config format
        # Reference: /shared/dhwanw2/models/OLMo-2-1124-7B/config.json
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 4096),
            "num_attention_heads": hf_config.get("num_attention_heads", 32),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 32),
            "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", 32)),
            "vocab_size": hf_config.get("vocab_size", 100352),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 4096),
            "rope_theta": hf_config.get("rope_theta", 500000.0),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "intermediate_size": hf_config.get("intermediate_size", 11008),
            "attention_bias": hf_config.get("attention_bias", False),
            "pad_token_id": hf_config.get("pad_token_id", 100277),
            "bos_token_id": hf_config.get("bos_token_id", None),
            "eos_token_id": hf_config.get("eos_token_id", 100257),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
            # Standard HuggingFace config attributes required by the framework
            "output_attentions": False,
            "output_hidden_states": False,
            "use_cache": hf_config.get("use_cache", True),
        }
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        # Create and return the config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


# ============================================================================
# Attention Classes
# ============================================================================

class NeuronOlmo2Attention(NeuronAttentionBase):
    """
    OLMo2 Attention implementation for NeuronX.
    
    Key differences from LLaMA attention:
    - Applies RMSNorm to Q and K projections BEFORE reshaping to heads
    - In OLMo2: q_norm operates on (batch, seq, num_heads * head_dim)
    - This is different from Qwen3's per-head normalization
    
    IMPORTANT: For TP > 1, we use ShardedRMSNorm which has a preshard_hook
    that handles extracting the correct slice of weights for each TP rank
    during checkpoint loading. This allows the framework to properly shard
    the q_norm/k_norm weights even though they're not in __SUPPORTED_SHARDED_MODULES.
    
    Reference: Olmo2Attention in modeling_olmo2.py
    - self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
    - self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
    - query_states = self.q_norm(self.q_proj(hidden_states))
    - key_states = self.k_norm(self.k_proj(hidden_states))
    """
    
    def __init__(self, config: Olmo2InferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tp_degree = config.neuron_config.tp_degree
        
        # Create rotary embedding for position encoding
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Initialize base attention WITHOUT q_layernorm/k_layernorm
        # We'll handle Q-K normalization ourselves since OLMo2 applies it
        # BEFORE reshaping to heads (different from Qwen3's per-head norm)
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            # No per-head layernorm - we handle normalization differently
            q_layernorm=None,
            k_layernorm=None,
            # OLMo2 uses no bias in attention projections
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
        )
        
        # OLMo2-specific: RMSNorm on Q and K projections (before head reshape)
        # We use ShardedRMSNorm which has a preshard_hook to handle TP sharding
        # during checkpoint loading. The norm weights are sharded to match the
        # sharded Q/K projection outputs.
        sharded_q_dim = (config.num_attention_heads // tp_degree) * head_dim
        sharded_k_dim = (config.num_key_value_heads // tp_degree) * head_dim
        full_q_dim = config.num_attention_heads * head_dim
        full_k_dim = config.num_key_value_heads * head_dim
        
        self.q_norm = ShardedRMSNorm(
            hidden_size=sharded_q_dim,
            full_hidden_size=full_q_dim,
            eps=config.rms_norm_eps,
            tp_degree=tp_degree,
        )
        self.k_norm = ShardedRMSNorm(
            hidden_size=sharded_k_dim,
            full_hidden_size=full_k_dim,
            eps=config.rms_norm_eps,
            tp_degree=tp_degree,
        )
    
    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """
        Override prep_qkv_tensors to apply OLMo2-style Q-K normalization.
        
        OLMo2 applies RMSNorm to Q and K projections BEFORE reshaping to heads,
        which is different from the base class behavior.
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
            move_heads_front,
        )
        
        # Get QKV projections through the base class's qkv_proj
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )
        
        # OLMo2-specific: Apply RMSNorm to Q and K BEFORE reshaping to heads
        # Q shape at this point: (batch, seq, num_heads/tp * head_dim)
        # K shape at this point: (batch, seq, num_kv_heads/tp * head_dim)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # Now reshape to heads (same as base class)
        bsz, q_len, _ = hidden_states.size()
        # Use getattr with default False for safety
        if getattr(self, 'qkv_proj_sp_enabled', False):
            q_len *= self.tensor_model_parallel_group.size()
        
        # BSHD -> BHSD layout
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        
        # Apply rotary embeddings
        if not skip_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        
        # Gather KV to full S when CP is enabled (same as base class)
        if past_key_value is None and getattr(self, 'cp_degree', 1) > 1:
            from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region_with_dim
            from neuronx_distributed_inference.modules.attention.attention_process_groups import get_context_parallel_attention_cp_group
            from neuronx_distributed_inference.modules.attention.utils import order_strided_tensor
            from neuronx_distributed_inference.modules.attention.attention_base import FlashAttentionStrategy
            
            stacked_kv = torch.stack([K, V], dim=0)
            stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                stacked_kv,
                gather_dim=3,
                process_group=get_context_parallel_attention_cp_group(),
            )
            if self.get_flash_attention_strategy_cp(q_len * self.cp_degree) == FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL:
                stacked_kv = order_strided_tensor(stacked_kv, 3, self.cp_degree)
            K, V = torch.unbind(stacked_kv, dim=0)
        
        return Q, K, V, cos_cache, sin_cache, residual
    
    # NOTE: We intentionally do NOT define a preshard_hook here.
    # The framework's invoke_preshard_hook function returns early if a module has preshard_hook,
    # which would prevent it from recursing into child modules (q_norm, k_norm, and the GQA class).
    # By not having preshard_hook here, the framework will:
    # 1. Recurse into q_norm and call ShardedRMSNorm.preshard_hook
    # 2. Recurse into k_norm and call ShardedRMSNorm.preshard_hook  
    # 3. Recurse into the GQA class and call its preshard_hook for QKV weight handling


# ============================================================================
# Decoder Layer
# ============================================================================

class NeuronOlmo2DecoderLayer(nn.Module):
    """
    OLMo2 Decoder Layer for NeuronX.
    
    Key architectural difference from LLaMA:
    - POST-layer normalization: RMSNorm is applied AFTER attention and AFTER MLP
    - In LLaMA, normalization is applied BEFORE attention and BEFORE MLP (pre-norm)
    
    Architecture flow (OLMo2 POST-norm):
        residual = hidden_states
        hidden_states = self_attn(hidden_states)
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states  
        hidden_states = mlp(hidden_states)
        hidden_states = post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
    
    Reference: Olmo2DecoderLayer in modeling_olmo2.py
    """
    
    def __init__(self, config: Olmo2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention (no pre-norm in OLMo2)
        self.self_attn = NeuronOlmo2Attention(config)
        
        # MLP (reuse LLaMA MLP - same architecture with SwiGLU)
        self.mlp = NeuronLlamaMLP(config)
        
        # Post-attention and post-feedforward normalization (OLMo2's key difference)
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = get_rmsnorm_cls()(
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
        Forward pass implementing OLMo2's post-normalization architecture.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key/value states for generation
            **kwargs: Additional arguments passed to attention
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Store residual for post-attention addition
        residual = hidden_states
        
        # Self Attention (no pre-norm in OLMo2)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        # Post-attention normalization (OLMo2's key difference from LLaMA)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Residual connection after attention
        hidden_states = residual + hidden_states
        
        # Store residual for post-MLP addition
        residual = hidden_states
        
        # MLP (no pre-norm in OLMo2)
        hidden_states = self.mlp(hidden_states)[0]
        
        # Post-feedforward normalization (OLMo2's key difference from LLaMA)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        
        # Residual connection after MLP
        hidden_states = residual + hidden_states
        
        # Return format consistent with NeuronX framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


# ============================================================================
# Model Classes
# ============================================================================

class NeuronOlmo2Model(NeuronBaseModel):
    """
    OLMo2 Model for NeuronX.
    
    Main model class that implements the OLMo2 architecture with:
    - Token embeddings
    - Stack of OLMo2 decoder layers
    - Final RMSNorm
    - LM head for language modeling
    
    Reference: Olmo2Model in modeling_olmo2.py
    """
    
    def setup_attr_for_model(self, config: Olmo2InferenceConfig):
        """Setup attributes required by the NeuronX framework."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: Olmo2InferenceConfig):
        """Initialize model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings with parallel sharding
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        
        # Stack of OLMo2 decoder layers
        self.layers = nn.ModuleList(
            [NeuronOlmo2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
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


class NeuronOlmo2ForCausalLM(NeuronBaseForCausalLM):
    """
    OLMo2 for Causal Language Modeling on NeuronX.
    
    This class extends NeuronBaseForCausalLM and provides:
    - Model class reference
    - HuggingFace model loading
    - State dict conversion from HF to NeuronX format
    - Config class reference
    
    Reference: Olmo2ForCausalLM in modeling_olmo2.py
    """
    
    _model_cls = NeuronOlmo2Model
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the original HuggingFace model."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace OLMo2 state dict to NeuronX format.
        
        Key mappings:
        - model.embed_tokens.weight -> embed_tokens.weight
        - model.layers.X.* -> layers.X.*
        - model.norm.weight -> norm.weight
        - lm_head.weight -> lm_head.weight
        
        OLMo2-specific:
        - q_norm and k_norm weights are kept at original shape [4096]
        - The ShardedRMSNorm class has a preshard_hook that shards these weights
          during checkpoint loading based on the TP rank
        
        Args:
            state_dict: Original HuggingFace state dictionary
            config: Model configuration
            
        Returns:
            Converted state dictionary for NeuronX
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        
        # Add rank utilities for vocab parallel and tensor parallel
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        num_layers = config.num_hidden_layers
        
        for i in range(num_layers):
            # Add rank utilities for attention layers
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
            
            # NOTE: q_norm and k_norm weights are NOT manually sharded here.
            # The ShardedRMSNorm class has a preshard_hook method that will
            # automatically shard these weights during checkpoint loading.
            # We just keep the original shape [4096].
        
        # Add rank utility for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Update state dict for tied weights if applicable."""
        # OLMo2 uses tie_word_embeddings=false by default, so nothing to do
        pass
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class."""
        return Olmo2InferenceConfig
