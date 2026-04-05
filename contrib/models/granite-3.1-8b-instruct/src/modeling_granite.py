# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
# Adapted for NeuronX Distributed Inference.
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
NeuronX Distributed Inference implementation of Granite model.

FIXED VERSION - Addresses critical accuracy issues:
1. attention_multiplier: Applied by scaling Q before attention computation
2. logits_scaling: Applied by dividing logits after lm_head
3. embedding_multiplier: Applied in forward pass (not to weights for tied embeddings)

Key differences from Llama:
1. embedding_multiplier: Scales input embeddings (default: 12.0)
2. logits_scaling: Scales output logits (default: 16.0)
3. residual_multiplier: Scales residual connections (default: 0.22)
4. attention_multiplier: Custom attention scaling (default: 0.0078125 = 1/head_dim)
"""

import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
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
from neuronx_distributed_inference.utils.distributed import get_tp_group

# Use HuggingFace's RMSNorm for CPU mode, CustomRMSNorm for Neuron
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.activations import ACT2FN

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    """
    Returns the appropriate RMSNorm class based on execution mode.
    CustomRMSNorm is optimized for Neuron devices.
    LlamaRMSNorm is used for CPU execution.
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class GraniteInferenceConfig(InferenceConfig):
    """
    Configuration class for Granite model inference on Neuron.
    
    Extends InferenceConfig with Granite-specific parameters:
    - embedding_multiplier: Scale factor for input embeddings
    - logits_scaling: Scale factor for output logits
    - residual_multiplier: Scale factor for residual connections
    - attention_multiplier: Scale factor for attention scores
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1
        # Granite uses standard attention without flash decoding by default
        
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
            # Granite-specific attributes
            "embedding_multiplier",
            "logits_scaling",
            "residual_multiplier",
            "attention_multiplier",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GraniteInferenceConfig":
        """
        Load configuration from a pretrained model directory.
        
        This method loads the HuggingFace config and creates a GraniteInferenceConfig
        that is compatible with NeuronX Distributed Inference.
        
        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments including neuron_config
            
        Returns:
            GraniteInferenceConfig: Configuration object for Granite model
        """
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        
        # Extract neuron_config from kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        
        if neuron_config is None:
            neuron_config = NeuronConfig()
        
        # Create config with load_config hook that loads from HuggingFace
        config = cls(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(model_path),
            **kwargs
        )
        
        return config


class NeuronGraniteMLP(nn.Module):
    """
    Granite MLP layer for NeuronX.
    
    Uses SwiGLU activation (same as Llama):
    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Replaces linear layers with column/row parallel layers for tensor parallelism.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Get MLP bias setting (Granite default is False)
        mlp_bias = getattr(config, "mlp_bias", False)
        
        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        
        if parallel_state.model_parallel_is_initialized():
            # Create parallel linear layers for tensor parallelism
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
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )
        else:
            # Use standard linear layers for non-parallel mode (e.g., testing)
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        Forward pass of the MLP layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            rmsnorm: Optional RMSNorm to apply before MLP (for fused operations)
            residual: Optional residual tensor for fused residual add
            adapter_ids: Optional adapter IDs for LoRA (not used in base implementation)
            
        Returns:
            Tuple of (output, residual) tensors
        """
        if rmsnorm is not None:
            x = rmsnorm(x)
            
        # SwiGLU activation: silu(gate_proj(x)) * up_proj(x)
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        
        return output, None


class NeuronGraniteAttention(NeuronAttentionBase):
    """
    Granite attention layer for NeuronX.
    
    CRITICAL FIX: Granite uses attention_multiplier (0.0078125 = 1/head_dim)
    instead of the standard 1/sqrt(head_dim) = 0.0884.
    
    The NeuronX attention kernels apply 1/sqrt(head_dim) scaling internally:
    - Context encoding (perform_prefill): Q = Q / sqrt(head_dim)
    - Token generation (compute_for_token_gen): scores = Q @ K^T / sqrt(head_dim)
    
    To convert from standard scaling to Granite's attention_multiplier:
    - Standard: Q @ K^T / sqrt(head_dim)
    - Granite: Q @ K^T * attention_multiplier
    
    We need to pre-scale Q by a correction factor:
    - (Q * correction) / sqrt(head_dim) = Q * attention_multiplier
    - correction = attention_multiplier * sqrt(head_dim)
    - correction = 0.0078125 * sqrt(128) = 0.0078125 * 11.31 = 0.0884
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        # Get Granite-specific attention multiplier
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.attention_multiplier = getattr(config, "attention_multiplier", 1.0 / head_dim)
        
        # Compute the correction factor to convert from standard 1/sqrt(head_dim) to attention_multiplier
        # The kernel applies: scores = Q @ K^T / sqrt(head_dim)
        # We want: scores = Q @ K^T * attention_multiplier
        # So: (Q * correction) @ K^T / sqrt(head_dim) = Q @ K^T * attention_multiplier
        # correction = attention_multiplier * sqrt(head_dim)
        self.q_scale_factor = self.attention_multiplier * math.sqrt(head_dim)
        
        # Initialize the base attention class
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=self._get_rope(config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
        )

    def _get_rope(self, config: InferenceConfig):
        """Get the rotary position embedding module for Granite."""
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
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
        Override prep_qkv_tensors to apply Granite's attention_multiplier.
        
        Since the flash attention kernel uses scale=1.0, we need to apply
        the attention_multiplier ourselves by scaling Q.
        """
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
            move_heads_front,
        )
        
        # Get QKV projections through the base class's qkv_proj
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )
        
        # Reshape to heads
        bsz, q_len, _ = hidden_states.size()
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
        
        # CRITICAL FIX: Apply Granite's attention_multiplier by scaling Q
        # The attention kernels compute: softmax((Q / sqrt(head_dim)) @ K^T) @ V
        # But Granite wants: softmax(Q @ K^T * attention_multiplier) @ V
        #
        # To convert: (Q * correction) / sqrt(head_dim) @ K^T = Q @ K^T * attention_multiplier
        # correction = attention_multiplier * sqrt(head_dim) = 0.0078125 * 11.31 = 0.0884
        Q = Q * self.q_scale_factor
        
        # Gather KV to full S when CP is enabled
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


class NeuronGraniteDecoderLayer(nn.Module):
    """
    Granite decoder layer for NeuronX.
    
    Structure:
    1. Input LayerNorm -> Self Attention -> Residual Add (with residual_multiplier)
    2. Post Attention LayerNorm -> MLP -> Residual Add (with residual_multiplier)
    
    Key difference from Llama: residual connections are scaled by residual_multiplier
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = getattr(config, "residual_multiplier", 1.0)

        # Self attention
        self.self_attn = NeuronGraniteAttention(
            config=config, 
            tensor_model_parallel_group=get_tp_group(config)
        )

        # MLP
        self.mlp = NeuronGraniteMLP(config)
        
        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key-value pairs for autoregressive generation
            adapter_ids: Optional adapter IDs for LoRA
            **kwargs: Additional arguments passed to attention
            
        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        residual = hidden_states

        # Input layer norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with Granite's residual multiplier
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        
        # Granite-specific: scale residual by residual_multiplier
        hidden_states = residual + attn_output.hidden_states * self.residual_multiplier

        # MLP with Granite's residual multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        
        # Granite-specific: scale residual by residual_multiplier
        hidden_states = residual + hidden_states * self.residual_multiplier

        outputs = (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)
        return outputs


class ScaledColumnParallelLinear(ColumnParallelLinear):
    """
    ColumnParallelLinear that applies logits_scaling after the linear projection.
    
    This is needed for Granite which divides logits by logits_scaling (16.0)
    after the lm_head projection.
    """
    
    def __init__(self, *args, logits_scaling: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_scaling = logits_scaling
    
    def forward(self, x):
        output = super().forward(x)
        # Apply Granite's logits_scaling
        return output / self.logits_scaling


class ScaledLinear(nn.Linear):
    """
    Linear layer that applies logits_scaling after the linear projection.
    For non-parallel mode (CPU testing).
    """
    
    def __init__(self, *args, logits_scaling: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_scaling = logits_scaling
    
    def forward(self, x):
        output = super().forward(x)
        return output / self.logits_scaling


class NeuronGraniteModel(NeuronBaseModel):
    """
    Granite model for NeuronX.
    
    CRITICAL FIXES:
    1. embedding_multiplier: Applied in forward pass via get_model_output override
    2. logits_scaling: Applied via ScaledColumnParallelLinear in lm_head
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        """Set up model attributes required by NeuronBaseModel."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        
        # Granite-specific multipliers
        self.embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)
        self.logits_scaling = getattr(config, "logits_scaling", 1.0)

    def init_model(self, config: InferenceConfig):
        """Initialize model components."""
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size
        
        # Token embeddings - embedding_multiplier is applied in forward(), not here
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
            )

            # CRITICAL FIX: Use ScaledColumnParallelLinear to apply logits_scaling
            self.lm_head = ScaledColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                logits_scaling=self.logits_scaling,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, 
                config.hidden_size, 
                self.padding_idx,
            )
            self.lm_head = ScaledLinear(
                config.hidden_size, 
                config.vocab_size, 
                bias=False,
                logits_scaling=self.logits_scaling,
            )

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGraniteDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        **kwargs,
    ):
        """
        Override get_model_output to apply Granite's embedding_multiplier.
        
        Granite multiplies embeddings by embedding_multiplier (12.0) AFTER the embedding
        lookup. This is critical for correct model behavior.
        """
        # Apply Granite embedding_multiplier if we need to compute embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Apply Granite's embedding_multiplier (12.0)
            inputs_embeds = inputs_embeds * self.embedding_multiplier
        
        # Call parent's get_model_output with pre-computed embeddings
        return super().get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,  # Pass scaled embeddings
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_ids,
            update_cache=update_cache,
            is_for_context_encoding=is_for_context_encoding,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            local_attn_mask=local_attn_mask,
            windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
            **kwargs,
        )


class NeuronGraniteForCausalLM(NeuronBaseForCausalLM):
    """
    Granite causal language model for NeuronX inference.
    
    Key differences from Llama:
    - Output logits are scaled by 1/logits_scaling
    """
    
    _model_cls = NeuronGraniteModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace model for weight conversion."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace state dict to Neuron format.
        
        IMPORTANT: The framework's preshard_hook in GroupQueryAttention_QKV and 
        GroupQueryAttention_O automatically handles the key renaming:
        - q_proj.weight -> qkv_proj.q_proj.weight
        - k_proj.weight -> qkv_proj.k_proj.weight
        - v_proj.weight -> qkv_proj.v_proj.weight
        - o_proj.weight -> o_proj.o_proj.weight
        
        So we should NOT rename these keys here.
        
        IMPORTANT: For Granite with tie_word_embeddings=True:
        - embedding_multiplier is applied in the forward pass, NOT to weights
        - lm_head.weight is tied to embed_tokens.weight (same weights)
        - logits_scaling is applied in the forward pass via ScaledColumnParallelLinear
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        # NOTE: Do NOT apply embedding_multiplier to weights!
        # For tied weights, this would incorrectly scale the lm_head weights.
        # Instead, embedding_multiplier is applied in the forward pass.
        
        # Add rank_util tensors required by NeuronAttentionBase
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embeddings and LM head."""
        # Granite uses tie_word_embeddings=True
        # The lm_head.weight should be the same as embed_tokens.weight
        # Note: embedding_multiplier is applied in forward pass, not to weights
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for Granite."""
        return GraniteInferenceConfig


# Export the main classes
__all__ = [
    "GraniteInferenceConfig",
    "NeuronGraniteModel",
    "NeuronGraniteForCausalLM",
    "NeuronGraniteMLP",
    "NeuronGraniteAttention",
    "NeuronGraniteDecoderLayer",
    "ScaledColumnParallelLinear",
    "ScaledLinear",
]
