# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GPT-2 model for NXD inference."""

import copy
import json
import logging
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import (
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
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

logger = logging.getLogger("Neuron")


class GPT2NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for GPT2

    CRITICAL: This class is REQUIRED for token generation to work.
    Without it, token generation HLO tracing fails with tensor shape mismatches.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # CRITICAL: Framework uses this during token generation tracing
        # Import will be set after class definition to avoid circular dependency
        self.attn_cls = None


class GPT2InferenceConfig(InferenceConfig):
    """Configuration class for GPT2 inference on Neuron"""

    def add_derived_config(self):
        """
        Add derived configuration parameters required by the framework

        CRITICAL: This method is called during initialization and MUST set
        all framework-required attributes
        """
        # REQUIRED: Framework uses this for attention computation distribution
        self.num_cores_per_group = 1

        # Calculate head_dim if not present in HF config
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads

        # REQUIRED: Framework expects all 4 of these attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True

        # Set bias flags for attention layers - GPT2 uses bias
        if not hasattr(self, 'qkv_bias'):
            self.qkv_bias = True  # GPT2 uses bias in attention
        if not hasattr(self, 'o_bias'):
            self.o_bias = True    # GPT2 uses bias in output projection

        # GPT2 specific attributes
        if not hasattr(self, 'layer_norm_epsilon'):
            self.layer_norm_epsilon = getattr(self, 'layer_norm_epsilon', 1e-5)

    def get_required_attributes(self) -> List[str]:
        """
        List of required attributes from HuggingFace config.json

        These attributes MUST be present in the HF config or provided during initialization
        """
        return [
            "hidden_size",              # Model hidden dimension (n_embd in GPT2)
            "num_attention_heads",      # Number of attention heads (n_head in GPT2)
            "num_hidden_layers",        # Number of transformer layers (n_layer in GPT2)
            "vocab_size",               # Vocabulary size
            "max_position_embeddings",  # Maximum sequence length (n_positions in GPT2)
            "layer_norm_epsilon",       # Layer normalization epsilon
            "activation_function",      # Activation function name
            "embd_pdrop",               # Embedding dropout probability
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """
        Return the NeuronConfig class to use

        CRITICAL: MUST return your custom NeuronConfig class, NOT base NeuronConfig
        Returning base NeuronConfig will cause token generation to fail
        """
        return GPT2NeuronConfig  # ✅ Return custom class, NOT NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load configuration from HuggingFace model directory

        Args:
            model_path: Path to HuggingFace model directory
            **kwargs: Additional config overrides
        """
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        def load_config_fn(config_instance):
            """Callback to load config attributes"""
            # Map GPT2 config names to standard names
            config_mapping = {
                'n_embd': 'hidden_size',
                'n_head': 'num_attention_heads', 
                'n_layer': 'num_hidden_layers',
                'n_positions': 'max_position_embeddings',
                'n_inner': 'intermediate_size',
            }
            
            for key, value in config_dict.items():
                if not key.startswith("_"):
                    # Use mapped name if available, otherwise use original
                    mapped_key = config_mapping.get(key, key)
                    setattr(config_instance, mapped_key, value)
                    
            # Set intermediate_size if not present (GPT2 uses 4 * hidden_size)
            if not hasattr(config_instance, 'intermediate_size') or config_instance.intermediate_size is None:
                n_inner = config_dict.get('n_inner')
                if n_inner is not None:
                    config_instance.intermediate_size = n_inner
                else:
                    config_instance.intermediate_size = 4 * config_instance.hidden_size
                    
            # Set num_key_value_heads (GPT2 uses MHA, so same as num_attention_heads)
            if not hasattr(config_instance, 'num_key_value_heads'):
                config_instance.num_key_value_heads = config_instance.num_attention_heads
                
            # Set embedding dropout (GPT2 default is 0.1)
            if not hasattr(config_instance, 'embd_pdrop'):
                config_instance.embd_pdrop = config_dict.get('embd_pdrop', 0.1)
                
            for key, value in kwargs.items():
                setattr(config_instance, key, value)

        # CRITICAL: Create default NeuronConfig if none provided
        # This must happen BEFORE calling __init__ to ensure proper initialization order
        if neuron_config is None:
            neuron_config = cls.get_neuron_config_cls()()

        return cls(neuron_config=neuron_config, load_config=load_config_fn)


class NeuronGPT2Attention(NeuronAttentionBase):
    """GPT2 attention implementation for NeuronX"""

    def __init__(self, config: GPT2InferenceConfig):
        """
        Initialize attention layer

        IMPORTANT: NO layer_idx parameter - GPT2 doesn't use rotary embeddings
        """
        # GPT2 doesn't use rotary position embeddings - uses absolute position embeddings
        rotary_emb = None

        # Initialize base attention with ALL required parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,

            # ✅ CRITICAL: Must pass num_cores_per_group
            # Missing this can cause incorrect tensor shapes during distributed execution
            num_cores_per_group=config.num_cores_per_group,

            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            sliding_window=None,  # GPT2 doesn't use sliding window
        )


class NeuronGPT2MLP(nn.Module):
    """GPT2 MLP implementation for NeuronX - Standard FFN with GELU activation"""

    def __init__(self, config: GPT2InferenceConfig):
        super().__init__()
        self.config = config

        # Input projection (hidden_size -> intermediate_size)
        self.c_fc = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,  # GPT2 uses bias
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # Activation function - GPT2 typically uses GELU
        if config.activation_function == "gelu":
            self.act = F.gelu
        elif config.activation_function == "gelu_new":
            self.act = lambda x: F.gelu(x, approximate="tanh")
        elif config.activation_function == "relu":
            self.act = F.relu
        elif config.activation_function == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")

        # Output projection (intermediate_size -> hidden_size)
        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,  # GPT2 uses bias
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for standard FFN

        Returns:
            Tuple of (output_tensor, None) - None for framework compatibility
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        # ✅ CRITICAL: Return tuple for framework compatibility
        # Standard MLPs return (output, None)
        return hidden_states, None


class NeuronGPT2DecoderLayer(nn.Module):
    """GPT2 decoder layer implementation for NeuronX"""

    def __init__(self, config: GPT2InferenceConfig):
        """
        Initialize decoder layer

        IMPORTANT: NO layer_idx parameter unless pattern requires it
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention (no layer_idx passed)
        self.self_attn = NeuronGPT2Attention(config)

        # MLP
        self.mlp = NeuronGPT2MLP(config)

        # Layer normalization - GPT2 uses LayerNorm (not RMSNorm)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,  # ✅ IMPORTANT: Capture extra framework arguments
    ) -> Tuple:
        """
        Forward pass for decoder layer

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, attn_weights)
        """
        # Self-attention with pre-normalization (GPT2 style)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention returns 4 values: (output, kv_cache, cos, sin)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # MLP with pre-normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # ✅ Handle MLP return based on architecture type
        mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            # Standard FFN returns (output, None)
            hidden_states = mlp_output[0]
        else:
            # SwiGLU returns single tensor
            hidden_states = mlp_output

        # Residual connection
        hidden_states = residual + hidden_states

        # Return 5-tuple expected by framework
        # (hidden_states, kv_cache, cos, sin, attention_weights)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronGPT2Model(NeuronBaseModel):
    """
    GPT2 base model for NeuronX

    IMPORTANT: Inherits from NeuronBaseModel, NOT NeuronBaseForCausalLM
    The CausalLM wrapper comes later
    """

    def setup_attr_for_model(self, config: GPT2InferenceConfig):
        """
        Setup attributes required by the framework

        Called BEFORE init_model() to set up instance attributes
        """
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: GPT2InferenceConfig):
        """
        Initialize model components

        Called AFTER setup_attr_for_model() to create layers
        """
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled
        )

        # Position embeddings (GPT2 uses absolute position embeddings)
        # Note: For now, we'll let the framework handle position embeddings
        # TODO: Add proper position embedding support
        self.wpe = ParallelEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            None,  # No padding for position embeddings
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled
        )

        # Embedding dropout - DISABLED for inference (set to 0.0)
        # GPT2 training uses dropout but inference should be deterministic
        self.dropout = nn.Dropout(0.0)  # Explicitly disable dropout for inference

        # Decoder layers
        # ✅ CRITICAL: Create layers WITHOUT layer_idx unless pattern requires it
        self.layers = nn.ModuleList(
            [NeuronGPT2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # Language modeling head
        # ✅ CRITICAL: lm_head belongs HERE in base model, not in CausalLM wrapper
        # Note: GPT2 vocab size (50257) may not be divisible by TP degree, so we use pad=True
        # We do NOT tie weights here - the state dict conversion handles weight sharing
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # GPT2 typically doesn't use bias in lm_head
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
            pad=True,  # Enable padding for non-divisible vocab sizes
        )
        
        # Note: We don't tie embeddings here because the framework's preshard_hook
        # expects separate weights. The state dict conversion handles weight sharing.

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds=None,
        kv_cache=None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
        **kwargs
    ):
        """
        Custom forward method for GPT2 that properly handles position embeddings
        
        This overrides the base class forward to ensure position embeddings are
        correctly added to token embeddings, fixing the repetitive output issue.
        """
        # ✅ CRITICAL FIX: Compute embeddings with position embeddings added
        if inputs_embeds is None and input_ids is not None:
            batch_size, seq_length = input_ids.shape
            
            # Token embeddings
            inputs_embeds = self.embed_tokens(input_ids)
            
            # Position embeddings - ensure correct shape and addition
            if position_ids is None:
                device = input_ids.device
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = position_ids.view(-1, seq_length).long()
            
            # Get position embeddings and add them to token embeddings
            position_embeds = self.wpe(position_ids)
            inputs_embeds = inputs_embeds + position_embeds
            
            # Apply embedding dropout
            inputs_embeds = self.dropout(inputs_embeds)
        
        # Now call the parent class forward method with the corrected embeddings
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            accepted_indices=accepted_indices,
            current_length=current_length,
            medusa_mask=medusa_mask,
            scatter_index=scatter_index,
            slot_mapping=slot_mapping,
            active_block_table=active_block_table,
            num_queries=num_queries,
            computed_context_lens=computed_context_lens,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            inputs_embeds=inputs_embeds,  # Pass our corrected embeddings
            kv_cache=kv_cache,
            active_mask=active_mask,
            rotary_position_id=rotary_position_id,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            **kwargs
        )


class NeuronGPT2ForCausalLM(NeuronBaseForCausalLM):
    """
    GPT2 causal language model for inference
    """
    _model_cls = NeuronGPT2Model
    # HuggingFace GPT-2 state dict keys have "transformer." prefix
    # Base class strips this before calling convert_hf_to_neuron_state_dict
    _STATE_DICT_MODEL_PREFIX = "transformer."

    @classmethod
    def get_config_cls(cls):
        """Return the configuration class for this model"""
        return GPT2InferenceConfig

    @classmethod
    def from_config(cls, config, model_path: str = ""):
        """
        Create a model from a configuration

        Args:
            config: Model configuration
            model_path: Path to model (can be empty for from_config)

        Returns:
            NeuronGPT2ForCausalLM: Model instance
        """
        return cls(model_path=model_path, config=config)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """
        Convert weights from HuggingFace format to NeuronX format

        Args:
            state_dict: HuggingFace state dictionary
            config: Model configuration

        Returns:
            Dict[str, torch.Tensor]: NeuronX format state dictionary
        """
        neuron_state_dict = {}

        # Token embeddings (base class strips "transformer." prefix)
        if "wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["wte.weight"].clone()

        # Position embeddings
        if "wpe.weight" in state_dict:
            neuron_state_dict["wpe.weight"] = state_dict["wpe.weight"].clone()

        # Final normalization
        if "ln_f.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["ln_f.weight"].clone()
        if "ln_f.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["ln_f.bias"].clone()

        # Language modeling head
        # GPT2 ties embeddings by default, so lm_head.weight = embed_tokens.weight
        # We need to provide the weight for the framework's preshard_hook
        if "lm_head.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
        elif "wte.weight" in state_dict:
            # Use embedding weight for tied embeddings
            neuron_state_dict["lm_head.weight"] = state_dict["wte.weight"].clone()

        # Decoder layers (base class strips "transformer." prefix, so keys are h.{i}.*)
        for i in range(config.num_hidden_layers):
            layer_prefix = f"h.{i}"
            neuron_prefix = f"layers.{i}"

            # Layer norms
            if f"{layer_prefix}.ln_1.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.input_layernorm.weight"] = state_dict[f"{layer_prefix}.ln_1.weight"].clone()
            if f"{layer_prefix}.ln_1.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.input_layernorm.bias"] = state_dict[f"{layer_prefix}.ln_1.bias"].clone()

            if f"{layer_prefix}.ln_2.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.post_attention_layernorm.weight"] = state_dict[f"{layer_prefix}.ln_2.weight"].clone()
            if f"{layer_prefix}.ln_2.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.post_attention_layernorm.bias"] = state_dict[f"{layer_prefix}.ln_2.bias"].clone()

            # Attention weights - GPT2 uses combined QKV projection
            if f"{layer_prefix}.attn.c_attn.weight" in state_dict:
                # Split the combined QKV weight
                qkv_weight = state_dict[f"{layer_prefix}.attn.c_attn.weight"].clone()
                hidden_size = config.hidden_size
                
                # GPT2 uses Conv1D which transposes the weight
                qkv_weight = qkv_weight.t().contiguous()  # Transpose and make contiguous
                
                q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
                
                # Use the correct weight names expected by the attention module
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.q_proj.weight"] = q_weight.contiguous()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.k_proj.weight"] = k_weight.contiguous()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.v_proj.weight"] = v_weight.contiguous()

            if f"{layer_prefix}.attn.c_attn.bias" in state_dict:
                # Split the combined QKV bias
                qkv_bias = state_dict[f"{layer_prefix}.attn.c_attn.bias"].clone()
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
                
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.q_proj.bias"] = q_bias
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.k_proj.bias"] = k_bias
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.v_proj.bias"] = v_bias

            # Attention output projection
            if f"{layer_prefix}.attn.c_proj.weight" in state_dict:
                # GPT2 uses Conv1D which transposes the weight
                weight = state_dict[f"{layer_prefix}.attn.c_proj.weight"].clone().t().contiguous()
                neuron_state_dict[f"{neuron_prefix}.self_attn.o_proj.weight"] = weight
            if f"{layer_prefix}.attn.c_proj.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.self_attn.o_proj.bias"] = state_dict[f"{layer_prefix}.attn.c_proj.bias"].clone()

            # MLP weights
            if f"{layer_prefix}.mlp.c_fc.weight" in state_dict:
                # GPT2 uses Conv1D which transposes the weight
                weight = state_dict[f"{layer_prefix}.mlp.c_fc.weight"].clone().t().contiguous()
                neuron_state_dict[f"{neuron_prefix}.mlp.c_fc.weight"] = weight
            if f"{layer_prefix}.mlp.c_fc.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.c_fc.bias"] = state_dict[f"{layer_prefix}.mlp.c_fc.bias"].clone()

            if f"{layer_prefix}.mlp.c_proj.weight" in state_dict:
                # GPT2 uses Conv1D which transposes the weight
                weight = state_dict[f"{layer_prefix}.mlp.c_proj.weight"].clone().t().contiguous()
                neuron_state_dict[f"{neuron_prefix}.mlp.c_proj.weight"] = weight
            if f"{layer_prefix}.mlp.c_proj.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.c_proj.bias"] = state_dict[f"{layer_prefix}.mlp.c_proj.bias"].clone()

        # Add rank information for tensor parallelism
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree

        # Add rank information for attention
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Add rank information for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return neuron_state_dict


# Set the attention class after all classes are defined
def _init_gpt2_neuron_config(self, **kwargs):
    """Initialize GPT2NeuronConfig with attention class"""
    super(GPT2NeuronConfig, self).__init__(**kwargs)
    # Set attention class after it's defined
    self.attn_cls = NeuronGPT2Attention

# Replace the __init__ method
GPT2NeuronConfig.__init__ = _init_gpt2_neuron_config