# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GPTNeoX model for NXD inference."""

"""
Key architectural features of GPTNeoX (Pythia):
- Rotary Position Embeddings (RoPE) with partial rotation
- Parallel residual connections (use_parallel_residual=True by default)
- LayerNorm (not RMSNorm)
- GELU activation in MLP
- Fused QKV projection (query_key_value)
- Multi-head attention (MHA, not GQA)

Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py
"""

import json
import logging
import os
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
from transformers import GPTNeoXForCausalLM
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


def get_layernorm_cls():
    """Get appropriate LayerNorm class based on execution mode.
    
    GPTNeoX uses standard LayerNorm, not RMSNorm.
    """
    return nn.LayerNorm


class GPTNeoXNeuronConfig(NeuronConfig):
    """Custom NeuronConfig for GPTNeoX.
    
    CRITICAL: This custom config class is REQUIRED for token generation to work.
    Without it, token generation HLO tracing fails.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CRITICAL: Framework uses this to determine attention class
        self.attn_cls = "NeuronGPTNeoXAttention"


class GPTNeoXInferenceConfig(InferenceConfig):
    """Inference configuration for GPTNeoX model.
    
    Maps HuggingFace GPTNeoXConfig parameters to NeuronX framework expectations.
    """
    
    def add_derived_config(self):
        """Add derived configuration parameters required by the framework."""
        # REQUIRED: For attention computation distribution
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                self.num_attention_heads, 
                self.num_key_value_heads, 
                self.neuron_config.tp_degree
            )
        
        # Calculate head_dim if missing
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        # REQUIRED: Framework attributes
        if not hasattr(self, 'output_attentions'):
            self.output_attentions = False
        if not hasattr(self, 'output_hidden_states'):
            self.output_hidden_states = False
        if not hasattr(self, 'use_return_dict'):
            self.use_return_dict = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        
        # GPTNeoX uses bias in attention
        if not hasattr(self, 'qkv_bias'):
            self.qkv_bias = getattr(self, 'attention_bias', True)
        if not hasattr(self, 'o_bias'):
            self.o_bias = getattr(self, 'attention_bias', True)
        
        # GPTNeoX specific: num_key_value_heads equals num_attention_heads (MHA)
        if not hasattr(self, 'num_key_value_heads'):
            self.num_key_value_heads = self.num_attention_heads
    
    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "layer_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the custom NeuronConfig class - REQUIRED for token generation."""
        return GPTNeoXNeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GPTNeoXInferenceConfig":
        """Load configuration from a pretrained model directory."""
        # Extract neuron_config from kwargs
        neuron_config = kwargs.pop("neuron_config", None)
        
        # Read config file
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            params = json.load(f)
        
        # Map GPTNeoX config to framework config
        # GPTNeoX uses num_attention_heads for both Q and KV (MHA, not GQA)
        config_dict = {
            "hidden_size": params.get("hidden_size", 2560),
            "num_attention_heads": params.get("num_attention_heads", 32),
            "num_hidden_layers": params.get("num_hidden_layers", 32),
            "num_key_value_heads": params.get("num_attention_heads", 32),  # MHA: same as num_attention_heads
            "vocab_size": params.get("vocab_size", 50304),
            "max_position_embeddings": params.get("max_position_embeddings", 2048),
            "layer_norm_eps": params.get("layer_norm_eps", 1e-5),
            "hidden_act": params.get("hidden_act", "gelu"),
            "intermediate_size": params.get("intermediate_size", 10240),
            "use_parallel_residual": params.get("use_parallel_residual", True),
            "attention_bias": params.get("attention_bias", True),
            "tie_word_embeddings": params.get("tie_word_embeddings", False),
            "bos_token_id": params.get("bos_token_id", 0),
            "eos_token_id": params.get("eos_token_id", 0),
            "pad_token_id": params.get("pad_token_id", 0),
            # RoPE parameters
            "rotary_pct": params.get("rotary_pct", 0.25),
            "rope_theta": params.get("rope_theta", 10000.0),
        }
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronGPTNeoXMLP(nn.Module):
    """GPTNeoX MLP layer ported to NeuronX.
    
    GPTNeoX uses a simple 2-layer MLP with GELU activation:
    - dense_h_to_4h: hidden_size -> intermediate_size
    - GELU activation
    - dense_4h_to_h: intermediate_size -> hidden_size
    
    Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py::GPTNeoXMLP
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        if parallel_state.model_parallel_is_initialized():
            self.dense_h_to_4h = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.dense_4h_to_h = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
    
    def forward(self, hidden_states):
        """Forward pass for MLP.
        
        Returns tuple (output, None) for framework compatibility.
        """
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states, None


def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_ndims):
    """Apply rotary position embedding to only the first rotary_ndims dimensions.
    
    GPTNeoX uses partial rotation (rotary_pct=0.25 by default), meaning only
    the first 25% of head dimensions get rotary embeddings applied.
    """
    cos = cos.unsqueeze(1)  # [batch, 1, seq, rotary_ndims]
    sin = sin.unsqueeze(1)
    
    # Split into rotated and pass-through parts
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    
    # Apply rotation to first part
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate rotated and pass-through parts
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class NeuronGPTNeoXAttention(NeuronAttentionBase):
    """GPTNeoX attention layer ported to NeuronX.
    
    Key features:
    - Fused QKV projection (query_key_value)
    - Partial Rotary Position Embeddings (rotary_pct=0.25)
    - Multi-head attention (MHA, not GQA)
    - Bias in attention projections
    
    Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py::GPTNeoXAttention
    """
    
    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_pct = getattr(config, 'rotary_pct', 0.25)
        self.rotary_ndims = int(head_dim * rotary_pct)
        
        # Create rotary embedding with partial dimension
        rotary_emb = RotaryEmbedding(
            self.rotary_ndims,  # Only rotary_ndims, not full head_dim
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
        )
        
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rms_norm_eps=config.layer_norm_eps,
        )
    
    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Override to use partial rotary embedding."""
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_partial_rotary_pos_emb(Q, K, cos_cache, sin_cache, self.rotary_ndims)
        return Q, K, cos_cache, sin_cache


class NeuronGPTNeoXDecoderLayer(nn.Module):
    """GPTNeoX decoder layer ported to NeuronX.
    
    Key features:
    - Parallel residual connections (use_parallel_residual=True):
      x = x + attn(ln1(x)) + mlp(ln2(x))
    - Sequential residual connections (use_parallel_residual=False):
      x = x + attn(ln1(x))
      x = x + mlp(ln2(x))
    
    Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py::GPTNeoXLayer
    """
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_parallel_residual = getattr(config, 'use_parallel_residual', True)
        
        self.self_attn = NeuronGPTNeoXAttention(
            config=config,
            tensor_model_parallel_group=get_tp_group(config),
        )
        self.mlp = NeuronGPTNeoXMLP(config)
        
        # GPTNeoX uses standard LayerNorm
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass for decoder layer.
        
        CRITICAL: Use tuple unpacking for attention output, not attribute access.
        This is required for token generation to work correctly.
        """
        residual = hidden_states
        
        # Input LayerNorm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention - CRITICAL: Use tuple unpacking
        attn_hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        if self.use_parallel_residual:
            # Parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_hidden_states = self.post_attention_layernorm(residual)
            mlp_output, _ = self.mlp(mlp_hidden_states)
            hidden_states = residual + attn_hidden_states + mlp_output
        else:
            # Sequential residual: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
            hidden_states = residual + attn_hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output, _ = self.mlp(hidden_states)
            hidden_states = residual + mlp_output
        
        # Return format expected by framework
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronGPTNeoXModel(NeuronBaseModel):
    """GPTNeoX model ported to NeuronX.
    
    Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py::GPTNeoXModel
    """
    
    def setup_attr_for_model(self, config: InferenceConfig):
        """Setup attributes required by the framework."""
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
    
    def init_model(self, config: InferenceConfig):
        """Initialize model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
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
        self.layers = nn.ModuleList([
            NeuronGPTNeoXDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class NeuronGPTNeoXForCausalLM(NeuronBaseForCausalLM):
    """GPTNeoX causal language model for NeuronX inference.
    
    Reference: transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py::GPTNeoXForCausalLM
    """
    
    _model_cls = NeuronGPTNeoXModel
    
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HuggingFace model for weight extraction."""
        return GPTNeoXForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert HuggingFace state dict to NeuronX format.
        
        Key mappings:
        - gpt_neox.embed_in.weight -> embed_tokens.weight
        - gpt_neox.layers.{i}.attention.query_key_value.weight -> layers.{i}.self_attn.qkv_proj.weight
        - gpt_neox.layers.{i}.attention.dense.weight -> layers.{i}.self_attn.o_proj.weight
        - gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight -> layers.{i}.mlp.dense_h_to_4h.weight
        - gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight -> layers.{i}.mlp.dense_4h_to_h.weight
        - gpt_neox.final_layer_norm.weight -> norm.weight
        - embed_out.weight -> lm_head.weight
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        
        # Embedding
        if "gpt_neox.embed_in.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict["gpt_neox.embed_in.weight"].clone()
        
        # Final LayerNorm
        if "gpt_neox.final_layer_norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["gpt_neox.final_layer_norm.weight"].clone()
        if "gpt_neox.final_layer_norm.bias" in state_dict:
            neuron_state_dict["norm.bias"] = state_dict["gpt_neox.final_layer_norm.bias"].clone()
        
        # LM Head
        if "embed_out.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = state_dict["embed_out.weight"].clone()
        
        # Decoder layers
        for i in range(num_layers):
            hf_prefix = f"gpt_neox.layers.{i}"
            neuron_prefix = f"layers.{i}"
            
            # Attention - GPTNeoX uses fused QKV with interleaved layout per head
            # Weight layout: [head0_Q, head0_K, head0_V, head1_Q, head1_K, head1_V, ...]
            # Shape: [3*hidden_size, hidden_size] = [num_heads * 3 * head_dim, hidden_size]
            if f"{hf_prefix}.attention.query_key_value.weight" in state_dict:
                qkv_weight = state_dict[f"{hf_prefix}.attention.query_key_value.weight"]
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                head_dim = hidden_size // num_heads
                
                # Reshape to [num_heads, 3, head_dim, hidden_size] then extract Q, K, V
                qkv_reshaped = qkv_weight.view(num_heads, 3, head_dim, hidden_size)
                q_weight = qkv_reshaped[:, 0, :, :].reshape(hidden_size, hidden_size)
                k_weight = qkv_reshaped[:, 1, :, :].reshape(hidden_size, hidden_size)
                v_weight = qkv_reshaped[:, 2, :, :].reshape(hidden_size, hidden_size)
                
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.q_proj.weight"] = q_weight.clone()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.k_proj.weight"] = k_weight.clone()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.v_proj.weight"] = v_weight.clone()
            
            if f"{hf_prefix}.attention.query_key_value.bias" in state_dict:
                qkv_bias = state_dict[f"{hf_prefix}.attention.query_key_value.bias"]
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                head_dim = hidden_size // num_heads
                
                # Reshape to [num_heads, 3, head_dim] then extract Q, K, V
                qkv_bias_reshaped = qkv_bias.view(num_heads, 3, head_dim)
                q_bias = qkv_bias_reshaped[:, 0, :].reshape(hidden_size)
                k_bias = qkv_bias_reshaped[:, 1, :].reshape(hidden_size)
                v_bias = qkv_bias_reshaped[:, 2, :].reshape(hidden_size)
                
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.q_proj.bias"] = q_bias.clone()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.k_proj.bias"] = k_bias.clone()
                neuron_state_dict[f"{neuron_prefix}.self_attn.qkv_proj.v_proj.bias"] = v_bias.clone()
            
            # Output projection - Note: o_proj is a GroupQueryAttention_O which has internal o_proj
            if f"{hf_prefix}.attention.dense.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.self_attn.o_proj.o_proj.weight"] = state_dict[f"{hf_prefix}.attention.dense.weight"].clone()
            if f"{hf_prefix}.attention.dense.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.self_attn.o_proj.o_proj.bias"] = state_dict[f"{hf_prefix}.attention.dense.bias"].clone()
            
            # MLP
            if f"{hf_prefix}.mlp.dense_h_to_4h.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.dense_h_to_4h.weight"] = state_dict[f"{hf_prefix}.mlp.dense_h_to_4h.weight"].clone()
            if f"{hf_prefix}.mlp.dense_h_to_4h.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.dense_h_to_4h.bias"] = state_dict[f"{hf_prefix}.mlp.dense_h_to_4h.bias"].clone()
            
            if f"{hf_prefix}.mlp.dense_4h_to_h.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.dense_4h_to_h.weight"] = state_dict[f"{hf_prefix}.mlp.dense_4h_to_h.weight"].clone()
            if f"{hf_prefix}.mlp.dense_4h_to_h.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.mlp.dense_4h_to_h.bias"] = state_dict[f"{hf_prefix}.mlp.dense_4h_to_h.bias"].clone()
            
            # LayerNorms
            if f"{hf_prefix}.input_layernorm.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.input_layernorm.weight"] = state_dict[f"{hf_prefix}.input_layernorm.weight"].clone()
            if f"{hf_prefix}.input_layernorm.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.input_layernorm.bias"] = state_dict[f"{hf_prefix}.input_layernorm.bias"].clone()
            
            if f"{hf_prefix}.post_attention_layernorm.weight" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.post_attention_layernorm.weight"] = state_dict[f"{hf_prefix}.post_attention_layernorm.weight"].clone()
            if f"{hf_prefix}.post_attention_layernorm.bias" in state_dict:
                neuron_state_dict[f"{neuron_prefix}.post_attention_layernorm.bias"] = state_dict[f"{hf_prefix}.post_attention_layernorm.bias"].clone()
            
            # Add rank utilities for tensor parallelism
            neuron_state_dict[f"{neuron_prefix}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        
        # Add rank utility for vocab parallel
        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )
        
        # Add rank utility for base model
        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        
        return neuron_state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights between embedding and lm_head."""
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
    
    @classmethod
    def get_config_cls(cls):
        """Return the configuration class."""
        return GPTNeoXInferenceConfig


# Module map for attention class lookup
_GPTNEOX_MODULE_MAP = {
    "NeuronGPTNeoXAttention": NeuronGPTNeoXAttention,
}
