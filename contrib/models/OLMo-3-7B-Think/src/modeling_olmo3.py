# coding=utf-8
# Copyright 2025 Allen AI and the HuggingFace Inc. team. All rights reserved.
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
PyTorch Olmo3 model for NXD inference - WITH SLIDING WINDOW ENABLED

Olmo3 Architecture Notes:
- Uses sliding window attention (4096 token window)
- Has Q/K normalization (RMSNorm) applied AFTER q_proj and k_proj, BEFORE RoPE
- Uses POST-normalization: post_attention_layernorm after attention output,
  post_feedforward_layernorm after MLP output
- MLP: SwiGLU activation (gate_proj, up_proj, down_proj)
- YARN rope scaling for extended context

NOTE: This version enables sliding window attention. Requires seq_len >= 512.
"""
import json
import math
import os
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

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


# RMSNorm implementation compatible with Olmo3
class Olmo3RMSNorm(nn.Module):
    """Olmo3 RMSNorm - equivalent to T5LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> Olmo3RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return Olmo3RMSNorm if cpu_mode() else CustomRMSNorm


class Olmo3InferenceConfig(InferenceConfig):
    """
    Configuration class for Olmo3 inference on Neuron.
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
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
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Olmo3InferenceConfig":
        """
        Load configuration from a pretrained model directory.
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)

        # Read config.json
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)

        # Map HuggingFace config to our config
        config_dict = {
            "hidden_size": hf_config.get("hidden_size", 4096),
            "num_attention_heads": hf_config.get("num_attention_heads", 32),
            "num_hidden_layers": hf_config.get("num_hidden_layers", 32),
            "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", 32)),
            "vocab_size": hf_config.get("vocab_size", 100278),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 65536),
            "rope_theta": hf_config.get("rope_theta", 500000.0),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "intermediate_size": hf_config.get("intermediate_size", 11008),
            "pad_token_id": hf_config.get("pad_token_id", 100277),
            "eos_token_id": hf_config.get("eos_token_id", 100257),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
            "attention_bias": hf_config.get("attention_bias", False),
            "sliding_window": hf_config.get("sliding_window", 4096),
            # Standard HuggingFace attributes needed by framework
            "output_attentions": False,
            "output_hidden_states": False,
            "use_cache": True,
        }

        # Override with any kwargs provided
        config_dict.update(kwargs)

        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)

        # Call add_derived_config
        config.add_derived_config()

        return config


class NeuronOlmo3Attention(NeuronAttentionBase):
    """
    Olmo3 Attention implementation for NeuronX.

    Key features:
    - Q/K normalization applied AFTER projection, BEFORE reshaping to heads
    - These norms operate on the full projection output (hidden_size), not per-head
    - Sliding window attention enabled (requires seq_len >= 512)
    """

    def __init__(self, config: Olmo3InferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Create rotary embedding
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Get sliding window size from config (default 4096 for Olmo3)
        sliding_window = getattr(config, "sliding_window", 4096)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            # Enable sliding window attention (requires seq_len >= 512)
            sliding_window=sliding_window,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
            # Disable base class Q/K norm - we handle it ourselves
            use_qk_norm=False,
            q_layernorm=None,
            k_layernorm=None,
        )

        # Create Q/K norms that match the HuggingFace checkpoint structure
        # These operate on full projection output (num_heads * head_dim = hidden_size)
        self.q_norm = get_rmsnorm_cls()(
            config.num_attention_heads * head_dim,
            eps=config.rms_norm_eps
        )
        self.k_norm = get_rmsnorm_cls()(
            config.num_key_value_heads * head_dim,
            eps=config.rms_norm_eps
        )

        # Store config for prep_qkv_tensors
        self._olmo3_config = config

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
        Override to apply Olmo3-style Q/K normalization.

        In Olmo3:
        1. Q = q_norm(q_proj(hidden_states))  # norm on full projection
        2. K = k_norm(k_proj(hidden_states))  # norm on full projection
        3. Then reshape to heads
        4. Apply RoPE
        """
        from neuronx_distributed_inference.modules.attention.utils import move_heads_front

        # Get Q, K, V projections from the base GQA module
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )

        # Apply Olmo3's Q/K normalization to full projection output (before reshaping)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Reshape to heads: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.qkv_proj_sp_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        # No per-head layernorm (already applied to full projection)
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Apply RoPE
        if not skip_rope:
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        # Gather KV for context parallel if needed (copy from base class)
        if past_key_value is None and self.cp_degree > 1:
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


class NeuronOlmo3DecoderLayer(nn.Module):
    """
    Olmo3 Decoder Layer with POST-normalization.

    Structure:
    1. residual = hidden_states
    2. hidden_states = self_attn(hidden_states)
    3. hidden_states = post_attention_layernorm(hidden_states)  # POST norm
    4. hidden_states = residual + hidden_states
    5. residual = hidden_states
    6. hidden_states = mlp(hidden_states)
    7. hidden_states = post_feedforward_layernorm(hidden_states)  # POST norm
    8. hidden_states = residual + hidden_states
    """

    def __init__(self, config: Olmo3InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Attention layer
        self.self_attn = NeuronOlmo3Attention(config)

        # MLP layer - reuse LlamaMLP since architecture is same (SwiGLU)
        self.mlp = NeuronLlamaMLP(config)

        # POST-normalization layers (different from Llama's PRE-norm)
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
        Forward pass with POST-normalization pattern.
        """
        # Save residual
        residual = hidden_states

        # Self Attention (no pre-norm for Olmo3)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # POST attention normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        # Save residual for MLP
        residual = hidden_states

        # MLP (no pre-norm for Olmo3)
        hidden_states = self.mlp(hidden_states)[0]

        # POST feedforward normalization
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronOlmo3Model(NeuronBaseModel):
    """
    The Neuron version of Olmo3Model.
    """

    def setup_attr_for_model(self, config: Olmo3InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = getattr(config, "sliding_window", 4096)

    def init_model(self, config: Olmo3InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
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
        self.layers = nn.ModuleList([
            NeuronOlmo3DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronOlmo3ForCausalLM(NeuronBaseForCausalLM):
    """
    Olmo3 for Causal Language Modeling on NeuronX.
    """

    _model_cls = NeuronOlmo3Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace Olmo3 model"""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace Olmo3 state dict to Neuron format.

        Key conversions:
        - q_norm/k_norm are kept as-is (full projection normalization)
        - Add rank utilities for tensor parallelism
        """
        neuron_config = config.neuron_config
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        # Add rank utilities for tensor parallelism
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
            # q_norm and k_norm are kept with their original names
            # They'll be loaded into self.q_norm and self.k_norm

        # Add rank utility for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Vocab parallel support
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights (embed_tokens and lm_head share weights if configured)"""
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Olmo3InferenceConfig
