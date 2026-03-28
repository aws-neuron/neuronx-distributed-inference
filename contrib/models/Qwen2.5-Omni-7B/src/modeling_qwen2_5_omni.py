# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
PyTorch Qwen2.5-Omni model for NXD inference (Text-only version)

This implementation ports the text model (Thinker) from Qwen2.5-Omni to NeuronX Distributed Inference.
It focuses on text-only inference, ignoring multimodal (audio/vision) components.

Based on:
- Reference: NeuronxDistributedInference/src/neuronx_distributed_inference/models/qwen2/modeling_qwen2.py
"""

import gc
import json
import os
from typing import List, Optional, Tuple, Type

import torch
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
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import _rotate_half
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# M-RoPE: Multimodal Rotary Position Embedding
# The Omni model's text backbone uses the same M-RoPE as Qwen2.5-VL.
# For text-only input all 3 axes get identical position IDs, but the
# frequency splitting/interleaving still differs from standard 1D RoPE.
# ---------------------------------------------------------------------------


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Apply M-RoPE to query and key tensors."""
    mrope_section = mrope_section * 2  # double for cos/sin pairs
    split_indices = [sum(mrope_section[: i + 1]) for i in range(len(mrope_section) - 1)]
    cos = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(cos, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(sin, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5OmniRotaryEmbedding(nn.Module):
    """3D rotary embedding for M-RoPE (temporal, height, width)."""

    def __init__(self, config, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = getattr(config, "rope_theta", 1000000.0)
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", None, persistent=False)
        self.inv_freq = self._compute_inv_freq(device)

    def _compute_inv_freq(self, device=None):
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    def forward(self, x, position_ids):
        # position_ids: (batch, seq) from NeuronBaseModel or (3, batch, seq) for M-RoPE
        # For text-only input, all 3 axes get the same positions.
        if position_ids.dim() == 2:
            # Expand 2D -> 3D: replicate across temporal, height, width
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # position_ids is now (3, batch, seq)
        # HF convention: dim 0 = 3 axes, dim 1 = batch, dim 2 = seq
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            position_ids.shape[0], -1, -1, 1
        )  # (3, 1, dim/2, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        # (3, batch, 1, seq)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5OmniNeuronConfig(NeuronConfig):
    """NeuronConfig for Qwen2.5-Omni model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronQwen2_5OmniAttention


class Qwen2_5OmniInferenceConfig(InferenceConfig):
    """
    Configuration class for Qwen2.5-Omni inference on NeuronX.

    This config handles the text model (Thinker) from Qwen2.5-Omni.
    The thinker_config.text_config contains the core text model parameters.
    """

    def add_derived_config(self):
        """Add derived configuration parameters"""
        self.num_cores_per_group = 1
        self.qkv_bias = True  # Qwen2.5-Omni has bias in Q/K/V projections
        self.o_bias = False  # No bias in output projection

        # Handle layer types for sliding window attention
        # Default to all full attention if not specified
        if not hasattr(self, "layer_types") or self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # Multimodal RoPE section for 3D position embeddings
        # [temporal, height, width] sections - for text-only, all positions are same
        if not hasattr(self, "mrope_section"):
            self.mrope_section = [16, 24, 24]  # Default from config

        # Add standard HuggingFace config attributes required by NeuronBaseModel
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = True
        if not hasattr(self, "use_cache"):
            self.use_cache = True

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
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen2_5OmniNeuronConfig]:
        """Return the NeuronConfig class to use"""
        return Qwen2_5OmniNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Qwen2_5OmniInferenceConfig":
        """
        Load configuration from a pretrained Qwen2.5-Omni model directory.

        The Qwen2.5-Omni config has a nested structure:
        config.json -> thinker_config -> text_config (the actual text model config)

        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration

        Returns:
            Qwen2_5OmniInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)

        # Try loading saved neuron config if not provided
        # Try multiple possible locations
        if neuron_config is None:
            possible_paths = [
                os.path.join(model_path, "neuron_config.json"),
                "neuron_config.json",  # Current directory
            ]

            for neuron_config_path in possible_paths:
                if os.path.exists(neuron_config_path):
                    print(f"Loading neuron_config from: {neuron_config_path}")
                    with open(neuron_config_path, "r") as f:
                        neuron_config_data = json.load(f)
                        # The saved config has the neuron_config nested
                        if "neuron_config" in neuron_config_data:
                            neuron_config_dict = neuron_config_data["neuron_config"]
                        else:
                            neuron_config_dict = neuron_config_data
                        neuron_config = cls.get_neuron_config_cls()(
                            **neuron_config_dict
                        )
                    break

        # Read the full config.json
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            full_config = json.load(f)

        # Navigate to the text model config
        # Path: config.json -> thinker_config -> text_config
        thinker_config = full_config.get("thinker_config", {})
        text_config = thinker_config.get("text_config", {})

        if not text_config:
            raise ValueError(
                f"Could not find text_config in {config_path}. "
                "Expected structure: config.json -> thinker_config -> text_config"
            )

        # Extract configuration parameters from text_config
        config_dict = {
            "hidden_size": text_config.get("hidden_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "vocab_size": text_config.get("vocab_size"),
            "max_position_embeddings": text_config.get("max_position_embeddings"),
            "intermediate_size": text_config.get("intermediate_size"),
            "rms_norm_eps": text_config.get("rms_norm_eps"),
            "rope_theta": text_config.get("rope_theta"),
            "hidden_act": text_config.get("hidden_act"),
            "sliding_window": text_config.get("sliding_window"),
            "use_sliding_window": text_config.get("use_sliding_window", False),
        }

        # Extract pad_token_id from thinker_config (not in text_config)
        config_dict["pad_token_id"] = thinker_config.get("pad_token_id")

        # Extract rope_scaling if present
        if "rope_scaling" in text_config and text_config["rope_scaling"]:
            rope_scaling = text_config["rope_scaling"]
            config_dict["rope_scaling"] = rope_scaling
            # Extract mrope_section for multimodal RoPE
            config_dict["mrope_section"] = rope_scaling.get(
                "mrope_section", [16, 24, 24]
            )

        # Handle layer_types for sliding window attention
        # Qwen2.5-Omni alternates between full and sliding attention
        num_layers = config_dict["num_hidden_layers"]
        if config_dict.get("use_sliding_window"):
            # Alternate between full and sliding attention
            config_dict["layer_types"] = [
                "sliding_attention" if i % 2 else "full_attention"
                for i in range(num_layers)
            ]
        else:
            # All layers use full attention
            config_dict["layer_types"] = ["full_attention"] * num_layers

        # Override with kwargs
        config_dict.update(kwargs)

        # Create and return config
        config = cls(neuron_config=neuron_config, **config_dict)
        return config


class NeuronQwen2_5OmniAttention(NeuronAttentionBase):
    """
    Qwen2.5-Omni attention with M-RoPE support.

    The Omni text backbone (thinker) uses M-RoPE with sections [16, 24, 24]
    for temporal, height, and width position axes. For text-only input, all
    3 axes receive the same position IDs.

    Reference:
    - HF: Qwen2_5OmniAttention in modeling_qwen2_5_omni.py
    - NXD: NeuronQwen2_5_VLAttention in Qwen2.5-VL-7B contrib
    """

    def __init__(self, config: Qwen2_5OmniInferenceConfig, layer_idx: int = 0):
        self.layer_idx = layer_idx

        # Determine if this layer uses sliding window
        sliding_window = None
        if hasattr(config, "layer_types") and config.layer_types:
            if config.layer_types[layer_idx] == "sliding_attention":
                sliding_window = getattr(config, "sliding_window", None)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            qkv_bias=config.qkv_bias,
            o_bias=config.o_bias,
            rotary_emb=Qwen2_5OmniRotaryEmbedding(config),
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
        )
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.mrope_section = config.mrope_section

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_multimodal_rotary_pos_emb(
                Q, K, cos_cache, sin_cache, self.mrope_section
            )
        return Q, K, cos_cache, sin_cache

        # Determine if this layer uses sliding window
        sliding_window = None
        if hasattr(config, "layer_types") and config.layer_types:
            if config.layer_types[layer_idx] == "sliding_attention":
                sliding_window = getattr(config, "sliding_window", None)

        # Initialize base attention
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            qkv_bias=config.qkv_bias,  # Qwen2.5-Omni has bias in QKV
            o_bias=config.o_bias,  # No bias in output
            rotary_emb=rotary_emb,
            sliding_window=sliding_window,
        )


class NeuronQwen2_5OmniDecoderLayer(nn.Module):
    """
    Qwen2.5-Omni decoder layer for NeuronX.

    Architecture (pre-norm):
    1. hidden = input + self_attn(norm(input))
    2. output = hidden + mlp(norm(hidden))

    Reference:
    - HF: Qwen2_5OmniDecoderLayer in modeling_qwen2_5_omni.py
    - NXD: NeuronQwen2DecoderLayer in modeling_qwen2.py
    """

    def __init__(self, config: Qwen2_5OmniInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention with layer-specific sliding window
        self.self_attn = NeuronQwen2_5OmniAttention(config, layer_idx=layer_idx)

        # MLP -- SwiGLU, same architecture as LLaMA
        self.mlp = NeuronLlamaMLP(config)

        # Layer normalization (RMSNorm)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Store attention type for this layer
        self.attention_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else "full_attention"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, None)
        """
        # Pre-norm: normalize before attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # Pre-norm: normalize before MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)[0]  # Take first element of tuple

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen2_5OmniModel(NeuronBaseModel):
    """
    Qwen2.5-Omni text model for NeuronX inference.

    This implements the core text model (Thinker) from Qwen2.5-Omni,
    focusing on text-only inference without multimodal components.

    Architecture:
    - Token embeddings
    - Stack of decoder layers with GQA and SwiGLU
    - RMSNorm
    - LM head for token generation

    Reference:
    - HF: Qwen2_5OmniThinkerTextModel in modeling_qwen2_5_omni.py
    - NXD: NeuronQwen2Model in modeling_qwen2.py
    """

    def setup_attr_for_model(self, config: Qwen2_5OmniInferenceConfig):
        """Setup attributes for model initialization"""
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen2_5OmniInferenceConfig):
        """Initialize the model components"""
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
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                NeuronQwen2_5OmniDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final normalization
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # LM head for token generation
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronQwen2_5OmniForCausalLM(NeuronBaseForCausalLM):
    """
    Qwen2.5-Omni for Causal Language Modeling on NeuronX.

    This is the main entry point for using Qwen2.5-Omni on NeuronX.
    It provides a HuggingFace-compatible interface for text generation.

    Usage:
        config = Qwen2_5OmniInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
        model = NeuronQwen2_5OmniForCausalLM(config)
        model.load_state_dict(state_dict)
        model.compile_model()
        model.save_pretrained(output_path)

    Reference:
    - HF: Qwen2_5OmniThinkerForConditionalGeneration in modeling_qwen2_5_omni.py
    - NXD: NeuronQwen2ForCausalLM in modeling_qwen2.py
    """

    _model_cls = NeuronQwen2_5OmniModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    @classmethod
    def get_config_cls(cls):
        return Qwen2_5OmniInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """
        Convert HuggingFace Qwen2.5-Omni state dict to NeuronX format.

        The Qwen2.5-Omni checkpoint has a nested structure with the text model under
        'model.thinker.model' prefix. This function extracts and renames the weights.

        NeuronAttentionBase expects QKV weights in the format:
        - layers.{i}.self_attn.qkv_proj.q_proj.weight (for separate Q/K/V)
        - layers.{i}.self_attn.qkv_proj.q_proj.bias

        Weight mappings:
        - thinker.model.embed_tokens.weight -> embed_tokens.weight
        - thinker.model.layers.{i}.self_attn.q_proj.* -> layers.{i}.self_attn.qkv_proj.q_proj.*
        - thinker.model.layers.{i}.self_attn.k_proj.* -> layers.{i}.self_attn.qkv_proj.k_proj.*
        - thinker.model.layers.{i}.self_attn.v_proj.* -> layers.{i}.self_attn.qkv_proj.v_proj.*
        - thinker.model.layers.{i}.self_attn.o_proj.* -> layers.{i}.self_attn.o_proj.*
        - thinker.model.layers.{i}.mlp.* -> layers.{i}.mlp.*
        - thinker.model.norm.weight -> norm.weight
        - thinker.lm_head.weight -> lm_head.weight

        Args:
            state_dict: HuggingFace state dictionary
            config: Model configuration

        Returns:
            Dictionary with NeuronX-formatted weights
        """
        import torch

        neuron_state_dict = {}

        # Remove prefixes: either "model.thinker.model." or just "thinker.model."
        # The actual prefix might vary
        possible_prefixes = [
            "model.thinker.model.",
            "thinker.model.",
            "model.thinker.",
            "thinker.",
            "",
        ]

        # Detect which prefix is used
        actual_prefix = ""
        for prefix in possible_prefixes:
            test_key = f"{prefix}embed_tokens.weight"
            if test_key in state_dict:
                actual_prefix = prefix
                break

        print(f"Detected HF weight prefix: '{actual_prefix}'")

        for name, param in state_dict.items():
            # Skip weights not belonging to the text model (thinker)
            if not name.startswith(actual_prefix) and actual_prefix != "":
                # Check if this is the lm_head
                lm_head_patterns = [
                    "model.thinker.lm_head.",
                    "thinker.lm_head.",
                    "lm_head.",
                ]
                is_lm_head = any(name.startswith(p) for p in lm_head_patterns)
                if not is_lm_head:
                    continue

            # Remove the prefix
            if actual_prefix and name.startswith(actual_prefix):
                new_name = name[len(actual_prefix) :]
            else:
                # Handle lm_head separately
                for lm_prefix in [
                    "model.thinker.lm_head.",
                    "thinker.lm_head.",
                    "lm_head.",
                ]:
                    if name.startswith(lm_prefix):
                        new_name = name[len(lm_prefix) :]
                        new_name = "lm_head." + new_name
                        break
                else:
                    new_name = name

            # Map attention weights to qkv_proj structure
            if ".self_attn.q_proj." in new_name:
                new_name = new_name.replace(
                    ".self_attn.q_proj.", ".self_attn.qkv_proj.q_proj."
                )
            elif ".self_attn.k_proj." in new_name:
                new_name = new_name.replace(
                    ".self_attn.k_proj.", ".self_attn.qkv_proj.k_proj."
                )
            elif ".self_attn.v_proj." in new_name:
                new_name = new_name.replace(
                    ".self_attn.v_proj.", ".self_attn.qkv_proj.v_proj."
                )

            # Clone and store the parameter
            neuron_state_dict[new_name] = param.clone()

        print(
            f"Converted {len(state_dict)} HF weights to {len(neuron_state_dict)} Neuron weights"
        )

        # Verify key weights exist
        required_keys = ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
        for key in required_keys:
            if key not in neuron_state_dict:
                print(
                    f"⚠️  Warning: Required key '{key}' not found in converted state dict"
                )

        # Verify layer 0 attention weights
        layer0_attn_keys = [
            "layers.0.self_attn.qkv_proj.q_proj.weight",
            "layers.0.self_attn.qkv_proj.k_proj.weight",
            "layers.0.self_attn.qkv_proj.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
        ]
        for key in layer0_attn_keys:
            if key not in neuron_state_dict:
                print(f"⚠️  Warning: Layer 0 attention key '{key}' not found")

        # Apply fused QKV if enabled
        if config.neuron_config.fused_qkv:
            neuron_state_dict = _convert_state_dict_to_fused_qkv(
                neuron_state_dict, config
            )

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Qwen2.5-Omni does not tie embeddings, but handle it gracefully."""
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()


# ---------------------------------------------------------------------------
# Fused QKV helpers -- must fuse both weight AND bias for QKV-bias models
# ---------------------------------------------------------------------------


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    q_key = f"layers.{layer_num}.self_attn.qkv_proj.q_proj.{attr}"
    k_key = f"layers.{layer_num}.self_attn.qkv_proj.k_proj.{attr}"
    v_key = f"layers.{layer_num}.self_attn.qkv_proj.v_proj.{attr}"
    fused_key = f"layers.{layer_num}.self_attn.Wqkv.{attr}"

    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        state_dict[fused_key] = torch.cat(
            [state_dict[q_key], state_dict[k_key], state_dict[v_key]]
        )
        del state_dict[q_key]
        del state_dict[k_key]
        del state_dict[v_key]


def _convert_state_dict_to_fused_qkv(state_dict, cfg):
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None) or []
    for layer in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, layer, "weight")
        _helper_concat_and_delete_qkv(state_dict, layer, "bias")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled
            or cfg.neuron_config.quantized
        ) and f"layers.{layer}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, layer, "scale")
    gc.collect()
    return state_dict
