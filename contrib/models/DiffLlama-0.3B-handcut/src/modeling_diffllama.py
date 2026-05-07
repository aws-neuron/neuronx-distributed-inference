# coding=utf-8
# Copyright 2024 DiffLlama team and AWS NeuronX Distributed Inference team.
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
"""DiffLlama model for NeuronX Distributed Inference.

Implements differential attention: the V tensor is transformed (chunk/cat/repeat)
before the attention matmul so that paired head groups share combined V values.
After attention, the output is split into two head groups and their difference
(scaled by learned lambda) is taken, followed by RMSNorm on 2*head_dim features.
"""

import math
import os
import json
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseModel,
    NeuronBaseForCausalLM,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    NeuronAttentionBaseOutput,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding, repeat_kv
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def lambda_init_fn(layer_idx):
    """Initialize lambda parameter based on layer index."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class Llama3RotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding with llama3 rope scaling support.

    Applies frequency-dependent scaling: high-frequency components are kept
    unchanged, low-frequency components are scaled down by `factor`, and
    mid-frequency components are smoothly interpolated.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, rope_scaling=None):
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)
        self.rope_scaling = rope_scaling

    def get_inv_freqs(self, device=None):
        inv_freq = super().get_inv_freqs(device)

        if self.rope_scaling is None:
            return inv_freq

        rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        if rope_type not in ("llama3",):
            return inv_freq

        factor = self.rope_scaling["factor"]
        low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = self.rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)


class DiffLlamaInferenceConfig(InferenceConfig):
    """Configuration class for DiffLlama model inference."""

    def __init__(self, neuron_config=None, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 128256)
        self.hidden_size = kwargs.pop("hidden_size", 2048)
        self.intermediate_size = kwargs.pop("intermediate_size", 8192)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 16)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 8)
        self.hidden_act = kwargs.pop("hidden_act", "silu")
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 131072)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-5)
        self.rope_theta = kwargs.pop("rope_theta", 500000.0)
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.lambda_std_dev = kwargs.pop("lambda_std_dev", 0.1)
        head_dim = kwargs.pop("head_dim", None)
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", 128000)
        self.eos_token_id = kwargs.pop("eos_token_id", 128001)
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict = True

        # Pop any remaining HF config keys that InferenceConfig doesn't expect
        for key in list(kwargs.keys()):
            if key not in ("neuron_config",):
                setattr(self, key, kwargs.pop(key))

        super().__init__(neuron_config, **kwargs)

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
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "DiffLlamaInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        # Remove keys that are metadata, not model config
        for meta_key in ("_name_or_path", "architectures", "model_type", "transformers_version"):
            config_dict.pop(meta_key, None)
        # torch_dtype is handled by neuron_config, not model config
        config_dict.pop("torch_dtype", None)
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


class NeuronDiffLlamaAttention(NeuronAttentionBase):
    """
    DiffLlama differential attention for NeuronX.

    Overrides standard_causal_attention_forward to implement the full
    differential attention mechanism with correct V transformation:

    1. Q, K, V projections + RoPE via NeuronAttentionBase.prep_qkv_tensors
    2. GQA expansion: repeat_kv on K and V
    3. V transformation: chunk V heads into two halves, concatenate along
       head_dim, then repeat to all heads -> V has 2*head_dim
    4. Standard attention: softmax(Q @ K^T / sqrt(d)) @ V_transformed
    5. Differential: split output into 2 head groups, subtract with lambda
    6. GroupNorm (RMSNorm) on 2*head_dim features
    7. Output projection
    """

    def __init__(self, config: DiffLlamaInferenceConfig, layer_idx: int):
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rotary_emb = Llama3RotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=rope_scaling,
            )
        else:
            rotary_emb = RotaryEmbedding(
                config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta,
        )

        self.layer_idx = layer_idx
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads

        # Differential attention parameters
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(config.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(config.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(config.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(config.head_dim))

        # GroupNorm for differential attention output (no learnable parameters)
        # Operates on 2*head_dim features (the concatenated V dimension)
        self.groupnorm = nn.RMSNorm(
            2 * config.head_dim, eps=config.rms_norm_eps, elementwise_affine=False
        )

    def standard_causal_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        active_mask=None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        rotary_position_ids=None,
        kv_mgr=None,
        get_kv_per_layer=False,
        update_kv_per_layer=False,
        residual=None,
        windowed_context_encoding_window_idx=-1,
        **kwargs,
    ):
        """
        Full differential attention forward.

        Instead of using NXDI's attention_tokengen/attention_context_encode
        (which can't handle the V transformation), we compute attention manually
        using standard PyTorch operations. This is correct for all sequence
        lengths and works with the Neuron compiler.
        """
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch_dtype)

        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        if rotary_position_ids is None:
            rotary_position_ids = position_ids

        if get_kv_per_layer:
            assert kv_mgr is not None
            past_key_value = kv_mgr.get_kv_by_layer_id(**kwargs)

        is_token_gen = past_key_value is not None
        if windowed_context_encoding_window_idx >= 0:
            is_token_gen = False
        if self.neuron_config.is_prefix_caching:
            is_token_gen = is_token_gen and q_len < 128

        # --- Step 1: Q, K, V projections + RoPE ---
        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rotary_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            residual=residual,
        )
        # Q: (B, num_heads, q_len, head_dim) e.g. (B, 32, q_len, 64)
        # K: (B, num_kv_heads, q_len, head_dim) e.g. (B, 8, q_len, 64)
        # V: (B, num_kv_heads, q_len, head_dim) e.g. (B, 8, q_len, 64)

        # --- Step 2: Handle KV cache ---
        if is_token_gen:
            K_cached, V_cached = past_key_value
            if self.k_cache_transposed:
                K_cached = K_cached.permute(0, 1, 3, 2)  # BHDS -> BHSD
            K_full = torch.cat([K_cached, K], dim=2)
            V_full = torch.cat([V_cached, V], dim=2)
        else:
            K_full = K
            V_full = V

        # --- Step 3: GQA expansion ---
        K_expanded = repeat_kv(K_full, self.num_kv_groups)  # (B, 32, S, 64)
        V_expanded = repeat_kv(V_full, self.num_kv_groups)  # (B, 32, S, 64)

        # --- Step 4: V transformation for differential attention ---
        # This matches the HF implementation exactly:
        # value_states = torch.cat(torch.chunk(value_states, 2, dim=1), dim=-1)
        # value_states = value_states.repeat(1, 2, 1, 1)
        V_first, V_second = torch.chunk(V_expanded, 2, dim=1)  # each (B, 16, S, 64)
        V_cat = torch.cat([V_first, V_second], dim=-1)  # (B, 16, S, 128)
        V_transformed = V_cat.repeat(1, 2, 1, 1)  # (B, 32, S, 128)

        # --- Step 5: Compute attention ---
        attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # (B, 32, q_len, S)

        # Apply causal mask. We generate our own to avoid shape mismatches with
        # the framework-provided mask (which may target NXDI's internal kernels).
        if not is_token_gen:
            # Context encoding: apply causal mask to prevent attending to future tokens
            kv_len = K_expanded.shape[-2]
            causal_mask = torch.triu(
                torch.full(
                    (q_len, kv_len),
                    torch.finfo(attn_weights.dtype).min,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask
        # Token gen: Q is a single token attending to all past positions (no mask needed)

        # Softmax in fp32 for numerical stability, then cast back
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

        attn_output = torch.matmul(attn_weights, V_transformed)  # (B, 32, q_len, 128)

        # --- Step 6: Differential mechanism ---
        attn_output1, attn_output2 = torch.chunk(
            attn_output, 2, dim=1
        )  # each (B, 16, q_len, 128)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, dtype=torch.float32)
        ).to(Q.dtype)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, dtype=torch.float32)
        ).to(Q.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_output = attn_output1 - lambda_full * attn_output2  # (B, 16, q_len, 128)

        # --- Step 7: GroupNorm (RMSNorm on 2*head_dim=128 features) ---
        attn_output = (1 - self.lambda_init) * self.groupnorm(
            attn_output
        )  # (B, 16, q_len, 128)

        # --- Step 8: Reshape and output projection ---
        # (B, 16, q_len, 128) -> (B, q_len, 16, 128) -> (B, q_len, 2048)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

        # --- Step 9: Prepare KV for cache update ---
        # Store original K, V (before repeat_kv / V transform) for the cache
        if self.k_cache_transposed:
            K = K.permute(0, 1, 3, 2)

        kv = (K, V)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        attn_output = attn_output.to(original_dtype)

        return NeuronAttentionBaseOutput(
            attn_output, kv, cos_cache, sin_cache, residual
        )


class NeuronDiffLlamaMLP(nn.Module):
    """DiffLlama MLP with SwiGLU activation for NeuronX."""

    def __init__(self, config: DiffLlamaInferenceConfig):
        super().__init__()
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
        self.act_fn = nn.SiLU()

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        output = self.down_proj(gate_output * up_output)
        return output, None


def get_rmsnorm_cls():
    return CustomRMSNorm


class NeuronDiffLlamaDecoderLayer(nn.Module):
    """DiffLlama decoder layer for NeuronX."""

    def __init__(self, config: DiffLlamaInferenceConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = NeuronDiffLlamaAttention(config, layer_idx)
        self.mlp = NeuronDiffLlamaMLP(config)
        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        adapter_ids=None,
        rotary_position_ids=None,
        residual=None,
        **kwargs,
    ):
        """
        Forward matching the NeuronBaseModel decoder layer interface.
        Returns: (hidden_states, kv, cos_cache, sin_cache, residual)
        """
        entry_hidden_states = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_ids,
            residual=residual,
            **kwargs,
        )

        residual = entry_hidden_states
        hidden_states = attn_output.hidden_states

        # Residual connection after attention
        hidden_states = residual + hidden_states
        residual = hidden_states

        # Post-attention norm + MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)

        # Residual connection after MLP
        hidden_states = residual + hidden_states
        residual = None

        return (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            residual,
        )


class NeuronDiffLlamaModel(NeuronBaseModel):
    """DiffLlama model for NeuronX."""

    def __init__(self, config: DiffLlamaInferenceConfig):
        super().__init__(config)

    def setup_attr_for_model(self, config: DiffLlamaInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: DiffLlamaInferenceConfig):
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.neuron_config.torch_dtype,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [
                NeuronDiffLlamaDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)


class NeuronDiffLlamaForCausalLM(NeuronBaseForCausalLM):
    """DiffLlama causal language model for NeuronX inference."""

    _model_cls = NeuronDiffLlamaModel

    @classmethod
    def get_config_cls(cls):
        return DiffLlamaInferenceConfig

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        import sys

        transformers_path = os.environ.get("DIFFLLAMA_TRANSFORMERS_PATH")
        if transformers_path and transformers_path not in sys.path:
            sys.path.insert(0, transformers_path)

        from transformers.models.diffllama.modeling_diffllama import (
            DiffLlamaForCausalLM,
        )
        from transformers.models.diffllama.configuration_diffllama import DiffLlamaConfig

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict.pop("_name_or_path", None)
        config = DiffLlamaConfig(**config_dict)

        model = DiffLlamaForCausalLM(config)

        from safetensors.torch import load_file as safe_load_file

        weights_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = safe_load_file(weights_path)
        else:
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")

        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert HF state dict to Neuron format.

        Keys arrive WITHOUT the 'model.' prefix (stripped by framework).
        QKV keys use self_attn.q_proj format (preshard_hook adds qkv_proj).
        """
        neuron_state_dict = {}
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree

        # Token embeddings
        if "embed_tokens.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = state_dict[
                "embed_tokens.weight"
            ]

        # Final normalization
        if "norm.weight" in state_dict:
            neuron_state_dict["norm.weight"] = state_dict["norm.weight"]

        # Find layer indices
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        layer_indices.add(int(parts[1]))
                    except ValueError:
                        pass

        for i in sorted(layer_indices):
            prefix = f"layers.{i}"

            # QKV projections
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key = f"{prefix}.self_attn.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key]

            # Output projection
            key = f"{prefix}.self_attn.o_proj.weight"
            if key in state_dict:
                neuron_state_dict[key] = state_dict[key]

            # Differential attention lambda parameters
            for param in ["lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"]:
                key = f"{prefix}.self_attn.{param}"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key]

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key]

            # Layer norms
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                key = f"{prefix}.{norm}.weight"
                if key in state_dict:
                    neuron_state_dict[key] = state_dict[key]

            # Rank utilities for tensor parallelism
            neuron_state_dict[f"{prefix}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Base model rank utilities
        neuron_state_dict["rank_util.rank"] = torch.arange(
            0, tp_degree, dtype=torch.int32
        )

        return neuron_state_dict


__all__ = [
    "DiffLlamaInferenceConfig",
    "NeuronDiffLlamaModel",
    "NeuronDiffLlamaForCausalLM",
    "NeuronDiffLlamaAttention",
    "NeuronDiffLlamaMLP",
    "NeuronDiffLlamaDecoderLayer",
    "Llama3RotaryEmbedding",
]
