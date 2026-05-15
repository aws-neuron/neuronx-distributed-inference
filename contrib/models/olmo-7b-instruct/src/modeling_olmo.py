# coding=utf-8
# Copyright 2024 Allen Institute for AI and the HuggingFace Inc. team.
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
PyTorch OLMo-7B-Instruct model for NeuronX Distributed Inference.

Forked from the Gemma 2 contrib port with the following key differences:
- Non-affine LayerNorm (no learnable weight/bias) instead of RMSNorm
- MHA with 32 heads (not GQA)
- SwiGLU activation (SiLU gating)
- No embedding scaling
- Fused QKV and fused MLP weights in HF checkpoint (hf_olmo format)
- No weight tying between embeddings and lm_head
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding


# ====================================================================================
# Configuration Classes
# ====================================================================================


class OlmoNeuronConfig(NeuronConfig):
    """NeuronConfig for OLMo model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronOlmoAttention


class OlmoInferenceConfig(InferenceConfig):
    """Configuration class for OLMo-7B-Instruct inference on NeuronX.

    Handles the hf_olmo config format which uses different field names
    (d_model, n_heads, n_layers, etc.) and fused weight dimensions.
    """

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1

        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_cache"):
            self.use_cache = True

        if not hasattr(self, "attention_bias"):
            self.attention_bias = False
        if not hasattr(self, "attention_dropout"):
            self.attention_dropout = 0.0

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[OlmoNeuronConfig]:
        return OlmoNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "OlmoInferenceConfig":
        """Load and map hf_olmo config.json to standard InferenceConfig fields."""
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            hf = json.load(f)

        # Map hf_olmo fields to standard names
        config_dict = {
            "hidden_size": hf.get("d_model", 4096),
            # intermediate_size = half of fused ff_proj (22016 / 2 = 11008)
            "intermediate_size": hf.get("mlp_hidden_size", 22016) // 2,
            "num_hidden_layers": hf.get("n_layers", 32),
            "num_attention_heads": hf.get("n_heads", 32),
            # MHA: num_kv_heads == num_attention_heads
            "num_key_value_heads": hf.get("n_heads", 32),
            # Use embedding_size (padded to 50304) for the model dimension
            "vocab_size": hf.get("embedding_size", hf.get("vocab_size", 50280)),
            "max_position_embeddings": hf.get("max_sequence_length", 2048),
            "rope_theta": hf.get("rope_theta", 10000.0),
            "attention_bias": hf.get("include_bias", False),
            "pad_token_id": hf.get("pad_token_id", 1),
            "eos_token_id": hf.get("eos_token_id", 50279),
            "tie_word_embeddings": hf.get("weight_tying", False),
        }

        config_dict.update(kwargs)

        if neuron_config is None:
            neuron_config = NeuronConfig()

        config = cls(neuron_config=neuron_config, **config_dict)
        return config


# ====================================================================================
# Model Components
# ====================================================================================


class OlmoLayerNorm(nn.Module):
    """Non-affine LayerNorm using explicit tensor ops for Neuron traceability.

    OLMo uses LayerNorm without learnable weight/bias.
    Implemented with primitive ops (mean, var, sqrt) to ensure correct XLA tracing,
    since F.layer_norm(weight=None) may not trace correctly on NeuronX.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        mean = x_float.mean(-1, keepdim=True)
        var = ((x_float - mean) ** 2).mean(-1, keepdim=True)
        return ((x_float - mean) / torch.sqrt(var + self.eps)).to(x.dtype)


class NeuronOlmoAttention(NeuronAttentionBase):
    """OLMo MHA with RoPE. No bias, no Q-K norm, no sliding window."""

    def __init__(self, config: OlmoInferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads

        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
        )


class NeuronOlmoMLP(nn.Module):
    """OLMo MLP: SwiGLU (gate/up/down with SiLU activation)."""

    def __init__(self, config: OlmoInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_output = self.down_proj(gate_output * up_output)
        return down_output, None


class NeuronOlmoDecoderLayer(nn.Module):
    """OLMo decoder layer: pre-norm -> attn -> residual -> pre-norm -> MLP -> residual.

    Uses non-affine LayerNorm (no learnable parameters).
    """

    def __init__(self, config: OlmoInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = NeuronOlmoAttention(config)
        self.mlp = NeuronOlmoMLP(config)

        self.input_layernorm = OlmoLayerNorm(self.hidden_size)
        self.post_attention_layernorm = OlmoLayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# Model Classes
# ====================================================================================


class NeuronOlmoModel(NeuronBaseModel):
    """OLMo base model: embedding + 32 decoder layers + final LayerNorm + lm_head."""

    def setup_attr_for_model(self, config: OlmoInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: OlmoInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        self.layers = nn.ModuleList(
            [NeuronOlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = OlmoLayerNorm(config.hidden_size)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronOlmoForCausalLM(NeuronBaseForCausalLM):
    """OLMo-7B-Instruct for causal LM on NeuronX.

    Handles weight conversion from fused hf_olmo format:
    - att_proj [3*H, H] -> q_proj, k_proj, v_proj (split dim=0)
    - ff_proj [2*I, H] -> gate_proj, up_proj (split dim=0)
    """

    _model_cls = NeuronOlmoModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert hf_olmo fused weights to NXDI split format."""
        neuron_config = config.neuron_config
        neuron_state_dict = {}
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        hidden_size = config.hidden_size

        # The base class strips "model." prefix before calling this function.
        # So hf_olmo keys arrive as "transformer.blocks.N.*" (not "model.transformer.*").
        is_olmo_format = any(k.startswith("transformer.") for k in state_dict)

        if not is_olmo_format:
            # Already in standard format — just add rank utilities
            for k, v in state_dict.items():
                neuron_state_dict[k] = v.detach().clone()
            for i in range(num_layers):
                neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                    0, tp_degree, dtype=torch.int32
                )
            neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
            return neuron_state_dict

        # === Convert from hf_olmo format (with "model." prefix already stripped) ===

        # Embeddings
        if "transformer.wte.weight" in state_dict:
            neuron_state_dict["embed_tokens.weight"] = (
                state_dict["transformer.wte.weight"].detach().clone()
            )

        # LM head
        if "transformer.ff_out.weight" in state_dict:
            neuron_state_dict["lm_head.weight"] = (
                state_dict["transformer.ff_out.weight"].detach().clone()
            )

        for i in range(num_layers):
            old = f"transformer.blocks.{i}"
            new = f"layers.{i}"

            # Split fused QKV: att_proj [3*H, H] -> Q, K, V each [H, H]
            att_key = f"{old}.att_proj.weight"
            if att_key in state_dict:
                q, k, v = state_dict[att_key].split(hidden_size, dim=0)
                neuron_state_dict[f"{new}.self_attn.q_proj.weight"] = q.detach().clone()
                neuron_state_dict[f"{new}.self_attn.k_proj.weight"] = k.detach().clone()
                neuron_state_dict[f"{new}.self_attn.v_proj.weight"] = v.detach().clone()

            # Output projection
            o_key = f"{old}.attn_out.weight"
            if o_key in state_dict:
                neuron_state_dict[f"{new}.self_attn.o_proj.weight"] = (
                    state_dict[o_key].detach().clone()
                )

            # Split fused MLP: ff_proj [2*I, H] -> up [I, H], gate [I, H]
            # OLMo SwiGLU: x, gate = chunk(2); return silu(gate) * x
            # First half = "up" (multiplied), second half = "gate" (SiLU)
            ff_key = f"{old}.ff_proj.weight"
            if ff_key in state_dict:
                ff_proj = state_dict[ff_key]
                half = ff_proj.shape[0] // 2
                up, gate = ff_proj.split(half, dim=0)
                neuron_state_dict[f"{new}.mlp.gate_proj.weight"] = gate.detach().clone()
                neuron_state_dict[f"{new}.mlp.up_proj.weight"] = up.detach().clone()

            # Down projection
            down_key = f"{old}.ff_out.weight"
            if down_key in state_dict:
                neuron_state_dict[f"{new}.mlp.down_proj.weight"] = (
                    state_dict[down_key].detach().clone()
                )

            # Rank utility for attention TP sharding
            neuron_state_dict[f"{new}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # No norm weights to map (non-affine LayerNorm)

        neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if neuron_config.vocab_parallel:
            neuron_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        return neuron_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """OLMo does not tie weights — no-op."""
        pass

    @classmethod
    def get_config_cls(cls):
        return OlmoInferenceConfig
