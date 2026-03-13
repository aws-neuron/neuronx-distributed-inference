# Copyright 2025 Ouro AI and Amazon Web Services. All rights reserved.
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
"""PyTorch Ouro model for NeuronX Distributed Inference.

Ouro is a Universal Transformer that runs its layers multiple times (total_ut_steps=4).
The UT loop is unrolled into total_ut_steps * num_hidden_layers physical layers so that
NXDI can iterate them in a single pass. Intermediate RMSNorm is applied at the end of
each UT step group (every num_hidden_layers layers).

HF Ouro forward (pseudocode):
  for ut in range(4):
      for layer in layers:
          hidden = layer(hidden, cache_idx=ut*24+layer_idx)
      hidden = norm(hidden)

NXDI unrolled (96 physical layers, weights shared across UT copies):
  for idx in range(96):
      hidden = layers[idx](hidden, cache_idx=idx)
      if (idx+1) % 24 == 0 and idx < 95:  # end of UT step (except last)
          hidden = intermediate_norm(hidden)
  hidden = final_norm(hidden)

Architectural novelties preserved:
- Dual layer norms: pre-norm + post-norm sandwich for both attention and MLP
- Full Universal Transformer loop via layer unrolling
- Separate KV cache per UT step (96 cache slots total)
"""

import logging
from typing import List, Optional, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.activations import ACT2FN

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger("Neuron")

# Ouro has 4 UT steps with 24 physical layers
TOTAL_UT_STEPS = 4


def get_rmsnorm_cls():
    if cpu_mode():
        class RMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

        return RMSNorm
    else:
        return CustomRMSNorm


class OuroInferenceConfig(InferenceConfig):
    """Configuration for Ouro model on NeuronX.

    Overrides num_hidden_layers to the unrolled count (total_ut_steps * hf_layers)
    so NXDI allocates the correct number of KV cache slots.
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Save the original HF layer count before overriding
        if not hasattr(self, "_hf_num_hidden_layers"):
            self._hf_num_hidden_layers = self.num_hidden_layers
        # Override num_hidden_layers to unrolled count for NXDI KV cache allocation
        total_ut_steps = getattr(self, "total_ut_steps", TOTAL_UT_STEPS)
        self.num_hidden_layers = self._hf_num_hidden_layers * total_ut_steps
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False

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
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig = None, **kwargs):
        if neuron_config is None:
            neuron_config = NeuronConfig(tp_degree=1, batch_size=1, seq_len=128)
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config = cls(
            neuron_config=neuron_config,
            load_config=load_pretrained_config(hf_config=hf_config),
            **kwargs,
        )
        return config


class NeuronOuroAttention(NeuronAttentionBase):
    """Ouro attention for NeuronX. Standard MHA with RoPE (no GQA)."""

    def __init__(self, config: OuroInferenceConfig, layer_idx: int):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        rotary_emb = RotaryEmbedding(
            head_dim,
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
            qkv_bias=False,
            o_bias=False,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=None,
        )


class NeuronOuroMLP(nn.Module):
    """Ouro MLP with SwiGLU activation."""

    def __init__(self, config: OuroInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronOuroDecoderLayer(nn.Module):
    """Ouro decoder layer with dual pre+post norm sandwich pattern.

    Optionally applies an intermediate RMSNorm at UT step boundaries.
    In the original model, norm is applied after each complete pass through
    all layers. We replicate this by adding intermediate_norm to the last
    layer of each UT step group (except the final one, handled by self.norm).
    """

    def __init__(self, config: OuroInferenceConfig, layer_idx: int, apply_intermediate_norm: bool = False):
        super().__init__()
        self.self_attn = NeuronOuroAttention(config, layer_idx)
        self.mlp = NeuronOuroMLP(config)

        # Dual norms: pre-norm + post-norm for attention
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        # Dual norms: pre-norm + post-norm for MLP
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # Intermediate norm at UT step boundaries
        self.apply_intermediate_norm = apply_intermediate_norm
        if apply_intermediate_norm:
            self.intermediate_norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        **kwargs,
    ):
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Post-attention norm + residual
        hidden_states = self.input_layernorm_2(attn_output.hidden_states)
        hidden_states = residual + hidden_states

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Post-MLP norm + residual
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Apply intermediate norm at UT step boundary
        if self.apply_intermediate_norm:
            hidden_states = self.intermediate_norm(hidden_states)

        return (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, None)


class NeuronOuroModel(NeuronBaseModel):
    """Ouro base model for NeuronX with unrolled Universal Transformer.

    Creates num_hidden_layers * total_ut_steps physical layers. Each UT step
    group ends with an intermediate RMSNorm (except the last, which uses self.norm).
    """

    def setup_attr_for_model(self, config: OuroInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = None

    def init_model(self, config: OuroInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        hf_layers = config._hf_num_hidden_layers
        total_ut_steps = config.num_hidden_layers // hf_layers
        total_layers = config.num_hidden_layers  # = hf_layers * total_ut_steps

        layers = []
        for idx in range(total_layers):
            # Apply intermediate norm at end of each UT step (except last)
            is_ut_boundary = (idx + 1) % hf_layers == 0 and idx < total_layers - 1
            layers.append(NeuronOuroDecoderLayer(config, idx, apply_intermediate_norm=is_ut_boundary))
        self.layers = nn.ModuleList(layers)

        # Final norm (applied after last UT step)
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronOuroForCausalLM(NeuronBaseForCausalLM):
    """Ouro causal LM for NeuronX inference with unrolled UT loop."""

    _model_cls = NeuronOuroModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Convert HF Ouro weights to Neuron format with UT loop unrolling.

        HF has 24 layers. We unroll to 96 layers (4 UT steps × 24), duplicating
        weights for each UT step copy.

        Weight mapping (for each UT step u=0..3, HF layer i=0..23):
          Neuron layer idx = u * 24 + i
          layers.{idx}.* = model.layers.{i}.*

        The intermediate_norm at UT boundaries gets model.norm.weight.
        """
        neuron_config = config.neuron_config
        hf_layers = config._hf_num_hidden_layers
        total_ut_steps = config.num_hidden_layers // hf_layers

        # Strip "model." prefix
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                stripped[key[6:]] = value
            else:
                stripped[key] = value

        new_state_dict = {}

        # Map embed_tokens, lm_head, final norm
        for key in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
            if key in stripped:
                new_state_dict[key] = stripped[key]

        # Pre-group layer keys by HF layer index (O(N) instead of O(96N) scanning)
        layer_keys = {i: [] for i in range(hf_layers)}
        for key, value in stripped.items():
            if not key.startswith("layers."):
                continue
            # Extract layer index and suffix
            rest = key[len("layers."):]
            dot_pos = rest.index(".")
            hf_idx = int(rest[:dot_pos])
            suffix = rest[dot_pos + 1:]

            # Rename attention keys to NXDI format
            if suffix.startswith("self_attn.q_proj."):
                suffix = suffix.replace("self_attn.q_proj.", "self_attn.qkv_proj.q_proj.")
            elif suffix.startswith("self_attn.k_proj."):
                suffix = suffix.replace("self_attn.k_proj.", "self_attn.qkv_proj.k_proj.")
            elif suffix.startswith("self_attn.v_proj."):
                suffix = suffix.replace("self_attn.v_proj.", "self_attn.qkv_proj.v_proj.")
            elif suffix.startswith("self_attn.o_proj."):
                suffix = suffix.replace("self_attn.o_proj.", "self_attn.o_proj.o_proj.")

            if hf_idx in layer_keys:
                layer_keys[hf_idx].append((suffix, value))

        # Unroll layer weights: duplicate each HF layer for each UT step
        for ut_step in range(total_ut_steps):
            for hf_idx in range(hf_layers):
                nxdi_idx = ut_step * hf_layers + hf_idx

                for suffix, value in layer_keys[hf_idx]:
                    # First UT step can reuse original tensors; subsequent steps need copies
                    new_state_dict[f"layers.{nxdi_idx}.{suffix}"] = value if ut_step == 0 else value.clone()

                # Map intermediate_norm at UT boundaries (model.norm.weight)
                is_ut_boundary = (nxdi_idx + 1) % hf_layers == 0 and nxdi_idx < config.num_hidden_layers - 1
                if is_ut_boundary and "norm.weight" in stripped:
                    new_state_dict[f"layers.{nxdi_idx}.intermediate_norm.weight"] = stripped["norm.weight"].clone()

        # Add rank utilities for TP
        tp_degree = neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Skip early_exit_gate and rotary_emb weights
        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return OuroInferenceConfig
