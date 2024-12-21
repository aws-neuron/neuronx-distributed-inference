# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phi-3 model for NXD inference."""

import gc
from typing import List, Optional, Tuple, Type
from transformers import Phi3ForCausalLM
import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
from torch import nn
import torch.utils.checkpoint

from transformers.activations import ACT2FN


from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    GroupQueryAttention_QKV,
    GroupQueryAttention_O,
)

from neuronx_distributed.parallel_layers import utils
from transformers.models.phi3.modeling_phi3 import (
    Phi3RotaryEmbedding,
    Phi3RMSNorm,
    Phi3LongRoPEScaledRotaryEmbedding,
)
from transformers.models.phi3.configuration_phi3 import Phi3Config
import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="debug.txt",  # This will write to a file named debug.log
    filemode="w",
)  # 'w' mode overwrites the file each time

# Create a logger
logger = logging.getLogger(__name__)
_CHECKPOINT_FOR_DOC = "microsoft/Phi-3-mini-4k-instruct"
_PHI3_ATTENTION_CLASSES = {}


def get_rmsnorm_cls():
    return Phi3RMSNorm


def _register_module(key: str, cls: Type[nn.Module]):
    _PHI3_ATTENTION_CLASSES[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronPhi3Attention")
        class NeuronPhi3Attention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def convert_state_dict_to_neuron(phi3_state_dict, cfg: InferenceConfig):
    for l in range(cfg.num_hidden_layers):
        # Keep the original fused weight as Wqkv.weight
        phi3_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = phi3_state_dict[
            f"layers.{l}.self_attn.qkv_proj.weight"
        ].clone().detach()

        # Get the fused QKV weight
        fused_weight = phi3_state_dict[f"layers.{l}.self_attn.qkv_proj.weight"].clone().detach()
        fused_gate_up = phi3_state_dict[f"layers.{l}.mlp.gate_up_proj.weight"].clone().detach()

        # Split the fused weight into Q, K, and V using torch.chunk
        q_weight, k_weight, v_weight = torch.chunk(fused_weight, 3, dim=0)
        gate, up = torch.chunk(fused_gate_up, 2, dim=0)

        # Add the split weights to the state dict
        phi3_state_dict[f"layers.{l}.self_attn.qkv_proj.q_proj.weight"] = q_weight
        phi3_state_dict[f"layers.{l}.self_attn.qkv_proj.k_proj.weight"] = k_weight
        phi3_state_dict[f"layers.{l}.self_attn.qkv_proj.v_proj.weight"] = v_weight
        phi3_state_dict[f"layers.{l}.mlp.gate_proj.weight"] = gate
        phi3_state_dict[f"layers.{l}.mlp.up_proj.weight"] = up

        # Remove the original qkv_proj weight
        del phi3_state_dict[f"layers.{l}.self_attn.qkv_proj.weight"]
        del phi3_state_dict[f"layers.{l}.mlp.gate_up_proj.weight"]

    gc.collect()

    return phi3_state_dict


class NeuronPhi3InferenceConfig(InferenceConfig):
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
            "pad_token_id",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronPhi3MLP(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config

        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
            )

            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
            )

            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )

    def forward(self, hidden_state):
        if self.sequence_parallel_enabled:
            x = _gather_along_dim(x, self.sequence_dimension)
        else:
            x = hidden_state

        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


@register_module("NeuronPhi3Attention")
class NeuronPhi3Attention(NeuronAttentionBase):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        self.init_custom_gqa_properties()

        self.init_rope()

    def init_custom_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=False,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=False,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )
        self.num_heads = utils.divide(
            self.qkv_proj.get_num_attention_heads(), self.tp_degree
        )
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type")
            if scaling_type == "longrope":
                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                    self.head_dim, self.config
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


class NeuronPhi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.self_attn = NeuronPhi3Attention(
            config=config,
        )
        self.hidden_size = config.hidden_size

        self.mlp = NeuronPhi3MLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states, present_key_value = attn_outs
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value)


class NeuronPhi3Model(NeuronBaseModel):
    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        # self._attn_implementation = config._attn_implementation

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                # We choose to shard across embedding dimension because this stops XLA from introducing
                # rank specific constant parameters into the HLO. We could shard across vocab, but that
                # would require us to use non SPMD parallel_model_trace.
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
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
                pad=True,
            )

        self.layers = nn.ModuleList(
            [
                NeuronPhi3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class NeuronPhi3ForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends Phi3ForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronPhi3Model

    @staticmethod
    def load_hf_model(model_path):
        return Phi3ForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """This function should be over-ridden in child classes as needed"""

        state_dict = convert_state_dict_to_neuron(state_dict, config)
        return state_dict

    @classmethod
    def get_config_cls(cls):
        return NeuronPhi3InferenceConfig
