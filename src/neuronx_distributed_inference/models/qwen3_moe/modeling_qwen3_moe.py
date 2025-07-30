# coding=utf-8
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
""" Qwen3 MOE model for NXD inference."""
import gc
import warnings
from typing import List, Optional, Tuple, Union

import torch

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Qwen3MoeForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe import initialize_moe_module

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config):
    """
    Helper function which converts the huggingface checkpoints to state dictionary compatible with the stucture of the neuron MoE model.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # to facilitate rank usage in base model
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]

        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        # Copy router weights
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        intermediate_size, hidden_size = neuron_state_dict[
            f"layers.{l}.mlp.experts.0.gate_proj.weight"
        ].shape
        device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype

        # copy the MLP parameters
        gate_up_proj = torch.empty(
            config.num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_experts):
            # Copy gate_proj and up_proj after concatenation
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                .T.detach()
                .clone()
            )
            up_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                .T.detach()
                .clone()
            )

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(
                gate_up_proj_slice, 2, intermediate_size, intermediate_size
            )
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(
            config.num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_experts):
            # Copy down_proj
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
                .T.detach()
                .clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return Qwen3MoeRMSNorm if cpu_mode() else CustomRMSNorm


class Qwen3MoeInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


class NeuronQwen3MoEAttention(NeuronAttentionBase):
    def __init__(self, config: Qwen3MoeInferenceConfig):
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
            rms_norm_eps=config.rms_norm_eps,
            # qk_norm in the base class is different from Qwen3RMSNorm
            use_qk_norm=False,
        )

        # Override q_layernorm and k_layernorm with RMSNorm
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3MoEAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )


class NeuronQwen3MoeDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config=config)

        self.mlp = initialize_moe_module(
            config=config,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
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
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.FloatTensor`, *optional*):
                position ids of size `(batch_size, sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronQwen3MoeModel(NeuronBaseModel):
    """
    NeuronQwen3MoeModel extends the Qwen3MoeModel to be traceable.
    The forward function of this class is traced.
    """

    def setup_attr_for_model(self, config: Qwen3MoeInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Qwen3MoeInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronQwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronQwen3MoeForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as Qwen3MoeForCausalLM
    """

    _model_cls = NeuronQwen3MoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen3MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3MoeInferenceConfig) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
