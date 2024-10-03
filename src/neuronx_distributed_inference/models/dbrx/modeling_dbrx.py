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
""" PyTorch Dbrx model for NXD inference."""
import gc
import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import DbrxForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.moe import initialize_moe_module

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

logger = logging.getLogger(__name__)


def convert_dbrx_to_neuron_state_dict(dbrx_state_dict, config):
    """
    Helper function which returns the model weights from the dbrx model in a state dictionary compatible with the stucture of the neuron MoE model.
    """

    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported for Dbrx Top-K model"
    neuron_state_dict = {}
    neuron_state_dict["embed_tokens.weight"] = dbrx_state_dict["wte.weight"].clone().detach()
    neuron_state_dict["norm.weight"] = dbrx_state_dict["norm_f.weight"].clone().detach()
    neuron_state_dict["lm_head.weight"] = dbrx_state_dict["lm_head.weight"].clone().detach()

    for l in range(config.n_layers):  # noqa: E741
        # Copy router weights
        neuron_state_dict[f"layers.{l}.ffn.router.linear_router.weight"] = (
            dbrx_state_dict[f"blocks.{l}.ffn.router.layer.weight"].clone().detach()
        )

        num_experts = config.ffn_config.moe_num_experts
        intermediate_size, hidden_size = (
            config.ffn_config.ffn_hidden_size,
            config.d_model,
        )

        # Copy gate_proj and up_proj after concatenation
        # [num_experts, hidden_size, 2 * intermediate_size]
        gate_proj_weights = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.w1"].view(
            num_experts, intermediate_size, hidden_size
        )
        up_proj_weights = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.v1"].view(
            num_experts, intermediate_size, hidden_size
        )
        gate_up_proj = torch.cat([gate_proj_weights, up_proj_weights], dim=1).transpose(1, 2)
        neuron_state_dict[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Copy down_proj
        # [num_experts, intermediate_size, hidden_size]
        down_proj = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.w2"].view(
            num_experts, intermediate_size, hidden_size
        )
        neuron_state_dict[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        neuron_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.attn.Wqkv.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.attn.out_proj.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.input_layernorm.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.norm_1.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.post_attention_layernorm.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.norm_2.weight"].clone().detach()
        )

    dbrx_state_dict.clear()
    gc.collect()

    return neuron_state_dict


class NeuronDbrxConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = True


class DbrxInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "d_model",
            "n_heads",
            "max_seq_len",
            "emb_pdrop",
            "resid_pdrop",
            "pad_token_id",
            "vocab_size",
            "attn_config",
            "ffn_config",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronDbrxConfig


class NeuronDbrxAttention(NeuronAttentionBase):
    def __init__(self, config: DbrxInferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_seq_len
        self.torch_dtype = config.neuron_config.torch_dtype
        self.padding_side = config.neuron_config.padding_side
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.rope_theta = config.attn_config.rope_theta

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronDbrxAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = config.attn_config.clip_qkv

        self.init_gqa_properties()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class NeuronDbrxBlock(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: DbrxInferenceConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.self_attn = NeuronDbrxAttention(config=config)

        self.ffn = initialize_moe_module(
            config=config, 
            num_experts=config.ffn_config.moe_num_experts, 
            top_k=config.ffn_config.moe_top_k, 
            hidden_size=config.d_model, 
            intermediate_size=config.ffn_config.ffn_hidden_size, 
            hidden_act=config.ffn_config.ffn_act_fn["name"],
        )

        self.input_layernorm = nn.LayerNorm(config.d_model, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, bias=False)

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
        hidden_states = self.input_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # FFN
        hidden_states = self.ffn(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)

        return outputs


class NeuronDbrxModel(NeuronBaseModel):
    """Transformer decoder consisting of *config.num_hidden_layers*. Each layer is a [`DbrxBlock`] layer.

    Args:
        config ([`DbrxConfig`]): Model configuration class with all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def setup_attr_for_model(self, config: DbrxInferenceConfig):
        self.emb_pdrop = config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: DbrxInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.d_model,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronDbrxBlock(config, block_idx) for block_idx in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model, bias=False)
        self.lm_head = ColumnParallelLinear(
            config.d_model,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronDbrxForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as DbrxForCausalLM
    """

    _STATE_DICT_MODEL_PREFIX = "transformer."

    _model_cls = NeuronDbrxModel

    @staticmethod
    def load_hf_model(model_path):
        return DbrxForCausalLM.from_pretrained(model_path)

    @classmethod
    def get_config_cls(cls):
        return DbrxInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: DbrxInferenceConfig) -> dict:
        return convert_dbrx_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Run collectives without pipelining
        compiler_args += " --tensorizer-options='--skip-pass=SimpleAllReduceTiling'"
        # Prevent auto-downcasting when running with fp32
        if self.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        return compiler_args
