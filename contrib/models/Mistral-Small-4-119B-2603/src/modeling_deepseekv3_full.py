# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
NeuronDeepseekV3ForCausalLM - Full model implementation for DeepSeek-V3 / Mistral-Small-4
on AWS Neuron (NxDI).

Combines:
- MLA (Multi-head Latent Attention) from existing NxDI DeepseekV3Attention
- MoE with shared experts from NxDI moe_v2 (same as Qwen3_moe/Llama4 pattern)
- NeuronBaseForCausalLM framework

Based on:
- neuronx_distributed_inference/models/deepseek/modeling_deepseek.py (MLA attention)
- neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py (MoE + shared experts pattern)
- neuronx_distributed_inference/models/mixtral/modeling_mixtral.py (base MoE pattern)
"""

import gc
import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3InferenceConfig,
    DeepseekV3RMSNorm,
    get_rmsnorm_cls,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

logger = logging.getLogger(__name__)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


class DeepseekV3MoEInferenceConfig(InferenceConfig):
    """
    Config class for DeepSeek-V3 / Mistral-Small-4 models with MoE + MLA.
    Extends InferenceConfig with MoE-specific fields.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Map n_routed_experts -> num_local_experts for MoE module compatibility
        if hasattr(self, "n_routed_experts") and not hasattr(self, "num_local_experts"):
            self.num_local_experts = self.n_routed_experts

        # Ensure n_shared_experts is set (default 0)
        if not hasattr(self, "n_shared_experts"):
            self.n_shared_experts = 0

        # MoE intermediate size for experts
        # moe_v2's initialize_moe_module reads config.intermediate_size for expert FFN size
        # But we need to preserve the original intermediate_size for shared experts
        self.shared_expert_intermediate_size = getattr(
            self, "intermediate_size", self.hidden_size * 3
        )
        # Set intermediate_size to moe_intermediate_size for the MoE module
        if hasattr(self, "moe_intermediate_size"):
            self.intermediate_size = self.moe_intermediate_size

        # Router config
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"

        # Normalize top-k affinities
        if hasattr(self, "norm_topk_prob") and self.norm_topk_prob:
            self.neuron_config.normalize_top_k_affinities = True

        # GLU MLP for SiLU activation
        self.neuron_config.glu_mlp = True

        # Set disable_numeric_cc_token as workaround (same as Qwen3_moe)
        self.neuron_config.disable_numeric_cc_token = True

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Ensure rope_scaling is accessible as dict
        if hasattr(self, "rope_scaling") and isinstance(self.rope_scaling, dict):
            # Already a dict, good
            pass

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            # MLA fields
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
            # MoE fields
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            # RoPE
            "rope_scaling",
            "rope_theta",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


def convert_deepseekv3_to_neuron_state_dict(neuron_state_dict, config):
    """
    Convert DeepSeek-V3 / Mistral-Small-4 HF state dict to NxDI format.

    Key transformations:
    1. Router: mlp.gate.weight -> mlp.router.linear_router.weight
    2. Experts: mlp.experts.gate_up_proj [N,H,2I] -> mlp.expert_mlps.mlp_op.gate_up_proj.weight
    3. Experts: mlp.experts.down_proj [N,H,I] -> mlp.expert_mlps.mlp_op.down_proj.weight
    4. Shared experts: mlp.shared_experts.{gate,up,down}_proj.weight -> mlp.shared_experts.*
    5. MLA attention keys are already compatible (q_a_proj, kv_a_proj_with_mqa, etc.)
    6. Add rank_util tensors

    Input state dict keys (after stripping "model." prefix by HF adapter):
      layers.N.self_attn.q_a_proj.weight
      layers.N.self_attn.q_a_layernorm.weight
      layers.N.self_attn.q_b_proj.weight
      layers.N.self_attn.kv_a_proj_with_mqa.weight
      layers.N.self_attn.kv_a_layernorm.weight
      layers.N.self_attn.kv_b_proj.weight
      layers.N.self_attn.o_proj.weight
      layers.N.input_layernorm.weight
      layers.N.post_attention_layernorm.weight
      layers.N.mlp.gate.weight                        (router)
      layers.N.mlp.experts.gate_up_proj                (grouped: [128, 4096, 4096])
      layers.N.mlp.experts.down_proj                   (grouped: [128, 4096, 2048])
      layers.N.mlp.shared_experts.gate_proj.weight
      layers.N.mlp.shared_experts.up_proj.weight
      layers.N.mlp.shared_experts.down_proj.weight
      embed_tokens.weight
      norm.weight
      lm_head.weight
    """
    assert config.neuron_config.glu_mlp is True, (
        "Only GLU MLP is supported for DeepSeek-V3"
    )

    # Add rank utility tensor
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Add per-layer rank utility for attention
        neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # ---- Router ----
        router_key = f"layers.{l}.mlp.gate.weight"
        if router_key in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict[router_key].detach().clone()
            )
            del neuron_state_dict[router_key]

        # ---- Routed Experts ----
        # The HF checkpoint stores grouped expert weights following nn.Linear convention
        # (weight shape = [out_features, in_features]):
        #   mlp.experts.gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
        #   mlp.experts.down_proj: [num_experts, hidden_size, moe_intermediate_size]
        #
        # NxDI MoE einsum "e...h,ehi->e...i" expects weights as [E, in_dim, out_dim]:
        #   mlp.expert_mlps.mlp_op.gate_up_proj.weight: [num_experts, hidden_size, 2*intermediate_size]
        #   mlp.expert_mlps.mlp_op.down_proj.weight: [num_experts, intermediate_size, hidden_size]
        #
        # Both need transposition of dims 1 and 2.
        # NOTE: For Mistral-Small-4, hidden_size == 2*moe_intermediate_size == 4096,
        # so the shapes are square [E, 4096, 4096]. The transpose changes the DATA LAYOUT
        # even though the shape appears unchanged.

        gate_up_key = f"layers.{l}.mlp.experts.gate_up_proj"
        if gate_up_key in neuron_state_dict:
            gate_up = neuron_state_dict[gate_up_key]
            # gate_up is [E, 2I, H] (HF convention) -> transpose to [E, H, 2I] (NxDI convention)
            neuron_state_dict[
                f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
            ] = gate_up.transpose(1, 2).contiguous()
            del neuron_state_dict[gate_up_key]

        down_key = f"layers.{l}.mlp.experts.down_proj"
        if down_key in neuron_state_dict:
            down = neuron_state_dict[down_key]
            # down is [E, H, I] (HF convention) -> transpose to [E, I, H] (NxDI convention)
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
                down.transpose(1, 2).contiguous()
            )
            del neuron_state_dict[down_key]

        # ---- Shared Experts ----
        # SharedExperts in NxDI moe_v2 expects:
        #   mlp.shared_experts.gate_proj.weight (or fused gate_up_proj)
        #   mlp.shared_experts.up_proj.weight
        #   mlp.shared_experts.down_proj.weight
        # The HF keys already match! No renaming needed for shared experts.
        # But we need to check if NxDI SharedExperts uses fused gate+up or separate.
        # From moe_v2.py: SharedExperts takes fused_gate_up_projection param.
        # If fused, it expects: shared_experts.gate_up_proj.weight
        # If not fused, it expects separate gate_proj and up_proj.
        # We'll use non-fused (separate) for simplicity since that matches the HF format.

        # ---- Remove FP8 scale keys if any remain ----
        keys_to_delete = []
        for key in list(neuron_state_dict.keys()):
            if key.startswith(f"layers.{l}.") and (
                "_scale_inv" in key or "activation_scale" in key
            ):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del neuron_state_dict[key]

        gc.collect()

    return neuron_state_dict


class NeuronDeepseekV3DecoderLayer(nn.Module):
    """
    DeepSeek-V3 decoder layer: MLA attention + MoE with shared experts.
    """

    def __init__(self, config: DeepseekV3MoEInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # MLA Attention
        self.self_attn = DeepseekV3Attention(
            config=config,
            layer_idx=layer_idx,
            tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        )

        # MoE with shared experts
        self.mlp = initialize_moe_module(config=config)

        # Layer norms
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
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Please use `attention_mask` instead."
            )

        # Convert from KV cache format (batch, 1, seq, 320) back to MLA format
        # past_key_value from cache manager is (k_cache, v_cache) tuple
        # k_cache has shape (batch, 1, seq_len, qk_rope_head_dim + kv_lora_rank)
        # We need to pass a single concatenated tensor to the attention
        mla_past_kv = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)):
                # From KV cache manager: (k_cache, v_cache)
                k_cache = past_key_value[0]
                # k_cache shape: (batch, 1, seq_len, 320)
                mla_past_kv = k_cache.squeeze(1)  # (batch, seq_len, 320)
            else:
                mla_past_kv = past_key_value

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # MLA Self Attention
        hidden_states, present_key_value_raw, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=mla_past_kv,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Convert MLA KV output to standard 4D KV cache format
        # present_key_value_raw = (k_pe[batch, seq, rope_dim], compressed_kv[batch, seq, kv_rank])
        k_pe, compressed_kv = present_key_value_raw
        # Concatenate along last dim: (batch, seq_len, qk_rope_head_dim + kv_lora_rank)
        concat_kv = torch.cat([k_pe, compressed_kv], dim=-1)
        # Add head dimension: (batch, 1, seq_len, 320)
        concat_kv = concat_kv.unsqueeze(1)
        # Create dummy V cache with same shape
        dummy_v = torch.zeros_like(concat_kv)
        present_key_value = (concat_kv, dummy_v)

        # MoE FFN (with shared experts handled internally by moe_v2)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronDeepseekV3Model(NeuronBaseModel):
    """
    NeuronDeepseekV3Model - traceable model for NxD compilation.
    """

    def setup_attr_for_model(self, config: DeepseekV3MoEInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # MLA: KV cache stores 1 compressed "head" per position
        # The compressed KV dim = qk_rope_head_dim + kv_lora_rank
        self.num_key_value_heads = 1
        # Override head_dim for KV cache sizing (not used by attention module)
        config.head_dim = config.qk_rope_head_dim + config.kv_lora_rank
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: DeepseekV3MoEInferenceConfig):
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
                NeuronDeepseekV3DecoderLayer(config, layer_idx)
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


class NeuronDeepseekV3ForCausalLM(NeuronBaseForCausalLM):
    """
    NeuronDeepseekV3ForCausalLM - Entry point for DeepSeek-V3 / Mistral-Small-4
    inference on Neuron.
    """

    _model_cls = NeuronDeepseekV3Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return DeepseekV3MoEInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: DeepseekV3MoEInferenceConfig
    ) -> dict:
        return convert_deepseekv3_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += (
            " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        # DMA optimization from DeepSeek attention
        compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
        return compiler_args
