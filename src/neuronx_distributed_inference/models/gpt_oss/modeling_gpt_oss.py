# Copyright (c) 2025, OpenAI. All rights reserved.
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

import copy
import gc
import json
import logging
import math
import os
import re
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_neuronx.xla_impl.ops import RmsNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.gpt_oss.mx_layout_transform import convert_hf_format_state_dict_mxfp4_compute, shuffle_hidden_dim, unshuffle_hidden_dim
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.generation.sampling import Sampler
from neuronx_distributed_inference.modules.kvcache.gpt_oss_kv_cache_manager import GptOssKVCacheManager
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

from transformers import GptOssForCausalLM


logger = logging.getLogger("Neuron")


def convert_gate_up_proj(tensor: torch.Tensor, is_bias: bool = False) -> torch.Tensor:
    """
    Convert the gate_up_proj tensor from GptOss reference format to NxDI format.

    Reference format: E, 2xI, H with interleaved gate and up projection
    NxDI format: E, H, 2xI with chunked gate and up project

    Args:
        tensor (torch.Tensor): the parameter to convert
        is_bias (bool): flag indicating if parameter is bias

    Returns:
        torch.Tensor: in format needed for NxDI MoE modules
    """
    gate, up_proj = tensor[:, ::2, ...], tensor[:, 1::2, ...]
    gate_up_proj = torch.cat((gate, up_proj), dim=1)
    return gate_up_proj if is_bias else gate_up_proj.transpose(1, 2)


def get_lm_head_pad_config(vocab_size: int, tp_degree: int, lm_head_pad_alignment_size: int = 1, skip_lm_head_pad: bool = False):
    """
    Check if lm_head padding is necessary to achieve good performance.

    Args:
        vocab_size (int): vocabulary size of the model
        tp_degree (int): tp_degree used for lm_head
        lm_head_pad_alignment_size (int): usually you want to set this to LNC degree
        skip_lm_head_pad (bool): always disable padding (for debug purpose)

    Returns:
        (bool, int): Tuple indiciating if we should pad and what the pad_alignment_size should be.
    """
    if vocab_size % (tp_degree * lm_head_pad_alignment_size) == 0 or skip_lm_head_pad:
        return False, 1

    return True, lm_head_pad_alignment_size


# Copied from GPT_OSS repo
# TODO: Add absolute link when the repo is public

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape} does not match {scales.shape}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    # TODO: Delete after making sure this is not necessary! since we go back to cpu in the end in create_quantized_param using .to(target_device)
    # Move back to CPU if needed
    # if need_to_move_back:
    #     out = out.cpu()
    del blocks, scales, lut
    return out


def convert_hf_format_state_dict_bf16_compute(state_dict: dict, config: InferenceConfig):

    neuron_config = config.neuron_config
    assert not neuron_config.is_mxfp4_compute, "mxfp4 compute must be disabled for convert_hf_format_state_dict_bf16_compute to be applicable"
    num_layers = config.num_hidden_layers

    for layer in range(num_layers):
        # add unpadded weight for layer norm
        state_dict[f"layers.{layer}.input_layernorm.weight_unpadded"] = state_dict[f"layers.{layer}.input_layernorm.weight"][:config.original_hidden_size].clone().contiguous()
        state_dict[f"layers.{layer}.post_attention_layernorm.weight_unpadded"] = state_dict[f"layers.{layer}.post_attention_layernorm.weight"][:config.original_hidden_size].clone().contiguous()

        # Sinks
        state_dict[f"layers.{layer}.self_attn.learned_sinks.sink"] = state_dict[f"layers.{layer}.self_attn.sinks"]
        # If attention is run in two different parallelisms across CTE and TKG, we duplicate the weights
        if config.neuron_config.attention_dp_degree != config.neuron_config.cp_degree:
            state_dict[f"layers.{layer}.self_attn.tkg_learned_sinks.sink"] = state_dict[f"layers.{layer}.self_attn.sinks"]
        del state_dict[f"layers.{layer}.self_attn.sinks"]

        # Router
        state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.weight"] = state_dict[f"layers.{layer}.mlp.router.weight"]
        del state_dict[f"layers.{layer}.mlp.router.weight"]

        state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.bias"] = state_dict[f"layers.{layer}.mlp.router.bias"]
        del state_dict[f"layers.{layer}.mlp.router.bias"]

        # Down Projection
        for proj in ["down_proj", "gate_up_proj"]:
            # TODO: check the dimension against the neuron implementation
            # convert FP4 back to BF16, the dimension will x2. Revisit when we support FP4 on neuron
            dequantized_weights = convert_moe_packed_tensors(
                state_dict[f"layers.{layer}.mlp.experts.{proj}_blocks"],
                state_dict[f"layers.{layer}.mlp.experts.{proj}_scales"]
            )

            if proj == "gate_up_proj":
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.weight"] = convert_gate_up_proj(dequantized_weights)
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.bias"] = convert_gate_up_proj(state_dict[f"layers.{layer}.mlp.experts.{proj}_bias"], is_bias=True)
            else:
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.weight"] = dequantized_weights.transpose(1, 2)
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.bias"] = state_dict[f"layers.{layer}.mlp.experts.{proj}_bias"]

            del state_dict[f"layers.{layer}.mlp.experts.{proj}_blocks"]
            del state_dict[f"layers.{layer}.mlp.experts.{proj}_scales"]
            del state_dict[f"layers.{layer}.mlp.experts.{proj}_bias"]
    return state_dict


class GptOssRMSNormV2(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Critical difference with LlamaRMSNorm: We multiply in full precision and then convert
        # to the target data type instead of converting hidden_states to the target data type and
        # then multiplying in full precision.
        output = self.weight * hidden_states
        return output.to(input_dtype)


class GptOssRMSNormV2Padded(nn.Module):
    def __init__(self, padded_hidden_size, unpadded_hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(padded_hidden_size))
        self.unpadded_hidden_size = unpadded_hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states[:, :, :self.unpadded_hidden_size].pow(2).mean(-1, keepdim=True)
        hidden_states[:, :, :self.unpadded_hidden_size] = hidden_states[:, :, :self.unpadded_hidden_size] * torch.rsqrt(variance + self.variance_epsilon)
        # Critical difference with LlamaRMSNorm: We multiply in full precision and then convert
        # to the target data type instead of converting hidden_states to the target data type and
        # then multiplying in full precision.
        output = self.weight * hidden_states
        output[:, :, self.unpadded_hidden_size:] = 0.0
        return output.to(input_dtype)


# When we have padded+shuffled hidden states, we cannot use unpadded+shuffled norm weights.
# Instead, we use padded+shuffled hidden states and padded+shuffled norm weights and adjust for
# padding inside the RMSNorm implementation.
class GptOssRMSNormV3PaddedShuffled(nn.Module):
    def __init__(self, padded_hidden_size, unpadded_hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(padded_hidden_size))
        self.unpadded_hidden_size = unpadded_hidden_size
        self.padded_hidden_size = padded_hidden_size
        self.variance_epsilon = eps
        self.norm_pad_scaling_factor = torch.tensor([padded_hidden_size / unpadded_hidden_size], dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # Important: adjust mean(hidden_states_padded^2) output by multiplying by (H_padded / H_actual). This
        # produces 1/H_actual*sum(hidden_states_padded^2) which is equivalent to mean(hidden_states_unpadded^2).
        variance = hidden_states.pow(2).mean(-1, keepdim=True) * self.norm_pad_scaling_factor
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Critical difference with LlamaRMSNorm: We multiply in full precision and then convert
        # to the target data type instead of converting hidden_states to the target data type and
        # then multiplying in full precision.
        output = self.weight * hidden_states
        # Ensure that padded regions of shuffled hidden states are zero
        output_shape = output.shape
        output = output.view(*output_shape[:-1], 4, self.padded_hidden_size // 4)
        output[..., self.unpadded_hidden_size // 4:] = 0.0
        output = output.view(output_shape)
        return output.to(input_dtype)


# RMSNorm produces incorrect results when operating on a padded weights.
# This implementation uses padded and unpadded weight.
# Padded weight is used during the kernel flow and unpadded weight is used during the flat compiler flow.
# The below implementation is for flat compiler flow. The output from RMSNorm is padded again before returning the result.
class CustomRMSNormV2Padded(CustomRMSNorm):
    def __init__(self, hidden_size, hidden_size_actual, eps):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.hidden_size_actual = hidden_size_actual
        self.pad = hidden_size - hidden_size_actual
        self.weight_unpadded = nn.Parameter(torch.ones(hidden_size_actual))

    def forward(self, hidden_states: torch.Tensor):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states[..., :self.hidden_size_actual]
        hidden_states = hidden_states.to(torch.float32)
        result = RmsNorm.apply(
            hidden_states, self.weight_unpadded, self.variance_epsilon, len(hidden_states.shape) - 1
        )
        output = torch.nn.functional.pad(result, (0, self.pad))
        return output.to(original_dtype)


def get_rmsnorm_cls(padded_hidden_size, unpadded_hidden_size, eps, is_hidden_dim_shuffled=False):
    # Initialize to the appropriate implementation of RMSNorm
    # If H dim is shuffled -> GptOssRMSNormV3PaddedShuffled
    # If infer on NXD AND H dim is not shuffled -> CustomRMSNormV2Padded
    # If infer on CPU -> GptOssRMSNormV2Padded (CustomRMSNorm does not work on CPU)
    if is_hidden_dim_shuffled:
        return GptOssRMSNormV3PaddedShuffled(padded_hidden_size, unpadded_hidden_size, eps)
    elif cpu_mode():
        return GptOssRMSNormV2Padded(padded_hidden_size, unpadded_hidden_size, eps)
    else:
        return CustomRMSNormV2Padded(padded_hidden_size, unpadded_hidden_size, eps)


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def get_modules_to_not_convert(neuron_config: NeuronConfig):
    """
    Returnes the modules_to_not_convert from the neuron configs

    Args:
    config (NeuronConfig): The neuron configuration for the model.

    Returns:
    lis[str]: A list of modules that should be skipped during per-layer configuration update

    """
    return getattr(neuron_config, "modules_to_not_convert", None)


def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a GptOss model.

    This function creates a list of InferenceConfig objects, one for each layer. It
    modifies the configurations for certain layers based on which modules should not
    be converted to quantized format. The function uses get_modules_to_not_convert()
    to determine which modules should not be converted.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version
                           with "quantized_mlp_kernel_enabled" as False for that specific layer.
    """
    updated_configs = []
    modules_to_not_convert = get_modules_to_not_convert(config.neuron_config)
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for i in range(config.num_hidden_layers):
        updated_config = copy.deepcopy(config)

        swa_layer = i % 2 == 0

        if swa_layer:
            updated_config.neuron_config.attention_dp_degree = config.neuron_config.sliding_window_attention_dp_degree

        if not swa_layer:
            updated_config.sliding_window = None

        # If any of the MLP modules for this layer are in modules_to_not_convert
        module_pattern = f"layers.{i}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            updated_config.neuron_config.quantized_mlp_kernel_enabled = False
            updated_config.neuron_config.activation_quantization_type = None
            updated_config.neuron_config.quantize_clamp_bound = float("inf")

        updated_configs.append(updated_config)

    return updated_configs


def _helper_concat_and_delete_qkv(GptOss_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        GptOss_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    GptOss_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            GptOss_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            GptOss_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            GptOss_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )

    del GptOss_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del GptOss_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del GptOss_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(GptOss_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(GptOss_state_dict, l, "weight")
        _helper_concat_and_delete_qkv(GptOss_state_dict, l, "bias")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(GptOss_state_dict, l, "scale")

    gc.collect()

    return GptOss_state_dict


class L2Norm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dtype = torch.float32

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


class GptOssNeuronConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sliding_window_attention_dp_degree = kwargs.pop("sliding_window_attention_dp_degree", 1)

        self.validate_attention_data_parallel(self.sliding_window_attention_dp_degree)


class GptOssInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

        if not hasattr(self, "num_local_experts") and hasattr(self, "num_experts"):
            self.num_local_experts = getattr(self, "num_experts")

        if not hasattr(self, "num_experts_per_tok") and hasattr(self, "experts_per_token"):
            self.num_experts_per_tok = getattr(self, "experts_per_token")

        if not hasattr(self, "rope_scaling_factor") and hasattr(self, "rope_scaling"):
            self.rope_scaling_factor = self.rope_scaling.get("factor")

        if not hasattr(self, "rope_ntk_alpha") and hasattr(self, "rope_scaling"):
            self.rope_ntk_alpha = self.rope_scaling.get("beta_slow")

        if not hasattr(self, "rope_ntk_beta") and hasattr(self, "rope_scaling"):
            self.rope_ntk_beta = self.rope_scaling.get("beta_fast")

    def get_required_attributes(self) -> List[str]:
        return [
            "num_hidden_layers",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "sliding_window",
            "initial_context_length",
            "rope_theta",
            "rope_scaling_factor",
            "rope_ntk_alpha",
            "rope_ntk_beta",
            "pad_token_id",
        ]

    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """

        def hasattr_nested(obj, attr_chain):
            attrs = attr_chain.split(".")
            for attr in attrs:
                if isinstance(obj, dict):
                    if attr not in obj:
                        return False
                    obj = obj[attr]
                else:
                    if not hasattr(obj, attr):
                        return False
                    obj = getattr(obj, attr)
            return True

        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr_nested(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

        if self.neuron_config.padded_hidden_size is not None:
            assert self.neuron_config.padded_hidden_size >= self.hidden_size, "Cannot reduce hidden size"
        if self.neuron_config.padded_intermediate_size is not None:
            assert self.neuron_config.padded_intermediate_size >= self.intermediate_size, f"Cannot reduce intermediate size, {self.intermediate_size}, {self.neuron_config.padded_intermediate_size}"

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return GptOssNeuronConfig

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rms_norm_eps = 1e-05
        self.hidden_act = "sigmoid"
        self.n_shared_experts = 0
        if not hasattr(self, 'original_hidden_size'):
            self.original_hidden_size = self.hidden_size
        if not hasattr(self, 'original_intermediate_size'):
            self.original_intermediate_size = self.intermediate_size
        self.hidden_size = (
            self.neuron_config.padded_hidden_size
            if self.neuron_config.padded_hidden_size is not None
            else self.hidden_size
        )

        self.intermediate_size = (
            self.neuron_config.padded_intermediate_size
            if self.neuron_config.padded_intermediate_size is not None
            else self.intermediate_size
        )


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 initial_context_length: int = 4096,
                 scaling_factor: float = 1.0,
                 ntk_alpha: float = 1.0,
                 ntk_beta: float = 32.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.register_buffer("inv_freq", None, persistent=False)
        self.concentration = None

    def get_inv_freqs_and_concentration(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
            / self.dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return inv_freq, concentration

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None or self.concentration is None:
            self.inv_freq, self.concentration = self.get_inv_freqs_and_concentration(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.concentration
        sin = emb.sin() * self.concentration
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronGptOssAttention(NeuronAttentionBase):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self.get_rope(config=config),
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_scaled_rope=None,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            learned_sinks_size=1,
            sliding_window=config.sliding_window
        )

    def get_rope(self, config: GptOssInferenceConfig):
        rotary_emb = GptOssRotaryEmbedding(dim=config.head_dim,
                                           base=config.rope_theta,
                                           initial_context_length=config.initial_context_length,
                                           scaling_factor=config.rope_scaling_factor,
                                           ntk_alpha=config.rope_ntk_alpha,
                                           ntk_beta=config.rope_ntk_beta)
        return rotary_emb


class NeuronGptOssMoE(nn.Module):
    def __init__(self, config: InferenceConfig, rmsnorm: Optional[nn.Module] = None):
        super().__init__()

        # TODO: Handle architecture related configuration separately from NeuronConfig
        config.neuron_config.router_config.dtype = torch.float32
        config.neuron_config.router_config.act_fn = "softmax"
        config.neuron_config.transpose_shared_experts_weights = False
        config.neuron_config.early_expert_affinity_modulation = False
        config.neuron_config.normalize_top_k_affinities = False
        config.neuron_config.glu_type = "swiglu"
        config.neuron_config.hidden_act_scaling_factor = 1.702
        config.neuron_config.hidden_act_bias = 1
        config.neuron_config.gate_clamp_upper_limit = 7.0
        config.neuron_config.gate_clamp_lower_limit = None
        config.neuron_config.up_clamp_upper_limit = 7.0
        config.neuron_config.up_clamp_lower_limit = -7.0

        self.moe = initialize_moe_module(config=config,
                                         rmsnorm=rmsnorm,
                                         init_tkg_module=not config.neuron_config.on_cpu,
                                         router_bias=True,
                                         experts_bias=True,
                                         apply_act_fn_over_topk=True)

    def forward(self, hidden_states, is_speculative_decoding=False):
        """Forward pass for the MOE module"""
        # return router_logit and expert_index for testing
        result = self.moe(hidden_states, is_speculative_decoding=is_speculative_decoding)
        hidden_states = result[0]
        router_logits = result[1] if self.moe.return_router_logits else None
        expert_index = result[-1] if self.moe.return_expert_index else None

        return tuple(x for x in (hidden_states, router_logits, expert_index) if x is not None)


class DecodeGptOssRotaryCacheManager():
    def __init__(self):
        self.global_sin_cache = None
        self.global_cos_cache = None
        self.window_sin_cache = None
        self.window_cos_cache = None

    def clear_cache(self):
        self.global_sin_cache = None
        self.global_cos_cache = None
        self.window_sin_cache = None
        self.window_cos_cache = None

    def set_cache(self, layer_index, sin_cache, cos_cache):
        if layer_index % 2 == 0:
            self.window_sin_cache = sin_cache
            self.window_cos_cache = cos_cache
        else:
            self.global_sin_cache = sin_cache
            self.global_cos_cache = cos_cache

    def get_cache(self, layer_index):
        if layer_index % 2 == 0:
            return self.window_cos_cache, self.window_sin_cache

        return self.global_cos_cache, self.global_sin_cache


class NeuronGptOssDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig, layer_idx: int, rotary_cache_manager: DecodeGptOssRotaryCacheManager):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rotary_cache_manager = rotary_cache_manager

        self.is_sliding_window_layer = layer_idx % 2 == 0

        self.self_attn = NeuronGptOssAttention(
            config=config,
        )

        self.post_attention_layernorm = get_rmsnorm_cls(
            padded_hidden_size=config.hidden_size,
            unpadded_hidden_size=config.original_hidden_size,
            eps=config.rms_norm_eps,
            # When using MXFP4 compute, MoE block input always has hidden dim shuffled
            is_hidden_dim_shuffled=config.neuron_config.is_mxfp4_compute,
        )
        self.feed_forward = NeuronGptOssMoE(config, rmsnorm=self.post_attention_layernorm)

        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls(
                padded_hidden_size=config.hidden_size,
                unpadded_hidden_size=config.original_hidden_size,
                eps=config.rms_norm_eps,
                # When using MXFP4 compute, attention block input has hidden dim shuffled if the full model is shuffled
                # NOTE: once e2e accuracy with MXFP4 compute is shuffled, switch this to use is_mxfp4_compute flag
                is_hidden_dim_shuffled=config.neuron_config.is_full_model_shuffled,
            )

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.is_prefill_stage = config.neuron_config.is_prefill_stage
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.config.neuron_config.is_prefill_stage:
            cos_cache, sin_cache = kwargs.pop("cos_cache"), kwargs.pop("sin_cache")
        else:
            cos_cache, sin_cache = self.rotary_cache_manager.get_cache(self.layer_idx)

            # The caches passed in via kwargs are ignored during decode
            kwargs.pop("cos_cache")
            kwargs.pop("sin_cache")

        residual = hidden_states.clone()

        # Shuffle H dim of residual when using mxfp4 weights, to ensure that residual add result is correct and shuffled. This is required
        #   because attention block input is unshuffled but attention block output is shuffled.
        # FIXME: remove hidden states shuffle/unshuffle logic once we have validated full model accuracy with mxfp4. Once we have
        #   validated full model accuracy, we will shuffle the H dim for the full model to avoid online shuffling/unshuffling hidden_states.
        if self.config.neuron_config.is_mxfp4_compute and not self.config.neuron_config.is_full_model_shuffled:
            residual = shuffle_hidden_dim(residual, dim=-1)

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        mask = local_mask
        if not self.is_sliding_window_layer or local_mask is None:
            mask = attention_mask

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states.clone()

        # Unshuffle H dim of residual when using mxfp4 weights, to ensure that residual add result is correct and unshuffled. This is required
        #   because MoE block input is shuffled but MoE block output is unshuffled.
        # FIXME: remove hidden states shuffle/unshuffle logic once we have validated full model accuracy with mxfp4. Once we have
        #   validated full model accuracy, we will shuffle the H dim for the full model to avoid online shuffling/unshuffling hidden_states.
        if self.config.neuron_config.is_mxfp4_compute and not self.config.neuron_config.is_full_model_shuffled:
            residual = unshuffle_hidden_dim(residual, dim=-1)

        is_speculative_decoding = self.config.neuron_config.enable_fused_speculation and (not self.config.neuron_config.is_prefill_stage)
        hidden_states = self.feed_forward(hidden_states, is_speculative_decoding)[0]
        hidden_states = residual + hidden_states

        if not self.config.neuron_config.is_prefill_stage:
            self.rotary_cache_manager.set_cache(self.layer_idx, sin_cache, cos_cache)

        # clear the cache at the end of decode
        if self.config.num_hidden_layers - 1 == self.layer_idx and not self.config.neuron_config.is_prefill_stage:
            self.rotary_cache_manager.clear_cache()

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronGptOssModel(NeuronBaseModel):

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.sliding_window = config.sliding_window

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            # Note: We pad LM Head to ensure proper sharding across all LNCs
            should_pad_lm_head, lm_head_pad_alignment_size = get_lm_head_pad_config(
                vocab_size=config.vocab_size,
                tp_degree=config.neuron_config.tp_degree,
                lm_head_pad_alignment_size=config.neuron_config.lm_head_pad_alignment_size * config.neuron_config.logical_nc_config,
                skip_lm_head_pad=not config.neuron_config.lm_head_pad)

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                # Note: bias is padded with dtype.min to mask out padding during sampling
                # because padding weight won't work since softmax used during sampling
                # will assign non-zero probability to zero values.
                bias=should_pad_lm_head,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size,
                keep_padded_output=should_pad_lm_head,
                dtype=config.neuron_config.torch_dtype,
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

        updated_configs = get_updated_configs(config)

        decode_rotary_cache_manager = DecodeGptOssRotaryCacheManager()

        self.layers = nn.ModuleList(
            [NeuronGptOssDecoderLayer(conf, idx, decode_rotary_cache_manager) for idx, conf in enumerate(updated_configs)]
        )

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls(
                padded_hidden_size=config.hidden_size,
                unpadded_hidden_size=config.original_hidden_size,
                eps=config.rms_norm_eps,
                # When using MXFP4 compute, attention block input has hidden dim shuffled if the full model is shuffled
                # NOTE: once e2e accuracy with MXFP4 compute is shuffled, switch this to use is_mxfp4_compute flag
                is_hidden_dim_shuffled=config.neuron_config.is_full_model_shuffled,
            )

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )

        self.has_mixed_attn = True

    def init_inference_optimization(self, config: InferenceConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        self.kv_mgr = GptOssKVCacheManager(config, num_kv_head=self.num_key_value_heads, global_rank=self.rank_util, sliding_window=self.sliding_window)


class NeuronGptOssForCausalLM(NeuronBaseForCausalLM):

    _model_cls = NeuronGptOssModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        checkpoint_format = os.environ.get("CHECKPOINT_FORMAT", "hf").lower()
        assert checkpoint_format == "hf", "Must use HF checkpoint to load HF model"

        return GptOssForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Choose the correct conversion function based on checkpoint format environment variable.
        """
        # Default to hf format
        checkpoint_format = os.environ.get("CHECKPOINT_FORMAT", "hf").lower()

        if checkpoint_format == "hf":
            NeuronGptOssForCausalLM._pad_hf_state_dict(state_dict, config)
            return NeuronGptOssForCausalLM._convert_hf_format_state_dict(state_dict, config)
        else:
            return NeuronGptOssForCausalLM._convert_neuron_format_state_dict(state_dict, config)

    @staticmethod
    def _convert_hf_format_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        Convert HuggingFace format state dict to Neuron format.
        """
        neuron_config = config.neuron_config

        num_layers = config.num_hidden_layers

        if neuron_config.is_mxfp4_compute:
            convert_hf_format_state_dict_mxfp4_compute(state_dict, config)
        else:
            convert_hf_format_state_dict_bf16_compute(state_dict, config)

        state_dict["norm.weight_unpadded"] = state_dict["norm.weight"][:config.original_hidden_size].clone().contiguous()

        should_pad_lm_head, _ = get_lm_head_pad_config(
            vocab_size=config.vocab_size,
            tp_degree=config.neuron_config.tp_degree,
            lm_head_pad_alignment_size=config.neuron_config.lm_head_pad_alignment_size * config.neuron_config.logical_nc_config,
            skip_lm_head_pad=not config.neuron_config.lm_head_pad)
        if should_pad_lm_head:
            state_dict["lm_head.bias"] = torch.zeros(state_dict["lm_head.weight"].shape[0], dtype=torch.float32)

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def _pad_hf_state_dict(state_dict: dict, config: InferenceConfig) -> None:
        """
        Pad hidden and intermediate sizes for better compatibility with Neuron hardware.
        """
        diff_H = config.hidden_size - config.original_hidden_size
        diff_I = config.intermediate_size - config.original_intermediate_size

        pad_map = {
            "lm_head.weight": (0, diff_H),
            "embed_tokens.weight": (0, diff_H),
            "norm.weight": (0, diff_H),

            "layers.*.mlp.experts.down_proj_bias": (0, diff_H),
            "layers.*.mlp.experts.down_proj_blocks": (0, 0, 0, diff_I // (16 * 2), 0, diff_H),
            "layers.*.mlp.experts.down_proj_scales": (0, diff_I // (16 * 2), 0, diff_H),
            "layers.*.mlp.experts.gate_up_proj_bias": (0, diff_I * 2),
            "layers.*.mlp.experts.gate_up_proj_blocks": (0, 0, 0, diff_H // (16 * 2), 0, diff_I * 2),
            "layers.*.mlp.experts.gate_up_proj_scales": (0, diff_H // (16 * 2), 0, diff_I * 2),
            "layers.*.mlp.router.weight": (0, diff_H),

            "layers.*.self_attn.k_proj.weight": (0, diff_H),
            "layers.*.self_attn.o_proj.bias": (0, diff_H),
            "layers.*.self_attn.o_proj.weight": (0, 0, 0, diff_H),
            "layers.*.self_attn.q_proj.weight": (0, diff_H),
            "layers.*.self_attn.v_proj.weight": (0, diff_H),
            "layers.*.post_attention_layernorm.weight": (0, diff_H),
            "layers.*.input_layernorm.weight": (0, diff_H),
        }

        for key in state_dict.keys():
            pad_map_key = re.sub("layers\\.\\d+", "layers.*", key)
            if pad_map_key in pad_map:
                shape_before = state_dict[key].shape
                state_dict[key] = F.pad(state_dict[key], pad_map[pad_map_key])
                logger.debug(f"Padded {key} from {shape_before} to {state_dict[key].shape}")

    @staticmethod
    def _convert_neuron_format_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config

        num_layers = config.num_hidden_layers
        num_attn_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim

        # Embedding
        state_dict["embed_tokens.weight"] = state_dict["embedding.weight"]
        del state_dict["embedding.weight"]

        q_hidden_size = num_attn_heads * head_dim
        kv_hidden_size = num_key_value_heads * head_dim

        for layer in range(num_layers):
            # Input Layer Norm
            state_dict[f"layers.{layer}.input_layernorm.weight"] = state_dict[f"block.{layer}.attn.norm.scale"]
            state_dict[f"layers.{layer}.input_layernorm.weight_unpadded"] = state_dict[f"block.{layer}.attn.norm.scale"][:config.original_hidden_size].clone()
            del state_dict[f"block.{layer}.attn.norm.scale"]

            # QKV Projection
            qkv_weight = state_dict[f"block.{layer}.attn.qkv.weight"]
            q, k, v = torch.split(qkv_weight, [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=0)
            state_dict[f"layers.{layer}.self_attn.q_proj.weight"] = q
            state_dict[f"layers.{layer}.self_attn.k_proj.weight"] = k
            state_dict[f"layers.{layer}.self_attn.v_proj.weight"] = v
            del state_dict[f"block.{layer}.attn.qkv.weight"]

            # QKV Bias
            qkv_bias = state_dict[f"block.{layer}.attn.qkv.bias"]
            qb, kb, vb = torch.split(qkv_bias, [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=0)
            state_dict[f"layers.{layer}.self_attn.q_proj.bias"] = qb
            state_dict[f"layers.{layer}.self_attn.k_proj.bias"] = kb
            state_dict[f"layers.{layer}.self_attn.v_proj.bias"] = vb
            del state_dict[f"block.{layer}.attn.qkv.bias"]

            # Sinks
            state_dict[f"layers.{layer}.self_attn.learned_sinks.sink"] = state_dict[f"block.{layer}.attn.sinks"]
            # If attention is run in two different parallelisms across CTE and TKG, we duplicate the weights
            if config.neuron_config.attention_dp_degree != config.neuron_config.cp_degree:
                state_dict[f"layers.{layer}.self_attn.tkg_learned_sinks.sink"] = state_dict[f"block.{layer}.attn.sinks"]
            del state_dict[f"block.{layer}.attn.sinks"]

            # O Projection
            state_dict[f"layers.{layer}.self_attn.o_proj.weight"] = state_dict[f"block.{layer}.attn.out.weight"]
            del state_dict[f"block.{layer}.attn.out.weight"]

            # O Bias
            state_dict[f"layers.{layer}.self_attn.o_proj.bias"] = state_dict[f"block.{layer}.attn.out.bias"]
            del state_dict[f"block.{layer}.attn.out.bias"]

            # Router
            state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.weight"] = state_dict[f"block.{layer}.mlp.gate.weight"]
            del state_dict[f"block.{layer}.mlp.gate.weight"]

            state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.bias"] = state_dict[f"block.{layer}.mlp.gate.bias"]
            del state_dict[f"block.{layer}.mlp.gate.bias"]

            # Post Attention Layer Norm
            state_dict[f"layers.{layer}.post_attention_layernorm.weight"] = state_dict[f"block.{layer}.mlp.norm.scale"]
            del state_dict[f"block.{layer}.mlp.norm.scale"]

            # Down Projection
            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.down_proj.weight"] = state_dict[f"block.{layer}.mlp.mlp2_weight"].transpose(1, 2)
            del state_dict[f"block.{layer}.mlp.mlp2_weight"]

            if f"block.{layer}.mlp.mlp2_scale" in state_dict:
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.down_proj.scale"] = state_dict[f"block.{layer}.mlp.mlp2_scale"].transpose(1, 2)

            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.down_proj.bias"] = state_dict[f"block.{layer}.mlp.mlp2_bias"]
            del state_dict[f"block.{layer}.mlp.mlp2_bias"]

            # Gate Up Projection
            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight"] = convert_gate_up_proj(state_dict[f"block.{layer}.mlp.mlp1_weight"])
            del state_dict[f"block.{layer}.mlp.mlp1_weight"]

            if f"block.{layer}.mlp.mlp1_scale" in state_dict:
                state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.scale"] = convert_gate_up_proj(state_dict[f"block.{layer}.mlp.mlp1_scale"])

            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.bias"] = convert_gate_up_proj(state_dict[f"block.{layer}.mlp.mlp1_bias"], is_bias=True)
            del state_dict[f"block.{layer}.mlp.mlp1_bias"]

        # LM Head Weight
        state_dict["lm_head.weight"] = state_dict["unembedding.weight"]
        del state_dict["unembedding.weight"]

        should_pad_lm_head, _ = get_lm_head_pad_config(
            vocab_size=config.vocab_size,
            tp_degree=config.neuron_config.tp_degree,
            lm_head_pad_alignment_size=config.neuron_config.lm_head_pad_alignment_size * config.neuron_config.logical_nc_config,
            skip_lm_head_pad=not config.neuron_config.lm_head_pad)
        if should_pad_lm_head:
            state_dict["lm_head.bias"] = torch.zeros(state_dict["lm_head.weight"].shape[0], dtype=torch.float32)

        # Final Norm Weight
        state_dict["norm.weight"] = state_dict["norm.scale"]
        del state_dict["norm.scale"]

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @classmethod
    def get_config_cls(cls):
        return GptOssInferenceConfig
