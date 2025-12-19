
# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
from types import SimpleNamespace
from typing import List, Optional, Tuple, Type

from neuronx_distributed_inference.models.llama4.utils.patch_llama4 import patch_llama4_text_moe_forward
from neuronx_distributed_inference.modules.kvcache.utils import get_layer_to_kv_cache_size_mapping_for_mixed_attn
import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn

from neuronx_distributed_inference.models.config import (  # noqa: E402
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.models.llama4.utils.layer_utils import is_before_nope_layer, is_after_nope_layer
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase, QKNormPlacement
)
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group

from neuronx_distributed_inference.modules.generation.sampling import Sampler
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

logger = logging.getLogger("Neuron")


class LlamaRMSNormV2(nn.Module):
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


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNormV2 if cpu_mode() else CustomRMSNorm


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
    Generate a list of configurations for each hidden layer in a Llama model.

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
        # If any of the MLP modules for this layer are in modules_to_not_convert
        module_pattern = f"layers.{i}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            non_quant_config = copy.deepcopy(config)
            non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
            non_quant_config.neuron_config.activation_quantization_type = None
            non_quant_config.neuron_config.quantize_clamp_bound = float("inf")
            updated_configs.append(non_quant_config)
        else:
            updated_configs.append(config)
    return updated_configs


def _helper_concat_and_delete_qkv(llama_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        llama_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    llama_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )

    del llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(llama_state_dict, l, "scale")

    gc.collect()

    return llama_state_dict


class L2Norm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dtype = torch.float32

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        required_attributes = [
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
            "num_local_experts",
            "num_experts_per_tok",
        ]
        return ["text_config." + atr for atr in required_attributes]

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

        assert (
            self.neuron_config.is_medusa is False and self.neuron_config.speculation_length == 0
        ), f"Speculative Decoding is not yet supported in this Model. \
                is_medusa was set to {self.neuron_config.is_medusa}. \
                speculation_length was set to {self.neuron_config.speculation_length}"

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # since kernels are not yet supported with chunked attention
        if isinstance(self.text_config, SimpleNamespace):
            self.text_config = vars(self.text_config)

        # We need to copy the model's _name_or_path to the text config
        self.text_config = InferenceConfig(self.neuron_config, **self.text_config)
        self.text_config._name_or_path = self._name_or_path

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        return json.dumps(config_dict, indent=2, sort_keys=True)


class NeuronLlama4MLP(NeuronLlamaMLP):
    def __init__(self, config: InferenceConfig):
        updated_config = copy.deepcopy(config)
        updated_config.intermediate_size = config.intermediate_size_mlp
        super().__init__(updated_config)


class NeuronLlama4Attention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(
        self,
        config: InferenceConfig,
        tensor_model_parallel_group=None,
        use_qk_norm=True,
        is_nope=False,
        is_post_global_attn_layer=False,
        is_pre_global_attn_layer=False,
    ):

        attn_chunk_size = None if is_nope else getattr(config, "attention_chunk_size", None)
        if attn_chunk_size and attn_chunk_size >= config.neuron_config.seq_len:
            attn_chunk_size = None
        optimize_interleave_attn = False
        if attn_chunk_size:
            optimize_interleave_attn = True

        # P314730317
        config.neuron_config.attn_block_tkg_nki_kernel_use_online_softmax = False
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            qk_norm_placement=QKNormPlacement.POST_ROPE,
            rope_theta=config.rope_theta,
            use_scaled_rope=getattr(config, "rope_scaling", None) is not None,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            tensor_model_parallel_group=tensor_model_parallel_group,
            attention_chunk_size=attn_chunk_size,
            is_post_global_attn_layer=is_post_global_attn_layer,
            is_pre_global_attn_layer=is_pre_global_attn_layer,
            optimize_interleave_attn=optimize_interleave_attn
        )
        # TODO: NeuronAttentionBase uses RMSNorm but LLama4 needs L2Norm
        self.qk_norm = L2Norm(config.rms_norm_eps) if use_qk_norm else None


class NeuronMoEDecoderLayer(nn.Module):
    def __init__(self, config: InferenceConfig, rmsnorm: Optional[nn.Module] = None):
        super().__init__()
        # The following list of configs are required for Llama4
        # HF config of Llama4 does not include `n_shared_experts`
        # We need to add this workaround to initialize Neuron's MOE module with shared experts
        config.n_shared_experts = 1
        config.neuron_config.router_config.dtype = torch.float32
        config.neuron_config.router_config.act_fn = "sigmoid"
        config.neuron_config.fused_shared_experts = False
        config.neuron_config.transpose_shared_experts_weights = True
        config.neuron_config.early_expert_affinity_modulation = True
        config.neuron_config.normalize_top_k_affinities = False

        self.moe = initialize_moe_module(config=config, rmsnorm=rmsnorm, init_tkg_module=True)

    def forward(self, hidden_states):
        """Forward pass for the MOE module"""
        hidden_states = self.moe(hidden_states)[0]
        return hidden_states


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.is_nope_layer = config.no_rope_layers[layer_idx] == 0

        # NoPE layers do full S attention, no QK norm, no RoPE
        # non-NoPE layers do a chunked attention, QK Norm, RoPE

        attention_config = copy.deepcopy(config)

        use_qk_norm = getattr(config, "use_qk_norm", False) and not self.is_nope_layer
        self.attention_chunk_size = getattr(config, "attention_chunk_size", None)
        if self.attention_chunk_size and self.attention_chunk_size >= config.neuron_config.seq_len or self.is_nope_layer:
            self.attention_chunk_size = None
        self.is_before_nope = False
        self.is_after_nope = False
        if self.attention_chunk_size:
            self.is_before_nope = is_before_nope_layer(config, layer_idx)
            self.is_after_nope = is_after_nope_layer(config, layer_idx)
        self.self_attn = NeuronLlama4Attention(
            config=attention_config,
            tensor_model_parallel_group=get_tp_group(config),
            use_qk_norm=use_qk_norm,
            is_nope=self.is_nope_layer,
            is_post_global_attn_layer=self.is_after_nope,
            is_pre_global_attn_layer=self.is_before_nope,
        )

        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        if (self.layer_idx + 1) % config.interleave_moe_layer_step == 0:
            self.feed_forward = NeuronMoEDecoderLayer(config, rmsnorm=self.post_attention_layernorm)
        else:
            self.feed_forward = NeuronLlama4MLP(config)

        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = config.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.is_prefill_stage = config.neuron_config.is_prefill_stage
        self.config = config

        if self.is_prefill_stage and self.config.neuron_config.is_mlp_quantized():
            # for CTE, quantized MLP kernel does not support fused rmsnorm
            self.mlp_kernel_fused_rmsnorm = False
        else:
            self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

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
        residual = hidden_states
        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm and (self.config.neuron_config.cp_degree == 1 or self.attention_chunk_size is None):
            hidden_states = self.input_layernorm(hidden_states)

        mask = local_mask
        if self.is_nope_layer or local_mask is None:
            mask = attention_mask

        # Self Attention
        attn_ouput_tuple = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            use_polar_compatible_rope=not self.is_nope_layer,
            **kwargs,
        )
        hidden_states, present_key_value, cos_cache, sin_cache = attn_ouput_tuple
        if self.is_prefill_stage and self.attention_chunk_size and self.config.neuron_config.cp_degree > 1:
            if self.is_before_nope:
                residual = torch.zeros_like(residual)  # residual add is already performed inside attn
            elif self.is_after_nope and attn_ouput_tuple.attn_input_hidden_states is not None:
                residual = attn_ouput_tuple.attn_input_hidden_states

        # MLP Flow (From NeuronLlamaDecoderLayer)
        if isinstance(self.feed_forward, NeuronLlama4MLP):
            if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
                assert (
                    not self.sequence_parallel_enabled
                ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
                # First residual add handled in the MLP kernel
                hidden_states, residual = self.feed_forward(
                    hidden_states,
                    rmsnorm=self.post_attention_layernorm,
                    residual=residual,
                    adapter_ids=adapter_ids,
                )
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                # RMSNorm (fused with QKV kernel when SP is disabled)
                if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                    rmsnorm = self.post_attention_layernorm
                else:
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    rmsnorm = None
                hidden_states, _ = self.feed_forward(
                    hidden_states,
                    rmsnorm=rmsnorm,
                    adapter_ids=adapter_ids,
                )

        # MoE Flow
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states

            hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

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


class NeuronLlama4TextModel(NeuronBaseModel):

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        # Concat vision and text embeddings during context encoding
        # Both inputs_embeds and vision_embeddings should be of the same shape: [BS, Total tokens (image + text), Hidden]
        # And vision_mask should of the shape [BS, Total tokens (image + text), 1]
        # Entries in vision_mask with value `True` represent vision tokens and with value `False` represent text tokens
        # For text-only inputs, vision_mask should be all `False`
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

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

            lm_head_pad = config.neuron_config.lm_head_pad
            lnc = config.neuron_config.logical_nc_config
            lm_head_pad_alignment_size = config.neuron_config.lm_head_pad_alignment_size * lnc
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=lm_head_pad,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size if lm_head_pad else 1,
                keep_padded_output=lm_head_pad,
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

        self.layers = nn.ModuleList(
            [NeuronLlamaDecoderLayer(conf, idx) for idx, conf in enumerate(updated_configs)]
        )

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length
        self.attention_chunk_size = getattr(config, "attention_chunk_size", None)
        if self.attention_chunk_size and config.neuron_config.seq_len <= self.attention_chunk_size:
            self.attention_chunk_size = None
        if self.attention_chunk_size:
            self.layer_to_cache_size_mapping = get_layer_to_kv_cache_size_mapping_for_mixed_attn(self.attention_chunk_size, self.config.neuron_config.seq_len, config.no_rope_layers)
        self.has_mixed_attn = True

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)

    def init_inference_optimization(self, config: InferenceConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)

        self.kv_mgr = KVCacheManager(config, num_kv_head=self.num_key_value_heads, global_rank=self.rank_util, attention_chunk_size=self.attention_chunk_size, layer_to_cache_size_mapping=self.layer_to_cache_size_mapping)


class NeuronLlama4TextForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlama4TextModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Llama4ForCausalLM

        model = Llama4ForCausalLM.from_pretrained(model_path)

        # Patch an accuracy issue that affects transformers v4.54-4.56.
        patch_llama4_text_moe_forward(model.model)

        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config

        state_dict["lm_head.weight"] = state_dict["language_model.lm_head.weight"]
        del state_dict["language_model.lm_head.weight"]

        if "language_model.lm_head.bias" not in state_dict and config.neuron_config.lm_head_pad:
            state_dict["lm_head.bias"] = torch.zeros(state_dict["lm_head.weight"].shape[0], dtype=torch.float32)

        dict_keys = list(state_dict.keys())
        for key in dict_keys:
            if key.startswith("language_model.model."):
                new_key = key.replace("language_model.model.", "")
                state_dict[new_key] = state_dict.pop(key)

        key_map = {
            "self_attn.qkv_proj.weight": "self_attn.Wqkv.weight",
            # router
            "feed_forward.router.weight": "feed_forward.moe.router.linear_router.weight",
            # experts
            "feed_forward.experts.gate_up_proj": "feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight",
            "feed_forward.experts.down_proj": "feed_forward.moe.expert_mlps.mlp_op.down_proj.weight",
            # shared experts
            "feed_forward.shared_expert.down_proj.weight": "feed_forward.moe.shared_experts.down_proj.weight",
            # scales
            "feed_forward.experts.gate_up_proj.scale": "feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.scale",
            "feed_forward.experts.down_proj.scale": "feed_forward.moe.expert_mlps.mlp_op.down_proj.scale",
        }

        if not config.neuron_config.fused_shared_experts:
            # if not fusing shared experts gate/up weights, we can directly rename the keys
            key_map["feed_forward.shared_expert.gate_proj.weight"] = "feed_forward.moe.shared_experts.gate_proj.weight"
            key_map["feed_forward.shared_expert.up_proj.weight"] = "feed_forward.moe.shared_experts.up_proj.weight"

        moe_intermediate_size = config.intermediate_size
        state_dict_keys = set(state_dict.keys())
        num_experts = config.num_local_experts
        for layer_n in range(config.num_hidden_layers):
            prefix = f"layers.{layer_n}."
            if prefix + "feed_forward.shared_expert.up_proj.weight" in state_dict_keys:
                exp_down_key = prefix + "feed_forward.experts.down_proj"
                state_dict[exp_down_key] = state_dict[exp_down_key].view(
                    num_experts, moe_intermediate_size, config.hidden_size
                )

                if config.neuron_config.fused_shared_experts:
                    shared_new_key = prefix + "feed_forward.moe.shared_experts.gate_up_proj.weight"
                    shared_swig_key = prefix + "feed_forward.shared_expert.up_proj.weight"
                    shared_in_key = prefix + "feed_forward.shared_expert.gate_proj.weight"

                    state_dict[shared_new_key] = torch.cat(
                        [state_dict[shared_in_key], state_dict[shared_swig_key]], dim=0
                    )
                    state_dict_keys.add(shared_new_key)

                    del state_dict[shared_swig_key]
                    del state_dict[shared_in_key]

            for old_key, new_key in key_map.items():
                if prefix + old_key in state_dict_keys:
                    state_dict[prefix + new_key] = state_dict[prefix + old_key]
                    del state_dict[prefix + old_key]

            gc.collect()

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
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
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
