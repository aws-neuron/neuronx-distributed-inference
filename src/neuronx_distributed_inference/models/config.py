import inspect
import json
import logging
import importlib
import os
import warnings
from typing import Dict, List, Type, Union, Optional

import torch
from neuronx_distributed.quantization.quantization_config import (
    ActivationQuantizationType,
    QuantizedDtype,
)
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig, RouterConfig
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed.utils.tensor_replacement.registry import RuntimeRegister
from neuronx_distributed_inference.utils.tensor_replacement.registry import TensorReplacementRegister

CONFIG_FILE = "neuron_config.json"
CHUNKED_ATTENTION_SUPPORTED_NEURON_CONFIG = {
    "flash_decoding_enabled": False,
    "attn_tkg_builtin_kernel_enabled": False,
    "attn_tkg_nki_kernel_enabled": False,
    "is_prefix_caching": False,
    "is_chunked_prefill": False,
    "padding_side": "right",
    "attention_dp_degree": 1,
}

# scratchpad page size since we get large tensors (used to set compiler and runtime)
# set to 1024 based on current model configurations and CC tiling factor
# this is stored in neuron_config since it needs to match between compile and runtime
LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE = 1024

# shard-on-I dimension per tp, intermediate size needs to be divisible by it
SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP = 256

# Non All-expert MoE TKG MK currently only support intermediate dim to be multiples of 128
MOE_TKG_MK_INTERMEDIATE_PER_TP = 128


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    assert dtype_str in dtype_mapping, f"Unsupported dtype: {dtype_str}"
    return dtype_mapping[dtype_str]


def validate_activation_quantization_type(type: str):
    assert type in ActivationQuantizationType, f"Unsupported activation quantization type: {type}"


def to_dict(obj):
    if type(obj) is dict:
        return {k: to_dict(v) for k, v in obj.items()}
    elif type(obj) is list:
        return [to_dict(v) for v in obj]
    elif inspect.isclass(obj):
        return {
            "__module__": obj.__module__,
            "__name__": obj.__name__,
        }
    elif hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif type(obj) is torch.dtype:
        return str(obj).split(".")[1]
    else:
        return obj


class IncompatibleConfigError(ValueError):
    pass


class NeuronConfig:
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.
    """

    def __init__(self, **kwargs) -> None:
        # Basic config for inference in NxD
        self.batch_size = kwargs.pop("batch_size", 1)
        self.padding_side = kwargs.pop("padding_side", "right")
        self.allow_input_truncation: bool = kwargs.pop("allow_input_truncation", False)
        # TODO: see if we can consolidate n_active_tokens and n_positions into one
        self.seq_len = kwargs.pop("seq_len", 128)
        self.n_active_tokens = kwargs.pop("n_active_tokens", self.seq_len)
        # Need to provide example input shape for tracing
        self.n_positions = kwargs.pop("n_positions", self.seq_len)
        self.on_cpu = kwargs.pop("on_cpu", False)
        self.output_logits = kwargs.pop("output_logits", False)
        self.is_prefill_stage = None  # set only in enable_<>() in NeuronBaseForCausalLM

        # Torch dtype
        if "torch_dtype" in kwargs:
            self.torch_dtype = kwargs.pop("torch_dtype")
            if isinstance(self.torch_dtype, str):
                self.torch_dtype = to_torch_dtype(self.torch_dtype)

            # This flag lets us avoid overriding torch_dtype in HFAdapter's load_pretrained_config.
            self.overrides_torch_dtype = kwargs.pop("overrides_torch_dtype", True)
        else:
            self.torch_dtype = torch.bfloat16
            self.overrides_torch_dtype = False

        self.cast_type = kwargs.pop("cast_type", "config")
        if self.cast_type not in ["config", "as-declared"]:
            raise ValueError("cast_type must be one of config, as-declared")

        self.rpl_reduce_dtype = kwargs.pop("rpl_reduce_dtype", None)
        if isinstance(self.rpl_reduce_dtype, str):
            self.rpl_reduce_dtype = to_torch_dtype(self.rpl_reduce_dtype)

        self.attention_dtype = kwargs.pop("attention_dtype", None)
        if isinstance(self.attention_dtype, str):
            self.attention_dtype = to_torch_dtype(self.attention_dtype)

        if self.attention_dtype != self.torch_dtype and self.attention_dtype is not None:
            assert self.cast_type == "as-declared", "attention dtype and torch dtype are different, however global cast is enabled. To enable running multiple precisions, set cast-type to 'as-declared'"

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = kwargs.pop("max_context_length", self.seq_len)
        self.max_new_tokens = kwargs.pop("max_new_tokens", self.seq_len - self.max_context_length)
        if self.max_new_tokens == 0:
            self.max_new_tokens = None
        self.max_length = kwargs.pop("max_length", self.seq_len)
        assert self.max_context_length <= self.max_length, "max_context_length cannot be more than max_length"

        # Embedding Config
        self.vocab_parallel = kwargs.pop("vocab_parallel", False)

        # TTFT Optimizations
        # This is currently ignored by the Application Base as it's causing logits regression , ticket: V1849736968
        self.enable_cte_modular_flow = kwargs.pop("enable_cte_modular_flow", False)

        # Layer boundary markers
        self.layer_boundary_markers = kwargs.pop("layer_boundary_markers", False)

        # Attention
        self.fused_qkv = kwargs.pop("fused_qkv", False)
        self.sequence_parallel_enabled = kwargs.pop("sequence_parallel_enabled", False)
        self.weight_gather_seq_len_threshold = kwargs.pop("weight_gather_seq_len_threshold", 32768)
        # TODO: Remove Llama attn_cls and multiple attention feature.
        self.attn_cls = kwargs.pop("attn_cls", "NeuronLlamaAttention")

        # Continuous batching
        # TODO: Check if we really need different batch size for CTE and TKG, given
        # that we anyway provide two different config instance for them.
        self.ctx_batch_size = kwargs.pop("ctx_batch_size", self.batch_size)
        self.tkg_batch_size = kwargs.pop("tkg_batch_size", self.batch_size)
        self.max_batch_size = kwargs.pop("max_batch_size", self.batch_size)
        self.is_continuous_batching = kwargs.pop("is_continuous_batching", False)

        # KV cache
        self.kv_cache_batch_size = kwargs.pop("kv_cache_batch_size", self.batch_size)
        # padding size for kv cache in batch-dimension (used in Data Parallel tokengen)
        # (this is useful when kv update needs to be discarded, but need to do write op to maintain SPMD)
        self.kv_cache_padding_size = kwargs.pop("kv_cache_padding_size", 0)

        # On-device sampling
        self.on_device_sampling_config = kwargs.pop("on_device_sampling_config", None)
        if type(self.on_device_sampling_config) is dict:
            self.on_device_sampling_config = OnDeviceSamplingConfig(
                **self.on_device_sampling_config
            )

        # async
        self.async_mode = kwargs.pop("async_mode", False)

        # Bucketing
        self.enable_bucketing = kwargs.pop("enable_bucketing", False)
        self.buckets = kwargs.pop("buckets", [self.seq_len])
        self.bucket_n_active_tokens = kwargs.pop("bucket_n_active_tokens", False)
        self.context_encoding_buckets = kwargs.pop("context_encoding_buckets", None)
        self.prefix_buckets = kwargs.pop("prefix_buckets", None)
        self.token_generation_buckets = kwargs.pop("token_generation_buckets", None)
        if self.context_encoding_buckets is not None:
            self.context_encoding_buckets.sort()
            assert (
                self.context_encoding_buckets[-1] <= self.max_context_length
            ), f"Context bucket {self.context_encoding_buckets[-1]} should be <= {self.max_context_length}"
        if self.token_generation_buckets is not None:
            self.token_generation_buckets.sort()
            assert (
                self.token_generation_buckets[-1] <= self.max_length
            ), f"Token generation bucket {self.token_generation_buckets[-1]} should be <= {self.max_length}"

        # Quantization
        self.quantized = kwargs.pop("quantized", False)
        self.quantized_checkpoints_path = kwargs.pop("quantized_checkpoints_path", None)
        if self.quantized is True:
            assert (
                self.quantized_checkpoints_path is not None
            ), "quantized_checkpoints_path is required"
        self.quantization_type: str = kwargs.pop("quantization_type", "per_tensor_symmetric")
        self.quantization_dtype: str = kwargs.pop("quantization_dtype", "int8")
        self.quantization_block_size = kwargs.pop("quantization_block_size", None)
        self.quantization_block_axis = kwargs.pop("quantization_block_axis", None)
        self.quantization_scale_dtype = kwargs.pop("quantization_scale_dtype", "f32")

        # TODO: move microscaling (MX) config into standalone config object
        self.is_mxfp4_compute = kwargs.pop("is_mxfp4_compute", False)
        self.is_hidden_dim_shuffled = self.is_mxfp4_compute
        self.is_intermediate_dim_shuffled = self.is_mxfp4_compute
        # TODO: remove is_full_model_shuffled flag once MXFP4 compute accuracy is validated
        self.is_full_model_shuffled = kwargs.pop("is_full_model_shuffled", False)
        assert not self.is_full_model_shuffled or self.is_mxfp4_compute, \
            "is_full_model_shuffled is enabled, but is_mxfp4_compute is not enabled. To enable is_full_model_shuffled, set is_mxfp4_compute=True"

        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        self.draft_model_modules_to_not_convert = kwargs.pop(
            "draft_model_modules_to_not_convert", None
        )
        # TODO: Add validation for quantized_checkpoints_path after the design discussions
        self.kv_cache_quant = kwargs.pop("kv_cache_quant", False)

        # Speculative decoding
        self.speculation_length = kwargs.pop("speculation_length", 0)
        self.spec_batch_size = kwargs.pop("spec_batch_size", self.batch_size)
        self.enable_fused_speculation = kwargs.pop("enable_fused_speculation", False)
        self.enable_eagle_speculation = kwargs.pop("enable_eagle_speculation", False)
        self.is_eagle3 = kwargs.pop("is_eagle3", False)
        self.is_eagle_draft = kwargs.pop("is_eagle_draft", False)
        self.enable_eagle_draft_input_norm = kwargs.pop("enable_eagle_draft_input_norm", False)

        if self.enable_eagle_speculation:
            self.enable_fused_speculation = True

        if self.speculation_length > 0 and self.async_mode and not self.enable_fused_speculation:
            warnings.warn(
                "Unfused Speculative Decoding + async mode may not result in performance improvements due to high likelihood of data dependent compute causing blocking calls."
            )

        # Added Token Tree config
        self.token_tree_config = kwargs.pop("token_tree_config", None)
        self.enable_token_tree = self.token_tree_config is not None

        # Medusa decoding
        self.is_medusa = kwargs.pop("is_medusa", False)
        self.medusa_speculation_length = kwargs.pop("medusa_speculation_length", 0)
        self.num_medusa_heads = kwargs.pop("num_medusa_heads", 0)
        self.medusa_tree = kwargs.pop("medusa_tree", None)

        if self.is_medusa and self.async_mode:
            raise IncompatibleConfigError("Medusa Decoding is not yet supported with async.")

        # Paged attention
        self.is_block_kv_layout = kwargs.pop("is_block_kv_layout", False)
        self.pa_num_blocks = kwargs.pop("pa_num_blocks", self.batch_size)
        self.pa_block_size = kwargs.pop("pa_block_size", self.seq_len)

        # Prefix caching
        self.is_prefix_caching = kwargs.pop("is_prefix_caching", False)

        # Windowed Context Encoding
        self.windowed_context_encoding_size = kwargs.pop("windowed_context_encoding_size", None)

        # Chunked prefill
        # When chunked prefill is enabled, max_context_length will be
        # used as chunk size. Batch size will be 1 for CTE because it
        # concatenates all requests along seq dim. cp_config.max_num_seqs
        # is used to denote how many sequences are there in one concatenated
        # request.
        self.chunked_prefill_config = kwargs.pop("chunked_prefill_config", None)
        if type(self.chunked_prefill_config) is dict:
            self.chunked_prefill_config = ChunkedPrefillConfig(
                **self.chunked_prefill_config
            )
        self.is_chunked_prefill = self.chunked_prefill_config is not None
        if self.is_chunked_prefill:
            assert (
                self.is_block_kv_layout
            ), "is_block_kv_layout has to be true for chunked prefill" \
                + " because it depends on block KV"
            if not self.chunked_prefill_config.tkg_model_enabled:
                assert (
                    self.batch_size == 1
                ), "batch_size has to be 1 for chunked prefill CTE model" \
                    + " because it concatenates all requests into one" \
                    + " long request"

        # Tensor replacement for debugging and accuracy verification
        tensor_replacement_config = kwargs.pop("tensor_replacement_config", None)
        if isinstance(tensor_replacement_config, dict):
            self.tensor_replacement_config = TensorReplacementConfig(**tensor_replacement_config)
        elif isinstance(tensor_replacement_config, TensorReplacementConfig):
            self.tensor_replacement_config = tensor_replacement_config
        else:
            if tensor_replacement_config is not None:  # Only warn if something invalid was passed
                warnings.warn(
                    f"tensor_replacement_config should be a TensorReplacementConfig object or dictionary, got {type(tensor_replacement_config)}. "
                    "Tensor replacemnet will be disabled."
                )
            self.tensor_replacement_config = None

        # Tensor capture for debugging and accuracy verification
        tensor_capture_config = kwargs.pop("tensor_capture_config", None)
        if isinstance(tensor_capture_config, dict):
            self.tensor_capture_config = TensorCaptureConfig(**tensor_capture_config)
        elif isinstance(tensor_capture_config, TensorCaptureConfig):
            self.tensor_capture_config = tensor_capture_config
        else:
            if tensor_capture_config is not None:  # Only warn if something invalid was passed
                warnings.warn(
                    f"tensor_capture_config should be a TensorCaptureConfig object or dictionary, got {type(tensor_capture_config)}. "
                    "Tensor capture will be disabled."
                )
            self.tensor_capture_config = None

        # Automatically set output_logits to True if tensor capture is enabled
        # This is required for tensor capture to work properly
        if self.tensor_capture_config and not self.output_logits:
            logging.info("Setting output_logits=True because tensor capture is enabled")
            self.output_logits = True
            if self.on_device_sampling_config is None:
                raise IncompatibleConfigError("Tensor capture requires on device sampling config to be set even if it's just defaults")

        # Lora
        self.lora_config = kwargs.pop("lora_config", None)
        if type(self.lora_config) is dict:
            self.lora_config = LoraServingConfig(**self.lora_config)

        # Distributed config
        self.tp_degree = kwargs.pop("tp_degree", 1)
        self.cp_degree = kwargs.pop("cp_degree", 1)
        self.mlp_cp_degree = kwargs.pop("mlp_cp_degree", 1)
        self.attention_dp_degree = kwargs.pop("attention_dp_degree", 1)
        self.pp_degree = kwargs.pop("pp_degree", 1)
        self.ep_degree = kwargs.pop("ep_degree", 1)
        self.save_sharded_checkpoint = kwargs.pop("save_sharded_checkpoint", False)
        self.skip_sharding = kwargs.pop("skip_sharding", False)

        if self.tp_degree % self.cp_degree != 0:
            raise ValueError("TP Degree must be evenly divisible by CP Degree")

        if self.mlp_cp_degree > 1 and not self.sequence_parallel_enabled:
            raise IncompatibleConfigError("Context Parallel for MLP requires Sequence Parallel to be enabled")

        # QK layer normalization
        self.qk_layernorm = kwargs.pop("qk_layernorm", False)
        # QK RMS normalization after QKV but before RoPE is Attn TKG MK
        self.pre_rope_rmsnorm = kwargs.pop("pre_rope_rmsnorm", False)

        self.world_size = kwargs.pop("world_size", None)
        if self.world_size is None:
            self.world_size = self.tp_degree * self.pp_degree * self.ep_degree

        self.start_rank_id = kwargs.pop("start_rank_id", 0)
        self.local_ranks_size = kwargs.pop("local_ranks_size", None)

        if self.local_ranks_size is None:
            self.local_ranks_size = self.world_size

        # Flash decoding
        self.flash_decoding_enabled = kwargs.pop("flash_decoding_enabled", False)

        # KV Cache tiling optimizations
        #   Tiling the sequence dimension of the KV cache enables specific
        #   compiler optimizations like cascaded reductions

        # Flag to allow users to disable kv cache tiling. If it exists, it'll be true.
        self.disable_kv_cache_tiling = kwargs.pop("disable_kv_cache_tiling", False)
        # If no user input has been taken in for kv cache tiling
        if not self.disable_kv_cache_tiling:
            self.kv_cache_tiling = False
            if self.enable_eagle_speculation or self.enable_fused_speculation:
                # TODO once compiler fixes CR 158191111 we can turn back output tiling on
                # For all models. For now only use it for fused speculation that needs
                # chaining of aliased tensors.
                if self.max_length > 128 and self.max_length % 128 == 0:
                    # Our tile size is 128. We can tile only if sequence length is
                    # divisible by 128.
                    self.kv_cache_tiling = True
        else:  # User requested disabling of kv cache tiling
            self.kv_cache_tiling = False

        self.k_cache_transposed = kwargs.pop("k_cache_transposed", False)

        # Kernels
        self.attn_kernel_enabled = kwargs.pop("attn_kernel_enabled", None)  # CTE attention kernel.
        self.strided_context_parallel_kernel_enabled = kwargs.pop("strided_context_parallel_kernel_enabled", False)
        self.qkv_kernel_enabled = kwargs.pop("qkv_kernel_enabled", False)
        self.qkv_nki_kernel_enabled = kwargs.pop("qkv_nki_kernel_enabled", False)
        self.qkv_cte_nki_kernel_fuse_rope = kwargs.pop("qkv_cte_nki_kernel_fuse_rope", False)
        if self.qkv_cte_nki_kernel_fuse_rope:
            assert self.qkv_kernel_enabled and self.qkv_nki_kernel_enabled, \
                f"When qkv_cte_nki_kernel_fuse_rope is set to True, qkv_kernel_enabled (currently: {self.qkv_kernel_enabled}) and qkv_nki_kernel_enabled (currently: {self.qkv_nki_kernel_enabled}) must also be set to True."
        self.qkv_kernel_nbsd_layout = kwargs.pop("qkv_kernel_nbsd_layout", False)
        self.mlp_kernel_enabled = kwargs.pop("mlp_kernel_enabled", False)
        self.mlp_tkg_nki_kernel_enabled = kwargs.pop("mlp_tkg_nki_kernel_enabled", False)
        self.fused_rmsnorm_skip_gamma = kwargs.pop("fused_rmsnorm_skip_gamma", False)
        self.mlp_kernel_fuse_residual_add = kwargs.pop("mlp_kernel_fuse_residual_add", False)
        self.qkv_kernel_fuse_residual_add = kwargs.pop("qkv_kernel_fuse_residual_add", False)
        self.quantized_mlp_kernel_enabled = kwargs.pop("quantized_mlp_kernel_enabled", False)
        self.out_proj_kernel_enabled = kwargs.pop("out_proj_kernel_enabled", False)
        self.activation_quantization_type = kwargs.pop("activation_quantization_type", None)
        if self.activation_quantization_type is not None:
            validate_activation_quantization_type(self.activation_quantization_type)

        if self.strided_context_parallel_kernel_enabled and not self.cp_degree > 1:
            raise ValueError("CP must also be enabled when strided_context_parallel_kernel_enabled is set.")

        # Token-gen attention kernels
        self.attn_tkg_nki_kernel_enabled = kwargs.pop("attn_tkg_nki_kernel_enabled", False)
        self.attn_tkg_builtin_kernel_enabled = kwargs.pop("attn_tkg_builtin_kernel_enabled", False)
        self.attn_block_tkg_nki_kernel_enabled = kwargs.pop("attn_block_tkg_nki_kernel_enabled", False)
        self.attn_block_cte_nki_kernel_enabled = kwargs.pop("attn_block_cte_nki_kernel_enabled", False)
        if self.attn_block_tkg_nki_kernel_enabled:
            assert (
                self.qkv_kernel_enabled or self.qkv_nki_kernel_enabled
            ), "When attn-block-tkg-nki-kernel-enabled, self.qkv_kernel_enabled or self.qkv_nki_kernel_enabled is also required."
        if self.attn_block_cte_nki_kernel_enabled:
            assert (
                self.attn_block_tkg_nki_kernel_enabled and (self.qkv_kernel_enabled or self.qkv_nki_kernel_enabled)
            ), "When attn-block-cte-nki-kernel-enabled, attn-block-tkg-nki-kernel-enabled and (qkv_kernel_enabled or qkv_nki_kernel_enabled) are also required."
        attn_tkg_kernel_enablement = [
            self.attn_tkg_nki_kernel_enabled,
            self.attn_tkg_builtin_kernel_enabled,
            self.attn_block_tkg_nki_kernel_enabled,
        ]
        assert sum(attn_tkg_kernel_enablement) <= 1, (
            "Multiple token-generation attention kernels enabled. Please enable no more than one of: "
            "[attn-tkg-nki-kernel-enabled, attn-tkg-builtin-kernel-enabled, attn-block-tkg-nki-kernel-enabled]"
        )

        self.attn_block_tkg_nki_kernel_cache_update = kwargs.pop("attn_block_tkg_nki_kernel_cache_update", False)
        self.attn_block_tkg_nki_kernel_cascaded_attention = kwargs.pop("attn_block_tkg_nki_kernel_cascaded_attention", False)
        if not self.attn_block_tkg_nki_kernel_enabled:
            assert not self.attn_block_tkg_nki_kernel_cache_update, \
                'attn-block-tkg-nki-kernel-cache-update can only be enabled with attn-block-tkg-nki-kernel-enabled'
            assert not self.attn_block_tkg_nki_kernel_cascaded_attention, \
                'attn_block_tkg_nki_kernel_cascaded_attention can only be enabled with attn-block-tkg-nki-kernel-enabled'

        self.attn_block_tkg_nki_kernel_use_online_softmax = kwargs.pop("attn_block_tkg_nki_kernel_use_online_softmax", True)
        self.kv_cache_update_with_kernel = False  # TODO: Set this to be true, dependent on compiler fix tracked through ticket V1970034499
        if self.attention_dp_degree > 1:
            self.kv_cache_batch_size = self.tkg_batch_size // self.attention_dp_degree
            if hardware(get_platform_target()) == hardware.TRN1 or not self.attn_block_tkg_nki_kernel_cascaded_attention:
                # TODO: Validate the dma skipping kernel works on trn1. It should theortically work and we can get rid of the old flow
                self.kv_cache_padding_size = 1  # use a padding size of 1 for garbage position writes in DP
            else:
                self.kv_cache_update_with_kernel = True
                self.kv_cache_padding_size = 0  # dma skipping used instead of padding for trn2+

        if self.attn_block_tkg_nki_kernel_cache_update:
            self.kv_cache_tiling = False  # Don't do kv cache tiling when kernel does cache update.

        # Check attention features incompatible with attention block kernel or transposed K cache.
        attn_tkg_incompatible_features = [
            (self.flash_decoding_enabled, 'flash decoding'),
            (self.is_chunked_prefill, 'contexted prefill'),
        ]
        k_cache_tp_incompatible_features = [
            (self.flash_decoding_enabled, 'flash decoding'),
            (self.is_chunked_prefill, 'contexted prefill'),
            (self.is_block_kv_layout, 'block KV cache'),
        ]
        if sum(attn_tkg_kernel_enablement) > 0:
            for flag, feature in attn_tkg_incompatible_features:
                assert not flag, f'Attention TKG kernels do not yet support {feature} feature.'
        if self.k_cache_transposed:
            for flag, feature in k_cache_tp_incompatible_features:
                assert not flag, f'Transposed K cache is not yet supported by {feature} feature.'

        if self.quantized_mlp_kernel_enabled or self.mlp_kernel_enabled:
            assert not (
                self.activation_quantization_type
            ), "native quantization does not work with quantized kernels"
        self.rmsnorm_quantize_kernel_enabled = kwargs.pop("rmsnorm_quantize_kernel_enabled", False)
        if self.rmsnorm_quantize_kernel_enabled:
            assert (
                self.quantized_mlp_kernel_enabled
            ), "quantized_mlp_kernel must be enabled to use rmsnorm_quantize_kernel!"
        self.quantize_clamp_bound = kwargs.pop("quantize_clamp_bound", float("inf"))

        if self.quantized or self.is_mlp_quantized():
            if self.qkv_kernel_enabled or self.qkv_nki_kernel_enabled:
                assert (
                    self.modules_to_not_convert
                ), "Could not find modules_to_not_convert for quantized model."
            elif not self.modules_to_not_convert:
                warnings.warn(
                    "modules_to_not_convert is not provided. Assuming all modules will be quantized.",
                    UserWarning,
                )

        self.moe_fused_nki_kernel_enabled = kwargs.pop("moe_fused_nki_kernel_enabled", None)
        self.router_topk_nki_kernel_enabled = kwargs.pop("router_topk_nki_kernel_enabled", None)
        self.expert_mlp_nki_kernel_enabled = kwargs.pop("expert_mlp_nki_kernel_enabled", None)
        self.shared_mlp_nki_kernel_enabled = kwargs.pop("shared_mlp_nki_kernel_enabled", None)

        # Logical NeuronCore Configuration (LNC)
        self.logical_nc_config = self._get_lnc(kwargs)

        # LM Head Padding for LNC>1
        self.lm_head_pad = kwargs.pop("lm_head_pad", self.logical_nc_config > 1)
        self.lm_head_pad_alignment_size = kwargs.pop("lm_head_pad_alignment_size", 1)

        if self.is_mlp_quantized():
            self.quantization_type = "per_channel_symmetric"
            self.quantization_dtype = "f8e4m3"

        # compiler flags
        self.cc_pipeline_tiling_factor = kwargs.pop("cc_pipeline_tiling_factor", 2)
        self.seq_len_threshold_for_cc_tiling = kwargs.pop("seq_len_threshold_for_cc_tiling", 16384)
        self.tile_cc = False
        # manual tiling of cc with NKI, when MLP kernels are enabled
        # the compiler flow with cc_tiling > 1 with kernels are not accurate
        if self.cc_pipeline_tiling_factor > 1 and (
                self.mlp_kernel_enabled or self.quantized_mlp_kernel_enabled):
            self.tile_cc = True

        self.target = kwargs.pop("target", None)

        # Flag to enable dge for spill reload DMAs in order to get reduction in DMA rings memory for long context
        self.enable_spill_reload_dge = kwargs.pop("enable_spill_reload_dge", False)

        # weights_to_skip_layout_optimization
        self.weights_to_skip_layout_optimization = []
        # skip WLO for lora weights if dynamic multi-lora serving is enabled
        if self.lora_config and self.lora_config.dynamic_multi_lora:
            self.weights_to_skip_layout_optimization.append(r".*lora.*")

        self._verify_quantized_config()

        self.apply_seq_ids_mask = kwargs.pop("apply_seq_ids_mask", False)

        # warmup flag
        self.skip_warmup = kwargs.pop("skip_warmup", False)

        # enable long context flags (to later set compiler/runtime flags)
        self.scratchpad_page_size = None
        self.enable_long_context_mode = self.max_context_length >= 32 * 1024
        if self.enable_long_context_mode:
            self.scratchpad_page_size = LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE

        # override if user provided a value
        self.scratchpad_page_size = kwargs.pop("scratchpad_page_size", self.scratchpad_page_size)
        if self.scratchpad_page_size:
            self.scratchpad_page_size = int(self.scratchpad_page_size)

        # flag to enable output tensor completion signal
        self.enable_output_completion_notifications = kwargs.pop(
            "enable_output_completion_notifications", False)

        self.padded_hidden_size = kwargs.pop("padded_hidden_size", None)
        self.padded_intermediate_size = kwargs.pop("padded_intermediate_size", None)

        self.disable_numeric_cc_token = kwargs.pop(
            "disable_numeric_cc_token", False)
        self.switch_cc = kwargs.pop("switch_cc", False)

        if kwargs:
            logging.warning(f"NeuronConfig init: Unexpected keyword arguments: {kwargs}")

        self.validate_attention_data_parallel(self.attention_dp_degree)

    def _verify_quantized_config(self):
        if not self.quantized:
            return
        assert self.quantized_checkpoints_path is not None, "quantized_checkpoints_path is required"
        # Verification for quantized dtype
        QuantizedDtype.has_dtype(self.quantization_dtype)
        if self.is_mlp_quantized():
            assert self.quantization_dtype == "f8e4m3"

    def validate_attention_data_parallel(self, attention_dp_degree):
        if self.tp_degree % attention_dp_degree != 0:
            raise ValueError("TP Degree must be evenly divisible by DP Degree")

        if self.tkg_batch_size % attention_dp_degree != 0:
            raise ValueError("Batch Size must be evenly divisible by DP Degree")

        if self.cp_degree > 1 and attention_dp_degree > 1 and self.cp_degree < attention_dp_degree:
            raise ValueError("Running DP Degree > CP Degree is not supported when CP and DP is enabled")

        if self.cp_degree > 1 and attention_dp_degree > 1 and self.cp_degree % attention_dp_degree != 0:
            raise ValueError("CP Degree % DP Degree should be 0 when CP and DP is enabled")

        if attention_dp_degree > 1 and not self.is_continuous_batching:
            raise ValueError("--is-continuous-batching must be enabled when running DP")

        # TODO: remove this after compiler fix V1970034499 for self.kv_cache_update_with_kernel
        if attention_dp_degree > 1 and self.attn_block_tkg_nki_kernel_cache_update and not self.kv_cache_update_with_kernel:
            raise ValueError("kv_cache_update_with_kernel must be set to enable attn_block_tkg_nki_kernel_cache_update when DP is used")

    def _get_lnc(self, kwargs):
        env_lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", -1))
        kwargs_lnc = -1
        platform_lnc = get_platform_lnc()
        if "logical_neuron_cores" in kwargs:
            # For backward compatibility, default to the deprecated logical_neuron_cores attribute if set.
            warning_message = (
                "'logical_neuron_cores' is deprecated and replaced by 'logical_nc_config'. "
                "In a future release, this attribute will be removed."
            )
            warnings.warn(warning_message, category=DeprecationWarning)
            kwargs_lnc = int(kwargs.pop("logical_neuron_cores"))

        if "logical_nc_config" in kwargs:
            kwargs_lnc = int(kwargs.pop("logical_nc_config"))

        # Warn that LNC will not match provided LNC if NEURON_LOGICAL_NC_CONFIG != logical_nc_config kwarg
        if env_lnc != -1 and kwargs_lnc != -1 and env_lnc != kwargs_lnc:
            warning_message = (
                f"NEURON_LOGICAL_NC_CONFIG={env_lnc} does not match provided logical_nc_config={kwargs_lnc}. "
                "Using NEURON_LOGICAL_NC_CONFIG to set LNC."
            )
            warnings.warn(warning_message, category=UserWarning)

        # LNC set with priority: NEURON_LOGICAL_NC_CONFIG > logical_nc_config kwarg > platform LNC
        if env_lnc != -1:
            return env_lnc
        elif kwargs_lnc != -1:
            return kwargs_lnc
        else:
            return platform_lnc

    def __getattribute__(self, name):
        # Maintain backward compatibility without serializing deprecated 'logical_neuron_cores' attr.
        if name == "logical_neuron_cores":
            warning_message = (
                "'logical_neuron_cores' is deprecated and will be removed in a future release. "
                "Use the 'logical_nc_config' attribute instead."
            )
            warnings.warn(warning_message, category=DeprecationWarning)
            return self.logical_nc_config

        if name == "trace_tokengen_model":
            warning_message = (
                "'trace_tokengen_model' is deprecated and will be removed in a future release. "
                + "Returning the expected value based on current configuration"
            )
            warnings.warn(warning_message, category=DeprecationWarning)
            return (
                not self.enable_fused_speculation
                or not self.speculation_length > 0
                or not self.medusa_speculation_length > 0
            )  # return true if speculation isn't enabled
        return super().__getattribute__(name)

    def is_mlp_quantized(self):
        return self.quantized_mlp_kernel_enabled or self.activation_quantization_type


class MultimodalVisionNeuronConfig(NeuronConfig):
    """
    for multimodal vision config on Neuron
    """

    def __init__(self, **kwargs) -> None:
        self.skip_vision = kwargs.pop("skip_vision", False)
        super().__init__(**kwargs)


class MoENeuronConfig(NeuronConfig):
    """
    Base class for mixture of experts (MoE) config on Neuron.
    """

    def __init__(
        self,
        capacity_factor: float = None,
        glu_mlp: bool = True,
        **kwargs,
    ) -> None:
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp
        self.glu_type = kwargs.pop("glu_type", "glu")
        self.hidden_act_scaling_factor = kwargs.pop("hidden_act_scaling_factor", 1.)
        self.hidden_act_bias = kwargs.pop("hidden_act_bias", 0.)
        self.gate_clamp_upper_limit = kwargs.pop("gate_clamp_upper_limit", None)
        self.gate_clamp_lower_limit = kwargs.pop("gate_clamp_lower_limit", None)
        self.up_clamp_upper_limit = kwargs.pop("up_clamp_upper_limit", None)
        self.up_clamp_lower_limit = kwargs.pop("up_clamp_lower_limit", None)

        self.use_index_calc_kernel = kwargs.pop("use_index_calc_kernel", False)
        self.moe_mask_padded_tokens = kwargs.pop("moe_mask_padded_tokens", False)
        self.early_expert_affinity_modulation = kwargs.pop("early_expert_affinity_modulation", False)
        self.normalize_top_k_affinities = not kwargs.pop("disable_normalize_top_k_affinities", False)
        self.fused_shared_experts = kwargs.pop("fused_shared_experts", False)
        self.shared_experts_sequence_parallel_enabled = kwargs.pop("shared_experts_sequence_parallel_enabled", False)
        self.return_expert_index = kwargs.pop("return_expert_index", False)
        self.return_router_logits = kwargs.pop("return_router_logits", False)
        self.hybrid_sharding_config = kwargs.pop("hybrid_sharding_config", None)
        if type(self.hybrid_sharding_config) is dict:
            self.hybrid_sharding_config = HybridShardingConfig(
                **self.hybrid_sharding_config
            )

        self.moe_tp_degree = kwargs.pop("moe_tp_degree", 1)
        self.moe_ep_degree = kwargs.pop("moe_ep_degree", 1)
        self.transpose_shared_experts_weights = kwargs.pop("transpose_shared_experts_weights", False)

        self.blockwise_matmul_config = kwargs.pop("blockwise_matmul_config", {})
        if isinstance(self.blockwise_matmul_config, dict):
            self.blockwise_matmul_config = BlockwiseMatmulConfig.from_kwargs(**self.blockwise_matmul_config)

        self.router_config = kwargs.pop("router_config", None)
        if isinstance(self.router_config, dict):
            self.router_config = RouterConfig(**self.router_config)
        else:
            self.router_config = RouterConfig.from_kwargs(**kwargs)
        super().__init__(**kwargs)


class InferenceConfig:
    # Alias map for attributes.
    attribute_map: Dict[str, str] = {}

    def __init__(
        self,
        neuron_config: NeuronConfig,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs
    ):
        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config
        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        self.metadata = metadata

        # Override config values from kwargs.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.add_derived_config()

        self.validate_config()

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def add_derived_config(self):
        """
        Override this in custom model InferenceConfig for flash decoding. See LlamaInferenceConfig
        """
        self.num_cores_per_group = 1
        pass

    def load_config(self):
        """
        Loads the config and sets attributes needed by the model you use.
        """
        pass

    def get_required_attributes(self) -> List[str]:
        """The list of attributes that must be present for validation to pass."""
        return []

    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """
        missing_attributes = [x for x in self.get_required_attributes() if not hasattr(self, x)]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"
        self._validate_chunked_attention_support()
        self._validate_windowed_context_encoding_support()

    def _validate_chunked_attention_support(self):
        if hasattr(self, "attention_chunk_size") and self.attention_chunk_size < self.neuron_config.seq_len:
            for config, config_value in CHUNKED_ATTENTION_SUPPORTED_NEURON_CONFIG.items():
                if getattr(self.neuron_config, config) != config_value:
                    raise ValueError(f"The Neuron config {config}: {getattr(self.neuron_config, config)} is not yet supported with chunked attention. Only config value of {config_value} is supported")
            assert self.attention_chunk_size % self.neuron_config.cp_degree == 0, f"attention_chunk_size: {self.attention_chunk_size} must be divisible by cp_degree: {self.neuron_config.cp_degree}"
            assert (self.neuron_config.seq_len % self.attention_chunk_size) % self.neuron_config.cp_degree == 0, f"The last chunk must be divisible by cp_degree: {self.neuron_config.cp_degree}"

    def _validate_windowed_context_encoding_support(self):
        wce_size = self.neuron_config.windowed_context_encoding_size
        if wce_size is not None and hasattr(self, "sliding_window") and self.sliding_window is not None:
            assert wce_size == self.sliding_window, f"Windowed context encoding size must equal sliding window size, if using both. Got windowed_context_encoding_size = {wce_size}, sliding_window = {self.sliding_window}"

    def save(self, model_path: Union[str, os.PathLike]):
        """
        Saves the config to a JSON file in the given model directory.
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config_file = os.path.join(model_path, CONFIG_FILE)
        self.to_json_file(config_file)

    def to_json_file(self, json_file: Union[str, os.PathLike]):
        with open(json_file, "w", encoding="utf-8") as writer:
            config_json = self.to_json_string()
            logging.debug(f"Saving config: {config_json}")
            writer.write(config_json + "\n")

    def to_json_string(self) -> str:
        config_dict = to_dict(self)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def get_text_config(self):
        """
        Returns text_config for the text model in multi-modal models.
        Returns original config for text models
        """
        if hasattr(self, "text_config") and self.text_config is not None:
            return self.text_config

        return self

    @staticmethod
    def get_draft_neuron_class(fused_spec_config_dict):
        if fused_spec_config_dict.get("draft_model_cls" , None):
            draft_model_cls_name = fused_spec_config_dict["draft_model_cls"]["__name__"]
            draft_neuron_module_path = fused_spec_config_dict["draft_model_cls"]["__module__"]
            draft_neuron_module = importlib.import_module(draft_neuron_module_path)
            return getattr(draft_neuron_module, draft_model_cls_name)
        return None

    @classmethod
    def load(cls, model_path: Union[str, os.PathLike], **kwargs) -> "InferenceConfig":
        """
        Loads the config from the given model directory.

        The given kwargs override any properties of the same name from the JSON file.
        """
        config_file = os.path.join(model_path, CONFIG_FILE)
        return cls.from_json_file(config_file, **kwargs)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs) -> "InferenceConfig":
        with open(json_file, "r", encoding="utf-8") as reader:
            config = cls.from_json_string(reader.read(), **kwargs)
            logging.info(f"Loaded Neuron config: {config.to_json_string()}")
            return config

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> "InferenceConfig":
        merged_kwargs = json.loads(json_string)
        merged_kwargs.update(kwargs)

        # Initialize NeuronConfig from dict.
        if "neuron_config" in merged_kwargs and isinstance(merged_kwargs["neuron_config"], dict):
            merged_kwargs["neuron_config"] = cls.get_neuron_config_cls()(
                **merged_kwargs["neuron_config"]
            )
        # Initialize FusedSpecNeuronConfig from dict.
        if "fused_spec_config" in merged_kwargs and isinstance(
            merged_kwargs["fused_spec_config"], dict
        ):
            draft_neuron_class = cls.get_draft_neuron_class(merged_kwargs["fused_spec_config"])
            if "draft_config" in merged_kwargs["fused_spec_config"] and isinstance(
                merged_kwargs["fused_spec_config"]["draft_config"], dict
            ):
                # Initialize NeuronConfig from dict.
                if "neuron_config" in merged_kwargs["fused_spec_config"][
                    "draft_config"
                ] and isinstance(
                    merged_kwargs["fused_spec_config"]["draft_config"]["neuron_config"], dict
                ):
                    merged_kwargs["fused_spec_config"]["draft_config"][
                        "neuron_config"
                    ] = draft_neuron_class.get_config_cls().get_neuron_config_cls()(
                        **merged_kwargs["fused_spec_config"]["draft_config"]["neuron_config"]
                    ) if draft_neuron_class is not None else cls.get_neuron_config_cls()(
                        **merged_kwargs["fused_spec_config"]["draft_config"]["neuron_config"]
                    )
                merged_kwargs["fused_spec_config"]["draft_config"] = draft_neuron_class.get_config_cls()(
                    **merged_kwargs["fused_spec_config"]["draft_config"]
                ) if draft_neuron_class is not None else cls(
                    **merged_kwargs["fused_spec_config"]["draft_config"]
                )

            fused_spec_config = FusedSpecNeuronConfig(
                **merged_kwargs["fused_spec_config"]
            )
            fused_spec_config.draft_model_cls = cls.get_draft_neuron_class(merged_kwargs["fused_spec_config"])

            model_cls_name = fused_spec_config.worker_cls["__name__"]
            neuron_module_path = fused_spec_config.worker_cls["__module__"]
            try:
                neuron_module = importlib.import_module(neuron_module_path)
                fused_spec_config.worker_cls = getattr(neuron_module, model_cls_name)
            except Exception as e:
                raise ModuleNotFoundError(
                    f"Failed to load class {model_cls_name} from module {neuron_module_path}. "
                    f"Make sure the module exists in NxDI and model class is supported. "
                    f"If the compiled model is from NxDI v0.2 or earlier, try recompiling the model."
                ) from e

            merged_kwargs["fused_spec_config"] = fused_spec_config

        return cls(**merged_kwargs)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class FusedSpecNeuronConfig:
    """
    Base class for fused speculative decoding on Neuron.
    """

    # attribute_map: Dict[str, str] = {}
    def __init__(
        self,
        worker_cls,
        draft_config: InferenceConfig = None,
        draft_model_path: str = None,
        draft_model_cls=None,
    ) -> None:
        self.worker_cls = worker_cls
        self.draft_model_cls = draft_model_cls
        self.draft_config = draft_config
        self.draft_model_path = draft_model_path


class OnDeviceSamplingConfig:
    def __init__(self, **kwargs):
        self.do_sample = kwargs.pop("do_sample", False)
        self.top_k = kwargs.pop("top_k", 1)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.dynamic = kwargs.pop("dynamic", False)
        self.deterministic = kwargs.pop("deterministic", False)
        self.global_topk = kwargs.pop("global_topk", 256)
        self.on_device_sampling_config = kwargs.pop("on_device_sampling_config", True)
        self.top_k_kernel_enabled = kwargs.pop("top_k_kernel_enabled", False)


class ChunkedPrefillConfig:
    def __init__(self, **kwargs):
        # max_num_seqs is the actual batch size for chunked prefill
        self.max_num_seqs = kwargs.pop("max_num_seqs", 0)

        # Whether to enable a separate TKG model for decoding-only requests.
        # With chunked prefill, the CTE model is always enabled and it can
        # process both prefilling and decoding requests in one iteration,
        # while a separate TKG model is OPTIONAL. The seprate TKG model can
        # process decoding-only requests, but in a faster way for now.
        self.tkg_model_enabled = kwargs.pop("tkg_model_enabled", True)

        # query len of the tile to be used by the kernel
        self.kernel_q_tile_size = kwargs.pop("kernel_q_tile_size", 128)
        # kv cache len of the tile to be used by the kernel
        self.kernel_kv_tile_size = kwargs.pop("kernel_kv_tile_size", 1024)


class HybridShardingConfig:
    def __init__(self, **kwargs):
        self.moe_cte_tp_degree = kwargs.pop("moe_cte_tp_degree", 1)
        self.moe_cte_ep_degree = kwargs.pop("moe_cte_ep_degree", 1)
        self.moe_tkg_tp_degree = kwargs.pop("moe_tkg_tp_degree", 1)
        self.moe_tkg_ep_degree = kwargs.pop("moe_tkg_ep_degree", 1)


def get_platform_lnc():
    """
    Get the Logical NeuronCore Configuration (LNC) for the current platform.
    """
    warning_message = (
        "neuronx_distributed_inference.models.config.get_platform_lnc() is deprecated. "
        "Use neuronx_distributed.utils.model_utils.get_platform_lnc() instead."
    )
    warnings.warn(warning_message, category=DeprecationWarning)

    target = get_platform_target()
    if target == "trn2" or target == "trn3":
        return 2
    else:
        return 1


class TensorCaptureConfig:
    """
    Configuration class for tensor capture settings.

    This class encapsulates all settings related to tensor capture for debugging
    and accuracy verification.
    """

    def __init__(self, **kwargs):
        # List of module names to capture tensors from
        self.modules_to_capture = kwargs.pop("modules_to_capture", [])

        # Maximum number of intermediate tensors to capture
        self.max_intermediate_tensors = kwargs.pop("max_intermediate_tensors", None)
        # Automatically capture moe tensors for expert stats
        self.auto_capture_moe_tensors = kwargs.pop("auto_capture_moe_tensors", False)

        # Whether to capture input tensors for the specified modules
        self.capture_inputs = kwargs.pop("capture_inputs", False)

        # add one max_intermediate_tensors since auto captured moe tensors will be stacked as one single tensor
        # to ensure fixed shape model output
        if self.auto_capture_moe_tensors:
            self.max_intermediate_tensors = self.max_intermediate_tensors + 1 if self.max_intermediate_tensors else 1

        # Validate that if explicit values were provided, at least one capture mechanism is enabled
        if "modules_to_capture" in kwargs or "max_intermediate_tensors" in kwargs:
            if not self.modules_to_capture and self.max_intermediate_tensors is None:
                raise ValueError(
                    "When configuring tensor capture, at least one of "
                    "'modules_to_capture' or 'max_intermediate_tensors' must be specified."
                )

    def get_offset(self):
        """
        Calculate the total number of tensors that will be captured.
        This is used to determine the offset in the output tensors.

        Returns:
            int: The total number of tensors to be captured
        """
        capture_offset = 0
        if self.modules_to_capture:
            capture_offset = len(self.modules_to_capture)
            if self.capture_inputs:
                capture_offset += len(self.modules_to_capture)
        if self.max_intermediate_tensors:
            capture_offset += self.max_intermediate_tensors
        return capture_offset


class TensorReplacementConfig:
    def __init__(self, **kwargs):
        if "ref_dir" not in kwargs:
            raise ValueError("Missing required argument: ref_dir")
        if 'neuron_dir' not in kwargs:
            raise ValueError("Missing required argument: neuron_dir")
        if 'tf_map' not in kwargs:
            raise ValueError("Missing required argument: tf_map")

        self.ref_dir = kwargs.pop("ref_dir", None)
        if not os.path.isdir(self.ref_dir):
            raise FileNotFoundError(f"ref_dir does not exist or is not a directory: {self.ref_dir}")
        self.neuron_dir = kwargs.pop("neuron_dir", None)
        if not os.path.isdir(self.neuron_dir):
            raise FileNotFoundError(f"neuron_dir does not exist or is not a directory: {self.neuron_dir}")
        self.tr_map = kwargs.pop("tf_map", None)
        if type(self.tr_map) is not dict:
            raise FileNotFoundError(f"Tensor replacement map is expected to be a dict but got: {type(self.tr_map)}")
        self.module_map = kwargs.pop("module_map", None)

        reg = TensorReplacementRegister.get_instance(
            ref_dir=self.ref_dir,
            neuron_dir=self.neuron_dir,
            tr_map=self.tr_map,
            ref_equiv_map=self.module_map,
        )
        RuntimeRegister.module_superset = reg.module_superset
