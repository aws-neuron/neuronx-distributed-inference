import json
import logging
import os
import warnings
from typing import Dict, List, Type, Union
import torch
from neuronx_distributed.quantization.quantization_config import QuantizedDtype, ActivationQuantizationType
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig

CONFIG_FILE = "neuron_config.json"


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

        self.rpl_reduce_dtype = kwargs.pop("rpl_reduce_dtype", self.torch_dtype)

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = kwargs.pop("max_context_length", self.seq_len)
        self.max_new_tokens = kwargs.pop("max_new_tokens", self.seq_len - self.max_context_length)
        if self.max_new_tokens == 0:
            self.max_new_tokens = None
        self.max_length = kwargs.pop("max_length", self.seq_len)

        # Embedding Config
        self.vocab_parallel = kwargs.pop("vocab_parallel", False)

        # Attention
        self.fused_qkv = kwargs.pop("fused_qkv", False)
        self.sequence_parallel_enabled = kwargs.pop("sequence_parallel_enabled", False)
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
        self.is_eagle_draft = kwargs.pop("is_eagle_draft", False)
        self.enable_eagle_draft_input_norm = kwargs.pop("enable_eagle_draft_input_norm", False)

        if self.enable_eagle_speculation:
            self.enable_fused_speculation = True

        if self.speculation_length > 0 and self.async_mode and not self.enable_fused_speculation:
            raise IncompatibleConfigError(
                "Unfused Speculative Decoding is not yet supported with async."
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
        # # Prefix caching
        self.is_prefix_caching = kwargs.pop("is_prefix_caching", False)
        # # Chunked prefill
        # When chunked prefill is enabled, max_context_length will be
        # used as chunk size. Batch size will be 1 because it concatenates
        # all requests along seq dim. cp_max_num_seqs is used to denote how
        # many sequences are there in one concatenated request.
        self.is_chunked_prefill = kwargs.pop("is_chunked_prefill", False)
        self.cp_max_num_seqs = kwargs.pop("cp_max_num_seqs", 0)
        self.cp_num_active_blocks = kwargs.pop("cp_num_active_blocks", 0)
        if self.is_chunked_prefill:
            assert (
                self.is_block_kv_layout
            ), "is_block_kv_layout has to be true for chunked prefill \
                because it depends on block KV"
            assert (
                self.batch_size == 1
            ), "batch_size has to be 1 for chunked prefill \
                because it concatenates all requests into one long request"

        # Lora
        self.lora_config = kwargs.pop("lora_config", None)
        if type(self.lora_config) is dict:
            self.lora_config = LoraServingConfig(**self.lora_config)

        # Distributed config
        self.tp_degree = kwargs.pop("tp_degree", 1)
        self.pp_degree = kwargs.pop("pp_degree", 1)
        self.ep_degree = kwargs.pop("ep_degree", 1)
        self.save_sharded_checkpoint = kwargs.pop("save_sharded_checkpoint", True)
        self.skip_sharding = kwargs.pop("skip_sharding", False)

        # QK layer normalization
        self.qk_layernorm = kwargs.pop("qk_layernorm", False)

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
        self.kv_cache_tiling = False
        if self.enable_eagle_speculation or self.enable_fused_speculation:
            # TODO once compiler fixes CR 158191111 we can turn back output tiling on
            # For all models. For now only use it for fused speculation that needs
            # chaining of aliased tensors.
            if self.max_length > 128 and self.max_length % 128 == 0:
                # Our tile size is 128. We can tile only if sequence length is
                # divisible by 128.
                self.kv_cache_tiling = True

        # Kernels
        self.attn_kernel_enabled = kwargs.pop("attn_kernel_enabled", False)
        self.qkv_kernel_enabled = kwargs.pop("qkv_kernel_enabled", False)
        self.qkv_kernel_nbsd_layout = kwargs.pop("qkv_kernel_nbsd_layout", False)
        self.mlp_kernel_enabled = kwargs.pop("mlp_kernel_enabled", False)
        self.mlp_kernel_fuse_residual_add = kwargs.pop("mlp_kernel_fuse_residual_add", False)
        self.quantized_mlp_kernel_enabled = kwargs.pop("quantized_mlp_kernel_enabled", False)
        self.activation_quantization_type = kwargs.pop("activation_quantization_type", None)
        if self.activation_quantization_type is not None:
            validate_activation_quantization_type(self.activation_quantization_type)

        if self.quantized_mlp_kernel_enabled or self.mlp_kernel_enabled:
            assert not (
                self.activation_quantization_type
            ), "native quantization does not work with quantized kernels"
        self.rmsnorm_quantize_kernel_enabled = kwargs.pop("rmsnorm_quantize_kernel_enabled", False)
        if self.rmsnorm_quantize_kernel_enabled:
            assert (
                self.quantized_mlp_kernel_enabled
            ), "quantized_mlp_kernel must be enabled to use rmsomrm_quantize_kernel!"
        self.quantize_clamp_bound = kwargs.pop("quantize_clamp_bound", float('inf'))

        if self.quantized or self.is_mlp_quantized():
            if self.qkv_kernel_enabled:
                assert (
                    self.modules_to_not_convert
                ), "Could not find modules_to_not_convert for quantized model."
            elif not self.modules_to_not_convert:
                warnings.warn(
                    "modules_to_not_convert is not provided. Assuming all modules will be quantized.",
                    UserWarning
                )

        # Logical NeuronCore Configuration (LNC)
        lnc = get_platform_lnc()
        if "logical_neuron_cores" in kwargs:
            # For backward compatibility, default to the deprecated logical_neuron_cores attribute if set.
            warning_message = (
                "'logical_neuron_cores' is deprecated and replaced by 'logical_nc_config'. "
                "In a future release, this attribute will be removed."
            )
            warnings.warn(warning_message, category=DeprecationWarning)
            lnc = kwargs.pop("logical_neuron_cores")
        self.logical_nc_config = int(
            os.environ.get("NEURON_LOGICAL_NC_CONFIG", kwargs.pop("logical_nc_config", lnc))
        )

        if self.is_mlp_quantized():
            self.quantization_type = "per_channel_symmetric"
            self.quantization_dtype = "f8e4m3"

        # compiler flags
        self.cc_pipeline_tiling_factor = kwargs.pop("cc_pipeline_tiling_factor", 2)
        self.target = kwargs.pop("target", None)

        # weights_to_skip_layout_optimization
        self.weights_to_skip_layout_optimization = []

        if kwargs:
            logging.warn(f"NeuronConfig init: Unexpected keyword arguments: {kwargs}")

        self._verify_quantized_config()

        # warmup flag
        self.skip_warmup = kwargs.pop("skip_warmup", False)

    def _verify_quantized_config(self):
        if not self.quantized:
            return
        assert self.quantized_checkpoints_path is not None, "quantized_checkpoints_path is required"
        # Verification for quantized dtype
        QuantizedDtype.has_dtype(self.quantization_dtype)
        if self.is_mlp_quantized():
            assert self.quantization_dtype == "f8e4m3"

    def __getattribute__(self, name):
        # Maintain backward compatibility without serializing deprecated 'logical_neuron_cores' attr.
        if name == "logical_neuron_cores":
            warning_message = (
                "'logical_neuron_cores' is deprecated and will be removed in a future release. "
                "Use the 'logical_nc_config' attribute instead."
            )
            warnings.warn(warning_message, category=DeprecationWarning)
            return self.logical_nc_config

        # Maintain backward compatibility without serializing deprecated 'logical_neuron_cores' attr.

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
        super().__init__(**kwargs)


class InferenceConfig:
    # Alias map for attributes.
    attribute_map: Dict[str, str] = {}

    def __init__(
        self, neuron_config: NeuronConfig, fused_spec_config=None, load_config=None, **kwargs
    ):
        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config
        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

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
                    ] = cls.get_neuron_config_cls()(
                        **merged_kwargs["fused_spec_config"]["draft_config"]["neuron_config"]
                    )
                merged_kwargs["fused_spec_config"]["draft_config"] = cls(
                    **merged_kwargs["fused_spec_config"]["draft_config"]
                )
            merged_kwargs["fused_spec_config"] = FusedSpecNeuronConfig(
                **merged_kwargs["fused_spec_config"]
            )

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
    ) -> None:
        self.worker_cls = worker_cls
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


def get_platform_lnc():
    """
    Get the Logical NeuronCore Configuration (LNC) for the current platform.
    """
    target = get_platform_target()
    if target == "trn2":
        return 2
    else:
        return 1
