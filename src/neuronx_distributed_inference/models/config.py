import json
import logging
import os
from typing import Dict, List, Type, Union

import torch

from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig

CONFIG_FILE = "neuron_config.json"


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    assert dtype_str in dtype_mapping, f"Unsupported dtype: {dtype_str}"
    return dtype_mapping[dtype_str]


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
        # TODO: see if we can consolidate n_active_tokens and n_positions into one
        self.seq_len = kwargs.pop("seq_len", 128)
        self.n_active_tokens = kwargs.pop("n_active_tokens", self.seq_len)
        # Need to provide example input shape for tracing
        self.n_positions = kwargs.pop("n_positions", self.seq_len)
        self.on_cpu = kwargs.pop("on_cpu", False)
        self.torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = to_torch_dtype(self.torch_dtype)

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = kwargs.pop("max_context_length", self.seq_len)
        self.max_new_tokens = kwargs.pop("max_new_tokens", self.seq_len - self.max_context_length)
        if self.max_new_tokens == 0:
            self.max_new_tokens = None
        self.max_length = kwargs.pop("max_length", self.seq_len)

        # Attention
        self.fused_qkv = kwargs.pop("fused_qkv", False)
        # TODO: Remove Llama attn_cls and multiple attention feature.
        self.attn_cls = kwargs.pop("attn_cls", "NeuronLlamaAttention")

        # Continuous batching
        # TODO: Check if we really need different batch size for CTE and TKG, given
        # that we anyway provide two different config instance for them.
        self.ctx_batch_size = kwargs.pop("ctx_batch_size", self.batch_size)
        self.tkg_batch_size = kwargs.pop("tkg_batch_size", self.batch_size)
        self.max_batch_size = kwargs.pop("max_batch_size", self.batch_size)
        self.is_continuous_batching = kwargs.pop("is_continuous_batching", False)

        # On-device sampling
        self.on_device_sampling = kwargs.pop("on_device_sampling", False)

        # Bucketing
        self.enable_bucketing = kwargs.pop("enable_bucketing", False)
        self.buckets = kwargs.pop("buckets", [self.seq_len])
        self.bucket_n_active_tokens = kwargs.pop("bucket_n_active_tokens", False)

        # Quantization
        self.quantized = kwargs.pop("quantized", False)
        self.quantized_checkpoints_path = kwargs.pop("quantized_checkpoints_path", None)
        if self.quantized is True:
            assert (
                self.quantized_checkpoints_path is not None
            ), "quantized_checkpoints_path is required"
        self.quantization_type = kwargs.pop("quantization_type", "per_tensor_symmetric")
        # TODO: Add validation for quantized_checkpoints_path after the design discussions

        # Speculative decoding
        self.trace_tokengen_model = kwargs.pop("trace_tokengen_model", True)
        self.speculation_length = kwargs.pop("speculation_length", 0)
        self.spec_batch_size = kwargs.pop("spec_batch_size", self.batch_size)

        # Medusa decoding
        self.is_medusa = kwargs.pop("is_medusa", False)
        self.medusa_speculation_length = kwargs.pop("medusa_speculation_length", 0)
        self.num_medusa_heads = kwargs.pop("num_medusa_heads", 0)
        self.medusa_tree = kwargs.pop("medusa_tree", 0)

        # Lora
        self.lora_config = kwargs.pop("lora_config", None)
        if type(self.lora_config) is dict:
            self.lora_config = LoraServingConfig(**self.lora_config)

        # Distributed config
        self.tp_degree = kwargs.pop("tp_degree", 1)
        self.pp_degree = kwargs.pop("pp_degree", 1)
        self.ep_degree = kwargs.pop("ep_degree", 1)

        self.world_size = kwargs.pop("world_size", None)
        if self.world_size is None:
            self.world_size = self.tp_degree * self.pp_degree * self.ep_degree

        self.start_rank_id = kwargs.pop("start_rank_id", 0)
        self.local_ranks_size = kwargs.pop("local_ranks_size", None)

        if self.local_ranks_size is None:
            self.local_ranks_size = self.world_size


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

    def __init__(self, neuron_config: NeuronConfig, load_config=None, **kwargs):
        self.neuron_config = neuron_config
        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        # Override config values from kwargs.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.validate_config()

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

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

        return cls(**merged_kwargs)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig
