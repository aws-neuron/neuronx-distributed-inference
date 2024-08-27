import torch
from transformers import PretrainedConfig

from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig


class NeuronConfig:
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.
    """

    def __init__(self, **kwargs) -> None:
        # Basic config for inference in NxD
        self.tp_degree = kwargs.pop("tp_degree", 1)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.padding_side = kwargs.pop("padding_side", "right")
        # TODO: see if we can consolidate n_active_tokens and n_positions into one
        self.seq_len = kwargs.pop("seq_len", 128)
        self.n_active_tokens = kwargs.pop("n_active_tokens", self.seq_len)
        # Need to provide example input shape for tracing
        self.n_positions = kwargs.pop("n_positions", self.seq_len)
        self.on_cpu = kwargs.pop("on_cpu", False)

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


class PretrainedConfigAdapter(PretrainedConfig):
    """
    Adapts PretrainedConfig to support a nested neuron_config attribute.
    """

    def __init__(self, neuron_config: NeuronConfig = None, **kwargs):
        self.neuron_config = neuron_config
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        if isinstance(self.neuron_config, NeuronConfig):
            output["neuron_config"] = to_dict(self.neuron_config)
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config: PretrainedConfigAdapter = super().from_dict(config_dict, **kwargs)
        if config.neuron_config is not None:
            merged_kwargs = config.neuron_config
            merged_kwargs.update(kwargs)
            config.neuron_config = cls.get_neuron_config_cls()(**merged_kwargs)
        return config

    @classmethod
    def get_neuron_config_cls(cls) -> NeuronConfig:
        raise NeuronConfig


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
