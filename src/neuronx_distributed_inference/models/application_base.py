import os
import warnings
from typing import List

import torch
from neuronx_distributed.quantization.quantization_config import QuantizationType
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.trace.model_builder import ModelBuilder
import neuronx_distributed.trace.hlo_utils as hlo_utils
from safetensors.torch import load_file

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
from neuronx_distributed_inference.modules.checkpoint import load_state_dict, prune_state_dict

COMPILED_MODEL_FILE_NAME = "model.pt"


def is_compiled(model_path):
    return os.path.isfile(model_path + COMPILED_MODEL_FILE_NAME)


class NeuronApplicationBase(torch.nn.Module):
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""

    # TODO: clear generation_config
    def __init__(
        self,
        model_path: str,
        config: InferenceConfig = None,
        neuron_config: NeuronConfig = None,
        generation_kwargs={"do_sample": True, "top_k": 1},
    ):
        if config is None:
            config = self.get_config_cls().load(model_path, **generation_kwargs)

        if neuron_config is not None:
            config.neuron_config = neuron_config

        self.validate_config(config)
        self.config = config
        self.neuron_config = config.neuron_config
        self.model_path = model_path
        self.models: List[ModelWrapper] = []
        self.traced_model = None
        self.is_compiled = is_compiled(model_path)
        self.is_loaded_to_neuron = False

        super().__init__()

    def forward(self, **kwargs):
        """Forward pass for this model."""
        raise NotImplementedError("forward is not implemented")

    @classmethod
    def validate_config(cls, config: InferenceConfig):
        """Checks whether the config is valid for this model."""
        if not hasattr(config, "neuron_config"):
            raise ValueError("Config must include a NeuronConfig")

    @classmethod
    def get_config_cls(cls) -> InferenceConfig:
        """Gets the config class for this model."""
        raise NotImplementedError("get_config_cls is not implemented")

    @classmethod
    def get_neuron_config_cls(cls) -> NeuronConfig:
        # TODO: improve the config access
        return cls.get_config_cls().get_neuron_config_cls()

    def get_compiler_args(self) -> str:
        """Gets the Neuron compiler arguments to use when compiling this model."""
        return None

    def compile(self, compiled_model_path):
        """Compiles this model and saves it to the given path."""
        self.config.save(compiled_model_path)

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        builder = ModelBuilder(
            router=None,
            tp_degree=self.neuron_config.tp_degree,
            pp_degree=self.neuron_config.pp_degree,
            ep_degree=self.neuron_config.ep_degree,
            world_size=self.neuron_config.world_size,
            start_rank_id=self.neuron_config.start_rank_id,
            local_ranks_size=self.neuron_config.local_ranks_size,
            checkpoint_loader=self.checkpoint_loader_fn,
            compiler_workdir=base_compile_work_dir,
        )

        for model in self.models:
            builder.add(
                key=model.tag,
                model_instance=model.get_model_instance(),
                example_inputs=model.input_generator(),
                compiler_args=model.compiler_args,
                bucket_config=model.bucket_config,
                priority_model_idx=model.priority_model_idx,
            )

        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, compiled_model_path + COMPILED_MODEL_FILE_NAME)
        del traced_model

        sharded_checkpoint_dir = os.path.join(compiled_model_path, "weights/")
        builder.shard_checkpoint(serialize_path=sharded_checkpoint_dir)

        if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
            builder.transform_weight_layout_with_overriden_option(
                sharded_checkpoint_dir=sharded_checkpoint_dir
            )

        self.is_compiled = True
        self.is_loaded_to_neuron = True

    def load(self, compiled_model_path):
        """Loads the compiled model checkpoint to the Neuron device."""
        self.traced_model = torch.jit.load(compiled_model_path + COMPILED_MODEL_FILE_NAME)

        self.load_weights(compiled_model_path)
        if self.neuron_config.torch_dtype == torch.bfloat16:
            self.bfloat16()

        for model_wrapper in self.models:
            model_wrapper.model = self.traced_model
        self.is_loaded_to_neuron = True

    def load_weights(self, compiled_model_path):
        """Loads the model weights to the Neuron device."""
        if self.traced_model is None:
            raise ValueError("Model is not loaded")

        weights = []
        for rank in range(self.neuron_config.start_rank_id, self.neuron_config.start_rank_id+self.neuron_config.local_ranks_size):
            ckpt = load_file(
                os.path.join(
                    compiled_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"
                )
            )
            weights.append(ckpt)
        self.traced_model.nxd_model.initialize(weights)

    def checkpoint_loader_fn(self, mmap: bool = False):
        """This function loads the model's state dictionary and weights from the hf model"""

        if self.config.neuron_config.quantized:
            return self.get_quantized_state_dict(self.config)
        else:
            model_sd = self.get_state_dict(self.model_path, self.config)
            if self.neuron_config.torch_dtype == torch.bfloat16:
                for name, param in model_sd.items():
                    model_sd[name] = param.bfloat16()
            return model_sd

    @classmethod
    def get_state_dict(cls, model_path: str, config: InferenceConfig) -> dict:
        """Gets the state dict for this model."""
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        if os.path.exists(model_path + "/medusa_heads.pt"):
            medusa_head = torch.load(model_path + "/medusa_heads.pt", map_location="cpu")
            model_sd.update(medusa_head)
        model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)
        return model_sd

    @classmethod
    def get_quantized_state_dict(cls, config: InferenceConfig, mmap: bool = False) -> dict:
        """
        This function loads the checkpointed float model state dictionary and weights from the quantized hf model
        This will be removed once we move to safe tensors in NxD
        """
        existing_checkpoint_path = config.neuron_config.quantized_checkpoints_path
        if not os.path.exists(existing_checkpoint_path):
            raise FileNotFoundError(
                f"Quantized checkpoint file not found: {existing_checkpoint_path}"
            )

        print(f"Using existing checkpoint: {existing_checkpoint_path}")
        model_quant_sd = torch.load(existing_checkpoint_path)
        model_quant_sd = cls.convert_hf_to_neuron_state_dict(model_quant_sd, config)

        # Make sure that the non quantized weights are in bfloat16 and not float32
        if config.neuron_config.torch_dtype == torch.bfloat16:
            for name, param in model_quant_sd.items():
                # TODO: Reduce and clean-up these warnings
                if param is not None and param.dtype == torch.float32:
                    if name.endswith(".scale"):
                        warnings.warn(
                            f"Found float32 weights in quantized checkpoint: {name}. Will skip converting to bfloat16 as its scale"
                        )
                    else:
                        warnings.warn(
                            f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16"
                        )
                        model_quant_sd[name] = param.bfloat16()

        return model_quant_sd

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        return state_dict

    @classmethod
    def save_quantized_state_dict(cls, model_path: str, config: InferenceConfig):
        """
        Quantizes the model and saves the quantized checkpoint to `config.neuron_config.quantized_checkpoints_path`.
        """
        quantized_state_dict = cls.generate_quantized_state_dict(model_path, config)

        # Prune None values in the quantized_state_dict. torch.save crashes if None values exist.
        quantized_state_dict = prune_state_dict(quantized_state_dict)
        torch.save(quantized_state_dict, config.neuron_config.quantized_checkpoints_path)

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, config: InferenceConfig) -> dict:
        """Generates the quantized state dict for this model."""
        hf_model = cls.load_hf_model(model_path)
        quantization_type = QuantizationType(config.neuron_config.quantization_type)
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(
                float_model=hf_model, inplace=True
            )
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_channel_symmetric(
                float_model=hf_model, inplace=True
            )
        else:
            raise RuntimeError(f"{config.neuron_config.quantization_type} not supported")

        return cls.prepare_quantized_state_dict(hf_model_quant)

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant) -> dict:
        """Can be overriden to customize the quantized state dict in generate_quantized_state_dict."""
        model_quant_sd = hf_model_quant.model.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        return model_quant_sd

    @staticmethod
    def load_hf_model(model_path):
        """Loads the HuggingFace model from the given checkpoint path."""
        raise NotImplementedError("load_hf_model is not implemented")

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def reset(self):
        """Resets the model state. Can be implemented by subclasses."""
        pass
