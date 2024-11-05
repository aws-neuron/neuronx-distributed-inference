import logging
import os
import warnings
from functools import partial
from typing import List

import neuronx_distributed.trace.hlo_utils as hlo_utils
import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import load_file

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    prune_state_dict,
    save_state_dict_safetensors,
)

COMPILED_MODEL_FILE_NAME = "model.pt"
logger = logging.getLogger("Neuron")


def is_compiled(model_path):
    return os.path.isfile(model_path + COMPILED_MODEL_FILE_NAME)


def init_custom_process_group_fn(config):
    if hasattr(config, "fused_spec_config") and config.fused_spec_config is not None:
        if config.fused_spec_config.draft_config.neuron_config.tp_degree is not None:
            draft_tp = config.fused_spec_config.draft_config.neuron_config.tp_degree
            parallel_state.initialize_speculative_draft_group(draft_tp)


class NeuronApplicationBase(torch.nn.Module):
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""
    _FUSED_PREFIX = ""

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig = None,
        neuron_config: NeuronConfig = None,
    ):
        super().__init__()
        if config is None:
            config = self.get_config_cls().load(model_path)

        if neuron_config is not None:
            config.neuron_config = neuron_config

        self.validate_config(config)
        self.config = config
        self.neuron_config = config.neuron_config
        self.fused_spec_config = config.fused_spec_config
        self.on_device_sampling = self.neuron_config.on_device_sampling_config is not None
        self.model_path = model_path
        self.models: List[ModelWrapper] = []
        self.traced_model = None
        self.is_compiled = is_compiled(model_path)
        self.is_loaded_to_neuron = False
        self._builder = None

    def get_builder(self, debug=False):
        if self._builder is None:
            base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

            # Use this function to intialize non standard TP/PP/DP distributed
            # process groups.
            custom_group_fn = partial(init_custom_process_group_fn, self.config)

            self._builder = ModelBuilder(
                router=None,
                tp_degree=self.neuron_config.tp_degree,
                pp_degree=self.neuron_config.pp_degree,
                ep_degree=self.neuron_config.ep_degree,
                world_size=self.neuron_config.world_size,
                start_rank_id=self.neuron_config.start_rank_id,
                local_ranks_size=self.neuron_config.local_ranks_size,
                checkpoint_loader=self.checkpoint_loader_fn,
                compiler_workdir=base_compile_work_dir,
                debug=debug,
                num_cores_per_group=self.config.num_cores_per_group,
                init_custom_process_group_fn=custom_group_fn,
                logical_neuron_cores=self.neuron_config.logical_neuron_cores,
                weights_to_skip_layout_optimization=self.config.neuron_config.weights_to_skip_layout_optimization,
            )
            for model in self.models:
                self._builder.add(
                    key=model.tag,
                    model_instance=model.get_model_instance(),
                    example_inputs=model.input_generator(),
                    compiler_args=model.compiler_args,
                    bucket_config=model.bucket_config,
                    priority_model_idx=model.priority_model_idx,
                )
        return self._builder

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

    def compile(self, compiled_model_path, debug=False):
        """Compiles this model and saves it to the given path."""
        self.config.save(compiled_model_path)

        traced_model = self.get_builder(debug).trace(initialize_model_weights=False)
        torch.jit.save(traced_model, compiled_model_path + COMPILED_MODEL_FILE_NAME)
        del traced_model

        sharded_checkpoint_dir = os.path.join(compiled_model_path, "weights/")
        if not self.neuron_config.save_sharded_checkpoint:
            logger.info(
                f"SKIPPING sharding the checkpoint at {sharded_checkpoint_dir}, assuming it's already sharded"
            )
        else:
            self.get_builder(debug).shard_checkpoint(serialize_path=sharded_checkpoint_dir)

            if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
                self.get_builder(debug).transform_weight_layout_with_overriden_option(
                    sharded_checkpoint_dir=sharded_checkpoint_dir
                )
        self.is_compiled = True

    def load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None):
        """Loads the compiled model checkpoint to the Neuron device."""
        self.traced_model = torch.jit.load(compiled_model_path + COMPILED_MODEL_FILE_NAME)

        self.load_weights(
            compiled_model_path, start_rank_id=start_rank_id, local_ranks_size=local_ranks_size
        )

        if self.neuron_config.torch_dtype != torch.float32:
            self.to(self.neuron_config.torch_dtype)

        for model_wrapper in self.models:
            model_wrapper.model = self.traced_model
        self.is_loaded_to_neuron = True

    def load_weights(self, compiled_model_path, start_rank_id=None, local_ranks_size=None):
        """Loads the model weights to the Neuron device."""
        if self.traced_model is None:
            raise ValueError("Model is not loaded")

        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        logging.info(
            f"loading models for ranks {start_rank_id}...{start_rank_id + local_ranks_size - 1}"
        )
        weights = []
        for rank in range(start_rank_id, start_rank_id + local_ranks_size):
            if self.neuron_config.save_sharded_checkpoint:
                ckpt = load_file(
                    os.path.join(
                        compiled_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"
                    )
                )
            else:
                print(f"There are no saved sharded checkpoints. Sharding and loading rank {rank}")
                source_model_key = list(self.get_builder().model_collection.keys())[0]
                ckpt = self.get_builder().shard_weights(
                    rank, self.get_builder().model_collection[source_model_key]
                )
            weights.append(ckpt)
        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.traced_model.nxd_model.initialize(weights, start_rank_tensor)

    def checkpoint_loader_fn(self, mmap: bool = False):
        """This function loads the model's state dictionary and weights from the hf model"""

        if self.config.neuron_config.quantized:
            return self.get_quantized_state_dict(self.config)
        elif self.config.neuron_config.enable_fused_speculation:
            assert self.fused_spec_config is not None
            self.__class__._FUSED_PREFIX = "draft_model"
            model_sd = self.get_state_dict(
                self.fused_spec_config.draft_model_path, self.fused_spec_config.draft_config
            )
            self.__class__._FUSED_PREFIX = "target_model"
            model_sd.update(self.get_state_dict(self.model_path, self.config))

            if self.neuron_config.torch_dtype != torch.float32:
                for name, param in model_sd.items():
                    if torch.is_floating_point(param):
                        model_sd[name] = param.to(self.neuron_config.torch_dtype)
            return model_sd
        else:
            model_sd = self.get_state_dict(self.model_path, self.config)
            if self.neuron_config.torch_dtype != torch.float32:
                for name, param in model_sd.items():
                    if torch.is_floating_point(param) and param.dtype not in [torch.float8_e4m3fn]:
                        # only cast floating types
                        model_sd[name] = param.to(self.neuron_config.torch_dtype)
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
        if getattr(config, "tie_word_embeddings", False):
            cls.update_state_dict_for_tied_weights(model_sd)

        param_name_list = list(model_sd.keys())
        if cls._FUSED_PREFIX != "":
            for param_name in param_name_list:
                model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                del model_sd[param_name]
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
        if os.path.isdir(existing_checkpoint_path):
            model_quant_sd = load_state_dict(existing_checkpoint_path)
        else:
            model_quant_sd = torch.load(existing_checkpoint_path)

        # For the case when huggingface models come with existing prefixes. We do not allow
        # Any prefix like "model."
        param_name_list = list(model_quant_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
                model_quant_sd[updated_param_name] = model_quant_sd[param_name]
                del model_quant_sd[param_name]

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
        if os.path.isdir(config.neuron_config.quantized_checkpoints_path):
            logging.info(
                "Saving quantized state dict as safetensors to: %s",
                config.neuron_config.quantized_checkpoints_path,
            )
            save_state_dict_safetensors(
                state_dict=quantized_state_dict,
                state_dict_dir=config.neuron_config.quantized_checkpoints_path,
            )
        else:
            logging.info(
                "Saving quantized state dict as torch pt file to: %s",
                config.neuron_config.quantized_checkpoints_path,
            )
            torch.save(quantized_state_dict, config.neuron_config.quantized_checkpoints_path)

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, config: InferenceConfig) -> dict:
        """Generates the quantized state dict for this model."""
        hf_model = cls.load_hf_model(model_path)
        quantization_type = QuantizationType(config.neuron_config.quantization_type)
        quantized_dtype = QuantizedDtype.get_dtype(config.neuron_config.quantization_dtype)
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(
                float_model=hf_model, inplace=True, dtype=quantized_dtype
            )
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_channel_symmetric(
                float_model=hf_model, inplace=True, dtype=quantized_dtype
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

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Implement state_dict update for each model class with tied weights"""
        raise NotImplementedError("State-dict update not implemented")

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
