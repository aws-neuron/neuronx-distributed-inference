"""
MFM(Module from Model) adapters are subclass of base adapters that
defines the define_module_cls and get_state_dict out-of-box for users.

They work by extracting individual modules (e.g., attention, MLP) and their
corresponding weights from the full NeuronApplicationBase model. Use when:
- Your test modules are part of the NeuronApplicationBase model
- Your NeuronApplicationBase model has get_state_dict() implemented
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional, List, Type

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from neuronx_distributed_inference.utils.testing import destroy_cpu_env, init_cpu_env
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.utils.model_utils import init_on_device
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG

from neuronx_distributed_inference.module_test.base_template.adapter_base import (
    HFAdapterBase,
    NxDINeuronAdapterBase,
    NxDISingleRankCPUAdapterBase,
)


class MFMHFAdapter(HFAdapterBase):
    def __init__(
        self,
        forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], Any],
        complete_model_cls: Type[PreTrainedModel],
        module_names: List[str],
        layer_id: int = 0,
        prefixes: Optional[List[str]] = None,
    ):
        """
        Args:
            forward_fn: User-defined forward function that will be patched to the module cls
            complete_model_cls: The complete HuggingFace model class (e.g., LlamaForCausalLM).
            module_names: List of module names to extract (e.g., ["attn", "mlp"]).
                                These should be the actual module attribute names in the complete model.
            layer_id: The layer index to extract modules from (default: 0).
                    Used to build the full module path like "{module_prefix}.{layer_id}.{module_name}".
            prefixes: Optional custom prefixes for each module. If None, uses "model.layers" for all.
                     Useful when modules have different parent paths in the model hierarchy.
        """
        super().__init__()
        self.forward_fn = forward_fn
        self.complete_model_cls = complete_model_cls

        # prefixes_map: module_name -> its prefix in the complete model
        self.prefixes_map = build_prefixes_map(
            prefixes=prefixes,
            needed_module_names=module_names,
            layer_id=layer_id,
            default_prefix="model.layers",
        )

    def define_module_cls(self):
        self.logger.info(
            f"[{self.__class__.__name__}] Defining module cls by extracting modules from the complete HF model on DRAM."
        )

        # to define module this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        # get all modules from the complete model and extract needed ones from it
        all_modules = dict(
            self.complete_model_cls.from_pretrained(self.hf_ckpt_path).named_modules()
        )
        sub = extract_submodules_by_prefixes(self.prefixes_map, all_modules)

        class Partial(nn.Module):
            def __init__(self):
                super().__init__()
                for k, m in sub.items():
                    setattr(self, k, m)

        self.module_cls = Partial
        self._patch_forward()
        self.logger.info(f"[{self.__class__.__name__}] Module cls defined successfully.")

    def _patch_forward(self):
        """Patch module forward with user-defined function."""
        assert self.module_cls
        self.module_cls.forward = self.forward_fn

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        self.logger.info(
            f"[{self.__class__.__name__}] Extracting weights for target modules from the complete HF model weight dict."
        )

        # to get state dict this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        # get full state dict of the complete model and extract needed ones from it
        full_sd = self.complete_model_cls.from_pretrained(self.hf_ckpt_path).state_dict()
        out = extract_subweights_by_prefixes(self.prefixes_map, full_sd)

        self.logger.info(f"[{self.__class__.__name__}] Weights extracted successfully.")
        return out


class MFMNxDICPUSingleRankAdapter(NxDISingleRankCPUAdapterBase):
    def __init__(
        self,
        forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], Any],
        complete_model_cls: Type[NeuronApplicationBase],
        complete_model_config: InferenceConfig,
        needed_module_names: List[str],
        layer_id: int = 0,
        prefixes: Optional[List[str]] = None,
        model_tag=CONTEXT_ENCODING_MODEL_TAG,
    ):
        """
        Args:
            forward_fn: User-defined forward function that will be patched to to the module cls.
            complete_model_cls: The complete NeuronApplicationBase model class.
            complete_model_config: InferenceConfig for the complete NeuronApplicationBase model cls.
            needed_module_names: List of module names to extract (e.g., ["attn", "mlp"]).
                                These should be the actual module attribute names in the complete model cls.
            layer_id: The layer index to extract modules from (default: 0).
                     Used to build the full module path like "{module_prefix}.{layer_id}.{module_name}".
            prefixes: Optional custom prefixes for each module. If None, uses "layers" for all.
                     Useful when modules have different parent paths in the model hierarchy.
            model_tag: Model tag identifier (default: CONTEXT_ENCODING_MODEL_TAG).
                      Used to select which model from the NeuronApplicationBase to extract from.
        """
        super().__init__()
        self.complete_model_config = complete_model_config
        self.forward_fn = forward_fn
        self.complete_model_cls = complete_model_cls
        self.model_tag = model_tag

        # prefixes_map: module_name -> its prefix in the complete model
        self.prefixes_map = build_prefixes_map(
            prefixes=prefixes,
            needed_module_names=needed_module_names,
            layer_id=layer_id,
            default_prefix="layers",
        )

    def define_module_cls(self):
        self.logger.info(
            f"[{self.__class__.__name__}] Defining module cls by extracting modules from the complete NxDI model on meta device."
        )

        # to define module this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        init_cpu_env()
        app = self.complete_model_cls(self.hf_ckpt_path, self.complete_model_config)
        builder = app.get_builder()
        container = builder.model_collection[self.model_tag]

        # NOTE :Uses meta device to load the complete model to prevent memory overhead.
        # This only works for NxDI, not HF from_pretrained() function.
        with init_on_device(torch.device("meta"), force_custom_init_on_device=True):
            container.model_instance.load_module()
        destroy_cpu_env()

        # get all modules from the complete model and extract needed ones from it
        all_modules = dict(container.model_instance.module.named_modules())
        sub = extract_submodules_by_prefixes(self.prefixes_map, all_modules)

        class Partial(nn.Module):
            def __init__(self):
                super().__init__()
                logger = logging.getLogger("Neuron")
                logger.info(f"[{self.__class__.__name__}] Converting needed modules to CPU.")
                for k, m in sub.items():
                    setattr(self, k, m.to_empty(device="cpu"))

        self.module_cls = Partial
        self._patch_forward()
        self.logger.info(f"[{self.__class__.__name__}] Module cls defined successfully.")

    def _patch_forward(self):
        """Patch module cls forward with user-defined function."""
        assert self.module_cls
        self.module_cls.forward = self.forward_fn

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        self.logger.info(
            f"[{self.__class__.__name__}] Extracting weights for target modules from the NxDI weight dict."
        )

        # to get state dict this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        # get full state dict of the complete model and extract needed ones from it
        full = self.complete_model_cls.get_state_dict(self.hf_ckpt_path, self.complete_model_config)
        out = extract_subweights_by_prefixes(self.prefixes_map, full)

        self.logger.info(f"[{self.__class__.__name__}] Weights extracted successfully.")
        return out


class MFMNxDINeuronAdapter(NxDINeuronAdapterBase):
    def __init__(
        self,
        forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], Any],
        complete_model_cls: Type[NeuronApplicationBase],
        complete_model_config: InferenceConfig,
        needed_module_names: List[str],
        layer_id: int = 0,
        prefixes: Optional[List[str]] = None,
        model_tag=CONTEXT_ENCODING_MODEL_TAG,
    ):
        """
        Args:
            forward_fn: User-defined forward function that will be patched to to the module cls.
            complete_model_cls: The complete NeuronApplicationBase model class.
            complete_model_config: InferenceConfig for the complete NeuronApplicationBase model cls.
            needed_module_names: List of module names to extract (e.g., ["attn", "mlp"]).
                                These should be the actual module attribute names in the complete model cls.
            layer_id: The layer index to extract modules from (default: 0).
                     Used to build the full module path like "{module_prefix}.{layer_id}.{module_name}".
            prefixes: Optional custom prefixes for each module. If None, uses "layers" for all.
                     Useful when modules have different parent paths in the model hierarchy.
            model_tag: Model tag identifier (default: CONTEXT_ENCODING_MODEL_TAG).
                      Used to select which model from the NeuronApplicationBase to extract from.
        """
        super().__init__(
            complete_model_config.neuron_config.tp_degree,
            complete_model_config.neuron_config.world_size,
        )
        self.forward_fn = forward_fn
        self.complete_model_cls = complete_model_cls
        self.complete_model_config = complete_model_config

        # prefixes_map: module_name -> its prefix in the complete model
        self.prefixes_map = build_prefixes_map(
            prefixes=prefixes,
            needed_module_names=needed_module_names,
            layer_id=layer_id,
            default_prefix="layers",
        )
        self.model_tag = model_tag

    def define_module_cls(self):
        self.logger.info(
            f"[{self.__class__.__name__}] Defining module cls by extracting modules from the complete NxDI model on meta device."
        )

        # to define module this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        app = self.complete_model_cls(self.hf_ckpt_path, self.complete_model_config)
        builder = app.get_builder()

        # NOTE :Uses meta device to load the complete model to prevent memory overhead.
        # This only works for NxDI, not HF from_pretrained() function.
        with mock_distributed(builder.world_size), init_on_device(
            torch.device("meta"), force_custom_init_on_device=True
        ):
            torch.distributed.init_process_group("xla", rank=0, world_size=builder.world_size)
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=builder.tp_degree,
                pipeline_model_parallel_size=builder.pp_degree,
                expert_model_parallel_size=builder.ep_degree,
                skip_collective_init=True,
            )
            container = builder.model_collection[self.model_tag]
            container.model_instance.load_module()
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

        # get all modules from the complete model and extract needed ones from it
        all_modules = dict(container.model_instance.module.named_modules())
        sub = extract_submodules_by_prefixes(self.prefixes_map, all_modules)

        # CPU version for compilation
        class Partial(nn.Module):
            def __init__(self):
                super().__init__()
                for k, m in sub.items():
                    setattr(self, k, m.to_empty(device="cpu"))

            def forward(self, *a, **kw):
                pass

        # weight sharding requires modules on Meta
        class PartialMeta(nn.Module):
            def __init__(self):
                super().__init__()
                for k, m in sub.items():
                    setattr(self, k, m)

            def forward(self, *a, **kw):
                pass

        self.module_cls = Partial
        self.meta_module_cls = PartialMeta
        self._patch_forward()
        self.logger.info(f"[{self.__class__.__name__}] Module cls defined successfully.")

    def _patch_forward(self):
        assert self.module_cls
        self.module_cls.forward = self.forward_fn  # type: ignore
        self.meta_module_cls.forward = self.forward_fn

    def get_state_dict(self):
        """Extract weights for target modules from specified layer in NxDI weight dict."""
        self.logger.info(
            f"[{self.__class__.__name__}] Extracting weights for target modules from the NxDI weight dict."
        )

        # to get state dict this way, the orchestrator has to set hf_ckpt_path first
        assert self.hf_ckpt_path

        # get full state dict of the complete model and extract needed ones from it
        full = self.complete_model_cls.get_state_dict(self.hf_ckpt_path, self.complete_model_config)
        out = extract_subweights_by_prefixes(self.prefixes_map, full)

        self.logger.info(f"[{self.__class__.__name__}] Weights extracted successfully.")
        return out


# ===========================================================
#          Utility Funcionts
# ===========================================================


def build_prefixes_map(
    prefixes: Optional[List[str]],
    needed_module_names: List[str],
    layer_id: int,
    default_prefix: str,
) -> Dict[str, str]:
    """
    Build a map of module names to their prefixes with testing layer_id appended.

    Args:
        prefixes: Optional list of prefixes (one per module). If None, uses default_prefix for all.
        needed_module_names: List of module names
        layer_id: Layer ID to append to each prefix
        default_prefix: Default prefix to use when prefixes is None

    Returns:
        Dictionary mapping module name to prefix (e.g., {"attn": "model.layers.0", "kv_cache_mgr": "model"})
    """
    if prefixes is None:
        # Use default prefix for all needed modules
        prefixes = [default_prefix] * len(needed_module_names)
    else:
        if len(prefixes) != len(needed_module_names):
            raise ValueError(
                f"prefixes must have the same length as needed_module_names "
                f"({len(prefixes)} vs {len(needed_module_names)})"
            )

    # Build map of module name to prefix with layer_id, omitting layer_id when prefix is empty
    return {
        name: f"{p}.{layer_id}" if "layer" in p else p
        for name, p in zip(needed_module_names, prefixes)
    }


def extract_submodules_by_prefixes(
    prefixes_map: Dict[str, str],
    all_modules: Dict[str, nn.Module],
) -> Dict[str, nn.Module]:
    """
    Extract specific modules from a layer using a prefix map.

    Args:
        prefixes_map: Dictionary mapping needed module name to prefix (e.g., {"attn": "model.layers.0"})
        all_modules: Dictionary of all available modules from the complete model

    Returns:
        Dictionary mapping needed module names to extracted module instances
    """
    out = {}
    for name, pref in prefixes_map.items():
        module_path = f"{pref}.{name}"
        out[name] = all_modules[module_path]
    return out


def extract_subweights_by_prefixes(
    prefixes_map: Dict[str, str],
    full_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Extract weights for needed modules from a full state dict using a prefix map.

    Args:
        prefixes_map: Dictionary mapping module name to prefix (e.g., {"attn": "model.layers.0"})
        full_sd: Full state dict containing model weights of the complete model

    Returns:
        Dictionary containing only the weights for the specified modules with prefix stripped
    """
    out = {}
    for name, pref in prefixes_map.items():
        for k, v in full_sd.items():
            if k.startswith(f"{pref}.{name}"):
                out[k.replace(f"{pref}.", "")] = v
    return out
