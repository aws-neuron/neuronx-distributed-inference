"""
Base adapter class for running a test on a specific backend(HF, NxDINeuron).

Adapters must define the following 4-steps:
1. define_module_cls() - Define module class and forward function
2. instantiate_module() - Initialize module with weights loaded
3. load_kv_cache() - [Optional] Load HF format KV cache to the specific backend for testing
4. run_inference() - Execute forward pass on the backend and return outputs
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type

from neuronx_distributed_inference.utils.testing import destroy_cpu_env, init_cpu_env
import torch
import torch.nn as nn

from neuronx_distributed.trace.parallel_context import NxDParallelState
from neuronx_distributed.trace.model_builder_v2 import ModelBuilder as ModelBuilderV2
from neuronx_distributed.trace.functions import shard_checkpoint
from neuronx_distributed.utils.model_utils import init_on_device
import logging


# ===========================================================
#                   BASE ADAPTER CLASS
# ===========================================================


class ModuleAdapterBase(ABC):
    """
    Abstract base for all test adapters.
    Defines the 4-step testing workflow.
    """

    def __init__(self):
        self.module_cls: Optional[Type[nn.Module]] = None
        self.module: Optional[nn.Module] = None
        self.logger = logging.getLogger("Neuron")
        self.torch_dtype: torch.dtype = torch.float16
        self.hf_ckpt_path: str = None

    # --------------------------------------------------
    # REQUIRED ADAPTER STEPS
    # --------------------------------------------------
    @abstractmethod
    def define_module_cls(self, *args, **kwargs):
        """
        Step 1: Define module class and its forward function.
        Sets self.module_cls.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.define_module_cls() must be implemented by subclasses."
        )

    @abstractmethod
    def instantiate_module(self, *args, **kwargs):
        """
        Step 2: Initialize module and load weights from checkpoint.
        Sets self.module.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.instantiate_module() must be implemented by subclasses."
        )

    def load_kv_cache(self, hf_kv_cache: Any):
        """
        Step 3 [Optional]: Load KV cache of hf format into module for testing.
        Override if module uses KV cache.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.load_kv_cache() is not implemented.")

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        """
        Step 4: Execute forward pass through the module.
        Returns module outputs for comparison.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.run_inference() must be implemented by subclasses."
        )

    # --------------------------------------------------
    # helper functions
    # --------------------------------------------------
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract state dict from self.hf_ckpt_path. This function is commonly used by instantiate_module. Override and implement it if you need it."""
        raise NotImplementedError(f"{self.__class__.__name__}.get_state_dict() is not implemented.")

    def set_torch_dtype(self, torch_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        self.torch_dtype = torch_dtype

    def set_hf_ckpt_path(self, hf_ckpt_path: str):
        self.hf_ckpt_path = hf_ckpt_path

    def free_resources(self):
        """
        Free adapter resources (module_cls and module) to release memory.
        Call this after adapter processing is complete.
        """
        if self.module is not None:
            del self.module
            self.module = None
        if self.module_cls is not None:
            del self.module_cls
            self.module_cls = None
        # Force garbage collection to ensure memory is released
        import gc
        gc.collect()


# ===========================================================
#                    HF ADAPTER
# ===========================================================


class HFAdapterBase(ModuleAdapterBase):
    def instantiate_module(self):
        state_dict = self.get_state_dict()
        assert self.module_cls is not None
        m = self.module_cls()
        m.load_state_dict(state_dict)
        self.module = m.to(self.torch_dtype)

    def run_inference(self, *args, **kwargs):
        assert self.module is not None
        self.module.eval()
        with torch.no_grad():
            return self.module(*args, **kwargs)


# ===========================================================
#               NxDI CPU (Single Rank) ADAPTER
# ===========================================================


class NxDISingleRankCPUAdapterBase(ModuleAdapterBase):
    """
    Adapter for NxDI modules on CPU (single rank).
    """

    def instantiate_module(self):
        state_dict = self.get_state_dict()
        assert self.module_cls is not None

        init_cpu_env()
        m = self.module_cls().to(self.torch_dtype)
        from neuronx_distributed.trace import trace

        trace.preprocess_checkpoint(m, state_dict)
        m.load_state_dict(state_dict, strict=False)
        destroy_cpu_env()

        self.module = m

    def run_inference(self, *args, **kwargs):
        init_cpu_env()
        with torch.no_grad():
            output = self.module(*args, **kwargs)
        destroy_cpu_env()
        return output


# ===========================================================
#                    NxDI NEURON ADAPTER
# ===========================================================


class NxDINeuronAdapterBase(ModuleAdapterBase):
    """
    Adapter for NxDI modules on Neuron hardware.
    """

    def __init__(self, tp_degree: int, world_size: int):
        super().__init__()
        self.meta_module_cls: Type[nn.Module] = None
        self.tp = tp_degree
        self.ws = world_size

    def instantiate_module(self, example_inputs):
        state_dict = self.get_state_dict()
        assert self.module_cls and self.meta_module_cls

        # Create sharded checkpoint on meta device
        with NxDParallelState(
            world_size=self.ws, tensor_model_parallel_size=self.tp
        ), init_on_device(torch.device("meta")):
            sharded = shard_checkpoint(state_dict, self.meta_module_cls())
        # Trace, compile, and load weights to Neuron
        with NxDParallelState(world_size=self.ws, tensor_model_parallel_size=self.tp):
            m = self.module_cls().to(self.torch_dtype)
            nm = ModelBuilderV2(m).trace(args=example_inputs).compile()
            nm.set_weights(sharded)
            nm.to_neuron()

        self.module = nm

    def run_inference(self, *args, **kwargs):
        assert self.module
        with NxDParallelState(
            world_size=self.ws,  # type: ignore
            tensor_model_parallel_size=self.tp,  # type: ignore
        ):
            return self.module(*args, **kwargs)
