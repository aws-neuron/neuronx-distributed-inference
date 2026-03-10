"""
Base orchestrator for module testing.
Orchestrators coordinate checkpoint preparation, input generation,
inference execution, and output validation to ensure consistency across backends (HuggingFace, NxDI CPU, NxDI Neuron).
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, List, Dict
import logging

from neuronx_distributed_inference.module_test.base_template.adapter_base import (
    NxDINeuronAdapterBase,
    ModuleAdapterBase,
)
import torch

from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings


# ===========================================================
#               TEST ORCHESTRATOR
# ===========================================================
@dataclass(kw_only=True)
class OrchestratorBaseConfig:
    batch_size: int  # batch size of the testing input
    torch_dtype: torch.dtype
    hf_weight_ckpt_path: str
    # tolerance for accuracy validation,
    # if None, use default from neuronx_distributed_inference.utils.accuracy
    atol: Optional[float] = None
    rtol: Optional[float] = None

    # -----------------------------
    # schemas for prep random input/kv_cache
    # NOTE: we use random input/cache generation for now, hence these classes.
    # We'll use real input/cache in the future and deprecate them.
    # -----------------------------
    @dataclass
    class PrepInputConfig:
        """Configuration for preparing test input tensors."""

        seq_len: int  # usually 1 for generation and ctx_len for encoding
        hf_hidden_size: int
        hidden_states_mean: float = 0
        hidden_states_std: float = 1

    @dataclass
    class PrepKVCacheConfig:
        """Configuration for preparing KV cache tensors."""

        ctx_len: int
        num_head: int
        hf_head_hidden_size: int
        kv_cache_mean: float = 0
        kv_cache_std: float = 1

    prep_input_config: PrepInputConfig
    prep_kv_cache_config: Optional[PrepKVCacheConfig] = None  # optional generate random KV cache


class OrchestratorBase:
    """
    Orchestrates module testing across multiple adapters (e.g. HF, NxDICPU, NxDINeuron).
    Generates consistent ckpt/inputs/KV caches and validates outputs match across adapters.
    """

    def __init__(
        self,
        adapters: List[ModuleAdapterBase],
        orchestratorConf: OrchestratorBaseConfig,
    ):
        self.adapters = adapters
        self.conf = orchestratorConf
        for adp in adapters:
            adp.set_torch_dtype(
                self.conf.torch_dtype
            )  # After initializing modules, all the adapter will convert module to this torch_dtype.
            # TODO: figure out how MX are dealt with.
            adp.set_hf_ckpt_path(self.conf.hf_weight_ckpt_path)

    def prepare_input_hf_format(self) -> dict[str, torch.Tensor]:
        """
        Generate input tensors for inference.
        For now, we use random input generation.
        """
        shape = (
            self.conf.batch_size,
            self.conf.prep_input_config.seq_len,
            self.conf.prep_input_config.hf_hidden_size,
        )

        # Get custom mean and std if specified, otherwise use defaults
        mean = (
            self.conf.prep_input_config.hidden_states_mean
            if self.conf.prep_input_config.hidden_states_mean is not None
            else 0.0
        )
        std = (
            self.conf.prep_input_config.hidden_states_std
            if self.conf.prep_input_config.hidden_states_std is not None
            else 1.0
        )

        # Generate normal distribution with specified mean and std
        x = torch.randn(shape, dtype=self.conf.torch_dtype) * std + mean

        input = {"hidden_states": x}
        return input

    def prepare_kv_cache_hf_format(self) -> torch.Tensor:
        """
        Generate KV cache tensors for testing.
        For now, we use random kv cache generation
        TODO: Fill this function
        """
        raise NotImplementedError(
            "prepare_kv_cache_hf_format is not implemented yet by the template."
        )

    @staticmethod
    def validate_result(
        adp_name_to_result: Dict[str, torch.Tensor],
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        """
        Compare outputs from all adapters pairwise.
        Fails if any pair exceeds tolerance thresholds.

        Args:
            adp_name_to_result: Map from adapter name to its output tensor
            atol: Absolute tolerance for accuracy comparison
            rtol: Relative tolerance for accuracy comparison
        """
        tolerance = {k: v for k, v in (("atol", atol), ("rtol", rtol)) if v is not None}

        # Get list of adapter names and their results
        adapter_names = list(adp_name_to_result.keys())

        # Compare all pairs of results
        for adp_name_a, adp_name_b in combinations(adapter_names, 2):
            out_a = adp_name_to_result[adp_name_a]
            out_b = adp_name_to_result[adp_name_b]

            passed, err = check_accuracy_embeddings(out_a, out_b, plot_outputs=True, **tolerance)

            if not passed:
                raise AssertionError(
                    f"[Mismatch] {adp_name_a} vs {adp_name_b}\n"
                    f"Error: {err}\n"
                    f"See plot.png for detailed comparison."
                )

    def run_validation(self):
        """
        Execute full validation workflow across all adapters:
        1. Define module classes
        2. Instantiate modules and load weights
        3. Load KV cache (if configured)
        4. Run inference with same inputs
        5. Free adapter resources to save memory
        6. Validate outputs match across adapters
        """
        logger = logging.getLogger("Neuron")
        logger.info(f"[{self.__class__.__name__}] Starting validation workflow")

        # Prepare inputs once (shared across all adapters)
        inputs = self.prepare_input_hf_format()
        example_inputs = tuple(inputs.values())

        # Prepare KV cache once if configured (shared across all adapters)
        kv_cache = None
        if self.conf.prep_kv_cache_config:
            logger.info(f"[{self.__class__.__name__}] Preparing KV cache")
            kv_cache = self.prepare_kv_cache_hf_format()
            logger.info(f"[{self.__class__.__name__}] KV cache prepared successfully")

        # Process each adapter sequentially to save memory
        adp_name_to_result: Dict[str, torch.Tensor] = {}
        for idx, adp in enumerate(self.adapters, 1):
            adp_name = adp.__class__.__name__
            logger.info(
                f"[{self.__class__.__name__}] Processing adapter {idx}/{len(self.adapters)}: {adp_name}"
            )

            # Step 1: Define module class for this adapter
            adp.define_module_cls()

            # Step 2: Instantiate module and load weights
            if isinstance(adp, NxDINeuronAdapterBase):
                adp.instantiate_module(example_inputs)
            else:
                adp.instantiate_module()

            # Step 2.5 (Optional): Load KV cache if configured
            if kv_cache is not None:
                adp.load_kv_cache(kv_cache)

            # Step 3: Run inference
            output = adp.run_inference(**inputs)

            # Store result in map
            adp_name_to_result[adp_name] = output

            # Step 4: Free adapter resources to save memory
            logger.info(f"[{self.__class__.__name__}] Freeing resources for {adp_name}")
            adp.free_resources()

            logger.info(f"[{self.__class__.__name__}] Completed processing {adp_name}")

        logger.info(f"[{self.__class__.__name__}] All adapters processed successfully")

        # Step 5: Validate all outputs match
        logger.info(f"[{self.__class__.__name__}] Validating outputs match across adapters")
        self.validate_result(adp_name_to_result, atol=self.conf.atol, rtol=self.conf.rtol)
        logger.info(f"[{self.__class__.__name__}] Validation successful - all outputs match!")
