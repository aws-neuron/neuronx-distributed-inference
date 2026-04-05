"""
Module From Model (MFM) orchestrator base for module testing.
Extends base orchestrator with layer-specific functionality for testing individual layers.
"""

from __future__ import annotations
from typing import Dict
import torch
from neuronx_distributed_inference.module_test.base_template.orchestrator_base import (
    OrchestratorBase,
    OrchestratorBaseConfig,
)
from dataclasses import dataclass


@dataclass(kw_only=True)
class MFMOrchestratorConfig(OrchestratorBaseConfig):
    layer_id: int  # Layer ID to extract and test
    model_tag: str  # model tag in the Application base


class MFMOrchestratorBase(OrchestratorBase):
    """
    Orchestrator for testing modules extracted from a specific layer of a full model.
    Handles layer-specific input preparation and KV cache management.
    """

    def prepare_input_hf_format(self) -> Dict[str, torch.Tensor]:
        # TODO: in the future, implement this function to use real input stored from S3
        return super().prepare_input_hf_format()

    def prepare_kv_cache_hf_format(self) -> torch.Tensor:
        # TODO: in the future, implement this function to use real input stored from S3
        return super().prepare_kv_cache_hf_format()
