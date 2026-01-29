# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

"""
Ovis2.5-9B model for NeuronX Distributed Inference.

This package implements a NeuronX port of the Ovis2.5-9B multimodal model.
For initial implementation, only the text-only LLM component (Qwen3-8B) is ported.

Main components:
- Ovis2_5_InferenceConfig: Configuration class
- NeuronOvis2_5_ForCausalLM: Model class for causal language modeling
"""

from .configuration_ovis2_5 import (
    Ovis2_5_InferenceConfig,
    Ovis2_5_NeuronConfig,
)
from .modeling_ovis2_5 import (
    NeuronOvis2_5_ForCausalLM,
    NeuronOvis2_5_Model,
)

__all__ = [
    "Ovis2_5_InferenceConfig",
    "Ovis2_5_NeuronConfig",
    "NeuronOvis2_5_ForCausalLM",
    "NeuronOvis2_5_Model",
]
