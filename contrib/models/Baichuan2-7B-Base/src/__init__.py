# coding=utf-8
# Copyright 2023 Baichuan Inc. All rights reserved.
"""
Baichuan2-7B-Base NeuronX Port
"""

from .modeling_baichuan2 import (
    Baichuan2InferenceConfig,
    NeuronBaichuan2ForCausalLM,
)

__all__ = [
    "Baichuan2InferenceConfig",
    "NeuronBaichuan2ForCausalLM",
]
