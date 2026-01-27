# coding=utf-8
# Copyright 2024 AWS Neuron. All rights reserved.
"""
Llama-2-7b-hf NeuronX Port

This package provides a NeuronX-compatible implementation of Meta's Llama-2-7b-hf
model for efficient inference on AWS Trainium hardware.
"""

from .modeling_llama2 import (
    Llama2InferenceConfig,
    NeuronLlama2ForCausalLM,
)

__all__ = [
    "Llama2InferenceConfig",
    "NeuronLlama2ForCausalLM",
]
