"""
Chandra OCR VLM on NeuronX Distributed Inference

Chandra (datalab-to/chandra) is a fine-tuned Qwen3-VL-8B model for OCR with
layout preservation. It uses NxDI's built-in Qwen3-VL multimodal pipeline
(NeuronQwen3VLForCausalLM + ImageToTextInferenceConfig) -- no custom modeling
code is required.

This module provides helper functions for configuration and inference.
"""

from .modeling_chandra import (
    get_chandra_neuron_configs,
    load_chandra_vllm,
    run_chandra_ocr,
)

__all__ = [
    "get_chandra_neuron_configs",
    "load_chandra_vllm",
    "run_chandra_ocr",
]
