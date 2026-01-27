"""
SmolLM3-3B NeuronX Port

This package contains the NeuronX Distributed Inference implementation
of SmolLM3-3B for AWS Trainium hardware.

Key Features:
- GQA with 16 query heads and 4 KV heads
- NoPE layers (every 4th layer skips RoPE)
- Tied embeddings
- SwiGLU activation

Usage:
    from neuronx_port import NeuronSmolLM3ForCausalLM, SmolLM3InferenceConfig

    # Create config
    config = SmolLM3InferenceConfig.from_pretrained(
        "/path/to/SmolLM3-3B",
        neuron_config=neuron_config
    )

    # Create model
    model = NeuronSmolLM3ForCausalLM(config)
    model.load("./compiled_model")

IMPORTANT: Must use TP=1 for this model.
"""

from .modeling_smollm3_neuron import (
    SmolLM3InferenceConfig,
    NeuronSmolLM3Model,
    NeuronSmolLM3ForCausalLM,
    NeuronSmolLM3Attention,
    NeuronSmolLM3MLP,
    NeuronSmolLM3DecoderLayer,
)

__all__ = [
    "SmolLM3InferenceConfig",
    "NeuronSmolLM3Model",
    "NeuronSmolLM3ForCausalLM",
    "NeuronSmolLM3Attention",
    "NeuronSmolLM3MLP",
    "NeuronSmolLM3DecoderLayer",
]

__version__ = "1.0.0"
