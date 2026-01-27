"""
Helium-1-2B NeuronX Port

This module provides a NeuronX-optimized implementation of the Helium-1-2B model
for AWS Trainium/Inferentia hardware.

Classes:
    HeliumInferenceConfig: Configuration class for Helium model
    NeuronHeliumForCausalLM: Main model class for causal language modeling

Usage:
    from neuronx_port import NeuronHeliumForCausalLM, HeliumInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig

    # Create config
    neuron_config = NeuronConfig(tp_degree=2, batch_size=1, seq_len=128)
    model_config = HeliumInferenceConfig.from_pretrained(
        "/path/to/hf_model",
        neuron_config=neuron_config,
    )

    # Create and compile model
    model = NeuronHeliumForCausalLM("/path/to/hf_model", model_config)
    model.compile("/path/to/output")

    # Load and run inference
    model = NeuronHeliumForCausalLM.from_pretrained(
        "/path/to/compiled",
        config=model_config,
    )
    model.load("/path/to/compiled")
    outputs = model(input_ids, position_ids=position_ids)
"""

from helium_config import HeliumInferenceConfig
from helium_model import (
    NeuronHeliumForCausalLM,
    NeuronHeliumModel,
    NeuronHeliumDecoderLayer,
    NeuronHeliumAttention,
    NeuronHeliumMLP,
)

__all__ = [
    "HeliumInferenceConfig",
    "NeuronHeliumForCausalLM",
    "NeuronHeliumModel",
    "NeuronHeliumDecoderLayer",
    "NeuronHeliumAttention",
    "NeuronHeliumMLP",
]

__version__ = "1.0.0"
