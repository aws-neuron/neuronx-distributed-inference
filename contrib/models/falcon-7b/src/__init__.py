# Falcon-7B NeuronX Port
#
# This package contains the NeuronX implementation of Falcon-7B for
# AWS Trainium/Inferentia hardware.
#
# Classes:
#   - NeuronFalconForCausalLM: Main model class for inference
#   - FalconInferenceConfig: Configuration class
#   - NeuronFalconAttention: Multi-Query Attention implementation
#   - NeuronFalconMLP: MLP layer implementation
#   - NeuronFalconDecoderLayer: Decoder layer with parallel attention + MLP
#   - NeuronFalconModel: Base transformer model
#
# Usage:
#   from neuronx_port import NeuronFalconForCausalLM, FalconInferenceConfig
#
# Port Version: v1
# Port Bank ID: 1949
# Validated: 2026-01-27

from .modeling_falcon import (
    FalconInferenceConfig,
    NeuronFalconAttention,
    NeuronFalconMLP,
    NeuronFalconDecoderLayer,
    NeuronFalconModel,
    NeuronFalconForCausalLM,
)

__all__ = [
    "FalconInferenceConfig",
    "NeuronFalconAttention",
    "NeuronFalconMLP",
    "NeuronFalconDecoderLayer",
    "NeuronFalconModel",
    "NeuronFalconForCausalLM",
]

__version__ = "1.0.0"
__port_bank_id__ = "1949"
