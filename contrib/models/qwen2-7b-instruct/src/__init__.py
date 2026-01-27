# Qwen2-7B-Instruct NeuronX Port
#
# This package contains the NeuronX implementation of the Qwen2-7B-Instruct model
# for AWS Trainium/Inferentia hardware.
#
# Usage:
#   from neuronx_port.modeling_qwen2 import NeuronQwen2ForCausalLM, Qwen2InferenceConfig
#
# See README.md for detailed usage instructions.

from .modeling_qwen2 import (
    NeuronQwen2ForCausalLM,
    Qwen2InferenceConfig,
    Qwen2NeuronConfig,
    NeuronQwen2Attention,
    NeuronQwen2DecoderLayer,
    NeuronQwen2Model,
)

__all__ = [
    "NeuronQwen2ForCausalLM",
    "Qwen2InferenceConfig",
    "Qwen2NeuronConfig",
    "NeuronQwen2Attention",
    "NeuronQwen2DecoderLayer",
    "NeuronQwen2Model",
]

__version__ = "1.0.0"
__port_version__ = "1272"
