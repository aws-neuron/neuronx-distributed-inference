# Qwen2.5-VL-7B NxDI Full Vision-Language Implementation
#
# This contrib provides complete Qwen2.5-VL support on Neuron, including:
# - Text decoder (identical to Qwen2-VL backbone)
# - Vision encoder with windowed attention + SwiGLU MLP + RMSNorm
# - VL orchestrator with ImageToTextInferenceConfig
#
# Usage:
#   from src.modeling_qwen2_5_vl import (
#       NeuronQwen2_5_VLForCausalLM,
#       Qwen2_5_VLInferenceConfig,
#   )
#   from src.modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding

from .modeling_qwen2_5_vl import (
    NeuronQwen2_5_VLForCausalLM,
    Qwen2_5_VLInferenceConfig,
)
from .modeling_qwen2_5_vl_text import (
    NeuronQwen2_5_VLTextForCausalLM,
)
from .modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding

__all__ = [
    "NeuronQwen2_5_VLForCausalLM",
    "Qwen2_5_VLInferenceConfig",
    "NeuronQwen2_5_VLTextForCausalLM",
    "NeuronQwen2_5_VLForImageEncoding",
]
