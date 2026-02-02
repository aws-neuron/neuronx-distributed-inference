# NeuronX RecurrentGemma Port
# 
# This package provides a NeuronX Distributed Inference compatible implementation
# of RecurrentGemma - a hybrid architecture combining recurrent blocks (RG-LRU) 
# with attention blocks.

from .modeling_recurrent_gemma import (
    RecurrentGemmaInferenceConfig,
    NeuronRecurrentGemmaModel,
    NeuronRecurrentGemmaForCausalLM,
    RecurrentGemmaRMSNorm,
    RecurrentGemmaMLP,
    RecurrentGemmaSdpaAttention,
    RecurrentGemmaRecurrentBlock,
    RecurrentGemmaRglru,
    RecurrentGemmaDecoderLayer,
)

__all__ = [
    "RecurrentGemmaInferenceConfig",
    "NeuronRecurrentGemmaModel", 
    "NeuronRecurrentGemmaForCausalLM",
    "RecurrentGemmaRMSNorm",
    "RecurrentGemmaMLP",
    "RecurrentGemmaSdpaAttention",
    "RecurrentGemmaRecurrentBlock",
    "RecurrentGemmaRglru",
    "RecurrentGemmaDecoderLayer",
]
