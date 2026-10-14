# Qwen3-Coder-Next NxDI contrib model
# Hybrid DeltaNet + GQA + Sparse MoE (80B total / 3B active per token)

from .modeling_qwen35_moe import (
    Qwen35MoeInferenceConfig,
    NeuronQwen35MoeForCausalLM,
    NeuronQwen35MoeModel,
    NeuronGatedDeltaNet,
    NeuronQwen35Attention,
    NeuronQwen35DecoderLayer,
    SigmoidGatedSharedExperts,
    Qwen35DecoderModelInstance,
    Qwen35ModelWrapper,
)

__all__ = [
    "Qwen35MoeInferenceConfig",
    "NeuronQwen35MoeForCausalLM",
    "NeuronQwen35MoeModel",
    "NeuronGatedDeltaNet",
    "NeuronQwen35Attention",
    "NeuronQwen35DecoderLayer",
    "SigmoidGatedSharedExperts",
    "Qwen35DecoderModelInstance",
    "Qwen35ModelWrapper",
]
