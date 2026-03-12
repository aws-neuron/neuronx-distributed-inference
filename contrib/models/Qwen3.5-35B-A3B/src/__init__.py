# Qwen3.5-35B-A3B NeuronX Port
# Export main classes

# Text decoder (existing)
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
from .nki_flash_attn_d256 import flash_attn_d256
from .nkilib_kernel_patch import get_nkilib_flash_attention_kernel, is_available

# Vision encoder
from .modeling_qwen35_moe_vision import (
    NeuronQwen35VisionModel,
    NeuronQwen35VisionModelWrapper,
    NeuronQwen35VisionAttention,
    NeuronQwen35VisionBlock,
    NeuronQwen35VisionMLP,
)

# Vision-Language orchestrator
from .modeling_qwen35_moe_vl import (
    NeuronQwen35MoeVLForCausalLM,
    Qwen35MoeVLInferenceConfig,
    get_rope_index,
)

__all__ = [
    # Text decoder
    "Qwen35MoeInferenceConfig",
    "NeuronQwen35MoeForCausalLM",
    "NeuronQwen35MoeModel",
    "NeuronGatedDeltaNet",
    "NeuronQwen35Attention",
    "NeuronQwen35DecoderLayer",
    "SigmoidGatedSharedExperts",
    "Qwen35DecoderModelInstance",
    "Qwen35ModelWrapper",
    "flash_attn_d256",
    "get_nkilib_flash_attention_kernel",
    "is_available",
    # Vision encoder
    "NeuronQwen35VisionModel",
    "NeuronQwen35VisionModelWrapper",
    "NeuronQwen35VisionAttention",
    "NeuronQwen35VisionBlock",
    "NeuronQwen35VisionMLP",
    # VL orchestrator
    "NeuronQwen35MoeVLForCausalLM",
    "Qwen35MoeVLInferenceConfig",
    "get_rope_index",
]
