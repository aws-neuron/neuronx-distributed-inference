# Qwen3.5-35B-A3B NeuronX Port
# Export main classes
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
from .nkilib_kernel_patch import patch_flash_attention_kernel, is_patched

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
    "flash_attn_d256",
    "patch_flash_attention_kernel",
    "is_patched",
]
