# Kimi-K2.5 Multimodal on Neuron via NxDI
#
# Text decoder: reuses K2 model code (modeling_kimi_k2.py)
# Multimodal: K2.5 checkpoint loader, vision fusion, MoonViT (modeling_kimi_k25.py, moonvit.py)

from .modeling_kimi_k2 import NeuronKimiK2ForCausalLM, KimiK2InferenceConfig
from .modeling_kimi_k25 import (
    apply_k25_patches,
    apply_k25_checkpoint_patch,
    build_k25_config,
    create_text_only_model_dir,
    preprocess_image,
    precompute_rope_tables,
    K25ImageToTextModelWrapper,
    BOS_TOKEN_ID,
    IM_USER_TOKEN_ID,
    IM_END_TOKEN_ID,
    IM_ASSISTANT_TOKEN_ID,
    MEDIA_PLACEHOLDER_TOKEN_ID,
)
from .moonvit import (
    NeuronMoonViTWrapper,
    load_vision_weights,
    precompute_rope_real,
)

__all__ = [
    # K2 text decoder
    "NeuronKimiK2ForCausalLM",
    "KimiK2InferenceConfig",
    # K2.5 multimodal
    "apply_k25_patches",
    "apply_k25_checkpoint_patch",
    "build_k25_config",
    "create_text_only_model_dir",
    "K25ImageToTextModelWrapper",
    # Vision
    "NeuronMoonViTWrapper",
    "load_vision_weights",
    # Preprocessing
    "preprocess_image",
    "precompute_rope_tables",
    "precompute_rope_real",
    # Token IDs
    "BOS_TOKEN_ID",
    "IM_USER_TOKEN_ID",
    "IM_END_TOKEN_ID",
    "IM_ASSISTANT_TOKEN_ID",
    "MEDIA_PLACEHOLDER_TOKEN_ID",
]
