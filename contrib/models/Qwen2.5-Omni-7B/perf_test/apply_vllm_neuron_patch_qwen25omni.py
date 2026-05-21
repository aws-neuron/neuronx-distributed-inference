#!/usr/bin/env python3
"""Add Qwen2.5-Omni model support to vllm-neuron.

This patch should be applied AFTER the MiMo/MiniMax patch (apply_vllm_neuron_patch.py).
It handles:
  1. Config extraction: Qwen2.5-Omni nests text config under thinker_config.text_config
  2. Architecture mapping: "Qwen2_5OmniModel" -> "qwen2_5_omni" model type
  3. Layer count extraction: get_num_layers_from_hf_config for nested config
"""

import os

# Patch 1 & 2: neuronx_distributed_model_loader.py
LOADER_FILE = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/vllm_neuron/worker/neuronx_distributed_model_loader.py"

with open(LOADER_FILE) as f:
    content = f.read()

# 1. In _get_model_configs: handle Qwen2.5-Omni nested config
content = content.replace(
    '    if architecture in NEURON_MULTI_MODAL_MODELS:\n'
    '        config = getattr(config, "text_config", None)\n'
    '    num_key_value_heads = getattr(config, "num_key_value_heads", None)',
    '    if architecture in NEURON_MULTI_MODAL_MODELS:\n'
    '        config = getattr(config, "text_config", None)\n'
    '    # Qwen2.5-Omni: text config is nested under thinker_config.text_config\n'
    '    if architecture == "Qwen2_5OmniModel":\n'
    '        thinker_config = getattr(config, "thinker_config", None)\n'
    '        if thinker_config is not None:\n'
    '            config = getattr(thinker_config, "text_config", config)\n'
    '    num_key_value_heads = getattr(config, "num_key_value_heads", None)',
)

# 2. In _get_neuron_model_cls: handle Qwen2_5OmniModel architecture
content = content.replace(
    '    try:\n'
    '        if "For" in architecture:',
    '    # Qwen2.5-Omni: architecture is "Qwen2_5OmniModel" (no "For" in name)\n'
    '    if architecture == "Qwen2_5OmniModel":\n'
    '        return MODEL_TYPES["qwen2_5_omni"]["causal-lm"]\n'
    '\n'
    '    try:\n'
    '        if "For" in architecture:',
)

with open(LOADER_FILE, "w") as f:
    f.write(content)

print("Patch 1/2: neuronx_distributed_model_loader.py updated")

# Patch 3: utils.py - handle Qwen2.5-Omni nested config for layer count
UTILS_FILE = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/vllm_neuron/worker/utils.py"

with open(UTILS_FILE) as f:
    content = f.read()

content = content.replace(
    '    # Sum nested configs (multimodal models)\n'
    '    total = 0\n'
    '    for attr in ["text_config", "vision_config"]:',
    '    # Qwen2.5-Omni: check thinker_config.text_config\n'
    '    thinker_config = getattr(hf_config, "thinker_config", None)\n'
    '    if thinker_config is not None:\n'
    '        text_config = getattr(thinker_config, "text_config", None)\n'
    '        if text_config is not None:\n'
    '            layers = getattr(text_config, "num_hidden_layers", None)\n'
    '            if layers is not None:\n'
    '                return layers\n'
    '\n'
    '    # Sum nested configs (multimodal models)\n'
    '    total = 0\n'
    '    for attr in ["text_config", "vision_config"]:',
)

with open(UTILS_FILE, "w") as f:
    f.write(content)

print("Patch 2/2: utils.py updated")
print()
print("Qwen2.5-Omni vllm-neuron patch applied successfully!")
print("  1. Added thinker_config.text_config extraction in _get_model_configs")
print("  2. Added Qwen2_5OmniModel -> qwen2_5_omni mapping in _get_neuron_model_cls")
print("  3. Added thinker_config.text_config layer count extraction in utils.py")
