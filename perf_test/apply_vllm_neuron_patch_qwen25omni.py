#!/usr/bin/env python3
"""Add Qwen2.5-Omni model type mapping to vllm-neuron.

This patch should be applied AFTER the MiMo/MiniMax patch (apply_vllm_neuron_patch.py).
It adds the Qwen2.5-Omni -> qwen2_5_omni model mapping in _get_neuron_model_cls.
"""

FILE = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/vllm_neuron/worker/neuronx_distributed_model_loader.py"

with open(FILE) as f:
    content = f.read()

# Add Qwen2.5-Omni model mapping in _get_neuron_model_cls
# The architecture "Qwen2_5OmniModel" normalizes to "qwen2_5omni" or similar
# Map it to "qwen2_5_omni" which is the NxDI MODEL_TYPES key
content = content.replace(
    '            # MiMo is based on Qwen2 architecture\n'
    '            if model == "mimo":\n'
    '                model = "qwen2"\n'
    '\n'
    '            if model == "qwen2vl":',
    '            # MiMo is based on Qwen2 architecture\n'
    '            if model == "mimo":\n'
    '                model = "qwen2"\n'
    '\n'
    '            # Qwen2.5-Omni: text backbone is Qwen2.5\n'
    '            if model in ("qwen2_5omni", "qwen25omni", "qwen2_5_omni"):\n'
    '                model = "qwen2_5_omni"\n'
    '\n'
    '            if model == "qwen2vl":',
)

with open(FILE, "w") as f:
    f.write(content)

print("Qwen2.5-Omni vllm-neuron patch applied successfully!")
print("  Added qwen2_5_omni model mapping in _get_neuron_model_cls")
