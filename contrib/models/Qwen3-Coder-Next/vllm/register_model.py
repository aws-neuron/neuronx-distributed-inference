"""
Register Qwen3-Coder-Next (qwen3_next) with NxDI's MODEL_TYPES registry.

This must be imported BEFORE vLLM loads the model so that
_get_neuron_model_cls("Qwen3NextForCausalLM") can find our class.

Usage:
    import register_model  # patches MODEL_TYPES
    # then launch vllm normally
"""

import sys
import os

# Ensure our contrib src is importable
CONTRIB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if CONTRIB_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(CONTRIB_SRC))

from neuronx_distributed_inference.utils.constants import MODEL_TYPES
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM

# Register under "qwen3next" (what vLLM derives from "Qwen3NextForCausalLM")
# The key format is: architecture.split("For")[0].lower() -> model name
# "Qwen3NextForCausalLM" -> model="qwen3next", task="causal-lm"
MODEL_TYPES["qwen3next"] = {"causal-lm": NeuronQwen35MoeForCausalLM}

print(
    f"[register_model] Registered 'qwen3next' -> {NeuronQwen35MoeForCausalLM.__name__}"
)
