#!/usr/bin/env python3
"""Patch vllm-neuron 0.5.0 to support Laguna-XS.2.

Registers the Laguna contrib model class into the NxDI MODEL_TYPES
registry so vllm-neuron can discover and load it.

Since Laguna is a standard causal LM (not multimodal), only one
registration layer is needed:
1. MODEL_TYPES — Add "laguna" -> NeuronLagunaForCausalLM mapping

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    export PYTHONPATH=contrib/models/Laguna-XS.2:$PYTHONPATH
    python contrib/models/Laguna-XS.2/vllm/patch_vllm_laguna.py
"""

import sys


def register_laguna():
    """Register NeuronLagunaForCausalLM in NxDI MODEL_TYPES."""
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES

    if "laguna" in MODEL_TYPES:
        print("[MODEL_TYPES] Laguna already registered — skipping")
        return

    # Import the contrib model class
    try:
        from src.modeling_laguna import NeuronLagunaForCausalLM
    except ImportError:
        print(
            "ERROR: Cannot import NeuronLagunaForCausalLM. "
            "Ensure PYTHONPATH includes contrib/models/Laguna-XS.2/",
            file=sys.stderr,
        )
        sys.exit(1)

    MODEL_TYPES["laguna"] = {"causal-lm": NeuronLagunaForCausalLM}
    print("[MODEL_TYPES] Registered laguna -> NeuronLagunaForCausalLM")


def main():
    register_laguna()
    print()
    print("Laguna registered with vLLM-neuron. To serve:")
    print()
    print("  export VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference'")
    print("  export NEURON_COMPILED_ARTIFACTS='/path/to/laguna-compiled'")
    print("  python -m vllm.entrypoints.openai.api_server \\")
    print("    --model /path/to/Laguna-XS.2 \\")
    print("    --tensor-parallel-size 4 \\")
    print("    --max-model-len 4096 \\")
    print("    --max-num-seqs 4 \\")
    print("    --block-size 128 \\")
    print("    --no-enable-prefix-caching \\")
    print("    --trust-remote-code")


if __name__ == "__main__":
    main()
