#!/usr/bin/env python3
"""
Patch NxDI moe.py to forward blockwise_matmul_config to ExpertMLPs.

Required for SDK 2.29+ where the NKI blockwise kernel (neuronxcc.nki._private.blockwise_mm)
was removed in NKI 0.3.0 GA. Without this patch, MoE models crash with:
  NotImplementedError: _call_shard_hidden_kernel is not available

The patch checks if `use_torch_block_wise=True` is set in the NeuronConfig's
blockwise_matmul_config and passes it to the ExpertMLPs constructor.

Usage:
    python patch_moe.py

Must be run BEFORE starting vLLM. Only needed on SDK 2.29+ (NKI 0.3.0 GA).
"""

import importlib
import os
import re
import sys


def get_moe_path():
    """Find the moe.py file in the installed NxDI package."""
    try:
        import neuronx_distributed_inference.modules.moe as moe_module

        return moe_module.__file__
    except ImportError:
        # Fallback: search common venv locations
        venv_paths = [
            "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/lib/python3.12/site-packages/neuronx_distributed_inference/modules/moe.py",
            "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/lib/python3.12/site-packages/neuronx_distributed_inference/modules/moe.py",
        ]
        for p in venv_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            "Cannot find neuronx_distributed_inference/modules/moe.py"
        )


def patch_moe():
    """Apply the torch_block_wise forwarding patch to moe.py."""
    fpath = get_moe_path()
    content = open(fpath).read()

    # Check if already patched
    if "use_torch_block_wise" in content:
        print(f"Already patched: {fpath}")
        return True

    # Find the initialize_moe_module function and add the forwarding logic
    # We need to find where ExpertMLPs is instantiated and add the kwarg
    # The pattern: look for "def initialize_moe_module" and add our check before ExpertMLPs(...)

    # Strategy: Insert a block at the top of initialize_moe_module that sets up extra kwargs
    pattern = r"(def initialize_moe_module\([^)]*\):)"
    match = re.search(pattern, content)
    if not match:
        print("ERROR: Could not find initialize_moe_module function")
        return False

    # Find the line after the function def to insert our patch
    func_start = match.end()
    # Skip docstring if present
    rest = content[func_start:]

    # Insert patch: check blockwise_matmul_config and add use_torch_block_wise
    patch_code = """
    # --- SDK 2.29 MoE patch: forward use_torch_block_wise to ExpertMLPs ---
    extra_kwargs = {}
    if hasattr(config, 'neuron_config') and hasattr(config.neuron_config, 'blockwise_matmul_config'):
        bmc = config.neuron_config.blockwise_matmul_config
        use_torch = getattr(bmc, "use_torch_block_wise", False)
        if use_torch:
            extra_kwargs["use_torch_block_wise"] = True
    # --- end patch ---
"""

    # Find the first newline after the function signature
    newline_pos = rest.find("\n")
    if newline_pos == -1:
        print("ERROR: Unexpected file structure")
        return False

    # Check if there's a docstring
    stripped = rest[newline_pos:].lstrip()
    if stripped.startswith('"""') or stripped.startswith("'''"):
        # Skip past the closing docstring
        quote = stripped[:3]
        docstring_end = rest.find(
            quote, newline_pos + rest[newline_pos:].find(quote) + 3
        )
        if docstring_end > 0:
            insert_pos = func_start + docstring_end + 3
            # Find next newline
            next_nl = content.find("\n", insert_pos)
            insert_pos = next_nl + 1
        else:
            insert_pos = func_start + newline_pos + 1
    else:
        insert_pos = func_start + newline_pos + 1

    content = content[:insert_pos] + patch_code + content[insert_pos:]

    # Now add **extra_kwargs to the ExpertMLPs constructor call
    # Find "ExpertMLPs(" and add extra_kwargs before the closing )
    expert_pattern = r"(ExpertMLPs\([^)]+)\)"
    matches = list(re.finditer(expert_pattern, content))
    if matches:
        # Patch the last match (most likely the real one inside initialize_moe_module)
        for m in reversed(matches):
            old = m.group(0)
            new = old[:-1] + ", **extra_kwargs)"
            content = content[: m.start()] + new + content[m.end() :]

        open(fpath, "w").write(content)
        print(f"Patched successfully: {fpath}")
        return True
    else:
        print(
            "WARNING: Could not find ExpertMLPs constructor. Manual patch may be needed."
        )
        print(f"File: {fpath}")
        open(fpath, "w").write(content)
        return False


if __name__ == "__main__":
    success = patch_moe()
    sys.exit(0 if success else 1)
