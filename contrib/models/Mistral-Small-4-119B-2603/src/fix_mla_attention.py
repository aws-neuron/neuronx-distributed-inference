#!/usr/bin/env python3
"""
Fix MLA attention out_absorb slicing bug in NxDI's stock DeepseekV3Attention.

Bug: Line 230 of modeling_deepseek.py uses:
    out_absorb = wkv_b[:, self.v_head_dim:, :]

This is correct ONLY when v_head_dim == qk_nope_head_dim (as in stock DeepSeek V3).
For Mistral-Small-4-119B (v_head_dim=128, qk_nope_head_dim=64), it produces a shape
mismatch causing RuntimeError during attention output reshape.

Fix: Change to:
    out_absorb = wkv_b[:, self.qk_nope_head_dim:, :]

The kv_b_proj weight is structured as [qk_nope_head_dim | v_head_dim] per head.
- First qk_nope_head_dim elements: used for Q nope absorption
- Remaining v_head_dim elements: used for V output absorption

Usage:
    python fix_mla_attention.py
"""

import os
import sys


def get_modeling_deepseek_path():
    """Find the modeling_deepseek.py file."""
    try:
        import neuronx_distributed_inference.models.deepseek.modeling_deepseek as mod

        return mod.__file__
    except ImportError:
        venv_paths = [
            "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/lib/python3.12/site-packages/neuronx_distributed_inference/models/deepseek/modeling_deepseek.py",
            "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/lib/python3.12/site-packages/neuronx_distributed_inference/models/deepseek/modeling_deepseek.py",
        ]
        for p in venv_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Cannot find modeling_deepseek.py")


def fix_mla():
    fpath = get_modeling_deepseek_path()
    content = open(fpath).read()

    old = "out_absorb = wkv_b[:, self.v_head_dim:, :]"
    new = "out_absorb = wkv_b[:, self.qk_nope_head_dim:, :]"

    if new in content:
        print(f"Already fixed: {fpath}")
        return True

    if old in content:
        content = content.replace(old, new)
        open(fpath, "w").write(content)
        print(f"Fixed MLA out_absorb slicing: {fpath}")
        return True
    else:
        print(f"ERROR: Pattern not found in {fpath}")
        print("  Expected: out_absorb = wkv_b[:, self.v_head_dim:, :]")
        return False


if __name__ == "__main__":
    success = fix_mla()
    sys.exit(0 if success else 1)
