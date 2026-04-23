#!/usr/bin/env python3
"""
Patch HuggingFace config.json for Ministral 8B compatibility with NxDI.

Ministral-8B-Instruct-2410 has two config issues that prevent NxDI compilation:
  1. sliding_window=32768 causes tensor shape mismatch in NxDI attention
  2. layer_types (alternating full/sliding attention) is not supported by NxDI

Both are safe to remove when max_model_len < 32768 (which is always true for
our benchmark config of 8192).

Usage:
    python patch_config.py --model-dir /path/to/Ministral-8B-Instruct-2410
"""

import argparse
import json
import os
import shutil


def patch_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        return False

    # Backup
    bak = config_path + ".bak_contrib"
    if not os.path.exists(bak):
        shutil.copy2(config_path, bak)
        print(f"  Backed up config.json")

    with open(config_path) as f:
        config = json.load(f)

    changed = False

    # Remove sliding_window
    if "sliding_window" in config and config["sliding_window"] is not None:
        old_val = config["sliding_window"]
        config["sliding_window"] = None
        print(f"  sliding_window: {old_val} -> null")
        changed = True
    elif "sliding_window" in config:
        print(f"  sliding_window: already null")
    else:
        print(f"  sliding_window: not present (OK)")

    # Remove layer_types
    if "layer_types" in config:
        old_val = config.pop("layer_types")
        n_full = old_val.count("full_attention") if isinstance(old_val, list) else "?"
        n_slide = (
            old_val.count("sliding_attention") if isinstance(old_val, list) else "?"
        )
        print(f"  layer_types: removed ({n_full} full, {n_slide} sliding)")
        changed = True
    else:
        print(f"  layer_types: not present (OK)")

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Config patched: {config_path}")
    else:
        print(f"\n  No changes needed")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch Ministral 8B config for NxDI compatibility"
    )
    parser.add_argument(
        "--model-dir",
        default="/mnt/models/Ministral-8B-Instruct-2410",
        help="Path to model directory",
    )
    args = parser.parse_args()
    patch_config(args.model_dir)
