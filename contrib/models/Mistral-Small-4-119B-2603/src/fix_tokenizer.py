#!/usr/bin/env python3
"""
Fix tokenizer_config.json for Mistral-Small-4-119B-2603 compatibility.

The HuggingFace tokenizer uses:
- tokenizer_class: "TokenizersBackend" (not available in transformers <5.3)
- extra_special_tokens: [...] (list format, but transformers expects dict)

This script fixes both issues for compatibility with SDK 2.29's transformers version.

Usage:
    python fix_tokenizer.py /path/to/model-dir
"""

import json
import os
import sys


def fix_tokenizer(model_dir):
    fpath = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(fpath):
        print(f"ERROR: {fpath} not found")
        return False

    with open(fpath) as f:
        cfg = json.load(f)

    modified = False

    # Fix tokenizer_class
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.pop("backend", None)
        modified = True
        print("  Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")

    # Fix extra_special_tokens (list -> remove)
    if "extra_special_tokens" in cfg and isinstance(cfg["extra_special_tokens"], list):
        cfg.pop("extra_special_tokens")
        modified = True
        print("  Removed incompatible extra_special_tokens list")

    if modified:
        with open(fpath, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Tokenizer config fixed: {fpath}")
    else:
        print(f"Tokenizer config already compatible: {fpath}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_tokenizer.py /path/to/model-dir")
        sys.exit(1)

    success = fix_tokenizer(sys.argv[1])
    sys.exit(0 if success else 1)
