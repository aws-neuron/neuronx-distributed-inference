#!/usr/bin/env python3
"""Remove leftover indented stubs from modular_isaac.py."""

import sys

paths = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "/mnt/models/Isaac-0.2-2B-Preview/modular_isaac.py",
        "/home/ubuntu/.cache/huggingface/modules/transformers_modules/"
        "Isaac_hyphen_0_dot_2_hyphen_2B_hyphen_Preview/modular_isaac.py",
    ]
)

INDENTED_STUBS = (
    "\n\n"
    "    class Event: pass\n"
    "    class Stream: pass\n"
    "    class TensorStream: pass\n"
    "    class TextType: pass\n"
    "    class VisionType: pass\n"
    "    def create_stream(*a, **kw): return None\n"
    "    def group_streams(*a, **kw): return None\n"
    "    def compute_mrope_pos_tensor(*a, **kw): return None\n"
    "    def modality_mask(*a, **kw): return None\n"
    "    def reconstruct_tensor_stream_from_compact_dict(*a, **kw): return None\n"
    "    def tensor_stream_token_view(*a, **kw): return None\n"
    "    def ts_slice(*a, **kw): return None"
)

for path in paths:
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"SKIP: {path}")
        continue

    if INDENTED_STUBS in content:
        content = content.replace(INDENTED_STUBS, "")
        with open(path, "w") as f:
            f.write(content)
        print(f"FIXED: removed indented stubs from {path}")
    else:
        print(f"OK: no indented stubs found in {path}")
