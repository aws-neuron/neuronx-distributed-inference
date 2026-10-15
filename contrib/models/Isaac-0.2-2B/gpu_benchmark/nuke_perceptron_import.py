#!/usr/bin/env python3
"""Remove perceptron.tensorstream import entirely from modular_isaac.py.
Replaces the try/except import block with direct stub definitions."""

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

# Replacement: just the stubs, no try/except, no import
REPLACEMENT = """# perceptron.tensorstream stubs (not available outside Perceptron environment)
class Event: pass
class Stream: pass
class TensorStream: pass
class TextType: pass
class VisionType: pass
def create_stream(*a, **kw): return None
def group_streams(*a, **kw): return None
def compute_mrope_pos_tensor(*a, **kw): return None
def modality_mask(*a, **kw): return None
def reconstruct_tensor_stream_from_compact_dict(*a, **kw): return None
def tensor_stream_token_view(*a, **kw): return None
def ts_slice(*a, **kw): return None"""

for path in paths:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"SKIP: {path} not found")
        continue

    # Find the try block that imports from perceptron
    try_start = None
    except_end = None
    in_except = False

    for i, line in enumerate(lines):
        if (
            line.strip() == "try:"
            and i + 1 < len(lines)
            and "perceptron" in lines[i + 1]
        ):
            try_start = i
        if try_start is not None and line.strip().startswith(
            "except ModuleNotFoundError"
        ):
            in_except = True
        if in_except and try_start is not None:
            # Find end of except block (next non-indented, non-blank line after except body)
            if i > try_start + 5:  # we're past the except line itself
                # Check if this line is NOT indented (new top-level statement)
                stripped = line.strip()
                if (
                    stripped
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and "def " not in lines[i - 1]
                    if i > 0
                    else True
                ):
                    # But also check it's not a continuation of the except body
                    pass

    # Simpler approach: find by content markers
    content = "".join(lines)

    # Pattern 1: Original unpatched try/except
    import re

    # Match everything from "try:\n    from perceptron" to the end of the except block
    pattern = r"try:\n    from perceptron\.tensorstream\.tensorstream import \(.*?\n(?:.*?\n)*?except ModuleNotFoundError.*?\n(?:    .*\n)*"
    match = re.search(pattern, content)
    if match:
        old_block = match.group(0)
        # Remove trailing newlines from old_block to be precise
        content = content.replace(old_block, REPLACEMENT + "\n\n")
        with open(path, "w") as f:
            f.write(content)
        print(f"SUCCESS: Replaced try/import block in {path}")
    else:
        # Check if already replaced
        if "# perceptron.tensorstream stubs" in content:
            print(f"ALREADY PATCHED: {path}")
        else:
            print(f"WARN: Could not find try/import block in {path}")
            # Show perceptron references
            for i, line in enumerate(lines):
                if "perceptron" in line.lower():
                    print(f"  Line {i + 1}: {line.rstrip()}")
