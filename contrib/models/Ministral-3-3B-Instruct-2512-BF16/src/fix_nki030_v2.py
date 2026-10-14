#!/usr/bin/env python3
"""
Fix attention_block_tkg_multi_kv.py for NKI 0.3.0 compatibility.

NKI 0.3.0 does NOT support keyword-only arguments (after *,).
Fix: remove *, entirely, give all non-defaulted params appropriate defaults.

For ndarray params: default = None (they MUST be provided at call time)
For bool params: default based on their usage (False for enable flags)
For float/int params: default = 0.0 / 0
For Optional params: default = None
"""

import sys
import re

fpath = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/lib/python3.12/site-packages/nkilib/experimental/transformer/attention_block_tkg_multi_kv.py"
)

with open(fpath) as f:
    content = f.read()

# Step 1: Remove the *, line
content = content.replace("    *,\n", "")
print("  Removed *,")

# Step 2: For params without defaults that come after params with defaults,
# add appropriate defaults. We need to find the function signature.
# The first defaulted param (that already had a default) determines the cutoff.

# Find function signature boundaries
func_start = content.find("def attention_block_tkg(")
paren_depth = 0
in_func = False
func_end = func_start
for i in range(func_start, len(content)):
    if content[i] == "(":
        paren_depth += 1
        in_func = True
    elif content[i] == ")":
        paren_depth -= 1
        if in_func and paren_depth == 0:
            func_end = i
            break

sig = content[func_start : func_end + 1]
lines = sig.split("\n")
new_lines = []

# Track if we've seen a defaulted param
seen_default = False

for line in lines:
    stripped = line.strip()

    # Skip comments, empty lines, def line, closing paren
    if (
        stripped.startswith("#")
        or stripped == ""
        or stripped.startswith("def ")
        or stripped == ")"
        or stripped == "),"
    ):
        new_lines.append(line)
        continue

    # Check if line has a default
    has_default = "=" in stripped and ":" in stripped

    if has_default:
        seen_default = True
        new_lines.append(line)
        continue

    # No default - need to add one if we've seen a defaulted param
    if not seen_default:
        new_lines.append(line)
        continue

    # Need to add a default. Determine appropriate default based on type.
    if "Optional[" in stripped:
        default = "None"
    elif ": nl.ndarray" in stripped:
        default = "None"
    elif ": bool" in stripped:
        default = "False"
    elif ": float" in stripped:
        default = "0.0"
    elif ": int" in stripped:
        default = "0"
    elif "QuantizationType" in stripped:
        default = "None"
    elif "SbufManager" in stripped:
        default = "None"
    else:
        default = "None"

    # Add default before trailing comma
    if stripped.endswith(","):
        line = line.rstrip().rstrip(",") + f" = {default},"
    else:
        line = line.rstrip() + f" = {default}"

    new_lines.append(line)

new_sig = "\n".join(new_lines)
content = content[:func_start] + new_sig + content[func_end + 1 :]

with open(fpath, "w") as f:
    f.write(content)

# Verify syntax
import py_compile

try:
    py_compile.compile(fpath, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Syntax error: {e}")
    sys.exit(1)

# Show first 30 lines of new signature for verification
with open(fpath) as f:
    all_lines = f.readlines()
for i, line in enumerate(all_lines):
    if "def attention_block_tkg" in line:
        for j in range(i, min(i + 45, len(all_lines))):
            print(f"{j + 1}: {all_lines[j]}", end="")
        break
