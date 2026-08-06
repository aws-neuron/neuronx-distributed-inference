#!/usr/bin/env python3
"""Patch modular_isaac.py on GPU to handle missing imports."""

import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "/home/ubuntu/.cache/huggingface/modules/transformers_modules/"
        "Isaac_hyphen_0_dot_2_hyphen_2B_hyphen_Preview/modular_isaac.py"
    )
)

with open(path, "r") as f:
    content = f.read()

fixes = 0

# Fix 1: DefaultFastImageProcessorKwargs
old1 = (
    "from transformers.image_processing_utils_fast import (\n"
    "    BaseImageProcessorFast,\n"
    "    DefaultFastImageProcessorKwargs,\n"
    "    SizeDict,\n"
    "    group_images_by_shape,\n"
    "    reorder_images,\n"
    ")"
)
new1 = (
    "from transformers.image_processing_utils_fast import (\n"
    "    BaseImageProcessorFast,\n"
    "    SizeDict,\n"
    "    group_images_by_shape,\n"
    "    reorder_images,\n"
    ")\n"
    "try:\n"
    "    from transformers.image_processing_utils_fast import DefaultFastImageProcessorKwargs\n"
    "except ImportError:\n"
    "    from typing import TypedDict\n"
    "    class DefaultFastImageProcessorKwargs(TypedDict, total=False):\n"
    "        pass"
)
if old1 in content:
    content = content.replace(old1, new1)
    fixes += 1
    print("Fix 1 applied: DefaultFastImageProcessorKwargs")
else:
    print("Fix 1: not found (may already be patched)")

# Fix 2: perceptron soft-fail
old2 = (
    "except ModuleNotFoundError as exc:  # pragma: no cover - import guard\n"
    "    raise ModuleNotFoundError(\n"
    '        "perceptron.tensorstream is required for the Isaac HuggingFace integration. "\n'
    '        "Ensure the TensorStream package is installed and on PYTHONPATH."\n'
    "    ) from exc"
)
new2 = (
    "except ModuleNotFoundError:  # pragma: no cover - import guard\n"
    "    import warnings as _warnings\n"
    '    _warnings.warn("perceptron.tensorstream not available; TensorStream features disabled")\n'
    "\n"
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
if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1
    print("Fix 2 applied: perceptron soft-fail")
else:
    print("Fix 2: not found (may already be patched)")

with open(path, "w") as f:
    f.write(content)
print(f"Done: {fixes} fixes applied to {path}")
