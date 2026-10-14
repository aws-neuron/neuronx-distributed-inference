"""Patch constants.py to add ZAYA model with torch.jit.script guard."""

CONSTANTS_PATH = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py"
BACKUP_PATH = CONSTANTS_PATH + ".bak"

# Restore from backup first
with open(BACKUP_PATH, "r") as f:
    content = f.read()

# Add ZAYA import with torch.jit.script monkey-patch BEFORE the import
zaya_import = """
# ZAYA contrib model (patched for vLLM integration)
# The Zyphra transformers fork uses @jit_fuser which calls torch.jit.script
# at module import time. This crashes in the Neuron environment, so we
# temporarily disable torch.jit.script during the import.
import sys as _sys
import torch as _torch
_zaya_src = "/home/ubuntu/zaya/contrib/models/ZAYA1-base/src"
if _zaya_src not in _sys.path:
    _sys.path.insert(0, _zaya_src)
_real_jit_script = _torch.jit.script
_torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)
try:
    from modeling_zaya import NeuronZayaForCausalLM
finally:
    _torch.jit.script = _real_jit_script
"""

# Insert after the pixtral vision import line
insert_after = "from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import NeuronPixtralForImageEncoding"
content = content.replace(insert_after, insert_after + zaya_import)

# Add zaya to MODEL_TYPES dict - find the last entry and add after it
old_last = '    "qwen3_vl": {"causal-lm": NeuronQwen3VLForCausalLM,\n                 "image-encoding": NeuronQwen3VLForImageEncoding},'
new_last = old_last + '\n    "zaya": {"causal-lm": NeuronZayaForCausalLM},'
content = content.replace(old_last, new_last)

with open(CONSTANTS_PATH, "w") as f:
    f.write(content)

print("constants.py patched successfully with jit.script guard")
