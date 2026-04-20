#!/usr/bin/env python3
"""Patch constants.py to add deepseekv3 to MODEL_TYPES."""

fpath = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py"
content = open(fpath).read()

# Check if already added
if '"deepseekv3"' in content:
    print("deepseekv3 already in MODEL_TYPES")
else:
    # Find the closing brace of MODEL_TYPES dict
    # It's the last "}" before some constants or end of MODEL_TYPES block
    # Strategy: find MODEL_TYPES = { ... } and add entry before the closing }

    # Find the last } that closes MODEL_TYPES
    mt_start = content.index("MODEL_TYPES = {")
    # Find matching closing brace
    brace_count = 0
    mt_end = -1
    for i in range(mt_start, len(content)):
        if content[i] == "{":
            brace_count += 1
        elif content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                mt_end = i
                break

    if mt_end > 0:
        # Insert new entry before the closing }
        new_entry = '    "deepseekv3": {"causal-lm": NeuronDeepseekV3ForCausalLM},\n'
        content = content[:mt_end] + new_entry + content[mt_end:]
        open(fpath, "w").write(content)
        print("Added deepseekv3 to MODEL_TYPES")
    else:
        print("ERROR: Could not find MODEL_TYPES closing brace")
