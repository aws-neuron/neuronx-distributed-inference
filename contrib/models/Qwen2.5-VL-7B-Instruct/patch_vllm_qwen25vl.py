"""
Patch vllm-neuron to add Qwen2.5-VL support.
Modifies 4 files:
1. vllm_neuron/worker/constants.py - Add to NEURON_MULTI_MODAL_MODELS
2. NxDI constants.py - Add to MODEL_TYPES
3. vllm_neuron/worker/neuronx_distributed_model_loader.py - Add wrapper class + dispatches
4. vllm_neuron/worker/neuronx_distributed_model_runner.py - Add multimodal data routing
"""

import os
import sys


def patch_file(path, check_string, old_string, new_string, description):
    """Patch a file by replacing old_string with new_string."""
    with open(path) as f:
        content = f.read()

    if check_string in content:
        print(f"SKIPPED (already patched): {path} - {description}")
        return False

    if old_string not in content:
        print(f"ERROR: Could not find target string in {path} for: {description}")
        print(f"  Looking for: {repr(old_string[:100])}")
        return False

    content = content.replace(old_string, new_string, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"PATCHED: {path} - {description}")
    return True


def append_to_file(path, check_string, text, description):
    """Append text to a file if check_string is not already present."""
    with open(path) as f:
        content = f.read()

    if check_string in content:
        print(f"SKIPPED (already patched): {path} - {description}")
        return False

    content += text
    with open(path, "w") as f:
        f.write(content)
    print(f"PATCHED: {path} - {description}")
    return True


# === File paths ===
VLLM_CONSTANTS = "/vllm/vllm_neuron/worker/constants.py"
NXDI_CONSTANTS = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/lib/python3.12/site-packages/neuronx_distributed_inference/utils/constants.py"
MODEL_LOADER = "/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py"
MODEL_RUNNER = "/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py"


# === 1. Add to NEURON_MULTI_MODAL_MODELS ===
patch_file(
    VLLM_CONSTANTS,
    check_string="Qwen2_5_VLForConditionalGeneration",
    old_string='    "Qwen3VLForConditionalGeneration",\n]',
    new_string='    "Qwen2_5_VLForConditionalGeneration",\n    "Qwen3VLForConditionalGeneration",\n]',
    description="Add Qwen2_5_VL to NEURON_MULTI_MODAL_MODELS",
)


# === 2. Register in NxDI MODEL_TYPES ===
registration_block = """

# --- Qwen2.5-VL registration (patched for qwen2.5-vl project) ---
import sys as _sys
if '/home/ubuntu/qwen25vl' not in _sys.path:
    _sys.path.insert(0, '/home/ubuntu/qwen25vl')
try:
    from src.modeling_qwen2_5_vl import NeuronQwen2_5_VLForCausalLM as _Qwen25VL_CausalLM
    from src.modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding as _Qwen25VL_ImageEnc
    MODEL_TYPES["qwen2_5_vl"] = {
        "causal-lm": _Qwen25VL_CausalLM,
        "image-encoding": _Qwen25VL_ImageEnc,
    }
except ImportError as _e:
    import logging as _logging
    _logging.getLogger("Neuron").warning(f"Qwen2.5-VL registration failed: {_e}")
# --- End Qwen2.5-VL registration ---
"""
append_to_file(
    NXDI_CONSTANTS,
    check_string="qwen2_5_vl",
    text=registration_block,
    description="Register qwen2_5_vl in MODEL_TYPES",
)


# === 3a. Add NeuronQwen2_5VLForCausalLM wrapper class ===
wrapper_class = """

class NeuronQwen2_5VLForCausalLM(NeuronQwen2VLForCausalLM):
    \"\"\"vLLM wrapper for Qwen2.5-VL. Inherits execute_model/forward from Qwen2-VL wrapper.\"\"\"
    def _save_pretrained_model(self, model_name: str):
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path


"""

# Find exact insertion point: after NeuronQwen3VLForCausalLM class ends
old_after_qwen3 = """class NeuronQwen3VLForCausalLM(NeuronQwen2VLForCausalLM):
    def _save_pretrained_model(self, model_name: str):
        from transformers import Qwen3VLForConditionalGeneration

        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path"""

new_after_qwen3 = old_after_qwen3 + wrapper_class

patch_file(
    MODEL_LOADER,
    check_string="NeuronQwen2_5VLForCausalLM",
    old_string=old_after_qwen3,
    new_string=new_after_qwen3,
    description="Add NeuronQwen2_5VLForCausalLM wrapper class",
)


# === 3b. Add dispatch in get_neuron_model ===
old_dispatch = """    elif architecture == "Qwen3VLForConditionalGeneration":
        model = NeuronQwen3VLForCausalLM(model_config.hf_config)
    else:"""
new_dispatch = """    elif architecture == "Qwen2_5_VLForConditionalGeneration":
        model = NeuronQwen2_5VLForCausalLM(model_config.hf_config)
    elif architecture == "Qwen3VLForConditionalGeneration":
        model = NeuronQwen3VLForCausalLM(model_config.hf_config)
    else:"""

patch_file(
    MODEL_LOADER,
    check_string="Qwen2_5_VLForConditionalGeneration",
    old_string=old_dispatch,
    new_string=new_dispatch,
    description="Add Qwen2.5-VL dispatch in get_neuron_model",
)


# === 4. Add multimodal data routing ===
old_mm_dispatch = """        elif self.model.model.config.model_type == "qwen3_vl":
            # Qwen3-VL uses the same processing as Qwen2-VL
            mm_kwargs = self._process_multi_modal_data_neuron_qwen2_vl(mm_kwargs)"""
new_mm_dispatch = """        elif self.model.model.config.model_type == "qwen2_5_vl":
            # Qwen2.5-VL uses the same processing as Qwen2-VL
            mm_kwargs = self._process_multi_modal_data_neuron_qwen2_vl(mm_kwargs)
        elif self.model.model.config.model_type == "qwen3_vl":
            # Qwen3-VL uses the same processing as Qwen2-VL
            mm_kwargs = self._process_multi_modal_data_neuron_qwen2_vl(mm_kwargs)"""

patch_file(
    MODEL_RUNNER,
    check_string="qwen2_5_vl",
    old_string=old_mm_dispatch,
    new_string=new_mm_dispatch,
    description="Add qwen2_5_vl multimodal data routing",
)


print("\nAll patches applied!")
