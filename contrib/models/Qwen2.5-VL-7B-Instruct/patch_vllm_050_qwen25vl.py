"""
Patch vllm-neuron 0.5.0 to add Qwen2.5-VL support.

vllm-neuron 0.5.0 is a pip-installable plugin (not a fork).
Install it first: git clone --branch release-0.5.0 https://github.com/vllm-project/vllm-neuron.git
                  pip install -e .

This script patches 4 files:
1. vllm_neuron/worker/constants.py - Add to NEURON_MULTI_MODAL_MODELS
2. NxDI constants.py - Register qwen2_5_vl in MODEL_TYPES
3. vllm_neuron/worker/neuronx_distributed_model_loader.py - Add wrapper class + dispatches
4. vllm_neuron/worker/neuronx_distributed_model_runner.py - Add multimodal data routing

Usage:
    python patch_vllm_050_qwen25vl.py [--vllm-dir /path/to/vllm-neuron] [--nxdi-constants /path/to/constants.py]

    If --vllm-dir is not specified, auto-detects from vllm_neuron package location.
    If --nxdi-constants is not specified, auto-detects from neuronx_distributed_inference package location.
"""

import argparse
import os
import sys


def patch_file(path, check_string, old_string, new_string, description):
    """Patch a file by replacing old_string with new_string."""
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path} - {description}")
        return False

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
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path} - {description}")
        return False

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


def find_vllm_neuron_dir():
    """Auto-detect vllm-neuron installation directory."""
    try:
        import vllm_neuron

        return os.path.dirname(os.path.dirname(vllm_neuron.__file__))
    except ImportError:
        # Try common locations
        for path in [
            os.path.expanduser("~/vllm-neuron"),
            "/home/ubuntu/vllm-neuron",
        ]:
            if os.path.exists(
                os.path.join(path, "vllm_neuron", "worker", "constants.py")
            ):
                return path
    return None


def find_nxdi_constants():
    """Auto-detect NxDI constants.py location."""
    try:
        import neuronx_distributed_inference.utils.constants as c

        return c.__file__
    except ImportError:
        pass
    # Fallback: search common venv locations
    for venv in [
        "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13",
    ]:
        path = os.path.join(venv, "lib")
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                if (
                    "constants.py" in files
                    and "neuronx_distributed_inference/utils" in root
                ):
                    return os.path.join(root, "constants.py")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Patch vllm-neuron 0.5.0 for Qwen2.5-VL support"
    )
    parser.add_argument("--vllm-dir", help="Path to vllm-neuron repo/install directory")
    parser.add_argument("--nxdi-constants", help="Path to NxDI constants.py")
    parser.add_argument(
        "--qwen25vl-dir",
        default="/home/ubuntu/qwen25vl",
        help="Path to qwen2.5-vl model code on instance (default: /home/ubuntu/qwen25vl)",
    )
    args = parser.parse_args()

    # Resolve paths
    vllm_dir = args.vllm_dir or find_vllm_neuron_dir()
    if vllm_dir is None:
        print("ERROR: Could not find vllm-neuron installation. Use --vllm-dir.")
        sys.exit(1)

    nxdi_constants = args.nxdi_constants or find_nxdi_constants()
    if nxdi_constants is None:
        print("ERROR: Could not find NxDI constants.py. Use --nxdi-constants.")
        sys.exit(1)

    VLLM_CONSTANTS = os.path.join(vllm_dir, "vllm_neuron", "worker", "constants.py")
    MODEL_LOADER = os.path.join(
        vllm_dir, "vllm_neuron", "worker", "neuronx_distributed_model_loader.py"
    )
    MODEL_RUNNER = os.path.join(
        vllm_dir, "vllm_neuron", "worker", "neuronx_distributed_model_runner.py"
    )
    NXDI_CONSTANTS = nxdi_constants
    qwen25vl_dir = args.qwen25vl_dir

    print(f"vllm-neuron dir: {vllm_dir}")
    print(f"NxDI constants:  {NXDI_CONSTANTS}")
    print(f"Qwen2.5-VL dir:  {qwen25vl_dir}")
    print()

    patched = 0
    skipped = 0
    errors = 0

    # === 1. Add to NEURON_MULTI_MODAL_MODELS ===
    result = patch_file(
        VLLM_CONSTANTS,
        check_string="Qwen2_5_VLForConditionalGeneration",
        old_string='    "Qwen3VLForConditionalGeneration",\n]',
        new_string='    "Qwen2_5_VLForConditionalGeneration",\n    "Qwen3VLForConditionalGeneration",\n]',
        description="Add Qwen2_5_VL to NEURON_MULTI_MODAL_MODELS",
    )
    patched += 1 if result else 0

    # === 2. Register in NxDI MODEL_TYPES ===
    registration_block = f"""

# --- Qwen2.5-VL registration (patched for qwen2.5-vl project) ---
import sys as _sys
if '{qwen25vl_dir}' not in _sys.path:
    _sys.path.insert(0, '{qwen25vl_dir}')
try:
    from src.modeling_qwen2_5_vl import NeuronQwen2_5_VLForCausalLM as _Qwen25VL_CausalLM
    from src.modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding as _Qwen25VL_ImageEnc
    MODEL_TYPES["qwen2_5_vl"] = {{
        "causal-lm": _Qwen25VL_CausalLM,
        "image-encoding": _Qwen25VL_ImageEnc,
    }}
except ImportError as _e:
    import logging as _logging
    _logging.getLogger("Neuron").warning(f"Qwen2.5-VL registration failed: {{_e}}")
# --- End Qwen2.5-VL registration ---
"""
    result = append_to_file(
        NXDI_CONSTANTS,
        check_string="qwen2_5_vl",
        text=registration_block,
        description="Register qwen2_5_vl in MODEL_TYPES",
    )
    patched += 1 if result else 0

    # === 3a. Add NeuronQwen2_5VLForCausalLM wrapper class ===
    wrapper_class = '''

class NeuronQwen2_5VLForCausalLM(NeuronQwen2VLForCausalLM):
    """vLLM wrapper for Qwen2.5-VL. Inherits execute_model/forward from Qwen2-VL wrapper."""
    def _save_pretrained_model(self, model_name: str):
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path


'''

    old_after_qwen3 = """class NeuronQwen3VLForCausalLM(NeuronQwen2VLForCausalLM):
    def _save_pretrained_model(self, model_name: str):
        from transformers import Qwen3VLForConditionalGeneration

        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path"""

    new_after_qwen3 = old_after_qwen3 + wrapper_class

    result = patch_file(
        MODEL_LOADER,
        check_string="NeuronQwen2_5VLForCausalLM",
        old_string=old_after_qwen3,
        new_string=new_after_qwen3,
        description="Add NeuronQwen2_5VLForCausalLM wrapper class",
    )
    patched += 1 if result else 0

    # === 3b. Add dispatch in get_neuron_model ===
    old_dispatch = """    elif architecture == "Qwen3VLForConditionalGeneration":
        model = NeuronQwen3VLForCausalLM(model_config.hf_config)
    else:"""
    new_dispatch = """    elif architecture == "Qwen2_5_VLForConditionalGeneration":
        model = NeuronQwen2_5VLForCausalLM(model_config.hf_config)
    elif architecture == "Qwen3VLForConditionalGeneration":
        model = NeuronQwen3VLForCausalLM(model_config.hf_config)
    else:"""

    result = patch_file(
        MODEL_LOADER,
        check_string='architecture == "Qwen2_5_VLForConditionalGeneration"',
        old_string=old_dispatch,
        new_string=new_dispatch,
        description="Add Qwen2.5-VL dispatch in get_neuron_model",
    )
    patched += 1 if result else 0

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

    result = patch_file(
        MODEL_RUNNER,
        check_string="qwen2_5_vl",
        old_string=old_mm_dispatch,
        new_string=new_mm_dispatch,
        description="Add qwen2_5_vl multimodal data routing",
    )
    patched += 1 if result else 0

    print(f"\nDone! {patched} patches applied.")


if __name__ == "__main__":
    main()
