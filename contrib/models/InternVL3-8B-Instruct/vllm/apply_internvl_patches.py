#!/usr/bin/env python3
"""
Apply 6 vLLM patches for InternVL3-8B-Instruct on Neuron.

Patches vllm-neuron 0.5.0 to support InternVL3 via NxDI contrib model.

Requires:
  - NxDI contrib files at /home/ubuntu/internvl3_contrib/src/
  - vLLM-neuron installed at /vllm/ (editable install)

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    python apply_internvl_patches.py [--dry-run] [--revert]
"""

import argparse
import os
import re
import shutil
import sys

VLLM_NEURON_DIR = "/vllm/vllm_neuron/worker"
CONSTANTS_FILE = os.path.join(VLLM_NEURON_DIR, "constants.py")
LOADER_FILE = os.path.join(VLLM_NEURON_DIR, "neuronx_distributed_model_loader.py")
RUNNER_FILE = os.path.join(VLLM_NEURON_DIR, "neuronx_distributed_model_runner.py")

CONTRIB_DIR = "/home/ubuntu/internvl3_contrib/src"


def backup_file(filepath):
    """Create .orig backup if it doesn't exist yet."""
    backup = filepath + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  Backed up: {filepath} -> {backup}")
    else:
        print(f"  Backup exists: {backup}")


def revert_file(filepath):
    """Revert from .orig backup."""
    backup = filepath + ".orig"
    if os.path.exists(backup):
        shutil.copy2(backup, filepath)
        print(f"  Reverted: {filepath}")
    else:
        print(f"  No backup found for: {filepath}")


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read()


def write_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)


def apply_patch_1_constants(dry_run=False):
    """Patch 1: Add InternVLChatModel to NEURON_MULTI_MODAL_MODELS."""
    print("\n--- Patch 1: Register InternVLChatModel in constants.py ---")

    content = read_file(CONSTANTS_FILE)

    if "InternVLChatModel" in content:
        print("  Already applied.")
        return True

    # Add after last entry in NEURON_MULTI_MODAL_MODELS list
    old = '    "Qwen3VLForConditionalGeneration",\n]'
    new = '    "Qwen3VLForConditionalGeneration",\n    "InternVLChatModel",\n]'

    if old not in content:
        print("  ERROR: Cannot find insertion point in constants.py")
        return False

    if not dry_run:
        backup_file(CONSTANTS_FILE)
        content = content.replace(old, new)
        write_file(CONSTANTS_FILE, content)

    print("  Applied: Added 'InternVLChatModel' to NEURON_MULTI_MODAL_MODELS")
    return True


def apply_patch_2_llm_config_fallback(dry_run=False):
    """Patch 2: Handle llm_config fallback in _get_model_configs()."""
    print("\n--- Patch 2: llm_config fallback in _get_model_configs ---")

    content = read_file(LOADER_FILE)

    if "llm_config" in content and 'getattr(config, "llm_config"' in content:
        print("  Already applied.")
        return True

    # Find the line: config = getattr(config, "text_config", None)
    # Replace with fallback that also checks llm_config
    old = '    if architecture in NEURON_MULTI_MODAL_MODELS:\n        config = getattr(config, "text_config", None)'
    new = '    if architecture in NEURON_MULTI_MODAL_MODELS:\n        config = getattr(config, "text_config", None)\n        if config is None:\n            config = getattr(config, "llm_config", None)'

    if old not in content:
        # Try alternative: maybe the indentation is different
        print("  ERROR: Cannot find insertion point for llm_config fallback")
        return False

    # Wait, the second getattr uses the already-None config.
    # We need: try text_config first, if None try llm_config from the ORIGINAL config.
    # Let me fix: store original config first.
    old = '    if architecture in NEURON_MULTI_MODAL_MODELS:\n        config = getattr(config, "text_config", None)'
    new = (
        "    if architecture in NEURON_MULTI_MODAL_MODELS:\n"
        "        # InternVL uses llm_config instead of text_config\n"
        '        config = getattr(config, "text_config", None) or getattr(config, "llm_config", None)'
    )

    if old not in content:
        print("  ERROR: Cannot find insertion point for llm_config fallback")
        return False

    if not dry_run:
        backup_file(LOADER_FILE)
        content = content.replace(old, new)
        write_file(LOADER_FILE, content)

    print("  Applied: Added llm_config fallback in _get_model_configs()")
    return True


def apply_patch_3_internvl_class(dry_run=False):
    """Patch 3: Add NeuronInternVL3ForCausalLM class with dynamic import from contrib."""
    print("\n--- Patch 3: Add NeuronInternVL3ForCausalLM class ---")

    content = read_file(LOADER_FILE)

    if "NeuronInternVL3ForCausalLM" in content:
        print("  Already applied.")
        return True

    # Insert the class right before the get_neuron_model function
    class_code = '''
class NeuronInternVL3ForCausalLM(NeuronMultiModalCausalLM):
    """InternVL3-8B-Instruct via NxDI contrib model."""

    def _save_pretrained_model(self, model_name: str):
        # InternVL3 uses trust_remote_code; save locally if needed
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        """Override to dynamically import contrib model class."""
        import sys
        contrib_path = "/home/ubuntu/internvl3_contrib/src"
        if contrib_path not in sys.path:
            sys.path.insert(0, contrib_path)

        from modeling_internvl3 import NeuronInternVL3ForCausalLM as ContribModel
        from neuronx_distributed_inference.utils.constants import MODEL_TYPES

        # Register the contrib model in MODEL_TYPES so _get_neuron_model_cls finds it
        if "internvl3" not in MODEL_TYPES:
            MODEL_TYPES["internvl3"] = {}
        MODEL_TYPES["internvl3"]["causal-lm"] = ContribModel

        return super().load_weights(model_name_or_path, architecture, **kwargs)

    def execute_model(self, model_input, **kwargs):
        """Forward multimodal inputs for InternVL3."""
        pixel_values = None
        if (
            model_input.multi_modal_kwargs is not None
            and model_input.multi_modal_kwargs.get("pixel_values") is not None
        ):
            pixel_values = model_input.multi_modal_kwargs["pixel_values"]

        # Build vision_mask from image token IDs
        # InternVL3 uses IMG_CONTEXT token (id=151667) for image placeholders
        IMG_CONTEXT_ID = 151667
        vision_mask = (model_input.input_tokens == IMG_CONTEXT_ID).unsqueeze(-1)

        hidden_states = self.forward(
            input_ids=model_input.input_tokens,
            positions=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            sampling_params=model_input.sampling_params,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
            **kwargs,
        )
        return hidden_states

    def forward(
        self,
        input_ids,
        positions,
        input_block_ids,
        sampling_params,
        pixel_values=None,
        vision_mask=None,
        **kwargs,
    ):
        """Forward pass with multimodal support for InternVL3."""
        # Cast vision tensors to the configured dtype
        if pixel_values is not None:
            dtype = self.model.config.vision_config.neuron_config.torch_dtype
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(dtype)
            elif isinstance(pixel_values, list):
                pixel_values = [p.to(dtype) for p in pixel_values]

        with self._reordered(
            input_block_ids,
            input_ids=input_ids,
            positions=positions,
            sampling_params=sampling_params,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
        ) as (sorted_ids, inputs, restore):
            output = self.model(
                inputs["input_ids"].to(torch.int32),
                attention_mask=None,
                position_ids=inputs["positions"].to(torch.int32),
                seq_ids=sorted_ids.flatten().to(torch.int32),
                pixel_values=inputs.get("pixel_values"),
                vision_mask=inputs.get("vision_mask"),
                sampling_params=inputs["sampling_params"],
                **kwargs,
            )

            if self.model.config.neuron_config.on_device_sampling_config:
                output = output.hidden_states
            else:
                output = output.logits[:, -1, :]

            return restore(output)


'''

    # Insert before: def _get_model_configs
    anchor = "\ndef _get_model_configs(config: PretrainedConfig) -> str:"
    if anchor not in content:
        print("  ERROR: Cannot find insertion anchor for NeuronInternVL3ForCausalLM")
        return False

    if not dry_run:
        # backup already done in patch 2, but just in case
        backup_file(LOADER_FILE)
        content = content.replace(anchor, class_code + anchor)
        write_file(LOADER_FILE, content)

    print("  Applied: Added NeuronInternVL3ForCausalLM class")
    return True


def apply_patch_4_get_neuron_model(dry_run=False):
    """Patch 4: Add InternVLChatModel branch in get_neuron_model()."""
    print("\n--- Patch 4: Add InternVLChatModel in get_neuron_model() ---")

    content = read_file(LOADER_FILE)

    if 'architecture == "InternVLChatModel"' in content:
        print("  Already applied.")
        return True

    # Add after Qwen3VL branch
    old = (
        '    elif architecture == "Qwen3VLForConditionalGeneration":\n'
        "        model = NeuronQwen3VLForCausalLM(model_config.hf_config)\n"
        "    else:\n"
        "        model = NeuronCausalLM(model_config.hf_config)"
    )
    new = (
        '    elif architecture == "Qwen3VLForConditionalGeneration":\n'
        "        model = NeuronQwen3VLForCausalLM(model_config.hf_config)\n"
        '    elif architecture == "InternVLChatModel":\n'
        "        model = NeuronInternVL3ForCausalLM(model_config.hf_config)\n"
        "    else:\n"
        "        model = NeuronCausalLM(model_config.hf_config)"
    )

    if old not in content:
        print("  ERROR: Cannot find insertion point for InternVLChatModel branch")
        return False

    if not dry_run:
        backup_file(LOADER_FILE)
        content = content.replace(old, new)
        write_file(LOADER_FILE, content)

    print("  Applied: Added InternVLChatModel branch in get_neuron_model()")
    return True


def apply_patch_5_multimodal_processor(dry_run=False):
    """Patch 5: Add internvl_chat to multimodal data processor."""
    print("\n--- Patch 5: Add internvl_chat in multimodal processor ---")

    content = read_file(RUNNER_FILE)

    if "internvl_chat" in content:
        print("  Already applied.")
        return True

    # Add after llama4 branch
    old = (
        '        elif self.model.model.config.model_type == "llama4":\n'
        "            pass  # llama4 doesn't require special processing\n"
        "        else:"
    )
    new = (
        '        elif self.model.model.config.model_type == "llama4":\n'
        "            pass  # llama4 doesn't require special processing\n"
        '        elif self.model.model.config.model_type == "internvl_chat":\n'
        "            pass  # InternVL3 handles vision processing internally via NxDI contrib\n"
        "        else:"
    )

    if old not in content:
        print("  ERROR: Cannot find insertion point for internvl_chat")
        return False

    if not dry_run:
        backup_file(RUNNER_FILE)
        content = content.replace(old, new)
        write_file(RUNNER_FILE, content)

    print("  Applied: Added internvl_chat to multimodal processor")
    return True


def apply_patch_6_model_cls(dry_run=False):
    """Patch 6: Special-case InternVLChatModel in _get_neuron_model_cls()."""
    print("\n--- Patch 6: Handle InternVLChatModel in _get_neuron_model_cls ---")

    content = read_file(LOADER_FILE)

    # Check if already applied: look for our special case comment in the function
    func_start = content.find("def _get_neuron_model_cls(")
    if func_start != -1 and "Special case: InternVLChatModel" in content[func_start:]:
        print("  Already applied.")
        return True

    # The issue: InternVLChatModel doesn't follow *ForConditionalGeneration pattern.
    # The parser splits on "For" and lowercases. "InternVLChatModel".split("For", 1)
    # gives ["InternVLChatModel"] (no "For"), hitting the KeyError path.
    # We need to add a special case BEFORE the split logic.

    old = '    try:\n        if "For" in architecture:'
    new = (
        "    # Special case: InternVLChatModel does not follow *For* naming convention\n"
        '    if architecture == "InternVLChatModel":\n'
        "        import sys\n"
        '        contrib_path = "/home/ubuntu/internvl3_contrib/src"\n'
        "        if contrib_path not in sys.path:\n"
        "            sys.path.insert(0, contrib_path)\n"
        "        from modeling_internvl3 import NeuronInternVL3ForCausalLM\n"
        "        return NeuronInternVL3ForCausalLM\n"
        "\n"
        '    try:\n        if "For" in architecture:'
    )

    if old not in content:
        print("  ERROR: Cannot find insertion point in _get_neuron_model_cls")
        return False

    if not dry_run:
        backup_file(LOADER_FILE)
        content = content.replace(old, new)
        write_file(LOADER_FILE, content)

    print("  Applied: Added InternVLChatModel special case in _get_neuron_model_cls()")
    return True


def main():
    parser = argparse.ArgumentParser(description="Apply InternVL3 vLLM patches")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show patches without applying"
    )
    parser.add_argument(
        "--revert", action="store_true", help="Revert all patches from backups"
    )
    args = parser.parse_args()

    if args.revert:
        print("Reverting all patches...")
        for f in [CONSTANTS_FILE, LOADER_FILE, RUNNER_FILE]:
            revert_file(f)
        print("\nDone. Reverted to original files.")
        return

    # Verify contrib files exist
    if not os.path.exists(CONTRIB_DIR):
        print(f"ERROR: Contrib directory not found: {CONTRIB_DIR}")
        sys.exit(1)
    for f in [
        "modeling_internvl3.py",
        "modeling_internvl3_text.py",
        "modeling_internvl3_vision.py",
    ]:
        if not os.path.exists(os.path.join(CONTRIB_DIR, f)):
            print(f"ERROR: Missing contrib file: {f}")
            sys.exit(1)

    print("=" * 60)
    print("Applying InternVL3-8B vLLM patches")
    print("=" * 60)
    print(f"vLLM-neuron dir: {VLLM_NEURON_DIR}")
    print(f"Contrib dir: {CONTRIB_DIR}")
    if args.dry_run:
        print("*** DRY RUN -- no files will be modified ***")

    results = []
    results.append(("Patch 1: constants.py", apply_patch_1_constants(args.dry_run)))
    results.append(
        (
            "Patch 2: llm_config fallback",
            apply_patch_2_llm_config_fallback(args.dry_run),
        )
    )
    results.append(
        ("Patch 3: NeuronInternVL3 class", apply_patch_3_internvl_class(args.dry_run))
    )
    results.append(
        ("Patch 4: get_neuron_model", apply_patch_4_get_neuron_model(args.dry_run))
    )
    results.append(
        (
            "Patch 5: multimodal processor",
            apply_patch_5_multimodal_processor(args.dry_run),
        )
    )
    results.append(
        ("Patch 6: _get_neuron_model_cls", apply_patch_6_model_cls(args.dry_run))
    )

    print("\n" + "=" * 60)
    print("Summary:")
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")

    failed = sum(1 for _, s in results if not s)
    if failed:
        print(f"\n{failed} patch(es) failed!")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} patches applied successfully.")


if __name__ == "__main__":
    main()
