#!/usr/bin/env python3
# Copyright 2025 (c) Amazon.com and Affiliates
"""Patch vllm-neuron 0.5.0 to support Isaac-0.2-2B VLM.

Applies the 4-layer registration:
1. constants.py — Add to NEURON_MULTI_MODAL_MODELS
2. model_loader.py — Add NeuronIsaacForConditionalGeneration wrapper class
3. model_loader.py — Add architecture dispatch in get_neuron_model() + fix Sampler import
4. model_runner.py — Add multimodal data routing

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    python patch_vllm_isaac.py
"""

import importlib
import os
import sys


def find_vllm_neuron_path():
    """Find the installed vllm_neuron package path."""
    try:
        spec = importlib.util.find_spec("vllm_neuron")
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
    except (ModuleNotFoundError, AttributeError):
        pass

    # Fallback: search common locations
    for base in sys.path:
        candidate = os.path.join(base, "vllm_neuron")
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot find vllm_neuron package. Is vllm-neuron installed?"
    )


def patch_constants(worker_dir):
    """Layer 1: Add Isaac to NEURON_MULTI_MODAL_MODELS."""
    path = os.path.join(worker_dir, "constants.py")
    with open(path, "r") as f:
        content = f.read()

    if "IsaacForConditionalGeneration" in content:
        print("[constants.py] Already patched — skipping")
        return

    # Add Isaac to the NEURON_MULTI_MODAL_MODELS list
    # Try various insertion points
    for marker in [
        '"Qwen3VLForConditionalGeneration",',
        '"Qwen2VLForConditionalGeneration",',
        '"Llama4ForConditionalGeneration",',
        '"LlavaForConditionalGeneration",',
    ]:
        if marker in content:
            content = content.replace(
                marker,
                marker + '\n    "IsaacForConditionalGeneration",',
            )
            break

    if "IsaacForConditionalGeneration" not in content:
        print("[constants.py] WARNING: Could not find insertion point")
        return

    with open(path, "w") as f:
        f.write(content)
    print(
        "[constants.py] Added IsaacForConditionalGeneration to NEURON_MULTI_MODAL_MODELS"
    )


def patch_model_loader(worker_dir):
    """Layer 2+3: Fix Sampler import, add Isaac wrapper class, add architecture dispatch."""
    path = os.path.join(worker_dir, "neuronx_distributed_model_loader.py")
    with open(path, "r") as f:
        content = f.read()

    # Fix Sampler import (shared issue with Gemma3)
    if "from vllm.v1.sample import sampler as Sampler" in content:
        content = content.replace(
            "from vllm.v1.sample import sampler as Sampler",
            "from vllm.v1.sample.sampler import Sampler",
        )
        print("[model_loader.py] Fixed Sampler import")

    if "NeuronIsaacForConditionalGeneration" in content:
        print("[model_loader.py] Already patched — skipping")
        with open(path, "w") as f:
            f.write(content)
        return

    # --- Add Isaac wrapper class before get_neuron_model or _get_model_configs ---
    isaac_class = '''

class NeuronIsaacForConditionalGeneration(NeuronLlama4ForCausalLM):
    """Isaac VLM using dynamically loaded NeuronIsaacForConditionalGeneration from contrib."""

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        import importlib

        neuronx_module = importlib.import_module("isaac_neuron.modeling_isaac")
        neuronx_model_cls = getattr(neuronx_module, "NeuronIsaacForConditionalGeneration")

        default_neuron_config = kwargs["neuron_config"]
        override_neuron_config = _validate_image_to_text_override_neuron_config(
            kwargs["override_neuron_config"]
        )

        vision_neuron_config = copy.deepcopy(default_neuron_config)
        vision_neuron_config.update(
            override_neuron_config.get("vision_neuron_config", {})
        )
        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **vision_neuron_config
        )

        text_neuron_config = copy.deepcopy(default_neuron_config)
        text_neuron_config.update(override_neuron_config.get("text_neuron_config", {}))
        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **text_neuron_config
        )

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        config = neuronx_model_cls.get_config_cls()(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            load_config=load_pretrained_config(hf_config=hf_config),
        )

        success, compiled_model_path, _ = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, config=config, **kwargs
        )

        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(model_name_or_path)

            self._compile_and_load_model(
                model_name_or_path, neuronx_model_cls, config, compiled_model_path
            )

        # Load tokenizer to get vision token ID
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.vision_token_id = tokenizer(
            "<|image_pad|>", add_special_tokens=False
        ).input_ids[0]
        return success, compiled_model_path

    def execute_model(self, model_input, **kwargs):
        """Execute model forward pass for Isaac VLM.

        Uses vision_token_id for vision mask (not model.config.image_token_index),
        calls base forward directly, and handles logits->token_id conversion since
        the Isaac compiled model returns logits (not on-device sampled tokens).
        """
        vision_mask = (
            model_input.input_tokens == self.vision_token_id
        ).unsqueeze(-1)

        pixel_values = None
        if (
            model_input.multi_modal_kwargs is not None
            and model_input.multi_modal_kwargs.get("pixel_values") is not None
        ):
            pixel_values = model_input.multi_modal_kwargs["pixel_values"]

        # Call base forward with Isaac-specific args
        with self._reordered(
            model_input.input_block_ids,
            input_ids=model_input.input_tokens,
            positions=model_input.position_ids,
            sampling_params=model_input.sampling_params,
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
            )

            # Isaac model returns logits (not on-device sampled tokens)
            # Extract last-token logits and argmax to get token IDs
            if hasattr(output, "hidden_states") and isinstance(output.hidden_states, torch.Tensor) and output.hidden_states.numel() > 0:
                result = output.hidden_states
            else:
                logits = output.logits[:, -1, :]  # [batch, vocab]
                result = torch.argmax(logits, dim=-1)  # [batch] - token IDs

            return restore(result)

'''

    # Insert class before _get_model_configs or get_neuron_model
    for marker in ["def _get_model_configs(", "def get_neuron_model("]:
        if marker in content:
            idx = content.index(marker)
            content = content[:idx] + isaac_class + "\n" + content[idx:]
            print("[model_loader.py] Added NeuronIsaacForConditionalGeneration class")
            break
    else:
        print("[model_loader.py] WARNING: Could not find insertion point for class")

    # --- Add architecture dispatch in get_neuron_model() ---
    # This function is in model_loader.py and dispatches based on architecture string
    dispatch_markers = [
        'elif architecture == "Qwen3VLForConditionalGeneration":',
        'elif architecture == "Qwen2VLForConditionalGeneration":',
        'elif architecture == "Llama4ForConditionalGeneration":',
    ]

    for marker in dispatch_markers:
        if marker in content:
            # Find the line after this elif + its body
            idx = content.index(marker)
            # Find next elif or else
            search_start = idx + len(marker)
            next_elif = content.find("\n    elif ", search_start)
            next_else = content.find("\n    else:", search_start)

            # Pick the closest one
            candidates = [c for c in [next_elif, next_else] if c > 0]
            if candidates:
                insert_point = min(candidates)
                insert_text = (
                    '\n    elif architecture == "IsaacForConditionalGeneration":'
                    "\n        model = NeuronIsaacForConditionalGeneration(model_config.hf_config)"
                )
                content = content[:insert_point] + insert_text + content[insert_point:]
                print(
                    "[model_loader.py] Added Isaac architecture dispatch in get_neuron_model()"
                )
                break
    else:
        print("[model_loader.py] WARNING: Could not find dispatch insertion point")

    with open(path, "w") as f:
        f.write(content)


def patch_model_runner(worker_dir):
    """Layer 4: Add multimodal data routing for Isaac model_type."""
    path = os.path.join(worker_dir, "neuronx_distributed_model_runner.py")
    with open(path, "r") as f:
        content = f.read()

    if '"isaac"' in content or "'isaac'" in content:
        print("[model_runner.py] Already patched — skipping")
        return

    changed = False

    # Add multimodal data routing for Isaac
    # Isaac uses pass-through (no special multimodal preprocessing needed, like Llama4)
    # Look for existing qwen3_vl routing and add after it
    routing_markers = [
        'elif self.model.model.config.model_type == "qwen3_vl":',
        'elif self.model.model.config.model_type == "qwen2_vl":',
        'elif self.model.model.config.model_type == "llava":',
    ]

    for marker in routing_markers:
        if marker in content:
            # Find the line(s) after this elif
            idx = content.index(marker)
            search_start = idx + len(marker)
            # Find next elif or else
            next_elif = content.find("\n        elif ", search_start)
            next_else = content.find("\n        else:", search_start)

            candidates = [c for c in [next_elif, next_else] if c > 0]
            if candidates:
                insert_point = min(candidates)
                insert_text = (
                    '\n        elif self.model.model.config.model_type == "isaac":'
                    "\n            pass  # Isaac does not require special multimodal preprocessing"
                )
                content = content[:insert_point] + insert_text + content[insert_point:]
                print("[model_runner.py] Added Isaac multimodal data routing")
                changed = True
                break

    if not changed:
        # Try alternative: check if there's a list-style routing
        for list_marker in [
            "in ['llama4'",
            'in ["llama4"',
            "in ['llama4', 'gemma3'",
            'in ["llama4", "gemma3"',
        ]:
            if list_marker in content:
                content = content.replace(
                    list_marker,
                    list_marker.rstrip("'\"") + "', 'isaac'"
                    if "'" in list_marker
                    else list_marker.rstrip("'\"") + '", "isaac"',
                )
                print("[model_runner.py] Added Isaac to multimodal list routing")
                changed = True
                break

    if not changed:
        print(
            "[model_runner.py] WARNING: Could not add multimodal routing — may need manual patch"
        )

    with open(path, "w") as f:
        f.write(content)


def main():
    vllm_neuron_path = find_vllm_neuron_path()
    worker_dir = os.path.join(vllm_neuron_path, "worker")
    print(f"Found vllm_neuron at: {vllm_neuron_path}")
    print(f"Worker directory: {worker_dir}")
    print()

    patch_constants(worker_dir)
    patch_model_loader(worker_dir)
    patch_model_runner(worker_dir)

    print()
    print("All patches applied. To use Isaac with vLLM:")
    print("  export VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference'")
    print("  export NEURON_COMPILED_ARTIFACTS='/mnt/models/traced_model/Isaac-0.2-2B'")
    print(
        "  PYTHONPATH='.../Isaac-0.2-2B/src:$PYTHONPATH' python -m vllm.entrypoints.openai.api_server ..."
    )


if __name__ == "__main__":
    main()
