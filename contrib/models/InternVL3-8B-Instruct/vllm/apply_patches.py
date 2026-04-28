#!/usr/bin/env python3
"""Apply vLLM-neuron patches for InternVL3 support."""

import sys

LOADER = "/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py"
RUNNER = "/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py"


def patch_loader():
    with open(LOADER, "r") as f:
        content = f.read()

    # === PATCH 2: _get_model_configs llm_config fallback ===
    old2 = '        config = getattr(config, "text_config", None)'
    new2 = '        config = getattr(config, "text_config", None) or getattr(config, "llm_config", None)'
    assert old2 in content, "Patch 2: old string not found"
    content = content.replace(old2, new2, 1)
    print("Patch 2 applied: llm_config fallback")

    # === PATCH 3: NeuronInternVL3ForCausalLM class ===
    insert_marker = "\ndef _get_model_configs(config: PretrainedConfig) -> str:"
    intern_class = '''

class NeuronInternVL3ForCausalLM(NeuronMultiModalCausalLM):
    """InternVL3 multimodal model using dynamically loaded contrib model."""

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        import importlib

        neuronx_module = importlib.import_module("modeling_internvl3")
        neuronx_model_cls = getattr(neuronx_module, "NeuronInternVL3ForCausalLM")

        default_neuron_config = kwargs["neuron_config"]
        override_neuron_config = _validate_image_to_text_override_neuron_config(
            kwargs["override_neuron_config"]
        )

        vision_neuron_config = copy.deepcopy(default_neuron_config)
        vision_neuron_config.update(
            override_neuron_config.get("vision_neuron_config", {})
        )
        # InternVL3 vision encoder has fused QKV weights
        vision_neuron_config["fused_qkv"] = True
        vision_neuron_config["buckets"] = [1]
        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **vision_neuron_config
        )

        text_neuron_config = copy.deepcopy(default_neuron_config)
        text_neuron_config.update(override_neuron_config.get("text_neuron_config", {}))
        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **text_neuron_config
        )

        config = neuronx_model_cls.get_config_cls().from_pretrained(
            model_name_or_path,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
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

        # Set vision token ID
        self.vision_token_id = 151667  # <IMG_CONTEXT>
        return success, compiled_model_path

'''
    assert insert_marker in content, "Patch 3: insert marker not found"
    content = content.replace(
        insert_marker,
        intern_class + "def _get_model_configs(config: PretrainedConfig) -> str:",
    )
    print("Patch 3 applied: NeuronInternVL3ForCausalLM class")

    # === PATCH 4: get_neuron_model elif ===
    old4 = (
        '    elif architecture == "Qwen3VLForConditionalGeneration":\n'
        "        model = NeuronQwen3VLForCausalLM(model_config.hf_config)\n"
        "    else:\n"
        "        model = NeuronCausalLM(model_config.hf_config)"
    )
    new4 = (
        '    elif architecture == "Qwen3VLForConditionalGeneration":\n'
        "        model = NeuronQwen3VLForCausalLM(model_config.hf_config)\n"
        '    elif architecture == "InternVLChatModel":\n'
        "        from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronInternVL3ForCausalLM\n"
        "        model = NeuronInternVL3ForCausalLM(model_config.hf_config)\n"
        "    else:\n"
        "        model = NeuronCausalLM(model_config.hf_config)"
    )
    assert old4 in content, "Patch 4: old string not found"
    content = content.replace(old4, new4, 1)
    print("Patch 4 applied: get_neuron_model elif")

    # === PATCH 6: _get_neuron_model_cls special case ===
    old6 = (
        '    # Handle Neuron class name (starts with "Neuron") - strip prefix\n'
        '    if architecture.startswith("Neuron") and "For" in architecture:'
    )
    new6 = (
        "    # Special case: InternVLChatModel doesn't follow *For* naming convention\n"
        '    if architecture == "InternVLChatModel":\n'
        "        import importlib\n"
        '        mod = importlib.import_module("modeling_internvl3")\n'
        '        return getattr(mod, "NeuronInternVL3ForCausalLM")\n'
        "\n"
        '    # Handle Neuron class name (starts with "Neuron") - strip prefix\n'
        '    if architecture.startswith("Neuron") and "For" in architecture:'
    )
    assert old6 in content, "Patch 6: old string not found"
    content = content.replace(old6, new6, 1)
    print("Patch 6 applied: _get_neuron_model_cls special case")

    with open(LOADER, "w") as f:
        f.write(content)
    print("All model_loader.py patches written.")


def patch_runner():
    with open(RUNNER, "r") as f:
        content = f.read()

    # === PATCH 5: _process_multi_modal_data_neuron ===
    old5 = (
        '        elif self.model.model.config.model_type == "llama4":\n'
        "            pass  # llama4 doesn't require special processing\n"
        "        else:"
    )
    new5 = (
        '        elif self.model.model.config.model_type == "llama4":\n'
        "            pass  # llama4 doesn't require special processing\n"
        '        elif self.model.model.config.model_type == "internvl_chat":\n'
        "            pass  # InternVL3 processes pixel_values directly\n"
        "        else:"
    )
    assert old5 in content, "Patch 5: old string not found"
    content = content.replace(old5, new5, 1)
    print("Patch 5 applied: internvl_chat handling in _process_multi_modal_data_neuron")

    with open(RUNNER, "w") as f:
        f.write(content)
    print("model_runner.py patch written.")


if __name__ == "__main__":
    patch_loader()
    patch_runner()
    print("\nAll 5 patches applied successfully (patches 2-6).")
