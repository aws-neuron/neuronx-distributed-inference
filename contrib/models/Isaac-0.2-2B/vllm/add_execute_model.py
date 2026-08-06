#!/usr/bin/env python3
"""Add execute_model and forward overrides to NeuronIsaacForConditionalGeneration in model_loader.py."""

import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py")
)

with open(path, "r") as f:
    content = f.read()

# We need to add execute_model and forward methods to NeuronIsaacForConditionalGeneration
# The class currently only has load_weights.

# The old code ends with:
OLD_END = """        self.vision_token_id = tokenizer(
            "<|image_pad|>", add_special_tokens=False
        ).input_ids[0]
        return success, compiled_model_path


def _get_model_configs"""

# The new code adds execute_model and forward after load_weights
NEW_END = '''        self.vision_token_id = tokenizer(
            "<|image_pad|>", add_special_tokens=False
        ).input_ids[0]
        return success, compiled_model_path

    def execute_model(self, model_input, **kwargs):
        """Execute model forward pass for Isaac VLM.

        Unlike Llama4, Isaac uses vision_token_id (set during load_weights)
        instead of model.config.image_token_index for vision mask creation.
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

        # Call the base NeuronMultiModalCausalLM.forward directly
        # (skip Llama4's forward which assumes Llama4-specific pixel_values format)
        hidden_states = NeuronMultiModalCausalLM.forward(
            self,
            input_ids=model_input.input_tokens,
            positions=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            sampling_params=model_input.sampling_params,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
        )
        return hidden_states


def _get_model_configs'''

if OLD_END in content:
    content = content.replace(OLD_END, NEW_END)
    with open(path, "w") as f:
        f.write(content)
    print(
        f"SUCCESS: Added execute_model override to NeuronIsaacForConditionalGeneration in {path}"
    )
else:
    print(f"ERROR: Could not find the expected code block in {path}")
    # Show what's around the class
    import re

    match = re.search(
        r"class NeuronIsaacForConditionalGeneration.*?(?=\nclass |\ndef _get_model_configs)",
        content,
        re.DOTALL,
    )
    if match:
        print(f"Found class at positions {match.start()}-{match.end()}")
        print("Last 200 chars of class:")
        print(match.group()[-200:])
    else:
        print("Could not find the class at all")
