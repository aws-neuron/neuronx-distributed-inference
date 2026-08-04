# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Upstream compatibility shims for Qwen2.5-Omni.
#
# These patches sit here (not in src/neuronx_distributed_inference/) so that
# this contrib package has zero direct invasion on the upstream source tree.
# Each patch is idempotent: if upstream has already fixed the issue the
# original object stays unchanged.

import logging
import inspect

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

logger = logging.getLogger(__name__)


def _patch_prepare_inputs_for_generation():
    """Fix upstream NameError in HuggingFaceGenerationAdapter.

    Upstream's prepare_inputs_for_generation builds model_inputs with
    "tensor_capture_hook": tensor_capture_hook
    but never defines tensor_capture_hook as a parameter or extracts it from
    **kwargs, so adapter.generate() raises NameError. Qwen2.5-Omni's Talker
    drives generation via adapter.generate(), so this breaks it out-of-the-box.

    This wrapper re-dispatches to the upstream method with tensor_capture_hook
    materialized as a local so the original body sees a defined name.
    """
    src = inspect.getsource(HuggingFaceGenerationAdapter.prepare_inputs_for_generation)
    references_hook = '"tensor_capture_hook": tensor_capture_hook' in src
    already_extracted = 'tensor_capture_hook = kwargs.get("tensor_capture_hook"' in src
    if already_extracted or not references_hook:
        return  # upstream already consistent

    original = HuggingFaceGenerationAdapter.prepare_inputs_for_generation

    def patched(self, input_ids, *args, **kwargs):
        # Inject the missing local via the frame's globals is fragile; the
        # cleanest fix is to guarantee the name exists in the caller's scope.
        # Since Python resolves bare names via function locals/globals, and
        # the original body has no local binding, we surface it through the
        # **kwargs extraction idiom already used for input_capture_hook.
        # The patched body below mirrors upstream but extracts the hook.
        import torch  # noqa: F401
        self.prev_kv_cache_populated = self.neuron_model.kv_cache_populated
        if self.neuron_model.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        past_key_values = kwargs.pop("past_key_values", None)
        attention_mask = kwargs.pop("attention_mask", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        sampling_params = kwargs.pop("sampling_params", None)
        adapter_ids = kwargs.pop("adapter_ids", None)
        divergence_idx = kwargs.pop("divergence_idx", None)

        accepted_indices = kwargs.get("accepted_indices", None)
        current_length = kwargs.get("current_length", None)
        medusa_mask = kwargs.get("medusa_mask", None)
        scatter_index = kwargs.get("scatter_index", None)
        position_ids = kwargs.get("position_ids", None)
        input_capture_hook = kwargs.get("input_capture_hook", None)
        tensor_capture_hook = kwargs.get("tensor_capture_hook", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            if self.input_start_offsets:
                if len(self.input_start_offsets) > 1:
                    import torch as _torch
                    position_ids += _torch.tensor(
                        self.input_start_offsets,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )[:, None]
                else:
                    position_ids += self.input_start_offsets[0]
                import torch as _torch
                for i, offset in enumerate(self.input_start_offsets):
                    position_ids[i, 0:offset] = _torch.arange(offset)
            else:
                position_ids.masked_fill_(attention_mask == 0, 1)

            if self.neuron_model.kv_cache_populated:
                import torch as _torch
                position_ids = _torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
                "attention_mask": attention_mask,
                "medusa_args": (accepted_indices, current_length, medusa_mask, scatter_index),
                "sampling_params": sampling_params,
                "input_capture_hook": input_capture_hook,
                "tensor_capture_hook": tensor_capture_hook,
                "adapter_ids": adapter_ids,
            }
        )

        tf_args = []
        if self.neuron_config.tensor_replacement_config:
            from neuronx_distributed_inference.utils.tensor_replacement.registry import (
                TensorReplacementRegister,
            )
            reg = TensorReplacementRegister.get_instance()
            tf, masks = reg.step_args(
                self.generation_step,
                divergence_idx=True if divergence_idx else False,
            )
            tf_args = tf + masks

        if tf_args:
            model_inputs["tf_args"] = tf_args

        additional_kwargs = self.neuron_model.get_required_kwargs()
        for arg in additional_kwargs:
            model_inputs.update({arg: kwargs.get(arg, None)})

        return model_inputs

    HuggingFaceGenerationAdapter.prepare_inputs_for_generation = patched
    HuggingFaceGenerationAdapter.prepare_inputs_for_generation.__doc__ = (
        "[Qwen2.5-Omni contrib patched] " + (original.__doc__ or "")
    )
    logger.info(
        "Qwen2.5-Omni contrib: patched HuggingFaceGenerationAdapter."
        "prepare_inputs_for_generation to extract tensor_capture_hook from kwargs."
    )


_patch_prepare_inputs_for_generation()
