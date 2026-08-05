# Upstream compatibility shims for Qwen3-Omni.
# Reuses the same patches as Qwen2.5-Omni since the upstream issues are shared.

import logging
import inspect

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

logger = logging.getLogger(__name__)


def _patch_prepare_inputs_for_generation():
    """Fix upstream NameError: tensor_capture_hook not defined."""
    src = inspect.getsource(HuggingFaceGenerationAdapter.prepare_inputs_for_generation)
    references_hook = '"tensor_capture_hook": tensor_capture_hook' in src
    already_extracted = 'tensor_capture_hook = kwargs.get("tensor_capture_hook"' in src
    if already_extracted or not references_hook:
        return

    original = HuggingFaceGenerationAdapter.prepare_inputs_for_generation

    def patched(self, input_ids, *args, **kwargs):
        import torch
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
                    position_ids += torch.tensor(
                        self.input_start_offsets,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )[:, None]
                else:
                    position_ids += self.input_start_offsets[0]
                for i, offset in enumerate(self.input_start_offsets):
                    position_ids[i, 0:offset] = torch.arange(offset)
            else:
                position_ids.masked_fill_(attention_mask == 0, 1)

            if self.neuron_model.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
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
    logger.info(
        "Qwen3-Omni contrib: patched HuggingFaceGenerationAdapter."
        "prepare_inputs_for_generation to extract tensor_capture_hook from kwargs."
    )


_patch_prepare_inputs_for_generation()


def _patch_vision_wrapper_load_state_dict():
    """Remap thinker.visual.* -> model.visual.* in safetensors loading.

    Qwen3-VL upstream expects model.visual.pos_embed.weight but
    Qwen3-Omni safetensors store it as thinker.visual.pos_embed.weight.
    """
    import neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision as vmod

    _original_load = vmod.load_state_dict

    def _remapped_load(state_dict_dir):
        sd = _original_load(state_dict_dir)
        if "model.visual.pos_embed.weight" not in sd and "thinker.visual.pos_embed.weight" in sd:
            remapped = {}
            for k, v in sd.items():
                if k.startswith("thinker.visual."):
                    remapped["model.visual." + k[len("thinker.visual."):]] = v
                else:
                    remapped[k] = v
            return remapped
        return sd

    vmod.load_state_dict = _remapped_load
    logger.info(
        "Qwen3-Omni contrib: patched vision wrapper load_state_dict "
        "to remap thinker.visual.* -> model.visual.*"
    )


_patch_vision_wrapper_load_state_dict()


def _patch_tensor_registry_clear():
    """Fix upstream NxD bug: TensorRegistry.clear() wipes modules_to_capture.

    Inside ``NeuronBaseModel._get_captured_tensors`` (called once per HLO
    trace, once per bucket), the final line is ``registry.clear()``. Upstream
    ``clear()`` replaces ``model_info`` with a fresh
    ``CapturedModelInfo([], 10, False)`` — resetting ``modules_to_capture``.
    The forward hooks installed by ``enable_tensor_capture`` still fire on
    the next bucket's trace, but ``register_tensor`` now falls through to
    the "manual" branch (module name no longer in ``modules_to_capture``).

    Net effect: only the FIRST bucket to trace captures a real tensor; every
    subsequent bucket emits the empty fallback ``torch.zeros(1, dtype=bf16)``,
    making layer-hidden-state capture (``layers.23`` for the talker pipeline)
    unusable for any prompt that doesn't fit the first bucket.

    Fix: preserve the configured modules/flags through clear() by stashing
    them on the singleton.
    """
    from neuronx_distributed.utils.tensor_capture.registry import (
        CapturedModelInfo, TensorRegistry,
    )

    if getattr(TensorRegistry, "_nxdi_clear_patched", False):
        return

    _orig_configure = TensorRegistry.configure

    def configure(self, enabled=False, modules=None, max_tensors=None, capture_inputs=False):
        cfg_modules = list(modules or [])
        if cfg_modules:
            self._nxdi_last_modules = cfg_modules
            self._nxdi_last_max_tensors = max_tensors
            self._nxdi_last_capture_inputs = capture_inputs
        _orig_configure(self, enabled=enabled, modules=modules,
                        max_tensors=max_tensors, capture_inputs=capture_inputs)

    def clear(self):
        modules = getattr(self, "_nxdi_last_modules", [])
        max_tensors = getattr(self, "_nxdi_last_max_tensors", 10)
        capture_inputs = getattr(self, "_nxdi_last_capture_inputs", False)
        self.model_info = CapturedModelInfo(modules, max_tensors, capture_inputs)

    TensorRegistry.configure = configure
    TensorRegistry.clear = clear
    TensorRegistry._nxdi_clear_patched = True
    logger.info(
        "Qwen3-Omni contrib: patched TensorRegistry.clear to preserve "
        "modules_to_capture across bucket traces."
    )


_patch_tensor_registry_clear()
