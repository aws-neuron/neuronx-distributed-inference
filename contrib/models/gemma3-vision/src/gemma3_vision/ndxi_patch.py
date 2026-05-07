from typing import Callable, List, Optional, Tuple, Union

from neuronx_distributed_inference.utils.tensor_replacement.registry import TensorReplacementRegister
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def patched_get_last_kv_window(window_size, position_ids, latest_k, latest_v, windowed_context_encoding_window_idx=-1, spec_len=0):
    """
    Replaces https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/attention/utils.py#L634
    to convert the index tensor in torch.gather to a LongTensor. Otherwise, the function will error out.
    """
    batch_size, num_head, _, head_dim = latest_k.shape
    latest_pos = torch.amax(position_ids, dim=1)
    if windowed_context_encoding_window_idx >= 1:  # if windowed cte, account for current window offset
        latest_pos -= windowed_context_encoding_window_idx * window_size

    # True window size
    window_size = window_size - 1 + spec_len - 1 if spec_len > 0 else window_size - 1

    end_idx = (latest_pos + 1).clamp(min=window_size)
    start_idx = (end_idx - window_size).clamp(min=0)
    orig_indices = start_idx[:, None] + torch.arange(window_size)

    # Calculate per-batch left shifts
    left_shifts = (window_size - (end_idx % window_size)) % window_size
    base = torch.arange(window_size).expand(batch_size, window_size)
    shifted_idx = (base + left_shifts[:, None]) % window_size

    # Determine per-batch shifted gather indices
    gather_idx = torch.gather(orig_indices, dim=1, index=shifted_idx.long())
    gather_idx = gather_idx[:, None, :, None].expand(batch_size, num_head, window_size, head_dim).to(device=latest_k.device)

    # Gather to create non-physically contiguous KV cache
    latest_k = torch.gather(latest_k, dim=2, index=gather_idx.long())
    latest_v = torch.gather(latest_v, dim=2, index=gather_idx.long())
    return latest_k, latest_v


def patched_base_image_to_text_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    seq_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    sampling_params: Optional[torch.FloatTensor] = None,
    prev_hidden: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    adapter_ids: Optional[torch.LongTensor] = None,
    medusa_args=None,
    return_dict: Optional[bool] = None,
    llava_args: Optional[List] = [],
    input_capture_hook: Optional[Callable] = None,
    slot_mapping: Optional[torch.LongTensor] = None,
    block_table: Optional[torch.LongTensor] = None,
    full_context_lens: Optional[torch.LongTensor] = None,
    computed_context_lens: Optional[torch.LongTensor] = None,
    vision_embeddings: Optional[torch.FloatTensor] = None,
    vision_mask: Optional[torch.BoolTensor] = None,
    tensor_capture_hook: Optional[Callable] = None, # Missing argument that triggers a NameError
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """
    # infer attention_mask from position_ids if not provided
    if attention_mask is None:
        attention_mask = self._infer_attention_mask(position_ids)

    if seq_ids is None:
        seq_ids = torch.arange(input_ids.shape[0])

    self.preprocess_inputs(
        input_ids=input_ids,
        seq_ids=seq_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        sampling_params=sampling_params,
        prev_hidden=prev_hidden,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        adapter_ids=adapter_ids,
        medusa_args=medusa_args,
        return_dict=return_dict,
        llava_args=llava_args,
        input_capture_hook=input_capture_hook,
        slot_mapping=slot_mapping,
        block_table=block_table,
        full_context_lens=full_context_lens,
        computed_context_lens=computed_context_lens,
    )

    if self.async_mode:
        outputs, is_run_on_neuron = self._get_model_outputs_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            adapter_ids=adapter_ids,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            medusa_args=medusa_args,
            llava_args=llava_args,
        )
    else:
        outputs, is_run_on_neuron = self._get_model_outputs(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            prev_hidden,
            adapter_ids,
            vision_embeddings,
            vision_mask,
            medusa_args,
            llava_args,
        )

    generation_model = self.get_generation_model()
    if not generation_model.is_neuron():
        self._copy_past_key_values(outputs)

    # Process outputs
    constructed_outputs = self._get_constructed_outputs(outputs, is_run_on_neuron)

    # Apply tensor_capture_hook if provided and tensors are captured
    if tensor_capture_hook and constructed_outputs.captured_tensors:
        # Apply the hook if captured tensors are found
        tensor_capture_hook(self, constructed_outputs.captured_tensors)

    return constructed_outputs


def patched_hf_adapter_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        sampling_params=None,
        adapter_ids=None,
        **kwargs,
    ):
        # Store KV cache flag before forward pass.
        self.prev_kv_cache_populated = self.neuron_model.kv_cache_populated
        if self.neuron_model.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        accepted_indices = kwargs.get("accepted_indices", None)
        current_length = kwargs.get("current_length", None)
        medusa_mask = kwargs.get("medusa_mask", None)
        scatter_index = kwargs.get("scatter_index", None)
        position_ids = kwargs.get("position_ids", None)
        input_capture_hook = kwargs.get("input_capture_hook", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            if self.input_start_offsets:
                if len(self.input_start_offsets) > 1:
                    position_ids += torch.tensor(self.input_start_offsets, dtype=position_ids.dtype, device=position_ids.device)[:, None]
                else:
                    position_ids += self.input_start_offsets[0]
                for i, offset in enumerate(self.input_start_offsets):
                    position_ids[i, 0:offset] = torch.arange(offset)
            else:
                position_ids.masked_fill_(attention_mask == 0, 1)

            if self.neuron_model.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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
                #"tensor_capture_hook": tensor_capture_hook, -> FIX: Otherwise raises a breaking NameError
                "adapter_ids": adapter_ids
            }
        )

        tf_args = []
        if self.neuron_config.tensor_replacement_config:
            if hasattr(self, 'generation_step'):
                self.generation_step += 1
            else:
                self.generation_step = 1
            reg = TensorReplacementRegister.get_instance()
            tf , masks = reg.step_args(self.generation_step)
            tf_args = tf + masks

        # Only add tf_args if not empty
        if tf_args:
            model_inputs["tf_args"] = tf_args

        # WARNING: This is needed for propagating additional kwargs to the neuron model
        additional_kwargs = self.neuron_model.get_required_kwargs()
        for arg in additional_kwargs:
            model_inputs.update({arg: kwargs.get(arg, None)})

        return model_inputs


def apply_patch() -> None:
    import neuronx_distributed_inference.modules.attention.utils as u
    u.get_last_kv_window = patched_get_last_kv_window
    import neuronx_distributed_inference.models.image_to_text_model_base as mm_base
    mm_base.NeuronBaseForImageToText.forward = patched_base_image_to_text_model_forward
    import neuronx_distributed_inference.utils.hf_adapter as hf_adapter
    hf_adapter.HuggingFaceGenerationAdapter.prepare_inputs_for_generation = patched_hf_adapter_prepare_inputs_for_generation
