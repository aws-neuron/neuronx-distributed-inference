from typing import Callable, List, Optional, Tuple, Union
import math
import logging

from neuronx_distributed_inference.utils.tensor_replacement.registry import (
    TensorReplacementRegister,
)
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def patched_get_last_kv_window(
    window_size,
    position_ids,
    latest_k,
    latest_v,
    windowed_context_encoding_window_idx=-1,
    spec_len=0,
):
    """
    Replaces https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/attention/utils.py#L634
    to convert the index tensor in torch.gather to a LongTensor. Otherwise, the function will error out.
    """
    batch_size, num_head, _, head_dim = latest_k.shape
    latest_pos = torch.amax(position_ids, dim=1)
    if (
        windowed_context_encoding_window_idx >= 1
    ):  # if windowed cte, account for current window offset
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
    gather_idx = (
        gather_idx[:, None, :, None]
        .expand(batch_size, num_head, window_size, head_dim)
        .to(device=latest_k.device)
    )

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
    tensor_capture_hook: Optional[
        Callable
    ] = None,  # Missing argument that triggers a NameError
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

    input_ids, attention_mask, position_ids, seq_ids, sampling_params = (
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
    )

    # Bypass _get_model_outputs entirely. NxDI 0.8.0 added a
    # deepstack_vision_embeds arg that gets forwarded to the CTE/TKG
    # models, but the ImageToTextModelWrapper.input_generator only
    # traces 24 inputs (no deepstack). Calling the models directly
    # with exactly 24 positional args avoids the mismatch.
    _empty = torch.empty(0)

    if self._is_prefill(position_ids):
        # Prefill: vision tensors must match the traced shapes even for
        # text-only inputs.  The CTE NEFF was traced with
        #   vision_embeddings=[batch, seq_len, hidden_size]
        #   vision_mask=[batch, seq_len, 1]
        # so we create zero-filled tensors when they are not provided.
        batch_size = input_ids.shape[0]
        n_active = input_ids.shape[1]  # == bucket seq_len
        if vision_embeddings is None or vision_embeddings.numel() == 0:
            dtype = getattr(self.config, "neuron_config", None)
            dtype = dtype.torch_dtype if dtype is not None else torch.bfloat16
            vision_embeddings = torch.zeros(
                batch_size, n_active, self.config.hidden_size, dtype=dtype
            )
        if vision_mask is None or vision_mask.numel() == 0:
            vision_mask = torch.zeros(batch_size, n_active, 1, dtype=torch.int32)

        outputs = self.context_encoding_model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            _empty,  # prev_hidden
            _empty,  # adapter_ids
            _empty,  # accepted_indices
            _empty,  # current_length
            _empty,  # medusa_mask
            _empty,  # scatter_index
            _empty,  # slot_mapping
            _empty,  # active_block_table
            _empty,  # num_queries
            _empty,  # computed_context_lens
            _empty,  # tile_q_indices
            _empty,  # tile_block_tables
            _empty,  # tile_masks
            _empty,  # inputs_embeds
            _empty,  # kv_cache
            _empty,  # active_mask
            _empty,  # rotary_position_ids
            vision_embeddings,
            vision_mask,
        )
        self.kv_cache_populated = True
        is_run_on_neuron = self.context_encoding_model.is_neuron()
    else:
        # Token generation: vision tensors must be empty (traced as [0]).
        outputs = self.token_generation_model(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            _empty,  # prev_hidden
            _empty,  # adapter_ids
            _empty,  # accepted_indices
            _empty,  # current_length
            _empty,  # medusa_mask
            _empty,  # scatter_index
            _empty,  # slot_mapping
            _empty,  # active_block_table
            _empty,  # num_queries
            _empty,  # computed_context_lens
            _empty,  # tile_q_indices
            _empty,  # tile_block_tables
            _empty,  # tile_masks
            _empty,  # inputs_embeds
            _empty,  # kv_cache
            _empty,  # active_mask
            _empty,  # rotary_position_ids
            _empty,  # vision_embeddings (empty for TKG)
            _empty,  # vision_mask (empty for TKG)
        )
        is_run_on_neuron = self.token_generation_model.is_neuron()

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
            "medusa_args": (
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            ),
            "sampling_params": sampling_params,
            "input_capture_hook": input_capture_hook,
            # "tensor_capture_hook": tensor_capture_hook, -> FIX: Otherwise raises a breaking NameError
            "adapter_ids": adapter_ids,
        }
    )

    tf_args = []
    if self.neuron_config.tensor_replacement_config:
        if hasattr(self, "generation_step"):
            self.generation_step += 1
        else:
            self.generation_step = 1
        reg = TensorReplacementRegister.get_instance()
        tf, masks = reg.step_args(self.generation_step)
        tf_args = tf + masks

    # Only add tf_args if not empty
    if tf_args:
        model_inputs["tf_args"] = tf_args

    # WARNING: This is needed for propagating additional kwargs to the neuron model
    additional_kwargs = self.neuron_model.get_required_kwargs()
    for arg in additional_kwargs:
        model_inputs.update({arg: kwargs.get(arg, None)})

    return model_inputs


# ---------------------------------------------------------------------------
# NKI Flash Attention kernel integration for head_dim > 128
# ---------------------------------------------------------------------------
# Lazy-loaded kernel reference (compiled on first use)
_nki_flash_attn_kernel = None


def _get_nki_flash_attn_kernel():
    """Lazy-load and JIT-compile the NKI flash attention kernel."""
    global _nki_flash_attn_kernel
    if _nki_flash_attn_kernel is not None:
        return _nki_flash_attn_kernel

    from nki_flash_attn_large_d import flash_attn_large_d

    _nki_flash_attn_kernel = flash_attn_large_d
    return _nki_flash_attn_kernel


def _nki_kernel_perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
    """
    Replacement for perform_prefill that uses our custom NKI kernel
    when head_dim > 128. Falls back to the original method otherwise.

    Input: Q, K, V in BHSD layout [batch, num_heads, seq_len, head_dim]
    Output: (attn_output in BHDS layout [batch, num_heads, head_dim, seq_len],
             FlashAttentionStrategy.UNSHARDED_KERNEL)
    """
    from neuronx_distributed_inference.modules.attention.attention_base import (
        FlashAttentionStrategy,
    )
    from neuronx_distributed_inference.modules.attention.attention_base import repeat_kv

    if self.head_dim <= 128:
        return self._orig_perform_prefill(Q, K, V, q_len, bsz, attention_mask)

    # head_dim > 128: use our NKI kernel
    kernel = _get_nki_flash_attn_kernel()

    # Q is BHSD: (bsz, num_heads, q_len, head_dim)
    # Reshape to (bsz * num_heads, q_len, head_dim) for kernel (tp_q=True layout)
    Q_3d = Q.reshape(bsz * self.num_heads, q_len, self.head_dim).to(self.torch_dtype)

    # GQA: replicate K/V heads if needed, then reshape
    num_kv_heads = self.num_key_value_heads
    K_active = K  # already (bsz, num_kv_heads, q_len, head_dim)
    V_active = V
    K_3d = K_active.reshape(bsz * num_kv_heads, q_len, self.head_dim).to(
        self.torch_dtype
    )
    V_3d = V_active.reshape(bsz * num_kv_heads, q_len, self.head_dim).to(
        self.torch_dtype
    )

    # NxDI already applies 1/sqrt(head_dim) scaling to Q in scaled_qk,
    # but for the kernel path it's applied before the kernel call (line 788)
    Q_3d = Q_3d / math.sqrt(self.head_dim)

    # Determine sliding window
    sw = self.sliding_window if self.sliding_window else 0

    # Grid: one program per KV group (kernel handles Q-head fan-out internally)
    grid_bs = bsz * num_kv_heads

    # Call kernel
    # kernel expects: q(bs, seq, d), k(bs_kv, seq, d), v(bs_kv, seq, d)
    # returns: o(bs, d, seq)
    attn_output = kernel[grid_bs](
        Q_3d,
        K_3d,
        V_3d,
        scale=1.0,  # scaling already applied to Q
        use_causal_mask=(attention_mask is not None),
        sliding_window=sw,
    )

    # Reshape output from (bsz * num_heads, head_dim, q_len) -> (bsz, num_heads, head_dim, q_len) = BHDS
    attn_output = attn_output.reshape(bsz, self.num_heads, self.head_dim, q_len)

    return attn_output, FlashAttentionStrategy.UNSHARDED_KERNEL


def _nki_kernel_perform_prefill_windowed_attn(
    self, Q, K, V, q_len, bsz, attention_mask, window_size
):
    """
    Replacement for perform_prefill_windowed_attn that uses our custom NKI kernel
    when head_dim > 128. Falls back to the original method otherwise.

    Input: Q, K, V in BHSD layout [batch, num_heads, seq_len, head_dim]
    Output: (attn_output in BHDS layout, FlashAttentionStrategy.UNSHARDED_KERNEL)
    """
    from neuronx_distributed_inference.modules.attention.attention_base import (
        FlashAttentionStrategy,
    )
    from neuronx_distributed_inference.modules.attention.attention_base import repeat_kv

    if self.head_dim <= 128:
        return self._orig_perform_prefill_windowed_attn(
            Q, K, V, q_len, bsz, attention_mask, window_size
        )

    # head_dim > 128: use our NKI kernel with sliding window
    kernel = _get_nki_flash_attn_kernel()

    Q_3d = Q.reshape(bsz * self.num_heads, q_len, self.head_dim).to(self.torch_dtype)

    # For windowed attn, K/V are already replicated by the caller
    K_active = repeat_kv(K, self.num_key_value_groups)
    V_active = repeat_kv(V, self.num_key_value_groups)
    K_3d = K_active.reshape(bsz * self.num_heads, q_len, self.head_dim).to(
        self.torch_dtype
    )
    V_3d = V_active.reshape(bsz * self.num_heads, q_len, self.head_dim).to(
        self.torch_dtype
    )

    Q_3d = Q_3d / math.sqrt(self.head_dim)

    sw = window_size if window_size else 0

    grid_bs = bsz * self.num_heads  # After repeat_kv, all heads are present

    attn_output = kernel[grid_bs](
        Q_3d,
        K_3d,
        V_3d,
        scale=1.0,
        use_causal_mask=True,
        sliding_window=sw,
    )

    attn_output = attn_output.reshape(bsz, self.num_heads, self.head_dim, q_len)

    return attn_output, FlashAttentionStrategy.UNSHARDED_KERNEL


def _patch_attention_modules_for_nki_kernel():
    """
    Monkey-patch NeuronAttentionBase.perform_prefill and
    perform_prefill_windowed_attn to use our NKI kernel when head_dim > 128.

    This is called once at import time. The original methods are preserved
    as _orig_perform_prefill and _orig_perform_prefill_windowed_attn.
    """
    from neuronx_distributed_inference.modules.attention.attention_base import (
        NeuronAttentionBase,
    )

    # Save originals
    NeuronAttentionBase._orig_perform_prefill = NeuronAttentionBase.perform_prefill
    NeuronAttentionBase._orig_perform_prefill_windowed_attn = (
        NeuronAttentionBase.perform_prefill_windowed_attn
    )

    # Replace with our wrappers
    NeuronAttentionBase.perform_prefill = _nki_kernel_perform_prefill
    NeuronAttentionBase.perform_prefill_windowed_attn = (
        _nki_kernel_perform_prefill_windowed_attn
    )

    logger.info("NKI flash attention kernel patch applied for head_dim > 128")


def apply_patch() -> None:
    import neuronx_distributed_inference.modules.attention.utils as u

    u.get_last_kv_window = patched_get_last_kv_window

    import neuronx_distributed_inference.models.image_to_text_model_base as mm_base

    mm_base.NeuronBaseForImageToText.forward = patched_base_image_to_text_model_forward

    # Patch attention for NKI kernel with head_dim > 128
    _patch_attention_modules_for_nki_kernel()

    try:
        import neuronx_distributed_inference.utils.hf_adapter as hf_adapter

        hf_adapter.HuggingFaceGenerationAdapter.prepare_inputs_for_generation = (
            patched_hf_adapter_prepare_inputs_for_generation
        )
    except ImportError:
        # hf_adapter may fail to import if transformers API changed
        # (e.g., SampleDecoderOnlyOutput renamed). This patch is only
        # needed for HF generate() integration, not core inference.
        pass
