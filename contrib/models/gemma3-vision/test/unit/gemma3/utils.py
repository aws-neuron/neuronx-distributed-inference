import torch

# context-encoding, non-sliding
# ref: https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_base.py#L209
def create_context_attn_mask(batch_size, n_positions, attention_mask=None, padding_side="right"):
    # Lower triangle causal mask for classic attention
    mask = torch.full(
        (n_positions, n_positions), True
    ).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)

    if padding_side == "right":
        return mask
    else:
        expanded_mask = (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, n_positions, n_positions)
            .to(torch.bool)
        )
        return torch.logical_and(mask, expanded_mask)

# context-encoding, sliding
# ref: https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_base.py#L245
def create_windowed_attn_mask_cte(batch_size, config) -> torch.Tensor:
    # Create a causal, window attention mask. E.g. n = 5, window_size = 2, mask is:
    #  [[1 0 0 0 0]
    #   [1 1 0 0 0]
    #   [0 1 1 0 0]
    #   [0 0 1 1 0]
    #   [0 0 0 1 1]]
    n_positions, window_size = config.neuron_config.n_positions, config.sliding_window
    i = torch.arange(n_positions).unsqueeze(1)
    j = torch.arange(n_positions).unsqueeze(0)
    mask = (j <= i) & (j >= (i - window_size + 1))  # Create mask: causal and within window
    mask = mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)
    return mask

# token-generation, non-sliding
# ref: https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_base.py#L295
def create_simple_attn_mask(attention_mask, n_positions):
    batch_size = attention_mask.shape[0]

    return (
        attention_mask[:, None, None, :].expand(batch_size, 1, 1, n_positions).to(torch.bool)
    )

# token-generation, sliding
# ref: https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/model_base.py#L317
def create_windowed_attn_mask_tkg(attention_mask, window_size, position_ids):
    # Create tkg mask for sliding window. E.g.:
    # position = 3, window_size = 4 -> mask = [1,1,1,0]
    # position = 5, window_size = 4 -> mask = [1,0,1,1]
    batch_size, _ = attention_mask.shape
    pos = position_ids[:, 0]
    idx = torch.arange(window_size, device=attention_mask.device).unsqueeze(0)
    base_mask = idx < pos.unsqueeze(1)  # for input_len <= window_size

    full_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=attention_mask.device)
    zero_pos = pos % window_size
    zero_mask = idx == zero_pos.unsqueeze(1)
    full_mask = torch.where(zero_mask, False, full_mask)  # for input_len > window_size

    seq_less_than_window = pos < window_size
    final_mask = torch.where(seq_less_than_window.unsqueeze(1), base_mask, full_mask)
    return final_mask[:, None, None, :]
    
def causal_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def window_mask(batch_size: int, seq_len: int, window_size: int):
    """create a causal, window attention mask"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
    for i in range(seq_len):
        if i >= window_size:
            mask[i, : i - window_size + 1] = False
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask


### HuggingFace Masks
def prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
    """
    https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma3/modeling_gemma3.py#L789C5-L844C27
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
            `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache,
            to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask

# ref: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma3/modeling_gemma3.py#L388
def apply_sliding_window_to_hf_attn_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sliding_window: int,
        cache_position: torch.Tensor,
        last_cache_position: torch.Tensor = None,
        attn_implementation: str = None,
    ):
    if last_cache_position == None:
        last_cache_position = cache_position[-1]
    # In prefill, we may be larger than sliding window
    effective_seq_len = max(cache_position.shape[0], sliding_window)
    # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
    # thus we must slice from the right (at most `effective_seq_len` elements)
    if attn_implementation == "flash_attention_2":
        attention_mask = attention_mask[:, -effective_seq_len:]
    # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
    # from the left, with an offset if we are beyond the sliding window
    else:
        min_dtype = torch.finfo(attention_mask.dtype).min
        sliding_window_mask = torch.tril(
            torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-sliding_window
        )
        attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
        # In case we are beyond the sliding window, we need to correctly offset the mask slicing
        # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo
        offset = last_cache_position - effective_seq_len
        # Should only be used when beyond the sliding window (i.e. offset > 0)
        offset = max(0, offset)
        attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]
    return attention_mask