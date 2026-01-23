"""
Fixed HybridCache KV cache manager from HuggingFace Transformers 4.52.4 source code.
See [Issue 37574](https://github.com/huggingface/transformers/issues/37574) -> Fixed in 4.52.0
Required for the integration test to pass, otherwise HuggingFace Transformers Cohere2 implementation generates wrong 
ground truth logits for the last token in the output sequence due to an incorrect KV cache rolling update in the SWA layers 
when generating the last token of a max_seq_len sequence.
To be removed once HuggingFace Transformers is updated to >=4.52.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# Utility functions for static/sliding cache update logic
def _static_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the static cache tensors in place.

    Args:
        k_cache (`torch.Tensor`): The key cache tensor to update.
        v_cache (`torch.Tensor`): The value cache tensor to update.
        key_states (`torch.Tensor`): The new key states to add.
        value_states (`torch.Tensor`): The new value states to add.
        cache_position (`Optional[torch.LongTensor]`): The position indices where the new states should be inserted.
                                                       If None, the entire cache is overwritten (prefill).

    Returns:
        Tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value cache tensors (modified in-place).
    """
    if cache_position is None:
        # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
        k_cache.copy_(key_states)
        v_cache.copy_(value_states)
    else:
        # Generation phase. Update specific positions.
        # Use index_copy_ for in-place update (compile-friendly).
        try:
            k_cache.index_copy_(2, cache_position, key_states)
            v_cache.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            k_cache[:, :, cache_position] = key_states
            v_cache[:, :, cache_position] = value_states
    return k_cache, v_cache


def _sliding_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.LongTensor,
    max_cache_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the sliding window cache tensors, returning the potentially modified tensors.

    Args:
        k_cache (`torch.Tensor`): The key cache tensor to update.
        v_cache (`torch.Tensor`): The value cache tensor to update.
        key_states (`torch.Tensor`): The new key states to add.
        value_states (`torch.Tensor`): The new value states to add.
        cache_position (`torch.LongTensor`): The position indices where the new states should be inserted.
        max_cache_len (`int`): The maximum length of the sliding window cache.

    Returns:
        Tuple[`torch.Tensor`, `torch.Tensor`]: The key and value tensors representing the cache state after the update.
                                               For prefill > window, these are the full input states.
                                               Otherwise, they are the updated cache tensors.
    """
    # Handle prefill phase when prompt length > sliding_window_size
    if cache_position.shape[0] > max_cache_len:
        new_k = key_states[:, :, -max_cache_len:, :]
        new_v = value_states[:, :, -max_cache_len:, :]
        k_cache.copy_(new_k)
        v_cache.copy_(new_v)
        return key_states, value_states

    # Sliding window logic for generation phase or prefill < window
    slicing = torch.arange(max_cache_len, device=value_states.device)
    current_seq_len = cache_position[-1] + 1  # Use last position to determine current length
    to_shift = current_seq_len > max_cache_len
    indices = (slicing + to_shift.sum()) % max_cache_len

    k_out_shifted = k_cache[:, :, indices]
    v_out_shifted = v_cache[:, :, indices]

    # Clamp cache_position to determine the *target index* within the shifted cache view
    update_position = cache_position.clamp(min=0, max=max_cache_len - 1)

    try:
        k_out_updated = k_out_shifted.index_copy(2, update_position, key_states)
        v_out_updated = v_out_shifted.index_copy(2, update_position, value_states)
    except NotImplementedError:
        # Fallback for MPS: clone and modify the clone
        k_out_updated = k_out_shifted.clone()
        v_out_updated = v_out_shifted.clone()
        k_out_updated[:, :, update_position] = key_states
        v_out_updated[:, :, update_position] = value_states

    k_cache.copy_(k_out_updated)
    v_cache.copy_(v_out_updated)
    return k_out_updated, v_out_updated


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention and ["StaticCache"]
    for global attention.For more information, see the documentation of each subcomponent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    is_compileable = True

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'hybrid' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        self.max_cache_len = max_cache_len if max_cache_len is not None else config.max_position_embeddings
        # Sliding layers can't be larger than the overall max cache len
        self.sliding_window_len = min(config.sliding_window, self.max_cache_len)
        self.max_batch_size = max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        layer_switch = config.sliding_window_pattern if hasattr(config, "sliding_window_pattern") else 2  # 2 is for BC
        self.is_sliding_list = [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        sliding_cache_shape = (self.max_batch_size, self.num_key_value_heads, self.sliding_window_len, self.head_dim)
        device = torch.device(device) if device is not None else None
        for i in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[i]
            else:
                layer_device = device
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            cache_shape = sliding_cache_shape if self.is_sliding_list[i] else global_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for HybridCache.")

        is_sliding_layer = self.is_sliding_list[layer_idx]

        # These two `if` blocks are only reached in multigpu and if `layer_device_map` is not passed. They are used
        # when the cache is initialized in the forward pass (e.g. Gemma2)
        if self.key_cache[layer_idx].device != key_states.device:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
        if self.value_cache[layer_idx].device != value_states.device:
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)

        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]
        key_states = key_states.to(k_cache.dtype)
        value_states = value_states.to(v_cache.dtype)

        if is_sliding_layer:
            return _sliding_cache_update(
                k_cache,
                v_cache,
                key_states,
                value_states,
                cache_position,
                k_cache.shape[2],  # Use actual cache dim as max cache len
            )
        else:
            return _static_cache_update(k_cache, v_cache, key_states, value_states, cache_position)

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
