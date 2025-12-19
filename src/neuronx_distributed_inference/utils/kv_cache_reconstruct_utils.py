import torch
from typing import Callable
from neuronx_distributed_inference.models.config import InferenceConfig
from torch.nn.modules.container import ParameterList

##############################################################################
# Types
##############################################################################
NeuronDeviceCache = list[dict[str, torch.Tensor]]
"""
Type for sharded KV Cache extracted from traced Neuron Model KV Cache

```
sharded_cache: NeuronDeviceCache = ...
```

1. `len(sharded_cache)` = tp_degree

2.  Each `sharded_cache[i]` represents `rank i`, mapping layer name to K/V tensors interleaved
```
sharded_cache[i].keys() = {
    kv_mgr.past_key_values[0] # layer 0 K Tensor
    kv_mgr.past_key_values[1] # layer 0 V Tensor
}
```
3. Each tensor is typically of shape `[batch_sz, num_kv_heads/tp_degree, seq_len, head_dim]`
- Note: we use num_kv_heads/tp_degree to denote the number of KV Heads per rank
```
# Typical KV tensor shape
sharded_cache[i][layer].shape = [batch_sz, num_kv_heads/tp_degree, seq_len, head_dim]
```
"""


ReconstructCacheLayerCallable = Callable[
    [NeuronDeviceCache, InferenceConfig, int, int, bool], torch.Tensor
]
"""
Prototype for a function used in `reconstruct_neuron_device_kv_cache_base` to reconstruct a single layer's
K or V Cache.

Args:
    neuron_cache (NeuronDeviceCacheType): Sharded KV Cache on Device
    inference_config (InferenceConfig): Neuron and Model Configuration
    num_kv_heads (int): Number of Key Value heads
    layer_idx (int): Specific layer to extract K or V Cache from
    is_v_cache (bool): True if extracting V cache; False if extracting K cache

Returns:
    torch.Tensor: Shape [batch_sz, num_kv_heads, seq_len, head_dim]
"""


##############################################################################
# CPU Reconstruction
##############################################################################
def reconstruct_neuron_cpu_kv_cache(kv_cache: ParameterList) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts KV Cache from `neuron_cpu_model.token_generation_model.model.kv_mgr.past_key_values` and
    returns a reconstructed K and V Cache Tensor

    - Supports only TP = 1

    Args:
        kv_cache (ParameterList):
            ParameterList where even indexes represent K Cache and Odd Indexes represent V Cache
            kv_cache[i].shape = [batch_sz, num_kv_heads, seq_len, head_dim]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - K cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
            - V cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
    """
    # Even indexes of the kv_cache represents K's while odd indexes represent V's
    k_cache = kv_cache[::2]
    v_cache = kv_cache[1::2]

    k_cache = torch.stack(list(k_cache), dim=1)
    v_cache = torch.stack(list(v_cache), dim=1)
    return k_cache, v_cache


##############################################################################
# Reconstruction Primitives
##############################################################################
def extract_all_k_or_v_heads_from_rank(
    neuron_cache: NeuronDeviceCache, rank_idx: int, layer_idx: int, is_v_head: bool
) -> torch.Tensor:
    """
    Base function which extracts the KV Cache tensor at layer `layer_idx` in rank `rank_idx` from Neuron Device to CPU.
    If `is_v_head` is True, then the v tensor is returned, else the k tensor is returned.

    Args:
        neuron_cache (NeuronDeviceCacheType): Sharded KV Cache on Device
        rank_idx (int): Specific rank to extract the KV Cache tensor from
        layer_idx (int): Specific layer to extract tensor from

    Returns:
        torch.Tensor: CPU KV Cache tensor from rank `rank_idx` at layer `layer_idx`
        - Note: returned tensor shape is typically [batch_sz, num_kv_heads/tp_degree, seq_len, head_dim]
        but can change for different strategies (i.e. tiling)
    """
    rank_dict = neuron_cache[rank_idx]

    # KV Cache layout at layer i: K at even indices (2*i), V at odd indices (2*i + 1)
    k_or_v_head = rank_dict[f"kv_mgr.past_key_values.{layer_idx * 2 + int(is_v_head)}"].to("cpu")
    return k_or_v_head


def reconstruct_neuron_device_kv_cache_base(
    neuron_cache: NeuronDeviceCache,
    inference_config: InferenceConfig,
    num_kv_heads: int,
    reconstruct_cache_layer_fn: ReconstructCacheLayerCallable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Base function for extracting and reconstructing KV cache from Neuron device to CPU.

    This function applies the provided reconstructing function to each layer of the model,
    handling the common logic of iterating through layers and stacking the results.
    It's designed to be wrapped by sharding-strategy-specific functions like
    `reconstruct_neuron_device_kv_cache_<SHARDING_STRATEGY>`.

    Args:
        neuron_cache (NeuronDeviceCacheType): Sharded KV cache tensors from Neuron device
        inference_config (InferenceConfig): Config containing model parameters (requires num_hidden_layers)
        num_kv_heads (int): Number of Key Value heads
        reconstruct_cache_layer_fn (ReconstructCacheLayerCallable): Function that implements reconstruction logic for a single layer

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - K cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
            - V cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]

    Raises:
        AttributeError: If num_hidden_layers is not in inference_config
    """
    # Step 1) Determine Number of Layers
    if not hasattr(inference_config, "num_hidden_layers"):
        raise AttributeError("num_hidden_layers not found in inference_config")

    num_layers = inference_config.num_hidden_layers

    # Step 2) Run reconstruction per layer
    def reconstruct_k_or_v_cache(is_v_cache: bool):
        all_layers_k_or_v: list[torch.Tensor] = []

        for layer_idx in range(num_layers):
            layer_k_or_v_cache = reconstruct_cache_layer_fn(
                neuron_cache, inference_config, num_kv_heads, layer_idx, is_v_cache
            )
            all_layers_k_or_v.append(layer_k_or_v_cache)

        # num_layers * [batch_sz, num_kv_heads, seq_len, head_dim] => [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
        reconstructed_k_or_v_cache = torch.stack(all_layers_k_or_v, dim=1)
        return reconstructed_k_or_v_cache

    # Step 3) Extract K and V Cache
    reconstructed_k_cache = reconstruct_k_or_v_cache(is_v_cache=False)
    reconstructed_v_cache = reconstruct_k_or_v_cache(is_v_cache=True)
    return reconstructed_k_cache, reconstructed_v_cache


# GQA Reconstruction Utils
def reconstruct_cache_layer_gqa_replicate_to_tp_degree(
    neuron_cache: NeuronDeviceCache,
    inference_config: InferenceConfig,
    num_kv_heads: int,
    layer_idx: int,
    is_v_cache: bool,
) -> torch.Tensor:
    """
    Reconstructs K or V Cache for a specific layer assuming `GQA.REPLICATE_TO_TP_DEGREE`

    Args:
        neuron_cache (NeuronDeviceCacheType): Sharded KV Cache on Device
        inference_config (InferenceConfig): Neuron and Model Configuration
        num_kv_heads (int): Number of Key Value heads
        layer_idx (int): Specific layer to extract K or V Cache from
        is_v_cache (bool): True if extracting V cache; False if extracting K cache

    Returns:
        torch.Tensor: K or V Cache; typical shape [batch_sz, num_kv_heads, seq_len, head_dim]
    """

    # Step 1) Determine tp_degree
    tp_degree = inference_config.neuron_config.tp_degree

    # Step 2) Find number of times KV Head is duplicated so that there is 1 KV Head per rank
    dup_degree = tp_degree // num_kv_heads

    # List of [B, KV/TP=1, S, D] tensors; length=num_kv_heads
    all_heads_k_or_v: list[torch.Tensor] = []

    # Step 3) Extract all unique KV Heads at layer `layer_idx`
    for rank_idx in range(0, tp_degree, dup_degree):
        # Extract KV Tensors and move to CPU
        k_or_v_head = extract_all_k_or_v_heads_from_rank(
            neuron_cache, rank_idx, layer_idx, is_v_head=is_v_cache
        )

        # Check `batch_sz`, `num_kv_heads/tp_degree`, and `seq_length`
        # Note: we assume num_kv_heads/tp_degree is 1 for `GQA.REPLICATE_TO_TP_DEGREE`
        # Since `head_dim` is not guaranteed to be in the HF config, we do not check for it here
        if k_or_v_head.shape[:3] != (inference_config.neuron_config.batch_size, 1, inference_config.neuron_config.seq_len):
            raise ValueError(
                f"Expected shape [{inference_config.neuron_config.batch_size}, 1, {inference_config.neuron_config.seq_len}, head_dim] but got {k_or_v_head.shape}"
            )

        all_heads_k_or_v.append(k_or_v_head)

    # Step 4) num_kv_heads * [batch_sz, num_kv_heads/tp_degree=1, seq_len, head_dim] => [batch_sz, num_kv_heads, seq_len, head_dim]
    all_heads_k_or_v_cache = torch.cat(all_heads_k_or_v, dim=1)

    return all_heads_k_or_v_cache


def reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(
    neuron_cache: NeuronDeviceCache,
    inference_config: InferenceConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstructs KV Cache from Neuron device to CPU for GQA models using REPLICATE_TO_TP_DEGREE sharding strategy.

    This function extracts and reshapes the key-value cache from a tensor-parallel distributed model where
    each KV head is replicated across multiple ranks (tp_degree/num_kv_heads ranks per head).

    Args:
        neuron_cache (NeuronDeviceCacheType): Sharded KV Cache tensors on Neuron device
        inference_config (InferenceConfig): Configuration containing model parameters and neuron configs
                                           (must include attributes num_hidden_layers and num_key_value_heads)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - K cache tensor; typical shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
            - V cache tensor; typical shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
    """

    if not hasattr(inference_config, "num_key_value_heads"):
        raise AttributeError("num_key_value_heads not found in inference_config")

    num_kv_heads = inference_config.num_key_value_heads

    k_cache, v_cache = reconstruct_neuron_device_kv_cache_base(
        neuron_cache,
        inference_config,
        num_kv_heads,
        reconstruct_cache_layer_gqa_replicate_to_tp_degree,
    )

    return k_cache, v_cache
