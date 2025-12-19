from unittest.mock import MagicMock
import pytest
import torch.nn as nn
import torch

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.utils.kv_cache_reconstruct_utils import (
    NeuronDeviceCache,
    reconstruct_cache_layer_gqa_replicate_to_tp_degree,
    reconstruct_neuron_cpu_kv_cache,
    reconstruct_neuron_device_kv_cache_base,
    reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree,
)

##############################################################################
# Constants
##############################################################################
LAYER_DIM = 1


##############################################################################
# CPU Reconstruction Unit Test
##############################################################################
@pytest.mark.parametrize(
    "batch_sz, num_layers, num_kv_heads, seq_len, head_dim",
    [
        (2, 4, 8, 256, 512),
    ],
)
def test_reconstruct_neuron_cpu_kv_cache(
    batch_sz: int, num_layers: int, num_kv_heads: int, seq_len: int, head_dim: int
):
    """
    Tests the `reconstruct_neuron_cpu_kv_cache` function by creating a mock ParameterList
    with alternating K and V tensors and verifying correct reconstruction.

    The function creates a ParameterList where even indices contain K cache tensors
    and odd indices contain V cache tensors, then verifies that the reconstruction
    function correctly separates and stacks them into the expected output format.

    Args:
        batch_sz (int): Batch size dimension
        num_layers (int): Number of transformer layers
        num_kv_heads (int): Number of key-value heads per layer
        seq_len (int): Sequence length dimension
        head_dim (int): Head dimension size

    Verifies:
        - Output tensors have correct shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
        - K cache contains tensors from even indices of input ParameterList
        - V cache contains tensors from odd indices of input ParameterList
    """

    print("Testing Neuron CPU KV Cache Reconstruction")

    # Initialize Shape
    cpu_cache_layer_shape = (batch_sz, num_kv_heads, seq_len, head_dim)
    expected_final_kv_cache_shape = (batch_sz, num_layers, num_kv_heads, seq_len, head_dim)

    # Step 1) Create the KV Cache which is a list of 2*n_layer tensors
    mock_kv_cache = nn.ParameterList()

    for layer in range(num_layers):
        k_head = torch.rand(cpu_cache_layer_shape)
        v_head = torch.rand(cpu_cache_layer_shape)

        mock_kv_cache.append(nn.Parameter(k_head))
        mock_kv_cache.append(nn.Parameter(v_head))

    # Step 2) Reconstruct CPU KV Cache
    k_cache, v_cache = reconstruct_neuron_cpu_kv_cache(mock_kv_cache)

    # Step 3) Verification
    assert k_cache.shape == expected_final_kv_cache_shape
    assert v_cache.shape == expected_final_kv_cache_shape

    for layer in range(num_layers):
        assert torch.equal(k_cache[:, layer], mock_kv_cache[2 * layer])
        assert torch.equal(v_cache[:, layer], mock_kv_cache[2 * layer + 1])


##############################################################################
# Neuron Device Base Reconstruct Unit Tests
##############################################################################
@pytest.mark.parametrize(
    "batch_sz, num_layers, num_kv_heads, seq_len, head_dim",
    # fmt: off
    [
        (2, 1, 8, 256, 512), 
        (2, 4, 8, 256, 512)
    ],
    # fmt: on
)
def test_reconstruct_neuron_device_kv_cache_base(
    batch_sz: int,
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    """
    Tests the `reconstruct_neuron_device_kv_cache_base` function using a mock layer reconstruction function.

    Creates a mock reconstruction function that returns tensors filled with the layer index value (`cache[:, layer, :, :, :] = layer_index`),
    then verifies that the base function correctly applies this function to each layer and
    stacks the results into the expected output format.

    Args:
        batch_sz (int): Batch size dimension
        num_layers (int): Number of transformer layers
        num_kv_heads (int): Number of key-value heads per layer
        seq_len (int): Sequence length dimension
        head_dim (int): Head dimension size

    Verifies:
        - Function correctly iterates through all layers
        - Output tensors have shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
        - Both K and V caches contain expected layer-indexed values
    """
    # Step 1) Create mock inputs
    mock_neuron_cache = MagicMock()
    inference_config = MagicMock()
    inference_config.num_hidden_layers = num_layers

    # Function to return tensor of shape [B, KV, S, D] = l_idx
    def mock_reconstruct_cache_layer_fn(
        neuron_cache: NeuronDeviceCache,
        inference_config: InferenceConfig,
        num_kv_heads: int,
        layer_idx: int,
        is_v_cache: bool,
    ) -> torch.Tensor:
        return torch.full((batch_sz, num_kv_heads, seq_len, head_dim), layer_idx)

    # Create Cache of shape [B, L, KV, S, D] where [:, L, :, :, :] = l_idx
    orig_cache = torch.zeros((batch_sz, num_layers, num_kv_heads, seq_len, head_dim))
    for layer in range(num_layers):
        orig_cache[:, layer, :, :, :] = layer

    # Step 2) Reconstruct Cache
    reconstructed_k_cache, reconstructed_v_cache = reconstruct_neuron_device_kv_cache_base(
        mock_neuron_cache, inference_config, num_kv_heads, mock_reconstruct_cache_layer_fn
    )

    # Step 3) Test Reconstruction
    assert reconstructed_k_cache.shape == orig_cache.shape
    assert reconstructed_v_cache.shape == orig_cache.shape
    assert torch.equal(reconstructed_k_cache, orig_cache)
    assert torch.equal(reconstructed_v_cache, orig_cache)


##############################################################################
# Neuron Device GQA.REPLICATE_TO_TP_DEGREE Unit Tests
##############################################################################
def create_mock_cache_replicate_to_tp_degree(
    batch_sz: int, num_layers: int, num_kv_heads: int, seq_len: int, head_dim: int, tp_degree: int
) -> tuple[torch.Tensor, torch.Tensor, NeuronDeviceCache]:
    """
    Creates mock KV cache data for testing GQA.REPLICATE_TO_TP_DEGREE sharding strategy.

    Generates original K/V cache tensors and their corresponding sharded representation
    where each KV head is replicated across multiple ranks (tp_degree // num_kv_heads ranks per head).

    Args:
        batch_sz (int): Batch size dimension
        num_layers (int): Number of transformer layers
        num_kv_heads (int): Number of key-value heads
        seq_len (int): Sequence length dimension
        head_dim (int): Head dimension size
        tp_degree (int): Tensor parallel degree (total number of ranks)

    Returns:
        tuple[torch.Tensor, torch.Tensor, NeuronDeviceCacheType]: Tuple containing:
            - Original K cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
            - Original V cache tensor with shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
            - Sharded cache in NeuronDeviceCacheType format for testing reconstruction functions
    """
    # Step 1) Create Base Golden Cache
    k_cache = torch.rand(batch_sz, num_layers, num_kv_heads, seq_len, head_dim)
    v_cache = torch.rand(batch_sz, num_layers, num_kv_heads, seq_len, head_dim)

    # Step 2) Shard Cache based on GQA.REPLICATE_TO_TP_DEGREE
    dup_degree = tp_degree // num_kv_heads

    sharded_cache: NeuronDeviceCache = []

    # For each core
    for core in range(tp_degree):

        # Generate all layers of the KV Cache
        rank_dict = {}

        # Map each Core to a specific head
        head = core // dup_degree
        for layer in range(num_layers):

            # Select a layer and head, keeping head dimension [B, L, KV, S, D] -> [B, KV=1, S, D]
            k_head = k_cache[:, layer, head : head + 1, :, :]
            v_head = v_cache[:, layer, head : head + 1, :, :]

            rank_dict[f"kv_mgr.past_key_values.{2 * layer}"] = k_head
            rank_dict[f"kv_mgr.past_key_values.{2 * layer + 1}"] = v_head

        sharded_cache.append(rank_dict)

    return k_cache, v_cache, sharded_cache


@pytest.mark.parametrize(
    "batch_sz, num_layers, num_kv_heads, seq_len, head_dim, tp_degree, layer_to_extract",
    [
        (2, 4, 32, 256, 512, 32, 1),  # Test KV=TP=32
        (2, 4, 8, 256, 512, 32, 1),  # Test KV=8, TP=32
        (2, 4, 8, 256, 512, 16, 3),  # Test KV=8, TP=16 (different TP degree, different layer)
    ],
)
def test_reconstruct_cache_layer_gqa_replicate_to_tp_degree(
    batch_sz: int,
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    tp_degree: int,
    layer_to_extract: int,
):
    """
    Tests the `reconstruct_cache_layer_gqa_replicate_to_tp_degree` function for a single layer.

    Creates mock sharded cache data and verifies that the function correctly extracts
    and reconstructs the K/V cache for a specific layer using GQA.REPLICATE_TO_TP_DEGREE strategy.

    Args:
        batch_sz (int): Batch size dimension
        num_layers (int): Number of transformer layers
        num_kv_heads (int): Number of key-value heads
        seq_len (int): Sequence length dimension
        head_dim (int): Head dimension size
        tp_degree (int): Tensor parallel degree (total number of ranks)
        layer_to_extract (int): Specific layer index to test reconstruction on

    Verifies:
        - Output tensors have shape [batch_sz, num_kv_heads, seq_len, head_dim]
        - Reconstructed K/V cache matches original cache for the specified layer
    """
    # Step 1) Create Mock Sharded Cache, K and V Cache Tensors, and Mock InferenceConfig
    inference_config = MagicMock()
    inference_config.neuron_config.batch_size = batch_sz
    inference_config.neuron_config.seq_len = seq_len
    inference_config.neuron_config.tp_degree = tp_degree

    k_cache, v_cache, neuron_cache = create_mock_cache_replicate_to_tp_degree(
        batch_sz, num_layers, num_kv_heads, seq_len, head_dim, tp_degree
    )

    # Step 2) Test Reconstruction Function; [B, KV, S, D]
    reconstructed_k_heads = reconstruct_cache_layer_gqa_replicate_to_tp_degree(
        neuron_cache, inference_config, num_kv_heads, layer_to_extract, is_v_cache=False
    )
    reconstructed_v_heads = reconstruct_cache_layer_gqa_replicate_to_tp_degree(
        neuron_cache, inference_config, num_kv_heads, layer_to_extract, is_v_cache=True
    )

    # Step 3) Compare Reconstructed Cache

    # Extract original KV Cache for a specific layer: [B, L, KV, S, D] -> [B, KV, S, D]
    orig_k_heads = k_cache[:, layer_to_extract, :, :, :]
    orig_v_heads = v_cache[:, layer_to_extract, :, :, :]

    assert orig_k_heads.shape == reconstructed_k_heads.shape
    assert orig_v_heads.shape == reconstructed_v_heads.shape
    assert torch.equal(orig_k_heads, reconstructed_k_heads)
    assert torch.equal(orig_v_heads, reconstructed_v_heads)


@pytest.mark.parametrize(
    "batch_sz, num_layers, num_kv_heads, seq_len, head_dim, tp_degree",
    [
        (2, 4, 32, 256, 512, 32),  # Test KV=32, TP=32
        (2, 4, 8, 256, 512, 32),  # Test KV=8, TP=32
        (2, 4, 8, 256, 512, 16),  # Test KV=8, TP=16
    ],
)
def test_reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(
    batch_sz: int,
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    tp_degree: int,
):
    """
    Tests the `reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree` function end-to-end.

    Creates mock sharded cache data and verifies that the function correctly reconstructs
    the entire KV cache across all layers using GQA.REPLICATE_TO_TP_DEGREE strategy.

    Args:
        batch_sz (int): Batch size dimension
        num_layers (int): Number of transformer layers
        num_kv_heads (int): Number of key-value heads
        seq_len (int): Sequence length dimension
        head_dim (int): Head dimension size
        tp_degree (int): Tensor parallel degree (total number of ranks)

    Verifies:
        - Output tensors have shape [batch_sz, num_layers, num_kv_heads, seq_len, head_dim]
        - Reconstructed K/V cache exactly matches original sharded cache
    """
    # Step 1) Create Mock KV Cache
    inference_config = MagicMock()
    inference_config.neuron_config.batch_size = batch_sz
    inference_config.num_hidden_layers = num_layers
    inference_config.num_key_value_heads = num_kv_heads
    inference_config.neuron_config.seq_len = seq_len
    inference_config.neuron_config.tp_degree = tp_degree

    k_cache, v_cache, neuron_cache = create_mock_cache_replicate_to_tp_degree(
        batch_sz, num_layers, num_kv_heads, seq_len, head_dim, tp_degree
    )

    # Step 2) Reconstruct Cache
    reconstructed_k_cache, reconstructed_v_cache = (
        reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(
            neuron_cache, inference_config
        )
    )

    # Step 3) Compare Caches
    assert k_cache.shape == reconstructed_k_cache.shape
    assert v_cache.shape == reconstructed_v_cache.shape

    assert torch.equal(k_cache, reconstructed_k_cache)
    assert torch.equal(v_cache, reconstructed_v_cache)


##############################################################################
# Example Unit Tests
##############################################################################
if __name__ == "__main__":
    test_reconstruct_neuron_cpu_kv_cache(
        num_layers=4, batch_sz=2, num_kv_heads=8, seq_len=256, head_dim=512
    )

    test_reconstruct_neuron_device_kv_cache_base(
        batch_sz=2, num_layers=4, num_kv_heads=8, seq_len=256, head_dim=512
    )

    test_reconstruct_cache_layer_gqa_replicate_to_tp_degree(
        batch_sz=2,
        num_layers=4,
        num_kv_heads=8,
        seq_len=256,
        head_dim=512,
        tp_degree=32,
        layer_to_extract=1,
    )

    test_reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(
        batch_sz=2, num_layers=4, num_kv_heads=32, seq_len=256, head_dim=512, tp_degree=32
    )
