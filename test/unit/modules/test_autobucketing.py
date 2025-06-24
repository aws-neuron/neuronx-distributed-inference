import pytest
import torch

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
    ChunkedPrefillConfig,
)
from neuronx_distributed_inference.models.model_wrapper import (
    ModelWrapper,
    SPECULATION_MODEL_TAG,
    CONTEXT_ENCODING_MODEL_TAG,
)

import neuronx_distributed_inference.modules.autobucketing as autobucketing


def test_generate_buckets():
    # Test with same min and max length
    assert autobucketing.generate_buckets(128, 128) == [128]
    
    # Test with different min and max lengths
    assert autobucketing.generate_buckets(128, 512) == [128, 256, 512]
    
    # Test with power of 2 values
    assert autobucketing.generate_buckets(64, 256) == [64, 128, 256]
    
    # Test with non-power of 2 max value
    assert autobucketing.generate_buckets(128, 513) == [128, 256, 513]


def test_generate_2d_buckets_for_prefix_caching():
    # Test basic case
    result = autobucketing.generate_2d_buckets_for_prefix_caching(128, 256, 128, 256, False)
    expected = [[128, 128], [128, 256], [256, 128], [256, 256]]
    assert result == expected
    
    # Test with context encoding
    result = autobucketing.generate_2d_buckets_for_prefix_caching(128, 256, 128, 256, True)
    expected = [[128, 0], [128, 128], [128, 256], [256, 0], [256, 128], [256, 256]]
    assert result == expected


def test_generate_buckets_on_chunk_size():
    # Test when max_context_len < q_tile_size
    assert autobucketing.generate_buckets_on_chunk_size(128, 64) == [128]
    
    # Test with small range and not a multiple of q_tile_size
    assert autobucketing.generate_buckets_on_chunk_size(128, 250) == [128, 256]
    
    # Test with larger range
    assert autobucketing.generate_buckets_on_chunk_size(128, 1024) == [128, 512, 1024]


def test_generate_buckets_for_chunked_prefill_cte_with_disabled_bucket():
    # Test with bucketing disabled
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=1024,
        max_length=1024,
        is_chunked_prefill=True,
        is_block_kv_layout=True,
        chunked_prefill_config=ChunkedPrefillConfig(
            kernel_q_tile_size=128,
            kernel_kv_tile_size=512,
            max_num_seqs=8
        )
    )
    config = InferenceConfig(neuron_config=n_config)

    result = autobucketing.generate_buckets_for_chunked_prefill_cte(config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == [1024, 128]

def test_generate_buckets_for_chunked_prefill_cte_with_enabled_bucket():
    # Test with bucketing enable
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=1024,
        max_length=1024,
        is_chunked_prefill=True,
        is_block_kv_layout=True,
        chunked_prefill_config=ChunkedPrefillConfig(
            kernel_q_tile_size=128,
            kernel_kv_tile_size=512,
            max_num_seqs=2
        )
    )
    config = InferenceConfig(neuron_config=n_config)

    result = autobucketing.generate_buckets_for_chunked_prefill_cte(config)

    chunk_size_buckets = [128, 512, 1024]
    tile_buckets = [1, 16, 32]
    expected = [[q, tile] for q in chunk_size_buckets for tile in tile_buckets]

    assert isinstance(result, list)
    assert len(result) == len(expected)
    for i in range(len(expected)):
        assert expected[i] == result[i]


def test_generate_buckets_for_cte():
    # Test with bucketing disabled and no prefix caching
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_cte(config)
    assert result == [1024]
    
    # Test with bucketing enabled and no custom buckets
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_cte(config)
    assert result == [128, 256, 512, 1024]


def test_generate_buckets_for_tkg():
    # Test with bucketing disabled and no prefix caching
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)
    assert result == [2048]
    
    # Test with bucketing enabled and no custom buckets
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)
    assert result == [128, 256, 512, 1024, 2048]


def test_generate_buckets_for_fused_spec():
    # Test with bucketing disabled and no prefix caching
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_fused_spec(config)
    assert result == [2048]
    
    # Test with bucketing enabled and no custom buckets
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_fused_spec(config)
    assert result == [128, 256, 512, 1024, 2048]


def test_generate_buckets_for_spec():
    # Test with bucketing disabled and no prefix caching
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_speculation(config)
    assert result == [2048]
    
    # Test with bucketing enabled and no custom buckets
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=1024,
        max_length=2048,
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_speculation(config)
    assert result == [128, 256, 512, 1024, 2048]
