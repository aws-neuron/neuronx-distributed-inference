import pytest

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
    ChunkedPrefillConfig,
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


def test_generate_buckets_for_tkg_batch_bucketing():
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=512,
        max_length=1024,
        tkg_batch_size=8,
        token_generation_batches=[2, 4],  # Max batch size 8 should be auto-added
        token_generation_buckets=[256, 512, 1024],
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)

    # Expected: batch buckets [8, 4, 2] (reverse sorted, 8 auto-added) × seq buckets
    expected = [
        [8, 256], [8, 512], [8, 1024],
        [4, 256], [4, 512], [4, 1024],
        [2, 256], [2, 512], [2, 1024],
    ]
    assert result == expected


def test_generate_buckets_for_tkg_batch_bucketing_max_only():
    # Test when only max batch size is specified
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=256,
        max_length=512,
        tkg_batch_size=4,
        token_generation_batches=[4],  # Only max batch size
        token_generation_buckets=[256, 512],
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)

    # Expected: only batch=4 buckets
    expected = [[4, 256], [4, 512]]
    assert result == expected


def test_generate_buckets_for_tkg_batch_bucketing_default_seq_buckets():
    # Test batch bucketing with auto-generated sequence buckets
    n_config = NeuronConfig(
        enable_bucketing=True,
        max_context_length=512,
        max_length=1024,
        tkg_batch_size=4,
        token_generation_batches=[2],
        # No token_generation_buckets specified - should auto-generate
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)

    # Expected: batch buckets [4, 2] × auto-generated seq buckets [128, 256, 512, 1024]
    expected = [
        [4, 128], [4, 256], [4, 512], [4, 1024],
        [2, 128], [2, 256], [2, 512], [2, 1024],
    ]
    assert result == expected


def test_generate_buckets_for_tkg_batch_bucketing_disabled_bucketing():
    # Test that batch bucketing can be enabled when sequence bucketing is disabled
    n_config = NeuronConfig(
        enable_bucketing=False,
        max_context_length=512,
        max_length=1024,
        tkg_batch_size=4,
        token_generation_batches=[2],  # Should be ignored
    )
    config = InferenceConfig(neuron_config=n_config)
    result = autobucketing.generate_buckets_for_tkg(config)

    # Expected: batch buckets x max_length
    expected = [[4, 1024], [2, 1024]]
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
