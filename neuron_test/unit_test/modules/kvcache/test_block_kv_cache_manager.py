import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import BlockKVCacheManager


def test_reading_kv_cache_for_paged_attention():
    tp_degree=1
    batch_size=1
    num_hidden_layers=3
    num_kv_head=4
    hidden_size=16
    num_hidden_layers=3
    num_attention_heads=8

    pa_num_blocks=15
    pa_block_size=4

    seq_len=32

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers*2
    cache_layout = (pa_num_blocks, pa_block_size, num_kv_head, hidden_size//num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    _pa_mock_kv_cache_in_mgr(kv_cache_mgr)
    block_table = torch.tensor([12,8,7,4,3,2,0,0])

    cache = kv_cache_mgr.get_cache(seq_len=seq_len, block_table=block_table)

    assert len(cache) == num_hidden_layers
    assert len(cache[0]) == 2

    for id, block_id in enumerate(block_table.tolist()):
        expected = block_id * torch.ones((pa_block_size, num_kv_head, hidden_size//num_attention_heads))

        # check k cache from the layer 0
        assert cache[0][0].shape == (len(block_table), pa_block_size, num_kv_head, hidden_size//num_attention_heads)
        actual = cache[0][0][id, :, :, :]
        assert torch.equal(actual, expected)

        # check v cache from the layer 2
        assert cache[2][1].shape == (len(block_table), pa_block_size, num_kv_head, hidden_size//num_attention_heads)
        actual = cache[2][1][id, :, :, :]
        assert torch.equal(actual, expected)


def _pa_mock_kv_cache_in_mgr(kv_cache_mgr: BlockKVCacheManager):
    for layer_id in range(len(kv_cache_mgr.past_key_values)):
        for block_id in range(kv_cache_mgr.pa_num_blocks):
            kv_cache_mgr.past_key_values[layer_id][block_id, :, :, :] = block_id


def test_writing_kv_cache_for_paged_attention():
    tp_degree=1
    batch_size=1
    num_hidden_layers=3
    num_kv_head=4
    hidden_size=16
    num_hidden_layers=3
    num_attention_heads=8

    pa_num_blocks=15
    pa_block_size=4

    seq_len=32

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers*2
    cache_layout = (pa_num_blocks, pa_block_size, num_kv_head, hidden_size//num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    latest = _pa_prepare_latest_kv_cache(
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=hidden_size//num_attention_heads,
        num_kv_heads_per_rank=num_kv_head,
        num_hidden_layers=num_hidden_layers,
    )

    # concatenated prompts with 3 seq [4,2,7], and it is padded with zero to
    # fit bucket size
    slot_mapping = torch.tensor(
        [24, 25, 26, 27,
         16, 17,
          8,  9, 10, 11, 12, 13, 14,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    )

    updated_cache = kv_cache_mgr.update_cache(
        is_for_context_encoding=False,
        seq_ids=None,
        position_ids=None,
        past_key_values=latest,
        seq_len=None,
        scatter_index=slot_mapping,
    )

    for seq_pos, slot in enumerate(slot_mapping.tolist()):
        if slot == 0:
            continue # skip for padding
        block_id = slot // pa_block_size
        block_offset = slot % pa_block_size

        # check the k cache for the layer 0
        actual = updated_cache[0]
        expected = latest[0][0]
        assert torch.equal(actual[block_id, block_offset, :, :], expected[0, :, seq_pos, :])

        # check the v cache for the layer 2
        actual = updated_cache[5]
        expected = latest[2][1]
        assert torch.equal(actual[block_id, block_offset, :, :], expected[0, :, seq_pos, :])


def _pa_prepare_latest_kv_cache(
    batch_size=1,
    seq_len=32,
    head_dim=4,
    num_kv_heads_per_rank=8,
    num_hidden_layers=3,
):
    latest_kv_cache = []
    for layer_id in range(num_hidden_layers):
        k_cache = torch.ones(batch_size, num_kv_heads_per_rank, seq_len, head_dim)
        v_cache = torch.ones(batch_size, num_kv_heads_per_rank, seq_len, head_dim)

        for seq_pos in range(seq_len):
            k_cache[:, :, seq_pos, :] *= (2*seq_pos)
            v_cache[:, :, seq_pos, :] *= (2*seq_pos+1)

        latest_kv_cache.append([k_cache, v_cache])

    return latest_kv_cache


def _pa_prepare_cache_mgr(
    tp_degree=1,
    batch_size=1,
    pa_num_blocks=75,
    pa_block_size=128,
    num_attention_heads=8,
    num_kv_head=4,
    hidden_size=32,
    num_hidden_layers=3,
):
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        is_paged_attention=True,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        torch_dtype=torch.float,
    )
    config = InferenceConfig(
        neuron_config=neuron_config,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        )
    kv_cache_mgr = BlockKVCacheManager(config=config, num_kv_head=num_kv_head)
    return kv_cache_mgr
