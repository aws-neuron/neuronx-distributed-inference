# fmt: off
import torch

from neuronx_distributed_inference.models.config import (
    InferenceConfig, 
    NeuronConfig, 
    ChunkedPrefillConfig
)
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import BlockKVCacheManager, generate_tokengen_slot_mapping, generate_fusedspec_slot_mapping
from neuronx_distributed_inference.utils.testing import build_function


def test_prefix_caching_reading_kv_cache():
    tp_degree = 1
    batch_size = 2
    num_hidden_layers = 3
    num_kv_head = 4
    hidden_size = 16
    num_attention_heads = 8

    pa_num_blocks = 16
    pa_block_size = 128
    seq_len = pa_num_blocks * pa_block_size // batch_size
    max_blocks_per_seq = pa_num_blocks // batch_size
    tiling_factor = 128 // max_blocks_per_seq

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        seq_len=seq_len,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        is_prefix_caching=True,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers * 2
    cache_layout = (pa_num_blocks +
                    BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                    tiling_factor, pa_block_size // tiling_factor, num_kv_head,
                    hidden_size // num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    _pa_mock_kv_cache_in_mgr(kv_cache_mgr)
    active_block_table = torch.randint(0, pa_num_blocks, (batch_size, max_blocks_per_seq))

    cache = kv_cache_mgr.get_cache(active_block_table=active_block_table)

    assert len(cache) == num_hidden_layers
    assert len(cache[0]) == 2
    assert cache[0][0].shape == (batch_size, num_kv_head, seq_len, hidden_size // num_attention_heads)

    for seq_id, active_block_per_seq in enumerate(active_block_table):
        expected = []
        for block_id in active_block_per_seq:
            expected.append(block_id * torch.ones((pa_block_size, num_kv_head, hidden_size // num_attention_heads)))
        expected = torch.cat(expected).reshape(seq_len, num_kv_head, hidden_size // num_attention_heads)
        expected = expected.permute(1, 0, 2) # SHD to HSD

        # check k cache from the layer 0
        actual = cache[0][0][seq_id]
        assert actual.shape == (
            num_kv_head,
            seq_len,
            hidden_size // num_attention_heads,
        )
        assert torch.equal(actual, expected)

        # check v cache from the layer 2
        actual = cache[2][1][seq_id]
        assert actual.shape == (
            num_kv_head,
            seq_len,
            hidden_size // num_attention_heads,
        )
        assert torch.equal(actual, expected)


def _pa_mock_kv_cache_in_mgr(kv_cache_mgr: BlockKVCacheManager):
    for layer_id in range(len(kv_cache_mgr.past_key_values)):
        for block_id in range(kv_cache_mgr.pa_num_blocks):
            kv_cache_mgr.past_key_values[layer_id][block_id, :, :, :] = block_id


def test_prefix_caching_writing_kv_cache_cte():
    tp_degree = 1
    batch_size = 1
    num_hidden_layers = 3
    num_kv_head = 4
    hidden_size = 16
    num_hidden_layers = 3
    num_attention_heads = 8

    pa_num_blocks = 16
    pa_block_size = 128
    seq_len = pa_num_blocks * pa_block_size // batch_size
    n_active_tokens = pa_num_blocks * pa_block_size // batch_size
    max_blocks_per_seq = pa_num_blocks // batch_size
    tiling_factor = 128 // max_blocks_per_seq

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        seq_len=seq_len,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        is_prefix_caching=True,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers * 2
    cache_layout = (pa_num_blocks +
                    BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                    tiling_factor, pa_block_size // tiling_factor, num_kv_head,
                    hidden_size // num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    latest = _pa_prepare_latest_kv_cache(
        batch_size=batch_size,
        n_active_tokens=n_active_tokens,
        head_dim=hidden_size // num_attention_heads,
        num_kv_heads_per_rank=num_kv_head,
        num_hidden_layers=num_hidden_layers,
    )

    slot_padding_id = -1
    # prompt with length of 4
    slot_mapping = torch.ones(batch_size, n_active_tokens, dtype=torch.int64) * slot_padding_id
    slot_mapping[0, :4] = torch.arange(24, 28)

    updated_cache = kv_cache_mgr.update_cache(
        is_for_context_encoding=False,
        seq_ids=None,
        position_ids=None,
        new_key_values=latest,
        seq_len=None,
        scatter_index=slot_mapping,
    )

    for slots_per_seq in slot_mapping:
        for seq_pos, slot in enumerate(slots_per_seq):
            if slot == slot_padding_id:
                continue  # skip for padding
            block_id = slot // pa_block_size
            block_offset = slot % pa_block_size

            # check the k cache for the layer 0
            actual = updated_cache[0].reshape(
                pa_num_blocks + BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                -1, num_kv_head, hidden_size // num_attention_heads)
            expected = latest[0][0]
            assert torch.equal(actual[block_id, block_offset, :, :],
                               expected[0, :, seq_pos, :])

            # check the v cache for the layer 2
            actual = updated_cache[5].reshape(
                pa_num_blocks + BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                -1, num_kv_head, hidden_size // num_attention_heads)
            expected = latest[2][1]
            assert torch.equal(actual[block_id, block_offset, :, :],
                               expected[0, :, seq_pos, :])


def test_prefix_caching_writing_kv_cache_tkg():
    tp_degree = 1
    batch_size = 3
    num_hidden_layers = 3
    num_kv_head = 4
    hidden_size = 16
    num_hidden_layers = 3
    num_attention_heads = 8

    pa_num_blocks = 12
    pa_block_size = 128
    seq_len = pa_num_blocks * pa_block_size // batch_size
    n_active_tokens = 1
    max_blocks_per_seq = pa_num_blocks // batch_size
    tiling_factor = 128 // max_blocks_per_seq

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        seq_len=seq_len,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        is_prefix_caching=True,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers * 2
    cache_layout = (pa_num_blocks +
                    BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                    tiling_factor, pa_block_size // tiling_factor, num_kv_head,
                    hidden_size // num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    latest = _pa_prepare_latest_kv_cache(
        batch_size=batch_size,
        n_active_tokens=n_active_tokens,
        head_dim=hidden_size // num_attention_heads,
        num_kv_heads_per_rank=num_kv_head,
        num_hidden_layers=num_hidden_layers,
    )

    slot_padding_id = -1
    # decode 3 seqs with context of length [4,2,7]
    slot_mapping = torch.ones(batch_size, n_active_tokens, dtype=torch.int64) * slot_padding_id
    slot_mapping[:, 0] = torch.tensor([4, 2, 7])

    updated_cache = kv_cache_mgr.update_cache(
        is_for_context_encoding=False,
        seq_ids=None,
        position_ids=None,
        new_key_values=latest,
        seq_len=None,
        scatter_index=slot_mapping,
    )

    for slots_per_seq in slot_mapping:
        for seq_pos, slot in enumerate(slots_per_seq):
            if slot == slot_padding_id:
                continue  # skip for padding
            block_id = slot // pa_block_size
            block_offset = slot % pa_block_size

            # check the k cache for the layer 0
            actual = updated_cache[0].reshape(
                pa_num_blocks + BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                -1, num_kv_head, hidden_size // num_attention_heads)
            expected = latest[0][0]
            assert torch.equal(actual[block_id, block_offset, :, :],
                               expected[0, :, seq_pos, :])

            # check the v cache for the layer 2
            actual = updated_cache[5].reshape(
                pa_num_blocks + BlockKVCacheManager._NUM_EXTRA_RESERVED_BLOCK,
                -1, num_kv_head, hidden_size // num_attention_heads)
            expected = latest[2][1]
            assert torch.equal(actual[block_id, block_offset, :, :],
                               expected[0, :, seq_pos, :])


def test_chunked_prefill_reading_kv_cache_cte():
    tp_degree=1
    batch_size=1
    num_hidden_layers=3
    num_kv_head=4
    hidden_size=16
    num_hidden_layers=3
    num_attention_heads=8

    pa_num_blocks=15
    pa_block_size=4

    cp_config = ChunkedPrefillConfig(tkg_model_enabled=True)

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        chunked_prefill_config=cp_config,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers*2
    cache_layout = (pa_num_blocks, num_kv_head, pa_block_size, hidden_size//num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert kv_cache_mgr.past_key_values[1].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    _pa_mock_kv_cache_in_mgr(kv_cache_mgr)
    active_block_table = torch.tensor([12,8,7,4,3,2,0,0])

    cache = kv_cache_mgr.get_cache(
        is_for_context_encoding=True,
        active_block_table=active_block_table,
    )

    assert len(cache) == num_hidden_layers
    assert len(cache[0]) == 2

    for block_id in range(pa_num_blocks):
        expected = block_id * torch.ones((num_kv_head, pa_block_size, hidden_size//num_attention_heads))

        # check k cache from the layer 0
        assert cache[0][0].shape == (pa_num_blocks, num_kv_head, pa_block_size, hidden_size//num_attention_heads)
        actual = cache[0][0][block_id, :, :, :]
        assert torch.equal(actual, expected)

        # check v cache from the layer 2
        assert cache[2][1].shape == (pa_num_blocks, num_kv_head, pa_block_size, hidden_size//num_attention_heads)
        actual = cache[2][1][block_id, :, :, :]
        assert torch.equal(actual, expected)


def test_chunked_prefill_reading_kv_cache_tkg():
    tp_degree=1
    batch_size=4
    num_hidden_layers=3
    num_kv_head=4
    hidden_size=16
    num_hidden_layers=3
    num_attention_heads=8

    pa_num_blocks=15
    pa_block_size=4

    cp_config = ChunkedPrefillConfig(tkg_model_enabled=True)

    kv_cache_mgr = _pa_prepare_cache_mgr(
        tp_degree=tp_degree,
        batch_size=batch_size,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        num_attention_heads=num_attention_heads,
        num_kv_head=num_kv_head,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        chunked_prefill_config=cp_config,
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers*2
    cache_layout = (pa_num_blocks, num_kv_head, pa_block_size, hidden_size//num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert kv_cache_mgr.past_key_values[1].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    _pa_mock_kv_cache_in_mgr(kv_cache_mgr)
    active_block_table = torch.tensor([12,8,7,4,3,2,0,0]).reshape(batch_size, -1)

    seq_len = active_block_table.shape[1] * pa_block_size

    cache = kv_cache_mgr.get_cache(
        is_for_context_encoding=False,
        active_block_table=active_block_table,
    )

    assert len(cache) == num_hidden_layers
    assert len(cache[0]) == 2

    actual_k = cache[0][0]
    assert actual_k.shape == (batch_size, num_kv_head, seq_len, hidden_size//num_attention_heads)

    actual_v = cache[2][1]
    assert actual_v.shape == (batch_size, num_kv_head, seq_len, hidden_size//num_attention_heads)

    num_blocks_per_seq = active_block_table.shape[1]
    for i in range(batch_size):
        for j in range(num_blocks_per_seq):
            block_id = active_block_table[i][j]
            expected = block_id * torch.ones((num_kv_head, pa_block_size, hidden_size//num_attention_heads))

            start_id = j * pa_block_size
            end_id = (j+1) * pa_block_size
            assert torch.equal(actual_k[i, :, start_id:end_id, :], expected)
            assert torch.equal(actual_v[i, :, start_id:end_id, :], expected)


def test_chunked_prefill_writing_kv_cache():
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
        chunked_prefill_config=ChunkedPrefillConfig(),
    )

    assert len(kv_cache_mgr.past_key_values) == num_hidden_layers*2
    cache_layout = (pa_num_blocks, num_kv_head, pa_block_size, hidden_size//num_attention_heads)
    assert kv_cache_mgr.past_key_values[0].shape == cache_layout
    assert kv_cache_mgr.past_key_values[1].shape == cache_layout
    assert torch.equal(kv_cache_mgr.past_key_values[0], torch.zeros(cache_layout))

    latest = _pa_prepare_latest_kv_cache(
        batch_size=batch_size,
        n_active_tokens=seq_len,
        head_dim=hidden_size//num_attention_heads,
        num_kv_heads_per_rank=num_kv_head,
        num_hidden_layers=num_hidden_layers,
    )

    # concatenated prompts with 3 seq [4,2,7], and it is padded with -1 to
    # fit bucket size
    slot_mapping = torch.tensor(
        [24, 25, 26, 27,
         16, 17,
          8,  9, 10, 11, 12, 13, 14,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )

    updated_cache = kv_cache_mgr.update_cache(
        is_for_context_encoding=False,
        seq_ids=None,
        position_ids=None,
        new_key_values=latest,
        seq_len=None,
        scatter_index=slot_mapping,
    )

    for seq_pos, slot in enumerate(slot_mapping.tolist()):
        if slot == -1:
            continue # skip for padding
        block_id = slot // pa_block_size
        block_offset = slot % pa_block_size

        # check the k cache for the layer 0
        actual = updated_cache[0]
        expected = latest[0][0]
        assert torch.equal(actual[block_id, :, block_offset, :], expected[0, :, seq_pos, :])

        # check the v cache for the layer 2
        actual = updated_cache[5]
        expected = latest[2][1]
        assert torch.equal(actual[block_id, :, block_offset, :], expected[0, :, seq_pos, :])


def _pa_prepare_latest_kv_cache(
    batch_size=1,
    n_active_tokens=32,
    head_dim=4,
    num_kv_heads_per_rank=8,
    num_hidden_layers=3,
):
    latest_kv_cache = []
    for layer_id in range(num_hidden_layers):
        k_cache = torch.ones(batch_size, num_kv_heads_per_rank, n_active_tokens, head_dim)
        v_cache = torch.ones(batch_size, num_kv_heads_per_rank, n_active_tokens, head_dim)

        for seq_pos in range(n_active_tokens):
            k_cache[:, :, seq_pos, :] *= 2 * seq_pos
            v_cache[:, :, seq_pos, :] *= 2 * seq_pos + 1

        latest_kv_cache.append([k_cache, v_cache])

    return latest_kv_cache


def _pa_prepare_cache_mgr(
    tp_degree=1,
    batch_size=1,
    pa_num_blocks=75,
    pa_block_size=128,
    seq_len=9600,
    num_attention_heads=8,
    num_kv_head=4,
    hidden_size=32,
    num_hidden_layers=3,
    is_prefix_caching=False,
    chunked_prefill_config=None,
):
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        is_block_kv_layout=True,
        pa_num_blocks=pa_num_blocks,
        pa_block_size=pa_block_size,
        torch_dtype=torch.float,
        is_chunked_prefill=chunked_prefill_config is not None,
        chunked_prefill_config=chunked_prefill_config,
        is_prefix_caching=is_prefix_caching,
        max_length=seq_len,
    )
    config = InferenceConfig(
        neuron_config=neuron_config,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    kv_cache_mgr = BlockKVCacheManager(config=config, num_kv_head=num_kv_head)
    return kv_cache_mgr


def test_generate_tokengen_slot_mapping():
    batch_size = 4
    num_blocks = 4
    block_size = 32

    example_inputs = [(
        torch.zeros((batch_size, 1), dtype=torch.int32),
        torch.zeros((batch_size, 1), dtype=torch.int32),
        torch.zeros((batch_size, num_blocks), dtype=torch.int32),
        torch.tensor(block_size, dtype=torch.int32)
    )]

    device_fn = build_function(generate_tokengen_slot_mapping, example_inputs)

    target_slot_mapping = torch.tensor([6, 40, 70, -1], dtype=torch.int32).unsqueeze(dim=1)
    position_ids = torch.tensor([6, 8, 126, 7], dtype=torch.int32).unsqueeze(dim=1)
    block_table = torch.tensor([[0, 0, 0, 0],
                                [1, 0, 0, 0],
                                [2, 3, 4, 5],
                                [0, 0, 0, 0]], dtype=torch.int32)

    # Generate CPU result
    cpu_result = generate_tokengen_slot_mapping(position_ids, target_slot_mapping, block_table, block_size)

    # Generate device result
    device_result = device_fn(position_ids, target_slot_mapping, block_table, torch.tensor(block_size, dtype=torch.int32))

    assert isinstance(cpu_result, torch.Tensor)
    assert isinstance(device_result, torch.Tensor)
    assert cpu_result.shape == device_result.shape
    assert cpu_result.dtype == device_result.dtype
    assert torch.allclose(cpu_result, device_result)


def test_generate_fusedspec_slot_mapping():
    batch_size = 4
    speculation_length = 5
    num_blocks = 8
    block_size = 32

    example_inputs = [(
        torch.zeros((batch_size, 1), dtype=torch.int32),
        torch.zeros((batch_size, speculation_length), dtype=torch.int32),
        torch.zeros((batch_size, num_blocks), dtype=torch.int32),
        torch.tensor(block_size, dtype=torch.int32)
    )]

    target_slot_mapping = torch.tensor([[41, 42, 43, 44, 45],
                                        [126, 127, 128, 129, 130],
                                        [-1, -1, -1, -1, -1],
                                        [-1, -1, -1, -1, -1]], dtype=torch.int32)
    position_ids = torch.tensor([41, 62, 41, 41], dtype=torch.int32).unsqueeze(dim=1)
    block_table = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0],
                                [2, 3, 4, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)

    device_fn = build_function(generate_fusedspec_slot_mapping, example_inputs)

    # Generate CPU result
    cpu_result = generate_fusedspec_slot_mapping(position_ids,
                                                 target_slot_mapping,
                                                 block_table,
                                                 block_size)

    # Generate device result
    device_result = device_fn(position_ids, target_slot_mapping, block_table, torch.tensor(block_size, dtype=torch.int32))


    assert isinstance(cpu_result, torch.Tensor)
    assert isinstance(device_result, torch.Tensor)
    assert cpu_result.shape == device_result.shape
    assert cpu_result.dtype == device_result.dtype
    assert torch.allclose(cpu_result, device_result)
