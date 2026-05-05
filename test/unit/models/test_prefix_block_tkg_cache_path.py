from unittest.mock import MagicMock

import pytest
import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.quantization.quantization_config import KVQuantizationConfig

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import (
    BlockKVCacheManager,
)


@pytest.fixture(autouse=True)
def mock_tensor_model_parallel_group():
    initial_tensor_model_parallel_group = parallel_state._TENSOR_MODEL_PARALLEL_GROUP
    initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
    initial_data_parallel_group = parallel_state._DATA_PARALLEL_GROUP
    initial_world_group = parallel_state._WORLD_GROUP

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = MagicMock(
        spec=torch.distributed.ProcessGroup
    )
    parallel_state._TENSOR_MODEL_PARALLEL_GROUP.size.return_value = 1
    parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
    parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
    parallel_state._DATA_PARALLEL_GROUP = MagicMock(spec=torch.distributed.ProcessGroup)
    parallel_state._DATA_PARALLEL_GROUP.size.return_value = 1
    parallel_state._WORLD_GROUP = MagicMock()
    parallel_state._WORLD_GROUP.size.return_value = 1

    yield

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = initial_tensor_model_parallel_group
    parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = initial_world_size
    parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = initial_rank
    parallel_state._DATA_PARALLEL_GROUP = initial_data_parallel_group
    parallel_state._WORLD_GROUP = initial_world_group


class _PassthroughDecoderLayer(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention

    def forward(self, hidden_states, **kwargs):
        attn_output = self.self_attn(hidden_states, **kwargs)
        return (
            attn_output.hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            attn_output.residual,
        )


class _ReturnPastDecoderLayer(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (
            hidden_states,
            kwargs["past_key_value"],
            kwargs.get("cos_cache"),
            kwargs.get("sin_cache"),
            kwargs.get("residual"),
        )


class _MinimalPrefixBlockTkgModel(NeuronBaseModel):
    def setup_attr_for_model(self, config):
        self.on_device_sampling = False
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config):
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [_PassthroughDecoderLayer(_make_minimal_attention(config))]
        )
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Identity()


def _make_minimal_attention(config):
    attn = NeuronAttentionBase.__new__(NeuronAttentionBase)
    torch.nn.Module.__init__(attn)

    neuron_config = config.neuron_config
    attn.config = config
    attn.neuron_config = neuron_config
    attn.torch_dtype = neuron_config.torch_dtype
    attn.attention_chunk_size = None
    attn.cp_degree = neuron_config.cp_degree
    attn.dp_degree = neuron_config.attention_dp_degree
    attn.sliding_window = None
    attn.sequence_parallel_enabled = neuron_config.sequence_parallel_enabled
    attn.tensor_model_parallel_group = None
    attn.attn_block_tkg_nki_kernel_enabled = (
        neuron_config.attn_block_tkg_nki_kernel_enabled
    )
    attn.attn_block_tkg_nki_kernel_cache_update = (
        neuron_config.attn_block_tkg_nki_kernel_cache_update
    )
    attn.k_cache_transposed = neuron_config.k_cache_transposed
    attn.logical_nc_config = neuron_config.logical_nc_config
    attn.learned_sinks_size = None
    attn.is_eagle3_draft = False
    return attn


def _make_config(attn_block_tkg_nki_kernel_enabled=True, kv_quant_config=None):
    neuron_config = NeuronConfig(
        batch_size=1,
        seq_len=1,
        n_active_tokens=1,
        n_positions=4,
        max_context_length=1,
        # Keep max_num_blocks_per_seq >= 128 so the non-block-TKG fallback test
        # exercises the ordinary 4D block cache layout instead of prefix tiling.
        max_length=512,
        is_prefix_caching=True,
        is_block_kv_layout=True,
        attn_block_tkg_nki_kernel_enabled=attn_block_tkg_nki_kernel_enabled,
        qkv_kernel_enabled=True,
        pa_num_blocks=4,
        pa_block_size=4,
        torch_dtype=torch.float32,
        on_cpu=True,
        logical_nc_config=1,
        kv_quant_config=kv_quant_config,
    )
    return InferenceConfig(
        neuron_config,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=16,
    )


def _install_cache_spies(model):
    events = []
    materialized_shapes = []

    original_materialize_bhsd = model.kv_mgr._get_block_cache_and_reshape_bhsd
    original_fetch_cache = model.kv_mgr._fetch_cache

    def materialize_bhsd_spy(*args, **kwargs):
        events.append("materialize_bhsd")
        result = original_materialize_bhsd(*args, **kwargs)
        materialized_shapes.append(tuple(result.shape))
        return result

    def fetch_cache_spy(*args, **kwargs):
        events.append("fetch_cache")
        return original_fetch_cache(*args, **kwargs)

    model.kv_mgr._get_block_cache_and_reshape_bhsd = MagicMock(
        side_effect=materialize_bhsd_spy
    )
    model.kv_mgr._fetch_cache = MagicMock(side_effect=fetch_cache_spy)
    return events, materialized_shapes


def _install_fake_block_tkg_attention(model, events, active_block_table, scatter_index):
    raw_cache_shapes_seen_by_attention = []

    def fake_attention_block_tkg(
        hidden_states,
        attention_mask,
        position_ids,
        kv_cache,
        active_mask,
        cos_cache,
        sin_cache,
        rmsnorm,
        rotary_position_ids,
        update_kv_per_layer,
        active_blocks_table,
        use_polar_compatible_rope=False,
    ):
        events.append("attention_block_tkg")
        raw_cache_shapes_seen_by_attention.append(
            (tuple(kv_cache[0].shape), tuple(kv_cache[1].shape))
        )
        assert torch.equal(position_ids, scatter_index)
        assert torch.equal(active_blocks_table, active_block_table)
        assert update_kv_per_layer is False
        return hidden_states, kv_cache, cos_cache, sin_cache

    model.layers[0].self_attn.attention_block_tokengen_nki_kernel = (
        fake_attention_block_tkg
    )
    return raw_cache_shapes_seen_by_attention


def _run_tokengen_model_output(model, active_block_table, scatter_index):
    input_ids = torch.tensor([[1]], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, 4), dtype=torch.bool)
    position_ids = torch.tensor([[1]], dtype=torch.long)
    active_mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    seq_ids = torch.tensor([0], dtype=torch.long)

    model.get_model_output(
        input_ids=input_ids,
        seq_ids=seq_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        active_mask=active_mask,
        update_cache=False,
        is_for_context_encoding=False,
        active_block_table=active_block_table,
        scatter_index=scatter_index,
        kvcache_buffer=None,
    )


def test_prefix_block_tkg_skips_bhsd_materialization_and_uses_raw_block_cache():
    config = _make_config()
    model = _MinimalPrefixBlockTkgModel(config)
    assert isinstance(model.kv_mgr, BlockKVCacheManager)

    events, materialized_shapes = _install_cache_spies(model)
    active_block_table = torch.tensor([[0]], dtype=torch.long)
    scatter_index = torch.tensor([[1]], dtype=torch.long)
    raw_cache_shapes_seen_by_attention = _install_fake_block_tkg_attention(
        model, events, active_block_table, scatter_index
    )

    _run_tokengen_model_output(model, active_block_table, scatter_index)

    attention_idx = events.index("attention_block_tkg")
    fetch_indices = [idx for idx, event in enumerate(events) if event == "fetch_cache"]

    assert model.kv_mgr._get_block_cache_and_reshape_bhsd.call_count == 0
    assert materialized_shapes == []
    assert model.kv_mgr._fetch_cache.call_count >= 1
    assert any(idx < attention_idx for idx in fetch_indices)
    assert raw_cache_shapes_seen_by_attention == [
        (tuple(model.kv_mgr.k_shape), tuple(model.kv_mgr.v_shape))
    ]


def test_prefix_without_block_tkg_still_materializes_bhsd():
    config = _make_config(attn_block_tkg_nki_kernel_enabled=False)
    model = _MinimalPrefixBlockTkgModel(config)
    model.layers[0] = _ReturnPastDecoderLayer()
    assert isinstance(model.kv_mgr, BlockKVCacheManager)

    _, materialized_shapes = _install_cache_spies(model)
    active_block_table = torch.tensor([[0]], dtype=torch.long)
    scatter_index = torch.tensor([[1]], dtype=torch.long)

    _run_tokengen_model_output(model, active_block_table, scatter_index)

    assert model.kv_mgr._get_block_cache_and_reshape_bhsd.call_count >= 1
    assert materialized_shapes == [(1, 1, 4, 4), (1, 1, 4, 4)]


def test_prefix_block_tkg_with_kv_quant_still_materializes_bhsd_for_now():
    kv_quant_config = KVQuantizationConfig(
        quant_dtype=torch.bfloat16,
        direct_cast=True,
    )
    config = _make_config(kv_quant_config=kv_quant_config)
    model = _MinimalPrefixBlockTkgModel(config)
    assert isinstance(model.kv_mgr, BlockKVCacheManager)

    events, materialized_shapes = _install_cache_spies(model)
    active_block_table = torch.tensor([[0]], dtype=torch.long)
    scatter_index = torch.tensor([[1]], dtype=torch.long)
    raw_cache_shapes_seen_by_attention = _install_fake_block_tkg_attention(
        model, events, active_block_table, scatter_index
    )

    _run_tokengen_model_output(model, active_block_table, scatter_index)

    assert model.kv_mgr._get_block_cache_and_reshape_bhsd.call_count >= 1
    assert model.kv_mgr._fetch_cache.call_count >= 2
    assert materialized_shapes == [(1, 1, 4, 4), (1, 1, 4, 4)]
    assert raw_cache_shapes_seen_by_attention == [
        (tuple(model.kv_mgr.k_shape), tuple(model.kv_mgr.v_shape))
    ]
