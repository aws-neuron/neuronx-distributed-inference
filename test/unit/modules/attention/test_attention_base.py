from unittest.mock import MagicMock, Mock, patch

import math
import pytest
import torch
from torch_xla.core import xla_model as xm

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import KVCacheManager
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    GroupQueryAttention_QKV,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.sliding_window.attention import (
    DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE, MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
)
from neuronx_distributed_inference.utils.testing import build_function

RANDOM_SEED = 0

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(RANDOM_SEED)

class MockTorchModule(MagicMock, torch.nn.Module):
    pass

@pytest.fixture
@patch("neuronx_distributed_inference.modules.attention.attention_base.init_data_parallel_attention_process_groups")
@patch("neuronx_distributed_inference.modules.attention.attention_base.init_context_parallel_attention_process_groups")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_tp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_tensor_model_parallel_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.GroupQueryAttention_QKV")
@patch("neuronx_distributed_inference.modules.attention.attention_base.GroupQueryAttention_O")
def attn_module(mock_GroupQueryAttention_O,
                mock_GroupQueryAttention_QKV,
                mock_get_tensor_model_parallel_group,
                mock_get_context_parallel_attention_tp_group,
                mock_init_context_parallel_attention_process_groups,
                mock_init_data_parallel_attention_process_groups,
                request):

    params = request.param if hasattr(request, "param") else {}
    tp_degree = params.get("tp_degree", 2)
    attention_chunk_size = params.get("attention_chunk_size", None)
    sliding_window = params.get("sliding_window", None)
    learned_sinks_size = params.get("learned_sinks_size", None)
    num_key_value_heads = params.get("num_key_value_heads", 4)

    mock_get_tensor_model_parallel_group.return_value = MagicMock()
    mock_get_tensor_model_parallel_group.return_value.size.return_value = tp_degree

    mock_qkv_instance = MagicMock()
    mock_qkv_instance.get_num_attention_heads.return_value = 4
    mock_qkv_instance.get_num_key_value_heads.return_value = num_key_value_heads
    mock_GroupQueryAttention_QKV.return_value = mock_qkv_instance

    neuron_config_kwargs = {
        "tp_degree": tp_degree, 
        "logical_nc_config": 1,
        "padding_side": "right",
        "torch_dtype": torch.float32,
        "on_cpu": True,
    }

    neuron_config_kwargs = {**neuron_config_kwargs, **params}
    neuron_config = NeuronConfig(**neuron_config_kwargs)

    # For instances where environment variables override the logical_nc_config, we use this as 
    # a workaround to respect what the attn unit test specified instead of the environment variable 
    if neuron_config.logical_nc_config != neuron_config_kwargs.get("logical_nc_config"):
        neuron_config.logical_nc_config = neuron_config_kwargs.get("logical_nc_config")

    config = InferenceConfig(neuron_config=neuron_config)

    module = _create_attn_module(
                config=config,
                hidden_size=16,
                num_attention_heads=4,
                num_key_value_heads=num_key_value_heads,
                attention_chunk_size=attention_chunk_size,
                sliding_window=sliding_window,
                learned_sinks_size=learned_sinks_size,
            )

    # set random seed explictly again since attn_module initialization increments the seed
    torch.manual_seed(RANDOM_SEED)
    
    return module

@pytest.mark.parametrize(
    "lnc, cp_degree, q_len, strided_context_parallel_kernel_enabled, expected_flash_attn_strategy, sliding_window",
    # fmt: off
    [
        (1, 4, 512, False, FlashAttentionStrategy.NONE, None), # LNC1
        (2, 4, 512, True, FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL, None), # LNC2, cp_enabled, q_len > head_dim, strided kernel enabled
        (2, 4, 512, False, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL, None), # LNC2, cp_enabled, q_len > head_dim
        (2, 4, 256, False, FlashAttentionStrategy.NONE, None), # LNC2, cp_enabled, q_len == head_dim
        (2, 4, 128, False, FlashAttentionStrategy.NONE, None), # LNC2, cp_enabled, q_len < head_dim
        (2, 1, 512, False, FlashAttentionStrategy.NONE, None), # LNC2, cp_disabled
        (2, 1, 512, True, None, None), # LNC2, cp_disabled, expected to error since strided CP requires CP is enabled
        (1, 4, 512, True, None, None), # LNC1, strided kernel enabled, expected to error since strided CP isn't supported when LNC = 1
        (2, 4, 128, True, None, None), # LNC2, q_len < head_dim, cp_enabled, expected to error since strided CP isn't supported when q_len <= head_dim
        (2, 4, 512, False, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL, 128), # LNC2, cp_enabled, q_len > head_dim, window_size set
        (2, 4, 512, True, None, 128), # LNC2, cp_enabled, q_len > head_dim, strided kernel enabled, window_size set
    ],
    # fmt: on
)
def test_get_flash_attention_strategy_cp(
    attn_module, lnc, cp_degree, q_len, strided_context_parallel_kernel_enabled, expected_flash_attn_strategy, sliding_window
):
    attn_module.logical_nc_config = lnc
    attn_module.head_dim = 256
    attn_module.cp_degree = cp_degree
    attn_module.strided_context_parallel_kernel_enabled = strided_context_parallel_kernel_enabled
    attn_module.sliding_window = sliding_window

    if expected_flash_attn_strategy is None:
        # Check that we correctly handle incorrect strided CP configs
        with pytest.raises(ValueError, match="Strided context parallel kernel is enabled but cannot be used"):
            attn_module.get_flash_attention_strategy_cp(q_len)
        return
    
    flash_attn_strategy = attn_module.get_flash_attention_strategy_cp(q_len)
    assert flash_attn_strategy == expected_flash_attn_strategy

@pytest.mark.parametrize(
    "platform, attn_kernel_enabled, lnc, cp_degree, q_len, has_attn_mask, sliding_window, expected_flash_attn_strategy",
    # fmt: off
    [
        ("trn2", None, 2, 4, 128, True, None, FlashAttentionStrategy.NONE), # LNC2, cp_enabled, q_len < head_dim
        ("trn2", None, 2, 1, 128, True, None, FlashAttentionStrategy.NONE),  # LNC2, q_len < 1024, not divisible by 256
        ("trn2", None, 2, 1, 3968, True, None, FlashAttentionStrategy.NONE),  # LNC2, q_len >= 1024, not divisible by 512
        ("trn2", True, 2, 1, 256, True, None, FlashAttentionStrategy.SHARDED_KERNEL), # LNC2, q_len < 1024, divisible by 256
        ("trn2", None, 2, 1, 256, True, None, FlashAttentionStrategy.SHARDED_KERNEL), # LNC2, q_len < 1024, divisible by 256
        ("trn2", True, 2, 1, 3968, True, None, FlashAttentionStrategy.NONE),  # LNC2, q_len >= 1024, not divisible by 512
        ("trn2", None, 2, 1, 4096, True, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        ("trn2", True, 2, 1, 4096, True, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        ("trn1", None, 1, 1, 4096, True, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        ("trn1", True, 1, 1, 4096, True, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        ("trn1", None, 1, 1, 1024, True, None, FlashAttentionStrategy.NONE),  # LNC1, 512 <= q_len < 4096
        ("trn1", True, 1, 1, 1024, True, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, enabled, 512 <= q_len < 4096
        ("trn1", None, 1, 1, 256, True, None, FlashAttentionStrategy.NONE),  # LNC1, q_len < 512
        ("trn1", True, 1, 1, 256, True, None, FlashAttentionStrategy.NONE),  # LNC1, enabled, q_len < 512
        ("trn2", None, 2, 1, 128, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, not divisible by 256
        ("trn2", None, 2, 1, 3968, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, not divisible by 512
        ("trn2", True, 2, 1, 256, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, divisible by 256
        ("trn2", None, 2, 1, 256, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, divisible by 256
        ("trn2", True, 2, 1, 3968, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, not divisible by 512
        ("trn2", None, 2, 1, 4096, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        ("trn2", True, 2, 1, 4096, False, None, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        # same as has_attn_mask=True for LNC1
        ("trn1", None, 1, 1, 4096, False, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        ("trn1", True, 1, 1, 4096, False, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        ("trn1", None, 1, 1, 1024, False, None, FlashAttentionStrategy.NONE),  # LNC1, 512 <= q_len < 4096
        ("trn1", True, 1, 1, 1024, False, None, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, enabled, 512 <= q_len < 4096
        ("trn1", None, 1, 1, 256, False, None, FlashAttentionStrategy.NONE),  # LNC1, q_len < 512
        ("trn1", True, 1, 1, 256, False, None, FlashAttentionStrategy.NONE),  # LNC1, enabled, q_len < 512

        # sliding window
        ("trn1", None, 1, 1, 256, True, 4096, FlashAttentionStrategy.NONE), # q_len < 512

        ("trn1", None, 1, 1, 2048, True, 4096, FlashAttentionStrategy.SLIDING_WINDOW_KERNEL), # q_len >= 512

        ("trn2", None, 2, 1, 256, True, 4096, FlashAttentionStrategy.SHARDED_KERNEL), # q_len < 512

        ("trn2", None, 2, 1, 2048, True, 4096, FlashAttentionStrategy.SHARDED_KERNEL), # q_len >= 512

        # turn off kernel explicitly
        ("trn2", False, 2, 4, 128, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 128, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 3968, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 256, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 256, False, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 3968, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 4096, True, None, FlashAttentionStrategy.NONE),
        ("trn2", False, 2, 1, 4096, False, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 4096, True, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 4096, False, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 1024, True, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 1024, False, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 256, True, None, FlashAttentionStrategy.NONE),
        ("trn1", False, 1, 1, 256, False, None, FlashAttentionStrategy.NONE),
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_platform_target")
def test_get_flash_attention_strategy(
    mock_get_platform_target, attn_module, platform, attn_kernel_enabled, lnc, cp_degree, q_len, has_attn_mask, sliding_window, expected_flash_attn_strategy
):
    mock_get_platform_target.return_value = platform
    # CP FA enablement is tested seperately, for the purpose of this function we are only interested that it's called.
    attn_module.get_flash_attention_strategy_cp = MagicMock()
    attn_module.get_flash_attention_strategy_cp.return_value = FlashAttentionStrategy.NONE

    attn_module.attn_kernel_enabled = attn_kernel_enabled
    attn_module.logical_nc_config = lnc
    attn_module.sliding_window = sliding_window
    flash_attn_strategy = attn_module.get_flash_attention_strategy(q_len, has_attn_mask)

    if cp_degree > 1 and attn_kernel_enabled:
        attn_module.get_flash_attention_strategy_cp.assert_called_once()
    elif cp_degree > 1 and not attn_kernel_enabled:
        attn_module.get_flash_attention_strategy_cp.assert_not_called()

    assert flash_attn_strategy == expected_flash_attn_strategy


@pytest.mark.parametrize(
    "attn_module, batch_size, seq_len, cp_degree, sequence_parallel",
    # fmt: off
    [
        ({}, 1, 8, 1, False),  # bs=1, context encoding, no sequence parallel, no context parallel
        ({}, 2, 1, 1, False),  # bs=2, context encoding, no sequence parallel, no context parallel
        ({}, 1, 8, 1, True),   # bs=1, token gen, sequence parallel, no context parallel
        ({}, 2, 1, 1, True),   # bs=2, token gen, sequence parallel, no context parallel
        ({"tp_degree": 4, "cp_degree": 2}, 1, 8, 2, False),   # bs=1, context encoding, no sequence parallel, context parallel
        ({"tp_degree": 4, "cp_degree": 2}, 1, 1, 2, True),   # bs=1, token gen, sequence parallel, context parallel
        ({"tp_degree": 4, "cp_degree": 2, "strided_context_parallel_kernel_enabled": True, "logical_nc_config": 2}, 1, 16, 2, False),   # bs=1, context encoding, no sequence parallel, context parallel
    ], indirect=["attn_module"],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_pos_emb")
@patch("neuronx_distributed_inference.modules.attention.attention_base.order_strided_tensor")
def test_prep_qkv_tensors_base_case(
    mock_order_strided_tensor, mock_apply_rotary_pos_emb, mock_get_context_parallel_attention_cp_group, 
    mock_gather_from_tensor_model_parallel_region_with_dim, attn_module, batch_size,
    cp_degree, seq_len, sequence_parallel
):
    # order_strided_tensor will act as an identity function
    mock_order_strided_tensor.side_effect = lambda kv, *_: kv

    # CP uses is_prefill_stage to determine whether to split on S or not
    attn_module.neuron_config.is_prefill_stage = seq_len > 1

    # When testing the CP flow we simply check that the right collectives are called and return the input as the output
    # The inputs are not really split across the CP groups
    mock_gather_from_tensor_model_parallel_region_with_dim.side_effect = lambda x, *args, **kwargs: x

    seq_len_factor = 1
    if sequence_parallel:
        seq_len_factor = 2
        _enable_sequence_parallel(attn_module, tp_degree=seq_len_factor)

    # Prepare qkv_proj mock.
    q_len = seq_len * seq_len_factor
    q = torch.rand((batch_size, q_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))
    attn_module.cte_qkv_proj = MockTorchModule(return_value=(q, k, v, None))
    attn_module.tkg_qkv_proj = MockTorchModule(return_value=(q, k, v, None))

    # PK Values is only used to check if it's not None in CP Flow
    past_key_values = [[
        torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim)),
        torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim)),
    ]]

    # Call function.
    position_ids = torch.ones((batch_size, seq_len))
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    actual_q, actual_k, actual_v, cos_cache, sin_cache, _ = attn_module.prep_qkv_tensors(
        position_ids=position_ids,
        hidden_states=hidden_states,
        past_key_value=past_key_values if seq_len == 1 else None,
    )

    # Check results.
    expected_q = (
        q.view(batch_size, q_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_k = (
        k.view(batch_size, q_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, q_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    mock_apply_rotary_pos_emb.assert_not_called()

    _check_qkv_proj_call(attn_module, hidden_states, cp_degree > 1, seq_len != 1)

    if attn_module.strided_context_parallel_kernel_enabled:
        mock_order_strided_tensor.assert_called_once()

    if cp_degree > 1 and seq_len > 1:
        mock_get_context_parallel_attention_cp_group.assert_called_once()
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_called_once()
    else:
        mock_get_context_parallel_attention_cp_group.assert_not_called()
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_not_called()


@pytest.mark.parametrize(
    "batch_size, seq_len, use_sin_cos_cache",
    # fmt: off
    [
        (1, 8, True),   # bs=1, context encoding, uses cache
        (2, 1, False),  # bs=2, token gen, no cache
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_pos_emb")
def test_prep_qkv_tensors_rotary_emb(
    mock_apply_rotary_pos_emb, attn_module, batch_size, seq_len, use_sin_cos_cache
):
    # Prepare qkv_proj mock.
    q = torch.rand((batch_size, seq_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))

    # Prepare rotary emb mock.
    cos_cache, sin_cache = Mock(), Mock()
    attn_module.rotary_emb = MagicMock(return_value=(cos_cache, sin_cache))

    # Mock apply_rotary_pos_emb as identity function for q, k.
    mock_apply_rotary_pos_emb.side_effect = lambda q, k, *_: (q, k)

    # Call function.
    position_ids = torch.ones((batch_size, seq_len))
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    actual_q, actual_k, actual_v, actual_cos_cache, actual_sin_cache, _ = attn_module.prep_qkv_tensors(
        position_ids=position_ids,
        hidden_states=hidden_states,
        past_key_value=None,  # Unused.
        cos_cache=cos_cache if use_sin_cos_cache else None,
        sin_cache=sin_cache if use_sin_cos_cache else None,
    )

    # Check output.
    expected_q = (
        q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_k = (
        k.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert actual_cos_cache == cos_cache
    assert actual_sin_cache == sin_cache

    # Check mocks.
    mock_apply_rotary_pos_emb.assert_called_once()
    torch.testing.assert_close(mock_apply_rotary_pos_emb.call_args.args[0], expected_q)
    torch.testing.assert_close(mock_apply_rotary_pos_emb.call_args.args[1], expected_k)
    assert mock_apply_rotary_pos_emb.call_args.args[2] == cos_cache
    assert mock_apply_rotary_pos_emb.call_args.args[3] == sin_cache

    if use_sin_cos_cache:
        attn_module.rotary_emb.assert_not_called()
    else:
        attn_module.rotary_emb.assert_called_once()
        torch.testing.assert_close(attn_module.rotary_emb.call_args.args[0], expected_v)
        assert torch.equal(attn_module.rotary_emb.call_args.args[1], position_ids)

    _check_qkv_proj_call(attn_module, hidden_states)


@pytest.mark.parametrize(
    "batch_size, seq_len, use_sin_cos_cache",
    # fmt: off
    [
        (1, 8, True),   # bs=1, context encoding, uses cache
        (2, 1, False),  # bs=2, token gen, no cache
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.precompute_freqs_cis")
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_polar_compatible")
def test_prep_qkv_tensors_polar_compatible_rotary_emb(
    mock_apply_rotary_polar_compatible, mock_precompute_freqs_cis, attn_module, batch_size, seq_len, use_sin_cos_cache
):
    # Prepare qkv_proj mock.
    q = torch.rand((batch_size, seq_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))

    rotary_freqs = torch.rand((seq_len * 2, attn_module.head_dim))
    mock_precompute_freqs_cis.return_value = rotary_freqs
    mock_apply_rotary_polar_compatible.return_value = (q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim), 
                                                       k.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim))

    # Call function.
    position_ids = torch.ones((batch_size, seq_len), dtype=torch.int32)
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    actual_q, actual_k, actual_v, actual_cos_cache, actual_sin_cache, _ = attn_module.prep_qkv_tensors(
        position_ids=position_ids,
        hidden_states=hidden_states,
        past_key_value=None,
        use_polar_compatible_rope=True,
    )

    # Check output.
    expected_q = (
        q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_k = (
        k.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)

    # Check mocks.
    mock_apply_rotary_polar_compatible.assert_called_once()
    mock_precompute_freqs_cis.assert_called_once()
    torch.testing.assert_close(mock_apply_rotary_polar_compatible.call_args.args[0], q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim))
    torch.testing.assert_close(mock_apply_rotary_polar_compatible.call_args.args[1], k.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim))
    torch.testing.assert_close(mock_apply_rotary_polar_compatible.call_args.args[2], rotary_freqs[position_ids])

@pytest.mark.parametrize(
    "batch_size, seq_len",
    # fmt: off
    [
        (1, 8),   # bs=1, context encoding
        (2, 1),   # bs=2, token gen
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_pos_emb")
def test_prep_qkv_tensors_skip_rope(mock_apply_rotary_pos_emb, attn_module, batch_size, seq_len):
    """Test prep_qkv_tensors skips rope computation when passed skip_rope flag."""
    # Prepare qkv_proj mock.
    q = torch.rand((batch_size, seq_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))
    attn_module.rotary_emb = Mock()

    # Call function.
    position_ids = torch.ones((batch_size, seq_len))
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    actual_q, actual_k, actual_v, cos_cache, sin_cache, _ = attn_module.prep_qkv_tensors(
        position_ids=position_ids, hidden_states=hidden_states, past_key_value=None, skip_rope=True
    )

    # Check results.
    expected_q = (
        q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_k = (
        k.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    attn_module.rotary_emb.assert_not_called()
    mock_apply_rotary_pos_emb.assert_not_called()

    _check_qkv_proj_call(attn_module, hidden_states)


@pytest.mark.parametrize(
    "batch_size, seq_len, qkv_cte_fuse_rope, is_prefill_stage",
    # fmt: off
    [
        (1, 8, True, True),    # bs=1, prefill stage, fuse rope enabled - should pass cos/sin to qkv_proj
        (2, 8, True, True),    # bs=2, prefill stage, fuse rope enabled - should pass cos/sin to qkv_proj
        (1, 8, False, True),   # bs=1, prefill stage, fuse rope disabled - should NOT pass cos/sin to qkv_proj
        (2, 8, False, True),   # bs=2, prefill stage, fuse rope disabled - should NOT pass cos/sin to qkv_proj
        (1, 1, True, False),   # bs=1, token gen, fuse rope enabled - should NOT pass cos/sin to qkv_proj
        (2, 1, True, False),   # bs=2, token gen, fuse rope enabled - should NOT pass cos/sin to qkv_proj
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_pos_emb")
def test_prep_qkv_tensors_qkv_cte_fuse_rope_nki_kernel(
    mock_apply_rotary_pos_emb, attn_module, batch_size, seq_len, qkv_cte_fuse_rope, is_prefill_stage
):
    """Test prep_qkv_tensors passes cos_cache and sin_cache to qkv_proj when qkv_cte_nki_kernel_fuse_rope is enabled during prefill stage."""
    # Configure qkv_cte_nki_kernel_fuse_rope and is_prefill_stage
    attn_module.qkv_cte_nki_kernel_fuse_rope = qkv_cte_fuse_rope
    attn_module.neuron_config.is_prefill_stage = is_prefill_stage
    
    # Prepare qkv_proj mock.
    q = torch.rand((batch_size, seq_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, seq_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))
    
    # Prepare rotary emb mock.
    cos_cache = torch.rand((batch_size, seq_len, attn_module.head_dim))
    sin_cache = torch.rand((batch_size, seq_len, attn_module.head_dim))
    attn_module.rotary_emb = MagicMock(return_value=(cos_cache, sin_cache))
    
    # Mock apply_rotary_pos_emb as identity function for q, k.
    mock_apply_rotary_pos_emb.side_effect = lambda q, k, *_: (q, k)

    # Call function
    position_ids = torch.ones((batch_size, seq_len))
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    actual_q, actual_k, actual_v, actual_cos_cache, actual_sin_cache, _ = attn_module.prep_qkv_tensors(
        position_ids=position_ids,
        hidden_states=hidden_states,
        past_key_value=None,
    )

    # Check results.
    expected_q = (
        q.view(batch_size, seq_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_k = (
        k.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, seq_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert torch.equal(actual_cos_cache, cos_cache)
    assert torch.equal(actual_sin_cache, sin_cache)

    # Determine if we should have passed cos/sin to qkv_proj
    should_pass_cos_sin_to_qkv = qkv_cte_fuse_rope and is_prefill_stage
    
    # Check mocks - rotary_emb should be called to compute cos/sin cache
    attn_module.rotary_emb.assert_called_once()
    assert torch.equal(attn_module.rotary_emb.call_args.args[1], position_ids)
    
    # Check that qkv_proj was called
    attn_module.qkv_proj.assert_called_once()
    qkv_proj_kwargs = attn_module.qkv_proj.call_args.kwargs
    torch.testing.assert_close(qkv_proj_kwargs["hidden_states"], hidden_states)
    
    if should_pass_cos_sin_to_qkv:
        # When fuse rope is enabled during context encoding, cos/sin should be passed to qkv_proj
        assert "cos_cache" in qkv_proj_kwargs
        assert "sin_cache" in qkv_proj_kwargs
        torch.testing.assert_close(qkv_proj_kwargs["cos_cache"], cos_cache)
        torch.testing.assert_close(qkv_proj_kwargs["sin_cache"], sin_cache)
        # apply_rotary_pos_emb should NOT be called since rope is fused in qkv kernel
        mock_apply_rotary_pos_emb.assert_not_called()
    else:
        # When fuse rope is disabled or not context encoding, cos/sin should NOT be passed to qkv_proj
        assert "cos_cache" not in qkv_proj_kwargs or qkv_proj_kwargs["cos_cache"] is None
        assert "sin_cache" not in qkv_proj_kwargs or qkv_proj_kwargs["sin_cache"] is None
        # apply_rotary_pos_emb should be called to apply rope separately
        mock_apply_rotary_pos_emb.assert_called_once()


def _check_qkv_proj_call(attn_module, hidden_states, is_context_parallel = False, is_cte = True):
    if is_context_parallel and is_cte:
        qkv_proj = attn_module.cte_qkv_proj
    elif is_context_parallel and not is_cte:
        qkv_proj = attn_module.tkg_qkv_proj
    else:
        qkv_proj = attn_module.qkv_proj

    qkv_proj.assert_called_once()
    qkv_proj_hidden_states = qkv_proj.call_args.kwargs["hidden_states"]
    assert torch.equal(qkv_proj_hidden_states, hidden_states)


@pytest.mark.parametrize(
    "batch_size, seq_len, expected_attn_output",
    # fmt: off
    [   # bs=1
        (1, 2, torch.tensor([[[[0.1759, 0.2698, 0.1507, 0.0317], [0.1949, 0.6581, 0.4875, 0.4498]],
                           [[0.5263, 0.2437, 0.5846, 0.0332], [0.3210, 0.2429, 0.7069, 0.4358]]]])),
        # bs=2
        (2, 2, torch.tensor([[[[0.7745, 0.4369, 0.5191, 0.6159], [0.7981, 0.7966, 0.2513, 0.4178]],
                           [[0.6965, 0.9143, 0.9351, 0.9412], [0.6425, 0.4419, 0.7186, 0.5217]]],
                          [[[0.0340, 0.9442, 0.8802, 0.0012], [0.2914, 0.7012, 0.6675, 0.1254]],
                           [[0.6923, 0.2038, 0.6833, 0.7529], [0.7883, 0.4838, 0.2903, 0.4184]]]])),
        (1, 4, torch.tensor([[[[0.7745, 0.4369, 0.5191, 0.6159],  [0.7981, 0.7966, 0.2513, 0.4178], [0.7654, 0.8145, 0.4770, 0.5902], [0.7185, 0.5962, 0.5026, 0.4822]],
                           [[0.0340, 0.9442, 0.8802, 0.0012], [0.2914, 0.7012, 0.6675, 0.1254], [0.4184, 0.5519, 0.6483, 0.2939], [0.5066, 0.6128, 0.5049, 0.2535]]]])),
    ],
    # fmt: on
)
def test_perform_prefill_no_flash_attn(attn_module, batch_size, seq_len, expected_attn_output):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)

    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attention_mask = _create_attn_mask(batch_size, seq_len)

    attn_output, flash_attn_strategy = attn_module.perform_prefill(
        q, k, v, seq_len, batch_size, attention_mask
    )
    assert flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(attn_output, expected_attn_output, atol=5e-5, rtol=2e-3)

@pytest.mark.parametrize(
    "batch_size, seq_len",
    # fmt: off
    [   # expected_attn_output shape (bs, n_heads, s, h_dim)
        
        (1, 4),
        (1, 3),
        (2, 4),
    ],
    # fmt: on
)
def test_perform_prefill_chunked_attn_no_flash_attn(attn_module, batch_size, seq_len):
    chunk_size = 2
    chunked_attn_mask  = _create_chunked_attn_mask(batch_size, seq_len, chunk_size)
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    expected_attn_output = _attn(q, k, v, chunked_attn_mask)
    attn_output, flash_attn_strategy = attn_module.perform_prefill_chunked_attn(
        q, k, v, seq_len, batch_size, chunked_attn_mask, chunk_size
    )
    assert flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(attn_output, expected_attn_output, atol=5e-5, rtol=2e-3)

@pytest.mark.parametrize(
    "batch_size, seq_len, window_size",
    # fmt: off
    [
        (1, 4, 2),
        (2, 8, 4),
        (2, 8, 1),
        (2, 8, 8),
    ],
    # fmt: on
)
def test_perform_prefill_window_attn_no_flash_attn(attn_module, batch_size, seq_len, window_size):
    windowed_attn_mask = _create_windowed_attention_mask(batch_size, seq_len, window_size)
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    expected_attn_output = _attn(q, k, v, windowed_attn_mask)
    attn_output, flash_attn_strategy = attn_module.perform_prefill_windowed_attn(
        q, k, v, seq_len, batch_size, windowed_attn_mask, window_size
    )
    assert flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(attn_output, expected_attn_output, atol=5e-5, rtol=2e-3)

@pytest.mark.parametrize("batch_size, seq_len", [(1, 1024), (2, 2048)])
@patch("neuronx_distributed_inference.modules.attention.attention_base.flash_fwd")
def test_perform_prefill_window_attn_flash_attn(mock_flash_fwd_call, attn_module, batch_size, seq_len):
    sliding_window = 512

    grid = (batch_size, attn_module.num_heads)
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim), dtype=torch.bfloat16)
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.bfloat16)
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.bfloat16)

    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.SLIDING_WINDOW_KERNEL)
    mock_flash_fwd_kernel = MagicMock()
    mock_flash_fwd_call.__getitem__.return_value = mock_flash_fwd_kernel

    _, flash_attn_strategy = attn_module.perform_prefill_windowed_attn(
        q, k, v, seq_len, batch_size, None, sliding_window
    )

    mock_flash_fwd_call.__getitem__.assert_called_once_with(grid)
    assert flash_attn_strategy == FlashAttentionStrategy.SLIDING_WINDOW_KERNEL

    _check_sliding_window_flash_attn_kernel_call(
        mock_flash_fwd_kernel, attn_module, batch_size, seq_len, sliding_window
    )

@pytest.mark.parametrize(
    "attn_module",
    [
        ({"num_key_value_heads": 4}),  # MHA config
        ({"num_key_value_heads": 2}),  # GQA config
    ],
    indirect=["attn_module"]
)
@pytest.mark.parametrize("batch_size", [1, 2])
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call_nki")
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_new_kernel", new=True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_native_gqa_tp_support", new=True)
def test_perform_prefill_unsharded_flash_attn(mock_flash_fwd_call, attn_module, batch_size):
    seq_len = 2  # Use a seq_len > 1 for prefill.
    _setup_perform_prefill_flash_attn_test(
        attn_module, batch_size, seq_len, seq_len, FlashAttentionStrategy.UNSHARDED_KERNEL
    )
    _check_flash_attn_kernel_call(mock_flash_fwd_call, attn_module, batch_size, seq_len)

@pytest.mark.parametrize(
    "attn_module",
    [
        ({"num_key_value_heads": 4}),  # MHA config
        ({"num_key_value_heads": 2}),  # GQA config
    ],
    indirect=["attn_module"]
)
@pytest.mark.parametrize("batch_size", [1, 2])
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call_nki")
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_new_kernel", new=True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_native_gqa_tp_support", new=True)
def test_perform_prefill_sharded_flash_attn(mock_flash_fwd_call, mock_nc, attn_module, batch_size):
    seq_len = 2  # Use a seq_len > 1 for prefill.

    # Mock accessing grid index.
    nc_value = Mock()
    mock_nc.return_value = nc_value
    grid = (nc_value,)
    mock_flash_fwd_kernel = MagicMock()
    mock_flash_fwd_call.__getitem__.return_value = mock_flash_fwd_kernel

    _setup_perform_prefill_flash_attn_test(
        attn_module, batch_size, seq_len, seq_len, FlashAttentionStrategy.SHARDED_KERNEL
    )

    mock_nc.assert_called_once_with(attn_module.logical_nc_config)
    mock_flash_fwd_call.__getitem__.assert_called_once_with(grid)

    _check_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len)

@pytest.mark.parametrize(
    "attn_module",
    [
        ({"num_key_value_heads": 4}),  # MHA config
        ({"num_key_value_heads": 2}),  # GQA config
    ],
    indirect=["attn_module"]
)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_new_pc_kernel", True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_native_gqa_tp_support", True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_pc_call_nki")
def test_perform_prefix_prefill_unsharded_flash_attn(mock_flash_fwd_call, attn_module):
    seq_len = 3  # Use a seq_len > 1 for prefill.
    seq_len_prior = 2
    batch_size = 1 # use batch size 1 for prefill

    _setup_perform_prefix_prefill_flash_attn_test(
        attn_module, batch_size, seq_len, seq_len, seq_len_prior, FlashAttentionStrategy.UNSHARDED_KERNEL
    )
    _check_prefix_prefill_flash_attn_kernel_call(mock_flash_fwd_call, attn_module, batch_size, seq_len, seq_len_prior)

@pytest.mark.parametrize(
    "attn_module",
    [
        ({"num_key_value_heads": 4}),  # MHA config
        ({"num_key_value_heads": 2}),  # GQA config
    ],
    indirect=["attn_module"]
)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_new_pc_kernel", True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_native_gqa_tp_support", True)
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_pc_call_nki")
def test_perform_prefix_prefill_sharded_flash_attn(mock_flash_fwd_call, mock_nc, attn_module):
    seq_len = 3  # Use a seq_len > 1 for prefill.
    seq_len_prior = 2
    batch_size = 1 # use batch size 1 for prefill

    # Mock accessing grid index.
    nc_value = Mock()
    mock_nc.return_value = nc_value
    grid = (nc_value,)
    mock_flash_fwd_kernel = MagicMock()
    mock_flash_fwd_call.__getitem__.return_value = mock_flash_fwd_kernel

    _setup_perform_prefix_prefill_flash_attn_test(
        attn_module, batch_size, seq_len, seq_len, seq_len_prior, FlashAttentionStrategy.SHARDED_KERNEL
    )

    mock_nc.assert_called_once_with(attn_module.logical_nc_config)
    mock_flash_fwd_call.__getitem__.assert_called_once_with(grid)

    _check_prefix_prefill_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len, seq_len_prior)

@pytest.mark.parametrize(
    "attn_module, batch_size, flash_attention_strategy",
    # fmt: off
    [
        # MHA configs
        ({"num_key_value_heads": 4}, 1, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 4}, 2, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 4}, 1, FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 4}, 2, FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL),
        ({"learned_sinks_size": 1, "num_key_value_heads": 4}, 1, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
        # GQA configs
        ({"num_key_value_heads": 2}, 1, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 2}, 2, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 2}, 1, FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL),
        ({"num_key_value_heads": 2}, 2, FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL),
        ({"learned_sinks_size": 1, "num_key_value_heads": 2}, 1, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL),
    ], indirect=["attn_module"]
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call_nki")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call_strided_context_parallel")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_platform_target")
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_new_kernel", new=True)
@patch("neuronx_distributed_inference.modules.attention.attention_base._has_native_gqa_tp_support", new=True)
def test_perform_prefill_context_parallel_attn_kernel(mock_get_platform_target, mock_flash_fwd_call_strided_context_parallel, mock_flash_fwd_call, mock_nc,
                                                      attn_module, batch_size, flash_attention_strategy):
    # Needed to use NKI kernel for non-strided CP attention
    mock_get_platform_target.return_value = "trn2"

    q_len = 16
    seq_len = 64
    attn_module.cp_degree = seq_len // q_len
    attn_module.neuron_config.is_prefill_stage = True

    # Mock accessing grid index.
    nc_value = Mock()
    mock_nc.return_value = nc_value

    grid = (nc_value,)
    mock_flash_fwd_kernel = MagicMock()
    mock_flash_fwd_call.__getitem__.return_value = mock_flash_fwd_kernel

    mock_flash_fwd_strided_kernel = MagicMock()
    mock_flash_fwd_call_strided_context_parallel.__getitem__.return_value = mock_flash_fwd_strided_kernel

    rank_util = Mock()
    attn_module.rank_util = rank_util
    rank_util.get_rank.return_value = torch.tensor(2) # test is running in tp = 2, cp_rank = 2 // 2 == 1

    _setup_perform_prefill_flash_attn_test(
        attn_module, batch_size, q_len, seq_len, flash_attention_strategy
    )

    mock_nc.assert_called_once_with(attn_module.logical_nc_config)

    assert mock_flash_fwd_call_strided_context_parallel.__getitem__.called != mock_flash_fwd_call.__getitem__.called
    if flash_attention_strategy == FlashAttentionStrategy.STRIDED_CONTEXT_PARALLEL_KERNEL:
        mock_flash_fwd_call_strided_context_parallel.__getitem__.assert_called_once_with(grid)
        _check_strided_context_parallel_flash_attn_kernel_call(mock_flash_fwd_strided_kernel, attn_module, batch_size, q_len, seq_len, flash_attention_strategy)

    if flash_attention_strategy == FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL:
        mock_flash_fwd_call.__getitem__.assert_called_once_with(grid)
        _check_context_parallel_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, q_len, seq_len, flash_attention_strategy)


def _setup_perform_prefill_flash_attn_test(attn_module, batch_size, q_len, seq_len, flash_attn_strategy):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=flash_attn_strategy)

    q = torch.rand((batch_size, attn_module.num_heads, q_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attention_mask = _create_attn_mask(batch_size, seq_len)

    _, flash_attn_strategy = attn_module.perform_prefill(
        q, k, v, q_len, batch_size, attention_mask
    )
    assert flash_attn_strategy == flash_attn_strategy

def _setup_perform_prefix_prefill_flash_attn_test(attn_module, batch_size, q_len, seq_len, seqlen_prior, flash_attn_strategy):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=flash_attn_strategy)
    q = torch.rand((batch_size, attn_module.num_heads, q_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    k_prior = torch.rand((batch_size, attn_module.num_key_value_heads, seqlen_prior, attn_module.head_dim))
    v_prior = torch.rand((batch_size, attn_module.num_key_value_heads, seqlen_prior, attn_module.head_dim))
    prior_mask = torch.rand((batch_size, seqlen_prior))
    past_key_value = [k_prior, v_prior]
    active_mask = _create_attn_mask(batch_size, seq_len)

    _, flash_attn_strategy = attn_module.perform_prefix_prefill(
        q, k, v, q_len, batch_size, prior_mask, past_key_value, active_mask
    )
    assert flash_attn_strategy == flash_attn_strategy

def _check_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert (len(flash_fwd_args) == 4 or len(flash_fwd_args) == 5)

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    assert q_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert k_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert v_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )

    assert flash_fwd_args[3] == 1.0
    assert flash_fwd_kwargs["kernel_name"] == "CausalAttentionMMSoftmaxMMWithoutSwap"
    assert flash_fwd_kwargs["do_out_tp"]
    assert flash_fwd_kwargs["tp_q"]
    assert flash_fwd_kwargs["tp_k"]

def _check_prefix_prefill_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len, seq_len_prior):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert (len(flash_fwd_args) == 4)

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    scale_fwd_arg = flash_fwd_args[3]
    assert q_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert k_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert v_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert scale_fwd_arg == 1.0

    assert (len(flash_fwd_kwargs) == 7)
    assert flash_fwd_kwargs["kernel_name"] == "CausalAttentionMMSoftmaxMMWithoutSwap"
    k_prior_fwd_arg = flash_fwd_kwargs["k_prior"]
    v_prior_fwd_arg = flash_fwd_kwargs["v_prior"]
    prior_used_len_fwd_arg = flash_fwd_kwargs["prior_used_len"]

    assert k_prior_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len_prior,
        attn_module.head_dim,
    )
    assert v_prior_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len_prior,
        attn_module.head_dim,
    )
    assert prior_used_len_fwd_arg.shape == (1,)
    assert flash_fwd_kwargs["do_out_tp"]
    assert flash_fwd_kwargs["tp_q"]
    assert flash_fwd_kwargs["tp_k"]

def _check_sliding_window_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len, sliding_window):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert len(flash_fwd_args) == 3

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    assert q_flash_fwd_arg.shape == (
        batch_size, 
        attn_module.num_heads,
        attn_module.head_dim,
        seq_len,
    )
    assert k_flash_fwd_arg.shape == (
        batch_size,
        attn_module.num_heads,
        attn_module.head_dim,
        seq_len,
    )
    assert v_flash_fwd_arg.shape == (
        batch_size,
        attn_module.num_heads,
        seq_len,
        attn_module.head_dim,
    )

    assert flash_fwd_kwargs["window_size"] == (sliding_window-1, -1)
    assert flash_fwd_kwargs["config"].seq_tile_size == (
        DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE if seq_len >= DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE else MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
    )

def _check_strided_context_parallel_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, q_len, seq_len, strategy):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert len(flash_fwd_args) == 5

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    assert q_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        q_len,
        attn_module.head_dim,
    )
    assert k_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        attn_module.head_dim,
        seq_len,
    )
    assert v_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        seq_len,
        attn_module.head_dim,
    )

    attn_output_flash_fwd_arg = flash_fwd_args[4]
    assert attn_output_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        attn_module.head_dim,
        q_len,
    )
    assert flash_fwd_args[3] == 1.0
    assert flash_fwd_kwargs["kernel_name"] == "CausalAttentionMMSoftmaxMMWithoutSwap"
    assert flash_fwd_kwargs["global_n_tiles"] == attn_module.cp_degree

    assert flash_fwd_kwargs["tile_i"] == 1 # we hardcode to global rank 1 for unit testing
    assert flash_fwd_kwargs["strided_q_slicing"]

def _check_context_parallel_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, q_len, seq_len, strategy):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert len(flash_fwd_args) == 4

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    assert q_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        q_len,
        attn_module.head_dim,
    )
    assert k_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )
    assert v_flash_fwd_arg.shape == (
        batch_size * attn_module.num_key_value_heads,
        seq_len,
        attn_module.head_dim,
    )

    assert flash_fwd_args[3] == 1.0
    assert flash_fwd_kwargs["kernel_name"] == "CausalAttentionMMSoftmaxMMWithoutSwap"
    assert flash_fwd_kwargs["global_cp_deg"] == attn_module.cp_degree

    assert flash_fwd_kwargs["cp_offset"].item() == 1 * q_len # we hardcode to global rank 1 for unit testing

    if attn_module.learned_sinks_size is not None:
        assert flash_fwd_kwargs["sink"].shape == (attn_module.num_heads, 1)

    assert flash_fwd_kwargs["do_out_tp"]
    assert flash_fwd_kwargs["tp_q"]
    assert flash_fwd_kwargs["tp_k"]

@pytest.mark.parametrize(
    "batch_size, seq_len, expected_attn_output",
    # fmt: off
    [
        # bs=1, basic TKG
        (1, 1, torch.tensor([[[[0.4416, 0.5398, 0.1560, 0.1593]], [[0.4122, 0.9235, 0.5826, 0.7991]]]])),
        # bs=1, speculation
        (1, 2, torch.tensor([[[[0.4528, 0.3471, 0.3211, 0.3019], [0.4926, 0.6658, 0.4152, 0.4677]],
                              [[0.6030, 0.5459, 0.7426, 0.4424], [0.4764, 0.3449, 0.7154, 0.4827]]]])),
        # bs=2, basic TKG
        (2, 1, torch.tensor([[[[0.4528, 0.3471, 0.3211, 0.3019]], [[0.4644, 0.9512, 0.4642, 0.5612]]],
                             [[[0.6030, 0.5459, 0.7426, 0.4424]], [[0.3629, 0.1561, 0.6844, 0.4984]]]])),
        # bs=2, speculation
        (2, 2, torch.tensor([[[[0.5184, 0.2393, 0.4476, 0.6107], [0.5210, 0.5512, 0.4504, 0.4682]],
                              [[0.6098, 0.6970, 0.7765, 0.8826], [0.7042, 0.5471, 0.7624, 0.5709]]],
                             [[[0.0426, 0.5981, 0.8600, 0.2530], [0.2262, 0.4586, 0.5537, 0.1997]],
                              [[0.5312, 0.5167, 0.7310, 0.3496], [0.6794, 0.4898, 0.4524, 0.2759]]]])),
    ],
    # fmt: on
)
def test_compute_for_token_gen(attn_module, batch_size, seq_len, expected_attn_output):
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    position_ids = torch.ones((batch_size, seq_len))
    past_k = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    past_v = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    past_key_value = (past_k, past_v)
    attention_mask = _create_attn_mask(batch_size, seq_len)
    active_mask = None
    if seq_len > 1:
        # Speculation case.
        active_mask = _create_attn_mask(batch_size, seq_len)

    attn_output = attn_module.compute_for_token_gen(
        q, k, v, position_ids, past_key_value, attention_mask, active_mask=active_mask
    )
    torch.testing.assert_close(attn_output, expected_attn_output, atol=5e-5, rtol=1e-4)

@pytest.mark.parametrize(
    "attn_module, batch_size, seq_len, sp_enabled",
    # fmt: off
    [
        ({"cp_degree": 2, "tp_degree": 4}, 1, 64, False),  # bs=1, context encoding, no sequence parallel
        ({"cp_degree": 2, "tp_degree": 4}, 2, 64, False),  # bs=2, context encoding, no sequence parallel
        ({"cp_degree": 2, "tp_degree": 4}, 1, 1, False),   # bs=1, token generation, no sequence parallel
        ({"cp_degree": 2, "tp_degree": 4}, 1, 64, True),   # bs=1, context encoding, sequence parallel
        ({"cp_degree": 2, "tp_degree": 4}, 2, 64, True),   # bs=2, context encoding, sequence parallel
        ({"cp_degree": 2, "attention_dp_degree": 2, "tp_degree": 4, "batch_size": 2, "is_continuous_batching": True}, 2, 1, False), # bs=4, context encoding, sequence parallel, cp with dp
    ], indirect=["attn_module"]
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_data_parallel_attention_dp_group")
def test_forward_context_parallel(mock_get_context_parallel_attention_dp_group, 
                                  mock_get_context_parallel_attention_cp_group, 
                                  mock_gather_from_tensor_model_parallel_region_with_dim, 
                                  attn_module, batch_size, seq_len, sp_enabled):

    attn_module.neuron_config.is_prefill_stage = seq_len > 1
    
    cp_degree = attn_module.cp_degree
    dp_degree = attn_module.dp_degree
    tp_degree = attn_module.tp_degree
    
    if sp_enabled:
        _enable_sequence_parallel(attn_module, tp_degree // cp_degree)
        
    attn_module.cp_degree = cp_degree

    rank_util = Mock()
    attn_module.rank_util = rank_util
    rank_util.get_rank.return_value = torch.tensor(0)

    q_len = seq_len // cp_degree if seq_len > 1 else seq_len

    # Prepare prep_qkv_tensors and o_proj mocks.
    q = torch.rand((batch_size // dp_degree, attn_module.num_heads, q_len, attn_module.head_dim))
    k = torch.rand((batch_size // dp_degree, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size // dp_degree, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attn_module.prep_qkv_tensors = MagicMock(return_value=(q, k, v, None, None, None))

    output_seq_len = seq_len if not sp_enabled else seq_len // cp_degree

    o_proj_output = torch.rand((batch_size // dp_degree, attn_module.num_attention_heads * attn_module.head_dim, attn_module.hidden_size))
    attn_output = torch.rand((batch_size // dp_degree, output_seq_len, attn_module.hidden_size))

    # o_proj output is defined based on if we do the final attention gather or not
    # In TKG or SP enabled, we don't gather so o_proj output is the final output
    o_proj_output = o_proj_output if seq_len > 1 and not sp_enabled else attn_output

    if dp_degree > 1:
        attn_module.o_proj = MockTorchModule(return_value=o_proj_output)
        o_proj = attn_module.o_proj
    elif seq_len == 1:
        attn_module.tkg_o_proj = MockTorchModule(return_value=o_proj_output)
        o_proj = attn_module.tkg_o_proj
    else:
        attn_module.cte_o_proj = MockTorchModule(return_value=o_proj_output)
        o_proj = attn_module.cte_o_proj

    mock_gather_from_tensor_model_parallel_region_with_dim.return_value = attn_output

    # If SP is enabled, pass in input // tp
    input_seq_len = seq_len if not sp_enabled else seq_len // tp_degree

    hidden_states = torch.rand((batch_size, input_seq_len, attn_module.hidden_size))
    attention_mask = torch.full((batch_size, 1, seq_len, seq_len), True)
    position_ids = torch.ones((batch_size, seq_len))

    past_key_value = None
    if seq_len == 1:
        past_k = torch.rand(
            (batch_size // dp_degree, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_v = torch.rand(
            (batch_size // dp_degree, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_key_value = (past_k, past_v)

    actual_output, past_key_value, cos_cache, sin_cache = attn_module.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
    )

    torch.testing.assert_close(actual_output, attn_output)
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    attn_module.prep_qkv_tensors.assert_called_once()
    o_proj.assert_called_once()

    if dp_degree > 1:
        mock_get_context_parallel_attention_dp_group.assert_called_once()
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_called_once()

        # Assert that the split happened
        attn_module.prep_qkv_tensors.call_args.args[1].shape == (batch_size // dp_degree, input_seq_len, attn_module.hidden_size)

    if seq_len > 1 and not sp_enabled:
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_called_once()
        mock_get_context_parallel_attention_cp_group.assert_called_once()
    elif sp_enabled:
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_not_called()
        mock_get_context_parallel_attention_cp_group.assert_not_called()
        mock_get_context_parallel_attention_dp_group.assert_not_called()

    o_proj_input = o_proj.call_args.args[0]
    assert o_proj_input.shape == (batch_size // dp_degree, q_len, attn_module.num_heads * attn_module.head_dim)

def _setup_chunked_attn_test_config(attn_module, seq_len, sp_enabled, optimize_interleave_attn, is_pre_global_attn_layer, is_post_global_attn_layer):
    """Setup test configuration for chunked attention test."""
    if sp_enabled:
        _enable_sequence_parallel_chunked_attn(
            attn_module, attn_module.tp_degree // attn_module.cp_degree, optimize_interleave_attn, is_pre_global_attn_layer, is_post_global_attn_layer
        )
    attn_module.neuron_config.is_prefill_stage = seq_len > 1
    
    return {
        'cp_degree': attn_module.cp_degree,
        'tp_degree': attn_module.tp_degree,
        'attn_chunk_size': attn_module.attention_chunk_size,
        'chunk_lens': _get_chunk_lens(seq_len, attn_module.attention_chunk_size),
        'n_chunks': math.ceil(seq_len / attn_module.attention_chunk_size) if seq_len > 1 else 1,
        'q_len': attn_module.attention_chunk_size if seq_len > 1 else seq_len
    }

def _setup_chunked_attn_mocks(mock_get_context_parallel_attention_cp_group, attn_module, config):
    """Setup mocks for chunked attention test."""
    attn_module.cp_degree = config['cp_degree']
    mock_get_context_parallel_attention_cp_group.return_value = MagicMock()
    mock_get_context_parallel_attention_cp_group.return_value.size().return_value = config['cp_degree']
    
    rank_util = Mock()
    attn_module.rank_util = rank_util
    rank_util.get_rank.return_value = torch.tensor(0)

def _setup_token_gen_mocks(attn_module, seq_len, config):
    """Setup mocks for token generation case."""
    q = torch.rand((1, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((1, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((1, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    
    attn_module.prep_qkv_tensors = MagicMock(return_value=(q, k, v, None, None, None))
    tkg_o_proj_output = torch.rand((1, 1, attn_module.hidden_size))
    attn_module.tkg_o_proj = MockTorchModule(return_value=tkg_o_proj_output)
    
    past_k = torch.rand((1, attn_module.num_key_value_heads, config['attn_chunk_size'], attn_module.head_dim))
    past_v = torch.rand((1, attn_module.num_key_value_heads, config['attn_chunk_size'], attn_module.head_dim))
    past_key_value = (past_k, past_v)
    
    if attn_module.neuron_config.attn_block_tkg_nki_kernel_enabled:
        attn_module.attention_block_tokengen_nki_kernel_chunked_attn = MagicMock(
            return_value=(tkg_o_proj_output, past_key_value, None, None)
        )
    
    return {'output': tkg_o_proj_output, 'past_key_value': past_key_value, 'k': k, 'v': v, 'o_proj': attn_module.tkg_o_proj}

def _setup_context_encode_mocks(attn_module, config, is_post_global_attn_layer, is_pre_global_attn_layer, optimize_interleave_attn, sp_enabled):
    """Setup mocks for context encoding case."""
    attn_module.compute_attention_per_chunk_with_context_parallel = MagicMock()
    attn_module.rotary_emb = MagicMock(return_value=(None, None))
    
    final_outputs = []
    last_k_chunk = None
    last_v_chunk = None
    
    for q_len in config['chunk_lens']:
        k = torch.rand((1, attn_module.num_key_value_heads, q_len, attn_module.head_dim))
        v = torch.rand((1, attn_module.num_key_value_heads, q_len, attn_module.head_dim))
        chunked_attn_output = torch.rand((1, q_len, attn_module.hidden_size))
        last_k_chunk = k
        last_v_chunk = v
        
        if is_post_global_attn_layer or is_pre_global_attn_layer:
            final_outputs.append(tuple([chunked_attn_output, k, v, torch.rand(1, config['attn_chunk_size'] // config['cp_degree'], attn_module.hidden_size)]))
        else:
            final_outputs.append(tuple([chunked_attn_output, k, v, None]))
    
    attn_module.compute_attention_per_chunk_with_context_parallel.side_effect = final_outputs
    
    cte_o_proj_output = None
    if not optimize_interleave_attn and not sp_enabled:
        cte_o_proj_output = torch.rand((1, sum(config['chunk_lens']), attn_module.hidden_size))
    elif optimize_interleave_attn:
        if is_pre_global_attn_layer:
            cte_o_proj_output = torch.rand((1, config['attn_chunk_size'] // config['cp_degree'], attn_module.hidden_size))
        else:
            cte_o_proj_output = torch.rand((1, config['attn_chunk_size'] // config['tp_degree'], attn_module.hidden_size))
    
    attn_module.cte_o_proj = MockTorchModule(return_value=cte_o_proj_output)
    
    return {'output': cte_o_proj_output, 'last_k': last_k_chunk, 'last_v': last_v_chunk, 'o_proj': attn_module.cte_o_proj}

def _setup_sequence_parallel_mocks(mock_gather_from_sequence_parallel_region, mock_gather_from_tensor_model_parallel_region, mock_split_along_dim, seq_len, config, is_post_global_attn_layer, is_pre_global_attn_layer, optimize_interleave_attn, attn_module):
    """Setup mocks for sequence parallel case."""
    if is_post_global_attn_layer or not optimize_interleave_attn:
        mock_gather_from_sequence_parallel_region.return_value = torch.rand((1, seq_len, attn_module.hidden_size))
    elif is_pre_global_attn_layer:
        gathered_hidden_states = torch.rand((1, seq_len // config['cp_degree'], attn_module.hidden_size))
        gathered_attn_output = torch.rand((1, config['attn_chunk_size'], attn_module.hidden_size))
        mock_gather_from_sequence_parallel_region.return_value = gathered_hidden_states
        mock_gather_from_tensor_model_parallel_region.return_value = gathered_attn_output
    else:
        mock_gather_from_sequence_parallel_region.return_value = torch.rand((1, seq_len // config['cp_degree'], attn_module.hidden_size))
    
    if not optimize_interleave_attn or is_pre_global_attn_layer:
        expected_output = torch.rand((1, seq_len // config['tp_degree'], attn_module.hidden_size))
        mock_split_along_dim.return_value = expected_output
        return expected_output
    return None

def _verify_chunked_attn_assertions(attn_module, seq_len, config, sp_enabled, optimize_interleave_attn, is_pre_global_attn_layer, mock_gather_from_sequence_parallel_region, mock_gather_from_tensor_model_parallel_region, mock_split_along_dim, past_key_value, last_k, last_v, actual_output, expected_output):
    """Verify assertions for chunked attention test."""
    if attn_module.neuron_config.attn_block_tkg_nki_kernel_enabled:
        attn_module.attention_block_tokengen_nki_kernel_chunked_attn.assert_called_once()
        return
    
    if seq_len > 1:
        attn_module.rotary_emb.assert_called_once()
        assert attn_module.compute_attention_per_chunk_with_context_parallel.call_count == config['n_chunks']
        assert torch.allclose(past_key_value[0], last_k), "K output is incorrect"
        assert torch.allclose(past_key_value[1], last_v), "V output is incorrect"
        
        if sp_enabled:
            if not optimize_interleave_attn:
                mock_gather_from_sequence_parallel_region.assert_called_once()
                mock_split_along_dim.assert_called_once()
            elif is_pre_global_attn_layer:
                mock_gather_from_sequence_parallel_region.assert_called_once()
                assert mock_gather_from_tensor_model_parallel_region.call_count == config['n_chunks']
                mock_split_along_dim.assert_called_once()
            else:
                mock_gather_from_sequence_parallel_region.assert_called_once()
        
        assert torch.allclose(actual_output, expected_output), 'Attention output is incorrect'

@pytest.mark.parametrize(
    "attn_module, seq_len, sp_enabled, optimize_interleave_attn, is_post_global_attn_layer, is_pre_global_attn_layer",
    # fmt: off
    [
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, False, False, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 1, False, False, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32, "fused_qkv": True, "attn_block_tkg_nki_kernel_enabled": True, "attn_block_tkg_nki_kernel_cache_update": True, "qkv_kernel_enabled": True}, 1, False, False, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, True, False, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 48, True, False, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, True, True, False, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, True, True, True, False),
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, True, True, False, True),
    ],
    indirect=["attn_module"],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_last_kv_chunk")
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_sequence_parallel_region")
@patch("neuronx_distributed_inference.modules.attention.attention_base.split_along_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_tensor_model_parallel_group")
def test_chunked_attn_forward_with_context_parallel(
    mock_get_tensor_model_parallel_group,
    mock_get_context_parallel_attention_cp_group,
    mock_split_along_dim,
    mock_gather_from_sequence_parallel_region,
    mock_get_last_kv_chunk,
    mock_gather_from_tensor_model_parallel_region,
    attn_module,
    seq_len,
    sp_enabled,
    optimize_interleave_attn,
    is_post_global_attn_layer,
    is_pre_global_attn_layer
):
    # Setup test configuration
    config = _setup_chunked_attn_test_config(attn_module, seq_len, sp_enabled, optimize_interleave_attn, is_pre_global_attn_layer, is_post_global_attn_layer)
    _setup_chunked_attn_mocks(mock_get_context_parallel_attention_cp_group, attn_module, config)
    
    # Setup mocks based on sequence length
    if seq_len == 1:
        token_gen_setup = _setup_token_gen_mocks(attn_module, seq_len, config)
        expected_attn_output = token_gen_setup['output']
        past_key_value = token_gen_setup['past_key_value']
        k, v = token_gen_setup['k'], token_gen_setup['v']
    else:
        context_setup = _setup_context_encode_mocks(attn_module, config, is_post_global_attn_layer, is_pre_global_attn_layer, optimize_interleave_attn, sp_enabled)
        expected_attn_output = context_setup['output']
        last_k_chunk, last_v_chunk = context_setup['last_k'], context_setup['last_v']
        past_key_value = None
        
        if sp_enabled:
            sp_expected = _setup_sequence_parallel_mocks(mock_gather_from_sequence_parallel_region, mock_gather_from_tensor_model_parallel_region, mock_split_along_dim, seq_len, config, is_post_global_attn_layer, is_pre_global_attn_layer, optimize_interleave_attn, attn_module)
            if sp_expected is not None:
                expected_attn_output = sp_expected
    
    # Prepare inputs
    input_seq_len = seq_len if not sp_enabled else seq_len // config['tp_degree']
    hidden_states = torch.rand((1, input_seq_len, attn_module.hidden_size))
    attention_mask = torch.full((1, 1, seq_len, seq_len), True)
    position_ids = torch.ones((1, seq_len))
    
    if seq_len > 1:
        mock_get_last_kv_chunk.return_value = (last_k_chunk, last_v_chunk)
    
    # Execute test
    actual_output, past_key_value, cos_cache, sin_cache = attn_module.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
    )
    
    # Verify results
    torch.testing.assert_close(actual_output, expected_attn_output)
    
    if seq_len == 1:
        _verify_chunked_attn_assertions(attn_module, seq_len, config, sp_enabled, optimize_interleave_attn, is_pre_global_attn_layer, mock_gather_from_sequence_parallel_region, mock_gather_from_tensor_model_parallel_region, mock_split_along_dim, past_key_value, k, v, actual_output, expected_attn_output)
    else:
        _verify_chunked_attn_assertions(attn_module, seq_len, config, sp_enabled, optimize_interleave_attn, is_pre_global_attn_layer, mock_gather_from_sequence_parallel_region, mock_gather_from_tensor_model_parallel_region, mock_split_along_dim, past_key_value, last_k_chunk, last_v_chunk, actual_output, expected_attn_output)

@pytest.mark.parametrize("position_ids, chunk_size, update_kv_per_layer, attn_block_tkg_nki_kernel_cache_update", [
    (torch.tensor([20]), 10, False, False),
    (torch.tensor([20, 10]), 10, False, True),
    (torch.tensor([5]), 10, True, True),
])
def test_attention_block_tokengen_nki_kernel_chunked_attn(attn_module, position_ids, chunk_size, update_kv_per_layer, attn_block_tkg_nki_kernel_cache_update):
    attn_module.attn_block_tkg_nki_kernel_cache_update = attn_block_tkg_nki_kernel_cache_update
    attn_module.attention_chunk_size = chunk_size

    kv_mgr = MagicMock()
    kv_mgr._fetch_cache.return_value = None

    attn_module.attention_block_tokengen_nki_kernel = MagicMock(return_value=(None, None, None, None))
    # The None values should be real tensors, but since we don't invoke the kernel in the UT, the inputs are not important
    attn_module.attention_block_tokengen_nki_kernel_chunked_attn(None, None, position_ids, kv_mgr, 
                                                                None, None, None, None, None, update_kv_per_layer,
                                                                idx=0, kvcache_buffer=None)

    attn_module.attention_block_tokengen_nki_kernel.assert_called_once()
    assert all(attn_module.attention_block_tokengen_nki_kernel.call_args.kwargs["position_ids"] == position_ids % chunk_size if attn_block_tkg_nki_kernel_cache_update else position_ids)
    assert attn_module.attention_block_tokengen_nki_kernel.call_args.kwargs["update_kv_per_layer"] == update_kv_per_layer

    if not attn_block_tkg_nki_kernel_cache_update and update_kv_per_layer:
        kv_mgr.update_kv_by_layer_id.assert_called_once()
        assert all(kv_mgr.update_kv_by_layer_id.call_args.kwargs["position_ids"] == position_ids)

@pytest.mark.parametrize(
    "attn_module, seq_len, chunk_idx, is_post_global_attn_layer, is_pre_global_attn_layer, optimize_interleave_attn",
    # fmt: off
    [
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, 0, False, False, False),  # 1st chunk, no interleave attn
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, 1, False, False, False),  # 2nd chunk, no interleave attn
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, 0, True, False, True),  # 1st chunk, interleave attn
        ({"cp_degree": 2, "tp_degree": 4, "attention_chunk_size": 32}, 64, 0, False, True, True),  # 1st chunk, interleave attn
    ],
    indirect=["attn_module"],
    # fmt: on
)
@patch(
    "neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group"
)
@patch(
    "neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim"
)
def test_compute_attention_per_chunk_with_context_parallel(
    mock_gather_from_tensor_model_parallel_region_with_dim,
    mock_get_context_parallel_attention_cp_group,
    attn_module,
    seq_len,
    chunk_idx,
    is_post_global_attn_layer,
    is_pre_global_attn_layer,
    optimize_interleave_attn,
):   
    cp_degree = attn_module.cp_degree
    attn_module.cp_degree = cp_degree
    attn_chunk_size = attn_module.attention_chunk_size
    attn_module.optimize_interleave_attn = optimize_interleave_attn
    attn_module.is_pre_global_attn_layer = is_pre_global_attn_layer
    attn_module.is_post_global_attn_layer = is_post_global_attn_layer

    mock_get_context_parallel_attention_cp_group.return_value = MagicMock()
    mock_get_context_parallel_attention_cp_group.return_value.size().return_value = (
        cp_degree
    )
    rank_util = Mock()
    attn_module.rank_util = rank_util
    rank_util.get_rank.return_value = torch.tensor(0)
    
    # Prepare inputs
    hidden_states = torch.rand((1, seq_len, attn_module.hidden_size))
    attention_mask = torch.full((1, 1, seq_len, seq_len), True)
    position_ids = torch.ones((1, seq_len))
    cos_cache = None
    sin_cache = None
    
    if optimize_interleave_attn and (is_post_global_attn_layer or is_pre_global_attn_layer):
        cp_hidden_states = hidden_states[:, chunk_idx * attn_chunk_size:chunk_idx * attn_chunk_size + attn_chunk_size // cp_degree, :]
    elif is_pre_global_attn_layer:
        cp_hidden_states = hidden_states[:, chunk_idx * attn_chunk_size:(chunk_idx + 1) * attn_chunk_size, :]
   
    # Prepare mocks
    q = torch.rand((1, attn_module.num_heads, attn_chunk_size // cp_degree, attn_module.head_dim))
    k = torch.rand((1, attn_module.num_key_value_heads, attn_chunk_size, attn_module.head_dim))
    v = torch.rand((1, attn_module.num_key_value_heads, attn_chunk_size, attn_module.head_dim))
    
    attn_module.prep_qkv_tensors = MagicMock(return_value=(q, k, v, None, None, None))
    expected_gathered_attn_output = torch.rand((1, attn_chunk_size, attn_module.head_dim))
    attn_output_before_reshape =  torch.rand((1, attn_chunk_size // cp_degree, attn_module.num_heads, attn_module.head_dim))
    expected_ungathered_attn_output = attn_output_before_reshape.reshape(-1, attn_chunk_size // cp_degree, attn_module.head_dim * attn_module.num_heads)
    attn_module.attention_context_encode = MagicMock(return_value=(attn_output_before_reshape, k, v))
    mock_gather_from_tensor_model_parallel_region_with_dim.side_effect = [expected_gathered_attn_output]


    # Call the tested function
    actual_output, actual_k, actual_v, actual_input_hidden_states  = attn_module.compute_attention_per_chunk_with_context_parallel(
        chunk_idx,
        attn_chunk_size,
        hidden_states,
        position_ids,
        attention_mask,
        cos_cache,
        sin_cache
    )

    ## Asserts
    attn_module.prep_qkv_tensors.assert_called_once, "prep_qkv_tensors has to be called once"
    attn_module.attention_context_encode.assert_called_once, "attn context encode has to be called once"
    assert torch.allclose(actual_k, k), "K output is incorrect"
    assert torch.allclose(actual_v, v), "V output is incorrect"
    if optimize_interleave_attn:
        assert torch.allclose(actual_output, expected_ungathered_attn_output), "attention output is incorrect"
        assert torch.allclose(actual_input_hidden_states, cp_hidden_states), "input hidden states is incorrect"
    else:
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_called_once()
        assert actual_input_hidden_states is None, "input hidden states is incorrect"
        assert torch.allclose(actual_output, expected_gathered_attn_output), "attention output is incorrect"


def _get_chunk_lens(seq_len, attn_chunk_size):
    chunk_lens = []
    while seq_len > 0:
        chunk_lens.append(min(attn_chunk_size, seq_len))
        seq_len -= attn_chunk_size
    return chunk_lens


@pytest.mark.parametrize(
    "batch_size, seq_len, sequence_parallel",
    # fmt: off
    [
        (1, 2, False),  # bs=1, context encoding, no sequence parallel
        (1, 2, True),   # bs=1, context encoding, sequence parallel
        (2, 2, False),  # bs=2, context encoding, no sequence parallel
        (2, 2, True),   # bs=2, context encoding, sequence parallel
        (1, 1, False),  # bs=1, token gen, no sequence parallel
        (2, 1, False),  # bs=2, token gen, no sequence parallel
        (1, 4, True),   # bs=2, context encoding, sequence parallel
    ],
    # fmt: on
)
def test_forward_base_case(attn_module, batch_size, seq_len, sequence_parallel):
    seq_len_factor = 1
    if sequence_parallel:
        seq_len_factor = 2
        _enable_sequence_parallel(attn_module, tp_degree=seq_len_factor)

    # Prepare qkv_proj and o_proj mocks.
    q_len = seq_len * seq_len_factor
    q = torch.rand((batch_size, q_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))
    attn_output = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    attn_module.o_proj = MockTorchModule(return_value=attn_output)

    # Prepare inputs.
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    attention_mask = torch.full((batch_size, 1, q_len, q_len), True)
    position_ids = torch.ones((batch_size, seq_len))
    past_key_value = None
    if seq_len == 1:
        past_k = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_v = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_key_value = (past_k, past_v)

    actual_output, past_key_value, cos_cache, sin_cache = attn_module.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
    )

    # Check outputs.
    past_k, past_v = past_key_value
    expected_k = (
        k.view(batch_size, q_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, q_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual_output, attn_output)
    torch.testing.assert_close(past_k, expected_k)
    torch.testing.assert_close(past_v, expected_v)
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    attn_module.qkv_proj.assert_called_once()
    qkv_proj_hidden_states = attn_module.qkv_proj.call_args.kwargs["hidden_states"]
    torch.testing.assert_close(qkv_proj_hidden_states, hidden_states)

    attn_module.o_proj.assert_called_once()
    o_proj_input = attn_module.o_proj.call_args.args[0]
    assert o_proj_input.shape == (batch_size, q_len, attn_module.num_heads * attn_module.head_dim)

@pytest.mark.parametrize(
    "batch_size, seq_len, is_token_gen",
    # fmt: off
    [
        (1, 1, True),  # bs=1, token gen
        (2, 1, True),  # bs=2, token gen
        (2, 5, True),  # bs=2, token gen (SD target)
        (1, 5, False), # bs=1, context encoding
    ],
    # fmt: on
)
def test_forward_attention_tokengen_kernel_builtin(attn_module, batch_size, seq_len, is_token_gen):
    """Test that forward() calls attention_tokengen_kernel_builtin with expected inputs during tokengen when enabled."""
    _enable_attn_tkg_builtin_kernel_enabled(attn_module)

    # Prepare mocks
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))

    attn_module.prep_qkv_tensors = MagicMock(return_value=(q, k, v, None, None, None))
    attn_module.attention_tokengen_kernel_builtin = MagicMock(
        return_value=(
            torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim)),
            k,
        )
    )
    attn_module.attention_context_encode = MagicMock(
        return_value=(
            torch.rand((batch_size, seq_len, attn_module.num_heads, attn_module.head_dim)),
            k,
            v,
        )
    )
    attn_module.o_proj = MockTorchModule(
        return_value=torch.rand((batch_size, seq_len, attn_module.hidden_size))
    )

    # Prepare inputs
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    attention_mask = torch.full((batch_size, 1, seq_len, seq_len), True)
    position_ids = torch.ones((batch_size, seq_len))
    past_key_value = None
    active_mask = None

    if is_token_gen:
        past_k = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_v = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_key_value = (past_k, past_v)

        if seq_len > 1:
            active_mask = torch.ones((batch_size, 1, seq_len, seq_len))

    # Call function
    actual_output, new_past_key_value, cos_cache, sin_cache = attn_module.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        active_mask=active_mask,
        is_for_context_encoding=not is_token_gen,
    )

    if is_token_gen:
        attn_module.attention_tokengen_kernel_builtin.assert_called_once_with(
            q,
            k,
            v,
            position_ids,
            past_key_value,
            attention_mask,
            active_mask,
            position_ids,
        )

        attn_module.attention_context_encode.assert_not_called()
    else:
        attn_module.attention_tokengen_kernel_builtin.assert_not_called()
        attn_module.attention_context_encode.assert_called_once()

    attn_module.o_proj.assert_called_once()
    o_proj_input = attn_module.o_proj.call_args.args[0]
    assert o_proj_input.shape == (batch_size, seq_len, attn_module.num_heads * attn_module.head_dim)


@pytest.mark.parametrize(
    "batch_size, seq_len",
    # fmt: off
    [
        (1, 1),  # bs=1, token gen
        (2, 1),  # bs=2, token gen
        (2, 5),  # bs=2, token gen (SD target)
    ],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch(
    "neuronx_distributed_inference.modules.attention.attention_base._attn_builtin_token_gen_call"
)
def test_attention_tokengen_kernel_builtin(
    mock_attn_builtin_token_gen_call, mock_nc, attn_module, batch_size, seq_len
):
    """Test attention_tokengen_kernel_builtin calls kernel with expected inputs."""
    _enable_attn_tkg_builtin_kernel_enabled(attn_module)

    nc_value = Mock()
    mock_nc.return_value = nc_value
    grid = (nc_value,)
    mock_attn_builtin_token_gen_kernel = MagicMock()
    mock_attn_builtin_token_gen_call.__getitem__.return_value = mock_attn_builtin_token_gen_kernel

    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))

    attention_mask = torch.full((batch_size, 1, seq_len, seq_len), True)
    position_ids = torch.ones((batch_size, seq_len))
    past_k = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    past_v = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    past_key_value = (past_k, past_v)
    active_mask = (
        torch.ones(batch_size, attn_module.num_heads, seq_len, seq_len)
        if seq_len == 1
        else torch.ones((batch_size, 1, seq_len, seq_len))
    )
    rotary_position_ids = position_ids

    # Act
    attn_output, K = attn_module.attention_tokengen_kernel_builtin(
        q, k, v, position_ids, past_key_value, attention_mask, active_mask, rotary_position_ids
    )

    # Assert
    mock_nc.assert_called_once_with(attn_module.logical_nc_config)
    mock_attn_builtin_token_gen_call.__getitem__.assert_called_once_with(grid)


@pytest.mark.parametrize(
    "batch_size, seq_len, is_for_speculation",
    # fmt: off
    [
        (1, 1, False), # bs=1, token gen
        (2, 1, False), # bs=2, token gen
        (2, 5, True),  # bs=2, token gen (speculation)
        (1, 5, False), # bs=1, context encoding
    ],
    # fmt: on
)
def test_forward_kv_cache(attn_module, batch_size, seq_len, is_for_speculation):
    neuron_config = NeuronConfig(
        tp_degree=2,
        torch_dtype=attn_module.torch_dtype,
        batch_size=batch_size,
        seq_len=seq_len,
        is_prefix_caching=False,
        is_chunked_prefill=False,
    )
    config = InferenceConfig(
        neuron_config,
        num_cores_per_group=1,
        num_attention_heads=attn_module.num_attention_heads,
        hidden_size=attn_module.hidden_size,
        num_hidden_layers=1,
    )

    # Prepare qkv_proj and o_proj mocks.
    q_len = seq_len
    q = torch.rand((batch_size, q_len, attn_module.num_heads * attn_module.head_dim))
    k = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    v = torch.rand((batch_size, q_len, attn_module.num_key_value_heads * attn_module.head_dim))
    attn_module.neuron_config = neuron_config
    attn_module.qkv_proj = MockTorchModule(return_value=(q, k, v, None))

    kv_mgr = KVCacheManager(config, num_kv_head=attn_module.num_key_value_heads)
    k_cache = torch.rand(kv_mgr.k_shape)
    v_cache = torch.rand(kv_mgr.v_shape)
    kv_mgr.get_kv_by_layer_id = Mock(return_value=(k_cache, v_cache))
    kv_mgr.update_kv_by_layer_id = Mock(return_value=(k_cache, v_cache))

    attn_output = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    attn_module.o_proj = MockTorchModule(return_value=attn_output)

    # Prepare inputs.
    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))
    attention_mask = torch.full((batch_size, 1, q_len, q_len), True)
    position_ids = torch.ones((batch_size, seq_len)).to(torch.int64)
    seq_ids = torch.arange(0, batch_size, dtype=torch.int32)
    is_for_context_encoding = seq_len > 1 and not is_for_speculation

    active_mask = None
    if is_for_speculation:
        # Speculation case.
        active_mask = _create_attn_mask(batch_size, seq_len)
        attention_mask = _create_attn_mask(batch_size, seq_len)

    if seq_len == 1:
        past_k = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_v = torch.rand(
            (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
        )
        past_key_value = (past_k, past_v)
        kv_mgr.get_kv_by_layer_id = MagicMock(return_value=past_key_value)

    actual_output, updated_kv_cache, cos_cache, sin_cache = attn_module.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        active_mask=active_mask,
        position_ids=position_ids,
        kv_mgr=kv_mgr,
        is_for_speculation=is_for_speculation,
        get_kv_per_layer=not is_for_context_encoding,
        update_kv_per_layer=True,
        seq_ids=seq_ids,
        idx=0,
        seq_len=seq_len,
    )

    torch.testing.assert_close(actual_output, attn_output)
    torch.testing.assert_close(k_cache, updated_kv_cache[0])
    torch.testing.assert_close(v_cache, updated_kv_cache[1])
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    attn_module.qkv_proj.assert_called_once()
    qkv_proj_hidden_states = attn_module.qkv_proj.call_args.kwargs["hidden_states"]
    torch.testing.assert_close(qkv_proj_hidden_states, hidden_states)

    if not is_for_context_encoding:
        assert kv_mgr.get_kv_by_layer_id.call_args.kwargs["idx"] == 0
        assert kv_mgr.get_kv_by_layer_id.call_args.kwargs["seq_len"] == seq_len
        torch.testing.assert_close(kv_mgr.get_kv_by_layer_id.call_args.kwargs["seq_ids"], seq_ids)
    else:
        kv_mgr.get_kv_by_layer_id.assert_not_called()

    expected_k = (
        k.view(batch_size, q_len, attn_module.num_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    expected_v = (
        v.view(batch_size, q_len, attn_module.num_key_value_heads, attn_module.head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    kv_mgr.update_kv_by_layer_id.assert_called_once()
    assert kv_mgr.update_kv_by_layer_id.call_args.kwargs["idx"] == 0
    torch.testing.assert_close(
        kv_mgr.update_kv_by_layer_id.call_args.kwargs["kv_per_layer"][0], expected_k
    )
    torch.testing.assert_close(
        kv_mgr.update_kv_by_layer_id.call_args.kwargs["kv_per_layer"][1], expected_v
    )
    torch.testing.assert_close(
        kv_mgr.update_kv_by_layer_id.call_args.kwargs["position_ids"], position_ids
    )
    torch.testing.assert_close(kv_mgr.update_kv_by_layer_id.call_args.kwargs["seq_ids"], seq_ids)
    assert kv_mgr.update_kv_by_layer_id.call_args.kwargs["seq_len"] == seq_len

    attn_module.o_proj.assert_called_once()
    o_proj_input = attn_module.o_proj.call_args.args[0]
    assert o_proj_input.shape == (batch_size, q_len, attn_module.num_heads * attn_module.head_dim)



@pytest.mark.parametrize(
    "attn_module, batch_size, seq_len",
    # fmt: off
    [
        ({"tp_degree": 2}, 1, 256),   # bs=1, context encoding, seqlen=256
    ], indirect=["attn_module"],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch(
    "neuronx_distributed_inference.modules.attention.attention_base.llama3_nki_attention_block_cte_kernel"
)
@patch(
    "neuronx_distributed_inference.modules.attention.attention_base.reduce_from_tensor_model_parallel_region"
)
def test_nki_attention_block_cte_kernel(
    mock_reduce_from_tensor_model_parallel_region, mock_llama3_nki_attention_block_cte_kernel, mock_nc, attn_module: NeuronAttentionBase, batch_size, seq_len
):
    """Test attention_tokengen_kernel_builtin calls kernel with expected inputs."""

    _enable_nki_attention_block_cte_kernel(attn_module)

    nc_value = Mock()
    mock_nc.return_value = nc_value
    grid = (nc_value,)

    mock_llama3_nki_attention_block_cte_kernel.__getitem__.return_value = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))

    hidden_states = torch.rand((batch_size, seq_len, attn_module.hidden_size))

    cos_cache = torch.rand((batch_size, seq_len, attn_module.head_dim))
    sin_cache = torch.rand((batch_size, seq_len, attn_module.head_dim))
    attn_module.rotary_emb = MagicMock(return_value=(cos_cache, sin_cache))

    rmsnorm = Mock()
    rmsnorm.weight = torch.rand((1, attn_module.hidden_size))

    # assert out_proj kernel is enabled - weight should be transpoed on CPU
    assert attn_module.o_proj.out_proj_kernel_enabled
    attn_module.o_proj = MockTorchModule()
    attn_module.o_proj.o_proj.weight = torch.rand((attn_module.num_heads * attn_module.head_dim, 
                                                    attn_module.hidden_size))
    position_ids = torch.ones((batch_size, seq_len))
    rotary_position_ids = position_ids

    # Act
    attn_module.forward(
        hidden_states, rmsnorm=rmsnorm, rotary_position_ids=rotary_position_ids,
        residual=hidden_states, 
    )


    # Assert
    mock_nc.assert_called_once_with(attn_module.logical_nc_config)
    mock_llama3_nki_attention_block_cte_kernel.__getitem__.assert_called_once_with(grid)


@pytest.mark.parametrize(
    "batch_size, seq_len",
    # fmt: off
    [
        (1, 2),
        (2, 2),
        (1, 4),
    ],
    # fmt: on
)
def test_perform_prefill_no_flash_attn_learned_sinks(attn_module, batch_size, seq_len):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)
    s = torch.rand(attn_module.num_heads)
    attn_module.learned_sinks = torch.nn.Parameter(s).to(xm.xla_device())

    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attention_mask = _create_attn_mask(batch_size, seq_len)

    def test_fn_closure(seq_len, batch_size):
        def test_fn(q, k, v, attention_mask):
            attn_output, _ = attn_module.perform_prefill(
                q, k, v, seq_len, batch_size, attention_mask
            )
            return attn_output
        return test_fn
    
    test_fn = test_fn_closure(seq_len, batch_size)
    example_inputs = [(
        torch.zeros((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool)
    )]
    device_fn = build_function(test_fn, example_inputs)
    device_output = device_fn(q, k, v, attention_mask)

    attn_module.learned_sinks = torch.nn.Parameter(s)
    expected_attn_output, expected_flash_attn_strategy = attn_module.perform_prefill(
        q, k, v, seq_len, batch_size, attention_mask
    )
    assert expected_flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(device_output, expected_attn_output, atol=5e-5, rtol=2e-3)

    attn_module.learned_sinks = None


@pytest.mark.parametrize(
    "batch_size, seq_len, window_size",
    # fmt: off
    [
        (1, 4, 2),
        (2, 8, 4),
        (2, 8, 1),
        (2, 8, 8),
    ],
    # fmt: on
)
def test_perform_prefill_window_attn_no_flash_attn_learned_sinks(attn_module, batch_size, seq_len, window_size):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)
    s = torch.rand(attn_module.num_heads)
    attn_module.learned_sinks = torch.nn.Parameter(s).to(xm.xla_device())

    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    windowed_attn_mask = _create_windowed_attention_mask(batch_size, seq_len, window_size)

    def test_fn_closure(seq_len, batch_size, window_size):
        def test_fn(q, k, v, windowed_attn_mask):
            attn_output, _ = attn_module.perform_prefill_windowed_attn(
                q, k, v, seq_len, batch_size, windowed_attn_mask, window_size
            )
            return attn_output
        return test_fn

    test_fn = test_fn_closure(seq_len, batch_size, window_size)
    example_inputs = [(
        torch.zeros((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool)
    )]
    device_fn = build_function(test_fn, example_inputs)
    device_output = device_fn(q, k, v, windowed_attn_mask)

    attn_module.learned_sinks = torch.nn.Parameter(s)
    expected_output, expected_flash_attn_strategy = attn_module.perform_prefill_windowed_attn(
        q, k, v, seq_len, batch_size, windowed_attn_mask, window_size
    )
    assert expected_flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(device_output, expected_output, atol=5e-5, rtol=2e-3)

    attn_module.learned_sinks = None

@pytest.mark.parametrize(
    "batch_size, seq_len",
    # fmt: off
    [
        # bs=1, basic TKG
        (1, 1),
        # bs=1, speculation
        (1, 2),
        # bs=2, basic TKG
        (2, 1),
        # bs=2, speculation
        (2, 2),
    ],
    # fmt: on
)
def test_compute_for_token_gen_learned_sinks(attn_module, batch_size, seq_len):
    s = torch.rand(attn_module.num_heads)
    attn_module.learned_sinks = torch.nn.Parameter(s).to(xm.xla_device())

    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    position_ids = torch.ones((batch_size, seq_len))
    past_k = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    past_v = torch.rand(
        (batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim)
    )
    attention_mask = _create_attn_mask(batch_size, seq_len)
    active_mask = None
    if seq_len > 1:
        # Speculation case.
        active_mask = _create_attn_mask(batch_size, seq_len).to(xm.xla_device())

    def test_fn_closure(active_mask):
        def test_fn(q, k, v, position_ids, past_k, past_v, attention_mask):
            past_key_value = (past_k, past_v)
            attn_output = attn_module.compute_for_token_gen(
                q, k, v, position_ids, past_key_value, attention_mask, active_mask=active_mask
            )
            return attn_output
        return test_fn

    test_fn = test_fn_closure(active_mask)
    example_inputs = [(
        torch.zeros((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, seq_len), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim), dtype=torch.float32),
        torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool)
    )]
    device_fn = build_function(test_fn, example_inputs)
    device_output = device_fn(q, k, v, position_ids, past_k, past_v, attention_mask)

    attn_module.learned_sinks = torch.nn.Parameter(s)
    if active_mask is not None:
        active_mask = active_mask.cpu()
    past_key_values = (past_k, past_v)
    expected_output = attn_module.compute_for_token_gen(
        q, k, v, position_ids, past_key_values, attention_mask, active_mask=active_mask
    )

    torch.testing.assert_close(device_output, expected_output, atol=5e-5, rtol=2e-3)


def _enable_sequence_parallel(attn_module, tp_degree):
    attn_module.tensor_model_parallel_group = MagicMock()
    attn_module.tensor_model_parallel_group.size.return_value = tp_degree
    attn_module.sequence_parallel_enabled = True
    attn_module.qkv_proj_sp_enabled = True
    attn_module.o_proj_sp_enabled = True

def _enable_sequence_parallel_chunked_attn(attn_module, tp_degree, optimize_interleave_attn, is_pre_global_attn_layer, is_post_global_attn_layer):
    attn_module.tensor_model_parallel_group = MagicMock()
    attn_module.tensor_model_parallel_group.size.return_value = tp_degree
    attn_module.neuron_config.sequence_parallel_enabled = True
    attn_module.sequence_parallel_enabled = True
    attn_module.qkv_proj_sp_enabled = False
    attn_module.o_proj_sp_enabled = False
    attn_module.optimize_interleave_attn = optimize_interleave_attn
    attn_module.is_pre_global_attn_layer = is_pre_global_attn_layer
    attn_module.is_post_global_attn_layer = is_post_global_attn_layer
    attn_module.o_proj_sp_enabled = False if is_pre_global_attn_layer or optimize_interleave_attn else True 



def _enable_attn_tkg_builtin_kernel_enabled(attn_module):
    attn_module.attn_tkg_builtin_kernel_enabled = True
    attn_module.inv_freqs = torch.rand(
        (
            attn_module.head_dim // 2,
            1,
        ),
        dtype=torch.float32,
    )

def _enable_nki_attention_block_cte_kernel(attn_module: NeuronAttentionBase):
    attn_module.attn_block_cte_nki_kernel_enabled = True
    attn_module.k_cache_transposed = True
    attn_module.fused_qkv = True

    attn_module.init_gqa_properties()


def _create_attn_module(
    config,
    hidden_size,
    num_attention_heads,
    num_key_value_heads,
    attention_chunk_size,
    sliding_window,
    **kwargs,
):
    attn_module = NeuronAttentionBase(config=config,
                                      hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      num_key_value_heads=num_key_value_heads,
                                      attention_chunk_size=attention_chunk_size,
                                      sliding_window=sliding_window,
                                      **kwargs)
    
    return attn_module

def _create_attn_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def _create_chunked_attn_mask(batch_size: int, seq_len: int, chunk_size: int) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def _create_windowed_attention_mask(batch_size: int, seq_len: int, window_size: int) -> torch.Tensor:
    """create a causal, window attention mask"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
    for i in range(seq_len):
        if i >= window_size:
            mask[i, : i - window_size + 1] = False
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def _attn(q, k, v, mask) -> torch.Tensor:
    """calculate expected attention output with mask"""
    attn = torch.einsum('bnsd,bntd->bnst', q / math.sqrt(q.shape[-1]), k)
    attn = attn.masked_fill(~mask, float('-inf'))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    attn = torch.einsum('bnts,bnsd->bntd', attn, v)
    return attn
