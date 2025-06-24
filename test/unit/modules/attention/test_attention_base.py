from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import KVCacheManager
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    GroupQueryAttention_QKV,
    NeuronAttentionBase,
)

RANDOM_SEED = 0

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(RANDOM_SEED)

class MockTorchModule(MagicMock, torch.nn.Module):
    pass

@pytest.fixture
@patch("neuronx_distributed_inference.modules.attention.attention_base.init_context_parallel_attention_process_groups")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_tp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_tensor_model_parallel_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_tensor_model_parallel_size")
def attn_module(mock_get_tensor_model_parallel_size, 
                mock_get_tensor_model_parallel_group, 
                mock_get_context_parallel_attention_tp_group, 
                mock_init_context_parallel_attention_process_groups, request):

    params = request.param if hasattr(request, "param") else {}

    tp_degree = params.get("tp_degree", 2)
    
    mock_get_tensor_model_parallel_size.return_value = tp_degree
    mock_get_tensor_model_parallel_group.return_value = None

    neuron_config_kwargs = {
        "tp_degree": tp_degree, 
        "logical_nc_config": 1,
        "padding_side": "right",
        "torch_dtype": torch.float32,
    }

    neuron_config_kwargs = {**neuron_config_kwargs, **params}

    neuron_config = NeuronConfig(**neuron_config_kwargs)

    config = InferenceConfig(neuron_config=neuron_config)

    module = _create_attn_module(
                config=config,
                hidden_size=16,
                num_attention_heads=4,
                num_key_value_heads=4,
            )

    # set random seed explictly again since attn_module initialization increments the seed
    torch.manual_seed(RANDOM_SEED)
    
    return module

@pytest.mark.parametrize(
    "attn_kernel_enabled, lnc, cp_degree, q_len, has_attn_mask, expected_flash_attn_strategy",
    # fmt: off
    [
        (None, 2, 4, 128, True, FlashAttentionStrategy.NONE), # LNC2, cp_enabled, q_len < head_dim
        (None, 2, 4, 512, True, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL), # LNC2, cp_enabled, q_len > head_dim
        (None, 2, 1, 128, True, FlashAttentionStrategy.NONE),  # LNC2, q_len < 1024, not divisible by 256
        (None, 2, 1, 3968, True, FlashAttentionStrategy.NONE),  # LNC2, q_len >= 1024, not divisible by 512
        (True, 2, 1, 256, True, FlashAttentionStrategy.SHARDED_KERNEL), # LNC2, q_len < 1024, divisible by 256
        (None, 2, 1, 256, True, FlashAttentionStrategy.SHARDED_KERNEL), # LNC2, q_len < 1024, divisible by 256
        (True, 2, 1, 3968, True, FlashAttentionStrategy.NONE),  # LNC2, q_len >= 1024, not divisible by 512
        (None, 2, 1, 4096, True, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        (True, 2, 1, 4096, True, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        (None, 1, 1, 4096, True, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (True, 1, 1, 4096, True, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (None, 1, 1, 1024, True, FlashAttentionStrategy.NONE),  # LNC1, 512 <= q_len < 4096
        (True, 1, 1, 1024, True, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, enabled, 512 <= q_len < 4096
        (None, 1, 1, 256, True, FlashAttentionStrategy.NONE),  # LNC1, q_len < 512
        (True, 1, 1, 256, True, FlashAttentionStrategy.NONE),  # LNC1, enabled, q_len < 512

        (None, 2, 4, 128, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, cp_enabled, q_len < head_dim
        (None, 2, 4, 512, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, cp_enabled, q_len > head_dim
        (None, 2, 1, 128, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, not divisible by 256
        (None, 2, 1, 3968, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, not divisible by 512
        (True, 2, 1, 256, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, divisible by 256
        (None, 2, 1, 256, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len < 1024, divisible by 256
        (True, 2, 1, 3968, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, not divisible by 512
        (None, 2, 1, 4096, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        (True, 2, 1, 4096, False, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len >= 1024, divisible by 512
        # same as has_attn_mask=True for LNC1
        (None, 1, 1, 4096, False, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (True, 1, 1, 4096, False, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (None, 1, 1, 1024, False, FlashAttentionStrategy.NONE),  # LNC1, 512 <= q_len < 4096
        (True, 1, 1, 1024, False, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, enabled, 512 <= q_len < 4096
        (None, 1, 1, 256, False, FlashAttentionStrategy.NONE),  # LNC1, q_len < 512
        (True, 1, 1, 256, False, FlashAttentionStrategy.NONE),  # LNC1, enabled, q_len < 512

        # turn off kernel explicitly
        (False, 2, 4, 128, True, FlashAttentionStrategy.NONE),
        (False, 2, 4, 512, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 128, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 3968, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 256, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 256, False, FlashAttentionStrategy.NONE),
        (False, 2, 1, 3968, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 4096, True, FlashAttentionStrategy.NONE),
        (False, 2, 1, 4096, False, FlashAttentionStrategy.NONE),
        (False, 1, 1, 4096, True, FlashAttentionStrategy.NONE),
        (False, 1, 1, 4096, False, FlashAttentionStrategy.NONE),
        (False, 1, 1, 1024, True, FlashAttentionStrategy.NONE),
        (False, 1, 1, 1024, False, FlashAttentionStrategy.NONE),
        (False, 1, 1, 256, True, FlashAttentionStrategy.NONE),
        (False, 1, 1, 256, False, FlashAttentionStrategy.NONE),
    ],
    # fmt: on
)
def test_get_flash_attention_strategy(
    attn_module, attn_kernel_enabled, lnc, cp_degree, q_len, has_attn_mask, expected_flash_attn_strategy
):
    attn_module.attn_kernel_enabled = attn_kernel_enabled
    attn_module.logical_nc_config = lnc
    attn_module.head_dim = 256
    attn_module.cp_degree = cp_degree
    flash_attn_strategy = attn_module.get_flash_attention_strategy(q_len, has_attn_mask)
    assert flash_attn_strategy == expected_flash_attn_strategy


@pytest.mark.parametrize(
    "attn_module, batch_size, seq_len, cp_degree, sequence_parallel",
    # fmt: off
    [
        ({}, 1, 8, 1, False),  # bs=1, context encoding, no sequence parallel, no context parallel
        ({}, 2, 1, 1, False),  # bs=2, context encoding, no sequence parallel, no context parallel
        ({}, 1, 8, 1, True),   # bs=1, token gen, sequence parallel, no context parallel
        ({}, 2, 1, 1, True),   # bs=2, token gen, sequence parallel, no context parallel
        ({"cp_degree": 2}, 1, 8, 2, False),   # bs=1, context encoding, no sequence parallel, context parallel
        ({"cp_degree": 2}, 1, 1, 2, True),   # bs=1, token gen, sequence parallel, context parallel
    ], indirect=["attn_module"],
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group")
@patch("neuronx_distributed_inference.modules.attention.attention_base.apply_rotary_pos_emb")
def test_prep_qkv_tensors_base_case(
    mock_apply_rotary_pos_emb, mock_get_context_parallel_attention_cp_group, 
    mock_gather_from_tensor_model_parallel_region_with_dim, attn_module, batch_size,
    cp_degree, seq_len, sequence_parallel
):
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
    "batch_size, seq_len, expected_attn_output",
    # fmt: off
    [   # expected_attn_output shape (bs, n_heads, s, h_dim)
        
        (1, 4, torch.tensor([[[[0.7745, 0.4369, 0.5191, 0.6159],  [0.7981, 0.7966, 0.2513, 0.4178], [0.6965, 0.9143, 0.9351, 0.9412], [0.6425, 0.4419, 0.7186, 0.5217]],
                           [[0.0340, 0.9442, 0.8802, 0.0012], [0.2914, 0.7012, 0.6675, 0.1254], [0.6923, 0.2038, 0.6833, 0.7529], [0.7883, 0.4838, 0.2903, 0.4184]]]])),
        # test case for when seq_len % chunk_size != 0
        (1, 3, torch.tensor([[[[0.2783, 0.4820, 0.8198, 0.9971],  [0.5041, 0.5280, 0.8281, 0.5717], [0.5932, 0.1123, 0.1535, 0.2417]],
                           [[0.7262, 0.7011, 0.2038, 0.6511], [0.7471, 0.5870, 0.3399, 0.6359], [0.8102, 0.9801, 0.1147, 0.3168]]]])),
        (2, 4, torch.tensor([[[[0.3251, 0.0902, 0.3936, 0.6069], [0.2518, 0.2769, 0.6193, 0.5299], [0.5139, 0.4569, 0.6012, 0.8179], [0.7067, 0.6081, 0.7578, 0.6694]],
                           [[0.0508, 0.2630, 0.8405, 0.4968], [0.1499, 0.1908, 0.4413, 0.2900], [0.3986, 0.7742, 0.7703, 0.0178], [0.5778, 0.4856, 0.6072, 0.1390]]],
                           [[[0.4037, 0.4018, 0.0513, 0.0683], [0.4139, 0.4607, 0.1760, 0.4174], [0.0500, 0.4663, 0.9397, 0.2961], [0.5858, 0.5939, 0.4102, 0.6053]],
                           [[0.4423, 0.2768, 0.8998, 0.0960], [0.4947, 0.3326, 0.8797, 0.3519], [0.7403, 0.6766, 0.3798, 0.3948], [0.4946, 0.7121, 0.5746, 0.5633]]]])),
        
    ],
    # fmt: on
)
def test_perform_prefill_chunked_attn_no_flash_attn(attn_module, batch_size, seq_len, expected_attn_output):
    chunk_size = 2
    chunked_attn_mask  = _create_chunked_attn_mask(batch_size, seq_len, chunk_size)
    attn_module.get_flash_attention_strategy = MagicMock(return_value=FlashAttentionStrategy.NONE)
    q = torch.rand((batch_size, attn_module.num_heads, seq_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attn_output, flash_attn_strategy = attn_module.perform_prefill_chunked_attn(
        q, k, v, seq_len, batch_size, chunked_attn_mask, chunk_size
    )
    assert flash_attn_strategy == FlashAttentionStrategy.NONE
    torch.testing.assert_close(attn_output, expected_attn_output, atol=5e-5, rtol=2e-3)


@pytest.mark.parametrize("batch_size", [1, 2])
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call")
def test_perform_prefill_unsharded_flash_attn(mock_flash_fwd_call, attn_module, batch_size):
    seq_len = 2  # Use a seq_len > 1 for prefill.
    _setup_perform_prefill_flash_attn_test(
        attn_module, batch_size, seq_len, seq_len, FlashAttentionStrategy.UNSHARDED_KERNEL
    )
    _check_flash_attn_kernel_call(mock_flash_fwd_call, attn_module, batch_size, seq_len)


@pytest.mark.parametrize("batch_size", [1, 2])
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call")
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

@pytest.mark.parametrize("batch_size", [1, 2])
@patch("neuronx_distributed_inference.modules.attention.attention_base.nc")
@patch("neuronx_distributed_inference.modules.attention.attention_base._flash_fwd_call")
def test_perform_prefill_context_parallel_attn_kernel(mock_flash_fwd_call, mock_nc, 
                                                      attn_module, batch_size):
    q_len = 16
    seq_len = 64
    attn_module.cp_degree = seq_len // q_len

    # Mock accessing grid index.
    nc_value = Mock()
    mock_nc.return_value = nc_value
    grid = (nc_value,)
    mock_flash_fwd_kernel = MagicMock()
    mock_flash_fwd_call.__getitem__.return_value = mock_flash_fwd_kernel

    global_rank = Mock()
    attn_module.global_rank = global_rank
    global_rank.get_rank.return_value = torch.tensor(0)

    _setup_perform_prefill_flash_attn_test(
        attn_module, batch_size, q_len, seq_len, FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL
    )

    mock_nc.assert_called_once_with(attn_module.logical_nc_config)
    mock_flash_fwd_call.__getitem__.assert_called_once_with(grid)

    _check_context_parallel_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, q_len, seq_len)

def _setup_perform_prefill_flash_attn_test(attn_module, batch_size, q_len, seq_len, flash_attn_strategy):
    attn_module.get_flash_attention_strategy = MagicMock(return_value=flash_attn_strategy)

    q = torch.rand((batch_size, attn_module.num_heads, q_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attention_mask = _create_attn_mask(batch_size, seq_len)

    attn_output, flash_attn_strategy = attn_module.perform_prefill(
        q, k, v, q_len, batch_size, attention_mask
    )
    assert attn_output.shape == (batch_size, attn_module.num_heads, attn_module.head_dim, q_len)
    assert flash_attn_strategy == flash_attn_strategy


def _check_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, seq_len):
    mock_flash_fwd_kernel.assert_called_once()
    flash_fwd_args, flash_fwd_kwargs = mock_flash_fwd_kernel.call_args
    assert len(flash_fwd_args) == 5

    q_flash_fwd_arg = flash_fwd_args[0]
    k_flash_fwd_arg = flash_fwd_args[1]
    v_flash_fwd_arg = flash_fwd_args[2]
    assert q_flash_fwd_arg.shape == (
        batch_size * attn_module.num_heads,
        attn_module.head_dim,
        seq_len,
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
        seq_len,
    )
    assert flash_fwd_args[3] == 1.0
    assert flash_fwd_kwargs["kernel_name"] == "CausalAttentionMMSoftmaxMMWithoutSwap"

def _check_context_parallel_flash_attn_kernel_call(mock_flash_fwd_kernel, attn_module, batch_size, q_len, seq_len):
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
    assert flash_fwd_kwargs["tile_i"] == 0 # we hardcode to global rank 0 for unit testing

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
        ({"cp_degree": 4, "tp_degree": 8}, 1, 64, False),  # bs=1, context encoding, no sequence parallel
        ({"cp_degree": 4, "tp_degree": 8}, 2, 64, False),  # bs=2, context encoding, no sequence parallel
        ({"cp_degree": 4, "tp_degree": 8}, 1, 1, False),   # bs=1, token generation, no sequence parallel
        ({"cp_degree": 4, "tp_degree": 8}, 1, 64, True),   # bs=1, context encoding, sequence parallel
        ({"cp_degree": 4, "tp_degree": 8}, 2, 64, True),   # bs=2, context encoding, sequence parallel
    ], indirect=["attn_module"]
    # fmt: on
)
@patch("neuronx_distributed_inference.modules.attention.attention_base.gather_from_tensor_model_parallel_region_with_dim")
@patch("neuronx_distributed_inference.modules.attention.attention_base.get_context_parallel_attention_cp_group")
def test_forward_context_parallel(mock_get_context_parallel_attention_cp_group, 
                                  mock_gather_from_tensor_model_parallel_region_with_dim, 
                                  attn_module, batch_size, seq_len, sp_enabled):

    attn_module.neuron_config.is_prefill_stage = seq_len > 1
    
    cp_degree = attn_module.cp_degree
    tp_degree = attn_module.tp_degree
    
    if sp_enabled:
        _enable_sequence_parallel(attn_module, tp_degree // cp_degree)
        
    attn_module.cp_degree = cp_degree

    global_rank = Mock()
    attn_module.global_rank = global_rank
    global_rank.get_rank.return_value = torch.tensor(0)

    q_len = seq_len // cp_degree if seq_len > 1 else seq_len

    # Prepare prep_qkv_tensors and o_proj mocks.
    q = torch.rand((batch_size, attn_module.num_heads, q_len, attn_module.head_dim))
    k = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    v = torch.rand((batch_size, attn_module.num_key_value_heads, seq_len, attn_module.head_dim))
    attn_module.prep_qkv_tensors = MagicMock(return_value=(q, k, v, None, None, None))

    output_seq_len = seq_len if not sp_enabled else seq_len // cp_degree

    o_proj_output = torch.rand((batch_size, attn_module.num_attention_heads * attn_module.head_dim, attn_module.hidden_size))
    attn_output = torch.rand((batch_size, output_seq_len, attn_module.hidden_size))

    # o_proj output is defined based on if we do the final attention gather or not
    # In TKG or SP enabled, we don't gather so o_proj output is the final output
    o_proj_output = o_proj_output if seq_len > 1 and not sp_enabled else attn_output

    if seq_len == 1:
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

    torch.testing.assert_close(actual_output, attn_output)
    assert cos_cache is None
    assert sin_cache is None

    # Check mocks.
    attn_module.prep_qkv_tensors.assert_called_once()
    o_proj.assert_called_once()

    if seq_len > 1 and not sp_enabled:
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_called_once()
        mock_get_context_parallel_attention_cp_group.assert_called_once()
    else:
        mock_gather_from_tensor_model_parallel_region_with_dim.assert_not_called()
        mock_get_context_parallel_attention_cp_group.assert_not_called()

    o_proj_input = o_proj.call_args.args[0]
    assert o_proj_input.shape == (batch_size, q_len, attn_module.num_heads * attn_module.head_dim)


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
    "batch_size, seq_len, head_dim, expected_padded_len",
    [
        (1, 577, 32, 578),  # Odd sequence length gets padded
        (2, 578, 32, 578),  # Even sequence length stays the same
        (1, 2, 32, 2),  # small odd length shouldn't be padded
        (1, 3, 32, 3),  # small even length shouldn't be padded
    ],
)
@patch("neuronx_distributed_inference.modules.attention.gqa._traced_qkv_kernel")
def test_kernel_qkv_forward_padding(
    mock_kernel, batch_size, seq_len, head_dim, expected_padded_len
):
    mock_self = MagicMock()
    mock_self.logical_nc_config = 2
    mock_self.Wqkv = MagicMock()
    mock_self.Wqkv.weight = torch.randn(head_dim, 3 * head_dim)
    mock_self.bias = False
    mock_self.rms_norm_eps = 1e-6
    mock_self.num_attention_heads = 16
    mock_self.num_key_value_heads = 16
    mock_self.tp_degree = 16
    mock_self.head_dim = head_dim
    mock_self.seq_len_threshold_for_cc_tiling = 16384

    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, head_dim)

    # Call the method
    GroupQueryAttention_QKV._kernel_qkv_forward(
        mock_self, hidden_states=hidden_states, rmsnorm=None, residual=None
    )

    # Verify kernel was called
    mock_kernel.__getitem__.assert_called_once()
    mock_kernel.__getitem__.return_value.assert_called_once()

    # Get the actual hidden states passed to kernel
    kernel_args = mock_kernel.__getitem__.return_value.call_args.args
    padded_hidden_states = kernel_args[0]

    # Verify padding
    assert padded_hidden_states.shape[1] == expected_padded_len


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


def _enable_sequence_parallel(attn_module, tp_degree):
    attn_module.tensor_model_parallel_group = MagicMock()
    attn_module.tensor_model_parallel_group.size.return_value = tp_degree
    attn_module.sequence_parallel_enabled = True


def _enable_attn_tkg_builtin_kernel_enabled(attn_module):
    attn_module.attn_tkg_builtin_kernel_enabled = True
    attn_module.inv_freqs = torch.rand(
        (
            attn_module.head_dim // 2,
            1,
        ),
        dtype=torch.float32,
    )


def _create_attn_module(
    config,
    hidden_size,
    num_attention_heads,
    num_key_value_heads,
):
    attn_module = NeuronAttentionBase(config=config,
                                      hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      num_key_value_heads=num_key_value_heads)
    
    return attn_module

def _create_attn_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask

def _create_chunked_attn_mask(batch_size, seq_len: int, chunk_size: int) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask
