import sys
import types
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch

_lora_pkg = types.ModuleType("neuronx_distributed_inference.modules.lora_serving")
_lora_pkg.__path__ = []
_lora_module = types.ModuleType(
    "neuronx_distributed_inference.modules.lora_serving.lora_module"
)
_lora_module.is_lora_module = lambda _module: False
sys.modules.setdefault(
    "neuronx_distributed_inference.modules.lora_serving",
    _lora_pkg,
)
sys.modules.setdefault(
    "neuronx_distributed_inference.modules.lora_serving.lora_module",
    _lora_module,
)

from neuronx_distributed_inference.modules.attention import gqa


def test_preshard_hook_preserves_qwen_qkv_gate_packed_weight():
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 4
    tp_degree = 4
    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim

    qkv_proj = gqa.GroupQueryAttention_QKV(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        tp_degree=tp_degree,
        fused_qkv=True,
        gather_output=False,
        bias=False,
    )
    qkv_proj.qwen_qkv_gate_packed = True

    q_weight = torch.arange(q_width * hidden_size, dtype=torch.float32).reshape(
        q_width, hidden_size
    )
    gate_weight = q_weight + 10_000
    k_weight = q_weight[:kv_width] + 20_000
    v_weight = q_weight[:kv_width] + 30_000
    packed_weight = torch.cat([q_weight, gate_weight, k_weight, v_weight], dim=0)
    state_dict = {"layers.0.self_attn.Wqkv.weight": packed_weight.clone()}

    qkv_proj.preshard_hook(state_dict, "layers.0.self_attn.weight")

    heads_per_rank = num_attention_heads // tp_degree
    q_heads = q_weight.reshape(num_attention_heads, head_dim, hidden_size)
    gate_heads = gate_weight.reshape(num_attention_heads, head_dim, hidden_size)
    q_gate_rank_blocks = []
    for rank in range(tp_degree):
        start = rank * heads_per_rank
        q_gate_rank_blocks.append(q_heads[start : start + heads_per_rank])
        q_gate_rank_blocks.append(gate_heads[start : start + heads_per_rank])
    expected_q_gate = torch.cat(q_gate_rank_blocks, dim=0).reshape(
        2 * q_width,
        hidden_size,
    )
    expected = torch.cat([expected_q_gate, k_weight, v_weight], dim=0)

    torch.testing.assert_close(state_dict["layers.0.self_attn.Wqkv.weight"], expected)
    assert getattr(qkv_proj.Wqkv.weight, "fused_qkv") is True
    assert getattr(qkv_proj.Wqkv.weight, "num_attention_heads") == num_attention_heads * 2
    assert getattr(qkv_proj.Wqkv.weight, "num_key_value_heads") == num_key_value_heads
    assert getattr(qkv_proj.Wqkv.weight, "head_dim") == head_dim


@pytest.mark.parametrize(
    "batch_size, seq_len, fuse_rope",
    # fmt: off
    [
        (1, 8, True),    # bs=1, context encoding, fuse rope enabled
        (2, 8, True),    # bs=2, context encoding, fuse rope enabled
        (1, 8, False),   # bs=1, context encoding, fuse rope disabled
        (2, 8, False),   # bs=2, context encoding, fuse rope disabled
        (1, 1, False),   # bs=1, token gen, fuse rope disabled
        (2, 1, False),   # bs=2, token gen, fuse rope disabled
    ],
    # fmt: on
)
@patch('neuronx_distributed_inference.modules.attention.gqa.qkv_kernel')
def test_kernel_qkv_forward_rope_fusion(mock_qkv_kernel, batch_size, seq_len, fuse_rope):
    """Test that qkv_kernel is called with correct arguments when rope fusion is enabled."""
    
    # Test parameters
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    tp_degree = 2
    
    # Prepare inputs
    hidden_states = torch.rand((batch_size, seq_len, hidden_size))
    cos_cache = torch.rand((batch_size, seq_len, head_dim)) if fuse_rope else None
    sin_cache = torch.rand((batch_size, seq_len, head_dim)) if fuse_rope else None
    
    # Mock qkv kernel
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim // tp_degree
    QKV = torch.rand((batch_size, seq_len, fused_qkv_size))
    
    mock_kernel_call = MagicMock(return_value=QKV)
    mock_qkv_kernel.__getitem__ = MagicMock(return_value=mock_kernel_call)
    
    # Create a mock GroupQueryAttention_QKV instance
    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = num_attention_heads
    qkv_proj.num_key_value_heads = num_key_value_heads
    qkv_proj.tp_degree = tp_degree
    qkv_proj.head_dim = head_dim
    qkv_proj.fused_rmsnorm = False
    qkv_proj.fused_rmsnorm_skip_gamma = False
    qkv_proj.logical_nc_config = 1
    qkv_proj.bias = False
    qkv_proj.seq_len_threshold_for_cc_tiling = 16834
    qkv_proj.tiling_factor = 1
    qkv_proj.qkv_kernel_nbsd_layout = False
    qkv_proj.qkv_nki_kernel_enabled = True
    qkv_proj.rms_norm_eps = 1e-6
    
    # Create a mock weight with correct shape (transposed for qkv_nki_kernel_enabled=True)
    qkv_proj.Wqkv = Mock()
    qkv_proj.Wqkv.weight = Mock()
    qkv_proj.Wqkv.weight.shape = (hidden_size, fused_qkv_size)
    qkv_proj.Wqkv.weight.dtype = torch.float32
    qkv_proj.Wqkv.bias = None
    qkv_proj.Wqkv.scale = None
    qkv_proj.Wqkv.input_scale = None
    
    # Mock _split_fused_qkv to return Q, K, V
    Q = torch.rand((batch_size, seq_len, num_attention_heads * head_dim // tp_degree))
    K = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    V = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    qkv_proj._split_fused_qkv = Mock(return_value=(Q, K, V))
    
    # Call the real _kernel_qkv_forward method with our mock instance
    result = gqa.GroupQueryAttention_QKV._kernel_qkv_forward(
        qkv_proj, hidden_states, None, None, cos_cache, sin_cache
    )
    
    # Verify the kernel was called
    mock_qkv_kernel.__getitem__.assert_called_once_with(qkv_proj.logical_nc_config)
    mock_kernel_call.assert_called_once()
    
    # Check the kernel arguments
    kernel_kwargs = mock_kernel_call.call_args.kwargs
    
    if fuse_rope:
        # When rope fusion is enabled, cos_cache and sin_cache should be passed
        torch.testing.assert_close(kernel_kwargs["cos_cache"], cos_cache)
        torch.testing.assert_close(kernel_kwargs["sin_cache"], sin_cache)
        assert kernel_kwargs["num_q_heads"] == num_attention_heads // tp_degree
        assert kernel_kwargs["num_kv_heads"] == num_key_value_heads // tp_degree
    else:
        # When rope fusion is disabled, cos_cache and sin_cache should NOT be passed
        assert kernel_kwargs["cos_cache"] is None
        assert kernel_kwargs["sin_cache"] is None

    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.NONE
    assert kernel_kwargs["qkv_w_scale"] is None
    assert kernel_kwargs["qkv_in_scale"] is None
    
    # Verify result is a tuple with Q, K, V, residual
    assert len(result) == 4
    Q, K, V, residual = result
    assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim // tp_degree)
    assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert residual is None


@patch('neuronx_distributed_inference.modules.attention.gqa.qkv_kernel')
def test_kernel_qkv_forward_passes_fp8_weight_scale(mock_qkv_kernel):
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    tp_degree = 2
    batch_size = 1
    seq_len = 8
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim // tp_degree

    hidden_states = torch.rand((batch_size, seq_len, hidden_size))
    QKV = torch.rand((batch_size, seq_len, fused_qkv_size))
    mock_kernel_call = MagicMock(return_value=QKV)
    mock_qkv_kernel.__getitem__ = MagicMock(return_value=mock_kernel_call)

    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = num_attention_heads
    qkv_proj.num_key_value_heads = num_key_value_heads
    qkv_proj.tp_degree = tp_degree
    qkv_proj.head_dim = head_dim
    qkv_proj.fused_rmsnorm = False
    qkv_proj.fused_rmsnorm_skip_gamma = False
    qkv_proj.logical_nc_config = 1
    qkv_proj.bias = False
    qkv_proj.qkv_kernel_nbsd_layout = False
    qkv_proj.rms_norm_eps = 1e-6

    qkv_proj.Wqkv = Mock()
    qkv_proj.Wqkv.weight = Mock()
    qkv_proj.Wqkv.weight.data = torch.rand((hidden_size, fused_qkv_size))
    qkv_scale = torch.rand((128, fused_qkv_size), dtype=torch.float32)
    qkv_proj.Wqkv.scale = Mock()
    qkv_proj.Wqkv.scale.data = qkv_scale
    qkv_proj.Wqkv.input_scale = None
    qkv_proj.Wqkv.bias = None

    Q = torch.rand((batch_size, seq_len, num_attention_heads * head_dim // tp_degree))
    K = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    V = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    qkv_proj._split_fused_qkv = Mock(return_value=(Q, K, V))

    result = gqa.GroupQueryAttention_QKV._kernel_qkv_forward(
        qkv_proj, hidden_states, None, None, None, None
    )

    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.ROW
    torch.testing.assert_close(kernel_kwargs["qkv_w_scale"], qkv_scale)
    assert kernel_kwargs["qkv_in_scale"] is None
    assert len(result) == 4


@patch("neuronx_distributed_inference.modules.attention.gqa.reduce_from_tensor_model_parallel_region")
@patch("neuronx_distributed_inference.modules.attention.gqa.output_projection_cte")
def test_kernel_o_proj_uses_bnds_layout_without_quantization(
    mock_output_projection_cte,
    mock_reduce_from_tensor_model_parallel_region,
):
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    tp_degree = 2
    batch_size = 1
    seq_len = 8
    heads_per_core = num_attention_heads // tp_degree
    nd = heads_per_core * head_dim

    attention_output = torch.rand((batch_size, seq_len, nd))
    kernel_out = torch.rand((batch_size, seq_len, hidden_size))
    mock_kernel_call = MagicMock(return_value=kernel_out)
    mock_output_projection_cte.__getitem__ = MagicMock(return_value=mock_kernel_call)
    mock_reduce_from_tensor_model_parallel_region.side_effect = lambda x, process_group=None: x

    o_proj = Mock(spec=gqa.GroupQueryAttention_O)
    o_proj.num_attention_heads = num_attention_heads
    o_proj.tp_degree = tp_degree
    o_proj.head_dim = head_dim
    o_proj.logical_nc_config = 1
    o_proj.bias = False
    o_proj.quantized = False
    o_proj.rpl_reduce_dtype = torch.float32
    o_proj.sequence_parallel_enabled = False
    o_proj.tensor_model_parallel_group = None
    o_proj.o_proj = Mock()
    o_proj.o_proj.weight = Mock()
    o_proj.o_proj.weight.shape = (nd, hidden_size)
    o_proj.o_proj.weight.dtype = torch.float32
    o_proj.o_proj.weight.data = torch.rand((nd, hidden_size))
    o_proj.o_proj.bias = None
    o_proj.o_proj.scale = None

    result = gqa.GroupQueryAttention_O._kernel_o_proj(o_proj, attention_output)

    mock_output_projection_cte.__getitem__.assert_called_once_with(o_proj.logical_nc_config)
    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["attention"].shape == (batch_size, heads_per_core, head_dim, seq_len)
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.NONE
    assert kernel_kwargs["weight_scales"] is None
    torch.testing.assert_close(result, kernel_out.to(torch.float32))


@patch("neuronx_distributed_inference.modules.attention.gqa.reduce_from_tensor_model_parallel_region")
@patch("neuronx_distributed_inference.modules.attention.gqa.output_projection_cte")
def test_kernel_o_proj_passes_fp8_row_weight_scales(
    mock_output_projection_cte,
    mock_reduce_from_tensor_model_parallel_region,
):
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    tp_degree = 2
    batch_size = 1
    seq_len = 8
    heads_per_core = num_attention_heads // tp_degree
    nd = heads_per_core * head_dim

    attention_output = torch.rand((batch_size, seq_len, nd))
    kernel_out = torch.rand((batch_size, seq_len, hidden_size))
    mock_kernel_call = MagicMock(return_value=kernel_out)
    mock_output_projection_cte.__getitem__ = MagicMock(return_value=mock_kernel_call)
    mock_reduce_from_tensor_model_parallel_region.side_effect = lambda x, process_group=None: x

    o_proj = Mock(spec=gqa.GroupQueryAttention_O)
    o_proj.num_attention_heads = num_attention_heads
    o_proj.tp_degree = tp_degree
    o_proj.head_dim = head_dim
    o_proj.logical_nc_config = 1
    o_proj.bias = False
    o_proj.quantized = True
    o_proj.rpl_reduce_dtype = torch.float32
    o_proj.sequence_parallel_enabled = False
    o_proj.tensor_model_parallel_group = None
    o_proj.o_proj = Mock()
    o_proj.o_proj.weight = Mock()
    o_proj.o_proj.weight.shape = (nd, hidden_size)
    o_proj.o_proj.weight.dtype = torch.float8_e4m3fn
    o_proj.o_proj.weight.data = torch.rand((nd, hidden_size)).to(torch.float8_e4m3fn)
    o_proj.o_proj.bias = None
    weight_scales = torch.rand((128, hidden_size), dtype=torch.float32)
    o_proj.o_proj.scale = Mock()
    o_proj.o_proj.scale.data = weight_scales

    result = gqa.GroupQueryAttention_O._kernel_o_proj(o_proj, attention_output)

    mock_output_projection_cte.__getitem__.assert_called_once_with(o_proj.logical_nc_config)
    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["attention"].shape == (batch_size, seq_len, heads_per_core, head_dim)
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.ROW
    torch.testing.assert_close(kernel_kwargs["weight_scales"], weight_scales)
    torch.testing.assert_close(result, kernel_out.to(torch.float32))


@patch("neuronx_distributed_inference.modules.attention.gqa.reduce_from_tensor_model_parallel_region")
@patch("neuronx_distributed_inference.modules.attention.gqa.output_projection_cte")
def test_kernel_o_proj_folds_large_head_dim_for_fp8_row_weight_scales(
    mock_output_projection_cte,
    mock_reduce_from_tensor_model_parallel_region,
):
    hidden_size = 16
    head_dim = 256
    num_attention_heads = 8
    tp_degree = 2
    batch_size = 1
    seq_len = 8
    heads_per_core = num_attention_heads // tp_degree
    nd = heads_per_core * head_dim

    attention_output = torch.rand((batch_size, seq_len, nd))
    kernel_out = torch.rand((batch_size, seq_len, hidden_size))
    mock_kernel_call = MagicMock(return_value=kernel_out)
    mock_output_projection_cte.__getitem__ = MagicMock(return_value=mock_kernel_call)
    mock_reduce_from_tensor_model_parallel_region.side_effect = lambda x, process_group=None: x

    o_proj = Mock(spec=gqa.GroupQueryAttention_O)
    o_proj.num_attention_heads = num_attention_heads
    o_proj.tp_degree = tp_degree
    o_proj.head_dim = head_dim
    o_proj.logical_nc_config = 1
    o_proj.bias = False
    o_proj.quantized = True
    o_proj.rpl_reduce_dtype = torch.float32
    o_proj.sequence_parallel_enabled = False
    o_proj.tensor_model_parallel_group = None
    o_proj.o_proj = Mock()
    o_proj.o_proj.weight = Mock()
    o_proj.o_proj.weight.shape = (nd, hidden_size)
    o_proj.o_proj.weight.dtype = torch.float8_e4m3fn
    o_proj.o_proj.weight.data = torch.rand((nd, hidden_size)).to(torch.float8_e4m3fn)
    o_proj.o_proj.bias = None
    weight_scales = torch.rand((128, hidden_size), dtype=torch.float32)
    o_proj.o_proj.scale = Mock()
    o_proj.o_proj.scale.data = weight_scales

    result = gqa.GroupQueryAttention_O._kernel_o_proj(o_proj, attention_output)

    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["attention"].shape == (
        batch_size,
        seq_len,
        heads_per_core * 2,
        head_dim // 2,
    )
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.ROW
    torch.testing.assert_close(kernel_kwargs["weight_scales"], weight_scales)
    torch.testing.assert_close(result, kernel_out.to(torch.float32))
