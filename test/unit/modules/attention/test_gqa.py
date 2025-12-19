from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch

from neuronx_distributed_inference.modules.attention import gqa


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
@patch('neuronx_distributed_inference.modules.attention.gqa._traced_qkv_kernel_nki')
def test_kernel_qkv_forward_rope_fusion(mock_traced_qkv_kernel_nki, batch_size, seq_len, fuse_rope):
    """Test that _traced_qkv_kernel_nki is called with correct arguments when rope fusion is enabled."""
    
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
    
    # Mock _traced_qkv_kernel_nki
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim // tp_degree
    QKV = torch.rand((batch_size, seq_len, fused_qkv_size))
    
    mock_kernel_call = MagicMock(return_value=QKV)
    mock_traced_qkv_kernel_nki.__getitem__ = MagicMock(return_value=mock_kernel_call)
    
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
    mock_traced_qkv_kernel_nki.__getitem__.assert_called_once()
    mock_kernel_call.assert_called_once()
    
    # Check the kernel arguments
    kernel_kwargs = mock_kernel_call.call_args.kwargs
    
    if fuse_rope:
        # When rope fusion is enabled, cos_cache and sin_cache should be passed
        assert "cos_cache" in kernel_kwargs
        assert "sin_cache" in kernel_kwargs
        assert "num_q_heads" in kernel_kwargs
        assert "num_kv_heads" in kernel_kwargs
        torch.testing.assert_close(kernel_kwargs["cos_cache"], cos_cache)
        torch.testing.assert_close(kernel_kwargs["sin_cache"], sin_cache)
        assert kernel_kwargs["num_q_heads"] == num_attention_heads // tp_degree
        assert kernel_kwargs["num_kv_heads"] == num_key_value_heads // tp_degree
    else:
        # When rope fusion is disabled, cos_cache and sin_cache should NOT be passed
        assert "cos_cache" not in kernel_kwargs
        assert "sin_cache" not in kernel_kwargs
        assert "num_q_heads" not in kernel_kwargs
        assert "num_kv_heads" not in kernel_kwargs
    
    # Verify result is a tuple with Q, K, V, residual
    assert len(result) == 4
    Q, K, V, residual = result
    assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim // tp_degree)
    assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert residual is None
