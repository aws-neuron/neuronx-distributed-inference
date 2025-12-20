import os
import pytest
from unittest.mock import patch, MagicMock
import torch
from neuronx_distributed_inference.experimental.functional.qkv.qkv import qkv_kernel
from torch_xla.core import xla_model as xm
from torch_neuronx.testing.validation import neuron_allclose

os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
torch.manual_seed(0)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_attention_heads,num_key_value_heads,head_dim,dtype", [
    (2, 4, 128, 8, 8, 16, torch.bfloat16),
    (1, 8, 256, 16, 8, 16, torch.bfloat16),
    (4, 16, 512, 32, 16, 16, torch.bfloat16),
])
def test_qkv_kernel_cpu(batch_size, seq_len, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, dtype):
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    weight = torch.randn(hidden_dim, fused_qkv_size, dtype=dtype)
    bias = torch.randn(fused_qkv_size, dtype=dtype)

    with patch("neuronx_distributed_inference.experimental.functional.qkv.qkv.nki_jit") as mock_nki_jit:
        Q, K, V = qkv_kernel(
            hidden_states=hidden_states,
            w_qkv=weight,
            bias=bias,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        mock_nki_jit.assert_called_once()
        assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim)
        assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
        assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_attention_heads,num_key_value_heads,head_dim,dtype", [
    (2, 512, 1024, 2, 2, 64, torch.float16),
    (2, 512, 1024, 1, 1, 64, torch.float16),
])
def test_qkv_kernel_without_rmsnorm(batch_size, seq_len, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, dtype):
    # CPU reference implementation
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    hidden_states_cpu = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    weight_cpu = torch.randn(fused_qkv_size, hidden_dim, dtype=dtype).T

    # Reference QKV computation - reshape for matrix multiplication
    hidden_reshaped = hidden_states_cpu.view(-1, hidden_dim)  # [batch*seq, hidden]
    expected_qkv = torch.matmul(hidden_reshaped, weight_cpu)  # [batch*seq, fused_qkv_size]
    expected_qkv = expected_qkv.view(batch_size, seq_len, fused_qkv_size)  # [batch, seq, fused_qkv_size]
    q_end = num_attention_heads * head_dim
    k_end = q_end + num_key_value_heads * head_dim
    expected_Q = expected_qkv[:, :, :q_end]
    expected_K = expected_qkv[:, :, q_end:k_end]
    expected_V = expected_qkv[:, :, k_end:]

    # On-device kernel test
    device = xm.xla_device()
    hidden_states = hidden_states_cpu.to(device)
    weight = weight_cpu.to(device)

    Q, K, V = qkv_kernel(
        hidden_states=hidden_states,
        w_qkv=weight,
        rmsnorm=None,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )

    assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim)
    assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
    assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim)

    # Compare the shapes
    assert Q.shape == expected_Q.shape
    assert K.shape == expected_K.shape
    assert V.shape == expected_V.shape

    assert torch.allclose(Q.cpu(), expected_Q, atol=1e-3, rtol=1e-3)
    assert torch.allclose(K.cpu(), expected_K, atol=1e-3, rtol=1e-3)
    assert torch.allclose(V.cpu(), expected_V, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_attention_heads,num_key_value_heads,head_dim,dtype", [
    (1, 128, 512, 2, 2, 64, torch.float16),
    (2, 512, 1024, 1, 1, 64, torch.float16),
    (4, 512, 1024, 2, 2, 64, torch.float16),
])
def test_qkv_kernel_with_rmsnorm(batch_size, seq_len, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, dtype):
    # CPU reference implementation
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    hidden_states_cpu = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    weight_cpu = torch.randn(fused_qkv_size, hidden_dim, dtype=dtype).T
    rmsnorm_weight_cpu = torch.randn(hidden_dim, dtype=dtype)

    # Reference RMSNorm + QKV computation on CPU
    eps = 1e-6
    variance = hidden_states_cpu.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states_cpu * torch.rsqrt(variance + eps)
    normed_hidden = normalized * rmsnorm_weight_cpu
    hidden_reshaped = normed_hidden.view(-1, hidden_dim)
    expected_qkv = torch.matmul(hidden_reshaped, weight_cpu)
    expected_qkv = expected_qkv.view(batch_size, seq_len, fused_qkv_size)
    q_end = num_attention_heads * head_dim
    k_end = q_end + num_key_value_heads * head_dim
    expected_Q = expected_qkv[:, :, :q_end]
    expected_K = expected_qkv[:, :, q_end:k_end]
    expected_V = expected_qkv[:, :, k_end:]

    # On-device kernel test
    device = xm.xla_device()
    hidden_states = hidden_states_cpu.to(device)
    weight = weight_cpu.to(device)
    rmsnorm = MagicMock()
    rmsnorm.weight = rmsnorm_weight_cpu.to(device)

    Q, K, V = qkv_kernel(
        hidden_states=hidden_states,
        w_qkv=weight,
        rmsnorm=rmsnorm,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )

    assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim)
    assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
    assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim)

    assert neuron_allclose(Q.cpu(), expected_Q, atol=1e-3, rtol=1e-3).allclose
    assert neuron_allclose(K.cpu(), expected_K, atol=1e-3, rtol=1e-3).allclose
    assert neuron_allclose(V.cpu(), expected_V, atol=1e-3, rtol=1e-3).allclose


def test_qkv_kernel_sequence_padding():
    batch_size, seq_len, hidden_dim = 2, 65, 128  # Odd sequence length > 64
    num_attention_heads, num_key_value_heads, head_dim = 8, 8, 16
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.randn(hidden_dim, fused_qkv_size)

    with patch("neuronx_distributed_inference.experimental.functional.qkv.qkv.nki_jit"):
        Q, K, V = qkv_kernel(
            hidden_states=hidden_states,
            w_qkv=weight,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        # Should still return original sequence length
        assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim)
        assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
        assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim)


def test_qkv_kernel_invalid_hidden_dim():
    batch_size, seq_len, hidden_dim = 2, 4, 128
    num_attention_heads, num_key_value_heads, head_dim = 8, 8, 16
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.randn(64, fused_qkv_size)  # Wrong hidden dim

    with pytest.raises(AssertionError):
        qkv_kernel(
            hidden_states=hidden_states,
            w_qkv=weight,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )


def test_qkv_kernel_invalid_rmsnorm_shape():
    batch_size, seq_len, hidden_dim = 2, 4, 128
    num_attention_heads, num_key_value_heads, head_dim = 8, 8, 16
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.randn(hidden_dim, fused_qkv_size)
    rmsnorm = MagicMock()
    rmsnorm.weight = torch.randn(64)  # Wrong size

    with pytest.raises(AssertionError):
        qkv_kernel(
            hidden_states=hidden_states,
            w_qkv=weight,
            rmsnorm=rmsnorm,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )


def test_qkv_kernel_no_bias():
    batch_size, seq_len, hidden_dim = 2, 4, 128
    num_attention_heads, num_key_value_heads, head_dim = 8, 8, 16
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.randn(hidden_dim, fused_qkv_size)

    with patch("neuronx_distributed_inference.experimental.functional.qkv.qkv.nki_jit") as mock_nki_jit:
        Q, K, V = qkv_kernel(
            hidden_states=hidden_states,
            w_qkv=weight,
            bias=None,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        mock_nki_jit.assert_called_once()
        assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim)
        assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
        assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim)
