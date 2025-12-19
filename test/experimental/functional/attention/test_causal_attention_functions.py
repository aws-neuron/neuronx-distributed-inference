import pytest
from unittest.mock import patch

import torch
	 
from neuronx_distributed_inference.experimental.functional import causal_scaled_dot_product_attention, scaled_dot_product_attention_kernel
from torch_xla.core import xla_model as xm
	 
torch.manual_seed(0)


@pytest.mark.parametrize("batch_size,num_heads,seq_len,dim,dtype,scale", [
    (4, 16, 512, 64, torch.float32, 2.0),
    (1, 128, 4096, 16, torch.float32, None),
])
def test_causal_scaled_dot_production_attention_no_kernel_cpu(batch_size, num_heads, seq_len, dim, dtype, scale):
    Q=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype)
    K=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype)
    V=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype)

    with patch("neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions.nki_jit") as mock_nki_jit:
        actual = causal_scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            scale=scale,
        )

        mock_nki_jit.assert_not_called()
        expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size,num_heads,seq_len,dim,dtype,scale", [
    (4, 4, 512, 4, torch.float32, 2.0),
    (1, 4, 512, 4, torch.float32, None),
])
def test_causal_scaled_dot_production_attention_no_kernel_flag_is_false(batch_size, num_heads, seq_len, dim, dtype, scale):
    device = xm.xla_device()
    Q=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    K=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    V=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)

    with patch("neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions.nki_jit") as mock_nki_jit:
        actual = causal_scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            scale=scale,
            try_using_kernel=False,
        )

        mock_nki_jit.assert_not_called()
        expected = torch.nn.functional.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=True, scale=scale)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("batch_size,num_heads,seq_len,dim,dtype,scale,atol,rtol", [
    (4, 16, 128, 32, torch.float32, 2.0, 1e-5, 1e-2),
    (1, 16, 128, 32, torch.float32, None, 1e-5, 1.3e-6),
])
def test_causal_scaled_dot_production_attention_no_kernel_invalid_shape(batch_size, num_heads, seq_len, dim, dtype, scale, atol, rtol):
    device = xm.xla_device()
    Q=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    K=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    V=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)

    with patch("neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions.nki_jit") as mock_nki_jit:
        actual = causal_scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            scale=scale,
        )

        mock_nki_jit.assert_not_called()
        expected = torch.nn.functional.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=True, scale=scale)
        torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


def test_causal_scaled_dot_production_attention_invalid_q_tensor_shape():
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="Q is expected to have 4 dimensions"):
        causal_scaled_dot_product_attention(Q=Q, K=K, V=V)
        


def test_causal_scaled_dot_production_attention_invalid_k_tensor_shape():
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="K is expected to have 4 dimensions"):
        causal_scaled_dot_product_attention(Q=Q, K=K, V=V)


def test_causal_scaled_dot_production_attention_invalid_v_tensor_shape():
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="V is expected to have 4 dimensions"):
        causal_scaled_dot_product_attention(Q=Q, K=K, V=V)


def test_causal_scaled_dot_production_attention_kv_shape_mismatch():
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len - 1, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="K, V sequence lengths must be identical"):
        causal_scaled_dot_product_attention(Q=Q, K=K, V=V)


@pytest.mark.parametrize("batch_size,num_heads,seq_len,dim,dtype,scale", [
    (4, 16, 512, 32, torch.float32, 2.0),
    (1, 16, 512, 32, torch.float32, None),
])
def test_causal_scaled_dot_production_attention_calls_kernel(batch_size, num_heads, seq_len, dim, dtype, scale):
    device = xm.xla_device()
    Q=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    K=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    V=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)

    with patch("neuronx_distributed_inference.experimental.functional.attention.causal_attention_functions.nki_jit") as mock_nki_jit:
        causal_scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            scale=scale,
        )

        mock_nki_jit.assert_called_once()


@pytest.mark.parametrize("batch_size,num_heads,seq_len,dim,dtype,is_causal,scale,atol,rtol", [
    # Global attention
    (4, 16, 512, 64, torch.bfloat16, False, 2.0, 8e-3, 1e-2),
    (4, 16, 512, 64, torch.bfloat16, False, None, 8e-3, 1e-5),
    (4, 16, 512, 64, torch.float32, False, None, 9e-3, 1e-5),
    (2, 16, 512, 64, torch.bfloat16, False, None, 9e-3, 1e-5),
    (2, 16, 512, 64, torch.float32, False, None, 9e-3, 1e-5),
    (1, 32, 1024, 64, torch.bfloat16, False, None, 8e-3, 1e-5),
    (1, 32, 1024, 64, torch.float32, False, None, 2e-2, 1e-5),
    (1, 1, 4096, 128, torch.bfloat16, False, None, 2e-3, 1e-5),
    (1, 1, 4096, 128, torch.float32, False, None, 2e-3, 1e-5),
    # Causal attention
    (4, 16, 512, 64, torch.bfloat16, True, 2.0, 2e-2, 1e-2),
    (4, 16, 512, 64, torch.bfloat16, True, None, 2e-2, 1e-5),
    (4, 16, 512, 64, torch.float32, True, None, 2e-2, 1e-5),
    (2, 16, 512, 64, torch.bfloat16, True, None, 2e-2, 1e-5),
    (2, 16, 512, 64, torch.float32, True, None, 2e-2, 1e-5),
    (1, 32, 1024, 64, torch.bfloat16, True, None, 2e-2, 1e-5),
    (1, 32, 1024, 64, torch.float32, True, None, 2e-2, 1e-5),
    (1, 1, 4096, 128, torch.bfloat16, True, None, 2e-2, 1e-5),
    (1, 1, 4096, 128, torch.float32, True, None, 2e-2, 1e-5),
])
def test_scaled_dot_production_attention_kernel(batch_size, num_heads, seq_len, dim, dtype, is_causal, scale, atol, rtol):
    device = xm.xla_device()
    Q=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    K=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    V=torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)

    actual = scaled_dot_product_attention_kernel(
        Q=Q,
        K=K,
        V=V,
        is_causal=is_causal,
        scale=scale,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=is_causal, scale=scale)
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)  


@pytest.mark.parametrize("q_len,k_len,v_len", [
    (4, 4, 8),
    (4, 8, 4),
    (8, 4, 4),
])
def test_scaled_dot_production_attention_kernel_causal_invalid_inputs(q_len, k_len, v_len):
    device = xm.xla_device()
    bsz = 16
    num_heads = 4
    head_dim = 64
    Q=torch.randn(bsz, num_heads, q_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, k_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, v_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="Q, K, V sequence lengths must be identical for causal attention kernel"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=True,
        )


@pytest.mark.parametrize("num_heads,is_causal", [(129, True), (129, False)])
def test_scaled_dot_production_attention_kernel_invalid_num_heads(num_heads, is_causal):
    device = xm.xla_device()
    bsz = 16
    seq_len = 512
    head_dim = 64
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="Num heads must be <= 128"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("seq_len,is_causal", [(511, True), (511, False)])
def test_scaled_dot_production_attention_kernel_invalid_seq_len(seq_len, is_causal):
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="Seq len must be >= 512"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("is_causal", [(True), (False)])
def test_scaled_dot_production_attention_kernel_invalid_kv_len(is_causal):
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, 1024, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="K, V sequence lengths must be identical"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("is_causal", [(True), (False)])
def test_scaled_dot_production_attention_kernel_invalid_q_tensor_shape(is_causal):
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="Q is expected to have 4 dimensions"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("is_causal", [(True), (False)])
def test_scaled_dot_production_attention_kernel_invalid_k_tensor_shape(is_causal):
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="K is expected to have 4 dimensions"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )


@pytest.mark.parametrize("is_causal", [(True), (False)])
def test_scaled_dot_production_attention_kernel_invalid_v_tensor_shape(is_causal):
    device = xm.xla_device()
    bsz = 16
    num_heads = 32
    head_dim = 64
    seq_len = 512
    Q=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K=torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    V=torch.randn(bsz, num_heads, seq_len, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match="V is expected to have 4 dimensions"):
        scaled_dot_product_attention_kernel(
            Q=Q,
            K=K,
            V=V,
            is_causal=is_causal,
        )
