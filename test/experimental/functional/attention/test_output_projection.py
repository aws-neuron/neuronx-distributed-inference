import os
import pytest
from unittest.mock import patch, MagicMock
import torch
from neuronx_distributed_inference.experimental.functional.attention.output_projection import o_proj_kernel_unreduced
from torch_xla.core import xla_model as xm

os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
torch.manual_seed(0)


@pytest.mark.parametrize("batch_size,seq_len,num_attention_heads,head_dim,hidden_size,dtype", [
    (2, 512, 2, 128, 512, torch.bfloat16),
    (1, 512, 4, 128, 1024, torch.bfloat16),
])
def test_o_proj_kernel_cpu(batch_size, seq_len, num_attention_heads, head_dim, hidden_size, dtype):
    hidden_states = torch.randn(batch_size, seq_len, num_attention_heads * head_dim, dtype=dtype)
    weight = torch.randn(num_attention_heads * head_dim, hidden_size, dtype=dtype)

    with patch("neuronx_distributed_inference.experimental.functional.attention.output_projection.nki_jit") as mock_nki_jit:
        mock_kernel_call = MagicMock()
        mock_wrapped = MagicMock()
        mock_wrapped.__getitem__.return_value = mock_kernel_call
        mock_nki_jit.return_value = MagicMock(return_value=mock_wrapped)

        output = o_proj_kernel_unreduced(
            hidden_states=hidden_states,
            w_o=weight,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )

        mock_nki_jit.assert_called_once()
        mock_kernel_call.assert_called_once()
        call_kwargs = mock_kernel_call.call_args.kwargs
        assert call_kwargs["active"].shape == (batch_size, num_attention_heads, head_dim, seq_len)
        assert torch.equal(call_kwargs["weight"], weight)
        assert call_kwargs["out"].shape == (batch_size, seq_len, hidden_size)
        assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize("batch_size,seq_len,num_attention_heads,head_dim,hidden_size,dtype", [
    (2, 512, 2, 64, 1024, torch.float16),
    (2, 512, 1, 64, 1024, torch.float16),
])
def test_o_proj_kernel_without_reduction(batch_size, seq_len, num_attention_heads, head_dim, hidden_size, dtype):
    hidden_states_cpu = torch.randn(batch_size, seq_len, num_attention_heads * head_dim, dtype=dtype)
    weight_cpu = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=dtype).T

    expected_output = compute_expected_output(
        hidden_states_cpu, weight_cpu, batch_size, seq_len, num_attention_heads, head_dim, hidden_size
    )
    hidden_states, weight = move_tensors_to_device(hidden_states_cpu, weight_cpu)

    output = o_proj_kernel_unreduced(
        hidden_states=hidden_states,
        w_o=weight,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
    )

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.shape == expected_output.shape
    assert torch.allclose(output.cpu(), expected_output, atol=1e-3, rtol=1e-3)


def test_o_proj_kernel_invalid_weight_shape():
    batch_size, seq_len, num_attention_heads, head_dim = 2, 4, 8, 16
    hidden_states = torch.randn(batch_size, seq_len, num_attention_heads * head_dim)
    weight = torch.randn(100, 128)  # Wrong first dimension

    with pytest.raises(AssertionError):
        o_proj_kernel_unreduced(
            hidden_states=hidden_states,
            w_o=weight,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )


def test_o_proj_kernel_invalid_attention_output_shape():
    batch_size, seq_len, num_attention_heads, head_dim, hidden_size = 2, 4, 8, 16, 128
    hidden_states = torch.randn(batch_size, seq_len, 100)  # Wrong size
    weight = torch.randn(num_attention_heads * head_dim, hidden_size)

    with pytest.raises(AssertionError):
        o_proj_kernel_unreduced(
            hidden_states=hidden_states,
            w_o=weight,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )


def test_o_proj_kernel_no_bias():
    batch_size, seq_len, num_attention_heads, head_dim, hidden_size = 2, 512, 2, 128, 512
    hidden_states_cpu = torch.randn(batch_size, seq_len, num_attention_heads * head_dim)
    weight_cpu = torch.randn(num_attention_heads * head_dim, hidden_size)

    expected_output = compute_expected_output(
        hidden_states_cpu, weight_cpu, batch_size, seq_len, num_attention_heads, head_dim, hidden_size
    )
    hidden_states, weight = move_tensors_to_device(hidden_states_cpu, weight_cpu)

    output = o_proj_kernel_unreduced(
        hidden_states=hidden_states,
        w_o=weight,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        bias=None,
    )

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert torch.allclose(output.cpu(), expected_output, atol=1e-3, rtol=1e-3)


def test_o_proj_kernel_with_bias():
    batch_size, seq_len, num_attention_heads, head_dim, hidden_size = 2, 512, 2, 128, 512
    hidden_states_cpu = torch.randn(batch_size, seq_len, num_attention_heads * head_dim)
    weight_cpu = torch.randn(num_attention_heads * head_dim, hidden_size)
    bias_cpu = torch.randn(hidden_size)

    expected_output = compute_expected_output(
        hidden_states_cpu, weight_cpu, batch_size, seq_len, num_attention_heads, head_dim, hidden_size, bias_cpu
    )
    hidden_states, weight, bias = move_tensors_to_device(hidden_states_cpu, weight_cpu, bias_cpu)

    output = o_proj_kernel_unreduced(
        hidden_states=hidden_states,
        w_o=weight,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        bias=bias,
        tp_degree=1,
    )

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert torch.allclose(output.cpu(), expected_output, atol=1e-3, rtol=1e-3)


# Helper functions
def compute_expected_output(x_cpu, weight_cpu, batch_size, seq_len, num_attention_heads, head_dim, hidden_size, bias_cpu=None):
    """Compute expected output using CPU reference implementation."""
    attn_reshaped = x_cpu.view(-1, num_attention_heads * head_dim)
    expected_output = torch.matmul(attn_reshaped, weight_cpu)
    expected_output = expected_output.view(batch_size, seq_len, hidden_size)
    if bias_cpu is not None:
        expected_output = expected_output + bias_cpu
    return expected_output


def move_tensors_to_device(*tensors):
    """Move tensors to XLA device."""
    device = xm.xla_device()
    return tuple(t.to(device) for t in tensors)


def create_mock_tp_group(size=2):
    """Create a mock tensor parallel group."""
    mock_tp_group = MagicMock()
    mock_tp_group.size.return_value = size
    return mock_tp_group
