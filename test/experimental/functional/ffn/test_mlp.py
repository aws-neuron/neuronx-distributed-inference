import pytest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronxcc.nki._pre_prod_kernels import ActFnType
from neuronx_distributed.quantization.quantization_layers import QuantizedColumnParallel, QuantizedRowParallel
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.experimental.functional.ffn.mlp import gated_mlp, gated_mlp_fused, gated_mlp_kernel_unreduced
from torch_xla.core import xla_model as xm

torch.manual_seed(0)


# TODO (sediriso@): Re-enable quantized layer tests after figuring out the issue with
# creating multiple process groups for these unit tests


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (2, 2, 4, 8, torch.float32, nn.Linear), 
    (2, 2, 4, 8, torch.float32, RowParallelLinear), 
    (2, 2, 4, 8, torch.float32, ColumnParallelLinear), 
    #(2, 2, 4, 8, torch.float32, QuantizedColumnParallel), 
    #(2, 2, 4, 8, torch.float32, QuantizedRowParallel),
    (5, 7, 13, 28, torch.float32, nn.Linear), 
    (5, 7, 13, 28, torch.float32, RowParallelLinear), 
    (5, 7, 13, 28, torch.float32, ColumnParallelLinear), 
    #(5, 7, 13, 28, torch.float32, QuantizedColumnParallel), 
    #(5, 7, 13, 28, torch.float32, QuantizedRowParallel),
])
def test_gated_mlp(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype)
    act_fn = F.silu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)

    actual = gated_mlp(
        w_up=w_up,
        w_gate=w_gate,
        w_down=w_down,
        act_fn=act_fn,
        hidden=hidden
    )

    assert actual.shape == (batch_size, seq_len, hidden_dim)
    expected = w_down(act_fn(w_gate(hidden)) * w_up(hidden))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (1, 128, 128, 256, torch.bfloat16, nn.Linear), 
    (1, 128, 128, 256, torch.bfloat16, RowParallelLinear), 
    (1, 128, 128, 256, torch.bfloat16, ColumnParallelLinear),
])
def test_gated_mlp_invokes_kernel(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    device = xm.xla_device()
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype, device=device)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype, device=device)
    act_fn = F.silu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
    tensor_parallel_group = [1]  # Dummmy group

    # Mock the kernel and reduction functions
    with patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.gated_mlp_kernel_unreduced") as mock_kernel, \
         patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.reduce_from_tensor_model_parallel_region") as mock_reduce:
        gated_mlp(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            act_fn=act_fn,
            hidden=hidden,
            tensor_parallel_group=tensor_parallel_group,
        )

        mock_kernel.assert_called_once()
        mock_reduce.assert_called_once()


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (1, 128, 128, 256, torch.bfloat16, nn.Linear), 
    (1, 128, 128, 256, torch.bfloat16, RowParallelLinear), 
    (1, 128, 128, 256, torch.bfloat16, ColumnParallelLinear),
])
def test_gated_mlp_skips_kernel_cpu_tensor(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype)
    act_fn = F.silu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    tensor_parallel_group = [1]  # Dummmy group

    # Patch the kernel and reduction functions to validate they were not reached
    with patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.gated_mlp_kernel_unreduced") as mock_kernel, \
         patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.reduce_from_tensor_model_parallel_region") as mock_reduce:
        actual = gated_mlp(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            act_fn=act_fn,
            hidden=hidden,
            tensor_parallel_group=tensor_parallel_group,
        )

        expected = w_down(act_fn(w_gate(hidden)) * w_up(hidden))
        torch.testing.assert_close(actual, expected)
        mock_kernel.assert_not_called()
        mock_reduce.assert_not_called()


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (1, 5, 32, 48, torch.bfloat16, nn.Linear), 
    (1, 5, 32, 48, torch.bfloat16, RowParallelLinear), 
    (1, 5, 32, 48, torch.bfloat16, ColumnParallelLinear),
])
def test_gated_mlp_skips_kernel_invalid_input(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    device = xm.xla_device()
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype, device=device)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype, device=device)
    act_fn = F.silu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
    tensor_parallel_group = [1]  # Dummmy group

    # Patch the reduction function to validate it was never reached
    with patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.reduce_from_tensor_model_parallel_region") as mock_reduce:
        actual = gated_mlp(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            act_fn=act_fn,
            hidden=hidden,
            tensor_parallel_group=tensor_parallel_group,
        )

        expected = w_down(act_fn(w_gate(hidden)) * w_up(hidden))
        torch.testing.assert_close(actual, expected)
        mock_reduce.assert_not_called()


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (1, 128, 128, 256, torch.bfloat16, nn.Linear), 
    (1, 128, 128, 256, torch.bfloat16, RowParallelLinear), 
    (1, 128, 128, 256, torch.bfloat16, ColumnParallelLinear),
])
def test_gated_mlp_skips_kernel_invalid_act_fn(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    device = xm.xla_device()
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype, device=device)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype, device=device)
    act_fn = F.relu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
    tensor_parallel_group = [1]  # Dummmy group

    # Patch the reduction function to validate it was never reached
    with patch("neuronx_distributed_inference.experimental.functional.ffn.mlp.reduce_from_tensor_model_parallel_region") as mock_reduce:
        actual = gated_mlp(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            act_fn=act_fn,
            hidden=hidden,
            tensor_parallel_group=tensor_parallel_group,
        )

        expected = w_down(act_fn(w_gate(hidden)) * w_up(hidden))
        torch.testing.assert_close(actual, expected)
        mock_reduce.assert_not_called()


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (2, 2, 4, 8, torch.float32, nn.Linear), 
    (2, 2, 4, 8, torch.float32, RowParallelLinear), 
    (2, 2, 4, 8, torch.float32, ColumnParallelLinear), 
    #(2, 2, 4, 8, torch.float32, QuantizedColumnParallel), 
    #(2, 2, 4, 8, torch.float32, QuantizedRowParallel),
    (5, 7, 13, 28, torch.float32, nn.Linear), 
    (5, 7, 13, 28, torch.float32, RowParallelLinear), 
    (5, 7, 13, 28, torch.float32, ColumnParallelLinear), 
    #(5, 7, 13, 28, torch.float32, QuantizedColumnParallel), 
    #(5, 7, 13, 28, torch.float32, QuantizedRowParallel),
])
def test_gated_mlp_fused(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    w_up_gate = layer_class(hidden_dim, 2*up_dim, bias=False, dtype=dtype)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype)
    act_fn = F.silu
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    # Initialize weights
    w_up_gate.weight.data = torch.cat([
        torch.ones(up_dim, hidden_dim, dtype=dtype),
        torch.ones(up_dim, hidden_dim, dtype=dtype) * 2,
    ], dim=0)
    w_down.weight.data = torch.ones(hidden_dim, up_dim, dtype=dtype) * 3

    actual = gated_mlp_fused(
        w_up_gate=w_up_gate,
        w_down=w_down,
        act_fn=act_fn,
        hidden=hidden
    )

    assert actual.shape == (batch_size, seq_len, hidden_dim)
    # Manually compute gated MLP without using any fused layers
    w_up = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype)
    w_up.weight.data = torch.ones(up_dim, hidden_dim, dtype=dtype)
    w_gate.weight.data = torch.ones(up_dim, hidden_dim, dtype=dtype) * 2
    expected = w_down(act_fn(w_gate(hidden)) * w_up(hidden))
    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(reason="Accuracy issue, maybe due to compiler regression. Need to ensure no autocast in jit flow")
@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,up_dim,dtype,layer_class", [
    (1, 128, 128, 256, torch.bfloat16, nn.Linear), 
    (1, 128, 128, 256, torch.bfloat16, RowParallelLinear), 
    (1, 128, 128, 256, torch.bfloat16, ColumnParallelLinear),
])
def test_gated_mlp_kernel_unreduced(batch_size, seq_len, hidden_dim, up_dim, dtype, layer_class):
    device = xm.xla_device()
    w_up = layer_class(hidden_dim, up_dim, bias=True, dtype=dtype, device=device)
    w_gate = layer_class(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_down = layer_class(up_dim, hidden_dim, bias=False, dtype=dtype, device=device)
    act_fn = ActFnType.SiLU
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)

    actual = gated_mlp_kernel_unreduced(
        w_up=w_up,
        w_gate=w_gate,
        w_down=w_down,
        hidden=hidden,
        act_fn=act_fn,
    )

    assert actual.shape == (batch_size, seq_len, hidden_dim)
    # Manually compute gated MLP without any kernels
    expected = w_down(F.silu(w_gate(hidden)) * w_up(hidden))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=6e-2)


# TODO: Uncomment test below (+ parametrize) once they are passing
"""
def test_gated_mlp_kernel_quantized():
    device = xm.xla_device()
    batch_size = 1
    seq_len = 128
    hidden_dim = 128
    up_dim = 256
    dtype = torch.float32
    w_up = QuantizedColumnParallel(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_gate = QuantizedColumnParallel(hidden_dim, up_dim, bias=False, dtype=dtype, device=device)
    w_down = QuantizedColumnParallel(up_dim, hidden_dim, bias=False, dtype=dtype, device=device)
    hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
    # Initialize weights
    w_up.weight.data = torch.ones(up_dim, hidden_dim, dtype=dtype, device=device)
    w_gate.weight.data = torch.ones(up_dim, hidden_dim, dtype=dtype, device=device) * 2
    w_down.weight.data = torch.ones(hidden_dim, up_dim, dtype=dtype, device=device) * 3

    actual = gated_mlp_kernel(
        w_up=w_up,
        w_gate=w_gate,
        w_down=w_down,
        hidden=hidden,
        quantized=True,
    )

    assert actual.shape == (batch_size, seq_len, hidden_dim)
    # Manually compute gated MLP without any kernels
    expected = w_down(F.silu(w_gate(hidden)) * w_up(hidden))
    torch.testing.assert_close(actual, expected)
"""