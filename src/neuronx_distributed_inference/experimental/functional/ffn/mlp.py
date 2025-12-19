import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from neuronx_distributed_inference.models.config import get_platform_lnc
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronxcc.nki.language import nc
from neuronxcc.nki._pre_prod_kernels import NormType, ActFnType
from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit


logger = logging.getLogger("Neuron")

_HIDDEN_NUM_DIM = 3
_TORCH_TO_KERNEL_ACT_FN = {F.silu: ActFnType.SiLU, F.gelu: ActFnType.GELU}


# -- public functions at the top


def gated_mlp(
    w_up: nn.Module,
    w_gate: nn.Module,
    w_down: nn.Module,
    hidden: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    tensor_parallel_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    r"""
    Implements the following:
        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    .. note::
        If a tensor_parallel_group is provided (required for reduction) and the inputs
        meet the criteria for invoking a kernel, we invoke a kernel to compute the gated MLP

    :param w_up: Linear layer for up proj
    :param w_gate: Linear layer for gate proj
    :param w_down: Linear layer for down proj
    :param hidden: Input tensor of shape [batch, seqlen, hidden_dim]
    :param act_fn: Activation function applied to result of gate projection
    :param tensor_parallel_group: Process group used for reduction when using a kernel.
                                  If not provided, this function will not invoke a kernel.

    :return: Result of passing the input through a gated MLP
    """

    # Validate weights
    assert (
        w_up.weight.shape == w_gate.weight.shape
    ), f"expected dim w_up and w_gate shapes to match but got {w_up.weight.shape=} and {w_gate.weight.shape=}"
    assert (
        w_up.weight.shape[0] == w_down.weight.shape[1]
    ), f"expected dim 1 of w_down weight to be {w_up.weight.shape[0]} but got {w_down.weight.shape[1]}"

    # Validate input
    assert len(hidden.shape) == _HIDDEN_NUM_DIM

    if _should_gated_mlp_try_kernel(tensor_parallel_group, hidden.device.type, act_fn):
        try:
            kernel_act_fn = _TORCH_TO_KERNEL_ACT_FN[act_fn]
            output = gated_mlp_kernel_unreduced(w_up=w_up, w_gate=w_gate, w_down=w_down, hidden=hidden, act_fn=kernel_act_fn)
            output = reduce_from_tensor_model_parallel_region(output, process_group=tensor_parallel_group)
            logger.debug("Gated MLP kernel was invoked")
            return output
        except AssertionError as e:
            logger.debug(
                f"Gated MLP kernel can not be used for the following reason: {e}. Falling back to non-kernel version"
            )

    return w_down(act_fn(w_gate(hidden)) * w_up(hidden))


def gated_mlp_fused(
    w_up_gate: nn.Module,
    w_down: nn.Module,
    hidden: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
) -> torch.Tensor:
    """
    Implements the following using a fused up and gate projection:
        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    Args:
        w_up_gate: Fused linear layer for up and gate proj
        w_down: Linear layer for down proj
        hidden: Input tensor of shape [batch, seqlen, hidden_dim]
        act_fn: Activation function applied to result of gate projection
    """

    # Validate weights
    assert (
        w_up_gate.weight.shape[0] % 2 == 0
    ), f"w_up_gate output dimension is expected to be divisible by 2, but got {w_up_gate.weight.shape[0]}"
    assert (
        w_up_gate.weight.shape[0] // 2 == w_down.weight.shape[1]
    ), f"expected dim 1 of w_down weight to be {w_up_gate.weight.shape[0] // 2} but got {w_up_gate.weight.shape[1]}"

    # Validate input
    assert len(hidden.shape) == _HIDDEN_NUM_DIM

    up, gate = _run_fused_up_and_gate_proj(
        w_up_gate=w_up_gate,
        hidden=hidden,
    )

    return w_down(act_fn(gate) * up)


def gated_mlp_kernel_unreduced(
    w_up: nn.Module,
    w_gate: nn.Module,
    w_down: nn.Module,
    hidden: torch.Tensor,
    act_fn: ActFnType = ActFnType.SiLU,
    quantized: bool = False,
    quantize_clamp_bound: float = 1200.0,
) -> torch.Tensor:
    r"""
    Implements the following using NKI kernels:
        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    .. note::
        The output of this is not reduced. Make sure to reduce the output after
        using this function if needed.

    :param w_up: Linear layer for up proj
    :param w_gate: Linear layer for gate proj
    :param w_down: Linear layer for down proj
    :param hidden: Input tensor of shape [batch, seqlen, hidden_dim]
    :param act_fn: Activation function applied to result of gate projection
    :param quantized: Whether or not to use a quantized kernel
    :param quantize_clamp_bound: Boundary for clamping quantized tensors

    :return: Result of passing the input through a gated MLP
    """

    # Validate weights
    MAX_INTERMEDIATE_DIM = 4096
    assert (
        w_up.weight.shape == w_gate.weight.shape
    ), f"expected dim w_up and w_gate shapes to match but got {w_up.weight.shape=} and {w_gate.weight.shape=}"
    assert (
        w_down.weight.shape[0] <= MAX_INTERMEDIATE_DIM
    ), f"Intermediate dim must be <= {MAX_INTERMEDIATE_DIM}, but got {w_down.weight.shape[1]}"

    # Validate input
    HIDDEN_DIM_ALIGNMENT = 128
    assert len(hidden.shape) == _HIDDEN_NUM_DIM
    assert (
        hidden.shape[2] % HIDDEN_DIM_ALIGNMENT == 0
    ), f"Hidden dim must be divisible by {HIDDEN_DIM_ALIGNMENT}, but got {hidden.shape[2]}"

    grid = (nc(get_platform_lnc()),)
    ln_w = torch.zeros(size=(1, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device)
    output_tensor = torch.zeros(
        size=hidden.shape,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    if quantized:
        # TODO: sediriso@ to complete implementation in follow-up CR
        raise NotImplementedError
        """
        nki_jit()(quant_mlp_isa_kernel)[grid](
            hidden=hidden,
            ln_w=ln_w,
            gate_w=w_gate.weight.data.T,
            gate_w_scale=w_gate.scale,
            up_w=w_up.weight.data.T,
            up_w_scale=w_up.scale,
            down_w=w_down.weight.data.T,
            down_w_scale=w_down.scale,
            clamp_bound=quantize_clamp_bound,
            act_fn=act_fn,
            out=output_tensor,
            kernel_name="MLP",
            norm_type=NormType.NO_NORM,
        )
        """
    else:
        nki_jit()(mlp_isa_kernel)[grid](
            hidden=hidden,
            ln_w=ln_w,
            gate_w=w_gate.weight.data.T,
            up_w=w_up.weight.data.T,
            down_w=w_down.weight.data.T,
            act_fn=act_fn,
            out=output_tensor,
            kernel_name="MLP",
            norm_type=NormType.NO_NORM,
        )

    return output_tensor


# -- private functions go below


def _should_gated_mlp_try_kernel(tensor_parallel_group: ProcessGroup, device_type: str, act_fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Returns whether or not the gated_mlp() function should try to call a kernel for execution
    """
    return (
        tensor_parallel_group is not None
        and device_type != "cpu"
        and act_fn in _TORCH_TO_KERNEL_ACT_FN
    )


def _run_fused_up_and_gate_proj(w_up_gate: nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes a fused up and gate projection, returning the result of each projection
    """
    # [batch, seqlen, 2 * up_dim]
    fused_up_gate: torch.Tensor = w_up_gate(hidden)

    up, gate = torch.tensor_split(
        fused_up_gate,
        2,
        dim=2,
    )

    return up, gate
