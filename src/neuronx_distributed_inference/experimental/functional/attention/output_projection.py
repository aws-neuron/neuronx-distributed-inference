import logging
import torch
from typing import Optional
from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._pre_prod_kernels.output_proj import output_proj_kernel

logger = logging.getLogger("Neuron")


# Public functions
def o_proj_kernel_unreduced(
    hidden_states: torch.Tensor,
    w_o: torch.Tensor,
    num_attention_heads: int,
    head_dim: int,
    bias: Optional[torch.Tensor] = None,
    tp_degree: int = None,
    logical_nc_config: int = 1,
) -> torch.Tensor:
    r"""
    Projects attention output from attention head space to hidden dimension space.
    This is the final linear transformation in attention mechanisms (W_o in the Transformer paper).

    Computes: ``output = (hidden_states @ w_o) + bias``, where hidden_states contains concatenated
    attention head outputs. The projection transforms from ``(num_heads * head_dim)`` to ``hidden_size``.

    .. note::
        This function does NOT perform tensor parallel reduction. Each TP rank computes a
        partial result that must be reduced (all-reduce or reduce-scatter) to get the final output.
        Bias is divided by tp_degree so that after reduction across ranks, the full bias is applied once.

    :param hidden_states: Attention output with concatenated heads before output projection.
                         Shape: ``(batch_size, seq_len, num_attention_heads * head_dim)``
    :param w_o: Sharded output projection weight matrix (per TP rank).
                Shape: ``(num_attention_heads * head_dim, hidden_size)``
    :param num_attention_heads: Number of attention heads (per TP rank).
    :param head_dim: Dimension of each attention head.
    :param bias: Optional bias tensor for output projection.
                 Bias is divided by tp_degree for correct accumulation after reduction.
                 Shape: ``(hidden_size,)``
    :param tp_degree: Tensor parallel degree, used for bias scaling. Required if bias is provided.
    :param logical_nc_config: Logical NeuronCore configuration for kernel grid. Default: 1

    :return: Output tensor after projection (no reduction applied).
             Shape: ``(batch_size, seq_len, hidden_size)``

    Example usage:
        >>> # Setup dimensions
        >>> batch_size, seq_len = 2, 512
        >>> num_attention_heads, head_dim = 8, 64
        >>> hidden_size = 512
        >>> tp_degree = 2
        >>>
        >>> # Create input tensors (multi-head attention output)
        >>> attn_output = torch.randn(batch_size, seq_len, num_attention_heads * head_dim)
        >>>
        >>> # Create weight matrix (transposed)
        >>> weight = torch.randn(num_attention_heads * head_dim, hidden_size)
        >>>
        >>> # Optional bias
        >>> bias = torch.randn(hidden_size)
        >>>
        >>> # Move to XLA device
        >>> device = xm.xla_device()
        >>> attn_output = attn_output.to(device)
        >>> weight = weight.to(device)
        >>> bias = bias.to(device)
        >>>
        >>> # Compute output projection (unreduced)
        >>> output = o_proj_kernel_unreduced(
        ...     hidden_states=attn_output,
        ...     w_o=weight,
        ...     num_attention_heads=num_attention_heads,
        ...     head_dim=head_dim,
        ...     bias=bias,
        ...     tp_degree=tp_degree,
        ... )

    """
    attn_dim, hidden_size = w_o.shape
    batch_size, seq_len, input_dim = hidden_states.shape

    _validate_inputs_o_proj_kernel_unreduced(attn_dim, input_dim, num_attention_heads, head_dim, hidden_states, tp_degree, bias)

    # hidden_states is (batch_size, seq_len, num_attention_heads * head_dim)
    # and kernel wants (batch_size, num_attention_heads, head_dim, seq_len) for input
    hidden_states = hidden_states.reshape(batch_size, seq_len, num_attention_heads, head_dim)
    kernel_attn_in = hidden_states.permute(0, 2, 3, 1)

    out = torch.zeros(batch_size, seq_len, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

    bias_per_tp_degree = None
    if bias is not None:
        bias_per_tp_degree = bias.unsqueeze(0) / tp_degree

    grid = (nc(logical_nc_config),)
    nki_jit()(output_proj_kernel)[grid](
        active=kernel_attn_in,
        weight=w_o,
        out=out,
        bias=bias_per_tp_degree,
    )

    return out


# Private Functions
def _validate_inputs_o_proj_kernel_unreduced(
    attn_dim: int,
    input_dim: int,
    num_attention_heads: int,
    head_dim: int,
    hidden_states: torch.Tensor,
    tp_degree: Optional[int],
    bias: Optional[torch.Tensor],
) -> None:
    """Validate input dimensions and parameters for o_proj_kernel_unreduced."""

    # Validate weight and input dimension compatibility
    assert attn_dim == input_dim, (
        f"Weight dimension ({attn_dim}) must match input dimension ({input_dim}). "
        f"Expected w_o.shape[0] == hidden_states.shape[2]"
    )

    # Validate attention head dimensions
    expected_attn_dim = num_attention_heads * head_dim
    assert attn_dim == expected_attn_dim, (
        f"Attention dimension mismatch: w_o.shape[0]={attn_dim} but "
        f"num_attention_heads * head_dim = {num_attention_heads} * {head_dim} = {expected_attn_dim}. "
        f"Input shape: {hidden_states.shape}"
    )

    # Validate bias and tp_degree consistency
    if bias is not None:
        assert tp_degree is not None, (
            "tp_degree must be provided when bias is specified. "
            "Bias scaling requires knowing the tensor parallel degree."
        )
