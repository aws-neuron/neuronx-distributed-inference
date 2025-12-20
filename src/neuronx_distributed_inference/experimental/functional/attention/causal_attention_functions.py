import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from neuronx_distributed_inference.models.config import get_platform_lnc
from neuronxcc.nki.language import nc
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit

logger = logging.getLogger("Neuron")

# -- public functions at the top


def causal_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    scale: Optional[float] = None,
    try_using_kernel: bool = True,
) -> Tensor:
    r"""
    Computes scaled dot product attention using the provided Q, K, V:
        softmax(QK^T * scale + causal_mask) @ V

    .. note::
        Uses SDPA kernel if the inputs meet the criteria

    :param Q: Query vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param K: Key vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param V: Value vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param scale: Scaling factor applied prior to softmax. If None, the default value is 1 / sqrt(head_dim)
    :param try_using_kernel: Indicates whether function should try to use a kernel for execution.
                             Default behavior is to use a kernel if possible. If set to false,
                             skips using the kernel even if input is eligible.

    :return: Result of causal scaled dot product attention computation
             Shape: (batch, num_heads, seq_len, head_dim)
    """
    _validate_inputs_for_sdpa(Q, K, V, scale)

    # Try calling kernel if eligible
    if _should_sdpa_try_kernel(Q.device.type, try_using_kernel):
        try:
            output = scaled_dot_product_attention_kernel(Q, K, V, is_causal=True, scale=scale)
            logger.debug("SDPA kernel was invoked")
            return output
        except AssertionError as e:
            logger.debug(
                f"SDPA kernel can not be used for the following reason: {e}. Falling back to non-kernel version"
            )

    head_dim = Q.shape[3]
    scale = 1 / math.sqrt(head_dim) if scale is None else scale

    # Computes causal scaled dot product attention: softmax(QK^T * scale + causal_mask) @ V
    scores = torch.matmul(Q, K.transpose(2, 3)) * scale
    _, _, source_seq_len, target_seq_len = scores.shape
    causal_mask = (
        _create_causal_attn_mask(source_seq_len, target_seq_len, scores.device)
        .expand(1, 1, source_seq_len, target_seq_len)
    )
    # You cannot use -inf well on Neuron, you will run into 1003 errors
    scores = torch.where(causal_mask, scores, torch.finfo(scores.dtype).min)
    scores = F.softmax(scores.float(), dim=-1).type_as(Q)
    output = torch.matmul(scores, V)

    return output


def scaled_dot_product_attention_kernel(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    r"""
    Computes scaled dot product attention on the provided Q, K, V using a NKI kernel.

    When is_causal is True, returns:
        softmax(QK^T * scale + causal_mask) @ V
    When is_causal is False, returns:
        softmax(QK^T * scale) @ V

    :param Q: Query vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param K: Key vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param V: Value vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param is_causal: Whether or not to compute causal attention
    :param scale: Scaling factor applied prior to softmax. If None, the default value is 1 / sqrt(head_dim)

    :return: Result of scaled dot product attention computation
             Shape: (batch, num_heads, seq_len, head_dim)
    """
    _validate_inputs_for_flash_attn_kernel(Q, K, V, is_causal, scale)

    bsz, num_heads, q_len, head_dim = Q.shape
    Q, K, V = _transform_qkv_for_flash_attn_kernel(Q, K, V, scale)

    attn_output = torch.zeros(bsz * num_heads, head_dim, q_len, dtype=Q.dtype, device=Q.device)
    kernel_name = "CausalAttentionMMSoftmaxMMWithoutSwap" if is_causal else "AttentionMMSoftmaxMMWithoutSwap"
    grid = (nc(get_platform_lnc()),)
    nki_jit()(attention_isa_kernel)[grid](
        q=Q,
        k=K,
        v=V,
        scale=1.0,
        out=attn_output,
        kernel_name=kernel_name,
    )

    # BHSD
    return attn_output.reshape((bsz, num_heads, head_dim, q_len)).transpose(2, 3)


def qkv_proj(wQKV, hidden, n_heads, n_kv_heads, d_head, tp_degree) -> tuple[Tensor, Tensor, Tensor]:

    # Validate weight
    assert len(wQKV.weight.shape) == 2
    assert wQKV.weight.shape[1] == (
        d_head * n_heads * tp_degree
    ), f"expected dim 1 of fused QKV weight to be {d_head * n_heads} but got {wQKV.weight.shape[1]}"
    assert (
        wQKV.weight.shape[0] == (n_heads + 2 * n_kv_heads) * d_head
    ), f"expected dim 0 of fused QKV weight to be {(n_heads + 2 * n_kv_heads) * d_head // tp_degree} but got {wQKV.weight.shape[0]}"

    # Validate input
    assert len(hidden.shape) >= 2
    assert hidden.shape[-1] == d_head * n_heads * tp_degree, "dim -1 of hidden should be hidden"

    qkv = wQKV(hidden)
    q, k, v = _split_fused_qkv(qkv, n_heads, n_kv_heads, d_head)

    return q, k, v


# -- private functions go below


def _should_sdpa_try_kernel(device_type: str, try_using_kernel: bool) -> bool:
    r"""
    Returns whether or not the scaled_dot_product_attention() function should try to call a kernel for execution

    :param device_type: Device type of input tensor
    :param try_using_kernel: Indicates whether we should try using the kernel
    """
    return (
        device_type != "cpu"
        and try_using_kernel
    )


def _validate_inputs_for_sdpa(Q: Tensor, K: Tensor, V: Tensor, scale: float):
    r"""
    Validates inputs to scaled_dot_product_attention() API

    :param Q: Query vectors used for attention computation. Shape: (batch, num_heads, seq_len, head_dim)
    :param K: Key vectors used for attention computation. Shape: (batch, num_heads, seq_len, head_dim)
    :param V: Value vectors used for attention computation. Shape: (batch, num_heads, seq_len, head_dim)
    :param scale: Scaling factor applied prior to softmax
    """
    QKV_NUM_DIMS = 4
    assert len(Q.shape) == QKV_NUM_DIMS, f"Q is expected to have {QKV_NUM_DIMS} dimensions, but got {len(Q.shape)}"
    assert len(K.shape) == QKV_NUM_DIMS, f"K is expected to have {QKV_NUM_DIMS} dimensions, but got {len(K.shape)}"
    assert len(V.shape) == QKV_NUM_DIMS, f"V is expected to have {QKV_NUM_DIMS} dimensions, but got {len(V.shape)}"
    assert K.shape[2] == V.shape[2], f"K, V sequence lengths must be identical but got {K.shape[2]}, {V.shape[2]}"
    assert scale is None or isinstance(scale, float), f"Expected scale to be of type float if provided, but got {type(scale)}"


def _validate_inputs_for_flash_attn_kernel(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_causal: bool,
    scale: Optional[float],
) -> None:
    """
    Validates inputs to flash attention kernel meet the kernel's constraints

    :param Q: Query vectors used for attention computation. Expected shape: (batch, num_heads, seq_len, head_dim)
    :param K: Key vectors used for attention computation. Expected shape: (batch, num_heads, seq_len, head_dim)
    :param V: Value vectors used for attention computation. Expected shape: (batch, num_heads, seq_len, head_dim)
    :param is_causal: Whether or not to compute causal attention
    :param scale: Scaling factor applied prior to softmax
    """
    MAX_NUM_HEADS = 128
    MIN_SEQ_LEN = 512
    QKV_NUM_DIMS = 4
    assert len(Q.shape) == QKV_NUM_DIMS, f"Q is expected to have {QKV_NUM_DIMS} dimensions, but got {len(Q.shape)}"
    assert len(K.shape) == QKV_NUM_DIMS, f"K is expected to have {QKV_NUM_DIMS} dimensions, but got {len(K.shape)}"
    assert len(V.shape) == QKV_NUM_DIMS, f"V is expected to have {QKV_NUM_DIMS} dimensions, but got {len(V.shape)}"
    if is_causal:
        assert (
            Q.shape[2] == K.shape[2] == V.shape[2]
        ), f"Q, K, V sequence lengths must be identical for causal attention kernel but got {Q.shape[2]}, {K.shape[2]}, {V.shape[2]}"
    assert Q.shape[1] <= MAX_NUM_HEADS, f"Num heads must be <= {MAX_NUM_HEADS}, but got {Q.shape[1]}"
    assert Q.shape[2] >= MIN_SEQ_LEN, f"Seq len must be >= {MIN_SEQ_LEN}, but got {Q.shape[2]}"
    assert K.shape[2] == V.shape[2], f"K, V sequence lengths must be identical but got {K.shape[2]}, {V.shape[2]}"
    assert scale is None or isinstance(scale, float), f"Expected scale to be of type float if provided, but got {type(scale)}"


def _transform_qkv_for_flash_attn_kernel(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    scale: Optional[float],
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Transforms Q, K, V tensors as required for invoking a flash attention kernel. Includes scaling of
    Q due to a compilation error when trying to scale within the kernel.

    :param Q: Query vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param K: Key vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param V: Value vectors used for attention computation
              Shape: (batch, num_heads, seq_len, head_dim)
    :param scale: Scaling factor applied prior to softmax. If None, the default value is 1 / sqrt(head_dim)

    :return: Q, K, V tensors that have been transformed for use by flash attention kernel. Notably, Q is scaled
             Shape (Q): (batch * num_heads, head_dim, seq_len)
             Shape (K): (batch * num_heads, head_dim, seq_len)
             Shape (V): (batch * num_heads, seq_len, head_dim)
    """
    bsz, num_heads, q_len, head_dim = Q.shape
    k_len = K.shape[2]
    scale = scale if scale is not None else 1 / math.sqrt(head_dim)

    Q = (
        Q.permute(0, 1, 3, 2)
        .reshape((bsz * num_heads, head_dim, q_len))
        # We scale Q here as the causal flash attention kernel runs into a compilation error when trying
        # to scale within the kernel
        * scale
    )
    K = (
        K.permute(0, 1, 3, 2)
        .reshape((bsz * num_heads, head_dim, k_len))
    )
    V = V.reshape((bsz * num_heads, k_len, head_dim))

    return Q, K, V


def _split_fused_qkv(QKV, n_heads, n_kv_heads, d_head) -> tuple[Tensor, Tensor, Tensor]:

    # shape of QKV is [batch, seqlen, fused_qkv_size]
    # we split the fused QKV (dim=2) into Q, K, V
    # for example:
    #   for 405B, TP=128, num_att_heads=128
    #   LNC=2/TP=64 will split QKV from [batch, seqlen, 512] into:
    #   Q [batch, seqlen, 256]
    #   K [batch, seqlen, 128]
    #   V [batch, seqlen, 128]
    # torch.split has accuracy issue and leads to more reshapes in hlo.
    # Using torch.tensor_split here. NAPP-3145
    q_end_index = n_heads * d_head
    k_end_index = q_end_index + n_kv_heads * d_head

    Q, K, V = torch.tensor_split(
        QKV,
        (
            q_end_index,
            k_end_index,
            # rest of the QKV will go to V output
        ),
        dim=2,
    )
    logger.debug(f"QKV shape before tensor_split: {QKV.shape}")
    logger.debug(f"Q shape after tensor_split: {Q.shape}")
    logger.debug(f"K shape after tensor_split: {K.shape}")
    logger.debug(f"V shape after tensor_split: {V.shape}")
    return Q, K, V


def _create_causal_attn_mask(source_seq_len, target_seq_len, device) -> Tensor:
    """
    Returns a causal attention mask of shape (source_seq_len, target_seq_len) on
    provided device
    """
    return torch.full(
        (source_seq_len, target_seq_len), True, device=device
    ).tril(diagonal=0).bool()
