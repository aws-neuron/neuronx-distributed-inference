import torch
import logging
import torch.nn.functional as F
from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_kernel
from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

logger = logging.getLogger("Neuron")


# -- public functions at the top
def qkv_kernel(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    bias: torch.Tensor = None,
    rmsnorm=None,
    rms_norm_eps: float = 1e-6,
    fused_rmsnorm_skip_gamma: bool = False,
    logical_nc_config: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Computes Query, Key, Value projections using NKI kernel with optional fused RMSNorm.

    :param hidden_states: Input hidden states tensor
                         Shape: (batch_size, seq_len, hidden_dim)
    :param w_qkv: Pre-transposed QKV projection weight matrix.
                  Shape: (hidden_dim, fused_qkv_size)
                  where fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    :param num_attention_heads: Number of attention heads (per TP rank)
    :param num_key_value_heads: Number of key/value heads (per TP rank, for GQA/MQA)
    :param head_dim: Dimension of each attention head
    :param bias: Optional bias tensor for QKV projection
                Shape: (fused_qkv_size,)
                Default: None
    :param rmsnorm: Optional RMSNorm module with weight attribute for fused normalization.
                   If provided, fused RMSNorm is enabled. If None, no normalization is applied.
                   rmsnorm.weight shape: (hidden_dim,)
                   Default: None
    :param rms_norm_eps: Epsilon value for RMSNorm computation
                        Default: 1e-6
    :param fused_rmsnorm_skip_gamma: Whether to skip gamma multiplication in RMSNorm
                                    Default: False
    :param logical_nc_config: Logical NeuronCore configuration for kernel grid
                             Default: 1

    :return: Tuple of Query, Key, Value tensors
            Q shape: (batch_size, seq_len, num_attention_heads * head_dim)
            K shape: (batch_size, seq_len, num_key_value_heads * head_dim)
            V shape: (batch_size, seq_len, num_key_value_heads * head_dim)
    """
    # TODO: Add assertions for cases when this kernel cannot be used. Reach out to NKI team for help.
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Validate weight matrix dimensions
    weight_hidden_dim, fused_qkv_size = w_qkv.shape
    assert hidden_dim == weight_hidden_dim, f"{hidden_dim=}, {weight_hidden_dim=}"

    # Pad sequence length to even number for CTE kernel
    # CTE kernel is used when batch_size * seq_len > 64 and requires even sequence length
    # for optimal memory access patterns and vectorization
    padded_seq_len = seq_len
    if batch_size * seq_len > 64 and seq_len % 2 != 0:
        hidden_states = F.pad(hidden_states, pad=(0, 0, 0, 1, 0, 0), value=1.0)
        padded_seq_len = seq_len + 1

    fused_rmsnorm_enabled = rmsnorm is not None
    norm_weights = None
    if fused_rmsnorm_enabled:
        norm_weights = rmsnorm.weight.unsqueeze(0)
        assert norm_weights.shape == (1, hidden_dim)

    grid = (nc(logical_nc_config),)

    qkv_output = torch.zeros(batch_size, padded_seq_len, fused_qkv_size, dtype=hidden_states.dtype, device=hidden_states.device)
    nki_jit()(rmsnorm_qkv_isa_kernel)[grid](
        hidden_states,
        w_qkv,
        norm_weights if norm_weights is not None else torch.ones((1, hidden_dim), device=hidden_states.device),
        qkv_output,
        kernel_name="QKV",
        eps=rms_norm_eps,
        fused_rmsnorm=fused_rmsnorm_enabled,
        bias=bias.unsqueeze(0) if bias is not None else None,
        skip_gamma=fused_rmsnorm_skip_gamma,
    )

    # Remove padding if it was added
    if padded_seq_len != seq_len:
        qkv_output = qkv_output[:, :-1, :]
    assert qkv_output.shape == (batch_size, seq_len, fused_qkv_size)

    Q, K, V = _split_fused_qkv(qkv_output, num_attention_heads, head_dim, num_key_value_heads)

    return Q, K, V


# -- private functions go below

def _split_fused_qkv(QKV, num_attention_heads, head_dim, num_key_value_heads):
    r"""
    Splits fused QKV tensor into separate Query, Key, Value tensors.

    :param QKV: Fused QKV tensor from kernel output
               Shape: (batch_size, seq_len, fused_qkv_size)
               where fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    :param num_attention_heads: Number of attention heads (per TP rank)
    :param head_dim: Dimension of each attention head
    :param num_key_value_heads: Number of key/value heads (per TP rank, for GQA/MQA)

    :return: Tuple of Query, Key, Value tensors
            Q shape: (batch_size, seq_len, num_attention_heads * head_dim)
            K shape: (batch_size, seq_len, num_key_value_heads * head_dim)
            V shape: (batch_size, seq_len, num_key_value_heads * head_dim)
    """
    logger.debug(f"Fused QKV tensor has shape {QKV.shape}")
    q_end_index = num_attention_heads * head_dim
    k_end_index = q_end_index + num_key_value_heads * head_dim
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
