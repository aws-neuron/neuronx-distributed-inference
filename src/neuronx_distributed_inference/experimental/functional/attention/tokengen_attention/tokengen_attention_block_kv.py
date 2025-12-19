import logging
from typing import Optional, Tuple

import torch

from neuronx_distributed_inference.experimental.functional.attention.tokengen_attention.tokengen_attention_standard_kv import (
    _check_nki_kernel_availability,
    _prepare_rope_coefficients,
    _prepare_attention_mask,
    _prepare_active_mask,
    _validate_common_shapes,
)

logger = logging.getLogger("Neuron")

# Import NKI kernel and related utilities
try:
    from neuronxcc.nki._pre_prod_kernels.attention_token_gen import llama3_nki_attention_block_token_gen_kernel
    from neuronxcc.nki.language import nc
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable token-gen attention block NKI kernel"
    )
    llama3_nki_attention_block_token_gen_kernel = None


def tokengen_attention_megakernel_block_kv(
    hidden_states: torch.Tensor,
    W_qkv: torch.Tensor,
    W_out: torch.Tensor,
    W_gamma: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    active_mask: torch.Tensor,
    position_ids: torch.Tensor,
    rmsnorm_eps: float,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    active_block_table: torch.Tensor,
    paged_attention_block_size: int,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    fused_rmsnorm: bool = True,
    skip_rope: bool = False,
    use_qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    rope_first_second_half_impl: bool = True,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""
    Token generation attention computation using NKI kernel with block-based KV cache management.

    .. note::
        Implementation details:
        The function calls the ``llama3_nki_attention_block_token_gen_kernel`` which runs with
        Logical NC Config = 2. The kernel performs fused operations including RMSNorm, QKV projection,
        RoPE application, attention computation, and output projection in a single optimized kernel.

    .. note::
        This implementation uses block-based KV caching which organizes the cache into fixed-size blocks
        rather than contiguous memory. This approach is more memory-efficient for variable-length sequences
        and supports paged attention patterns.

    :param hidden_states: Input hidden states from the transformer layer.
                         Shape: ``(batch_size, sequence_length, hidden_size)``
                         These are the token embeddings that will be transformed into Q, K, V projections.
    :param W_qkv: Combined weight matrix for Query, Key, and Value projections.
                  Shape: ``(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim)``
                  Contains concatenated weights for Q (num_heads * head_dim), K (num_kv_heads * head_dim),
                  and V (num_kv_heads * head_dim) projections.
    :param W_out: Output projection weight matrix applied after attention computation.
                  Shape: ``(num_heads * head_dim, hidden_size)``
                  Projects the concatenated attention output back to hidden_size.
    :param W_gamma: RMSNorm scaling parameter (gamma) applied to hidden states before attention.
                    Shape: ``(1, hidden_size)``
                    Used in the fused RMSNorm operation when ``fused_rmsnorm=True``.
    :param K_cache: Block-organized Key cache for storing past key states.
                    Shape: ``(total_blocks, paged_attention_block_size, num_kv_heads, head_dim)``
                    Each block stores key vectors for ``paged_attention_block_size`` tokens.
                    The ``active_block_table`` determines which blocks are used for each sequence.
    :param V_cache: Block-organized Value cache for storing past value states.
                    Shape: ``(total_blocks, paged_attention_block_size, num_kv_heads, head_dim)``
                    Organized identically to K_cache for efficient block-based access patterns.
    :param attention_mask: Causal attention mask controlling which positions can attend to each other.
                          Shape: ``(batch_size, 1, sequence_length, max_context_length)`` or
                          ``(batch_size, num_heads, sequence_length, max_context_length)``
                          Values should be 0 for positions that can be attended to, -inf for masked positions.
    :param active_mask: Mask indicating which positions in the current sequence are active.
                       Shape: ``(batch_size, 1, sequence_length, sequence_length)``
                       Used to mask out padding tokens within the current generation step.
                       For single token generation (sequence_length=1), this is automatically set to all ones.
    :param position_ids: Position indices for each token in the sequence.
                        Shape: ``(batch_size, sequence_length)``
                        Used for RoPE (Rotary Position Embedding) computation to encode positional information.
    :param rmsnorm_eps: Epsilon value for RMSNorm computation to prevent division by zero.
                       Typically set to 1e-6 or 1e-5. Used in the fused RMSNorm operation.
    :param head_dim: Dimension of each attention head. Must be consistent across Q, K, V projections.
                    Common values are 64, 80, 96, 128 depending on model architecture.
    :param num_heads: Number of attention heads for Query projection.
                     Determines the parallelism in attention computation.
    :param num_kv_heads: Number of attention heads for Key and Value projections.
                        For Grouped Query Attention (GQA), this is typically less than ``num_heads``.
                        Must satisfy: ``num_heads % num_kv_heads == 0``.
                        For block KV implementation, currently requires ``num_kv_heads == 1``.
    :param active_block_table: Table mapping sequence positions to cache block indices.
                              Shape varies based on implementation but typically
                              ``(batch_size, max_blocks_per_sequence)``
                              Determines which blocks in the cache are used for each sequence.
    :param paged_attention_block_size: Number of tokens stored in each cache block.
                                      Common values are 16, 32, 64. Affects memory access patterns
                                      and should be chosen based on hardware characteristics.
    :param cos_cache: Precomputed cosine values for RoPE (Rotary Position Embedding).
                     Shape: ``(head_dim // 2, batch_size, sequence_length)``
                     If None, zero tensors are used (effectively disabling RoPE).
    :param sin_cache: Precomputed sine values for RoPE (Rotary Position Embedding).
                     Shape: ``(head_dim // 2, batch_size, sequence_length)``
                     If None, zero tensors are used (effectively disabling RoPE).
    :param fused_rmsnorm: Whether to apply RMSNorm within the kernel before attention computation.
                         When True, uses ``W_gamma`` and ``rmsnorm_eps`` for normalization.
                         When False, assumes input is already normalized.
    :param skip_rope: Whether to skip RoPE (Rotary Position Embedding) application.
                     When True, position encoding is not applied to Q and K projections.
                     When False, uses ``cos_cache`` and ``sin_cache`` for position encoding.
    :param use_qk_norm: Whether to apply normalization to Query and Key projections before attention.
                       Helps with training stability in some model architectures.
    :param qk_norm_eps: Epsilon value for Q/K normalization when ``use_qk_norm=True``.
                       Prevents division by zero in the normalization computation.
    :param rope_first_second_half_impl: Implementation variant for RoPE application.
                                       Controls how the rotation is applied to the first and second
                                       halves of the head dimension.

    :return: Tuple containing:
             - ``attn_output``: Attention output tensor with shape ``(batch_size, sequence_length, hidden_size)``
             - ``(K_cache, V_cache)``: Updated cache tensors with new key/value states stored in appropriate blocks
             - ``cos_cache``: Cosine cache tensor (same as input or newly created if input was None)
             - ``sin_cache``: Sine cache tensor (same as input or newly created if input was None)
    """
    LOGICAL_NC_CONFIG = 2  # Underlying NKI kernel runs with LNC=2

    # Check if NKI kernel is available
    _check_nki_kernel_availability()

    # Validate inputs for blocked KV implementation
    _validate_block_kv_inputs(
        hidden_states, W_qkv, W_out, W_gamma, K_cache, V_cache,
        attention_mask, active_mask, position_ids, cos_cache, sin_cache,
        head_dim, num_heads, num_kv_heads, paged_attention_block_size
    )

    bsz, q_len, h = hidden_states.size()

    # Prepare RoPE coefficients
    cos_cache, sin_cache = _prepare_rope_coefficients(
        cos_cache, sin_cache, head_dim, bsz, q_len,
        hidden_states.dtype, hidden_states.device
    )

    # Prepare attention masks
    attention_mask = _prepare_attention_mask(attention_mask, bsz, num_heads, q_len)
    active_mask = _prepare_active_mask(active_mask, bsz, num_heads, q_len, hidden_states)

    # Initialize output tensor
    attn_output = torch.zeros(
        head_dim, bsz, num_heads * q_len, dtype=hidden_states.dtype, device=hidden_states.device
    )

    # Call the NKI kernel
    grid = (nc(LOGICAL_NC_CONFIG),)
    try:
        attn_output, K_cache, V_cache = llama3_nki_attention_block_token_gen_kernel[grid](
            X=hidden_states,
            W_qkv=W_qkv,
            W_gamma=W_gamma,
            rmsnorm_eps=rmsnorm_eps,
            cos=cos_cache,
            sin=sin_cache,
            W_out=W_out,
            K_cache=K_cache,
            V_cache=V_cache,
            mask_cache=attention_mask,
            mask_active=active_mask,
            position_ids=position_ids.to(torch.int32),
            update_cache=True,
            active_blocks_table=active_block_table,
            K_cache_transposed=False,
            fused_rmsnorm=fused_rmsnorm,
            skip_rope=skip_rope,
            rope_first_second_half_impl=rope_first_second_half_impl,
            qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
    except Exception as e:
        logger.error(f"NKI kernel execution failed: {e}")
        raise RuntimeError(f"Failed to execute NKI attention kernel: {e}")

    attn_output = attn_output.reshape((bsz, q_len, h))

    return attn_output, (K_cache, V_cache), cos_cache, sin_cache


def _validate_block_kv_inputs(
    hidden_states: torch.Tensor,
    W_qkv: torch.Tensor,
    W_out: torch.Tensor,
    W_gamma: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    attention_mask: torch.Tensor,
    active_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cos_cache: Optional[torch.Tensor],
    sin_cache: Optional[torch.Tensor],
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    paged_attention_block_size: int,
) -> None:
    """
    Validate input tensors for blocked KV cache implementation.
    """
    # Validate common shapes and dimensions
    bsz, q_len, h = _validate_common_shapes(
        hidden_states, W_qkv, W_out, W_gamma,
        attention_mask, active_mask, position_ids,
        cos_cache, sin_cache, head_dim, num_heads, num_kv_heads
    )

    # Validate block KV cache specific requirements
    total_blocks = K_cache.shape[0]
    expected_cache_shape = (total_blocks, paged_attention_block_size, num_kv_heads, head_dim)
    assert K_cache.shape == expected_cache_shape, (
        f'K_cache shape mismatch: {K_cache.shape} vs {expected_cache_shape}'
    )
    assert V_cache.shape == expected_cache_shape, (
        f'V_cache shape mismatch: {V_cache.shape} vs {expected_cache_shape}'
    )
    assert num_kv_heads == 1, "Blocked KV cache implementation requires num_kv_heads == 1"
