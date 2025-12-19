import logging
from typing import Optional, Tuple

import torch

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


def tokengen_attention_megakernel_standard_kv(
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
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    k_cache_transposed: bool = False,
    update_cache_in_kernel: bool = True,
    fused_rmsnorm: bool = True,
    skip_rope: bool = False,
    use_qk_norm: bool = False,
    qk_norm_eps: float = 1e-6,
    rope_first_second_half_impl: bool = True,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""
    Token generation attention computation using NKI kernel with standard contiguous KV cache management.

    .. note::
        Implementation details:
        The function calls the ``llama3_nki_attention_block_token_gen_kernel`` which runs with
        Logical NC Config = 2. The kernel performs fused operations including RMSNorm, QKV projection,
        RoPE application, attention computation, and output projection in a single optimized kernel.
        Cache updates can be performed either within the kernel or externally based on ``update_cache_in_kernel``.

    .. note::
        This implementation uses standard contiguous KV caching where the entire cache is allocated
        as contiguous memory blocks. This approach is simpler than block-based caching but may be
        less memory-efficient for highly variable sequence lengths.

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
    :param K_cache: Contiguous Key cache for storing past key states.
                    Shape: ``(batch_size, num_kv_heads, max_context_length, head_dim)`` when ``k_cache_transposed=False``
                    or ``(batch_size, num_kv_heads, head_dim, max_context_length)`` when ``k_cache_transposed=True``
                    Stores key vectors for all past tokens in a contiguous memory layout.
    :param V_cache: Contiguous Value cache for storing past value states.
                    Shape: ``(batch_size, num_kv_heads, max_context_length, head_dim)``
                    Stores value vectors for all past tokens in a contiguous memory layout.
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
    :param cos_cache: Precomputed cosine values for RoPE (Rotary Position Embedding).
                     Shape: ``(head_dim // 2, batch_size, sequence_length)``
                     If None, zero tensors are used (effectively disabling RoPE).
    :param sin_cache: Precomputed sine values for RoPE (Rotary Position Embedding).
                     Shape: ``(head_dim // 2, batch_size, sequence_length)``
                     If None, zero tensors are used (effectively disabling RoPE).
    :param k_cache_transposed: Whether the K_cache is stored in transposed format.
                              When True, K_cache shape is ``(batch_size, num_kv_heads, head_dim, max_context_length)``
                              When False, K_cache shape is ``(batch_size, num_kv_heads, max_context_length, head_dim)``
                              Affects memory access patterns and kernel optimization.
    :param update_cache_in_kernel: Whether to update the KV cache within the NKI kernel.
                                  When True, cache updates are performed efficiently within the kernel.
                                  When False, cache updates are handled externally with additional tensor operations.
    :param fused_rmsnorm: Whether to apply RMSNorm within the kernel before attention computation.
                         When True, uses ``W_gamma`` and ``rmsnorm_eps`` for normalization.
                         When False, assumes input is already normalized.
    :param skip_rope: Whether to skip RoPE (Rotary Position Embedding) application.
                     When True, position encoding is not applied to Q and K projections.
                     When False, uses ``cos_cache`` and ``sin_cache`` for position encoding.
    :param use_qk_norm: Whether to apply normalization to Query and Key projections before attention.
    :param qk_norm_eps: Epsilon value for Q/K normalization when ``use_qk_norm=True``.
    :param rope_first_second_half_impl: Implementation variant for RoPE application.
                                       Controls how the rotation is applied to the first and second
                                       halves of the head dimension.

    :return: Tuple containing:
             - ``attn_output``: Attention output tensor with shape ``(batch_size, sequence_length, hidden_size)``
             - ``(K_cache, V_cache)``: Updated cache tensors with new key/value states appended
             - ``cos_cache``: Cosine cache tensor (same as input or newly created if input was None)
             - ``sin_cache``: Sine cache tensor (same as input or newly created if input was None)
    """
    LOGICAL_NC_CONFIG = 2  # Underlying NKI kernel runs with LNC=2

    # Check if NKI kernel is available
    _check_nki_kernel_availability()

    # Validate inputs for standard implementation
    _validate_standard_kv_inputs(
        hidden_states, W_qkv, W_out, W_gamma, K_cache, V_cache,
        attention_mask, active_mask, position_ids, cos_cache, sin_cache,
        head_dim, num_heads, num_kv_heads, k_cache_transposed
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

    # Prepare cache tensors
    if update_cache_in_kernel:
        K = K_cache
        V = V_cache
    else:
        K = torch.zeros(head_dim, bsz, q_len, dtype=hidden_states.dtype, device=hidden_states.device)
        V = torch.zeros(bsz, q_len, head_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    # Call the NKI kernel
    grid = (nc(LOGICAL_NC_CONFIG),)
    try:
        attn_output, K, V = llama3_nki_attention_block_token_gen_kernel[grid](
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
            update_cache=update_cache_in_kernel,
            active_blocks_table=None,
            K_cache_transposed=k_cache_transposed,
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

    # Process cache outputs if not updated in kernel
    if not update_cache_in_kernel:
        K = K.permute(1, 0, 2) if k_cache_transposed else K.permute(1, 2, 0)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

    return attn_output, (K, V), cos_cache, sin_cache


def _check_nki_kernel_availability():
    """Check if NKI kernel is available."""
    if llama3_nki_attention_block_token_gen_kernel is None:
        raise RuntimeError(
            "NKI attention block token generation kernel is not available. "
            "Please use a more recent neuron compiler version."
        )


def _validate_common_shapes(
    hidden_states: torch.Tensor,
    W_qkv: torch.Tensor,
    W_out: torch.Tensor,
    W_gamma: torch.Tensor,
    attention_mask: torch.Tensor,
    active_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cos_cache: Optional[torch.Tensor],
    sin_cache: Optional[torch.Tensor],
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[int, int, int]:
    """
    Validate common shapes and dimensions

    Args:
        Same subset of arguments used by both validation functions

    Returns:
        Tuple[int, int, int]: batch_size, sequence_length, hidden_size
    """
    # Basic dimension constants
    HIDDEN_STATES_DIMS = 3
    ATTENTION_MASK_DIMS = 4

    # Validate hidden states
    assert len(hidden_states.shape) == HIDDEN_STATES_DIMS, (
        f"hidden_states should be {HIDDEN_STATES_DIMS}D, got {len(hidden_states.shape)}D"
    )
    bsz, q_len, h = hidden_states.shape

    # Validate weights
    assert W_qkv.shape == (h, (num_heads + 2) * head_dim), (
        f"W_qkv shape {W_qkv.shape} != ({h}, {(num_heads + 2) * head_dim})"
    )
    assert W_out.shape == (num_heads * head_dim, h), (
        f"W_out shape {W_out.shape} != ({num_heads * head_dim}, {h})"
    )
    assert W_gamma.shape == (1, h), f"W_gamma shape {W_gamma.shape} != (1, {h})"

    # Validate position IDs
    assert position_ids.shape == (bsz, q_len), (
        f"position_ids shape {position_ids.shape} != ({bsz}, {q_len})"
    )

    # Validate attention masks
    assert len(attention_mask.shape) == ATTENTION_MASK_DIMS, (
        f"attention_mask should be {ATTENTION_MASK_DIMS}D, got {len(attention_mask.shape)}D"
    )
    if active_mask is not None:
        assert len(active_mask.shape) == ATTENTION_MASK_DIMS, (
            f"active_mask should be {ATTENTION_MASK_DIMS}D, got {len(active_mask.shape)}D"
        )

    # Validate RoPE coefficients if provided
    _validate_rope_coefficients(cos_cache, sin_cache, head_dim, bsz, q_len)

    # Validate dimensions
    _validate_attention_dimensions(head_dim, num_heads, num_kv_heads)

    return bsz, q_len, h


def _validate_rope_coefficients(
    cos_cache: Optional[torch.Tensor],
    sin_cache: Optional[torch.Tensor],
    head_dim: int,
    batch_size: int,
    seq_len: int,
) -> None:
    """
    Validate RoPE coefficient tensors.
    """
    ROPE_COEFF_DIVISOR = 2

    if cos_cache is not None:
        expected_rope_shape = (head_dim // ROPE_COEFF_DIVISOR, batch_size, seq_len)
        assert cos_cache.shape == expected_rope_shape, (
            f"cos_cache shape {cos_cache.shape} != {expected_rope_shape}"
        )

    if sin_cache is not None:
        expected_rope_shape = (head_dim // ROPE_COEFF_DIVISOR, batch_size, seq_len)
        assert sin_cache.shape == expected_rope_shape, (
            f"sin_cache shape {sin_cache.shape} != {expected_rope_shape}"
        )


def _validate_attention_dimensions(head_dim: int, num_heads: int, num_kv_heads: int) -> None:
    """
    Validate attention-related dimensions.
    """
    assert head_dim > 0, f"head_dim must be positive, got {head_dim}"
    assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
    assert num_kv_heads > 0, f"num_kv_heads must be positive, got {num_kv_heads}"
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )


def _validate_standard_kv_inputs(
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
    k_cache_transposed: bool,
) -> None:
    """
    Validate input tensors for standard KV cache implementation.
    """
    # Validate common shapes and dimensions
    bsz, q_len, h = _validate_common_shapes(
        hidden_states, W_qkv, W_out, W_gamma,
        attention_mask, active_mask, position_ids,
        cos_cache, sin_cache, head_dim, num_heads, num_kv_heads
    )

    # Validate standard KV cache specific requirements
    s_max_ctx = V_cache.shape[2]
    expected_k_cache_shape = (
        (bsz, num_kv_heads, head_dim, s_max_ctx)
        if k_cache_transposed
        else (bsz, num_kv_heads, s_max_ctx, head_dim)
    )
    assert K_cache.shape == expected_k_cache_shape, (
        f"Expected K cache shape: {expected_k_cache_shape}, got {K_cache.shape}"
    )


def _prepare_rope_coefficients(
    cos_cache: Optional[torch.Tensor],
    sin_cache: Optional[torch.Tensor],
    head_dim: int,
    bsz: int,
    q_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare RoPE coefficients.
    """
    if cos_cache is None or sin_cache is None:
        logger.warning("cos_cache or sin_cache not provided, using zero tensors")
        expected_rope_coeff_shape = (head_dim // 2, bsz, q_len)
        cos_cache = torch.zeros(expected_rope_coeff_shape, dtype=dtype, device=device)
        sin_cache = torch.zeros(expected_rope_coeff_shape, dtype=dtype, device=device)

    return cos_cache, sin_cache


def _prepare_attention_mask(attention_mask, bsz, num_heads, q_len):
    """
    Helper function to prepare attention mask.
    """
    s_prior = attention_mask.shape[-1]
    expected_cache_mask_shape = [(bsz, 1, q_len, s_prior), (bsz, num_heads, q_len, s_prior)]
    assert attention_mask.shape in expected_cache_mask_shape

    return attention_mask.expand(-1, num_heads, -1, -1)


def _prepare_active_mask(active_mask, bsz, num_heads, q_len, hidden_states):
    """
    Helper function to prepare active mask.
    """
    expected_active_mask_shape = (bsz, 1, q_len, q_len)

    if q_len == 1:
        active_mask = torch.ones(expected_active_mask_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    else:
        assert active_mask.shape == expected_active_mask_shape

    return active_mask.expand(-1, num_heads, -1, -1)
