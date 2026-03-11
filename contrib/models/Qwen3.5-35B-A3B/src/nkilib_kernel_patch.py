"""
Prepare the modified nkilib flash attention kernel for direct invocation.

NxDI uses `neuronxcc.nki._pre_prod_kernels.attn_fwd` which has head_dim <= 128.
Our modified nkilib `attention_cte` supports head_dim up to 256 via d-tiling.

Instead of monkey-patching the global `_flash_fwd_call_nki`, this module provides
the kernel as a callable that `perform_prefill` invokes directly with `tp_out=False`.
This avoids the `tp_out=True` limitation for d>128 (which would require tiling the
output write-back path).

STATUS: BLOCKED by NKI compiler bug (SDK 2.28, neuronx-cc 2.23.x)
    The NKI compiler's TraceKernel.inline_function does not propagate the
    trace context ('builtin' injection) to sub-functions called from within
    the kernel. Since nkilib's attention_cte uses _attention_cte_impl as a
    sub-function, all nisa.* ISA operations inside it fail with:
        NameError: name 'builtin' is not defined

    The standalone nkilib kernel works correctly in baremetal/simulation mode
    (all unit tests pass). The issue is specific to torchxla mode tracing,
    which is required for NxDI integration.

    See: issue_nki_inline_function_builtin.md for full details.

Usage (when compiler bug is fixed):
    from nkilib_kernel_patch import get_nkilib_flash_attention_kernel

    kernel = get_nkilib_flash_attention_kernel()
    if kernel is not None:
        output = kernel(q, k, v, scale, causal_mask=True, tp_q=True, tp_k=True, tp_out=False)
        # Output shape with tp_out=False: (B*H, seqlen, d) -- BHSD when reshaped

Prerequisites:
    pip install git+https://github.com/jimburtoft/nki-library.git@feature/head-dim-256
"""

import logging

logger = logging.getLogger(__name__)

_kernel = None
_kernel_built = False
_max_head_dim = None


def _build_kernel():
    """Build the nkilib attention kernel wrapped for PyTorch XLA execution.

    Uses torch_neuronx.nki_jit() to wrap the nkilib GenericKernel directly.

    CURRENTLY BLOCKED: Returns None because the NKI compiler's sub-function
    inlining doesn't propagate trace context in torchxla mode. When the
    compiler bug is fixed, remove the early return below.

    Returns the decorated kernel function, or None if not available.
    """
    global _max_head_dim

    from torch_neuronx import nki_jit
    import nkilib.core.attention.attention_cte as cte_module

    # Verify our modified kernel is loaded (not the bundled one)
    _max_head_dim = getattr(cte_module, "_MAX_HEAD_DIM", None)
    if _max_head_dim is None or _max_head_dim <= 128:
        logger.warning(
            f"nkilib attention_cte._MAX_HEAD_DIM = {_max_head_dim}. "
            "Expected > 128 from the modified nki-library fork. "
            "Install with: pip install git+https://github.com/jimburtoft/nki-library.git@feature/head-dim-256"
        )
        return None

    logger.info(
        f"nkilib attention_cte._MAX_HEAD_DIM = {_max_head_dim} -- modified kernel detected"
    )

    # BLOCKED: NKI compiler bug -- sub-function inlining doesn't propagate
    # trace context. nisa.* calls in _attention_cte_impl fail with:
    #   NameError: name 'builtin' is not defined
    # When this is fixed in neuronx-cc, remove this early return.
    logger.warning(
        "nkilib flash attention kernel available but DISABLED due to NKI compiler bug "
        "(sub-function inlining doesn't propagate trace context in torchxla mode). "
        "Using PyTorch softmax fallback for head_dim > 128."
    )
    return None

    # --- Code below is ready to use once the compiler bug is fixed ---
    # Wrap the GenericKernel with nki_jit for PyTorch XLA execution.
    attention_cte = cte_module.attention_cte
    logger.info(
        f"Wrapping attention_cte ({type(attention_cte).__name__}) with nki_jit()"
    )
    decorated = nki_jit()(attention_cte)
    logger.info(f"Wrapped kernel type: {type(decorated).__name__}")
    return decorated


def get_nkilib_flash_attention_kernel():
    """Get the prepared nkilib flash attention kernel.

    Returns the decorated kernel function, or None if not available.

    IMPORTANT: For head_dim > 128, use tp_out=False. The kernel does not
    support tp_out=True for d > 128 (would require tiling the output path).

    Returns None if:
    - nkilib standalone is not installed
    - nkilib _MAX_HEAD_DIM <= 128 (bundled version)
    - NKI compiler bug blocks torchxla integration (current state)
    - Build fails for any reason
    """
    global _kernel, _kernel_built

    if _kernel_built:
        return _kernel

    try:
        _kernel = _build_kernel()
    except Exception as e:
        logger.error(
            f"Failed to build nkilib flash attention kernel: {e}", exc_info=True
        )
        _kernel = None

    _kernel_built = True
    return _kernel


def get_max_head_dim():
    """Get the maximum head_dim supported by the loaded nkilib kernel."""
    return _max_head_dim


def is_available():
    """Check if the modified nkilib kernel is available."""
    return get_nkilib_flash_attention_kernel() is not None
