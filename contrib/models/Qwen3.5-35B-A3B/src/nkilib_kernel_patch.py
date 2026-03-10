"""
Monkey-patch NxDI's flash attention kernel with our modified nkilib kernel.

NxDI uses `neuronxcc.nki._pre_prod_kernels.attn_fwd` which has head_dim <= 128.
Our modified nkilib `attention_cte` supports head_dim up to 256 via d-tiling.

This module:
1. Imports our modified `attention_cte` from nkilib (standalone override must be
   pip-installed from github.com/jimburtoft/nki-library branch feature/head-dim-256)
2. Wraps it in an adapter matching the NxDI calling convention
3. Applies the same decorator stack NxDI uses (peel, re-jit torchxla, skip_middle_end,
   enable_stack_allocator)
4. Replaces `attention_base._flash_fwd_call_nki` with the adapted kernel

Usage:
    from nkilib_kernel_patch import patch_flash_attention_kernel
    patch_flash_attention_kernel()

    # Now NxDI will use our modified nkilib kernel for flash attention,
    # supporting head_dim up to 256.

Prerequisites:
    pip install git+https://github.com/jimburtoft/nki-library.git@feature/head-dim-256

    This installs nkilib_src.nkilib which automatically overrides the bundled nkilib
    in neuronx-cc via the nkilib import mechanism.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def _build_adapted_kernel():
    """Build the adapted nkilib kernel with NxDI's decorator stack.

    The key challenge is that NxDI and nkilib use different parameter names:
      NxDI: do_out_tp, kernel_name, use_dma_transpose
      nkilib: tp_out, causal_mask, (no dma_transpose param)

    We peel the @nki.jit decorator from nkilib's attention_cte, write a thin
    wrapper that translates parameter names and calls the undecorated kernel
    body, then apply NxDI's full decorator stack to the wrapper. This produces
    a single NKI compilation unit (no nested jit).

    Returns the decorated kernel function, or None on failure.
    """
    from neuronx_distributed_inference.utils.decorator_peeling import peel_decorations
    from neuronxcc.nki import jit
    from neuronxcc.nki.compiler import (
        skip_middle_end_transformations,
        enable_stack_allocator,
    )
    from torch_neuronx.utils import get_platform_target
    import nkilib.core.attention.attention_cte as cte_module

    # Verify our modified kernel is loaded (not the bundled one)
    max_head_dim = getattr(cte_module, "_MAX_HEAD_DIM", None)
    if max_head_dim is None or max_head_dim <= 128:
        logger.warning(
            f"nkilib attention_cte._MAX_HEAD_DIM = {max_head_dim}. "
            "Expected > 128 from the modified nki-library fork. "
            "Install with: pip install git+https://github.com/jimburtoft/nki-library.git@feature/head-dim-256"
        )
        return None

    logger.info(
        f"nkilib attention_cte._MAX_HEAD_DIM = {max_head_dim} -- modified kernel detected"
    )

    # Peel the @nki.jit decorator from attention_cte to get the raw kernel function.
    # This is the undecorated function body that contains all the NKI primitives.
    attention_cte_raw = peel_decorations(cte_module.attention_cte)
    logger.info(f"Peeled attention_cte, raw function: {attention_cte_raw}")

    # Build the adapter wrapper around the raw (undecorated) kernel.
    # This wrapper translates NxDI's parameter convention to nkilib's convention,
    # then falls through to the raw kernel body. Since neither the wrapper nor the
    # inner function is @nki.jit-decorated, this is a single flat function from the
    # compiler's perspective.
    def nkilib_attention_adapter(
        q,
        k,
        v,
        scale,
        do_out_tp=True,
        tp_q=True,
        tp_k=False,
        use_dma_transpose=False,  # absorbed -- nkilib decides internally
        kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
        # Prefix caching params (passed through if present)
        k_prior=None,
        v_prior=None,
        prior_used_len=None,
        # Optional params (passed through)
        sink=None,
        sliding_window=None,
    ):
        """Adapter: translates NxDI calling convention -> nkilib attention_cte."""
        # Translate kernel_name -> causal_mask
        causal_mask = "Causal" in kernel_name

        # Translate do_out_tp -> tp_out
        tp_out = do_out_tp

        # Call the raw (undecorated) nkilib kernel body
        return attention_cte_raw(
            q=q,
            k=k,
            v=v,
            scale=scale,
            causal_mask=causal_mask,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            k_prior=k_prior,
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            sink=sink,
            sliding_window=sliding_window,
        )

    # Apply NxDI's decorator stack.
    # This matches what import_nki_cte_attention_kernel() does in attention_base.py:111-117:
    #   1. jit(mode='torchxla', platform_target=...) -- fixes platform detection bug
    #   2. skip_middle_end_transformations -- required for CTE kernels
    #   3. enable_stack_allocator -- required for CTE kernels
    platform_target = get_platform_target()
    logger.info(f"Applying NxDI decorator stack with platform_target={platform_target}")

    decorated = jit(
        nkilib_attention_adapter,
        mode="torchxla",
        platform_target=platform_target,
        show_compiler_tb=True,
        debug_kernel=True,
    )
    decorated = skip_middle_end_transformations(decorated)
    decorated = enable_stack_allocator(decorated, log_level=logging.INFO)

    return decorated


def patch_flash_attention_kernel():
    """Replace NxDI's flash attention kernel with our modified nkilib kernel.

    This monkey-patches attention_base._flash_fwd_call_nki and related globals.
    Must be called BEFORE model compilation (i.e., before the first forward pass).

    Returns True if the patch was applied, False otherwise.
    """
    global _patched

    if _patched:
        logger.info("Flash attention kernel already patched, skipping")
        return True

    try:
        adapted_kernel = _build_adapted_kernel()
    except Exception as e:
        logger.error(f"Failed to build adapted nkilib kernel: {e}", exc_info=True)
        return False

    if adapted_kernel is None:
        logger.warning("Could not build adapted kernel, patch not applied")
        return False

    # Monkey-patch the module-level globals in attention_base
    import neuronx_distributed_inference.modules.attention.attention_base as attn_base

    attn_base._flash_fwd_call_nki = adapted_kernel
    # Also patch the prefix caching variant (NxDI sets _flash_fwd_pc_call_nki = _flash_fwd_call_nki)
    attn_base._flash_fwd_pc_call_nki = adapted_kernel
    # Ensure _has_new_kernel is True so NxDI uses the return-value code path
    # (when _has_new_kernel=True, the kernel returns output directly instead of
    # writing to a pre-allocated tensor)
    attn_base._has_new_kernel = True

    _patched = True
    logger.info(
        "Successfully patched NxDI flash attention kernel with modified nkilib "
        f"(head_dim up to {getattr(cte_module_ref(), '_MAX_HEAD_DIM', '?')})"
    )
    return True


def cte_module_ref():
    """Lazy reference to the cte module for logging."""
    try:
        import nkilib.core.attention.attention_cte as m

        return m
    except Exception:
        return None


def is_patched():
    """Check if the flash attention kernel has been patched."""
    return _patched
