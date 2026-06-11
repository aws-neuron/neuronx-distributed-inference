"""
Monkeypatch: Native multi-KV-head NKI kernel adapter for NxDI.

Replaces NxDI's `NeuronAttentionBase.attention_block_tokengen_nki_kernel` dispatch
method to call our modified nki-library kernel (attention_block_tkg_multi_kv)
instead of the bundled `llama3_nki_attention_block_token_gen_kernel`.

The bundled kernel hardcodes kv_heads=1 per rank, and the per-group monkeypatch
(patch_attn_block_multi_kv.py) hits a compiler ICE (NCC_ITEN404) at S_ctx >= 512.

This adapter translates NxDI's calling convention to the nki-library kernel's
interface, handling:
  - Mask merging: mask_cache + mask_active -> single attention_mask in NKI layout
  - Parameter renaming (W_gamma -> rmsnorm_X_gamma, etc.)
  - Passing n_kv_heads for multi-KV-head support
  - Default values for nki-library-only parameters (quantization=NONE, etc.)

For kv_heads_per_rank == 1, the patch is a no-op passthrough to the original method.

Usage:
    from . import patch_native_multi_kv
    patch_native_multi_kv.apply_patch()
"""

import logging
import torch

logger = logging.getLogger(__name__)

_original_method = None
_patched = False


def _patched_attention_block_tokengen_nki_kernel(
    self,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value,
    active_mask,
    cos_cache,
    sin_cache,
    rmsnorm,
    rotary_position_ids,
    update_kv_per_layer,
    active_block_table,
    use_polar_compatible_rope=False,
):
    """
    Native multi-KV-head dispatch using the nki-library kernel fork.

    When kv_heads_per_rank == 1, delegates to the original NxDI method.
    When kv_heads_per_rank > 1, calls our attention_block_tkg kernel with
    n_kv_heads parameter, handling the interface translation.
    """
    kv_heads = self.num_key_value_heads
    q_heads = self.num_heads

    # Fast path: kv_heads=1 per rank, use original unmodified method
    if kv_heads == 1:
        return _original_method(
            self,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            active_mask,
            cos_cache,
            sin_cache,
            rmsnorm,
            rotary_position_ids,
            update_kv_per_layer,
            active_block_table,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

    # --- Multi-KV-head path: use nki-library kernel with n_kv_heads ---
    assert q_heads % kv_heads == 0, (
        f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})"
    )

    # Sequence parallel gather (same as original method)
    if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
        from neuronx_distributed.parallel_layers.mappings import (
            gather_from_sequence_parallel_region,
        )

        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            self.sequence_dimension,
            process_group=self.tensor_model_parallel_group,
        )

    from .attention_block_tkg_multi_kv import attention_block_tkg

    bsz, q_len, h = hidden_states.size()

    # ---- RoPE cos/sin preparation (same as original NxDI method) ----
    skip_rope = False

    if self.rotary_emb is not None:
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)
            cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
            sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)
    elif use_polar_compatible_rope:
        from neuronx_distributed_inference.modules.attention.attention_base import (
            precompute_freqs_cis,
        )

        rotary_freqs = precompute_freqs_cis(
            self.head_dim,
            self.neuron_config.max_context_length * 2,
            self.rope_theta,
            self.use_scaled_rope,
            device=hidden_states.device,
        )
        rotary_freqs = rotary_freqs[position_ids]
        cos_cache = rotary_freqs.cos().permute(2, 0, 1)
        sin_cache = rotary_freqs.sin().permute(2, 0, 1)
    else:
        expected_rope_coeff_shape = (self.head_dim // 2, bsz, q_len)
        cos_cache = torch.zeros(expected_rope_coeff_shape).to(hidden_states)
        sin_cache = torch.zeros(expected_rope_coeff_shape).to(hidden_states)
        skip_rope = True

    # ---- KV Cache ----
    # .data unwraps PlaceholderParameter -> plain Tensor for NKI tracer compatibility
    K_prior = past_key_value[0].data
    V_prior = past_key_value[1].data

    the_dtype = hidden_states.dtype
    the_device = hidden_states.device

    # ---- Mask preparation ----
    # NxDI provides:
    #   attention_mask (mask_cache): [bsz, 1, q_len, s_prior] or [bsz, q_heads, q_len, s_prior]
    #   active_mask: [bsz, 1, q_len, q_len]
    # Our nki-library kernel expects:
    #   attention_mask: [S_ctx, B, q_heads, S_tkg] (NKI layout)
    # where S_ctx = s_prior and S_tkg = q_len

    s_prior = attention_mask.shape[-1]

    # Expand to full q_heads if needed
    attention_mask = attention_mask.expand(-1, q_heads, -1, -1).contiguous()

    expected_active_mask_shape = (bsz, 1, q_len, q_len)
    if q_len == 1:
        active_mask = torch.ones(
            expected_active_mask_shape, dtype=the_dtype, device=the_device
        )
    active_mask = active_mask.expand(-1, q_heads, -1, -1).contiguous()

    # Merge: overwrite the last q_len positions of s_prior with active_mask
    attention_mask[:, :, :, -q_len:] = active_mask

    # Transpose to NKI layout: [bsz, q_heads, q_len, s_prior] -> [s_prior, bsz, q_heads, q_len]
    attention_mask_nki = attention_mask.permute(3, 0, 1, 2).contiguous()

    # Create per-group mask for multi-KV-head attention (virtual batch approach)
    # For the virtual batch approach, attention_tkg is called ONCE with
    # B_virt = bsz * kv_heads, q_head = q_per_kv_group.
    # Mask shape: [s_prior, B_virt, q_per_kv_group, q_len]
    # All heads share the same causal mask, so expand bsz -> B_virt by repeating.
    q_per_kv_group = q_heads // kv_heads
    group_attention_mask = attention_mask_nki[:, :, :q_per_kv_group, :].contiguous()
    # group_attention_mask: [s_prior, bsz, q_per_kv_group, q_len]
    # Expand to [s_prior, bsz * kv_heads, q_per_kv_group, q_len]
    group_attention_mask = group_attention_mask.repeat(1, kv_heads, 1, 1).contiguous()

    # ---- Weights ----
    # CRITICAL: Use .data to unwrap Parameter/PlaceholderParameter into plain Tensor.
    # The NKI tracer cannot access .shape on Parameter objects — it treats them as "none"
    # in the AST. All tensors passed to @nki.jit must be plain Tensor, not Parameter.
    W_qkv = self.get_qkv_proj().Wqkv.weight.data
    W_qkv_bias = (
        self.get_qkv_proj().Wqkv.bias.data.unsqueeze(0) if self.qkv_bias else None
    )

    fused_rmsnorm = rmsnorm is not None
    W_gamma = (
        rmsnorm.weight.data.unsqueeze(0)
        if fused_rmsnorm
        else torch.ones((1, h), device=the_device)
    )

    update_cache_in_kernel = (
        update_kv_per_layer and self.attn_block_tkg_nki_kernel_cache_update
    )

    W_out = self.get_o_proj().o_proj.weight.data
    h_out = h // 2 if getattr(self, "is_eagle3_draft", False) else h
    assert W_out.shape == (q_heads * self.head_dim, h_out), (
        f"W_out.shape = {W_out.shape}"
    )

    W_out_bias = (
        self.get_o_proj().o_proj.bias.data.unsqueeze(0) if self.o_bias else None
    )
    if W_out_bias is not None:
        W_out_bias = W_out_bias / self.tp_degree

    # ---- Output buffers ----
    if update_cache_in_kernel:
        K = K_prior
        V = V_prior
    else:
        # Multi-KV-head: K output is [d, B, kv_heads, q_len], V is [B, kv_heads, q_len, d]
        K = torch.zeros(
            self.head_dim, bsz, kv_heads, q_len, dtype=the_dtype, device=the_device
        )
        V = torch.zeros(
            bsz, kv_heads, q_len, self.head_dim, dtype=the_dtype, device=the_device
        )

    # ---- V active HBM buffer (multi-KV-head) ----
    # Pre-allocate in PyTorch to avoid NCC_IBIR440 (DRAM allocator failure for
    # kernel-internal nl.ndarray(..., buffer=nl.shared_hbm) with multi-KV-head).
    # Shape: [B_virt, 1, q_len, head_dim] where B_virt = bsz * kv_heads
    if kv_heads > 1:
        B_virt = bsz * kv_heads
        v_active_hbm_buf = torch.zeros(
            B_virt, 1, q_len, self.head_dim, dtype=the_dtype, device=the_device
        )
    else:
        v_active_hbm_buf = None

    # ---- RoPE parameters ----
    # skip_rope: pass None for cos/sin to skip RoPE in nki-library kernel
    cos_for_kernel = None if skip_rope else cos_cache
    sin_for_kernel = None if skip_rope else sin_cache

    # rope_contiguous_layout = rope_first_second_half_impl = not use_polar_compatible_rope
    rope_contiguous_layout = not use_polar_compatible_rope

    # ---- QK norm (pre-rope or post-rope) ----
    # For Mistral3: use_qk_norm is typically False
    # The bundled kernel has a single qk_norm boolean; nki-library splits into pre/post
    qk_norm = self.use_qk_norm
    pre_rope_rmsnorm = self.neuron_config.pre_rope_rmsnorm

    # If qk_norm is enabled and pre_rope_rmsnorm placement:
    rmsnorm_QK_pre_rope_enabled = qk_norm and pre_rope_rmsnorm
    rmsnorm_QK_post_rope_enabled = qk_norm and not pre_rope_rmsnorm

    # QK norm weights — the bundled kernel derives these internally from W_qkv
    # For now, pass None and let the kernel use unit scaling (identity norm)
    # This is safe because Mistral3 doesn't use QK norm
    rmsnorm_QK_pre_rope_W_Q = None
    rmsnorm_QK_pre_rope_W_K = None
    rmsnorm_QK_post_rope_W_Q = None
    rmsnorm_QK_post_rope_W_K = None

    # ---- kv_cache_update_idx ----
    # NxDI passes position_ids as [B, q_len] int32
    # nki-library kernel expects kv_cache_update_idx as [B, 1] uint32
    # For TKG (q_len=1), position_ids is already [B, 1]
    kv_cache_update_idx = position_ids.to(torch.int32)
    if kv_cache_update_idx.dim() == 1:
        kv_cache_update_idx = kv_cache_update_idx.unsqueeze(1)

    # ---- kv_cache_update_idx_virt (multi-KV-head) ----
    # Replicate position indices for each KV head within each batch.
    # Shape: [B_virt, 1] where B_virt = bsz * kv_heads
    # Each batch's position index is repeated kv_heads times.
    if kv_heads > 1:
        kv_cache_update_idx_virt = kv_cache_update_idx.repeat_interleave(
            kv_heads, dim=0
        )
    else:
        kv_cache_update_idx_virt = None

    # ---- H_actual (for padded checkpoints) ----
    X_hidden_dim_actual = getattr(self.config, "original_hidden_size", None)

    # ---- Grid ----
    # The bundled kernel uses nc(lnc_config) which returns a VNC object.
    # Our nki.jit kernel needs a plain integer grid.
    # Extract the integer value from the logical_nc_config.
    lnc = self.logical_nc_config
    grid = lnc if isinstance(lnc, int) else int(lnc)

    # WORKAROUND: Force grid=1 for multi-KV-head to avoid NCC_IXLV002 barrier
    # mismatch when attention_tkg runs with B_virt > 1 on LNC=2.
    # With grid=1, the kernel runs on a single program (no LNC split).
    # Performance impact: ~5-10% slower cache update (no K/V core parallelism),
    # but correctness is preserved.
    if kv_heads > 1:
        grid = 1

    # ---- Call our nki-library kernel ----
    from nkilib.core.utils.common_types import QuantizationType

    attn_output, K, V = attention_block_tkg[grid](
        # -- input
        X=hidden_states,
        X_hidden_dim_actual=X_hidden_dim_actual,
        # -- rmsnorm X
        rmsnorm_X_enabled=fused_rmsnorm,
        rmsnorm_X_eps=self.rms_norm_eps,
        rmsnorm_X_gamma=W_gamma,
        # -- qkv projections
        W_qkv=W_qkv,
        bias_qkv=W_qkv_bias,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        # -- Q/K processing: pre-RoPE RMSNorm
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=self.rms_norm_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        # -- Q/K processing: RoPE
        cos=cos_for_kernel,
        sin=sin_for_kernel,
        rope_contiguous_layout=rope_contiguous_layout,
        # -- Q/K processing: post-RoPE RMSNorm
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=self.rms_norm_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        # -- attention
        K_cache_transposed=self.k_cache_transposed,
        active_blocks_table=active_block_table,
        K_cache=K_prior,
        V_cache=V_prior,
        attention_mask=attention_mask_nki,
        sink=None,
        softmax_scale=None,
        # -- KV cache update
        update_cache=update_cache_in_kernel,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=None,
        v_scale=None,
        # -- output projection
        W_out=W_out,
        bias_out=W_out_bias,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        # -- output
        out_in_sb=False,
        sbm=None,
        skip_attention=False,
        # -- Multi-KV-head
        n_kv_heads=kv_heads,
        # -- Number of query heads per rank (explicit, avoids W_qkv.shape on PlaceholderParameter)
        n_q_heads=q_heads,
        # -- Head dimension (explicit, avoids NKI .shape on 4D cache)
        head_dim=self.head_dim,
        # -- Max context length (from KV cache shape, accessible in PyTorch)
        s_max_ctx=V_prior.shape[2],
        # -- Per-group attention mask (avoids NKI reshape-on-slice inside kernel)
        group_attention_mask=group_attention_mask,
        # -- Pre-allocated HBM buffer for V active tokens (multi-KV-head only)
        v_active_hbm=v_active_hbm_buf,
        # -- Replicated kv_cache_update_idx for multi-KV-head cache update
        kv_cache_update_idx_virt=kv_cache_update_idx_virt,
    )

    # ---- Post-processing: reshape output ----
    # Our kernel returns attn_output with O-proj already applied
    # Shape depends on transposed_out and out_in_sb:
    #   transposed_out=False, out_in_sb=False: [B*S_tkg, H] @ HBM
    attn_output = attn_output.reshape((bsz, q_len, h_out))

    # All-reduce or reduce-scatter across TP ranks
    from neuronx_distributed.parallel_layers.mappings import (
        reduce_from_tensor_model_parallel_region,
    )

    if self.sequence_parallel_enabled:
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_scatter_to_sequence_parallel_region,
        )

        attn_output = reduce_scatter_to_sequence_parallel_region(
            attn_output, 1, process_group=self.tensor_model_parallel_group
        )
    else:
        attn_output = reduce_from_tensor_model_parallel_region(
            attn_output, process_group=self.tensor_model_parallel_group
        )

    # ---- KV output handling ----
    if not update_cache_in_kernel:
        # K from kernel: [d, B, kv_heads, q_len] or [d, B, q_len] (if kv_heads=1)
        # V from kernel: [B, kv_heads, q_len, d] or [B, 1, q_len, d] (if kv_heads=1)
        # NxDI expects:
        #   K: [B, kv_heads, d, q_len] if k_cache_transposed else [B, kv_heads, q_len, d]
        #   V: [B, kv_heads, q_len, d]
        if K.dim() == 4:
            # [d, B, kv_heads, q_len] -> [B, kv_heads, d, q_len]
            if self.k_cache_transposed:
                K = K.permute(1, 2, 0, 3)
            else:
                # [d, B, kv_heads, q_len] -> [B, kv_heads, q_len, d]
                K = K.permute(1, 2, 3, 0)
        else:
            # Single head: [d, B, q_len]
            if self.k_cache_transposed:
                K = K.permute(1, 0, 2).unsqueeze(1)  # [B, 1, d, q_len]
            else:
                K = K.permute(1, 2, 0).unsqueeze(1)  # [B, 1, q_len, d]
        if V.dim() == 3:
            V = V.unsqueeze(1)  # [B, q_len, d] -> [B, 1, q_len, d]
        # V is already [B, kv_heads, q_len, d] from our kernel

    return attn_output, (K, V), cos_cache, sin_cache


def apply_patch():
    """
    Apply the native multi-KV-head kernel adapter to NeuronAttentionBase.

    Must be called after NxDI imports but before model compilation.
    """
    global _original_method, _patched

    from neuronx_distributed_inference.modules.attention.attention_base import (
        NeuronAttentionBase,
    )

    _original_method = NeuronAttentionBase.attention_block_tokengen_nki_kernel
    NeuronAttentionBase.attention_block_tokengen_nki_kernel = (
        _patched_attention_block_tokengen_nki_kernel
    )
    _patched = True
    logger.info(
        "Patched NeuronAttentionBase.attention_block_tokengen_nki_kernel "
        "with native multi-KV-head nki-library kernel adapter"
    )
    return True
