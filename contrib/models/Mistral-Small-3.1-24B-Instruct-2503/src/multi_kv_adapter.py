# ============================================================
# MULTI-KV-HEAD TKG KERNEL ADAPTER
# ============================================================
# Appended to attention_base.py by setup_patches.py
# Provides multi-KV-head support for the TKG fused attention NKI kernel.
# When kv_heads_per_rank == 1, delegates to the stock NxDI TKG method.
# When kv_heads_per_rank > 1, calls the Leanstral-derived forked kernel
# using a "virtual batch" approach (each KV head becomes a virtual batch entry).

import logging as _mkv_logging

_mkv_logger = _mkv_logging.getLogger("multi_kv_tkg_adapter")

_original_attn_block_tkg_nki = NeuronAttentionBase.attention_block_tokengen_nki_kernel


def _multi_kv_attention_block_tokengen_nki_kernel(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    active_mask=None,
    cos_cache=None,
    sin_cache=None,
    rmsnorm=None,
    rotary_position_ids=None,
    update_kv_per_layer=True,
    active_block_table=None,
    use_polar_compatible_rope=False,
):
    """
    Multi-KV-head TKG kernel adapter.

    When kv_heads_per_rank == 1, delegates to the original NxDI method.
    When kv_heads_per_rank > 1, calls the Leanstral forked kernel with
    n_kv_heads parameter, handling the interface translation.
    """
    import torch

    kv_heads = self.num_key_value_heads
    q_heads = self.num_heads

    # Fast path: kv_heads=1 per rank, use original unmodified method
    if kv_heads == 1:
        return _original_attn_block_tkg_nki(
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

    # --- Multi-KV-head path ---
    assert q_heads % kv_heads == 0, (
        f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})"
    )

    # Sequence parallel gather
    if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            self.sequence_dimension,
            process_group=self.tensor_model_parallel_group,
        )

    from nkilib.experimental.transformer.attention_block_tkg_multi_kv import (
        attention_block_tkg,
    )

    bsz, q_len, h = hidden_states.size()
    h_out = h // 2 if getattr(self, "is_eagle3_draft", False) else h

    # ---- RoPE cos/sin preparation ----
    skip_rope = False
    rope_contiguous_layout = not use_polar_compatible_rope

    if self.rotary_emb is not None:
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(hidden_states, rotary_position_ids)
            cos_cache = cos_cache[..., : cos_cache.shape[-1] // 2].permute(2, 0, 1)
            sin_cache = sin_cache[..., : sin_cache.shape[-1] // 2].permute(2, 0, 1)
    elif use_polar_compatible_rope:
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
        expected_shape = (self.head_dim // 2, bsz, q_len)
        cos_cache = torch.zeros(expected_shape).to(hidden_states)
        sin_cache = torch.zeros(expected_shape).to(hidden_states)
        skip_rope = True

    cos_for_kernel = None if skip_rope else cos_cache
    sin_for_kernel = None if skip_rope else sin_cache

    # ---- KV Cache ----
    K_prior = past_key_value[0].data
    V_prior = past_key_value[1].data

    the_dtype = hidden_states.dtype
    the_device = hidden_states.device

    # ---- Mask preparation ----
    s_prior = attention_mask.shape[-1]
    attention_mask = attention_mask.expand(-1, q_heads, -1, -1).contiguous()

    expected_active_mask_shape = (bsz, 1, q_len, q_len)
    if q_len == 1:
        active_mask = torch.ones(
            expected_active_mask_shape, dtype=the_dtype, device=the_device
        )
    active_mask = active_mask.expand(-1, q_heads, -1, -1).contiguous()
    attention_mask[:, :, :, -q_len:] = active_mask
    attention_mask_nki = attention_mask.permute(3, 0, 1, 2).contiguous()

    # Per-group mask for virtual batch approach
    q_per_kv_group = q_heads // kv_heads
    group_attention_mask = attention_mask_nki[:, :, :q_per_kv_group, :].contiguous()
    group_attention_mask = group_attention_mask.repeat(1, kv_heads, 1, 1).contiguous()

    # ---- Weights ----
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
        K = torch.zeros(
            self.head_dim, bsz, kv_heads, q_len, dtype=the_dtype, device=the_device
        )
        V = torch.zeros(
            bsz, kv_heads, q_len, self.head_dim, dtype=the_dtype, device=the_device
        )

    # ---- V active HBM buffer (workaround NCC_IBIR440) ----
    B_virt = bsz * kv_heads
    v_active_hbm_buf = torch.zeros(
        B_virt, 1, q_len, self.head_dim, dtype=the_dtype, device=the_device
    )

    # ---- kv_cache_update_idx ----
    kv_cache_update_idx = position_ids[:, :1].to(torch.int32)

    # ---- Replicated update idx for multi-KV ----
    kv_cache_update_idx_virt = kv_cache_update_idx.repeat_interleave(kv_heads, dim=0)

    # ---- QK norm ----
    has_qk_layernorm = self.q_layernorm is not None and self.k_layernorm is not None
    qk_norm_eps = self.rms_norm_eps if self.rms_norm_eps else 1e-6
    is_pre_rope_qk_norm = (
        has_qk_layernorm and self.qk_norm_placement == QKNormPlacement.PRE_ROPE
    )
    is_post_rope_qk_norm = (
        has_qk_layernorm and self.qk_norm_placement == QKNormPlacement.POST_ROPE
    )
    rmsnorm_QK_pre_rope_W_Q = (
        self.q_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None
    )
    rmsnorm_QK_pre_rope_W_K = (
        self.k_layernorm.weight.data.unsqueeze(0) if is_pre_rope_qk_norm else None
    )
    rmsnorm_QK_post_rope_W_Q = (
        self.q_layernorm.weight.data.unsqueeze(0) if is_post_rope_qk_norm else None
    )
    rmsnorm_QK_post_rope_W_K = (
        self.k_layernorm.weight.data.unsqueeze(0) if is_post_rope_qk_norm else None
    )

    # ---- Grid ----
    lnc = self.logical_nc_config
    grid = lnc if isinstance(lnc, int) else int(lnc)
    # WORKAROUND: Force grid=1 for multi-KV to avoid NCC_IXLV002
    if kv_heads > 1:
        grid = 1

    # ---- Call multi-KV kernel ----
    from nkilib.core.utils.common_types import QuantizationType

    attn_output, K, V = attention_block_tkg[grid](
        X=hidden_states,
        X_hidden_dim_actual=getattr(self.config, "original_hidden_size", None),
        rmsnorm_X_enabled=fused_rmsnorm,
        rmsnorm_X_eps=self.rms_norm_eps,
        rmsnorm_X_gamma=W_gamma,
        W_qkv=W_qkv,
        bias_qkv=W_qkv_bias,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=is_pre_rope_qk_norm,
        rmsnorm_QK_pre_rope_eps=qk_norm_eps if is_pre_rope_qk_norm else 0.0,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos_for_kernel,
        sin=sin_for_kernel,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=is_post_rope_qk_norm,
        rmsnorm_QK_post_rope_eps=qk_norm_eps if is_post_rope_qk_norm else 0.0,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        K_cache_transposed=self.k_cache_transposed,
        active_blocks_table=active_block_table,
        K_cache=K_prior,
        V_cache=V_prior,
        attention_mask=attention_mask_nki,
        sink=None,
        softmax_scale=None if self.softmax_scale is None else (1 / self.softmax_scale),
        update_cache=update_cache_in_kernel,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=W_out_bias,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        # Multi-KV-head parameters
        n_kv_heads=kv_heads,
        n_q_heads=q_heads,
        head_dim=self.head_dim,
        s_max_ctx=V_prior.shape[2],
        group_attention_mask=group_attention_mask,
        v_active_hbm=v_active_hbm_buf,
        kv_cache_update_idx_virt=kv_cache_update_idx_virt,
    )

    # ---- Post-processing ----
    attn_output = attn_output.reshape((bsz, q_len, h_out))

    # All-reduce or reduce-scatter across TP ranks
    if self.sequence_parallel_enabled:
        attn_output = reduce_scatter_to_sequence_parallel_region(
            attn_output, 1, process_group=self.tensor_model_parallel_group
        )
    else:
        from neuronx_distributed_inference.modules.attention.attention_base import (
            EPDispatchOption,
        )

        if self.ep_dispatch_cc_option == EPDispatchOption.AR_AG:
            attn_output = reduce_from_tensor_model_parallel_region(
                attn_output, process_group=self.tensor_model_parallel_group
            )
        elif self.ep_dispatch_cc_option == EPDispatchOption.RS_AG:
            attn_output = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                attn_output,
                partition_dim=0,
                process_group=self.tensor_model_parallel_group,
            )
        elif self.ep_dispatch_cc_option == EPDispatchOption.AG_AR:
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_data_parallel_attention_dp_group,
            )

            attn_output = gather_from_tensor_model_parallel_region_with_dim(
                attn_output,
                gather_dim=0,
                process_group=get_data_parallel_attention_dp_group(),
            )
        else:
            raise ValueError(f"Unknown EPDispatchOption: {self.ep_dispatch_cc_option}")

    # ---- KV output ----
    if not update_cache_in_kernel:
        if K.dim() == 4:
            if self.k_cache_transposed:
                K = K.permute(1, 2, 0, 3)
            else:
                K = K.permute(1, 2, 3, 0)
        else:
            if self.k_cache_transposed:
                K = K.permute(1, 0, 2).unsqueeze(1)
            else:
                K = K.permute(1, 2, 0).unsqueeze(1)
        if V.dim() == 3:
            V = V.unsqueeze(1)

    return attn_output, (K, V), cos_cache, sin_cache


NeuronAttentionBase.attention_block_tokengen_nki_kernel = (
    _multi_kv_attention_block_tokengen_nki_kernel
)
_mkv_logger.info("Applied multi-KV-head TKG kernel adapter to NeuronAttentionBase")
