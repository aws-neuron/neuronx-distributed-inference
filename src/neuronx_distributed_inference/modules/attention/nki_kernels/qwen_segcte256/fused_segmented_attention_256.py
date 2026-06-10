# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused segmented attention: replaces per-segment _attention_cte + external reduction
with a single flash-attention loop over all segments.

Uses kv_section_idx=0 so K/V indexing always starts at tile 0 (each segment has
its own K/V in SBUF), while section_idx > 0 triggers the flash attention
accumulation path in _write_back_impl and _update_max_impl. This keeps the PV
accumulation in float32 SBUF across segments, matching _attention_cte's internal
precision.
"""

import math

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.attention_reduce import _MAX_FREE_TILES, reduce_one_batch
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE, div_ceil
from nkilib.core.utils.modular_allocator import ModularAllocator
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nki.isa import reduce_cmd
from nki.language.opcode import maximum as _maximum
from nkilib.core.attention.attention_cte import (
    _FLOAT32_MIN,
    _K_TILE_SZ,
    _LARGE_TILE_SZ,
    _Q_GRP_SZ,
    _V_TILE_SZ,
    AttnConfig,
    AttnInternalBuffers,
    SectionParams,
    _allocate_attention_buffers,
    _compute_tile_parameters,
    _exp_impl,
    _fused_qkmax_and_pv_impl,
    _get_kv_tile_apc,
    _has_any_compute_causal,
    _has_any_compute_swa,
    _load_q_impl,
    _pv_impl,
    _qk_and_max_impl,
    _setup_range_select_bounds,
    _update_max_impl,
    _write_back_impl,
)


def _run_groups(grp_start, grp_end, ac, atp, sp, bufs, q, batch_id, o, sbuf_addr, sink=None):
    """Run Q-group loop with software pipelining."""
    n = grp_end - grp_start
    if n <= 1:
        _load_q_impl(grp_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start, ac, atp, sp, bufs, sink)
        _exp_impl(grp_start, ac, atp, sp, bufs, sink)
        _pv_impl(grp_start, ac, atp, sp, bufs)
        _write_back_impl(grp_start, ac, atp, sp, bufs, o, batch_id)
    else:
        _load_q_impl(grp_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start, ac, atp, sp, bufs, sink)
        _exp_impl(grp_start, ac, atp, sp, bufs, sink)

        _load_q_impl(grp_start + 1, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start + 1, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start + 1, ac, atp, sp, bufs, sink)

        for grp_i in range(grp_start, grp_end - 2):
            _load_q_impl(grp_i + 2, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
            _exp_impl(grp_i + 1, ac, atp, sp, bufs, sink)
            _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs, batch_id)
            _write_back_impl(grp_i, ac, atp, sp, bufs, o, batch_id)
            _update_max_impl(grp_i + 2, ac, atp, sp, bufs, sink)

        _pv_impl(grp_end - 2, ac, atp, sp, bufs)
        _write_back_impl(grp_end - 2, ac, atp, sp, bufs, o, batch_id)
        _exp_impl(grp_end - 1, ac, atp, sp, bufs, sink)
        _pv_impl(grp_end - 1, ac, atp, sp, bufs)
        _write_back_impl(grp_end - 1, ac, atp, sp, bufs, o, batch_id)


def _make_ac_atp(
    seqlen_q, seqlen_k, head_dim, dtype, causal, scale, tp_q, tp_out, num_sections, use_cp=False, global_cp_deg=None
):
    """Create AttnConfig + AttnTileParams."""
    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k_active=seqlen_k,
        seqlen_k_prior=None,
        d=head_dim,
        tp_q=tp_q,
        tp_k=False,
        tp_out=tp_out,
        is_prefix_caching=False,
        causal_mask=causal,
        use_swa=False,
        sliding_window=0,
        use_cp=use_cp,
        global_cp_deg=global_cp_deg,
        cp_strided_q_slicing=False,
        cp_striped_input=False,
        scale=scale,
        cache_softmax=True,
        skip_output_normalization=True,
        dtype=dtype,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        is_sequence_packed=False,
    )
    atp = _compute_tile_parameters(ac, is_seqlen_sharded=False)
    if head_dim == 256:
        atp.num_q_grps_per_load = min(4, atp.num_grps)
    atp.num_sections = num_sections
    return ac, atp


def _kvp_partial_prior_attention(
    q_hbm,
    k_cache_sbuf,
    v_cache_sbuf,
    k_prior_sbuf,
    v_prior_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    kvp_offset_active_hbm,
    kvp_offset,
    prior_block_offset,
    partial_prior_tokens,
    num_k_tiles_active,
    num_v_tiles_active,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    n_grps,
    head_dim,
    block_size,
    sb_p,
    scale,
    tp_q,
    allocator,
    attention_cte_fn,
    sink=None,
    k_scale_sb=None,
):
    """KVP partial prior: two separate attention_cte calls (active + prior) then reduce.

    Splits into two calls to avoid the unsupported use_cp + is_prefix_caching combination:
      1. Active-only: causal_mask=True with cp_offset=kvp_offset_active
      2. Prior-only: causal_mask=False with effective_prior_used_len
    Results are reduced via online softmax into o_prev_hbm in-place.
    """
    init_sbuf_addr = allocator.get_current_address()

    # Call 1: active-only with causal mask + cp_offset.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=True,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_active],
        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_active],
        out_o_hbm=o_prev_hbm,
        out_neg_max_hbm=neg_max_prev_hbm,
        out_sum_hbm=sum_prev_hbm,
        init_sbuf_addr=init_sbuf_addr,
        sink=sink,
        cp_offset=kvp_offset_active_hbm,
        global_cp_deg=1,
        k_scale_sb=k_scale_sb,
    )
    allocator.set_current_address(init_sbuf_addr)

    # Compute effective_prior_used_len = max(0, min(partial_prior_tokens, kvp_offset - prior_block_offset*block_size))
    effective_prior_used_len = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    prior_seg_start_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    nisa.tensor_scalar(dst=prior_seg_start_sbuf, data=prior_block_offset, op0=nl.multiply, operand0=block_size)
    nisa.tensor_tensor(dst=effective_prior_used_len, data1=kvp_offset, data2=prior_seg_start_sbuf, op=nl.subtract)
    nisa.tensor_tensor(
        dst=effective_prior_used_len, data1=effective_prior_used_len, data2=partial_prior_tokens, op=nl.minimum
    )
    nisa.tensor_scalar(dst=effective_prior_used_len, data=effective_prior_used_len, op0=nl.maximum, operand0=0)

    # Re-zero active KV sbuf before Call 2 (prior-only, causal_mask=False).
    for k_idx in range(num_k_tiles_active):
        nisa.memset(k_cache_sbuf[k_idx][...], value=0.0)
    for v_idx in range(num_v_tiles_active):
        nisa.memset(v_cache_sbuf[v_idx][...], value=0.0)

    call2_sbuf_addr = allocator.get_current_address()
    # Call 2: prior-only with causal_mask=False and effective_prior_used_len (SBUF).
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=False,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_per_seg],
        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_per_seg],
        k_prior_sbuf=k_prior_sbuf,
        v_prior_sbuf=v_prior_sbuf,
        prior_used_len=effective_prior_used_len,
        out_o_hbm=o_curr_hbm,
        out_neg_max_hbm=neg_max_curr_hbm,
        out_sum_hbm=sum_curr_hbm,
        init_sbuf_addr=call2_sbuf_addr,
    )
    allocator.set_current_address(call2_sbuf_addr)

    # Reduce active (o_prev_hbm) + prior (o_curr_hbm) into o_prev_hbm.
    softmax_pat = [[n_grps, sb_p], [1, n_grps]]
    o_pat = [[head_dim, sb_p], [1, head_dim]]
    num_free = min(n_grps, _MAX_FREE_TILES)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()
    reduce_one_batch(
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        n_grps,
        head_dim,
        n_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb_buf,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )
    allocator.set_current_address(init_sbuf_addr)


def _nonkvp_partial_prior_attention(
    q_hbm,
    k_cache,
    v_cache,
    block_tables,
    k_cache_sbuf,
    v_cache_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    prior_block_offset,
    partial_prior_tokens,
    num_k_tiles_active,
    num_v_tiles_active,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    num_blocks_per_seg,
    num_v_tiles_for_prior,
    b_i,
    h_i,
    n_grps,
    head_dim,
    sb_p,
    scale,
    tp_q,
    allocator,
    attention_cte_fn,
    load_kv_cache_fn,
    sink=None,
):
    """Non-KVP partial prior: two sequential attention_cte calls then reduce.

    Mirrors _kvp_partial_prior_attention's 2-pass shape but without cp_offset /
    global_cp_deg, and uses the static partial_prior_tokens directly as
    prior_used_len (no dynamic effective_prior_used_len math needed).

    Pass 1: active-only, causal_mask=True, sink applied.
    ---- Allocate k_prior_sbuf/v_prior_sbuf ALIASED onto k_cache_sbuf /
         v_cache_sbuf (same physical SBUF region). Pass 1 has already reduced
         its active-K result into o_prev_hbm, so the active K data in
         k_cache_sbuf is no longer needed. Load prior K/V into the aliased
         region.
    Pass 2: prior-only, causal_mask=False, prior_used_len=partial_prior_tokens.
    Reduce via online softmax into o_prev_hbm in-place.

    Saves ~48 KB/partition of peak SBUF at head_dim=128 prior_seg_size=8192
    compared to the previous single-fused-call design (which held both
    k_cache_sbuf and a separate k_prior_sbuf concurrently live through APC).
    """
    init_sbuf_addr = allocator.get_current_address()

    # Pass 1: active-only with causal mask. No k_prior_sbuf reference.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=True,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_active],
        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_active],
        out_o_hbm=o_prev_hbm,
        out_neg_max_hbm=neg_max_prev_hbm,
        out_sum_hbm=sum_prev_hbm,
        init_sbuf_addr=init_sbuf_addr,
        sink=sink,
    )
    allocator.set_current_address(init_sbuf_addr)

    # Alias k_prior_sbuf/v_prior_sbuf onto the first N tiles of
    # k_cache_sbuf/v_cache_sbuf via Python list slicing — same physical SBUF,
    # no new allocation. k_cache_sbuf is sized with
    # max(num_k_tiles_active, num_k_tiles_per_seg) at the caller, so the slice
    # is always in range.
    kernel_assert(
        num_k_tiles_per_seg <= len(k_cache_sbuf),
        "k_cache_sbuf must be sized >= num_k_tiles_per_seg for aliased reuse",
    )
    kernel_assert(
        num_v_tiles_for_prior <= len(v_cache_sbuf),
        "v_cache_sbuf must be sized >= num_v_tiles_for_prior for aliased reuse",
    )
    k_prior_sbuf = k_cache_sbuf[:num_k_tiles_per_seg]
    v_prior_sbuf = v_cache_sbuf[:num_v_tiles_for_prior]

    # Load prior K/V into the aliased region. This overwrites the active K/V
    # from Pass 1, which is safe because Pass 1's results are already in
    # o_prev_hbm.
    load_kv_cache_fn(
        k_cache,
        v_cache,
        block_tables,
        k_prior_sbuf,
        v_prior_sbuf,
        b_i,
        h_i,
        prior_block_offset,
        num_blocks_per_seg,
        allocator,
    )

    call2_sbuf_addr = allocator.get_current_address()
    # Pass 2: non-APC call treating the aliased prior data as the active K/V.
    # `kv_used_len=partial_prior_tokens` dynamically masks K positions beyond
    # the used prior range. Previously this was an APC call (k_prior_sbuf +
    # k_cache_sbuf both pointing to the aliased memory), but that caused the
    # kernel to attend the prior data TWICE — once as "active" (unmasked) and
    # once as "prior" (masked by prior_used_len) — inflating sum_curr_hbm by
    # ~2× and skewing the reduce_one_batch combination. Using kv_used_len in
    # non-APC mode keeps Bucket B's SBUF aliasing AND produces correct output.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=False,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_prior_sbuf,
        v_cache_sbuf=v_prior_sbuf,
        kv_used_len=partial_prior_tokens,
        out_o_hbm=o_curr_hbm,
        out_neg_max_hbm=neg_max_curr_hbm,
        out_sum_hbm=sum_curr_hbm,
        init_sbuf_addr=call2_sbuf_addr,
    )
    allocator.set_current_address(call2_sbuf_addr)

    # Reduce active (o_prev_hbm) + prior (o_curr_hbm) into o_prev_hbm.
    softmax_pat = [[n_grps, sb_p], [1, n_grps]]
    o_pat = [[head_dim, sb_p], [1, head_dim]]
    num_free = min(n_grps, _MAX_FREE_TILES)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()
    reduce_one_batch(
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        n_grps,
        head_dim,
        n_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb_buf,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )
    allocator.set_current_address(init_sbuf_addr)


_allocate_attention_buffers_base = _allocate_attention_buffers
_load_q_impl_base = _load_q_impl
_qk_and_max_impl_base = _qk_and_max_impl
_pv_impl_base = _pv_impl


def _zero_k_tiles(k_sbuf, num_tiles, head_dim):
    if head_dim == 256:
        for k_idx in range(num_tiles):
            nisa.memset(k_sbuf[k_idx][0][...], value=0.0)
            nisa.memset(k_sbuf[k_idx][1][...], value=0.0)
    else:
        for k_idx in range(num_tiles):
            nisa.memset(k_sbuf[k_idx][...], value=0.0)


def _repeat_ref(value, count):
    values = []
    for _ in range(count):
        values.append(value)
    return values


def _allocate_attention_buffers(
    allocator,
    ac: AttnConfig,
    atp,
    bufs: AttnInternalBuffers,
    sink=None,
    k_cache_sbuf=None,
    v_cache_sbuf=None,
):
    if ac.d <= 128:
        return _allocate_attention_buffers_base(allocator, ac, atp, bufs, sink, k_cache_sbuf, v_cache_sbuf)

    kernel_assert(ac.d == 256, f"qwen_segcte256 only supports head_dim=256, got {ac.d}")
    kernel_assert(not ac.tp_out, "qwen_segcte256 uses tp_out=False to keep head_dim on the free axis")

    mm1_p, mm1_n = atp.sb_p, nl.tile_size.psum_fmax
    mm2_p, mm2_n = atp.sb_p, ac.d
    num_q_slots = div_ceil(atp.num_grps, atp.num_q_grps_per_load)

    if k_cache_sbuf is not None and len(k_cache_sbuf) > 0:
        bufs.k_sb = k_cache_sbuf
    else:
        k_lo = allocator.alloc_sbuf_tensor(
            shape=(128, _K_TILE_SZ),
            dtype=nl.bfloat16,
            block_dim=[atp.num_k_tiles_per_section],
            num_free_tiles=[atp.num_k_tiles_per_section],
            align_to=32,
        )
        k_hi = allocator.alloc_sbuf_tensor(
            shape=(128, _K_TILE_SZ),
            dtype=nl.bfloat16,
            block_dim=[atp.num_k_tiles_per_section],
            num_free_tiles=[atp.num_k_tiles_per_section],
            align_to=32,
        )
        bufs.k_sb = []
        for i in range(atp.num_k_tiles_per_section):
            bufs.k_sb.append((k_lo[i], k_hi[i]))

    if v_cache_sbuf is not None and len(v_cache_sbuf) > 0:
        bufs.v_sb = v_cache_sbuf
    else:
        bufs.v_sb = allocator.alloc_sbuf_tensor(
            shape=(_V_TILE_SZ, ac.d),
            dtype=nl.bfloat16,
            block_dim=[atp.num_v_tiles_per_section],
            num_free_tiles=[atp.num_v_tiles_per_section],
        )

    # This kernel runs Q groups sequentially, not with the upstream
    # software-pipelined 3-group schedule. Keep one physical scratch window and
    # alias all logical group slots to it so pfx256 does not allocate per-group
    # MM1/MM2 scratch for the full CTE bucket.
    q_sb_lo = allocator.alloc_sbuf_tensor(
        shape=(128, atp.sb_p * atp.num_q_grps_per_load),
        dtype=nl.bfloat16,
        align_to=32,
    )
    q_sb_hi = allocator.alloc_sbuf_tensor(
        shape=(128, atp.sb_p * atp.num_q_grps_per_load),
        dtype=nl.bfloat16,
        align_to=32,
    )
    bufs.q_sb_lo = _repeat_ref(q_sb_lo, num_q_slots)
    bufs.q_sb_hi = _repeat_ref(q_sb_hi, num_q_slots)

    flash_attn_correction_factor = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
    )
    bufs.flash_attn_correction_factor = _repeat_ref(flash_attn_correction_factor, atp.num_grps)
    mm1_partial_max_n_elts = atp.num_k_tiles_per_section + (sink is not None)
    mm1_partial_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, mm1_partial_max_n_elts),
        dtype=nl.float32,
        align_to=4,
    )
    bufs.mm1_partial_max = _repeat_ref(mm1_partial_max, atp.num_grps)
    mm1_section_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
    )
    bufs.mm1_section_max = _repeat_ref(mm1_section_max, atp.num_grps)
    n_final_reduce_sum_elts = div_ceil(atp.section_len, atp.exp_inst_elems) + (sink is not None)
    exp_partial_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, n_final_reduce_sum_elts),
        dtype=nl.float32,
    )
    bufs.exp_partial_sum = _repeat_ref(exp_partial_sum, atp.num_grps)
    exp_section_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
    )
    bufs.exp_section_sum = _repeat_ref(exp_section_sum, atp.num_grps)
    prev_mm1_running_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
    )
    bufs.prev_mm1_running_max = _repeat_ref(prev_mm1_running_max, atp.num_grps)
    prev_exp_running_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
    )
    bufs.prev_exp_running_sum = _repeat_ref(prev_exp_running_sum, atp.num_grps)
    mm2_prev_output = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d),
        dtype=ac.mm_out_dtype,
    )
    bufs.mm2_prev_output = _repeat_ref(mm2_prev_output, atp.num_grps)
    mm2_accum_flash_attn = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d),
        dtype=nl.float32,
    )
    bufs.mm2_accum_flash_attn = _repeat_ref(mm2_accum_flash_attn, atp.num_grps)
    mm2_final = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d),
        dtype=ac.mm_out_dtype,
    )
    bufs.mm2_final = _repeat_ref(mm2_final, atp.num_grps)
    mm2_sb = allocator.alloc_sbuf_tensor(
        shape=(mm2_p, mm2_n),
        dtype=ac.mm_out_dtype,
    )
    bufs.mm2_sb = _repeat_ref(mm2_sb, atp.num_grps)
    mm1_masked_tiles = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.float32,
        block_dim=[atp.num_large_tiles_per_section],
        num_free_tiles=[1],
    )
    mm1_masked_row = []
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        mm1_masked_row.append(mm1_masked_tiles[large_tile_idx])
    bufs.mm1_masked = _repeat_ref(mm1_masked_row, atp.num_grps)
    exp_sb_tiles = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.bfloat16,
        block_dim=[atp.num_large_tiles_per_section],
        num_free_tiles=[1],
    )
    exp_sb_row = []
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        exp_sb_row.append(exp_sb_tiles[large_tile_idx])
    bufs.exp_sb = _repeat_ref(exp_sb_row, atp.num_grps)

    bufs.mm1_psum = []
    for grp_idx in range(atp.num_grps):
        mm1_psum_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            tile_row = []
            for k_tile_idx in range(4):
                tile_row.append(
                    nl.ndarray(
                        (mm1_p, mm1_n),
                        dtype=ac.mm_out_dtype,
                        buffer=nl.psum,
                        address=(0, (k_tile_idx % 4) * PSUM_BANK_SIZE),
                    )
                )
            mm1_psum_row.append(tile_row)
        bufs.mm1_psum.append(mm1_psum_row)

    if not atp.dynamic_sel_mask:
        mm1_copy_tiles = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=ac.mm_out_dtype,
            block_dim=[atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1],
        )
        mm1_affine_select_output_tiles = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=ac.mm_out_dtype,
            block_dim=[atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1],
        )
        mm1_copy_row = []
        mm1_affine_select_output_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            mm1_copy_tile_row = []
            mm1_affine_select_output_tile_row = []
            for k_tile_idx in range(4):
                mm1_copy_tile_row.append(mm1_copy_tiles[large_tile_idx][k_tile_idx])
                mm1_affine_select_output_tile_row.append(
                    mm1_affine_select_output_tiles[large_tile_idx][k_tile_idx]
                )
            mm1_copy_row.append(mm1_copy_tile_row)
            mm1_affine_select_output_row.append(mm1_affine_select_output_tile_row)
        bufs.mm1_copy_sb = _repeat_ref(mm1_copy_row, atp.num_grps)
        bufs.mm1_affine_select_output = _repeat_ref(mm1_affine_select_output_row, atp.num_grps)

    exp_tp_tiles = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.mm2_grp_sz),
        dtype=nl.bfloat16,
        block_dim=[atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp],
        num_free_tiles=[1, atp.num_tps_in_mm2_grp],
        align_to=32,
    )
    exp_tp_row = []
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        exp_tp_tile_row = []
        for tp_idx in range(atp.num_tps_in_mm2_grp):
            exp_tp_tile_row.append(exp_tp_tiles[large_tile_idx][tp_idx])
        exp_tp_row.append(exp_tp_tile_row)
    bufs.exp_tp_sb = _repeat_ref(exp_tp_row, atp.num_grps)

    bufs.mm2_psum = []
    for grp_idx in range(atp.num_grps):
        mm2_psum_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            mm2_psum_row.append(
                nl.ndarray(
                    (mm2_p, mm2_n),
                    dtype=ac.mm_out_dtype,
                    buffer=nl.psum,
                    address=(0, ((4 + (large_tile_idx % 4)) * PSUM_BANK_SIZE)),
                )
            )
        bufs.mm2_psum.append(mm2_psum_row)


def _load_q_impl(grp_i, ac: AttnConfig, atp, sp: SectionParams, bufs: AttnInternalBuffers, q, batch_id, sbuf_addr):
    if ac.d <= 128:
        return _load_q_impl_base(grp_i, ac, atp, sp, bufs, q, batch_id, sbuf_addr)

    kernel_assert(ac.d == 256, f"qwen_segcte256 only supports head_dim=256, got {ac.d}")
    if grp_i % atp.num_q_grps_per_load != 0:
        return

    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac, atp.num_q_grps_per_load)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if not has_any_compute_pred:
        return

    q_seqlen_offset = grp_i * _Q_GRP_SZ
    q_slot = grp_i // atp.num_q_grps_per_load
    num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ * atp.num_q_grps_per_load)
    kernel_assert(str(q.dtype) == str(nl.bfloat16), "qwen_segcte256 currently expects bf16 Q input")

    if ac.tp_q:
        _, seqlen, _ = q.shape
        for d_half in range(2):
            d_offset = d_half * 128
            q_dst = bufs.q_sb_lo[q_slot] if d_half == 0 else bufs.q_sb_hi[q_slot]
            nisa.dma_transpose(
                dst=q_dst.ap([[_Q_GRP_SZ * atp.num_q_grps_per_load, 128], [1, 1], [1, 1], [1, num_f]]),
                src=q.ap(
                    [[ac.d, num_f], [1, 1], [1, 1], [1, 128]],
                    offset=batch_id * seqlen * ac.d + q_seqlen_offset * ac.d + d_offset,
                ),
            )
    else:
        _, _, seqlen = q.shape
        for d_half in range(2):
            d_offset = d_half * 128
            q_dst = bufs.q_sb_lo[q_slot] if d_half == 0 else bufs.q_sb_hi[q_slot]
            nisa.dma_copy(
                dst=q_dst.ap(pattern=[[_Q_GRP_SZ * atp.num_q_grps_per_load, 128], [1, num_f]], offset=0),
                src=q.ap(
                    pattern=[[seqlen, 128], [1, num_f]],
                    offset=batch_id * ac.d * seqlen + d_offset * seqlen + q_seqlen_offset,
                ),
            )


def _qk_and_max_impl(grp_i, ac: AttnConfig, atp, sp: SectionParams, bufs: AttnInternalBuffers, batch_id: int = 0):
    if ac.d <= 128:
        return _qk_and_max_impl_base(grp_i, ac, atp, sp, bufs, batch_id)

    has_any_compute_pred = (
        _has_any_compute_causal(grp_i, sp.section_offset_active, ac)
        if (atp.is_causal and not sp.section_contains_prefix)
        else True
    )
    if has_any_compute_pred:
        nisa.memset(bufs.mm1_partial_max[grp_i], value=_FLOAT32_MIN)
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            _qk_and_max_large_tile_impl_256(grp_i, large_tile_idx, ac, atp, sp, bufs, batch_id)


def _qk_and_max_large_tile_impl_256(qkmax_grp, large_tile_idx, ac, atp, sp, bufs, batch_id: int = 0):
    q_seqlen_offset = qkmax_grp * atp.sb_p
    num_k_tiles_in_large_tile = _LARGE_TILE_SZ // _K_TILE_SZ
    for k_tile_idx in range(num_k_tiles_in_large_tile):
        mm1_psum_tile = bufs.mm1_psum[qkmax_grp][large_tile_idx][k_tile_idx]
        if not atp.dynamic_sel_mask:
            mm1_copy_sb_tile = bufs.mm1_copy_sb[qkmax_grp][large_tile_idx][k_tile_idx]
            mm1_affine_select_output_tile = bufs.mm1_affine_select_output[qkmax_grp][large_tile_idx][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[qkmax_grp][large_tile_idx]
        mm1_partial_max_tile = bufs.mm1_partial_max[qkmax_grp]

        k_tile_idx_in_section = large_tile_idx * num_k_tiles_in_large_tile + k_tile_idx
        _kv_sec_idx = sp.kv_section_idx if sp.kv_section_idx is not None else sp.section_idx
        k_tile_idx_global = atp.num_k_tiles_per_section * _kv_sec_idx + k_tile_idx_in_section
        is_prior_tile, seqlen_k, k_start_pos, _ = _get_kv_tile_apc(
            ac.is_prefix_caching,
            False,
            True,
            atp.seqlen_k_active_updated,
            ac.seqlen_k_prior,
            k_tile_idx_global * _K_TILE_SZ,
            None,
        )

        if atp.is_causal and not is_prior_tile:
            matmul_selection = _has_any_compute_causal(qkmax_grp, k_start_pos, ac)
            if ac.use_swa:
                matmul_selection = matmul_selection and _has_any_compute_swa(qkmax_grp, k_start_pos, _K_TILE_SZ, ac)
        else:
            matmul_selection = True

        if q_seqlen_offset >= ac.seqlen_q or k_start_pos >= seqlen_k:
            matmul_selection = False

        if matmul_selection and k_tile_idx_in_section < atp.num_k_tiles_per_section:
            num_f = min(seqlen_k - k_start_pos, _K_TILE_SZ)
            num_q_free = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            if is_prior_tile and bufs.k_sb_prior is not None:
                k_tile_to_use = bufs.k_sb_prior[k_start_pos // _K_TILE_SZ]
            elif bufs.k_sb_prior is not None:
                k_tile_to_use = bufs.k_sb[k_start_pos // _K_TILE_SZ]
            else:
                k_tile_to_use = bufs.k_sb[k_tile_idx_in_section]

            q_slot = qkmax_grp // atp.num_q_grps_per_load
            q_offset = (qkmax_grp % atp.num_q_grps_per_load) * _Q_GRP_SZ
            nisa.nc_matmul(
                mm1_psum_tile[:num_q_free, :num_f],
                bufs.q_sb_lo[q_slot][:128, nl.ds(q_offset, num_q_free)],
                k_tile_to_use[0][:128, :num_f],
            )
            nisa.nc_matmul(
                mm1_psum_tile[:num_q_free, :num_f],
                bufs.q_sb_hi[q_slot][:128, nl.ds(q_offset, num_q_free)],
                k_tile_to_use[1][:128, :num_f],
            )

            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            num_f = min(seqlen_k - k_start_pos, _K_TILE_SZ)
            diagonal_sel_mask = (
                matmul_selection and ((qkmax_grp * _Q_GRP_SZ) < (k_start_pos + _K_TILE_SZ))
                if (atp.is_causal and not is_prior_tile and not atp.dynamic_sel_mask)
                else False
            )
            if ac.use_swa and atp.is_causal and not is_prior_tile:
                diagonal_sel_mask = not atp.dynamic_sel_mask

            if diagonal_sel_mask:
                nisa.tensor_copy(mm1_copy_sb_tile[:num_p, :num_f], mm1_psum_tile[:num_p, :num_f])
                nisa.affine_select(
                    mm1_affine_select_output_tile[:num_p, :num_f],
                    pattern=[[-1, num_f]],
                    offset=qkmax_grp * atp.sb_p - k_start_pos,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=mm1_copy_sb_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                )
                if ac.use_swa:
                    nisa.affine_select(
                        mm1_affine_select_output_tile[:num_p, :num_f],
                        pattern=[[1, num_f]],
                        offset=(k_start_pos + ac.sliding_window - 1 - qkmax_grp * atp.sb_p),
                        channel_multiplier=-1,
                        cmp_op=nl.greater_equal,
                        on_true_tile=mm1_affine_select_output_tile[:num_p, :num_f],
                        on_false_value=_FLOAT32_MIN,
                    )
                nisa.tensor_scalar_reduce(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_affine_select_output_tile[:num_p, :num_f],
                    op0=nl.multiply,
                    operand0=ac.scale,
                    reduce_op=nl.maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                )
            elif atp.dynamic_sel_mask or is_prior_tile:
                if is_prior_tile:
                    bound0 = bufs.range_sel_lbs_prior[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                    bound1 = bufs.range_sel_ubs_prior[:num_p, qkmax_grp]
                    comp_op1 = nl.less
                elif ac.is_sequence_packed:
                    bound0 = bufs.range_sel_lbs[:num_p, nl.ds(qkmax_grp, 1)]
                    bound1 = bufs.range_sel_ubs[:num_p, nl.ds(qkmax_grp, 1)]
                    comp_op1 = nl.less_equal if atp.is_causal else nl.less
                else:
                    bound0 = bufs.range_sel_lbs[:num_p, qkmax_grp] if ac.use_swa else bufs.zero_bias_tensor
                    bound1 = bufs.range_sel_ubs[:num_p, qkmax_grp]
                    comp_op1 = nl.less_equal

                kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
                nisa.range_select(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    on_true_tile=mm1_psum_tile[:num_p, :num_f],
                    on_false_value=_FLOAT32_MIN,
                    comp_op0=nl.greater_equal,
                    comp_op1=comp_op1,
                    bound0=bound0[:num_p, :1],
                    bound1=bound1[:num_p, :1],
                    reduce_op=_maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                    reduce_cmd=reduce_cmd.reset_reduce,
                    range_start=k_start_pos,
                )
            else:
                nisa.tensor_scalar_reduce(
                    mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                    data=mm1_psum_tile[:num_p, :num_f],
                    op0=nl.multiply,
                    operand0=ac.scale,
                    reduce_op=nl.maximum,
                    reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
                )


def _run_groups(grp_start, grp_end, ac, atp, sp, bufs, q, batch_id, o, sbuf_addr, sink=None):
    for grp_i in range(grp_start, grp_end):
        _load_q_impl(grp_i, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_i, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_i, ac, atp, sp, bufs, sink)
        _exp_impl(grp_i, ac, atp, sp, bufs, sink)
        _pv_impl_base(grp_i, ac, atp, sp, bufs)
        _write_back_impl(grp_i, ac, atp, sp, bufs, o, batch_id)


def _run_attention_from_sbuf(
    q_hbm,
    k_sbuf,
    v_sbuf,
    out_o_hbm,
    out_neg_max_hbm,
    out_sum_hbm,
    seqlen_q,
    seqlen_k,
    head_dim,
    sb_p,
    n_grps,
    scale,
    causal,
    tp_q,
    allocator,
    sink=None,
    kv_used_len=None,
):
    ac, atp = _make_ac_atp(seqlen_q, seqlen_k, head_dim, q_hbm.dtype, causal, scale, tp_q, False, 2)
    ac.has_kv_used_len = kv_used_len is not None
    bufs = AttnInternalBuffers()
    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)
    bufs.k_scale_sb = None
    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    if sink is not None:
        bufs.sink_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
        nisa.dma_copy(dst=bufs.sink_sb[0, 0], src=sink[0, 0])
        stream_shuffle_broadcast(src=bufs.sink_sb, dst=bufs.sink_sb)
    _allocate_attention_buffers(allocator, ac, atp, bufs, sink, k_sbuf, v_sbuf)
    _setup_range_select_bounds(ac, atp, bufs, allocator, None, None, None, None, batch_id=0, kv_used_len=kv_used_len)
    sp = SectionParams(
        section_idx=0,
        section_offset=0,
        section_offset_active=0,
        next_section_offset_active=seqlen_k,
        section_contains_prefix=False,
        next_section_contains_prefix=False,
        kv_section_idx=0,
    )
    sbuf_inner = allocator.get_current_address()
    _run_groups(0, n_grps, ac, atp, sp, bufs, q_hbm, 0, out_o_hbm, sbuf_inner, sink=sink)
    nisa.dma_copy(dst=out_neg_max_hbm.ap(pattern=[[n_grps, sb_p], [1, n_grps]], offset=0), src=bufs.mm1_running_max)
    nisa.dma_copy(dst=out_sum_hbm.ap(pattern=[[n_grps, sb_p], [1, n_grps]], offset=0), src=bufs.exp_running_sum)


def _nonkvp_partial_prior_attention(
    q_hbm,
    k_cache,
    v_cache,
    block_tables,
    k_cache_sbuf,
    v_cache_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    prior_block_offset,
    partial_prior_tokens,
    num_k_tiles_active,
    num_v_tiles_active,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    num_blocks_per_seg,
    num_v_tiles_for_prior,
    b_i,
    h_i,
    n_grps,
    head_dim,
    sb_p,
    scale,
    tp_q,
    allocator,
    attention_cte_fn,
    load_kv_cache_fn,
    sink=None,
):
    init_sbuf_addr = allocator.get_current_address()
    seqlen_q = q_hbm.shape[1] if tp_q else q_hbm.shape[2]
    _run_attention_from_sbuf(
        q_hbm,
        k_cache_sbuf[:num_k_tiles_active],
        v_cache_sbuf[:num_v_tiles_active],
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        seqlen_q,
        seqlen_q,
        head_dim,
        sb_p,
        n_grps,
        scale,
        True,
        tp_q,
        allocator,
        sink=sink,
    )
    allocator.set_current_address(init_sbuf_addr)

    kernel_assert(num_k_tiles_per_seg <= len(k_cache_sbuf), "k_cache_sbuf must fit prior segment")
    kernel_assert(num_v_tiles_for_prior <= len(v_cache_sbuf), "v_cache_sbuf must fit prior segment")
    k_prior_sbuf = k_cache_sbuf[:num_k_tiles_per_seg]
    v_prior_sbuf = v_cache_sbuf[:num_v_tiles_for_prior]
    load_kv_cache_fn(
        k_cache,
        v_cache,
        block_tables,
        k_prior_sbuf,
        v_prior_sbuf,
        b_i,
        h_i,
        prior_block_offset,
        num_blocks_per_seg,
        allocator,
    )

    call2_sbuf_addr = allocator.get_current_address()
    _run_attention_from_sbuf(
        q_hbm,
        k_prior_sbuf,
        v_prior_sbuf,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        seqlen_q,
        num_k_tiles_per_seg * _K_TILE_SZ,
        head_dim,
        sb_p,
        n_grps,
        scale,
        False,
        tp_q,
        allocator,
        kv_used_len=partial_prior_tokens,
    )
    allocator.set_current_address(call2_sbuf_addr)

    softmax_pat = [[n_grps, sb_p], [1, n_grps]]
    o_pat = [[head_dim, sb_p], [1, head_dim]]
    num_free = min(n_grps, _MAX_FREE_TILES)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()
    reduce_one_batch(
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        n_grps,
        head_dim,
        n_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb_buf,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )
    allocator.set_current_address(init_sbuf_addr)


def fused_segmented_attention_impl(
    q_hbm,
    num_batches,
    k_cache,
    v_cache,
    block_tables,
    k_cache_sbuf,
    v_cache_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    prior_tokens_sbuf,
    num_full_prior_segments_i32,
    partial_prior_tokens,
    is_partial_prior_segment,
    is_not_partial_prior_segment,
    active_block_offset,
    prior_block_offset,
    allocator,
    prior_seg_size,
    block_size,
    scale,
    head_dim,
    num_grps,
    num_active_blocks,
    num_k_tiles_active,
    num_v_tiles_active,
    num_blocks_per_seg,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    b_i=0,
    h_i=0,
    tp_q=True,
    tp_out=False,
    load_kv_cache_fn=None,
    attention_cte_fn=None,
    sink=None,
    kvp_offset=None,
    k_pre_transposed=False,
    k_scale_sb=None,
):
    """Fused segmented-attention impl with SBUF aliasing across active/prior passes.

    Uses kv_section_idx=0 so K/V indexing starts at tile 0 for every segment,
    while section_idx controls flash attention accumulation:
      - Active segment: section_idx=0 (init running stats)
      - Prior segments: section_idx=1 (accumulate via _write_back_impl)

    The PV accumulation stays in float32 SBUF across all segments, matching
    _attention_cte's internal flash attention precision.
    """
    orig_addr = allocator.get_current_address()

    kernel_assert(
        kvp_offset == None,
        "qwen_segcte256 KVP mode is not production validated; use the "
        "non-KVP segmented CTE path",
    )
    kernel_assert(
        not k_pre_transposed,
        "qwen_segcte256 supports only k_pre_transposed=False; "
        "the transposed-K path has not been production validated",
    )

    is_kvp = False
    # KVP: compute kvp_offset_active = kvp_offset - prior_tokens_sbuf (for active segment cp_offset)
    # and allocate kvp_offset_prior_sbuf/hbm for per-iteration prior segment cp_offset.
    kvp_offset_active_hbm = None
    kvp_offset_prior_sbuf = None
    kvp_offset_prior_hbm = None
    if is_kvp:
        kvp_offset_active_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        kvp_offset_active_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        nisa.tensor_tensor(dst=kvp_offset_active_sbuf, data1=kvp_offset, data2=prior_tokens_sbuf, op=nl.subtract)
        nisa.dma_copy(dst=kvp_offset_active_hbm, src=kvp_offset_active_sbuf)
        kvp_offset_prior_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        kvp_offset_prior_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    seqlen_q = q_hbm.shape[1] if tp_q else q_hbm.shape[2]
    seqlen_k_active = seqlen_q  # actual tokens, not rounded to tile boundary
    seqlen_k_prior = prior_seg_size  # actual tokens, not rounded to tile boundary

    # num_sections must be > 1 to enable flash attention accumulation path
    max_blocks_per_seq = block_tables.shape[1]
    max_prior_segments = math.ceil(max_blocks_per_seq * block_size / prior_seg_size)
    total_sections = max(max_prior_segments + 1, 2)

    # Prior config: KVP uses causal=True + cp to handle shifted causal mask; non-KVP uses causal=False
    ac_p, atp_p = _make_ac_atp(
        seqlen_q,
        seqlen_k_prior,
        head_dim,
        q_hbm.dtype,
        is_kvp,
        scale,
        tp_q,
        False,
        total_sections,
        use_cp=is_kvp,
        global_cp_deg=1 if is_kvp else None,
    )

    sb_p = atp_p.sb_p
    n_grps = atp_p.num_grps

    # Running buffers (persist across all segments in SBUF)
    bufs = AttnInternalBuffers()
    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)
    bufs.k_scale_sb = k_scale_sb
    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)

    sbuf_outer = allocator.get_current_address()

    # Load sink token into SBUF if provided
    if sink is not None:
        bufs.sink_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
        nisa.dma_copy(dst=bufs.sink_sb[0, 0], src=sink[0, 0])
        stream_shuffle_broadcast(src=bufs.sink_sb, dst=bufs.sink_sb)

    active_stream_tokens = min(prior_seg_size, seqlen_k_active)
    kernel_assert(
        active_stream_tokens % block_size == 0,
        "qwen_segcte256 active streaming requires an active stream chunk divisible by block_size",
    )
    num_active_stream_sections = math.ceil(seqlen_k_active / active_stream_tokens)
    num_blocks_per_active_stream = active_stream_tokens // block_size
    num_k_tiles_per_active_stream = math.ceil(active_stream_tokens / _K_TILE_SZ)
    num_v_tiles_per_active_stream = num_k_tiles_per_active_stream * (_K_TILE_SZ // _V_TILE_SZ)

    active_stream_offset = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
    active_stream_addr = allocator.get_current_address()
    for active_section_idx in range(num_active_stream_sections):
        if active_section_idx == 0:
            nisa.tensor_copy(dst=active_stream_offset, src=active_block_offset)
        else:
            nisa.tensor_scalar(
                dst=active_stream_offset,
                data=active_block_offset,
                op0=nl.add,
                operand0=active_section_idx * num_blocks_per_active_stream,
            )

        load_kv_cache_fn(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf,
            v_cache_sbuf,
            b_i,
            h_i,
            active_stream_offset,
            num_blocks_per_active_stream,
            allocator,
            k_pre_transposed=k_pre_transposed,
        )

        section_offset_active = active_section_idx * active_stream_tokens
        next_section_offset_active = min(section_offset_active + active_stream_tokens, seqlen_k_active)
        ac_a, atp_a = _make_ac_atp(
            seqlen_q,
            next_section_offset_active,
            head_dim,
            q_hbm.dtype,
            True,
            scale,
            tp_q,
            False,
            total_sections,
        )
        atp_a.section_len = active_stream_tokens
        atp_a.num_large_tiles_per_section = div_ceil(active_stream_tokens, _LARGE_TILE_SZ)
        atp_a.num_k_tiles_per_section = num_k_tiles_per_active_stream
        atp_a.num_v_tiles_per_section = num_v_tiles_per_active_stream
        sp_active = SectionParams(
            section_idx=active_section_idx,
            section_offset=section_offset_active,
            section_offset_active=section_offset_active,
            next_section_offset_active=next_section_offset_active,
            section_contains_prefix=False,
            next_section_contains_prefix=False,
            kv_section_idx=active_section_idx,
        )
        allocator.set_current_address(sbuf_outer)
        _allocate_attention_buffers(
            allocator,
            ac_a,
            atp_a,
            bufs,
            sink,
            k_cache_sbuf[:num_k_tiles_per_active_stream],
            v_cache_sbuf[:num_v_tiles_per_active_stream],
        )
        _setup_range_select_bounds(ac_a, atp_a, bufs, allocator, None, None, None, None, batch_id=0)
        sbuf_inner = allocator.get_current_address()
        _run_groups(0, n_grps, ac_a, atp_a, sp_active, bufs, q_hbm, 0, o_prev_hbm, sbuf_inner, sink=sink)
        allocator.set_current_address(active_stream_addr)

    is_partial_reg = nisa.register_alloc()
    nisa.register_load(dst=is_partial_reg, src=is_partial_prior_segment)

    for _ in nl.dynamic_range(0, is_partial_reg):
        load_kv_cache_fn(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf,
            v_cache_sbuf,
            b_i,
            h_i,
            prior_block_offset,
            num_blocks_per_seg,
            allocator,
            k_pre_transposed=k_pre_transposed,
        )
        allocator.set_current_address(sbuf_outer)
        ac_partial, atp_partial = _make_ac_atp(
            seqlen_q,
            seqlen_k_prior,
            head_dim,
            q_hbm.dtype,
            False,
            scale,
            tp_q,
            False,
            total_sections,
        )
        ac_partial.has_kv_used_len = True
        atp_partial.dynamic_sel_mask = True
        _allocate_attention_buffers(
            allocator,
            ac_partial,
            atp_partial,
            bufs,
            sink,
            k_cache_sbuf[:num_k_tiles_per_seg],
            v_cache_sbuf[:num_v_tiles_per_seg],
        )
        _setup_range_select_bounds(
            ac_partial,
            atp_partial,
            bufs,
            allocator,
            None,
            None,
            None,
            None,
            batch_id=0,
            kv_used_len=partial_prior_tokens,
        )
        sp_partial = SectionParams(
            section_idx=1,
            section_offset=0,
            section_offset_active=0,
            next_section_offset_active=seqlen_k_prior,
            section_contains_prefix=False,
            next_section_contains_prefix=False,
            kv_section_idx=0,
        )
        sbuf_inner_partial = allocator.get_current_address()
        _run_groups(
            0,
            n_grps,
            ac_partial,
            atp_partial,
            sp_partial,
            bufs,
            q_hbm,
            0,
            o_prev_hbm,
            sbuf_inner_partial,
            sink=sink,
        )
        allocator.set_current_address(active_stream_addr)

    sm_pat = [[num_grps, sb_p], [1, num_grps]]

    # --- PRIOR SEGMENTS (section_idx=1, kv_section_idx=0, dynamic loop) ---
    # section_idx=1 triggers accumulation: _write_back_impl loads prev output from o_prev_hbm,
    # applies correction factor, adds fresh PV, writes back. Running stats update in SBUF.
    # kv_section_idx=0 ensures K/V indexing starts at tile 0 (each segment's own SBUF data).
    sp_prior = SectionParams(
        section_idx=1,
        section_offset=0,
        section_offset_active=0,
        next_section_offset_active=seqlen_k_prior,
        section_contains_prefix=False,
        next_section_contains_prefix=False,
        kv_section_idx=0,
    )

    prior_offset_save = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
    nisa.tensor_copy(dst=prior_offset_save, src=prior_block_offset)

    num_prior_reg = nisa.register_alloc()
    nisa.register_load(dst=num_prior_reg, src=num_full_prior_segments_i32)

    loop_addr = allocator.get_current_address()

    for _ in nl.dynamic_range(0, num_prior_reg):
        nisa.tensor_scalar(
            dst=prior_block_offset, data=prior_block_offset, op0=nl.subtract, operand0=num_blocks_per_seg
        )

        load_kv_cache_fn(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf,
            v_cache_sbuf,
            b_i,
            h_i,
            prior_block_offset,
            num_blocks_per_seg,
            allocator,
            k_pre_transposed=k_pre_transposed,
        )

        allocator.set_current_address(sbuf_outer)
        _allocate_attention_buffers(allocator, ac_p, atp_p, bufs, sink, k_cache_sbuf, v_cache_sbuf)

        # KVP: compute kvp_offset_prior = kvp_offset - prior_block_offset * block_size
        if is_kvp:
            nisa.tensor_scalar(dst=kvp_offset_prior_sbuf, data=prior_block_offset, op0=nl.multiply, operand0=block_size)
            nisa.tensor_tensor(dst=kvp_offset_prior_sbuf, data1=kvp_offset, data2=kvp_offset_prior_sbuf, op=nl.subtract)
            nisa.dma_copy(dst=kvp_offset_prior_hbm, src=kvp_offset_prior_sbuf)

        prior_cp_offset = kvp_offset_prior_hbm if is_kvp else None

        if is_kvp:
            # KVP: use attention_cte_fn with cp_offset, then reduce into accumulated output
            init_sbuf_addr = allocator.get_current_address()
            attention_cte_fn(
                q_hbm,
                None,
                None,
                scale=scale,
                causal_mask=True,
                tp_q=tp_q,
                tp_k=False,
                tp_out=False,
                cache_softmax=True,
                skip_output_normalization=True,
                k_cache_sbuf=k_cache_sbuf[:num_k_tiles_per_seg],
                v_cache_sbuf=v_cache_sbuf[:num_v_tiles_per_seg],
                out_o_hbm=o_curr_hbm,
                out_neg_max_hbm=neg_max_curr_hbm,
                out_sum_hbm=sum_curr_hbm,
                init_sbuf_addr=init_sbuf_addr,
                cp_offset=prior_cp_offset,
                global_cp_deg=1,
                k_scale_sb=k_scale_sb,
            )
            allocator.set_current_address(init_sbuf_addr)

            # Reduce current segment into accumulated output
            softmax_pat = [[num_grps, sb_p], [1, num_grps]]
            o_pat = [[head_dim, sb_p], [1, head_dim]]
            num_free = min(num_grps, _MAX_FREE_TILES)
            neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            o_prev_sb = allocator.alloc_sbuf_tensor(
                shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
            )
            o_curr_sb = allocator.alloc_sbuf_tensor(
                shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
            )
            o_new_sb = allocator.alloc_sbuf_tensor(
                shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
            )
            batch_loop_addr = allocator.get_current_address()
            reduce_one_batch(
                o_prev_hbm,
                neg_max_prev_hbm,
                sum_prev_hbm,
                o_curr_hbm,
                neg_max_curr_hbm,
                sum_curr_hbm,
                0,
                0,
                num_grps,
                head_dim,
                num_grps,
                sb_p,
                softmax_pat,
                o_pat,
                neg_max_prev_sb,
                sum_prev_sb,
                neg_max_curr_sb,
                sum_curr_sb_buf,
                o_prev_sb,
                o_curr_sb,
                o_new_sb,
                batch_loop_addr,
                allocator,
            )
            # Reload updated stats into SBUF running buffers
            nisa.dma_copy(dst=bufs.mm1_running_max, src=neg_max_prev_hbm.ap(pattern=softmax_pat, offset=0))
            nisa.dma_copy(dst=bufs.exp_running_sum, src=sum_prev_hbm.ap(pattern=softmax_pat, offset=0))
        else:
            _setup_range_select_bounds(ac_p, atp_p, bufs, allocator, None, None, None, None, batch_id=0)
            sbuf_inner_p = allocator.get_current_address()
            _run_groups(0, n_grps, ac_p, atp_p, sp_prior, bufs, q_hbm, 0, o_prev_hbm, sbuf_inner_p)

        allocator.set_current_address(loop_addr)

    # Restore
    nisa.tensor_copy(dst=prior_block_offset, src=prior_offset_save)

    # Write running stats to HBM for caller's normalization
    sm_pat = [[num_grps, sb_p], [1, num_grps]]
    nisa.dma_copy(dst=neg_max_prev_hbm.ap(pattern=sm_pat, offset=0), src=bufs.mm1_running_max)
    nisa.dma_copy(dst=sum_prev_hbm.ap(pattern=sm_pat, offset=0), src=bufs.exp_running_sum)

    allocator.set_current_address(orig_addr)
