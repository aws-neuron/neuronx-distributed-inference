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

"""Full PairformerLayer mega-kernel — SPMD grid=[2] variant.

Extends the z-only mega-kernel (fused_z_ops_spmd.py) to include ALL 7 sub-operations
of a PairformerLayer:

  Step 1:  s = s + PairBiasAttention(s, z)     # O(N^2) attention on s, biased by z
  Step 2:  z = z + TriMulOut(z, pair_mask)      # O(N^3) einsum
  Step 3:  z = z + TriMulIn(z, pair_mask)       # O(N^3) einsum
  Step 4:  z = z + TriAttnStart(z)              # O(N^3) row-wise attention
  Step 5:  z = z + TriAttnEnd(z)                # O(N^3) column-wise attention
  Step 6a: s = s + Transition_s(s)              # SwiGLU FFN on s (C_s=384, hidden=1536)
  Step 6b: z = z + Transition_z(z)              # SwiGLU FFN on z (C_z=128, hidden=512)

Key insight: Steps 1 and 6a operate on s; steps 2-5 and 6b operate on z.
Step 1 reads z (for pair bias) but does not modify it. Steps 2-6b do not read s.
Therefore step 1 and steps 2-5 can execute concurrently if we split SPMD work
accordingly. However, for simplicity and correctness, we run them sequentially
in the initial implementation.

New challenges vs z-only kernel:
  - C_s = 384 = 3 * P_MAX: weight matrices up to [384, 384] and [1536, 384]
    require tiled matmuls across multiple P_MAX-sized chunks
  - PairBiasAttention: head_dim=24, 16 heads, pair bias from z (128→16)
  - Transition_s: hidden_dim=1536 = 12 * P_MAX, large weight matrices
  - Transition_s has NO biases on fc1/fc2/fc3 (same as Transition_z: bias=False)

Hardware: NeuronCore v3 (trn2), LNC=2, 2 physical cores per logical core
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

from fused_z_ops_spmd import (
    P_MAX,
    _layer_norm_tile,
    _matmul_with_w_t,
    _prepare_weight,
    _transition_z_phase_spmd,
    _transpose_to_sbuf,
    _triattn_phase_spmd,
    _trimul_phase_spmd,
)


# ============================================================================
# Helpers for C_s=384 tiled operations
# ============================================================================

C_S_CHUNKS = 3  # C_s=384 = 3 * P_MAX
HIDDEN_S_CHUNKS = 12  # hidden_dim_s=1536 = 12 * P_MAX


# ============================================================================
# Phase 6a: Transition_s (SwiGLU FFN on s, dim=384, hidden=1536)
# SPMD variant: split tiles by pid
# ============================================================================
def _transition_s_phase_spmd(
    s_in_hbm,
    s_out_hbm,
    norm_w_hbm,
    norm_b_hbm,
    fc1_w_hbm,
    fc2_w_hbm,
    fc3_w_hbm,
    N,
    C_s,
    hidden_dim_s,
    eps,
    pid,
):
    """Execute Transition_s: SwiGLU FFN on s (SPMD variant).

    s is [N, C_s=384]. Hidden dim is 1536 = 12*128.
    fc1: [1536, 384] (expansion, SiLU branch) — NO bias
    fc2: [1536, 384] (expansion, gate branch) — NO bias
    fc3: [384, 1536] (projection back) — NO bias
    Same as Transition_z: the Transition class uses bias=False for all linear layers.

    Tiling strategy:
      - s tiles: N // P_MAX tiles of [P_MAX, 384]
      - C_s = 3 * P_MAX = 3 input chunks
      - hidden = 12 * P_MAX = 12 hidden chunks
      - fc1/fc2: for each of 12 output chunks, accumulate across 3 input chunks
      - fc3: for each of 3 output chunks, accumulate across 12 input chunks
    """
    n_tiles = N // P_MAX
    n_in = C_s // P_MAX  # 3
    n_hid = hidden_dim_s // P_MAX  # 12

    # SPMD work split
    tiles_per_engine = n_tiles // 2
    my_tile_start = pid * tiles_per_engine

    # Load LayerNorm weights [P_MAX, C_s] (tiled to [P_MAX, C_s])
    ln_w = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_w, src=norm_w_hbm)
    ln_b = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_b, src=norm_b_hbm)

    # WEIGHT-STATIONARY: Pre-load and transpose ALL weight chunks.
    # fc1: [1536, 384] = 12 output chunks × 3 input chunks
    # fc2: [1536, 384] = same
    # fc3: [384, 1536] = 3 output chunks × 12 input chunks
    # NKI compiler doesn't support mutation, generator expressions, or list
    # comprehensions inside traced code. Must use explicit named variables
    # with if/elif chains inside nl.static_range loops (same pattern as
    # fused_z_ops_spmd.py Transition_z).

    # fc1: [1536, 384] — 12 hidden chunks, each [P_MAX, 384] with 3 input chunks
    # fc2: [1536, 384] — same structure
    # For fc1/fc2: iterate h=0..11 hidden chunks, accumulate across i=0..2 input chunks
    # For fc3: iterate o=0..2 output chunks, accumulate across h=0..11 hidden chunks
    #
    # Since 12*3=36 named variables per matrix is excessive, we restructure:
    # For each hidden chunk h, load s_chunks[i] and compute fc1/fc2 by accumulating
    # across i. The fc1/fc2 weights for a given h are just 3 P_MAX×P_MAX tiles.
    # We pre-load all 12*3=36 tiles per matrix as named vars.
    # But instead of if/elif with 36 branches, we pre-load per-h in the h loop.
    # Wait — we can't do that either because the loop var h can't index anything.
    #
    # Alternative: pass weights as [hidden_dim, C_s] and slice inside the loop.
    # The HBM slicing [h*P_MAX:(h+1)*P_MAX, i*P_MAX:(i+1)*P_MAX] should work
    # since h and i are nl.static_range loop vars (compile-time constants).
    # We just call _prepare_weight directly with HBM slices inside the loop.
    # This means weights are loaded per-tile per-iteration, NOT weight-stationary.
    # Less optimal but correct and compilable.

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        # Load s tile: [P_MAX, C_s=384]
        s_tile = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=s_tile, src=s_in_hbm[tile_start : tile_start + P_MAX, 0:C_s])

        # LayerNorm
        s_normed = _layer_norm_tile(s_tile, ln_w, ln_b, C_s, eps)

        # Split into 3 chunks and transpose each
        s_chunk_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_chunk_0, src=s_normed[0:P_MAX, 0:P_MAX])
        s_chunk_0_t = _transpose_to_sbuf(s_chunk_0)

        s_chunk_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_chunk_1, src=s_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        s_chunk_1_t = _transpose_to_sbuf(s_chunk_1)

        s_chunk_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_chunk_2, src=s_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        s_chunk_2_t = _transpose_to_sbuf(s_chunk_2)

        s_chunks_t = (s_chunk_0_t, s_chunk_1_t, s_chunk_2_t)

        # --- fc1 and fc2: [P_MAX, 384] @ [1536, 384]^T → [P_MAX, 1536] ---
        # For each hidden chunk h (0..11), accumulate across 3 input chunks
        # Then apply SwiGLU: SiLU(fc1) * fc2

        # Accumulate fc3 output across hidden chunks
        fc3_acc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc3_acc_0, value=0.0)
        fc3_acc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc3_acc_1, value=0.0)
        fc3_acc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc3_acc_2, value=0.0)

        fc3_acc = (fc3_acc_0, fc3_acc_1, fc3_acc_2)

        for h in nl.static_range(12):
            # fc1 chunk h: accumulate across 3 input chunks
            fc1_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=fc1_acc, value=0.0)
            fc2_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=fc2_acc, value=0.0)

            for i in nl.static_range(3):
                fc1_w_hi = _prepare_weight(
                    fc1_w_hbm[h * P_MAX : (h + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p1, stationary=s_chunks_t[i], moving=fc1_w_hi)
                p1s = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p1s, src=p1)
                nisa.tensor_tensor(dst=fc1_acc, data1=fc1_acc, data2=p1s, op=nl.add)

                fc2_w_hi = _prepare_weight(
                    fc2_w_hbm[h * P_MAX : (h + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p2, stationary=s_chunks_t[i], moving=fc2_w_hi)
                p2s = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p2s, src=p2)
                nisa.tensor_tensor(dst=fc2_acc, data1=fc2_acc, data2=p2s, op=nl.add)

            # SwiGLU: SiLU(fc1) * fc2 — no bias to add
            fc1_for_silu = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc1_for_silu, src=fc1_acc)

            fc1_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(
                dst=fc1_sig, op=nl.sigmoid, data=fc1_for_silu, bias=None, scale=1.0
            )
            fc1_silu = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=fc1_silu, data1=fc1_for_silu, data2=fc1_sig, op=nl.multiply
            )

            fc2_for_gate = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc2_for_gate, src=fc2_acc)

            swiglu = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=swiglu, data1=fc1_silu, data2=fc2_for_gate, op=nl.multiply
            )

            # fc3: accumulate this hidden chunk's contribution to each output chunk
            sg_t = _transpose_to_sbuf(swiglu)
            for o in nl.static_range(3):
                fc3_w_oh = _prepare_weight(
                    fc3_w_hbm[o * P_MAX : (o + 1) * P_MAX, h * P_MAX : (h + 1) * P_MAX]
                )
                p3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p3, stationary=sg_t, moving=fc3_w_oh)
                p3s = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p3s, src=p3)
                nisa.tensor_tensor(
                    dst=fc3_acc[o], data1=fc3_acc[o], data2=p3s, op=nl.add
                )

        # Residual: store fc3 output + original s
        for o in nl.static_range(3):
            output_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=output_chunk, src=fc3_acc[o])

            # Load original s chunk for residual
            s_orig_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=s_orig_chunk,
                src=s_in_hbm[
                    tile_start : tile_start + P_MAX,
                    o * P_MAX : (o + 1) * P_MAX,
                ],
            )
            s_updated = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=s_updated, data1=s_orig_chunk, data2=output_chunk, op=nl.add
            )
            nisa.dma_copy(
                dst=s_out_hbm[
                    tile_start : tile_start + P_MAX,
                    o * P_MAX : (o + 1) * P_MAX,
                ],
                src=s_updated,
            )


# ============================================================================
# Phase 1: PairBiasAttention (attention on s, biased by z)
# SPMD variant: split work by pid
# ============================================================================
def _pair_bias_attn_phase_spmd(
    s_in_hbm,
    z_in_hbm,
    s_out_hbm,
    mask_hbm,
    # PairBiasAttention weights
    norm_s_w_hbm,
    norm_s_b_hbm,
    norm_z_w_hbm,
    norm_z_b_hbm,
    proj_q_w_hbm,
    proj_q_b_hbm,
    proj_k_w_hbm,
    proj_v_w_hbm,
    proj_z_w_hbm,
    proj_g_w_hbm,
    proj_o_w_hbm,
    # Scratch buffers in shared HBM
    q_buf,  # [N, C_s] bf16 — Q projections
    k_buf,  # [N, C_s] bf16 — K projections
    v_buf,  # [N, C_s] bf16 — V projections
    gate_buf,  # [N, C_s] bf16 — gate values
    z_bias_buf,  # [N, N, H_s] bf16 — pair bias from z
    N,
    C_s,
    C_z,
    H_s,  # 16
    d_s,  # 24
    eps,
    pid,
):
    """Execute PairBiasAttention: multi-head attention on s, biased by z.

    Step 1 of PairformerLayer.

    s: [N, C_s=384], z: [N*N, C_z=128], mask: [N, 1]
    Q/K/V projections: [384, 384], pair bias: z → Linear(128→16) → [N, N, 16]
    Attention: 16 heads, d=24, logits = Q@K^T/sqrt(24) + pair_bias

    Work split:
      - Pass A (Q/K/V/gate projection on s): split N tiles by pid
      - Pass B (pair bias from z): split N*N tiles by pid
      - Pass C (attention): split rows by pid
      - Pass D (output gating + projection): split N tiles by pid
    """
    n_tiles = N // P_MAX
    n_in = C_s // P_MAX  # 3
    Hd_s = H_s * d_s  # 384
    scale_s = 1.0 / (d_s**0.5)

    # SPMD work split for s tiles
    tiles_per_engine = n_tiles // 2
    my_tile_start = pid * tiles_per_engine

    # SPMD work split for z flat tiles
    n_z_flat = N * N
    n_z_tiles = n_z_flat // P_MAX
    z_tiles_per_engine = n_z_tiles // 2
    my_z_tile_start = pid * z_tiles_per_engine

    # SPMD work split for attention rows
    rows_per_engine = N // 2
    my_row_start = pid * rows_per_engine

    # ---- Pass A: LayerNorm(s) + Q/K/V/gate projections ----
    # Load s LayerNorm weights [P_MAX, C_s]
    ln_s_w = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_s_w, src=norm_s_w_hbm)
    ln_s_b = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_s_b, src=norm_s_b_hbm)

    # Pre-load Q/K/V/gate weight chunks: each [384, 384] = 3x3 blocks
    # NKI compiler doesn't support mutation, tuple(), or generator expressions.
    # Load weight tiles directly from HBM inside nl.static_range loops.
    # Since nl.static_range is unrolled, h/i/o become literals and HBM slices
    # resolve to constant offsets at compile time.

    # Pre-load Q bias: [P_MAX, C_s=384] tiled as 3 chunks along free dim
    qb_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qb_0, src=proj_q_b_hbm[0:P_MAX, 0:P_MAX])
    qb_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qb_1, src=proj_q_b_hbm[0:P_MAX, P_MAX : 2 * P_MAX])
    qb_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qb_2, src=proj_q_b_hbm[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
    q_bias_chunks = (qb_0, qb_1, qb_2)

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        # Load s tile [P_MAX, 384]
        s_tile = nl.ndarray((P_MAX, C_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=s_tile, src=s_in_hbm[tile_start : tile_start + P_MAX, 0:C_s])

        s_normed = _layer_norm_tile(s_tile, ln_s_w, ln_s_b, C_s, eps)

        # Split into chunks and transpose
        sn_c0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sn_c0, src=s_normed[0:P_MAX, 0:P_MAX])
        sn_c0_t = _transpose_to_sbuf(sn_c0)

        sn_c1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sn_c1, src=s_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        sn_c1_t = _transpose_to_sbuf(sn_c1)

        sn_c2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sn_c2, src=s_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        sn_c2_t = _transpose_to_sbuf(sn_c2)

        sn_chunks_t = (sn_c0_t, sn_c1_t, sn_c2_t)

        # Q/K/V/gate: each [P_MAX, 384] = 3 output chunks
        for o in nl.static_range(3):
            # Q chunk o
            q_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=q_acc, value=0.0)
            k_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=k_acc, value=0.0)
            v_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=v_acc, value=0.0)
            g_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=g_acc, value=0.0)

            for i in nl.static_range(3):
                q_w_oi = _prepare_weight(
                    proj_q_w_hbm[
                        o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX
                    ]
                )
                pq = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pq, stationary=sn_chunks_t[i], moving=q_w_oi)
                pqs = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pqs, src=pq)
                nisa.tensor_tensor(dst=q_acc, data1=q_acc, data2=pqs, op=nl.add)

                k_w_oi = _prepare_weight(
                    proj_k_w_hbm[
                        o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX
                    ]
                )
                pk = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pk, stationary=sn_chunks_t[i], moving=k_w_oi)
                pks = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pks, src=pk)
                nisa.tensor_tensor(dst=k_acc, data1=k_acc, data2=pks, op=nl.add)

                v_w_oi = _prepare_weight(
                    proj_v_w_hbm[
                        o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX
                    ]
                )
                pv = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pv, stationary=sn_chunks_t[i], moving=v_w_oi)
                pvs = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pvs, src=pv)
                nisa.tensor_tensor(dst=v_acc, data1=v_acc, data2=pvs, op=nl.add)

                g_w_oi = _prepare_weight(
                    proj_g_w_hbm[
                        o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX
                    ]
                )
                pg = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pg, stationary=sn_chunks_t[i], moving=g_w_oi)
                pgs = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pgs, src=pg)
                nisa.tensor_tensor(dst=g_acc, data1=g_acc, data2=pgs, op=nl.add)

            # Add Q bias
            qb_f32 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qb_f32, src=q_bias_chunks[o])
            nisa.tensor_tensor(dst=q_acc, data1=q_acc, data2=qb_f32, op=nl.add)

            # Store Q/K/V/gate chunks to scratch HBM
            q_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_bf16, src=q_acc)
            nisa.dma_copy(
                dst=q_buf[tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=q_bf16,
            )

            k_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_bf16, src=k_acc)
            nisa.dma_copy(
                dst=k_buf[tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=k_bf16,
            )

            v_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_bf16, src=v_acc)
            nisa.dma_copy(
                dst=v_buf[tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=v_bf16,
            )

            # Gate: apply sigmoid before storing
            g_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_bf16, src=g_acc)
            g_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_bf16, bias=None, scale=1.0)
            nisa.dma_copy(
                dst=gate_buf[
                    tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
                src=g_sig,
            )

    # ---- Pass B: Pair bias from z ----
    # z: [N*N, C_z=128], proj_z: [H_s=16, C_z=128]
    # Output: z_bias_buf [N*N, H_s=16]
    # Load z LayerNorm weights
    ln_z_w = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_z_w, src=norm_z_w_hbm)
    ln_z_b = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_z_b, src=norm_z_b_hbm)

    # proj_z: [H_s=16, C_z=128]. Pad to [P_MAX, P_MAX] and transpose for matmul.
    proj_z_raw = nl.ndarray((H_s, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=proj_z_raw, src=proj_z_w_hbm)
    proj_z_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=proj_z_padded, value=0.0)
    nisa.tensor_copy(dst=proj_z_padded[0:H_s, 0:C_z], src=proj_z_raw)
    proj_z_w_t = _transpose_to_sbuf(proj_z_padded)

    # BARRIER: Pass A must complete before Pass B uses q_buf etc.
    # But Pass B doesn't use q_buf, so we can overlap. However both passes
    # are writing to shared HBM buffers, so we barrier after Pass A + B.
    # Actually, Pass B is independent — it only reads z_in_hbm and writes z_bias_buf.
    # We can run it without waiting for Pass A.

    for tile_local in nl.sequential_range(z_tiles_per_engine):
        tile_idx = my_z_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        z_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_tile, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])

        z_normed = _layer_norm_tile(z_tile, ln_z_w, ln_z_b, C_z, eps)
        z_n_t = _transpose_to_sbuf(z_normed)

        # Matmul: [P_MAX, 128] @ [16, 128]^T → [P_MAX, 16]
        bias_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bias_psum, stationary=z_n_t, moving=proj_z_w_t)
        bias_full = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bias_full, src=bias_psum)
        bias_slice = nl.ndarray((P_MAX, H_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bias_slice, src=bias_full[0:P_MAX, 0:H_s])

        nisa.dma_copy(
            dst=z_bias_buf[tile_start : tile_start + P_MAX, 0:H_s],
            src=bias_slice,
        )

    # BARRIER: Both cores must finish Pass A and Pass B before attention
    nisa.core_barrier(data=q_buf, cores=(0, 1))
    nisa.core_barrier(data=z_bias_buf, cores=(0, 1))

    # ---- Pass C: Multi-head attention on s with pair bias ----
    # Q/K/V in q_buf/k_buf/v_buf: [N, 384] = [N, 16*24]
    # z_bias_buf: [N*N, 16] = [N, N, 16]
    # For each head h (0..15): Q_h = Q[:, h*24:(h+1)*24], etc.
    # Attention: logits[j, k] = Q_h[j,:] @ K_h[k,:]^T / sqrt(24) + bias[j, k, h]
    # This is standard attention on N positions (not row-wise like TriAttn)

    # Strides for Q/K/V: shape [N, H_s*d_s=384], stored contiguously
    s_stride_row = C_s  # stride between positions

    # Strides for bias: shape [N, N, H_s=16], stored as [N*N, 16]
    bias_s_stride_q = N * H_s  # stride along query dim
    bias_s_stride_k = H_s  # stride along key dim

    # Output projection weights loaded inline from HBM inside loops below

    for h in nl.affine_range(H_s):
        hd_start = h * d_s

        for j_tile in nl.affine_range(n_tiles):
            j_start = j_tile * P_MAX

            # Determine if this tile belongs to this engine
            # For simplicity in attention, split by j_tile (output query tile)
            # Both engines process all heads but different j_tiles
            # We do this by splitting the j_tile loop rather than the row loop
            # since attention here is over the full N (not per-row)

            # Load Q tile: [P_MAX, d_s=24] from q_buf
            q_tile = nl.ndarray((P_MAX, d_s), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=q_tile,
                src=q_buf.ap(
                    pattern=[[s_stride_row, P_MAX], [1, d_s]],
                    offset=j_start * s_stride_row + hd_start,
                ),
            )

            # Pad Q to [P_MAX, P_MAX] and transpose for nc_matmul
            q_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=q_padded, value=0.0)
            nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:d_s], src=q_tile)
            q_t = _transpose_to_sbuf(q_padded)

            # Online softmax accumulators
            m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=m_prev, value=-1e30)
            l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=l_prev, value=0.0)
            o_acc = nl.ndarray((P_MAX, d_s), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=o_acc, value=0.0)

            for k_tile_idx in nl.sequential_range(n_tiles):
                k_start = k_tile_idx * P_MAX

                # Load K tile: [P_MAX, d_s=24]
                k_tile_sb = nl.ndarray((P_MAX, d_s), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=k_tile_sb,
                    src=k_buf.ap(
                        pattern=[[s_stride_row, P_MAX], [1, d_s]],
                        offset=k_start * s_stride_row + hd_start,
                    ),
                )

                # Load V tile: [P_MAX, d_s=24]
                v_tile_sb = nl.ndarray((P_MAX, d_s), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=v_tile_sb,
                    src=v_buf.ap(
                        pattern=[[s_stride_row, P_MAX], [1, d_s]],
                        offset=k_start * s_stride_row + hd_start,
                    ),
                )

                # Load pair bias tile: [P_MAX, P_MAX] from z_bias_buf
                # z_bias_buf is [N*N, H_s] = [N, N, H_s]
                # bias[j, k, h] at offset j*N*H_s + k*H_s + h
                bias_tile = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=bias_tile,
                    src=z_bias_buf.ap(
                        pattern=[[bias_s_stride_q, P_MAX], [bias_s_stride_k, P_MAX]],
                        offset=j_start * bias_s_stride_q
                        + k_start * bias_s_stride_k
                        + h,
                    ),
                )

                # Q @ K^T
                k_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.memset(dst=k_padded, value=0.0)
                nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:d_s], src=k_tile_sb)
                k_t = _transpose_to_sbuf(k_padded)

                logits_psum = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)
                logits = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=logits, src=logits_psum)

                # Scale
                logits_scaled = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=logits_scaled,
                    data=logits,
                    op0=nl.multiply,
                    operand0=scale_s,
                    engine=nisa.vector_engine,
                )

                # Add pair bias
                bias_fp32 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=bias_fp32, src=bias_tile)
                logits_biased = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=logits_biased,
                    data1=logits_scaled,
                    data2=bias_fp32,
                    op=nl.add,
                )

                # Online softmax
                tile_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=tile_max, op=nl.maximum, data=logits_biased, axis=1
                )

                m_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=m_new, data1=m_prev, data2=tile_max, op=nl.maximum
                )

                m_diff = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=m_diff, data1=m_prev, data2=m_new, op=nl.subtract
                )
                correction = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=correction, op=nl.exp, data=m_diff, bias=None, scale=1.0
                )

                logits_shifted = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=logits_shifted,
                    data=logits_biased,
                    op0=nl.subtract,
                    operand0=m_new,
                    engine=nisa.vector_engine,
                )
                exp_logits = nl.ndarray(
                    (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(
                    dst=exp_logits,
                    op=nl.exp,
                    data=logits_shifted,
                    bias=None,
                    scale=1.0,
                )

                l_corrected = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=l_corrected, data1=l_prev, data2=correction, op=nl.multiply
                )
                tile_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_logits, axis=1)
                l_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=l_new, data1=l_corrected, data2=tile_sum, op=nl.add
                )

                o_scaled = nl.ndarray((P_MAX, d_s), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=o_scaled,
                    data=o_acc,
                    op0=nl.multiply,
                    operand0=correction,
                    engine=nisa.vector_engine,
                )

                # exp_logits @ V
                exp_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=exp_bf16, src=exp_logits)
                exp_t = _transpose_to_sbuf(exp_bf16)

                pv_psum = nl.ndarray((P_MAX, d_s), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)
                pv_sbuf = nl.ndarray((P_MAX, d_s), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                nisa.tensor_tensor(dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add)
                nisa.tensor_copy(dst=m_prev, src=m_new)
                nisa.tensor_copy(dst=l_prev, src=l_new)

            # Finalize: o = o_acc / l
            inv_l = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_l, data=l_prev)
            o_final = nl.ndarray((P_MAX, d_s), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=o_final,
                data=o_acc,
                op0=nl.multiply,
                operand0=inv_l,
                engine=nisa.vector_engine,
            )

            # Store attention output back to q_buf (reuse buffer)
            # q_buf[j_start:j_start+P_MAX, hd_start:hd_start+d_s]
            o_out = nl.ndarray((P_MAX, d_s), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_out, src=o_final)
            nisa.dma_copy(
                dst=q_buf.ap(
                    pattern=[[s_stride_row, P_MAX], [1, d_s]],
                    offset=j_start * s_stride_row + hd_start,
                ),
                src=o_out,
            )

    # BARRIER: attention output must be complete before output projection
    nisa.core_barrier(data=q_buf, cores=(0, 1))

    # ---- Pass D: Output gating + projection + residual ----
    # attn_output in q_buf: [N, 384]
    # gate in gate_buf: [N, 384] (already sigmoided)
    # result = proj_o(gate * attn_output) + s_input

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        # Load gate and attention output, apply gating per chunk
        gc0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gc0, src=q_buf[tile_start : tile_start + P_MAX, 0:P_MAX])
        gg0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gg0, src=gate_buf[tile_start : tile_start + P_MAX, 0:P_MAX])
        gated_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated_0, data1=gg0, data2=gc0, op=nl.multiply)
        gated_0_t = _transpose_to_sbuf(gated_0)

        gc1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gc1, src=q_buf[tile_start : tile_start + P_MAX, P_MAX : 2 * P_MAX]
        )
        gg1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gg1, src=gate_buf[tile_start : tile_start + P_MAX, P_MAX : 2 * P_MAX]
        )
        gated_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated_1, data1=gg1, data2=gc1, op=nl.multiply)
        gated_1_t = _transpose_to_sbuf(gated_1)

        gc2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gc2, src=q_buf[tile_start : tile_start + P_MAX, 2 * P_MAX : 3 * P_MAX]
        )
        gg2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gg2,
            src=gate_buf[tile_start : tile_start + P_MAX, 2 * P_MAX : 3 * P_MAX],
        )
        gated_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated_2, data1=gg2, data2=gc2, op=nl.multiply)
        gated_2_t = _transpose_to_sbuf(gated_2)

        gated_chunks_t = (gated_0_t, gated_1_t, gated_2_t)

        # Output projection: [P_MAX, 384] @ [384, 384]^T → [P_MAX, 384]
        for o in nl.static_range(3):
            proj_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=proj_acc, value=0.0)
            for i in nl.static_range(3):
                o_w_oi = _prepare_weight(
                    proj_o_w_hbm[
                        o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX
                    ]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=gated_chunks_t[i], moving=o_w_oi)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=proj_acc, data1=proj_acc, data2=ps, op=nl.add)

            proj_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=proj_bf16, src=proj_acc)

            # Residual: s_updated = s_input + proj_output
            s_orig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=s_orig,
                src=s_in_hbm[
                    tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
            )
            s_updated = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s_updated, data1=s_orig, data2=proj_bf16, op=nl.add)
            nisa.dma_copy(
                dst=s_out_hbm[
                    tile_start : tile_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
                src=s_updated,
            )


# ============================================================================
# Main Entry Point: Full PairformerLayer — SPMD grid=[2]
# ============================================================================
@nki.jit
def full_pairformer_layer_spmd(
    # Main tensors
    s,  # [N, C_s=384] bf16
    z,  # [N*N, C_z=128] bf16
    pair_mask,  # [N*N, 1] bf16
    mask,  # [N, 1] bf16 (for PairBiasAttention masking — unused for now)
    # PairBiasAttention weights (Step 1)
    pba_norm_s_w,
    pba_norm_s_b,
    pba_norm_z_w,
    pba_norm_z_b,
    pba_q_w,
    pba_q_b,
    pba_k_w,
    pba_v_w,
    pba_z_w,
    pba_g_w,
    pba_o_w,
    # TriMulOut weights (Step 2)
    tmul_out_norm_in_w,
    tmul_out_norm_in_b,
    tmul_out_p_in_w,
    tmul_out_g_in_w,
    tmul_out_norm_out_w,
    tmul_out_norm_out_b,
    tmul_out_p_out_w,
    tmul_out_g_out_w,
    # TriMulIn weights (Step 3)
    tmul_in_norm_in_w,
    tmul_in_norm_in_b,
    tmul_in_p_in_w,
    tmul_in_g_in_w,
    tmul_in_norm_out_w,
    tmul_in_norm_out_b,
    tmul_in_p_out_w,
    tmul_in_g_out_w,
    # TriAttnStart weights (Step 4)
    tatt_s_ln_w,
    tatt_s_ln_b,
    tatt_s_bias_proj_w,
    tatt_s_q_w,
    tatt_s_k_w,
    tatt_s_v_w,
    tatt_s_g_w,
    tatt_s_o_w,
    # TriAttnEnd weights (Step 5)
    tatt_e_ln_w,
    tatt_e_ln_b,
    tatt_e_bias_proj_w,
    tatt_e_q_w,
    tatt_e_k_w,
    tatt_e_v_w,
    tatt_e_g_w,
    tatt_e_o_w,
    # Transition_s weights (Step 6a) — no bias on fc1/fc2/fc3 (bias=False)
    trans_s_norm_w,
    trans_s_norm_b,
    trans_s_fc1_w,
    trans_s_fc2_w,
    trans_s_fc3_w,
    # Transition_z weights (Step 6b)
    trans_z_norm_w,
    trans_z_norm_b,
    trans_z_fc1_w,
    trans_z_fc2_w,
    trans_z_fc3_w,
    # Pre-allocated scratch buffers (external shared HBM)
    scratch_buf,  # [6 * N*N, C_z] bf16 — for z operations
    bias_buf,  # [N*N, H_z=4] bf16 — for TriAttn bias
    s_scratch_q,  # [N, C_s=384] bf16 — Q buffer / attn output for PBA
    s_scratch_k,  # [N, C_s=384] bf16 — K buffer
    s_scratch_v,  # [N, C_s=384] bf16 — V buffer
    s_scratch_gate,  # [N, C_s=384] bf16 — gate buffer
    z_bias_scratch,  # [N*N, H_s=16] bf16 — pair bias from z
    s_intermediate,  # [N, C_s=384] bf16 — s after step 1 (before step 6a)
    # Constants
    N: int = 256,
    C_s: int = 384,
    C_z: int = 128,
    H_z: int = 4,
    d_z: int = 32,
    H_s: int = 16,
    d_s: int = 24,
    eps: float = 1e-5,
):
    """Full PairformerLayer mega-kernel — SPMD grid=[2].

    Executes all 7 sub-operations of a PairformerLayer in a single kernel:
      Step 1:  s = s + PairBiasAttn(s, z)
      Step 2:  z = z + TriMulOut(z)
      Step 3:  z = z + TriMulIn(z)
      Step 4:  z = z + TriAttnStart(z)
      Step 5:  z = z + TriAttnEnd(z)
      Step 6a: s = s + Transition_s(s)
      Step 6b: z = z + Transition_z(z)

    Returns: (s_out, z_out)
    """
    # SPMD grid=[2] splits s tiles as N//P_MAX//2 per engine.
    # At N=128 (1 tile), tiles_per_engine=0 and s operations are silently skipped,
    # producing all-zero s_out. Require N >= 256 until a single-engine fallback is added.
    assert N >= 256, (
        f"N={N} too small for SPMD grid=[2]: s has only {N // 128} tile(s), "
        f"need >= 2. Use N >= 256."
    )

    pid = nl.program_id(0)

    hidden_dim_z = 4 * C_z  # 512
    hidden_dim_s = 4 * C_s  # 1536
    n_flat = N * N

    s_out = nl.ndarray((N, C_s), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    z_out = nl.ndarray((n_flat, C_z), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # z-operations scratch offsets (same as fused_z_ops_spmd)
    off_a = 0 * n_flat
    off_b = 1 * n_flat
    off_gate = 2 * n_flat
    off_d = 3 * n_flat
    off_z1 = 4 * n_flat
    off_z2 = 5 * n_flat

    # ================================================================
    # Step 1: PairBiasAttention (s = s + PBA(s, z))
    # ================================================================
    _pair_bias_attn_phase_spmd(
        s_in_hbm=s,
        z_in_hbm=z,
        s_out_hbm=s_intermediate,
        mask_hbm=mask,
        norm_s_w_hbm=pba_norm_s_w,
        norm_s_b_hbm=pba_norm_s_b,
        norm_z_w_hbm=pba_norm_z_w,
        norm_z_b_hbm=pba_norm_z_b,
        proj_q_w_hbm=pba_q_w,
        proj_q_b_hbm=pba_q_b,
        proj_k_w_hbm=pba_k_w,
        proj_v_w_hbm=pba_v_w,
        proj_z_w_hbm=pba_z_w,
        proj_g_w_hbm=pba_g_w,
        proj_o_w_hbm=pba_o_w,
        q_buf=s_scratch_q,
        k_buf=s_scratch_k,
        v_buf=s_scratch_v,
        gate_buf=s_scratch_gate,
        z_bias_buf=z_bias_scratch,
        N=N,
        C_s=C_s,
        C_z=C_z,
        H_s=H_s,
        d_s=d_s,
        eps=eps,
        pid=pid,
    )

    # BARRIER: Step 1 must complete before Step 6a reads s_intermediate
    nisa.core_barrier(data=s_intermediate, cores=(0, 1))

    # ================================================================
    # Steps 2-5: z operations (same as fused_z_ops_spmd)
    # ================================================================

    # Step 2: TriMulOut
    _trimul_phase_spmd(
        z_in_hbm=z,
        buf=scratch_buf,
        off_out=off_z1,
        off_gate=off_gate,
        off_a=off_a,
        off_b=off_b,
        off_result=off_d,
        pair_mask_hbm=pair_mask,
        norm_in_w=tmul_out_norm_in_w,
        norm_in_b=tmul_out_norm_in_b,
        p_in_w=tmul_out_p_in_w,
        g_in_w=tmul_out_g_in_w,
        norm_out_w=tmul_out_norm_out_w,
        norm_out_b=tmul_out_norm_out_b,
        p_out_w=tmul_out_p_out_w,
        g_out_w=tmul_out_g_out_w,
        N=N,
        C_z=C_z,
        eps=eps,
        is_incoming=False,
        pid=pid,
    )
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Step 3: TriMulIn
    _trimul_phase_spmd(
        z_in_hbm=scratch_buf[off_z1 : off_z1 + n_flat, 0:C_z],
        buf=scratch_buf,
        off_out=off_z2,
        off_gate=off_gate,
        off_a=off_a,
        off_b=off_b,
        off_result=off_d,
        pair_mask_hbm=pair_mask,
        norm_in_w=tmul_in_norm_in_w,
        norm_in_b=tmul_in_norm_in_b,
        p_in_w=tmul_in_p_in_w,
        g_in_w=tmul_in_g_in_w,
        norm_out_w=tmul_in_norm_out_w,
        norm_out_b=tmul_in_norm_out_b,
        p_out_w=tmul_in_p_out_w,
        g_out_w=tmul_in_g_out_w,
        N=N,
        C_z=C_z,
        eps=eps,
        is_incoming=True,
        pid=pid,
    )
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Step 4: TriAttnStart
    _triattn_phase_spmd(
        z_in_hbm=scratch_buf[off_z2 : off_z2 + n_flat, 0:C_z],
        buf=scratch_buf,
        off_out=off_z1,
        off_q=off_a,
        off_k=off_b,
        off_v=off_gate,
        bias_buf=bias_buf,
        pair_mask_hbm=pair_mask,
        ln_w_hbm=tatt_s_ln_w,
        ln_b_hbm=tatt_s_ln_b,
        bias_proj_w_hbm=tatt_s_bias_proj_w,
        q_w_hbm=tatt_s_q_w,
        k_w_hbm=tatt_s_k_w,
        v_w_hbm=tatt_s_v_w,
        gate_w_hbm=tatt_s_g_w,
        out_w_hbm=tatt_s_o_w,
        N=N,
        C_z=C_z,
        H=H_z,
        d=d_z,
        eps=eps,
        is_ending=False,
        pid=pid,
    )
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Step 5: TriAttnEnd
    _triattn_phase_spmd(
        z_in_hbm=scratch_buf[off_z1 : off_z1 + n_flat, 0:C_z],
        buf=scratch_buf,
        off_out=off_z2,
        off_q=off_a,
        off_k=off_b,
        off_v=off_gate,
        bias_buf=bias_buf,
        pair_mask_hbm=pair_mask,
        ln_w_hbm=tatt_e_ln_w,
        ln_b_hbm=tatt_e_ln_b,
        bias_proj_w_hbm=tatt_e_bias_proj_w,
        q_w_hbm=tatt_e_q_w,
        k_w_hbm=tatt_e_k_w,
        v_w_hbm=tatt_e_v_w,
        gate_w_hbm=tatt_e_g_w,
        out_w_hbm=tatt_e_o_w,
        N=N,
        C_z=C_z,
        H=H_z,
        d=d_z,
        eps=eps,
        is_ending=True,
        pid=pid,
    )
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # ================================================================
    # Step 6a: Transition_s (s = s_intermediate + Trans_s(s_intermediate))
    # ================================================================
    _transition_s_phase_spmd(
        s_in_hbm=s_intermediate,
        s_out_hbm=s_out,
        norm_w_hbm=trans_s_norm_w,
        norm_b_hbm=trans_s_norm_b,
        fc1_w_hbm=trans_s_fc1_w,
        fc2_w_hbm=trans_s_fc2_w,
        fc3_w_hbm=trans_s_fc3_w,
        N=N,
        C_s=C_s,
        hidden_dim_s=hidden_dim_s,
        eps=eps,
        pid=pid,
    )

    # ================================================================
    # Step 6b: Transition_z (same as fused_z_ops_spmd)
    # ================================================================
    _transition_z_phase_spmd(
        z_in_hbm=scratch_buf[off_z2 : off_z2 + n_flat, 0:C_z],
        z_out_hbm=z_out,
        norm_w_hbm=trans_z_norm_w,
        norm_b_hbm=trans_z_norm_b,
        fc1_w_hbm=trans_z_fc1_w,
        fc2_w_hbm=trans_z_fc2_w,
        fc3_w_hbm=trans_z_fc3_w,
        N=N,
        C_z=C_z,
        hidden_dim=hidden_dim_z,
        eps=eps,
        pid=pid,
    )

    return s_out, z_out
