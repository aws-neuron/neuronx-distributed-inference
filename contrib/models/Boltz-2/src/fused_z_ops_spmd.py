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

"""Fused z-operations mega-kernel — SPMD grid=[2] variant.

This is a variant of fused_z_ops_seq.py that uses NKI SPMD with grid=[2]
to split work across the 2 physical NeuronCores within one logical core
(LNC=2 mode on trn2). Each physical core executes the same kernel code
but operates on a different subset of tiles/iterations.

Key differences from fused_z_ops_seq.py:
  - scratch_buf and bias_buf use shared_hbm (not private_hbm) so both
    cores can read each other's phase outputs
  - Each phase function accepts a `pid` parameter and splits work by pid
  - nisa.core_barrier() is inserted between phases where phase N+1 reads
    phase N's output from shared HBM
  - Main entry uses nl.program_id(0) and is launched with [2] grid

Work splitting strategy:
  - Flat tile loops: pid=0 processes tiles 0..n/2-1, pid=1 does n/2..n-1
  - TriMul Pass 1b (d-loop): pid=0 does d=0..C_z/2-1, pid=1 does C_z/2..C_z-1
  - TriAttn Pass 3b (i_row): pid=0 does rows 0..N/2-1, pid=1 does N/2..N-1
  - All phases are embarrassingly parallel within each phase; no collectives needed

Hardware: NeuronCore v3 (trn2), LNC=2, 2 physical cores per logical core
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


# ============================================================================
# Helper: LayerNorm on a tile [P_MAX, F] in SBUF
# ============================================================================
def _layer_norm_tile(x_tile, weight_tiled, bias_tiled, F, eps=1e-5):
    """LayerNorm on a tile in SBUF.

    Computes: (x - mean) / sqrt(var + eps) * weight + bias
    Reduces over the free dimension (axis 1, size F).

    Args:
        x_tile: [P_MAX, F] bf16 in SBUF
        weight_tiled: [P_MAX, F] bf16 -- pre-tiled LN weight (each row identical)
        bias_tiled:   [P_MAX, F] bf16 -- pre-tiled LN bias (each row identical)
        F: int, free dimension size
        eps: float

    Returns:
        normalized: [P_MAX, F] bf16 in SBUF
    """
    inv_F = 1.0 / float(F)

    # Cast to float32
    x_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_f32, src=x_tile)

    # Mean
    sum_x = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_x, op=nl.add, data=x_f32, axis=1)
    mean = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_F, engine=nisa.vector_engine
    )

    # Centered
    centered = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=centered,
        data=x_f32,
        op0=nl.subtract,
        operand0=mean,
        engine=nisa.vector_engine,
    )

    # Variance
    sq = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=centered, data2=centered, op=nl.multiply)
    sum_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=sq, axis=1)
    var = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=var, data=sum_sq, op0=nl.multiply, operand0=inv_F, engine=nisa.vector_engine
    )

    # rsqrt(var + eps)
    var_eps = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=var_eps, data=var, op0=nl.add, operand0=eps, engine=nisa.vector_engine
    )
    rsqrt_std = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rsqrt_std, op=nl.rsqrt, data=var_eps, bias=None, scale=1.0)

    # Normalize
    normalized_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=normalized_f32,
        data=centered,
        op0=nl.multiply,
        operand0=rsqrt_std,
        engine=nisa.vector_engine,
    )

    # Scale by weight + bias (both [P_MAX, F], pre-tiled on host)
    weight_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=weight_f32, src=weight_tiled)
    scaled = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=scaled, data1=normalized_f32, data2=weight_f32, op=nl.multiply
    )

    bias_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bias_f32, src=bias_tiled)
    result_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=result_f32, data1=scaled, data2=bias_f32, op=nl.add)

    result = nl.ndarray((P_MAX, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_f32)
    return result


# ============================================================================
# Helper: matmul x @ W^T where x [P_MAX, K], W [M, K], both K=P_MAX
# Returns [P_MAX, M] in bf16 on SBUF
# ============================================================================
def _linear_128x128(x_t, w_hbm):
    """Compute x @ W^T for [P_MAX, 128] @ [128, 128] -> [P_MAX, 128].

    Args:
        x_t: [P_MAX, P_MAX] bf16 in SBUF -- already transposed x
        w_hbm: [P_MAX, P_MAX] bf16 in HBM -- weight matrix

    x_t is the nc_transpose of x. nc_matmul(stationary=x_t, moving=w_t)
    computes x_t^T @ w_t = x @ W^T when w_t = nc_transpose(W).
    """
    w = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    w_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=w_t_psum, data=w)
    w_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_t, src=w_t_psum)

    result_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=result_psum, stationary=x_t, moving=w_t)
    result = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_psum)
    return result


def _prepare_weight(w_hbm):
    """Load and transpose a [P_MAX, P_MAX] weight matrix from HBM to SBUF.

    Call ONCE before a tile loop, then pass the result to _matmul_with_w_t
    inside the loop. Eliminates redundant weight DMA + transpose per tile.

    Args:
        w_hbm: [P_MAX, P_MAX] bf16 in HBM -- weight matrix

    Returns:
        w_t: [P_MAX, P_MAX] bf16 in SBUF -- transposed weight, ready for matmul
    """
    w = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    w_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=w_t_psum, data=w)
    w_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_t, src=w_t_psum)
    return w_t


def _matmul_with_w_t(x_t, w_t):
    """Compute x @ W^T using a pre-transposed weight already in SBUF.

    Args:
        x_t: [P_MAX, P_MAX] bf16 in SBUF -- already transposed activation
        w_t: [P_MAX, P_MAX] bf16 in SBUF -- pre-transposed weight (from _prepare_weight)

    Returns:
        result: [P_MAX, P_MAX] bf16 in SBUF -- x @ W^T
    """
    result_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=result_psum, stationary=x_t, moving=w_t)
    result = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_psum)
    return result


def _transpose_to_sbuf(x):
    """nc_transpose x from SBUF -> PSUM -> SBUF."""
    x_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=x_t_psum, data=x)
    x_t = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_t, src=x_t_psum)
    return x_t


# ============================================================================
# Phase 1/2: Triangle Multiplication (Outgoing / Incoming)
# SPMD variant: work split by pid
# ============================================================================
def _trimul_phase_spmd(
    z_in_hbm,
    buf,
    off_out,
    off_gate,
    off_a,
    off_b,
    off_result,
    pair_mask_hbm,
    norm_in_w,
    norm_in_b,
    p_in_w,
    g_in_w,
    norm_out_w,
    norm_out_b,
    p_out_w,
    g_out_w,
    N,
    C_z,
    eps,
    is_incoming,
    pid,
):
    """Execute one triangle multiplication phase (SPMD variant).

    Work is split by pid:
      - Pass 1a (projection): pid processes its half of flat tiles
      - Pass 1b (einsum): pid processes its half of d iterations
      - Pass 1c (output): pid processes its half of flat tiles
    """
    n_flat = N * N
    n_tiles_flat = n_flat // P_MAX
    n_tiles_spatial = N // P_MAX
    stride_i = N * C_z
    stride_k = C_z

    # SPMD work split for flat tile loops
    tiles_per_engine = n_tiles_flat // 2
    my_tile_start = pid * tiles_per_engine

    # SPMD work split for d-loop
    d_per_engine = C_z // 2
    my_d_start = pid * d_per_engine

    # ---- Pass 1a: Projection (split tiles by pid) ----
    ln_w = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_w, src=norm_in_w)
    ln_b = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_b, src=norm_in_b)

    # WEIGHT-STATIONARY: Pre-load and transpose all 4 projection weights ONCE.
    p_in_w_t_0 = _prepare_weight(p_in_w[0:C_z, 0:C_z])
    p_in_w_t_1 = _prepare_weight(p_in_w[C_z : 2 * C_z, 0:C_z])
    g_in_w_t_0 = _prepare_weight(g_in_w[0:C_z, 0:C_z])
    g_in_w_t_1 = _prepare_weight(g_in_w[C_z : 2 * C_z, 0:C_z])

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        z_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_tile, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])

        z_normed = _layer_norm_tile(z_tile, ln_w, ln_b, C_z, eps)

        nisa.dma_copy(
            dst=buf[off_gate + tile_start : off_gate + tile_start + P_MAX, 0:C_z],
            src=z_normed,
        )

        z_n_t = _transpose_to_sbuf(z_normed)

        for half in nl.static_range(2):
            p_w_t = p_in_w_t_0 if half == 0 else p_in_w_t_1
            g_w_t = g_in_w_t_0 if half == 0 else g_in_w_t_1
            proj_tile = _matmul_with_w_t(z_n_t, p_w_t)
            gate_tile = _matmul_with_w_t(z_n_t, g_w_t)

            gate_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(
                dst=gate_sig, op=nl.sigmoid, data=gate_tile, bias=None, scale=1.0
            )
            gated = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gated, data1=proj_tile, data2=gate_sig, op=nl.multiply
            )

            mask_tile = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            mask_tile_bf16 = nl.ndarray((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=mask_tile_bf16,
                src=pair_mask_hbm[tile_start : tile_start + P_MAX, 0:1],
            )
            nisa.tensor_copy(dst=mask_tile, src=mask_tile_bf16)
            gated_f32 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gated_f32, src=gated)
            masked_f32 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=masked_f32,
                data=gated_f32,
                op0=nl.multiply,
                operand0=mask_tile,
                engine=nisa.vector_engine,
            )
            masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=masked, src=masked_f32)

            if half == 0:
                nisa.dma_copy(
                    dst=buf[off_a + tile_start : off_a + tile_start + P_MAX, 0:C_z],
                    src=masked,
                )
            else:
                nisa.dma_copy(
                    dst=buf[off_b + tile_start : off_b + tile_start + P_MAX, 0:C_z],
                    src=masked,
                )

    # BARRIER: Both cores must finish pass 1a before pass 1b reads A and B
    nisa.core_barrier(data=buf, cores=(0, 1))

    # ---- Pass 1b: Matmul (split d iterations by pid) ----
    for d_local in nl.sequential_range(d_per_engine):
        d = my_d_start + d_local

        for i_tile in nl.affine_range(n_tiles_spatial):
            i_start = i_tile * P_MAX
            for j_tile in nl.affine_range(n_tiles_spatial):
                j_start = j_tile * P_MAX

                acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc, value=0.0)

                for k_tile_idx in nl.sequential_range(n_tiles_spatial):
                    k_start = k_tile_idx * P_MAX

                    a_tile = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    b_tile = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )

                    if is_incoming:
                        nisa.dma_copy(
                            dst=a_tile,
                            src=buf.ap(
                                pattern=[[stride_k, P_MAX], [stride_i, P_MAX]],
                                offset=off_a * C_z
                                + k_start * stride_i
                                + i_start * stride_k
                                + d,
                            ),
                        )
                        nisa.dma_copy(
                            dst=b_tile,
                            src=buf.ap(
                                pattern=[[stride_k, P_MAX], [stride_i, P_MAX]],
                                offset=off_b * C_z
                                + k_start * stride_i
                                + j_start * stride_k
                                + d,
                            ),
                        )
                    else:
                        nisa.dma_copy(
                            dst=a_tile,
                            src=buf.ap(
                                pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                                offset=off_a * C_z
                                + i_start * stride_i
                                + k_start * stride_k
                                + d,
                            ),
                        )
                        nisa.dma_copy(
                            dst=b_tile,
                            src=buf.ap(
                                pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                                offset=off_b * C_z
                                + j_start * stride_i
                                + k_start * stride_k
                                + d,
                            ),
                        )

                    a_t = _transpose_to_sbuf(a_tile)
                    b_t = _transpose_to_sbuf(b_tile)

                    partial_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=partial_psum, stationary=a_t, moving=b_t)
                    partial = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=partial, src=partial_psum)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=partial, op=nl.add)

                out_tile = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile, src=acc)
                nisa.dma_copy(
                    dst=buf.ap(
                        pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                        offset=off_result * C_z
                        + i_start * stride_i
                        + j_start * stride_k
                        + d,
                    ),
                    src=out_tile,
                )

    # BARRIER: Both cores must finish pass 1b before pass 1c reads result
    nisa.core_barrier(data=buf, cores=(0, 1))

    # ---- Pass 1c: Output processing + residual (split tiles by pid) ----
    ln_out_w = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_out_w, src=norm_out_w)
    ln_out_b = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_out_b, src=norm_out_b)

    # WEIGHT-STATIONARY
    p_out_w_t = _prepare_weight(p_out_w)
    g_out_w_t = _prepare_weight(g_out_w)

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        r_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=r_tile,
            src=buf[off_result + tile_start : off_result + tile_start + P_MAX, 0:C_z],
        )

        r_normed = _layer_norm_tile(r_tile, ln_out_w, ln_out_b, C_z, eps)
        r_n_t = _transpose_to_sbuf(r_normed)

        proj_out = _matmul_with_w_t(r_n_t, p_out_w_t)

        z_normed_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=z_normed_tile,
            src=buf[off_gate + tile_start : off_gate + tile_start + P_MAX, 0:C_z],
        )
        zn_t = _transpose_to_sbuf(z_normed_tile)

        gate_out = _matmul_with_w_t(zn_t, g_out_w_t)
        gate_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(
            dst=gate_sig, op=nl.sigmoid, data=gate_out, bias=None, scale=1.0
        )

        output_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=output_tile, data1=proj_out, data2=gate_sig, op=nl.multiply
        )

        z_orig = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_orig, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])
        z_updated = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z_updated, data1=z_orig, data2=output_tile, op=nl.add)
        nisa.dma_copy(
            dst=buf[off_out + tile_start : off_out + tile_start + P_MAX, 0:C_z],
            src=z_updated,
        )


# ============================================================================
# Phase 3/4: Triangle Attention (Starting / Ending Node)
# SPMD variant: work split by pid
# ============================================================================
def _triattn_phase_spmd(
    z_in_hbm,
    buf,
    off_out,
    off_q,
    off_k,
    off_v,
    bias_buf,
    pair_mask_hbm,
    ln_w_hbm,
    ln_b_hbm,
    bias_proj_w_hbm,
    q_w_hbm,
    k_w_hbm,
    v_w_hbm,
    gate_w_hbm,
    out_w_hbm,
    N,
    C_z,
    H,
    d,
    eps,
    is_ending,
    pid,
):
    """Execute one triangle attention phase (SPMD variant).

    Work is split by pid:
      - Pass 3a (LN+QKV+bias): pid processes its half of flat tiles
      - Pass 3b (attention): pid processes its half of i_row iterations
      - Pass 3c (output): pid processes its half of flat tiles
    """
    Hd = H * d
    n_flat = N * N
    n_tiles_flat = n_flat // P_MAX
    n_tiles_spatial = N // P_MAX
    scale = 1.0 / (d**0.5)

    q_stride_row = N * Hd
    q_stride_col = Hd

    bias_stride_q = N * H
    bias_stride_k = H

    # SPMD work split for flat tile loops
    tiles_per_engine = n_tiles_flat // 2
    my_tile_start = pid * tiles_per_engine

    # SPMD work split for i_row loop
    rows_per_engine = N // 2
    my_row_start = pid * rows_per_engine

    # ---- Pass 3a: LayerNorm + QKV projection + bias computation (split tiles by pid) ----
    ln_w = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_w, src=ln_w_hbm)
    ln_b = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_b, src=ln_b_hbm)

    # WEIGHT-STATIONARY: Pre-load and transpose Q, K, V projection weights ONCE.
    q_w_t = _prepare_weight(q_w_hbm)
    k_w_t = _prepare_weight(k_w_hbm)
    v_w_t = _prepare_weight(v_w_hbm)

    # Pre-load and transpose bias projection weight.
    bias_w = nl.ndarray((H, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_w[0:H, 0:C_z], src=bias_proj_w_hbm)
    bias_w_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=bias_w_padded, value=0.0)
    nisa.tensor_copy(dst=bias_w_padded[0:H, 0:C_z], src=bias_w[0:H, 0:C_z])
    bias_w_t = _transpose_to_sbuf(bias_w_padded)

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        z_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_tile, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])

        z_normed = _layer_norm_tile(z_tile, ln_w, ln_b, C_z, eps)
        z_n_t = _transpose_to_sbuf(z_normed)

        q_tile = _matmul_with_w_t(z_n_t, q_w_t)
        k_tile = _matmul_with_w_t(z_n_t, k_w_t)
        v_tile = _matmul_with_w_t(z_n_t, v_w_t)

        nisa.dma_copy(
            dst=buf[off_q + tile_start : off_q + tile_start + P_MAX, 0:C_z], src=q_tile
        )
        nisa.dma_copy(
            dst=buf[off_k + tile_start : off_k + tile_start + P_MAX, 0:C_z], src=k_tile
        )
        nisa.dma_copy(
            dst=buf[off_v + tile_start : off_v + tile_start + P_MAX, 0:C_z], src=v_tile
        )

        # Bias projection
        bias_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bias_psum, stationary=z_n_t, moving=bias_w_t)
        bias_tile = nl.ndarray((P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        bias_full = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bias_full, src=bias_psum)
        nisa.tensor_copy(dst=bias_tile, src=bias_full[0:P_MAX, 0:H])

        nisa.dma_copy(dst=bias_buf[tile_start : tile_start + P_MAX, 0:H], src=bias_tile)

    # BARRIER: Both cores must finish pass 3a before pass 3b reads Q, K, V, bias
    nisa.core_barrier(data=buf, cores=(0, 1))
    nisa.core_barrier(data=bias_buf, cores=(0, 1))

    # ---- Pass 3b: Attention (split i_row by pid) ----
    for row_local in nl.sequential_range(rows_per_engine):
        i_row = my_row_start + row_local

        for h in nl.affine_range(H):
            hd_start = h * d

            for j_tile in nl.affine_range(n_tiles_spatial):
                j_start = j_tile * P_MAX

                q_tile = nl.ndarray((P_MAX, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                if is_ending:
                    nisa.dma_copy(
                        dst=q_tile,
                        src=buf.ap(
                            pattern=[[q_stride_row, P_MAX], [1, d]],
                            offset=off_q * C_z
                            + j_start * q_stride_row
                            + i_row * q_stride_col
                            + hd_start,
                        ),
                    )
                else:
                    nisa.dma_copy(
                        dst=q_tile,
                        src=buf.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=off_q * C_z
                            + i_row * q_stride_row
                            + j_start * q_stride_col
                            + hd_start,
                        ),
                    )

                q_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.memset(dst=q_padded, value=0.0)
                nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:d], src=q_tile)
                q_t = _transpose_to_sbuf(q_padded)

                m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=m_prev, value=-1e30)
                l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=l_prev, value=0.0)
                o_acc = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=o_acc, value=0.0)

                for k_tile_idx in nl.sequential_range(n_tiles_spatial):
                    k_start = k_tile_idx * P_MAX

                    k_tile_sb = nl.ndarray(
                        (P_MAX, d), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    if is_ending:
                        nisa.dma_copy(
                            dst=k_tile_sb,
                            src=buf.ap(
                                pattern=[[q_stride_row, P_MAX], [1, d]],
                                offset=off_k * C_z
                                + k_start * q_stride_row
                                + i_row * q_stride_col
                                + hd_start,
                            ),
                        )
                    else:
                        nisa.dma_copy(
                            dst=k_tile_sb,
                            src=buf.ap(
                                pattern=[[q_stride_col, P_MAX], [1, d]],
                                offset=off_k * C_z
                                + i_row * q_stride_row
                                + k_start * q_stride_col
                                + hd_start,
                            ),
                        )

                    v_tile_sb = nl.ndarray(
                        (P_MAX, d), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    if is_ending:
                        nisa.dma_copy(
                            dst=v_tile_sb,
                            src=buf.ap(
                                pattern=[[q_stride_row, P_MAX], [1, d]],
                                offset=off_v * C_z
                                + k_start * q_stride_row
                                + i_row * q_stride_col
                                + hd_start,
                            ),
                        )
                    else:
                        nisa.dma_copy(
                            dst=v_tile_sb,
                            src=buf.ap(
                                pattern=[[q_stride_col, P_MAX], [1, d]],
                                offset=off_v * C_z
                                + i_row * q_stride_row
                                + k_start * q_stride_col
                                + hd_start,
                            ),
                        )

                    bias_tile = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    if is_ending:
                        nisa.dma_copy(
                            dst=bias_tile,
                            src=bias_buf.ap(
                                pattern=[
                                    [bias_stride_k, P_MAX],
                                    [bias_stride_q, P_MAX],
                                ],
                                offset=k_start * bias_stride_q
                                + j_start * bias_stride_k
                                + h,
                            ),
                        )
                    else:
                        nisa.dma_copy(
                            dst=bias_tile,
                            src=bias_buf.ap(
                                pattern=[
                                    [bias_stride_q, P_MAX],
                                    [bias_stride_k, P_MAX],
                                ],
                                offset=j_start * bias_stride_q
                                + k_start * bias_stride_k
                                + h,
                            ),
                        )

                    k_padded = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.memset(dst=k_padded, value=0.0)
                    nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:d], src=k_tile_sb)
                    k_t = _transpose_to_sbuf(k_padded)

                    logits_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)
                    logits = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=logits, src=logits_psum)

                    logits_scaled = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=logits_scaled,
                        data=logits,
                        op0=nl.multiply,
                        operand0=scale,
                        engine=nisa.vector_engine,
                    )

                    bias_fp32 = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
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
                    correction = nl.ndarray(
                        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
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

                    l_corrected = nl.ndarray(
                        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_tensor(
                        dst=l_corrected, data1=l_prev, data2=correction, op=nl.multiply
                    )
                    tile_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_logits, axis=1)
                    l_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=l_new, data1=l_corrected, data2=tile_sum, op=nl.add
                    )

                    o_scaled = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=o_scaled,
                        data=o_acc,
                        op0=nl.multiply,
                        operand0=correction,
                        engine=nisa.vector_engine,
                    )

                    exp_bf16 = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=exp_bf16, src=exp_logits)
                    exp_t = _transpose_to_sbuf(exp_bf16)

                    pv_psum = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)
                    pv_sbuf = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                    nisa.tensor_tensor(
                        dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add
                    )
                    nisa.tensor_copy(dst=m_prev, src=m_new)
                    nisa.tensor_copy(dst=l_prev, src=l_new)

                inv_l = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_l, data=l_prev)
                o_final = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=o_final,
                    data=o_acc,
                    op0=nl.multiply,
                    operand0=inv_l,
                    engine=nisa.vector_engine,
                )
                o_out = nl.ndarray((P_MAX, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_out, src=o_final)

                if is_ending:
                    nisa.dma_copy(
                        dst=buf.ap(
                            pattern=[[q_stride_row, P_MAX], [1, d]],
                            offset=off_q * C_z
                            + j_start * q_stride_row
                            + i_row * q_stride_col
                            + hd_start,
                        ),
                        src=o_out,
                    )
                else:
                    nisa.dma_copy(
                        dst=buf.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=off_q * C_z
                            + i_row * q_stride_row
                            + j_start * q_stride_col
                            + hd_start,
                        ),
                        src=o_out,
                    )

    # BARRIER: Both cores must finish pass 3b before pass 3c reads attention output
    nisa.core_barrier(data=buf, cores=(0, 1))

    # ---- Pass 3c: Output gating + projection + residual (split tiles by pid) ----
    # WEIGHT-STATIONARY
    gate_w_t = _prepare_weight(gate_w_hbm)
    out_w_t = _prepare_weight(out_w_hbm)

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        attn_out = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=attn_out,
            src=buf[off_q + tile_start : off_q + tile_start + P_MAX, 0:C_z],
        )

        z_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_tile, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])
        z_normed = _layer_norm_tile(z_tile, ln_w, ln_b, C_z, eps)
        zn_t = _transpose_to_sbuf(z_normed)

        gate_out = _matmul_with_w_t(zn_t, gate_w_t)
        gate_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(
            dst=gate_sig, op=nl.sigmoid, data=gate_out, bias=None, scale=1.0
        )

        gated = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated, data1=attn_out, data2=gate_sig, op=nl.multiply)

        gated_t = _transpose_to_sbuf(gated)
        proj_out = _matmul_with_w_t(gated_t, out_w_t)

        z_updated = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z_updated, data1=z_tile, data2=proj_out, op=nl.add)
        nisa.dma_copy(
            dst=buf[off_out + tile_start : off_out + tile_start + P_MAX, 0:C_z],
            src=z_updated,
        )


# ============================================================================
# Phase 5: Transition_z (SwiGLU FFN)
# SPMD variant: split tiles by pid
# ============================================================================
def _transition_z_phase_spmd(
    z_in_hbm,
    z_out_hbm,
    norm_w_hbm,
    norm_b_hbm,
    fc1_w_hbm,
    fc2_w_hbm,
    fc3_w_hbm,
    N,
    C_z,
    hidden_dim,
    eps,
    pid,
):
    """Execute Transition_z: SwiGLU FFN on z (SPMD variant)."""
    n_flat = N * N
    n_tiles_flat = n_flat // P_MAX
    n_hidden_tiles = hidden_dim // P_MAX

    # SPMD work split for flat tile loops
    tiles_per_engine = n_tiles_flat // 2
    my_tile_start = pid * tiles_per_engine

    ln_w = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_w, src=norm_w_hbm)
    ln_b = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_b, src=norm_b_hbm)

    # WEIGHT-STATIONARY: Pre-load ALL FFN weight chunks ONCE before tile loop.
    fc1_w_t_0 = _prepare_weight(fc1_w_hbm[0:P_MAX, 0:C_z])
    fc1_w_t_1 = _prepare_weight(fc1_w_hbm[P_MAX : 2 * P_MAX, 0:C_z])
    fc1_w_t_2 = _prepare_weight(fc1_w_hbm[2 * P_MAX : 3 * P_MAX, 0:C_z])
    fc1_w_t_3 = _prepare_weight(fc1_w_hbm[3 * P_MAX : 4 * P_MAX, 0:C_z])

    fc2_w_t_0 = _prepare_weight(fc2_w_hbm[0:P_MAX, 0:C_z])
    fc2_w_t_1 = _prepare_weight(fc2_w_hbm[P_MAX : 2 * P_MAX, 0:C_z])
    fc2_w_t_2 = _prepare_weight(fc2_w_hbm[2 * P_MAX : 3 * P_MAX, 0:C_z])
    fc2_w_t_3 = _prepare_weight(fc2_w_hbm[3 * P_MAX : 4 * P_MAX, 0:C_z])

    fc3_w_t_0 = _prepare_weight(fc3_w_hbm[0:C_z, 0:P_MAX])
    fc3_w_t_1 = _prepare_weight(fc3_w_hbm[0:C_z, P_MAX : 2 * P_MAX])
    fc3_w_t_2 = _prepare_weight(fc3_w_hbm[0:C_z, 2 * P_MAX : 3 * P_MAX])
    fc3_w_t_3 = _prepare_weight(fc3_w_hbm[0:C_z, 3 * P_MAX : 4 * P_MAX])

    for tile_local in nl.sequential_range(tiles_per_engine):
        tile_idx = my_tile_start + tile_local
        tile_start = tile_idx * P_MAX

        z_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_tile, src=z_in_hbm[tile_start : tile_start + P_MAX, 0:C_z])

        z_normed = _layer_norm_tile(z_tile, ln_w, ln_b, C_z, eps)
        z_n_t = _transpose_to_sbuf(z_normed)

        output_acc = nl.ndarray((P_MAX, C_z), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=output_acc, value=0.0)

        for h_tile in nl.static_range(n_hidden_tiles):
            if h_tile == 0:
                fc1_w_t = fc1_w_t_0
                fc2_w_t = fc2_w_t_0
                fc3_w_t = fc3_w_t_0
            elif h_tile == 1:
                fc1_w_t = fc1_w_t_1
                fc2_w_t = fc2_w_t_1
                fc3_w_t = fc3_w_t_1
            elif h_tile == 2:
                fc1_w_t = fc1_w_t_2
                fc2_w_t = fc2_w_t_2
                fc3_w_t = fc3_w_t_2
            else:
                fc1_w_t = fc1_w_t_3
                fc2_w_t = fc2_w_t_3
                fc3_w_t = fc3_w_t_3

            fc1_chunk = _matmul_with_w_t(z_n_t, fc1_w_t)
            fc2_chunk = _matmul_with_w_t(z_n_t, fc2_w_t)

            fc1_sig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(
                dst=fc1_sig, op=nl.sigmoid, data=fc1_chunk, bias=None, scale=1.0
            )
            fc1_silu = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=fc1_silu, data1=fc1_chunk, data2=fc1_sig, op=nl.multiply
            )
            swiglu = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=swiglu, data1=fc1_silu, data2=fc2_chunk, op=nl.multiply
            )

            sg_t = _transpose_to_sbuf(swiglu)

            partial_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=partial_psum, stationary=sg_t, moving=fc3_w_t)
            partial = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=partial, src=partial_psum)
            nisa.tensor_tensor(
                dst=output_acc, data1=output_acc, data2=partial, op=nl.add
            )

        output_tile = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_tile, src=output_acc)

        z_updated = nl.ndarray((P_MAX, C_z), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z_updated, data1=z_tile, data2=output_tile, op=nl.add)
        nisa.dma_copy(
            dst=z_out_hbm[tile_start : tile_start + P_MAX, 0:C_z], src=z_updated
        )


# ============================================================================
# Main Mega-Kernel Entry Point — SPMD grid=[2] variant
# ============================================================================
@nki.jit
def fused_z_ops_spmd(
    # Main tensors (all in HBM)
    z,
    pair_mask,
    # TriMulOut weights
    tmul_out_norm_in_w,
    tmul_out_norm_in_b,
    tmul_out_p_in_w,
    tmul_out_g_in_w,
    tmul_out_norm_out_w,
    tmul_out_norm_out_b,
    tmul_out_p_out_w,
    tmul_out_g_out_w,
    # TriMulIn weights
    tmul_in_norm_in_w,
    tmul_in_norm_in_b,
    tmul_in_p_in_w,
    tmul_in_g_in_w,
    tmul_in_norm_out_w,
    tmul_in_norm_out_b,
    tmul_in_p_out_w,
    tmul_in_g_out_w,
    # TriAttnStart weights
    tatt_s_ln_w,
    tatt_s_ln_b,
    tatt_s_bias_proj_w,
    tatt_s_q_w,
    tatt_s_k_w,
    tatt_s_v_w,
    tatt_s_g_w,
    tatt_s_o_w,
    # TriAttnEnd weights
    tatt_e_ln_w,
    tatt_e_ln_b,
    tatt_e_bias_proj_w,
    tatt_e_q_w,
    tatt_e_k_w,
    tatt_e_v_w,
    tatt_e_g_w,
    tatt_e_o_w,
    # Transition_z weights
    trans_z_norm_w,
    trans_z_norm_b,
    trans_z_fc1_w,
    trans_z_fc2_w,
    trans_z_fc3_w,
    # Pre-allocated scratch buffers (passed from PyTorch as external shared HBM)
    scratch_buf,
    bias_buf,
    # Constants
    N: int = 256,
    C_z: int = 128,
    H: int = 4,
    d: int = 32,
    eps: float = 1e-5,
):
    """Fused z-operations mega-kernel — SPMD grid=[2] variant.

    Launches on 2 physical NeuronCores within one logical core (LNC=2).
    Each core executes the same program but processes different subsets
    of work. core_barrier() synchronizes between phases.

    IMPORTANT: scratch_buf and bias_buf must be pre-allocated by the caller
    as PyTorch tensors on the XLA device. They live in shared HBM and are
    visible to both SPMD instances. Internal nl.ndarray(shared_hbm) does NOT
    work for intermediate buffers in SPMD kernels (compiler limitation).

    scratch_buf: shape [6 * N*N, C_z], dtype bf16
    bias_buf: shape [N*N, H], dtype bf16

    Launch with: fused_z_ops_spmd[2](z, pair_mask, ..., scratch_buf, bias_buf)
    """
    pid = nl.program_id(0)

    hidden_dim = 4 * C_z
    n_flat = N * N

    z_out = nl.ndarray((n_flat, C_z), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    off_a = 0 * n_flat
    off_b = 1 * n_flat
    off_gate = 2 * n_flat
    off_d = 3 * n_flat
    off_z1 = 4 * n_flat
    off_z2 = 5 * n_flat

    # Phase 1: TriMulOut
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

    # BARRIER: Phase 2 reads Phase 1's z1 output
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Phase 2: TriMulIn
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

    # BARRIER: Phase 3 reads Phase 2's z2 output
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Phase 3: TriAttnStart
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
        H=H,
        d=d,
        eps=eps,
        is_ending=False,
        pid=pid,
    )

    # BARRIER: Phase 4 reads Phase 3's z1 output
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Phase 4: TriAttnEnd
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
        H=H,
        d=d,
        eps=eps,
        is_ending=True,
        pid=pid,
    )

    # BARRIER: Phase 5 reads Phase 4's z2 output
    nisa.core_barrier(data=scratch_buf, cores=(0, 1))

    # Phase 5: Transition_z
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
        hidden_dim=hidden_dim,
        eps=eps,
        pid=pid,
    )

    return z_out
