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

"""Triangular attention kernel for AlphaFold-family pairformer architectures.

Implements the core attention computation from the triangular attention operation
used in Boltz-2, AlphaFold2, and AlphaFold3 pairformer layers. This operation
applies multi-head attention independently along each row (or column) of the
pair representation matrix, with an additive bias derived from the pair
representation projected to attention heads.

The "starting node" (row-wise) variant is implemented directly. The "ending node"
(column-wise) variant is obtained by transposing the pair representation (swapping
spatial dims 0 and 1) before and after calling this kernel.

Algorithm:
    For each row i of the pair matrix, independently:
        1. Extract Q[i,:,:], K[i,:,:], V[i,:,:] — each [N, Hd]
        2. For each head h:
            logits[j,k] = Q[i,j,h] @ K[i,k,h]^T * scale + bias[j,k,h]
            weights[j,:] = softmax(logits[j,:])
            output[i,j,h] = weights[j,:] @ V[i,:,h]

    The bias is a full N x N matrix per head, shared across all rows i.
    It is derived from the pair representation projected to H heads:
        bias = permute(Linear(LayerNorm(z)), (2, 0, 1))  ->  [H, N, N]
    and stored as (N, N, H) for natural DMA access.

    Online softmax (Milakov & Gimelshein, 2018) is used to tile over the key
    dimension without materializing the full N x N attention matrix.

IO Shapes:
    q:      (N, N, H * d)   — query, after QKV projection
    k:      (N, N, H * d)   — key
    v:      (N, N, H * d)   — value
    bias:   (N, N, H)       — triangle bias, bias[q, k, h] for query pos q, key pos k
    output: (N, N, H * d)   — attention output before gating/output projection

Tiling:
    - Partition dimension (axis 0 of SBUF tiles): P_MAX = 128
    - N must be a multiple of P_MAX
    - Head dimension d must be <= P_MAX (typically d=32 for Boltz-2, H=4)
    - For Q@K^T: d is padded to P_MAX for nc_matmul contraction axis
    - For attn@V: contraction over P_MAX key positions (natural fit)
    - Bias is loaded as (P_MAX, P_MAX) tiles per (j_tile, k_tile) pair

Reference:
    Wohlwend et al., "Boltz-2: Predicting the Structure and Interactions of
    Biomolecular Complexes", 2025. https://github.com/jwohlwend/boltz
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension maximum. nl.tile_size.pmax returns a non-integer at
# import time; using a Python int is required for static shape declarations.
P_MAX = 128


@nki.jit
def triangular_attention_fwd(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    bias: nl.ndarray,
    scale: float = 0.1767766952966369,
) -> nl.ndarray:
    """Compute triangular attention (starting node / row-wise).

    Performs multi-head attention independently for each row of the pair
    representation matrix, with a full 2D additive triangle bias per head.
    Uses online softmax to handle large sequence lengths without materializing
    the full N x N attention matrix.

    Dimensions:
        N:  Sequence length (number of tokens/residues). Must be a multiple of 128.
        Hd: Total head dimension (H * d). Typically 128 for Boltz-2 (H=4, d=32).
        H:  Number of attention heads.
        d:  Per-head dimension (Hd // H).

    Args:
        q (nl.ndarray): Query tensor, shape (N, N, Hd), dtype bfloat16.
        k (nl.ndarray): Key tensor, shape (N, N, Hd), dtype bfloat16.
        v (nl.ndarray): Value tensor, shape (N, N, Hd), dtype bfloat16.
        bias (nl.ndarray): Triangle bias, shape (N, N, H), dtype bfloat16.
            bias[q_pos, k_pos, h] is the additive bias for query position
            q_pos, key position k_pos, head h. This bias is shared across
            all rows i — the same bias matrix applies to every row's
            independent attention computation.
        scale (float): Attention scaling factor, typically 1/sqrt(d).
            Default is 1/sqrt(32) for Boltz-2.

    Returns:
        nl.ndarray: Attention output, shape (N, N, Hd), dtype bfloat16.

    Notes:
        - Linear projections (QKV, gating, output) are handled outside this
          kernel for modularity. The caller applies them before/after.
        - For the "ending node" (column-wise) variant, transpose the pair
          representation's spatial dimensions before and after calling this kernel.
        - Online softmax operates in float32 for numerical stability; the final
          output is cast back to the input dtype (bfloat16).
        - Q@K^T uses nc_matmul with d padded to P_MAX on the contraction axis.
          For d=32 this wastes 75% of TensorEngine lanes but is correct.

    Pseudocode:
        output = zeros(N, N, Hd)
        for i_row in range(N):                     # independent per row
            for h in range(H):                      # independent per head
                for j_tile in range(N // P_MAX):    # output query tile
                    q_tile = load q[i_row, j_start:j_end, h*d:(h+1)*d]
                    # Online softmax over key tiles
                    m, l, o_acc = -inf, 0, 0
                    for k_tile in range(N // P_MAX):
                        k_t = load k[i_row, k_start:k_end, h*d:(h+1)*d]
                        v_t = load v[i_row, k_start:k_end, h*d:(h+1)*d]
                        bias_tile = load bias[j_start:j_end, k_start:k_end, h]
                        logits = q_tile @ k_t^T * scale + bias_tile
                        m_new = max(m, max(logits))
                        correction = exp(m - m_new)
                        o_acc = o_acc * correction + exp(logits - m_new) @ v_t
                        l = l * correction + sum(exp(logits - m_new))
                        m = m_new
                    output[i_row, j_start:j_end, h*d:(h+1)*d] = o_acc / l
        return output
    """
    N, N2, Hd = q.shape
    N_bias, N_bias2, H = bias.shape
    d = Hd // H

    assert N == N2, f"Q must be square: got ({N}, {N2}, {Hd})"
    assert N == N_bias and N == N_bias2, (
        f"Bias shape ({N_bias}, {N_bias2}, {H}) must match Q N ({N})"
    )
    assert N % P_MAX == 0, f"N ({N}) must be a multiple of P_MAX ({P_MAX})"
    assert d <= P_MAX, f"Head dim d ({d}) must be <= P_MAX ({P_MAX})"

    # Output tensor in HBM — same shape as input
    output = nl.ndarray((N, N, Hd), dtype=q.dtype, buffer=nl.shared_hbm)

    # Access pattern constants for loading from 3D HBM tensors.
    # For a tensor of shape (N, N, Hd), the row-major strides are:
    #   dim 0 stride = N * Hd,  dim 1 stride = Hd,  dim 2 stride = 1
    # We load 2D tiles (P_MAX, d) from a specific (i_row, j_start, hd_start).
    q_stride_row = N * Hd  # stride along dim 0 (row i)
    q_stride_col = Hd  # stride along dim 1 (position j/k within row)

    # For bias tensor of shape (N, N, H), strides are:
    #   dim 0 stride = N * H,  dim 1 stride = H,  dim 2 stride = 1
    # We load 2D tiles (P_MAX, P_MAX) from bias[j_start, k_start, h] for each
    # (j_tile, k_tile, head) combination. The tile has P_MAX query positions
    # (strided by N*H) and P_MAX key positions (strided by H).
    bias_stride_q = N * H  # stride along dim 0 (query position)
    bias_stride_k = H  # stride along dim 1 (key position)

    # Process each row i independently (embarrassingly parallel across rows)
    for i_row in nl.affine_range(N):
        # Base HBM offset for row i: i_row * N * Hd
        row_base = i_row * q_stride_row

        # Process each head independently (parallel across heads)
        for h in nl.affine_range(H):
            hd_start = h * d

            # Tile over query positions j (output positions), P_MAX at a time
            for j_tile in nl.affine_range(N // P_MAX):
                j_start = j_tile * P_MAX

                # ---- Load Q tile: (P_MAX, d) ----
                # From q[i_row, j_start : j_start+P_MAX, hd_start : hd_start+d]
                q_tile = nl.ndarray((P_MAX, d), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=q_tile,
                    src=q.ap(
                        pattern=[[q_stride_col, P_MAX], [1, d]],
                        offset=row_base + j_start * q_stride_col + hd_start,
                    ),
                )

                # ---- Prepare Q^T padded for nc_matmul ----
                # nc_matmul contraction dim must be P_MAX (partition axis).
                # With d < P_MAX, we pad Q to (P_MAX, P_MAX), transpose, then
                # use the transposed version as stationary operand.
                q_padded = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.memset(dst=q_padded, value=0.0)
                nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:d], src=q_tile)

                q_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=q_t_psum, data=q_padded)
                q_t = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_t, src=q_t_psum)

                # ---- Online softmax accumulators ----
                m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=m_prev, value=-1e30)

                l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=l_prev, value=0.0)

                o_acc = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=o_acc, value=0.0)

                # ---- Tile over key positions (sequential for online softmax) ----
                for k_tile_idx in nl.sequential_range(N // P_MAX):
                    k_start = k_tile_idx * P_MAX

                    # Load K tile: (P_MAX, d)
                    # From k[i_row, k_start : k_start+P_MAX, hd_start : hd_start+d]
                    k_tile_sb = nl.ndarray((P_MAX, d), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=k_tile_sb,
                        src=k.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=row_base + k_start * q_stride_col + hd_start,
                        ),
                    )

                    # Load V tile: (P_MAX, d)
                    # From v[i_row, k_start : k_start+P_MAX, hd_start : hd_start+d]
                    v_tile_sb = nl.ndarray((P_MAX, d), dtype=v.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=v_tile_sb,
                        src=v.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=row_base + k_start * q_stride_col + hd_start,
                        ),
                    )

                    # ---- Load bias tile: (P_MAX, P_MAX) ----
                    # From bias[j_start : j_start+P_MAX, k_start : k_start+P_MAX, h]
                    # This is a full 2D bias tile for query positions j and key
                    # positions k. Each element bias[j, k, h] is the additive bias
                    # for query pos j, key pos k, head h.
                    bias_tile = nl.ndarray(
                        (P_MAX, P_MAX), dtype=bias.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=bias_tile,
                        src=bias.ap(
                            pattern=[[bias_stride_q, P_MAX], [bias_stride_k, P_MAX]],
                            offset=j_start * bias_stride_q
                            + k_start * bias_stride_k
                            + h,
                        ),
                    )

                    # ---- Compute Q @ K^T via nc_matmul ----
                    # Pad K to (P_MAX, P_MAX), transpose, then matmul.
                    # Result: logits (P_MAX, P_MAX) = Q^T^T @ K^T = Q @ K^T
                    k_padded = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.memset(dst=k_padded, value=0.0)
                    nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:d], src=k_tile_sb)

                    k_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=k_t_psum, data=k_padded)
                    k_t = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=k_t, src=k_t_psum)

                    logits_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)

                    logits = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=logits, src=logits_psum)

                    # Scale logits by 1/sqrt(d)
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

                    # Add full 2D triangle bias: (P_MAX, P_MAX) + (P_MAX, P_MAX)
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

                    # ---- Online softmax (Milakov & Gimelshein, 2018) ----
                    # Step 1: tile max
                    tile_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(
                        dst=tile_max, op=nl.maximum, data=logits_biased, axis=1
                    )

                    # Step 2: running max
                    m_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=m_new, data1=m_prev, data2=tile_max, op=nl.maximum
                    )

                    # Step 3: correction factor exp(m_prev - m_new)
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

                    # Step 4: exp(logits - m_new)
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

                    # Step 5: update running sum l = l * correction + sum(exp_logits)
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

                    # Step 6: rescale previous output accumulator
                    o_scaled = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=o_scaled,
                        data=o_acc,
                        op0=nl.multiply,
                        operand0=correction,
                        engine=nisa.vector_engine,
                    )

                    # ---- exp_logits @ V via nc_matmul ----
                    # Contraction is over P_MAX key positions (natural fit).
                    # Need to transpose exp_logits for nc_matmul convention.
                    exp_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=exp_bf16, src=exp_logits)

                    exp_t_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum
                    )
                    nisa.nc_transpose(dst=exp_t_psum, data=exp_bf16)
                    exp_t = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=exp_t, src=exp_t_psum)

                    pv_psum = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)

                    pv_sbuf = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                    # Step 7: accumulate o_acc = o_scaled + pv
                    nisa.tensor_tensor(
                        dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add
                    )

                    # Update running state
                    nisa.tensor_copy(dst=m_prev, src=m_new)
                    nisa.tensor_copy(dst=l_prev, src=l_new)

                # ---- Finalize: output = o_acc / l ----
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

                # Cast back to input dtype and store to HBM
                o_out = nl.ndarray((P_MAX, d), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_out, src=o_final)

                nisa.dma_copy(
                    dst=output.ap(
                        pattern=[[q_stride_col, P_MAX], [1, d]],
                        offset=row_base + j_start * q_stride_col + hd_start,
                    ),
                    src=o_out,
                )

    return output


# ---------------------------------------------------------------------------
# CPU reference implementation for testing
# ---------------------------------------------------------------------------


def triangular_attention_ref(q, k, v, bias, scale):
    """NumPy reference implementation for triangular attention.

    Args:
        q: (N, N, H*d) float32 — query tensor.
        k: (N, N, H*d) float32 — key tensor.
        v: (N, N, H*d) float32 — value tensor.
        bias: (N, N, H) float32 — triangle bias. bias[q_pos, k_pos, h] is the
            additive bias for query position q_pos, key position k_pos, head h.
            Shared across all rows.
        scale: float — 1/sqrt(d).

    Returns:
        output: (N, N, H*d) float32 — attention output.
    """
    N, _, Hd = q.shape
    H = bias.shape[2]
    d = Hd // H

    output = np.zeros_like(q)

    for h in range(H):
        hd_s = h * d
        hd_e = (h + 1) * d

        q_h = q[:, :, hd_s:hd_e]  # (N, N, d)
        k_h = k[:, :, hd_s:hd_e]  # (N, N, d)
        v_h = v[:, :, hd_s:hd_e]  # (N, N, d)
        bias_h = bias[:, :, h]  # (N, N) — full 2D bias for head h

        for i in range(N):
            q_row = q_h[i]  # (N, d)
            k_row = k_h[i]  # (N, d)
            v_row = v_h[i]  # (N, d)

            # logits: (N, N) = q_row @ k_row^T
            logits = q_row @ k_row.T * scale  # (N, N)

            # Add full 2D bias: bias_h[j, k] for each (query j, key k) pair
            logits += bias_h  # (N, N) + (N, N)

            # Softmax over k (axis=-1)
            logits_max = logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            attn_weights = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

            # Weighted sum: (N, N) @ (N, d) -> (N, d)
            output[i, :, hd_s:hd_e] = attn_weights @ v_row

    return output
