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

"""Triangular multiplicative update kernel for AlphaFold-family pairformers.

Implements the core einsum contraction from the triangular multiplicative update
operation used in Boltz-2, AlphaFold2, and AlphaFold3 pairformer layers. This
operation computes a contraction over one spatial dimension of two gated
projections of the pair representation, producing a rank-2 update per channel.

Two variants exist:
    - Outgoing: result[i,j,d] = sum_k a[i,k,d] * b[j,k,d]   (einsum "bikd,bjkd->bijd")
    - Incoming: result[i,j,d] = sum_k a[k,i,d] * b[k,j,d]   (einsum "bkid,bkjd->bijd")

This kernel implements the outgoing variant directly. The incoming variant is
obtained by transposing the spatial dimensions of a and b before calling.

Algorithm:
    For each channel d independently:
        result[:,:,d] = a[:,:,d] @ b[:,:,d].T       (N x N matrix multiply)

    This decomposes the 4D einsum into D independent NxN matmuls, one per
    channel. Each matmul is tiled into P_MAX x P_MAX blocks and accumulated
    in float32.

IO Shapes:
    a:      (N, N, D)   — first gated projection, after sigmoid gating and masking
    b:      (N, N, D)   — second gated projection
    output: (N, N, D)   — contraction result, in float32 semantics

    Where D = 128 for Boltz-2 (dim of pair representation).

Tiling:
    - Partition dimension (axis 0 of SBUF tiles): P_MAX = 128
    - N must be a multiple of P_MAX
    - D must be <= P_MAX (D=128 for Boltz-2)
    - For each channel d: tile the NxN matmul a_d @ b_d^T into P_MAX blocks
    - Accumulate in float32, output in bfloat16

Precision:
    - Boltz-2 explicitly casts a and b to float32 before the einsum
    - The kernel accepts bfloat16 inputs (post-gating) and accumulates in float32
    - nc_matmul on NeuronCore accumulates in float32 natively for bf16 inputs
    - Output is cast to bfloat16 for the subsequent LayerNorm + output gating

Reference:
    Wohlwend et al., "Boltz-2: Predicting the Structure and Interactions of
    Biomolecular Complexes", 2025. https://github.com/jwohlwend/boltz
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


@nki.jit
def triangular_mul_fwd(
    a: nl.ndarray,
    b: nl.ndarray,
) -> nl.ndarray:
    """Compute triangular multiplicative update (outgoing variant).

    Performs the core einsum contraction: result[i,j,d] = sum_k a[i,k,d] * b[j,k,d].
    This is equivalent to D independent NxN matrix multiplies: for each channel d,
    result[:,:,d] = a[:,:,d] @ b[:,:,d].T.

    For the incoming variant (bkid,bkjd->bijd), transpose the spatial dimensions
    of a and b before calling: a_incoming[i,k,d] = a_original[k,i,d].

    Dimensions:
        N: Sequence length (number of tokens/residues). Must be a multiple of 128.
        D: Channel dimension. Must be <= 128. Typically 128 for Boltz-2.

    Args:
        a (nl.ndarray): First projection tensor, shape (N, N, D), dtype bfloat16.
            After gating: a = sigmoid(gate_a(z_norm)) * proj_a(z_norm), masked.
        b (nl.ndarray): Second projection tensor, shape (N, N, D), dtype bfloat16.
            After gating: b = sigmoid(gate_b(z_norm)) * proj_b(z_norm), masked.

    Returns:
        nl.ndarray: Contraction result, shape (N, N, D), dtype bfloat16.
            The caller applies LayerNorm + output gating + output projection.

    Notes:
        - Linear projections, gating, masking, LayerNorm, and output projection
          are handled outside this kernel for modularity.
        - Float32 accumulation is handled natively by nc_matmul for bf16 inputs.
        - Output is cast to bfloat16. Boltz-2 applies LayerNorm (which uses
          float32 internally) immediately after, so bf16 output is acceptable.
        - For each channel d, the kernel performs a standard tiled matrix multiply
          a_d @ b_d^T using nc_matmul with P_MAX-sized tiles.
    """
    N, N2, D = a.shape
    N_b, N2_b, D_b = b.shape

    assert N == N2, f"a must be square: got ({N}, {N2}, {D})"
    assert N == N_b and N2 == N2_b and D == D_b, (
        f"a shape ({N}, {N2}, {D}) must match b shape ({N_b}, {N2_b}, {D_b})"
    )
    assert N % P_MAX == 0, f"N ({N}) must be a multiple of P_MAX ({P_MAX})"
    assert D <= P_MAX, f"D ({D}) must be <= P_MAX ({P_MAX})"

    # Output tensor in HBM
    output = nl.ndarray((N, N, D), dtype=a.dtype, buffer=nl.shared_hbm)

    # Strides for 3D tensor (N, N, D) in row-major layout
    # a[i, k, d] is at offset i * N * D + k * D + d
    stride_i = N * D  # stride along dim 0
    stride_k = D  # stride along dim 1

    n_tiles = N // P_MAX

    # Process each channel d independently
    for d in nl.affine_range(D):
        # For each output tile (i_tile, j_tile)
        for i_tile in nl.affine_range(n_tiles):
            i_start = i_tile * P_MAX

            for j_tile in nl.affine_range(n_tiles):
                j_start = j_tile * P_MAX

                # Accumulator for result[i_start:i_end, j_start:j_end, d]
                # This is the partial sum over k of a[i,k,d] * b[j,k,d]
                acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc, value=0.0)

                # Tile over contraction axis k
                for k_tile_idx in nl.sequential_range(n_tiles):
                    k_start = k_tile_idx * P_MAX

                    # Load a_tile: a[i_start:i_end, k_start:k_end, d]
                    # Shape: (P_MAX, P_MAX) — P_MAX i-positions, P_MAX k-positions
                    # Stride: dim0 = N*D (between i rows), dim1 = D (between k cols)
                    a_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=a_tile,
                        src=a.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=i_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    # Load b_tile: b[j_start:j_end, k_start:k_end, d]
                    # Same layout as a
                    b_tile = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=b_tile,
                        src=b.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=j_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    # Compute a_tile @ b_tile^T using nc_matmul
                    # nc_matmul: dst = stationary^T @ moving
                    # We want a_tile @ b_tile^T = (a_tile^T)^T @ b_tile^T
                    # So: stationary = a_tile (nc_matmul transposes it internally)
                    #     moving = b_tile
                    # Wait -- nc_matmul computes: dst[f1, f2] = sum_p stationary[p, f1] * moving[p, f2]
                    # = stationary^T @ moving
                    #
                    # We want: result[i, j] = sum_k a[i,k] * b[j,k]
                    # = a @ b^T
                    # = (b @ a^T)^T
                    #
                    # Using nc_matmul with stationary=b_tile, moving=a_tile:
                    #   dst[f1, f2] = sum_p b[p, f1] * a[p, f2] = b^T @ a = (a^T @ b)^...
                    #
                    # Actually let me be precise:
                    # nc_matmul: dst[i,j] = sum_k stationary[k,i] * moving[k,j]
                    # This computes stationary^T @ moving.
                    #
                    # We want: result[i,j] = sum_k a[i,k] * b[j,k] = a @ b^T
                    #
                    # If we transpose a first: a^T[k,i] = a[i,k]
                    # Then nc_matmul with stationary=a^T, moving=b^T:
                    #   dst[i,j] = sum_k a^T[k,i]^T... no that's wrong
                    #
                    # nc_matmul(stationary=X, moving=Y) = X^T @ Y
                    #   dst[i,j] = sum_k X[k,i] * Y[k,j]
                    #
                    # We want sum_k a[i,k] * b[j,k].
                    # Let X[k,i] = a[i,k] => X = a^T. Let Y[k,j] = b[j,k] => Y = b^T.
                    # Then dst[i,j] = sum_k a^T[k,i] * b^T[k,j] ...
                    # = sum_k a[i,k] * b[j,k]. YES!
                    #
                    # So: stationary = a^T (transpose a_tile), moving = b^T (transpose b_tile)

                    # Transpose a_tile
                    a_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=a_t_psum, data=a_tile)
                    a_t = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=a_t, src=a_t_psum)

                    # Transpose b_tile
                    b_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=b_t_psum, data=b_tile)
                    b_t = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=b_t, src=b_t_psum)

                    # nc_matmul: stationary=a^T, moving=b^T => a @ b^T
                    partial_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=partial_psum, stationary=a_t, moving=b_t)
                    partial = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=partial, src=partial_psum)

                    # Accumulate
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=partial, op=nl.add)

                # Cast to output dtype and store
                out_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile, src=acc)

                # Store result[i_start:i_end, j_start:j_end, d]
                # Output has same (N, N, D) layout as input
                nisa.dma_copy(
                    dst=output.ap(
                        pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                        offset=i_start * stride_i + j_start * stride_k + d,
                    ),
                    src=out_tile,
                )

    return output


# ---------------------------------------------------------------------------
# CPU reference implementation for testing
# ---------------------------------------------------------------------------


def triangular_mul_ref(a, b):
    """NumPy reference: outgoing triangular multiplicative update.

    Computes result[i,j,d] = sum_k a[i,k,d] * b[j,k,d] for all (i,j,d).
    Equivalent to: for each d, result[:,:,d] = a[:,:,d] @ b[:,:,d].T

    Args:
        a: (N, N, D) float32
        b: (N, N, D) float32

    Returns:
        output: (N, N, D) float32
    """
    N, _, D = a.shape
    output = np.zeros_like(a)
    for d in range(D):
        output[:, :, d] = a[:, :, d] @ b[:, :, d].T
    return output
