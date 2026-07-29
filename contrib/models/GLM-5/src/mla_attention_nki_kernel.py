# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA Attention NKI kernel for Token Generation (decode).

Implements fused MLA attention for GLM-5 / DeepSeek-V3 architecture:
  1. Dual-score: (q_pe @ k_pe^T) + (q_nope @ c_kv^T) with K-tiled matmuls
  2. Online softmax across cache length (CACHE_TILE_SIZE=512 chunks)
  3. V accumulation: softmax_weights @ c_kv
  4. V absorption: v_accum @ v_absorb^T

Target: trn2 (NKI 0.3.0, nc_version >= 3)

Hardware constraints:
  - P_MAX = 128 (partition dimension)
  - PSUM_F_MAX = 512 (PSUM free dimension, gen2/3)
  - nc_matmul stationary: [K<=128, M<=128]
  - nc_matmul moving: [K<=128, N<=512]
  - nc_transpose: tile <= 128x128

Dimensions (GLM-5):
  - d_rope = 64 (RoPE dimension)
  - d_c = 512 (compressed KV / kv_lora_rank)
  - d_v = 256 (value head dimension after absorption)
  - d_cache = d_rope + d_c = 576 (KV cache layout: [k_pe | c_kv])
  - BH = batch * n_heads_per_core (number of queries)

Algorithm (per query):
  For each cache tile of CACHE_TILE_SIZE positions:
    1. Load k_pe_cache[S_tile, d_rope] and c_kv_cache[S_tile, d_c]
    2. Compute rope_scores = q_pe[1, d_rope] @ k_pe^T -> [1, S_tile]
       (d_rope=64 fits in single matmul: stationary=[64, 1], moving=[64, S_tile])
    3. Compute nope_scores via K-tiled matmuls over d_c=512:
       For each K_tile of 128: partial = q_nope_chunk[1, 128] @ c_kv_chunk^T[128, S_tile]
       Accumulate into same PSUM -> [1, S_tile]
    4. Combined scores = rope_scores + nope_scores (in PSUM via accumulation)
    5. Apply scale and mask
    6. Online softmax: update running_max, correction, running_sum
    7. V accumulation: weights[1, S_tile] @ c_kv[S_tile, d_c] -> accumulate [1, d_c]
       (d_c=512 exceeds PSUM_F_MAX=512, so single tile suffices for free dim)
  After all cache tiles:
    8. Normalize V accumulation by 1/running_sum
    9. V absorption: v_accum[1, d_c] @ v_absorb^T[d_c, d_v] -> [1, d_v]
       (d_c=512, tile K in 128 chunks, d_v=256 fits in moving free dim)
"""

import nki
import nki.isa as nisa
import nki.language as nl


# === Hardware constants ===
P_MAX = 128  # SBUF partition dimension max
PSUM_F_MAX = 512  # PSUM free dimension max (gen2/3)
CACHE_TILE_SIZE = 512  # Process 512 cache positions per tile
K_TILE = 128  # K-dimension tile size for matmuls


def div_ceil(n: int, d: int) -> int:
    """Ceiling division."""
    return (n + d - 1) // d


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


# Large negative value for masked positions (bf16-safe, NOT -inf which can cause NaN)
LARGE_NEG = -9984.0


@nki.jit
def mla_attention_tkg(
    q_pe,  # [BH, d_rope]
    q_nope,  # [BH, d_c]
    kv_cache,  # [B, S, d_cache]  where d_cache = d_rope + d_c
    v_absorb,  # [H, d_v, d_c]
    attn_mask,  # [BH, S] bool/uint8
    softmax_scale=0.0625,
    n_heads=2,
    d_rope=64,
    d_c=512,
    d_v=256,
    seq_len=512,
):
    """
    MLA attention kernel for token generation (s_active=1).

    Processes each query (BH) independently. For each query:
    - Computes dual scores against the KV cache
    - Applies online softmax
    - Accumulates V from compressed KV
    - Applies V absorption weight

    Args:
        q_pe: Query RoPE component [BH, d_rope]
        q_nope: Query absorbed nope component [BH, d_c]
        kv_cache: KV cache [B, S, d_cache] with layout [k_pe(d_rope) | c_kv(d_c)]
        v_absorb: V absorption weight [H, d_v, d_c]
        attn_mask: Boolean attention mask [BH, S]
        softmax_scale: Attention scale factor (typically 1/sqrt(d_c + d_rope) or similar)
        n_heads: Number of attention heads per core
        d_rope: RoPE dimension (64)
        d_c: Compressed KV dimension (512)
        d_v: Output value dimension (256)
        seq_len: Actual sequence length in cache

    Returns:
        output: [BH, d_v] attention output after V absorption
    """
    BH = q_pe.shape[0]
    B = BH // n_heads

    # Validate dimensions
    kernel_assert(d_rope <= P_MAX, f"d_rope={d_rope} must be <= P_MAX={P_MAX}")
    kernel_assert(d_c % K_TILE == 0, f"d_c={d_c} must be divisible by K_TILE={K_TILE}")
    kernel_assert(d_v <= PSUM_F_MAX, f"d_v={d_v} must be <= PSUM_F_MAX={PSUM_F_MAX}")

    # Number of cache tiles
    num_cache_tiles = div_ceil(seq_len, CACHE_TILE_SIZE)
    # Number of K-tiles for d_c dimension
    num_k_tiles_dc = d_c // K_TILE  # 512/128 = 4

    # Allocate output in shared HBM
    output = nl.ndarray((BH, d_v), dtype=q_pe.dtype, buffer=nl.shared_hbm)

    # Process each query independently
    for bh in nl.affine_range(BH):
        b_idx = bh // n_heads
        h_idx = bh % n_heads

        # =====================================================================
        # Load query vectors into SBUF
        # q_pe[bh, :] -> [1, d_rope] -> need as [d_rope, 1] for stationary
        # q_nope[bh, :] -> [1, d_c] -> kept as row, sliced per K-tile
        # =====================================================================

        # Load q_pe as column vector [d_rope, 1] via dma_transpose
        # HBM shape: q_pe[bh:bh+1, 0:d_rope] = [1, d_rope]
        # dma_transpose loads as [d_rope, 1] directly
        q_pe_col = nl.ndarray((d_rope, 1), dtype=q_pe.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=q_pe_col[0:d_rope, 0:1], src=q_pe[bh : bh + 1, 0:d_rope])

        # OPTIMIZATION: Pre-load all q_nope chunks as columns [K_TILE, 1] ONCE,
        # outside the cache tile loop. These are reused every iteration.
        # (avoids repeated dma_transpose per cache tile - saves 4 DMA ops per tile)
        q_nope_cols = [None] * num_k_tiles_dc
        for k_idx in nl.affine_range(num_k_tiles_dc):
            k_start = k_idx * K_TILE
            q_nope_cols[k_idx] = nl.ndarray(
                (K_TILE, 1), dtype=q_nope.dtype, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=q_nope_cols[k_idx][0:K_TILE, 0:1],
                src=q_nope[bh : bh + 1, k_start : k_start + K_TILE],
            )

        # =====================================================================
        # Online softmax state: running_max, running_sum, v_accum
        # =====================================================================

        # Running max for online softmax: scalar in [1, 1]
        running_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=running_max[0:1, 0:1], value=LARGE_NEG)

        # Running sum of exp(scores - max): scalar in [1, 1]
        running_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=running_sum[0:1, 0:1], value=0.0)

        # V accumulation buffer: [1, d_c] in float32
        # We accumulate weighted c_kv values, then absorb at the end
        v_accum = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=v_accum[0:1, 0:d_c], value=0.0)

        # =====================================================================
        # Main loop over cache tiles (online softmax)
        # =====================================================================

        for tile_idx in nl.sequential_range(num_cache_tiles):
            s_start = tile_idx * CACHE_TILE_SIZE
            s_size = min(CACHE_TILE_SIZE, seq_len - s_start)

            # -----------------------------------------------------------------
            # Load cache tile: kv_cache[b_idx, s_start:s_start+s_size, :]
            # Layout in cache: [S_tile, d_cache] where d_cache = d_rope + d_c
            # We need:
            #   k_pe_cache: [S_tile, d_rope] -> transpose to [d_rope, S_tile] for matmul
            #   c_kv_cache: [S_tile, d_c] -> transpose chunks to [K_TILE, S_tile]
            # -----------------------------------------------------------------

            # Load k_pe portion: kv_cache[b_idx, s_start:s_end, 0:d_rope]
            # Shape [s_size, d_rope] in HBM. We need [d_rope, s_size] in SBUF.
            # d_rope=64 fits as partition, s_size=512 fits as free
            k_pe_sb = nl.ndarray((d_rope, s_size), dtype=kv_cache.dtype, buffer=nl.sbuf)
            # Load transposed: HBM [s_size, d_rope] -> SBUF [d_rope, s_size]
            # Use dma_transpose for this (loads with transpose)
            nisa.dma_transpose(
                dst=k_pe_sb[0:d_rope, 0:s_size],
                src=kv_cache[b_idx, s_start : s_start + s_size, 0:d_rope],
            )

            # -----------------------------------------------------------------
            # Compute RoPE scores: q_pe_col^T @ k_pe_sb
            # q_pe_col is [d_rope, 1] (stationary), k_pe_sb is [d_rope, s_size] (moving)
            # Result: [1, s_size] in PSUM
            # -----------------------------------------------------------------

            # Allocate PSUM for combined scores [1, s_size]
            scores_psum = nl.ndarray((1, s_size), dtype=nl.float32, buffer=nl.psum)

            # RoPE score: stationary=[d_rope, 1], moving=[d_rope, s_size] -> [1, s_size]
            nisa.nc_matmul(
                dst=scores_psum[0:1, 0:s_size],
                stationary=q_pe_col[0:d_rope, 0:1],
                moving=k_pe_sb[0:d_rope, 0:s_size],
            )

            # -----------------------------------------------------------------
            # Compute nope scores: q_nope^T @ c_kv^T
            # q_nope is [1, d_c], c_kv is [s_size, d_c] in HBM
            # We need: sum over K of q_nope_chunk[K_TILE, 1]^T @ c_kv_chunk[K_TILE, s_size]
            # This accumulates into the SAME scores_psum -> hardware accumulation!
            # -----------------------------------------------------------------

            for k_idx in nl.affine_range(num_k_tiles_dc):
                k_start = k_idx * K_TILE

                # Load c_kv chunk transposed: HBM [s_size, K_TILE] -> SBUF [K_TILE, s_size]
                c_kv_chunk = nl.ndarray(
                    (K_TILE, s_size), dtype=kv_cache.dtype, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=c_kv_chunk[0:K_TILE, 0:s_size],
                    src=kv_cache[
                        b_idx,
                        s_start : s_start + s_size,
                        d_rope + k_start : d_rope + k_start + K_TILE,
                    ],
                )

                # Use pre-loaded q_nope column (hoisted out of cache tile loop)
                # Matmul: stationary=[K_TILE, 1], moving=[K_TILE, s_size] -> [1, s_size]
                # Accumulates into scores_psum (same dst = hardware PSUM accumulation)
                nisa.nc_matmul(
                    dst=scores_psum[0:1, 0:s_size],
                    stationary=q_nope_cols[k_idx][0:K_TILE, 0:1],
                    moving=c_kv_chunk[0:K_TILE, 0:s_size],
                )

            # -----------------------------------------------------------------
            # Copy scores from PSUM to SBUF, apply scale
            # scores_psum [1, s_size] -> scores_sb [1, s_size]
            # -----------------------------------------------------------------

            scores_sb = nl.ndarray((1, s_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=scores_sb[0:1, 0:s_size], src=scores_psum[0:1, 0:s_size]
            )

            # Apply softmax scale: scores = scores * scale
            nisa.tensor_scalar(
                dst=scores_sb[0:1, 0:s_size],
                data=scores_sb[0:1, 0:s_size],
                op0=nl.multiply,
                operand0=softmax_scale,
            )

            # -----------------------------------------------------------------
            # Apply attention mask
            # Load mask for this tile: attn_mask[bh, s_start:s_start+s_size]
            # Where mask=0, add LARGE_NEG to scores (drives softmax weight to ~0)
            # Formula: masked_scores = scores + (1 - mask) * LARGE_NEG
            # For mask=1: scores + 0 = scores (unchanged)
            # For mask=0: scores + LARGE_NEG (effectively -inf for softmax)
            # -----------------------------------------------------------------

            mask_sb = nl.ndarray((1, s_size), dtype=attn_mask.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=mask_sb[0:1, 0:s_size],
                src=attn_mask[bh : bh + 1, s_start : s_start + s_size],
            )

            # Convert mask to float: mask_f = float(mask)
            mask_f = nl.ndarray((1, s_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=mask_f[0:1, 0:s_size], src=mask_sb[0:1, 0:s_size])

            # OPTIMIZATION: Reduced from 5 ops to 3 ops.
            # Compute mask_penalty = (1 - mask) * LARGE_NEG using fused ops:
            # Step 1: mask_penalty = mask_f * (-LARGE_NEG) + LARGE_NEG
            #   = mask * (-LARGE_NEG) + LARGE_NEG = LARGE_NEG * (1 - mask)
            # Wait - we want (1-mask)*LARGE_NEG:
            #   When mask=1: 0, when mask=0: LARGE_NEG
            # Compute: penalty = mask * (-LARGE_NEG) + LARGE_NEG
            #   When mask=1: -LARGE_NEG + LARGE_NEG = 0. When mask=0: 0 + LARGE_NEG = LARGE_NEG. Correct!
            mask_penalty = nl.ndarray((1, s_size), dtype=nl.float32, buffer=nl.sbuf)
            # Fused: activation(data=mask_f, scale=-LARGE_NEG, bias=LARGE_NEG_as_bias)
            # But activation needs an op like exp/sigmoid which we don't want here.
            # Use tensor_scalar chain instead (still 2 ops, down from 3 for inv_mask):
            nisa.tensor_scalar(
                dst=mask_penalty[0:1, 0:s_size],
                data=mask_f[0:1, 0:s_size],
                op0=nl.multiply,
                operand0=-LARGE_NEG,
            )
            nisa.tensor_scalar(
                dst=mask_penalty[0:1, 0:s_size],
                data=mask_penalty[0:1, 0:s_size],
                op0=nl.add,
                operand0=LARGE_NEG,
            )

            # Step 2: scores = scores + mask_penalty
            nisa.tensor_tensor(
                dst=scores_sb[0:1, 0:s_size],
                data1=scores_sb[0:1, 0:s_size],
                data2=mask_penalty[0:1, 0:s_size],
                op=nl.add,
            )

            # -----------------------------------------------------------------
            # Online softmax update
            # 1. tile_max = max(scores_sb)
            # 2. new_max = max(running_max, tile_max)
            # 3. correction = exp(running_max - new_max)
            # 4. exp_scores = exp(scores - new_max)
            # 5. running_sum = running_sum * correction + sum(exp_scores)
            # 6. v_accum = v_accum * correction + exp_scores @ c_kv
            # 7. running_max = new_max
            # -----------------------------------------------------------------

            # Step 1: tile_max = max(scores along free dim)
            tile_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=tile_max[0:1, 0:1],
                data=scores_sb[0:1, 0:s_size],
                op=nl.maximum,
                axis=(1,),
            )

            # Step 2: new_max = max(running_max, tile_max)
            new_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=new_max[0:1, 0:1],
                data1=running_max[0:1, 0:1],
                data2=tile_max[0:1, 0:1],
                op=nl.maximum,
            )

            # Step 3: correction = exp(running_max - new_max)
            # Use fused: exp(running_max * 1.0 + (-new_max)) = exp(running_max - new_max)
            correction = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            # First compute diff = running_max - new_max
            diff_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=diff_max[0:1, 0:1],
                data1=running_max[0:1, 0:1],
                data2=new_max[0:1, 0:1],
                op=nl.subtract,
            )
            # Then correction = exp(diff_max)
            nisa.activation(
                dst=correction[0:1, 0:1],
                op=nl.exp,
                data=diff_max[0:1, 0:1],
            )

            # Step 4: exp_scores = exp(scores - new_max)
            # Use fused activation: exp(scores_sb * 1.0 + neg_new_max)
            # nisa.activation with bias=[P,1] broadcasts across free dim
            neg_new_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=neg_new_max[0:1, 0:1],
                data=new_max[0:1, 0:1],
                op0=nl.multiply,
                operand0=-1.0,
            )
            exp_scores = nl.ndarray((1, s_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_scores[0:1, 0:s_size],
                op=nl.exp,
                data=scores_sb[0:1, 0:s_size],
                bias=neg_new_max[0:1, 0:1],
            )

            # Step 5: running_sum = running_sum * correction + sum(exp_scores)
            # running_sum *= correction
            nisa.tensor_tensor(
                dst=running_sum[0:1, 0:1],
                data1=running_sum[0:1, 0:1],
                data2=correction[0:1, 0:1],
                op=nl.multiply,
            )
            # tile_sum = sum(exp_scores)
            tile_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=tile_sum[0:1, 0:1],
                data=exp_scores[0:1, 0:s_size],
                op=nl.add,
                axis=(1,),
            )
            # running_sum += tile_sum
            nisa.tensor_tensor(
                dst=running_sum[0:1, 0:1],
                data1=running_sum[0:1, 0:1],
                data2=tile_sum[0:1, 0:1],
                op=nl.add,
            )

            # Step 6: v_accum = v_accum * correction + exp_scores @ c_kv
            # Scale existing accumulation by correction factor
            # v_accum is [1, d_c], correction is [1, 1]
            # Strategy: create correction_wide[1, d_c] = exp(diff_max) replicated,
            # using nisa.activation(op=exp, data=zeros, bias=diff_max) which broadcasts bias
            correction_wide = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=correction_wide[0:1, 0:d_c],
                op=nl.exp,
                data=v_accum[0:1, 0:d_c],  # dummy data (will be scaled to 0)
                scale=0.0,
                bias=diff_max[0:1, 0:1],
            )
            # correction_wide = exp(v_accum * 0.0 + diff_max) = exp(diff_max) = correction
            # Now multiply with matching free dims
            nisa.tensor_tensor(
                dst=v_accum[0:1, 0:d_c],
                data1=v_accum[0:1, 0:d_c],
                data2=correction_wide[0:1, 0:d_c],
                op=nl.multiply,
            )

            # V matmul: exp_scores[1, s_size] @ c_kv[s_size, d_c] -> [1, d_c]
            # nc_matmul: stationary[K, M] @ moving[K, N] -> dst[M, N]
            # We want [1, d_c] = [1, s_size] @ [s_size, d_c]
            # Approach: tile over s_size in chunks of S_SUB_TILE=128
            # For each chunk: stationary = exp_scores_chunk^T = [128, 1]
            #                 moving = c_kv_chunk = [128, d_c=512]
            #                 result: [1, 512] accumulated in PSUM
            #
            # Transpose [1, 128] -> [128, 1] via HBM round-trip with dma_transpose
            # (nc_transpose limited to 32x32 in NKI 0.3.0)

            # Temp HBM buffer for exp_scores transpose (allocated once, reused)
            exp_scores_hbm = nl.ndarray(
                (1, s_size), dtype=nl.float32, buffer=nl.shared_hbm
            )
            nisa.dma_copy(
                dst=exp_scores_hbm[0:1, 0:s_size], src=exp_scores[0:1, 0:s_size]
            )

            # V accumulation tiled over sequence sub-chunks:
            # Split s_size into S_SUB_TILE = 128 chunks
            S_SUB_TILE = 128
            num_s_sub_tiles = div_ceil(s_size, S_SUB_TILE)

            v_psum = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.psum)

            for s_sub_idx in nl.affine_range(num_s_sub_tiles):
                s_sub_start = s_sub_idx * S_SUB_TILE
                s_sub_size = min(S_SUB_TILE, s_size - s_sub_start)

                # Load exp_scores chunk as column [s_sub_size, 1] via dma_transpose
                # HBM src: exp_scores_hbm[0:1, s_sub_start:s_sub_start+s_sub_size] = [1, s_sub_size]
                # dma_transpose -> SBUF [s_sub_size, 1]
                exp_chunk_col = nl.ndarray(
                    (s_sub_size, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=exp_chunk_col[0:s_sub_size, 0:1],
                    src=exp_scores_hbm[0:1, s_sub_start : s_sub_start + s_sub_size],
                )

                # Cast to match kv_cache dtype for nc_matmul (both operands must match)
                exp_chunk_col_cast = nl.ndarray(
                    (s_sub_size, 1), dtype=kv_cache.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(
                    dst=exp_chunk_col_cast[0:s_sub_size, 0:1],
                    src=exp_chunk_col[0:s_sub_size, 0:1],
                )

                # Load c_kv chunk: kv_cache[b_idx, s_start+s_sub_start:..., d_rope:d_rope+d_c]
                # Shape [s_sub_size, d_c] in HBM -> [s_sub_size, d_c] in SBUF
                # s_sub_size=128 (partition), d_c=512 (free) - fits in SBUF!
                c_kv_v_chunk = nl.ndarray(
                    (s_sub_size, d_c), dtype=kv_cache.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=c_kv_v_chunk[0:s_sub_size, 0:d_c],
                    src=kv_cache[
                        b_idx,
                        s_start + s_sub_start : s_start + s_sub_start + s_sub_size,
                        d_rope : d_rope + d_c,
                    ],
                )

                # Matmul: stationary=[s_sub_size, 1], moving=[s_sub_size, d_c] -> [1, d_c]
                # K=s_sub_size<=128, M=1, N=d_c=512 = PSUM_F_MAX. OK.
                # Accumulates into v_psum across s_sub_tiles (PSUM accumulates in float32)
                nisa.nc_matmul(
                    dst=v_psum[0:1, 0:d_c],
                    stationary=exp_chunk_col_cast[0:s_sub_size, 0:1],
                    moving=c_kv_v_chunk[0:s_sub_size, 0:d_c],
                )

            # Copy V matmul result from PSUM to SBUF
            v_tile_result = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_tile_result[0:1, 0:d_c], src=v_psum[0:1, 0:d_c])

            # Add to running v_accum
            nisa.tensor_tensor(
                dst=v_accum[0:1, 0:d_c],
                data1=v_accum[0:1, 0:d_c],
                data2=v_tile_result[0:1, 0:d_c],
                op=nl.add,
            )

            # Step 7: Update running_max
            nisa.tensor_copy(dst=running_max[0:1, 0:1], src=new_max[0:1, 0:1])

        # =====================================================================
        # After all cache tiles: normalize v_accum and apply V absorption
        # =====================================================================

        # Normalize: v_accum = v_accum / running_sum
        # OPTIMIZATION: Use -log(running_sum) directly instead of reciprocal + log + exp.
        # exp(-log(running_sum)) = 1/running_sum, broadcast via activation bias.
        neg_log_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        # Compute log(running_sum)
        nisa.activation(
            dst=neg_log_sum[0:1, 0:1],
            op=nl.log,
            data=running_sum[0:1, 0:1],
        )
        # Negate: neg_log_sum = -log(running_sum)
        nisa.tensor_scalar(
            dst=neg_log_sum[0:1, 0:1],
            data=neg_log_sum[0:1, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
        )
        # Broadcast: inv_sum_wide = exp(0 + neg_log_sum) = 1/running_sum replicated to [1, d_c]
        inv_sum_wide = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_sum_wide[0:1, 0:d_c],
            op=nl.exp,
            data=v_accum[0:1, 0:d_c],  # dummy (scaled to 0)
            scale=0.0,
            bias=neg_log_sum[0:1, 0:1],
        )
        # v_accum *= inv_sum_wide
        nisa.tensor_tensor(
            dst=v_accum[0:1, 0:d_c],
            data1=v_accum[0:1, 0:d_c],
            data2=inv_sum_wide[0:1, 0:d_c],
            op=nl.multiply,
        )

        # =====================================================================
        # V absorption: output = v_accum @ v_absorb[h_idx]^T
        # v_accum is [1, d_c=512], v_absorb[h_idx] is [d_v=256, d_c=512]
        # output = v_accum @ v_absorb^T = [1, d_c] @ [d_c, d_v] = [1, d_v]
        #
        # nc_matmul: stationary[K, M] @ moving[K, N] -> dst[M, N]
        # We want [1, d_v] = [1, d_c] @ [d_c, d_v]
        # Rewrite: stationary = v_accum^T = [d_c, 1], moving = v_absorb^T = [d_c, d_v]
        # But v_absorb[h_idx] shape in HBM is [d_v, d_c]
        # v_absorb^T has shape [d_c, d_v] - load transposed: HBM[d_v, d_c] -> SBUF[d_c, d_v]
        #
        # K = d_c = 512, exceeds K limit of 128 per matmul step.
        # Need to tile K dimension: sum over K_TILE=128 chunks.
        # stationary chunk: [K_TILE=128, M=1]
        # moving chunk: [K_TILE=128, N=d_v=256]
        # Result: [1, d_v=256] accumulated across 4 K-tiles
        # =====================================================================

        absorb_psum = nl.ndarray((1, d_v), dtype=nl.float32, buffer=nl.psum)

        # Store v_accum to HBM for transposed reload per K-chunk
        # (nc_transpose limited to 32x32, v_accum chunks are [1, 128])
        v_accum_hbm = nl.ndarray((1, d_c), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=v_accum_hbm[0:1, 0:d_c], src=v_accum[0:1, 0:d_c])

        for k_idx in nl.affine_range(num_k_tiles_dc):
            k_start = k_idx * K_TILE

            # Load v_accum chunk as column [K_TILE, 1] via dma_transpose
            # HBM src: v_accum_hbm[0:1, k_start:k_start+K_TILE] = [1, K_TILE]
            # dma_transpose -> SBUF [K_TILE, 1]
            v_accum_chunk_col = nl.ndarray(
                (K_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=v_accum_chunk_col[0:K_TILE, 0:1],
                src=v_accum_hbm[0:1, k_start : k_start + K_TILE],
            )

            # Cast to match v_absorb dtype for nc_matmul
            v_accum_chunk_cast = nl.ndarray(
                (K_TILE, 1), dtype=v_absorb.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=v_accum_chunk_cast[0:K_TILE, 0:1],
                src=v_accum_chunk_col[0:K_TILE, 0:1],
            )

            # Load v_absorb[h_idx] chunk: HBM shape [d_v, d_c]
            # We need [K_TILE, d_v] from v_absorb[h_idx, :, k_start:k_start+K_TILE]
            # v_absorb is [H, d_v, d_c], so v_absorb[h_idx, :, k_start:k_start+K_TILE]
            # gives [d_v, K_TILE] in HBM. We need it as [K_TILE, d_v] in SBUF.
            # Use dma_transpose: HBM [d_v, K_TILE] -> SBUF [K_TILE, d_v]
            v_absorb_chunk = nl.ndarray(
                (K_TILE, d_v), dtype=v_absorb.dtype, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=v_absorb_chunk[0:K_TILE, 0:d_v],
                src=v_absorb[h_idx, 0:d_v, k_start : k_start + K_TILE],
            )

            # Matmul: stationary=[K_TILE, 1], moving=[K_TILE, d_v] -> [1, d_v]
            # Accumulates across K-tiles into absorb_psum
            nisa.nc_matmul(
                dst=absorb_psum[0:1, 0:d_v],
                stationary=v_accum_chunk_cast[0:K_TILE, 0:1],
                moving=v_absorb_chunk[0:K_TILE, 0:d_v],
            )

        # Copy absorption result from PSUM to SBUF
        absorb_result = nl.ndarray((1, d_v), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=absorb_result[0:1, 0:d_v], src=absorb_psum[0:1, 0:d_v])

        # Cast to output dtype if needed and store to HBM
        if q_pe.dtype != nl.float32:
            absorb_out = nl.ndarray((1, d_v), dtype=q_pe.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=absorb_out[0:1, 0:d_v], src=absorb_result[0:1, 0:d_v])
            nisa.dma_copy(dst=output[bh : bh + 1, 0:d_v], src=absorb_out[0:1, 0:d_v])
        else:
            nisa.dma_copy(dst=output[bh : bh + 1, 0:d_v], src=absorb_result[0:1, 0:d_v])

    return output
