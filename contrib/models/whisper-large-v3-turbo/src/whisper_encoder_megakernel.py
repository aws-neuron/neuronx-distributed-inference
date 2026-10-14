# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper encoder layer megakernel — hand-written NKI ISA.

Fuses an entire Whisper encoder layer into a single @nki.jit kernel call,
following the Boltz-2 full_pairformer_layer_spmd approach.

Per-layer operation sequence (32 layers total):
  Phase 1:  LayerNorm (pre-attention)
  Phase 2:  Fused QKV projection (single tiled matmul)
  Phase 3:  Bidirectional flash attention (online softmax, no causal mask)
  Phase 4:  Output projection + residual add
  Phase 5:  LayerNorm (pre-MLP)
  Phase 6:  MLP up-projection (FC1)
  Phase 7:  GELU activation
  Phase 8:  MLP down-projection (FC2) + residual add

Whisper large-v3-turbo encoder dimensions:
  hidden_dim (d_model) = 1280  (10 tiles of 128)
  num_heads            = 20
  head_dim             = 64    (50% PE utilization on Q@K^T)
  seq_len              = 1500  (pad to 1536 = 12 tiles for kernel)
  MLP intermediate     = 5120  (40 tiles of 128)
  num_layers           = 32
  dtype                = bf16

Hardware: NeuronCore v3 (trn2), 128-wide partition dimension.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128

# Whisper encoder constants
D_MODEL = 1280  # hidden dimension
N_HEADS = 20  # attention heads
HEAD_DIM = 64  # per-head dimension
MLP_DIM = 5120  # MLP intermediate dimension
SEQ_PAD = 1536  # padded sequence length (1500 -> 1536 = 12 * 128)
SEQ_ACTUAL = 1500  # actual sequence length

# Tile counts
N_HIDDEN = D_MODEL // P_MAX  # 10 tiles for hidden dim
N_MLP = MLP_DIM // P_MAX  # 40 tiles for MLP intermediate
N_SEQ = SEQ_PAD // P_MAX  # 12 tiles for padded sequence


# ============================================================================
# Helper: Transpose SBUF -> PSUM -> SBUF
# ============================================================================
def _transpose_to_sbuf(x):
    """nc_transpose x from SBUF -> PSUM -> SBUF."""
    x_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=x_t_psum, data=x)
    x_t = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_t, src=x_t_psum)
    return x_t


# ============================================================================
# Helper: Prepare weight (load from HBM + transpose)
# ============================================================================
def _prepare_weight(w_hbm):
    """Load and transpose a [P_MAX, P_MAX] weight tile from HBM."""
    w = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    w_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=w_t_psum, data=w)
    w_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_t, src=w_t_psum)
    return w_t


# ============================================================================
# Helper: matmul with pre-transposed weight
# ============================================================================
def _matmul_with_w_t(x_t, w_t):
    """Compute x @ W^T using pre-transposed weight in SBUF. Returns bf16."""
    result_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=result_psum, stationary=x_t, moving=w_t)
    result = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_psum)
    return result


# ============================================================================
# Helper: LayerNorm on a tile [P_MAX, F] in SBUF
# ============================================================================
def _layer_norm_tile(x_tile, weight_tiled, bias_tiled, F, eps=1e-5):
    """LayerNorm on a [P_MAX, F] tile entirely in SBUF.

    Args:
        x_tile: [P_MAX, F] bf16 in SBUF
        weight_tiled: [P_MAX, F] bf16 -- pre-tiled (each row identical)
        bias_tiled: [P_MAX, F] bf16 -- pre-tiled (each row identical)
        F: int, free dimension size
        eps: float
    Returns:
        normalized: [P_MAX, F] bf16 in SBUF
    """
    inv_F = 1.0 / float(F)

    # Cast to f32
    x_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_f32, src=x_tile)

    # Mean: reduce over free dim
    sum_x = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_x, op=nl.add, data=x_f32, axis=1)
    mean = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_F, engine=nisa.vector_engine
    )

    # Center: x - mean
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
    norm_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=norm_f32,
        data=centered,
        op0=nl.multiply,
        operand0=rsqrt_std,
        engine=nisa.vector_engine,
    )

    # Scale + bias (both [P_MAX, F], pre-tiled on host)
    w_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_f32, src=weight_tiled)
    scaled = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=norm_f32, data2=w_f32, op=nl.multiply)

    b_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_f32, src=bias_tiled)
    result_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=result_f32, data1=scaled, data2=b_f32, op=nl.add)

    result = nl.ndarray((P_MAX, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_f32)
    return result


# ============================================================================
# Phase: Bidirectional flash attention for one sequence tile
# ============================================================================
def _attention_for_seq_tile(
    q_hbm,  # [N_SEQ * P_MAX, N_HEADS * HEAD_DIM] -- all Q in private_hbm
    k_hbm,  # [N_SEQ * P_MAX, N_HEADS * HEAD_DIM] -- all K in private_hbm
    v_hbm,  # [N_SEQ * P_MAX, N_HEADS * HEAD_DIM] -- all V in private_hbm
    out_hbm,  # [N_SEQ * P_MAX, N_HEADS * HEAD_DIM] -- output in private_hbm
    j_tile,  # query tile index (compile-time from static_range)
    n_seq_tiles,  # number of sequence tiles
    n_heads,  # number of heads
    head_dim,  # per-head dimension
    scale,  # 1/sqrt(head_dim)
):
    """Bidirectional flash attention for query tile j_tile, all heads.

    Uses online softmax (Milakov & Gimelshein 2018). No causal mask.
    Q/K/V layout in HBM: [seq_tiles * P_MAX, n_heads * head_dim]
    Each seq tile is P_MAX positions, heads are concatenated along dim 1.
    """
    Hd = n_heads * head_dim
    j_start = j_tile * P_MAX

    for h in nl.affine_range(n_heads):
        hd_start = h * head_dim

        # Load Q tile: [P_MAX, head_dim] from q_hbm
        q_tile = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_tile,
            src=q_hbm[j_start : j_start + P_MAX, hd_start : hd_start + head_dim],
        )

        # Pad Q to [P_MAX, P_MAX] for nc_matmul (head_dim=64 < P_MAX=128)
        q_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=q_padded, value=0.0)
        nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:head_dim], src=q_tile)
        q_t = _transpose_to_sbuf(q_padded)

        # Online softmax accumulators
        m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=m_prev, value=-1e30)
        l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=l_prev, value=0.0)
        o_acc = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=o_acc, value=0.0)

        # Sequential over key tiles (required for online softmax)
        for k_tile_idx in nl.sequential_range(n_seq_tiles):
            k_start = k_tile_idx * P_MAX

            # Load K tile: [P_MAX, head_dim]
            k_tile_sb = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_tile_sb,
                src=k_hbm[k_start : k_start + P_MAX, hd_start : hd_start + head_dim],
            )

            # Load V tile: [P_MAX, head_dim]
            v_tile_sb = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_tile_sb,
                src=v_hbm[k_start : k_start + P_MAX, hd_start : hd_start + head_dim],
            )

            # Pad K to [P_MAX, P_MAX], transpose
            k_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=k_padded, value=0.0)
            nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:head_dim], src=k_tile_sb)
            k_t = _transpose_to_sbuf(k_padded)

            # Q @ K^T -> logits [P_MAX, P_MAX]
            logits_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)
            logits = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=logits, src=logits_psum)

            # Scale by 1/sqrt(d)
            logits_scaled = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=logits_scaled,
                data=logits,
                op0=nl.multiply,
                operand0=scale,
                engine=nisa.vector_engine,
            )

            # Online softmax step 1: tile max
            tile_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=logits_scaled, axis=1)

            # Step 2: running max
            m_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m_new, data1=m_prev, data2=tile_max, op=nl.maximum)

            # Step 3: correction = exp(m_prev - m_new)
            m_diff = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m_diff, data1=m_prev, data2=m_new, op=nl.subtract)
            correction = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=correction, op=nl.exp, data=m_diff, bias=None, scale=1.0
            )

            # Step 4: exp(logits - m_new)
            logits_shifted = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logits_shifted,
                data=logits_scaled,
                op0=nl.subtract,
                operand0=m_new,
                engine=nisa.vector_engine,
            )
            exp_logits = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_logits, op=nl.exp, data=logits_shifted, bias=None, scale=1.0
            )

            # Step 5: update l = l * correction + sum(exp_logits)
            l_corrected = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=l_corrected, data1=l_prev, data2=correction, op=nl.multiply
            )
            tile_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_logits, axis=1)
            l_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=l_new, data1=l_corrected, data2=tile_sum, op=nl.add)

            # Step 6: rescale output accumulator
            o_scaled = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=o_scaled,
                data=o_acc,
                op0=nl.multiply,
                operand0=correction,
                engine=nisa.vector_engine,
            )

            # exp_logits @ V: [P_MAX, P_MAX] @ [P_MAX, head_dim] -> [P_MAX, head_dim]
            exp_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=exp_bf16, src=exp_logits)
            exp_t = _transpose_to_sbuf(exp_bf16)

            pv_psum = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)
            pv_sbuf = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

            # Step 7: accumulate
            nisa.tensor_tensor(dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add)

            # Update running state
            nisa.tensor_copy(dst=m_prev, src=m_new)
            nisa.tensor_copy(dst=l_prev, src=l_new)

        # Finalize: output = o_acc / l
        inv_l = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_l, data=l_prev)
        o_final = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=o_final,
            data=o_acc,
            op0=nl.multiply,
            operand0=inv_l,
            engine=nisa.vector_engine,
        )

        # Cast to bf16 and store
        o_out = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_out, src=o_final)
        nisa.dma_copy(
            dst=out_hbm[j_start : j_start + P_MAX, hd_start : hd_start + head_dim],
            src=o_out,
        )


# ============================================================================
# Main entry point: Whisper encoder layer megakernel
# ============================================================================
@nki.jit
def whisper_encoder_layer_fwd(
    x_in,  # [N_SEQ * P_MAX, D_MODEL] bf16 -- input activations
    # Pre-attention LayerNorm weights (pre-tiled to [P_MAX, D_MODEL])
    attn_ln_w,  # [P_MAX, D_MODEL] bf16
    attn_ln_b,  # [P_MAX, D_MODEL] bf16
    # Fused QKV weight: [3 * D_MODEL, D_MODEL]
    qkv_w,  # [3 * D_MODEL, D_MODEL] bf16
    qkv_b,  # [P_MAX, 3 * D_MODEL] bf16 (pre-tiled, each row identical)
    # Output projection: [D_MODEL, D_MODEL]
    out_w,  # [D_MODEL, D_MODEL] bf16
    out_b,  # [P_MAX, D_MODEL] bf16 (pre-tiled, each row identical)
    # Pre-MLP LayerNorm weights (pre-tiled to [P_MAX, D_MODEL])
    mlp_ln_w,  # [P_MAX, D_MODEL] bf16
    mlp_ln_b,  # [P_MAX, D_MODEL] bf16
    # MLP FC1 (up projection): [MLP_DIM, D_MODEL]
    fc1_w,  # [MLP_DIM, D_MODEL] bf16
    fc1_b,  # [P_MAX, MLP_DIM] bf16 (pre-tiled, each row identical)
    # MLP FC2 (down projection): [D_MODEL, MLP_DIM]
    fc2_w,  # [D_MODEL, MLP_DIM] bf16
    fc2_b,  # [P_MAX, D_MODEL] bf16 (pre-tiled, each row identical)
    # Scalars
    n_seq_tiles: int = N_SEQ,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    eps: float = 1e-5,
) -> nl.ndarray:
    """Execute one Whisper encoder layer as a single fused kernel.

    Sequence: LN -> QKV -> FlashAttn -> OutProj -> Residual ->
              LN -> FC1 -> GELU -> FC2 -> Residual

    Input x_in is [n_seq_tiles * P_MAX, D_MODEL] stored flat in HBM.
    Returns x_out of the same shape.
    """
    S = n_seq_tiles * P_MAX  # padded sequence length
    Hd = n_heads * head_dim
    scale = 1.0 / (head_dim**0.5)

    # Output tensor
    x_out = nl.ndarray((S, D_MODEL), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Scratch buffers in private HBM for Q, K, V, attn_out
    q_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    k_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    v_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    attn_out_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    # Scratch for post-attention residual (need original x for residual add)
    post_attn_buf = nl.ndarray((S, D_MODEL), dtype=nl.bfloat16, buffer=nl.private_hbm)

    # Load LayerNorm weights once (shared across all seq tiles)
    ln1_w = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_w, src=attn_ln_w)
    ln1_b = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_b, src=attn_ln_b)

    # ================================================================
    # Phase 1-2: LayerNorm + QKV projection for each sequence tile
    # QKV weight is [3*D_MODEL, D_MODEL] = [3840, 1280].
    # For each seq tile: x_normed [P_MAX, 1280] @ W_qkv^T [1280, 3840]
    # Output: 3 * 10 = 30 output chunks of [P_MAX, P_MAX]
    # We split into Q[P_MAX, Hd], K[P_MAX, Hd], V[P_MAX, Hd]
    # where Hd = n_heads * head_dim = 1280
    # ================================================================
    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        # Load input tile [P_MAX, D_MODEL]
        x_tile = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_in[s_start : s_start + P_MAX, 0:D_MODEL])

        # LayerNorm
        x_normed = _layer_norm_tile(x_tile, ln1_w, ln1_b, D_MODEL, eps)

        # Split normed input into N_HIDDEN chunks and transpose each
        # x_normed is [P_MAX, 1280] = [P_MAX, 10*128]
        # We need x_chunks_t: tuple of 10 transposed [P_MAX, P_MAX] tiles
        xc_t_0 = _transpose_to_sbuf(
            nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        )
        # Can't do that -- need to copy first, then transpose.
        # Let me do it properly:

        # Extract and transpose each hidden chunk
        xc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_0, src=x_normed[0:P_MAX, 0:P_MAX])
        xc_t_0 = _transpose_to_sbuf(xc_0)

        xc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_1, src=x_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        xc_t_1 = _transpose_to_sbuf(xc_1)

        xc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_2, src=x_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        xc_t_2 = _transpose_to_sbuf(xc_2)

        xc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_3, src=x_normed[0:P_MAX, 3 * P_MAX : 4 * P_MAX])
        xc_t_3 = _transpose_to_sbuf(xc_3)

        xc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_4, src=x_normed[0:P_MAX, 4 * P_MAX : 5 * P_MAX])
        xc_t_4 = _transpose_to_sbuf(xc_4)

        xc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_5, src=x_normed[0:P_MAX, 5 * P_MAX : 6 * P_MAX])
        xc_t_5 = _transpose_to_sbuf(xc_5)

        xc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_6, src=x_normed[0:P_MAX, 6 * P_MAX : 7 * P_MAX])
        xc_t_6 = _transpose_to_sbuf(xc_6)

        xc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_7, src=x_normed[0:P_MAX, 7 * P_MAX : 8 * P_MAX])
        xc_t_7 = _transpose_to_sbuf(xc_7)

        xc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_8, src=x_normed[0:P_MAX, 8 * P_MAX : 9 * P_MAX])
        xc_t_8 = _transpose_to_sbuf(xc_8)

        xc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_9, src=x_normed[0:P_MAX, 9 * P_MAX : 10 * P_MAX])
        xc_t_9 = _transpose_to_sbuf(xc_9)

        xc_t = (
            xc_t_0,
            xc_t_1,
            xc_t_2,
            xc_t_3,
            xc_t_4,
            xc_t_5,
            xc_t_6,
            xc_t_7,
            xc_t_8,
            xc_t_9,
        )

        # QKV matmul: for each of 30 output chunks (3*10), accumulate across
        # 10 input chunks. Output goes to Q/K/V buffers in private_hbm.
        # qkv_w is [3*D_MODEL, D_MODEL] = [3840, 1280]
        # Layout: rows 0..1279 = Q weights, 1280..2559 = K, 2560..3839 = V
        # Each section: 10 output chunks of [P_MAX, P_MAX]
        # For output chunk o (0..9 for Q, 10..19 for K, 20..29 for V):
        #   acc = sum over i=0..9 of: x_chunks_t[i] @ W[o*128:(o+1)*128, i*128:(i+1)*128]^T

        # Q projection: 10 output chunks
        for o in nl.static_range(N_HIDDEN):
            q_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=q_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=q_acc, data1=q_acc, data2=ps, op=nl.add)
            # Add bias and store
            q_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_out, src=q_acc)
            # Bias: load [P_MAX, P_MAX] slice (pre-tiled on host)
            q_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_bias, src=qkv_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            q_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q_biased, data1=q_out, data2=q_bias, op=nl.add)
            nisa.dma_copy(
                dst=q_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=q_biased,
            )

        # K projection: 10 output chunks (rows D_MODEL..2*D_MODEL of qkv_w)
        for o in nl.static_range(N_HIDDEN):
            k_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=k_acc, value=0.0)
            w_row_off = D_MODEL + o * P_MAX
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[w_row_off : w_row_off + P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=k_acc, data1=k_acc, data2=ps, op=nl.add)
            k_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_out, src=k_acc)
            k_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_bias,
                src=qkv_b[0:P_MAX, D_MODEL + o * P_MAX : D_MODEL + (o + 1) * P_MAX],
            )
            k_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_biased, data1=k_out, data2=k_bias, op=nl.add)
            nisa.dma_copy(
                dst=k_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=k_biased,
            )

        # V projection: 10 output chunks (rows 2*D_MODEL..3*D_MODEL of qkv_w)
        for o in nl.static_range(N_HIDDEN):
            v_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=v_acc, value=0.0)
            w_row_off = 2 * D_MODEL + o * P_MAX
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[w_row_off : w_row_off + P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=v_acc, data1=v_acc, data2=ps, op=nl.add)
            v_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_out, src=v_acc)
            v_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_bias,
                src=qkv_b[
                    0:P_MAX, 2 * D_MODEL + o * P_MAX : 2 * D_MODEL + (o + 1) * P_MAX
                ],
            )
            v_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_biased, data1=v_out, data2=v_bias, op=nl.add)
            nisa.dma_copy(
                dst=v_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=v_biased,
            )

    # ================================================================
    # Phase 3: Bidirectional flash attention
    # Q/K/V are in private_hbm: [S, Hd] where Hd = n_heads * head_dim
    # Process each query tile independently (parallel over seq tiles)
    # ================================================================
    for j_tile in nl.static_range(N_SEQ):
        _attention_for_seq_tile(
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            j_tile,
            n_seq_tiles,
            n_heads,
            head_dim,
            scale,
        )

    # ================================================================
    # Phase 4: Output projection + residual add
    # attn_out is [S, Hd=1280], out_w is [D_MODEL, D_MODEL] = [1280, 1280]
    # For each seq tile: attn_out[P_MAX, 1280] @ out_w^T -> [P_MAX, 1280]
    # Then add bias and residual (original x_in)
    # Result stored to post_attn_buf for Phase 5-8
    # ================================================================
    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        # Load attn output tile, split into chunks, transpose each
        ao_c_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ao_c_0, src=attn_out_buf[s_start : s_start + P_MAX, 0:P_MAX])
        ao_t_0 = _transpose_to_sbuf(ao_c_0)

        ao_c_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_1, src=attn_out_buf[s_start : s_start + P_MAX, P_MAX : 2 * P_MAX]
        )
        ao_t_1 = _transpose_to_sbuf(ao_c_1)

        ao_c_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_2,
            src=attn_out_buf[s_start : s_start + P_MAX, 2 * P_MAX : 3 * P_MAX],
        )
        ao_t_2 = _transpose_to_sbuf(ao_c_2)

        ao_c_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_3,
            src=attn_out_buf[s_start : s_start + P_MAX, 3 * P_MAX : 4 * P_MAX],
        )
        ao_t_3 = _transpose_to_sbuf(ao_c_3)

        ao_c_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_4,
            src=attn_out_buf[s_start : s_start + P_MAX, 4 * P_MAX : 5 * P_MAX],
        )
        ao_t_4 = _transpose_to_sbuf(ao_c_4)

        ao_c_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_5,
            src=attn_out_buf[s_start : s_start + P_MAX, 5 * P_MAX : 6 * P_MAX],
        )
        ao_t_5 = _transpose_to_sbuf(ao_c_5)

        ao_c_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_6,
            src=attn_out_buf[s_start : s_start + P_MAX, 6 * P_MAX : 7 * P_MAX],
        )
        ao_t_6 = _transpose_to_sbuf(ao_c_6)

        ao_c_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_7,
            src=attn_out_buf[s_start : s_start + P_MAX, 7 * P_MAX : 8 * P_MAX],
        )
        ao_t_7 = _transpose_to_sbuf(ao_c_7)

        ao_c_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_8,
            src=attn_out_buf[s_start : s_start + P_MAX, 8 * P_MAX : 9 * P_MAX],
        )
        ao_t_8 = _transpose_to_sbuf(ao_c_8)

        ao_c_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_9,
            src=attn_out_buf[s_start : s_start + P_MAX, 9 * P_MAX : 10 * P_MAX],
        )
        ao_t_9 = _transpose_to_sbuf(ao_c_9)

        ao_t = (
            ao_t_0,
            ao_t_1,
            ao_t_2,
            ao_t_3,
            ao_t_4,
            ao_t_5,
            ao_t_6,
            ao_t_7,
            ao_t_8,
            ao_t_9,
        )

        # Tiled matmul: for each output chunk, accumulate across 10 input chunks
        for o in nl.static_range(N_HIDDEN):
            out_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=out_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    out_w[o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=ao_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=out_acc, data1=out_acc, data2=ps, op=nl.add)

            # Cast, add bias, add residual
            proj_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=proj_chunk, src=out_acc)
            ob = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=ob, src=out_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            proj_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=proj_biased, data1=proj_chunk, data2=ob, op=nl.add)
            x_orig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_orig,
                src=x_in[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
            )
            x_res = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_res, data1=x_orig, data2=proj_biased, op=nl.add)
            nisa.dma_copy(
                dst=post_attn_buf[
                    s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
                src=x_res,
            )

    # ================================================================
    # Phase 5-8: LayerNorm + MLP (FC1 -> GELU -> FC2) + residual
    # post_attn_buf has the post-attention residual output.
    # FC1: [MLP_DIM, D_MODEL] = [5120, 1280], 40 out x 10 in chunks
    # FC2: [D_MODEL, MLP_DIM] = [1280, 5120], 10 out x 40 in chunks
    # ================================================================
    # Load MLP LayerNorm weights
    ln2_w = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_w, src=mlp_ln_w)
    ln2_b = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_b, src=mlp_ln_b)

    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        # Load post-attention tile
        pa_tile = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=pa_tile,
            src=post_attn_buf[s_start : s_start + P_MAX, 0:D_MODEL],
        )

        # LayerNorm
        x_normed = _layer_norm_tile(pa_tile, ln2_w, ln2_b, D_MODEL, eps)

        # Split and transpose normed input (10 chunks)
        mc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_0, src=x_normed[0:P_MAX, 0:P_MAX])
        mc_t_0 = _transpose_to_sbuf(mc_0)

        mc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_1, src=x_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        mc_t_1 = _transpose_to_sbuf(mc_1)

        mc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_2, src=x_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        mc_t_2 = _transpose_to_sbuf(mc_2)

        mc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_3, src=x_normed[0:P_MAX, 3 * P_MAX : 4 * P_MAX])
        mc_t_3 = _transpose_to_sbuf(mc_3)

        mc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_4, src=x_normed[0:P_MAX, 4 * P_MAX : 5 * P_MAX])
        mc_t_4 = _transpose_to_sbuf(mc_4)

        mc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_5, src=x_normed[0:P_MAX, 5 * P_MAX : 6 * P_MAX])
        mc_t_5 = _transpose_to_sbuf(mc_5)

        mc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_6, src=x_normed[0:P_MAX, 6 * P_MAX : 7 * P_MAX])
        mc_t_6 = _transpose_to_sbuf(mc_6)

        mc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_7, src=x_normed[0:P_MAX, 7 * P_MAX : 8 * P_MAX])
        mc_t_7 = _transpose_to_sbuf(mc_7)

        mc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_8, src=x_normed[0:P_MAX, 8 * P_MAX : 9 * P_MAX])
        mc_t_8 = _transpose_to_sbuf(mc_8)

        mc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_9, src=x_normed[0:P_MAX, 9 * P_MAX : 10 * P_MAX])
        mc_t_9 = _transpose_to_sbuf(mc_9)

        mc_t = (
            mc_t_0,
            mc_t_1,
            mc_t_2,
            mc_t_3,
            mc_t_4,
            mc_t_5,
            mc_t_6,
            mc_t_7,
            mc_t_8,
            mc_t_9,
        )

        # FC1 + GELU + FC2 fused with accumulation into output chunks
        # FC2 accumulators: 10 output chunks (accumulated across 40 hidden chunks)
        fc2_acc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_0, value=0.0)
        fc2_acc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_1, value=0.0)
        fc2_acc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_2, value=0.0)
        fc2_acc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_3, value=0.0)
        fc2_acc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_4, value=0.0)
        fc2_acc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_5, value=0.0)
        fc2_acc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_6, value=0.0)
        fc2_acc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_7, value=0.0)
        fc2_acc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_8, value=0.0)
        fc2_acc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_9, value=0.0)

        fc2_acc = (
            fc2_acc_0,
            fc2_acc_1,
            fc2_acc_2,
            fc2_acc_3,
            fc2_acc_4,
            fc2_acc_5,
            fc2_acc_6,
            fc2_acc_7,
            fc2_acc_8,
            fc2_acc_9,
        )

        # For each hidden chunk h (0..39):
        #   1. FC1: accumulate across 10 input chunks -> [P_MAX, P_MAX]
        #   2. Add FC1 bias
        #   3. GELU activation
        #   4. FC2: accumulate this hidden chunk into each of 10 output chunks
        for h in nl.static_range(N_MLP):
            # FC1 for hidden chunk h: accumulate across 10 input chunks
            fc1_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=fc1_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    fc1_w[h * P_MAX : (h + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=mc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=fc1_acc, data1=fc1_acc, data2=ps, op=nl.add)

            # Add FC1 bias
            fc1_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc1_bf16, src=fc1_acc)
            fb1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=fb1, src=fc1_b[0:P_MAX, h * P_MAX : (h + 1) * P_MAX])
            fc1_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=fc1_biased, data1=fc1_bf16, data2=fb1, op=nl.add)

            # GELU approximation: x * sigmoid(1.702 * x)
            # This matches PyTorch's F.gelu() "tanh" approximation closely
            scaled_x = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled_x,
                data=fc1_biased,
                op0=nl.multiply,
                operand0=1.702,
                engine=nisa.vector_engine,
            )
            sig_x = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(
                dst=sig_x, op=nl.sigmoid, data=scaled_x, bias=None, scale=1.0
            )
            gelu_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gelu_out, data1=fc1_biased, data2=sig_x, op=nl.multiply
            )

            # FC2: accumulate this hidden chunk's contribution to each output chunk
            gelu_t = _transpose_to_sbuf(gelu_out)
            for o in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    fc2_w[o * P_MAX : (o + 1) * P_MAX, h * P_MAX : (h + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=gelu_t, moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(
                    dst=fc2_acc[o], data1=fc2_acc[o], data2=ps, op=nl.add
                )

        # Add FC2 bias, residual, and store to output
        for o in nl.static_range(N_HIDDEN):
            fc2_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc2_chunk, src=fc2_acc[o])
            fb2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=fb2, src=fc2_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            fc2_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=fc2_biased, data1=fc2_chunk, data2=fb2, op=nl.add)

            # Residual: add post-attention value
            pa_orig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pa_orig,
                src=post_attn_buf[
                    s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
            )
            x_final = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_final, data1=pa_orig, data2=fc2_biased, op=nl.add)
            nisa.dma_copy(
                dst=x_out[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=x_final,
            )

    return x_out
