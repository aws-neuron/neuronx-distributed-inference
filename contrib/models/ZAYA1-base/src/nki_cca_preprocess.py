"""CCA preprocessing NKI kernel for ZAYA attention.

Fuses L2 normalization + sqrt(d) scaling + temperature scaling for Q and K.

NKI 0.3.0 API: All ISA functions take dst as first arg.
  nisa.tensor_tensor(dst, data1, data2, op) — all operands must match on partition dim
  nisa.tensor_scalar(dst, data, op0, operand0) — operand0 is scalar or (par_dim, 1) vector
  nisa.activation(dst, op, data, bias=0.0)
  nisa.tensor_copy(dst, src)
  nisa.tensor_reduce(dst, op, data, axis)

Layout: head data on FREE axis (dim 1) as (1, head_dim) so tensor_reduce works.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import math


@nki.jit
def cca_l2_norm_and_scale(
    q_ref: nl.ndarray,  # [B, S, Q_dim]
    k_ref: nl.ndarray,  # [B, S, K_dim]
    temp_ref: nl.ndarray,  # [num_kv_heads]
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple:
    """Fused L2 normalization + temperature scaling for CCA Q/K."""
    B = q_ref.shape[0]
    S = q_ref.shape[1]
    Q_dim = q_ref.shape[2]
    K_dim = k_ref.shape[2]

    sqrt_d = math.sqrt(float(head_dim))

    q_out = nl.ndarray((B, S, Q_dim), dtype=q_ref.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((B, S, K_dim), dtype=k_ref.dtype, buffer=nl.shared_hbm)

    # Load temperature to SBUF — shape (num_kv_heads, 1) for use as tensor_scalar operand
    temp_sb = nl.ndarray((num_kv_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for kv_h in nl.affine_range(num_kv_heads):
        temp_raw = nl.ndarray((1, 1), dtype=temp_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=temp_raw[0:1, 0:1], src=temp_ref[kv_h : kv_h + 1])
        temp_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=temp_f32[0:1, 0:1], src=temp_raw[0:1, 0:1])
        nisa.tensor_copy(dst=temp_sb[kv_h : kv_h + 1, 0:1], src=temp_f32[0:1, 0:1])

    for b in nl.affine_range(B):
        for s in nl.affine_range(S):
            # --- Q heads ---
            for qh in nl.affine_range(num_q_heads):
                offset = b * S * Q_dim + s * Q_dim + qh * head_dim

                # Load head data as (1, head_dim) — head_dim on FREE axis
                head_sb = nl.ndarray((1, head_dim), dtype=q_ref.dtype, buffer=nl.sbuf)
                q_flat = q_ref.reshape((B * S * Q_dim,))
                nisa.dma_copy(
                    dst=head_sb[0:1, 0:head_dim],
                    src=q_flat.ap(pattern=[[1, 1], [1, head_dim]], offset=offset),
                )

                # bf16 -> f32
                head_f32 = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=head_f32[0:1, 0:head_dim], src=head_sb[0:1, 0:head_dim]
                )

                # x^2
                sq = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    sq[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    nl.multiply,
                )

                # sum(x^2) via tensor_reduce on free axis
                norm_sq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    norm_sq[0:1, 0:1], nl.add, sq[0:1, 0:head_dim], axis=(1,)
                )

                # rsqrt(sum_sq) = 1/||x||
                rsqrt_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    rsqrt_val[0:1, 0:1], nl.rsqrt, norm_sq[0:1, 0:1], bias=0.0
                )

                # out = head_f32 * rsqrt_val * sqrt_d
                # tensor_scalar: (1, head_dim) * (1,1) operand * scalar
                # Use two-op form: (data * operand0) * operand1
                scaled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    scaled[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    nl.multiply,
                    rsqrt_val[0:1, 0:1],
                    op1=nl.multiply,
                    operand1=sqrt_d,
                )

                # f32 -> bf16
                out_bf16 = nl.ndarray((1, head_dim), dtype=q_ref.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=out_bf16[0:1, 0:head_dim], src=scaled[0:1, 0:head_dim]
                )

                # Write
                q_flat_out = q_out.reshape((B * S * Q_dim,))
                nisa.dma_copy(
                    dst=q_flat_out.ap(pattern=[[1, 1], [1, head_dim]], offset=offset),
                    src=out_bf16[0:1, 0:head_dim],
                )

            # --- K heads ---
            for kh in nl.affine_range(num_kv_heads):
                offset = b * S * K_dim + s * K_dim + kh * head_dim

                head_sb = nl.ndarray((1, head_dim), dtype=k_ref.dtype, buffer=nl.sbuf)
                k_flat = k_ref.reshape((B * S * K_dim,))
                nisa.dma_copy(
                    dst=head_sb[0:1, 0:head_dim],
                    src=k_flat.ap(pattern=[[1, 1], [1, head_dim]], offset=offset),
                )

                head_f32 = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=head_f32[0:1, 0:head_dim], src=head_sb[0:1, 0:head_dim]
                )

                sq = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    sq[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    nl.multiply,
                )

                norm_sq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    norm_sq[0:1, 0:1], nl.add, sq[0:1, 0:head_dim], axis=(1,)
                )

                rsqrt_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    rsqrt_val[0:1, 0:1], nl.rsqrt, norm_sq[0:1, 0:1], bias=0.0
                )

                # scale = head_f32 * rsqrt * sqrt_d * temp[kh]
                # Step 1: data * rsqrt * sqrt_d (two-op tensor_scalar)
                scaled_no_temp = nl.ndarray(
                    (1, head_dim), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    scaled_no_temp[0:1, 0:head_dim],
                    head_f32[0:1, 0:head_dim],
                    nl.multiply,
                    rsqrt_val[0:1, 0:1],
                    op1=nl.multiply,
                    operand1=sqrt_d,
                )

                # Step 2: multiply by temperature
                # temp_sb[kh] is (1,1) — need to use tensor_scalar with (1,1) operand
                scaled = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    scaled[0:1, 0:head_dim],
                    scaled_no_temp[0:1, 0:head_dim],
                    nl.multiply,
                    temp_sb[kh : kh + 1, 0:1],
                )

                out_bf16 = nl.ndarray((1, head_dim), dtype=k_ref.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=out_bf16[0:1, 0:head_dim], src=scaled[0:1, 0:head_dim]
                )

                k_flat_out = k_out.reshape((B * S * K_dim,))
                nisa.dma_copy(
                    dst=k_flat_out.ap(pattern=[[1, 1], [1, head_dim]], offset=offset),
                    src=out_bf16[0:1, 0:head_dim],
                )

    return q_out, k_out
