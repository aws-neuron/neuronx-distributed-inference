"""CCA NKI kernel v3: fuses mean-residual add + L2 norm + sqrt(d) scale + temperature.

v3 improvements over v2:
- nl.static_range replaces deprecated nl.affine_range
- Accepts [S, B, heads*dim] layout — eliminates 2 input permutes on caller side
- Same separate Q/K processing (combined tile fails due to partition slice limitation)

Layout: (num_heads, head_dim) — partition=heads, free=128
NKI 0.3.0 API: All ISA functions take dst as first arg.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import math


@nki.jit
def cca_postconv_fused(
    q_conv_ref,  # [S, B, num_q_heads * head_dim]
    k_conv_ref,  # [S, B, num_kv_heads * head_dim]
    q_mean_ref,  # [S, B, num_q_heads * head_dim]
    k_mean_ref,  # [S, B, num_kv_heads * head_dim]
    temp_ref,  # [num_kv_heads]
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Fused mean-residual + L2 norm + sqrt(d) scale + temperature for CCA Q/K.

    Accepts [S, B, ...] layout (no permute needed on caller side).
    Returns q_out, k_out in [S, B, heads*dim].
    """
    S = q_conv_ref.shape[0]
    B = q_conv_ref.shape[1]
    Q_dim = num_q_heads * head_dim
    K_dim = num_kv_heads * head_dim

    sqrt_d = math.sqrt(float(head_dim))

    q_out = nl.ndarray((S, B, Q_dim), dtype=q_conv_ref.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((S, B, K_dim), dtype=k_conv_ref.dtype, buffer=nl.shared_hbm)

    # Load temperature once: [num_kv_heads] -> (num_kv_heads, 1) f32 in SBUF
    temp_sb = nl.ndarray((num_kv_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for kh in nl.static_range(num_kv_heads):
        t_raw = nl.ndarray((1, 1), dtype=temp_ref.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_raw[0:1, 0:1], src=temp_ref[kh : kh + 1])
        t_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_f32[0:1, 0:1], src=t_raw[0:1, 0:1])
        nisa.tensor_copy(dst=temp_sb[kh : kh + 1, 0:1], src=t_f32[0:1, 0:1])

    for s in nl.static_range(S):
        for b in nl.static_range(B):
            # ---- Process Q heads ----
            q_conv_sb = nl.ndarray(
                (num_q_heads, head_dim), dtype=q_conv_ref.dtype, buffer=nl.sbuf
            )
            q_mean_sb = nl.ndarray(
                (num_q_heads, head_dim), dtype=q_mean_ref.dtype, buffer=nl.sbuf
            )

            for qh in nl.static_range(num_q_heads):
                nisa.dma_copy(
                    dst=q_conv_sb[qh : qh + 1, 0:head_dim],
                    src=q_conv_ref[s, b, qh * head_dim : (qh + 1) * head_dim],
                )
                nisa.dma_copy(
                    dst=q_mean_sb[qh : qh + 1, 0:head_dim],
                    src=q_mean_ref[s, b, qh * head_dim : (qh + 1) * head_dim],
                )

            # Cast to f32
            q_conv_f32 = nl.ndarray(
                (num_q_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            q_mean_f32 = nl.ndarray(
                (num_q_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=q_conv_f32, src=q_conv_sb)
            nisa.tensor_copy(dst=q_mean_f32, src=q_mean_sb)

            # Mean residual add
            q_sum = nl.ndarray(
                (num_q_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(q_sum, q_conv_f32, q_mean_f32, nl.add)

            # L2 norm
            q_sq = nl.ndarray((num_q_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(q_sq, q_sum, q_sum, nl.multiply)

            q_norm_sq = nl.ndarray((num_q_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(q_norm_sq, nl.add, q_sq, axis=(1,))

            q_rsqrt = nl.ndarray((num_q_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(q_rsqrt, nl.rsqrt, q_norm_sq, bias=0.0)

            # Fused scale: q_out = q_sum * rsqrt * sqrt_d
            q_scaled = nl.ndarray(
                (num_q_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                q_scaled,
                q_sum,
                nl.multiply,
                q_rsqrt,
                op1=nl.multiply,
                operand1=sqrt_d,
            )

            # Cast back and write
            q_out_bf16 = nl.ndarray(
                (num_q_heads, head_dim), dtype=q_conv_ref.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=q_out_bf16, src=q_scaled)

            for qh in nl.static_range(num_q_heads):
                nisa.dma_copy(
                    dst=q_out[s, b, qh * head_dim : (qh + 1) * head_dim],
                    src=q_out_bf16[qh : qh + 1, 0:head_dim],
                )

            # ---- Process K heads ----
            k_conv_sb = nl.ndarray(
                (num_kv_heads, head_dim), dtype=k_conv_ref.dtype, buffer=nl.sbuf
            )
            k_mean_sb = nl.ndarray(
                (num_kv_heads, head_dim), dtype=k_mean_ref.dtype, buffer=nl.sbuf
            )

            for kh in nl.static_range(num_kv_heads):
                nisa.dma_copy(
                    dst=k_conv_sb[kh : kh + 1, 0:head_dim],
                    src=k_conv_ref[s, b, kh * head_dim : (kh + 1) * head_dim],
                )
                nisa.dma_copy(
                    dst=k_mean_sb[kh : kh + 1, 0:head_dim],
                    src=k_mean_ref[s, b, kh * head_dim : (kh + 1) * head_dim],
                )

            k_conv_f32 = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            k_mean_f32 = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=k_conv_f32, src=k_conv_sb)
            nisa.tensor_copy(dst=k_mean_f32, src=k_mean_sb)

            k_sum = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(k_sum, k_conv_f32, k_mean_f32, nl.add)

            k_sq = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(k_sq, k_sum, k_sum, nl.multiply)

            k_norm_sq = nl.ndarray((num_kv_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(k_norm_sq, nl.add, k_sq, axis=(1,))

            k_rsqrt = nl.ndarray((num_kv_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(k_rsqrt, nl.rsqrt, k_norm_sq, bias=0.0)

            k_scaled_no_temp = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                k_scaled_no_temp,
                k_sum,
                nl.multiply,
                k_rsqrt,
                op1=nl.multiply,
                operand1=sqrt_d,
            )

            # Temperature
            k_scaled = nl.ndarray(
                (num_kv_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                k_scaled,
                k_scaled_no_temp,
                nl.multiply,
                temp_sb[0:num_kv_heads, 0:1],
            )

            k_out_bf16 = nl.ndarray(
                (num_kv_heads, head_dim), dtype=k_conv_ref.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=k_out_bf16, src=k_scaled)

            for kh in nl.static_range(num_kv_heads):
                nisa.dma_copy(
                    dst=k_out[s, b, kh * head_dim : (kh + 1) * head_dim],
                    src=k_out_bf16[kh : kh + 1, 0:head_dim],
                )

    return q_out, k_out
