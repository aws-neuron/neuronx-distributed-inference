"""
Flash attention for d=256 with causal masking.

NKI kernel for SDK 2.28+. Uses neuronxcc.nki.jit (compat API, works within
torch_neuronx.trace). Supports head_dim=256 via 2x128 QK contraction tiling.
Uses NKI affine_select for causal masking, activation_reduce for fused exp+sum.

Improvements over original (Task 17):
  1. V layout [kv, d] -- already in place from original kernel
  2. activation_reduce for fused exp+sum -- already in place
  3. 1D grid (bs * kv_heads) -- required for nki.jit, correct for GQA
  4. Post-matmul scaling -- avoids bf16 tensor_scalar, scale applied in f32
  5. dma_transpose for P: NOT applied -- deferred to Task 20 (pipelined kernel)

Run on trn2:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python3 nki_flash_attn_d256.py  # unit test
"""

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import math
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

B_P = 128  # partition dim max
B_F = 512  # free dim max for matmul moving operand
D_TILE = 128  # head_dim tile size


@nki.jit
def flash_attn_d256(q, k, v, use_causal_mask=True):
    """
    Flash attention for head_dim=256.

    Args:
        q: (bs, n_heads, 256, seq_q) -- bfloat16
        k: (bs, nk_heads, 256, seq_k) -- bfloat16
        v: (bs, nv_heads, seq_v, 256) -- bfloat16
        use_causal_mask: bool

    Returns:
        o: (bs, n_heads, seq_q, 256) -- bfloat16

    Grid: kernel[bs * nk_heads](...) -- 1D grid over batch*kv_heads.
    Each program handles all Q heads sharing one KV head (GQA).

    The QK matmul is tiled: QK = Q0^T @ K0 + Q1^T @ K1
    where Q0/Q1 are the first/second 128 dims of Q along head_dim.
    """
    b, h, d, seqlen_q = q.shape
    _, k_h, _, seqlen_k = k.shape
    assert d == 256
    assert seqlen_k % B_F == 0

    q_h_per_k_h = h // k_h

    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # 1D grid: flatten batch and KV-head dims
    pid = nl.program_id(axis=0)
    batch_id = pid // k_h
    head_id = pid % k_h  # KV head index

    scale = 1.0 / (d**0.5)
    n_q_tiles = seqlen_q // B_P
    n_kv_tiles = seqlen_k // B_F
    NEG_INF = -9984.0

    for i_q_h in nl.affine_range(q_h_per_k_h):
        q_head_idx = head_id * q_h_per_k_h + i_q_h

        for qi in nl.sequential_range(n_q_tiles):
            # Accumulators
            o_acc = nl.zeros((par_dim(B_P), d), dtype=np.float32, buffer=nl.sbuf)
            m_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)
            l_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)

            # Load Q tile: 2 chunks of (128, 128)
            q0 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16)
            q0[:, :] = nl.load(
                q[batch_id, q_head_idx, nl.ds(0, D_TILE), nl.ds(qi * B_P, B_P)]
            )
            q1 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16)
            q1[:, :] = nl.load(
                q[batch_id, q_head_idx, nl.ds(D_TILE, D_TILE), nl.ds(qi * B_P, B_P)]
            )

            for kvi in nl.sequential_range(n_kv_tiles):
                # Causal: skip if Q tile is entirely before K tile
                if use_causal_mask:
                    skip_condition = qi * B_P < kvi * B_F
                else:
                    skip_condition = False

                if not skip_condition:
                    # Load K: 2 chunks of (par_dim(128), 512)
                    k0 = nl.ndarray((par_dim(D_TILE), B_F), dtype=nl.bfloat16)
                    k0[:, :] = nl.load(
                        k[batch_id, head_id, nl.ds(0, D_TILE), nl.ds(kvi * B_F, B_F)]
                    )
                    k1 = nl.ndarray((par_dim(D_TILE), B_F), dtype=nl.bfloat16)
                    k1[:, :] = nl.load(
                        k[
                            batch_id,
                            head_id,
                            nl.ds(D_TILE, D_TILE),
                            nl.ds(kvi * B_F, B_F),
                        ]
                    )

                    # Tiled QK matmul: Q0^T @ K0 + Q1^T @ K1 -> (B_P, B_F) in PSUM
                    qk = nl.ndarray(
                        (par_dim(B_P), B_F), dtype=np.float32, buffer=nl.psum
                    )
                    qk[:, :] = nl.matmul(q0, k0, transpose_x=True)
                    qk[:, :] += nl.matmul(q1, k1, transpose_x=True)

                    # Move to SBUF for masking and softmax
                    qk_sbuf = nl.ndarray(
                        (par_dim(B_P), B_F), dtype=np.float32, buffer=nl.sbuf
                    )

                    # Apply causal mask (transfers PSUM -> SBUF)
                    if use_causal_mask:
                        i_q, i_k = nl.mgrid[0:B_P, 0:B_F]
                        q_pos = qi * B_P + i_q
                        k_pos = kvi * B_F + i_k
                        pred_causal = q_pos >= k_pos

                        qk_sbuf[:, :] = nisa.affine_select(
                            pred=pred_causal,
                            on_true_tile=qk,
                            on_false_value=NEG_INF,
                            dtype=np.float32,
                        )
                    else:
                        qk_sbuf[:, :] = nl.copy(qk, dtype=np.float32)

                    # Apply scale post-matmul (in f32 SBUF, avoids bf16 precision loss)
                    qk_sbuf[...] = nisa.tensor_scalar(qk_sbuf, nl.multiply, scale)

                    # Row max
                    new_max = nisa.tensor_reduce(
                        np.max, qk_sbuf, axis=(1,), dtype=np.float32, negate=False
                    )

                    m_prev = nl.copy(m_acc[:, 0])
                    m_acc[:, 0] = nl.maximum(m_prev, new_max)
                    m_cur = m_acc[:, 0]

                    # Rescale previous output: o_acc *= exp(m_prev - m_cur)
                    alpha = nisa.activation(np.exp, m_cur, bias=m_prev, scale=-1.0)
                    o_acc[...] = nl.multiply(o_acc, alpha)

                    # exp(qk - max) and row sum via activation_reduce
                    p = nl.ndarray((par_dim(B_P), B_F), dtype=nl.bfloat16)
                    p_sum = nl.ndarray((par_dim(B_P), 1), dtype=np.float32)
                    p[:, :] = nisa.activation_reduce(
                        np.exp,
                        qk_sbuf,
                        bias=-1 * m_cur,
                        scale=1.0,
                        reduce_op=nl.add,
                        reduce_res=p_sum[:, 0],
                        dtype=nl.bfloat16,
                    )

                    # Load V: (B_F//B_P, par_dim(B_P), 256)
                    n_v_sub = B_F // B_P
                    v_tile = nl.ndarray((n_v_sub, par_dim(B_P), d), dtype=nl.bfloat16)
                    for vi in nl.affine_range(n_v_sub):
                        v_tile[vi, :, :] = nl.load(
                            v[batch_id, head_id, nl.ds(kvi * B_F + vi * B_P, B_P), :],
                            dtype=nl.bfloat16,
                        )

                    # Transpose P for PV matmul: P[B_P, B_F] in 4 chunks of 128x128
                    p_t = nl.ndarray((par_dim(B_P), B_F), dtype=nl.bfloat16)
                    for ti in nl.affine_range(B_F // B_P):
                        p_t_tmp = nl.ndarray(
                            (par_dim(B_P), B_P), dtype=np.float32, buffer=nl.psum
                        )
                        p_t_tmp[:, :] = nisa.nc_transpose(p[:, nl.ds(ti * B_P, B_P)])
                        p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                            p_t_tmp, dtype=nl.bfloat16
                        )

                    # PV matmul: (B_P, B_F) @ (B_F, 256) -> (B_P, 256) in PSUM
                    pv = nl.zeros(
                        (par_dim(B_P), d),
                        dtype=np.float32,
                        buffer=nl.psum,
                        lazy_initialization=True,
                    )
                    for vi in nl.affine_range(n_v_sub):
                        pv[:, :] += nl.matmul(
                            p_t[:, nl.ds(vi * B_P, B_P)],
                            v_tile[vi, :, :],
                            transpose_x=True,
                        )

                    o_acc[:, :] = nl.add(o_acc, pv)

                    # Update log-sum-exp: l = m + log(exp(l - m) + p_sum)
                    exp_l = nisa.activation(nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0)
                    l_acc[:, 0] = nl.add(
                        m_cur, nisa.activation(nl.log, exp_l, bias=p_sum[:, 0])
                    )

            # Final rescale and store
            final_exp = nisa.activation(
                np.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
            )
            out = nl.multiply(o_acc, final_exp, dtype=nl.bfloat16)
            nl.store(
                o[batch_id, q_head_idx, nl.ds(qi * B_P, B_P), :],
                out,
            )

    return o


# ---- Unit test ----
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import time

    def reference_causal_attention(q, k, v):
        """CPU reference: q(b,h,d,sq), k(b,h,d,sk), v(b,h,sk,d) -> (b,h,sq,d)"""
        d = q.shape[2]
        q_t = q.permute(0, 1, 3, 2).float()
        k_t = k.permute(0, 1, 3, 2).float()
        v_t = v.float()
        scale = 1.0 / (d**0.5)
        attn = q_t @ k_t.transpose(-2, -1) * scale
        mask = torch.triu(
            torch.ones(q_t.shape[2], k_t.shape[2], dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return attn @ v_t

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    tests = [
        {"seq": 512, "bs": 1, "heads": 1, "kv_heads": 1, "label": "d256 seq=512 1:1"},
        {"seq": 1024, "bs": 1, "heads": 1, "kv_heads": 1, "label": "d256 seq=1024 1:1"},
        {
            "seq": 512,
            "bs": 1,
            "heads": 4,
            "kv_heads": 1,
            "label": "d256 seq=512 GQA 4:1",
        },
    ]

    for t in tests:
        seq_len = t["seq"]
        bs = t["bs"]
        heads = t["heads"]
        kv_heads = t["kv_heads"]
        d = 256
        print(f"\n=== Testing: {t['label']} ===")
        torch.manual_seed(42)
        q = torch.randn(bs, heads, d, seq_len, dtype=torch.bfloat16)
        k = torch.randn(bs, kv_heads, d, seq_len, dtype=torch.bfloat16)
        v = torch.randn(bs, kv_heads, seq_len, d, dtype=torch.bfloat16)

        # CPU reference uses per-head attention
        ref_parts = []
        for h_idx in range(heads):
            kv_idx = h_idx // (heads // kv_heads)
            ref_h = reference_causal_attention(
                q[:, h_idx : h_idx + 1],
                k[:, kv_idx : kv_idx + 1],
                v[:, kv_idx : kv_idx + 1],
            )
            ref_parts.append(ref_h)
        ref = torch.cat(ref_parts, dim=1)

        q_dev = q.to(device)
        k_dev = k.to(device)
        v_dev = v.to(device)
        t0 = time.time()
        out = flash_attn_d256[bs * kv_heads](q_dev, k_dev, v_dev, use_causal_mask=True)
        xm.mark_step()
        out_cpu = out.cpu().float()
        t1 = time.time()

        cos = F.cosine_similarity(
            ref.reshape(-1).unsqueeze(0), out_cpu.reshape(-1).unsqueeze(0)
        ).item()
        maxd = (ref - out_cpu).abs().max().item()
        print(f"  Time: {t1 - t0:.1f}s (includes compile)")
        print(f"  Cosine sim: {cos:.6f}")
        print(f"  Max diff: {maxd:.6f}")
        print(f"  {'PASS' if cos > 0.999 else 'FAIL'}")
