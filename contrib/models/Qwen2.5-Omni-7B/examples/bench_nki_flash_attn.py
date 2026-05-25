#!/usr/bin/env python3
"""NKI ``attention_cte`` micro-bench for the Qwen2.5-Omni Token2Wav DiT shape.

Probes whether replacing the current "explicit matmul" attention in
``modeling_qwen25_omni_token2wav.py`` with the ``attention_cte`` NKI
kernel (the one already wired up in NxDI's Flux diffuser and the LTX2
contrib model) would help, and at what numerical cost.

DiT attention shape (batch=2 CFG, heads=16, head_dim=64, mel=1024); zero
mask matches the look_local / look_back / look_ahead masks the DiT
emits today (those tensors are float zeros under our random-tensor bench
setup, so this is the right reference for timing).

This script:

  1. computes a CPU fp32 reference via explicit matmul
  2. traces + times the current explicit matmul path on Neuron (fp32)
  3. traces + times ``attention_cte`` (fp32 and bf16) and reports
     timing + numerical drift vs the CPU reference

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0 python examples/bench_nki_flash_attn.py
"""

import argparse
import math
import os
import statistics
import sys
import time
import traceback

import torch


BATCH = 2          # CFG: cond + uncond
HEADS = 16
HEAD_DIM = 64
SEQ_LEN = 1024


def make_inputs(dtype, device="cpu"):
    torch.manual_seed(0)
    q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    mask = torch.zeros(BATCH, 1, SEQ_LEN, SEQ_LEN, dtype=dtype, device=device)
    return q, k, v, mask


def explicit_matmul_attn(q, k, v, mask):
    """Mirrors src/modeling_qwen25_omni_token2wav.py:_monkeypatch_dit_attention."""
    scale = HEAD_DIM ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn + mask
    attn = torch.nn.functional.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class ExplicitModule(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return explicit_matmul_attn(q, k, v, mask)


class AttentionISAModule(torch.nn.Module):
    """Mirrors src/neuronx_distributed_inference/experimental/functional/attention/
    causal_attention_functions.py:scaled_dot_product_attention_kernel.

    attention_isa_kernel is a private NKI kernel (under
    neuronxcc.nki._private_kernels.attention) used by Mixtral / DBRX /
    Qwen3-MoE and the Trinity contrib model. It is invoked through
    ``nki_jit()(attention_isa_kernel)[grid](q, k, v, scale, out, kernel_name)``.

    Two kernel_name modes:
      * ``"AttentionMMSoftmaxMMWithoutSwap"`` — bi-directional (DiT case)
      * ``"CausalAttentionMMSoftmaxMMWithoutSwap"`` — causal

    Layout expected by the kernel:
      Q: (B*H, head_dim, q_len)   ← Q is pre-scaled by caller
      K: (B*H, head_dim, k_len)
      V: (B*H, k_len, head_dim)
      out: (B*H, head_dim, q_len) buffer the kernel writes into

    Constraints (from _validate_inputs_for_flash_attn_kernel):
      * num_heads <= 128 — we have 16 ✓
      * seq_len >= 512   — we have 1024 ✓
    """

    def __init__(self, causal=False):
        super().__init__()
        try:
            from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
        except ImportError:
            from neuronxcc.nki.kernels.attention import attention_isa_kernel
        from torch_neuronx.xla_impl.ops import nki_jit
        try:
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_platform_lnc,
            )
            from neuronxcc.nki.compiler.backends.neuron.dimensions import nc
            grid = (nc(get_platform_lnc()),)
        except Exception:
            try:
                from neuronxcc.nki.compiler.backends.neuron.dimensions import nc
                grid = (nc(1),)
            except Exception:
                grid = None
        self._kernel = nki_jit()(attention_isa_kernel)
        self._grid = grid
        self._kernel_name = (
            "CausalAttentionMMSoftmaxMMWithoutSwap" if causal
            else "AttentionMMSoftmaxMMWithoutSwap"
        )
        self._scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, q, k, v, mask):
        bsz, n_head, q_len, d_head = q.shape
        # caller-side Q scale, kernel layout: Q,K = (B*H, d, S), V = (B*H, S, d)
        Qf = q.permute(0, 1, 3, 2).reshape(bsz * n_head, d_head, q_len) * self._scale
        Kf = k.permute(0, 1, 3, 2).reshape(bsz * n_head, d_head, q_len)
        Vf = v.reshape(bsz * n_head, q_len, d_head)
        out_buf = torch.zeros(
            bsz * n_head, d_head, q_len, dtype=q.dtype, device=q.device,
        )
        if self._grid is not None:
            self._kernel[self._grid](
                q=Qf, k=Kf, v=Vf, scale=1.0, out=out_buf,
                kernel_name=self._kernel_name,
            )
        else:
            self._kernel(
                q=Qf, k=Kf, v=Vf, scale=1.0, out=out_buf,
                kernel_name=self._kernel_name,
            )
        return out_buf.reshape(bsz, n_head, d_head, q_len).transpose(2, 3)


class FlashFwdModule(torch.nn.Module):
    """Public NKI flash kernel from neuronxcc.nki.kernels.attention.flash_fwd.

    Asserted layouts (from the kernel's own assertion errors):
      Q,K: (B, H, d, S)
      V:   (B, H, S, d)
      out: (B, H, S, d)
    Launch grid: ``flash_fwd[B, H](...)``.

    Hard shape constraints (also asserted by the kernel):
      * head_dim <= 128
      * seqlen_k % 2048 == 0   ← DiT's S=1024 fails this — flash_fwd
                                  is unusable for the Token2Wav DiT
                                  shape on this SDK version. Caller
                                  should raise from __init__ when the
                                  kernel won't accept S.

    DiT is bi-directional so we pass ``use_causal_mask=False``.
    """

    def __init__(self):
        super().__init__()
        if SEQ_LEN % 2048 != 0:
            raise RuntimeError(
                f"flash_fwd requires seqlen_k % 2048 == 0; DiT S={SEQ_LEN} "
                f"is unsupported by this kernel."
            )
        from neuronxcc.nki.kernels.attention import flash_fwd
        self._kernel = flash_fwd
        self._scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, q, k, v, mask):
        # Asserted layouts (from runtime error): Q,K = (B, H, d, S),
        # V = (B, H, S, d). Output = (B, H, S, d).
        b, h, s, d = q.shape
        q_t = q.transpose(-2, -1).contiguous()   # (B, H, d, S)
        k_t = k.transpose(-2, -1).contiguous()   # (B, H, d, S)
        v_t = v.contiguous()                     # (B, H, S, d)
        out = self._kernel[b, h](
            q_t, k_t, v_t,
            seed=0.125,
            softmax_scale=self._scale,
            use_causal_mask=False,
            mixed_precision=False,
            dropout_p=0.0,
        )
        return out


class AttentionCTEModule(torch.nn.Module):
    """Mirrors NxDI Flux's attention_wrapper_sharded_without_swap.

    attention_cte signature (from
    src/neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py):

        attention_cte(q, k, v, scale, causal_mask=False,
                      tp_q=True, tp_k=True, tp_out=False)

    where q/k/v are (B*H, S, d). The kernel emits (B*H, S, d).

    With ``tp_k=True`` the kernel transposes K internally for better DMA.
    """

    def __init__(self, sharded=False):
        super().__init__()
        from nkilib.core.attention.attention_cte import attention_cte
        self._kernel = attention_cte
        self._sharded = sharded
        self._scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, q, k, v, mask):
        # mask is unused for the zero-mask path; attention_cte does not
        # take a generic additive bias. We compare value-equivalent
        # (zero-mask) outputs only — the real DiT can pick this kernel
        # only on the no-mask code path (block.look_backward_block == 0
        # and block.look_ahead_block == 0).
        bs, n_head, q_len, d_head = q.shape
        qf = q.reshape(bs * n_head, q_len, d_head)
        kf = k.reshape(bs * n_head, q_len, d_head)
        vf = v.reshape(bs * n_head, q_len, d_head)

        if self._sharded:
            out = self._kernel[2](
                qf, kf, vf, self._scale,
                causal_mask=False,
                tp_q=True, tp_k=True, tp_out=False,
            )
        else:
            out = self._kernel(
                qf, kf, vf, self._scale,
                causal_mask=False,
                tp_q=True, tp_k=True, tp_out=False,
            )
        return out.reshape(bs, n_head, q_len, d_head)


def probe_nki():
    info = {}
    try:
        import neuronxcc
        info["neuronxcc"] = getattr(neuronxcc, "__version__", "unknown")
    except ImportError as e:
        info["neuronxcc"] = f"ImportError: {e}"
    try:
        import nkilib
        info["nkilib"] = getattr(nkilib, "__version__", "unknown")
    except ImportError as e:
        info["nkilib"] = f"ImportError: {e}"
        return info
    try:
        from nkilib.core.attention import attention_cte as mod
        info["nkilib.core.attention.attention_cte"] = "imported"
        info["available_symbols"] = sorted(
            n for n in dir(mod) if not n.startswith("_")
        )
    except ImportError as e:
        info["nkilib.core.attention.attention_cte"] = f"ImportError: {e}"
    # Also probe neuronxcc.nki.kernels.attention for reference
    try:
        from neuronxcc.nki.kernels import attention as nki_attn
        info["neuronxcc.nki.kernels.attention"] = sorted(
            n for n in dir(nki_attn) if not n.startswith("_")
        )
    except ImportError as e:
        info["neuronxcc.nki.kernels.attention"] = f"ImportError: {e}"
    return info


def trace_and_time(label, mod, inputs, runs=10):
    import torch_neuronx
    print(f"\n[{label}] tracing ...", flush=True)
    t0 = time.time()
    try:
        traced = torch_neuronx.trace(mod, inputs)
    except Exception as e:
        print(f"[{label}] TRACE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        return None
    print(f"[{label}] traced in {time.time() - t0:.1f}s")

    out = traced(*inputs)  # warmup
    walls = []
    for _ in range(runs):
        t1 = time.time()
        out = traced(*inputs)
        walls.append(time.time() - t1)
    return {
        "median_ms": statistics.median(walls) * 1000,
        "min_ms": min(walls) * 1000,
        "max_ms": max(walls) * 1000,
        "out": out,
    }


def compare(ref, out, label):
    o32 = out.to(torch.float32)
    diff = (ref - o32).abs()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), o32.flatten().unsqueeze(0),
    ).item()
    print(
        f"[{label}] max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} cosine={cos:.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--skip-bf16", action="store_true")
    p.add_argument("--sharded", action="store_true",
                   help="Use attention_cte[2] (requires NEURON_RT_VIRTUAL_CORE_SIZE=2)")
    args = p.parse_args()

    print("=" * 60)
    print("NKI attention_cte Micro-Bench (Token2Wav DiT shape)")
    print("=" * 60)
    print(f"  Shape: B={BATCH} H={HEADS} D={HEAD_DIM} T={SEQ_LEN}")
    print(f"  Visible cores: {os.environ.get('NEURON_RT_VISIBLE_CORES', '?')}")
    print(f"  VC size:       {os.environ.get('NEURON_RT_VIRTUAL_CORE_SIZE', '1')}")

    print("\n--- NKI probe ---")
    for k, v in probe_nki().items():
        print(f"  {k}: {v}")

    print("\n--- CPU fp32 reference ---")
    q32, k32, v32, m32 = make_inputs(torch.float32)
    t0 = time.time()
    ref = explicit_matmul_attn(q32, k32, v32, m32)
    print(f"  computed in {(time.time() - t0) * 1000:.0f} ms")

    # ---------------- Neuron: explicit matmul fp32 (baseline) ----------------
    r_exp = trace_and_time(
        "explicit_matmul_fp32",
        ExplicitModule(),
        (q32, k32, v32, m32),
        runs=args.runs,
    )
    if r_exp:
        print(f"  median={r_exp['median_ms']:.2f} ms "
              f"min={r_exp['min_ms']:.2f} max={r_exp['max_ms']:.2f}")
        compare(ref, r_exp["out"], "explicit_matmul_fp32")

    # bf16 inputs must be cast from the fp32 ones, NOT freshly drawn:
    # `torch.randn(...,dtype=bf16)` with the same seed produces DIFFERENT
    # random data than `torch.randn(...,dtype=fp32)`, which makes the
    # bf16 outputs uncorrelated with the fp32 CPU reference (cosine ~0).
    qb, kb, vb, mb = (q32.bfloat16(), k32.bfloat16(),
                      v32.bfloat16(), m32.bfloat16())

    def _run(name, mod_fn, inputs):
        try:
            mod = mod_fn()
        except Exception as e:
            print(f"\n[{name}] kernel not available: {type(e).__name__}: {e}")
            return
        r = trace_and_time(name, mod, inputs, runs=args.runs)
        if r:
            print(f"  median={r['median_ms']:.2f} ms "
                  f"min={r['min_ms']:.2f} max={r['max_ms']:.2f}")
            compare(ref, r["out"], name)
            if r_exp:
                sp = r_exp["median_ms"] / r["median_ms"]
                print(f"  speedup vs explicit_matmul_fp32: {sp:.2f}x")

    # ---------------- attention_cte ----------------
    _run("attention_cte_fp32",
         lambda: AttentionCTEModule(sharded=args.sharded),
         (q32, k32, v32, m32))
    if not args.skip_bf16:
        _run("attention_cte_bf16",
             lambda: AttentionCTEModule(sharded=args.sharded),
             (qb, kb, vb, mb))

    # ---------------- attention_isa_kernel (private) ----------------
    _run("attention_isa_fp32",
         lambda: AttentionISAModule(causal=False),
         (q32, k32, v32, m32))
    if not args.skip_bf16:
        _run("attention_isa_bf16",
             lambda: AttentionISAModule(causal=False),
             (qb, kb, vb, mb))

    # ---------------- flash_fwd (public) ----------------
    _run("flash_fwd_fp32",
         lambda: FlashFwdModule(),
         (q32, k32, v32, m32))
    if not args.skip_bf16:
        _run("flash_fwd_bf16",
             lambda: FlashFwdModule(),
             (qb, kb, vb, mb))


if __name__ == "__main__":
    main()
