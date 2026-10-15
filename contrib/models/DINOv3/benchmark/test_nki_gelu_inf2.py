"""Test the exact-erf GELU NKI kernel on the inf2 TRACE path.

Why this is worth testing even though a prior attempt regressed 24%:
  - GELU is the #1 forward hotspot: 34.2% of forward Vector+Scalar instruction
    time, and the top source line at 56.5 ms cumulative engine time (ViT-L).
  - The inference forward is Vector-bound (86% active) while the Tensor engine
    sits at 57%, which is exactly the profile an activation-table offload helps.
  - neuronx-cc lowers exact erf-GELU as 1664 matmuls + 256 reciprocals + 2816
    vector ops; the kernel does it in 26 activation ops.
  - The prior -24% inference result used a DIFFERENT kernel (tanh approximation)
    through a DIFFERENT dispatch path with F.pad-to-128, so it does not settle
    the exact-erf question on this path.

Three gates, in order. Stop at the first failure.
  Gate 1: does `nl.gelu` exist and produce exact results on gen2 (inf2)?
  Gate 2: does a GELU-only module trace and match CPU FP32?
  Gate 3: does full ViT-L with all MLP activations swapped beat baseline?

Usage:
    python test_nki_gelu_inf2.py --gate 1
    python test_nki_gelu_inf2.py --gate 3 --model vit_l --batch-size 8
"""

import argparse
import os
import sys
import time

# NOTE: do NOT set NEURON_PLATFORM_TARGET_OVERRIDE=gen2 here. That override is for
# standalone nki.simulate / bare-NKI compilation. On the torch_neuronx.trace() path
# it leaks into neuronx-cc as `--target gen2`, which is rejected:
#   "argument --target: invalid choice: gen2 (choose from trn1, inf2, ...)"
# torch_neuronx.trace() already selects the correct target (inf2) from the instance.

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch
import torch.nn.functional as F
import torch_neuronx
from torch import nn

PMAX = 128


def build_kernel():
    """Import the trace-compatible exact-erf GELU kernel.

    See nki_gelu_trace.py for why the older nki_jit convention
    (neuronxcc.nki + output-as-parameter + nl.load/nl.store) is required
    instead of the standalone-nki / shared_hbm-return form.
    """
    from nki_gelu_trace import NkiGELU, nki_gelu
    return nki_gelu


def _NkiGELU():
    from nki_gelu_trace import NkiGELU
    return NkiGELU


def gate1():
    """Does nl.gelu exist on gen2, and is it numerically exact?"""
    print("=== Gate 1: nl.gelu availability + exactness on gen2 (inf2) ===")
    try:
        import nki.language as nl
        print(f"  nki imported; hasattr(nl,'gelu') = {hasattr(nl, 'gelu')}")
        if not hasattr(nl, "gelu"):
            print("  FAIL: nl.gelu missing")
            return False
    except Exception as e:
        print(f"  FAIL importing nki: {type(e).__name__}: {e}")
        return False

    NkiGELU = _NkiGELU()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = NkiGELU()

        def forward(self, x):
            return self.act(x)

    x = torch.randn(256, 1024)
    ref = F.gelu(x)  # exact erf by default

    try:
        # --auto-cast=none so we measure the kernel's own precision, not autocast's
        m = torch_neuronx.trace(M(), x, compiler_args=["--auto-cast", "none"])
    except Exception as e:
        print(f"  FAIL tracing NKI GELU: {type(e).__name__}: {e}")
        return False

    got = m(x)
    cos = F.cosine_similarity(ref.flatten().unsqueeze(0),
                              got.flatten().unsqueeze(0)).item()
    mad = (ref - got).abs().max().item()
    print(f"  cos_sim={cos:.8f}  max_abs_diff={mad:.3e}")
    ok = cos > 0.9999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def gate3(model_key, batch_size, img_size, iters):
    """Full-model A/B: baseline vs all-MLP-GELU swapped."""
    from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, trace_dinov3

    print(f"=== Gate 3: full {model_key} A/B, BS={batch_size} ===")
    cfg = MODEL_REGISTRY[model_key]
    repo = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")

    def bench(m, label):
        x = torch.randn(batch_size, 3, img_size, img_size)
        for _ in range(10):
            m(x)
        lat = []
        for _ in range(iters):
            t = time.time(); m(x); lat.append((time.time() - t) * 1000)
        lat = np.array(lat)
        tput = batch_size * 1000 / lat.mean()
        print(f"  {label:28s} {tput:>8.1f} img/s  p50={np.median(lat):>7.2f}ms")
        return tput

    # Baseline
    base_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=repo)
    t0 = time.time()
    base_n = trace_dinov3(base_cpu, is_convnext=(cfg["arch"] == "convnext"),
                          img_size=img_size, batch_size=batch_size,
                          inline_weights=True)
    print(f"  baseline compile: {time.time() - t0:.1f}s")
    ref_out = None
    with torch.no_grad():
        ref_out = base_cpu(torch.randn(1, 3, img_size, img_size))
    base_t = bench(base_n, "baseline (native GELU)")
    del base_n, base_cpu

    # NKI-swapped
    NkiGELU = _NkiGELU()
    nki_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=repo)
    n_swapped = 0
    for blk in nki_cpu.blocks:
        mlp = blk.mlp
        if hasattr(mlp, "act") and isinstance(mlp.act, nn.GELU):
            mlp.act = NkiGELU()
            n_swapped += 1
    print(f"  swapped {n_swapped} nn.GELU -> NkiGELU")
    if n_swapped == 0:
        print("  SKIP: this model has no nn.GELU MLP (SwiGLU variant?)")
        return

    t0 = time.time()
    try:
        nki_n = trace_dinov3(nki_cpu, is_convnext=(cfg["arch"] == "convnext"),
                             img_size=img_size, batch_size=batch_size,
                             inline_weights=True)
    except Exception as e:
        print(f"  FAIL tracing full NKI model: {type(e).__name__}: {e}")
        return
    print(f"  NKI compile: {time.time() - t0:.1f}s")

    # Accuracy vs the same CPU model
    xv = torch.randn(batch_size, 3, img_size, img_size)
    with torch.no_grad():
        cpu_out = nki_cpu(xv)
    nki_out = nki_n(xv)
    cos = F.cosine_similarity(cpu_out.flatten().unsqueeze(0).float(),
                              nki_out.flatten().unsqueeze(0).float()).item()
    print(f"  accuracy vs CPU FP32: cos_sim={cos:.6f}")

    nki_t = bench(nki_n, "NKI exact-erf GELU")
    delta = (nki_t / base_t - 1) * 100
    print(f"\n  RESULT: {delta:+.1f}%  ({base_t:.1f} -> {nki_t:.1f} img/s)")
    print(f"  {'GO' if delta > 2 else 'NO-GO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--model", default="vit_l")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if args.gate == 1:
        gate1()
    elif args.gate == 3:
        gate3(args.model, args.batch_size, args.img_size, args.iters)


if __name__ == "__main__":
    main()
