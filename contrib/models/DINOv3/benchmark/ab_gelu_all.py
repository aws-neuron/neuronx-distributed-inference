"""A/B the NKI exact-erf GELU kernel across all DINOv3 models that use nn.GELU.

Only ViT-L was validated in the first round (+32% at BS=1, -4.6% at BS=8).
This sweeps every applicable model x batch size so the README can state a
per-model recommendation instead of extrapolating from one model.

Applicable models (ffn_layer='mlp' or ConvNeXt blocks): ViT-S, ViT-B, ViT-L,
ConvNeXt-T, ConvNeXt-B. NOT ViT-H+ / ViT-7B -- those use SwiGLU (SiLU gate).

The GELU swap is generic: it walks the whole module tree and replaces every
nn.GELU instance, so it handles ViT (blocks[].mlp.act) and ConvNeXt (which nests
GELU inside its own block structure) without model-specific code.

Usage:
    python ab_gelu_all.py --models vit_s,vit_b,convnext_tiny,convnext_base \
        --batch-sizes 1,4,8 --out results_gelu_ab.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, trace_dinov3

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def swap_all_gelu(module, NkiGELU):
    """Recursively replace every nn.GELU with NkiGELU. Returns the count.

    Generic on purpose: ViT keeps GELU at blocks[i].mlp.act, ConvNeXt nests it
    inside its own block type. Walking the tree covers both.
    """
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, NkiGELU())
            n += 1
        else:
            n += swap_all_gelu(child, NkiGELU)
    return n


def count_gelu(module):
    return sum(1 for m in module.modules() if isinstance(m, nn.GELU))


def bench(m, bs, img_size, iters, warmup):
    x = torch.randn(bs, 3, img_size, img_size)
    for _ in range(warmup):
        m(x)
    lat = []
    for _ in range(iters):
        t = time.time(); m(x); lat.append((time.time() - t) * 1000)
    lat = np.array(lat)
    return {
        "throughput_img_s": float(bs * 1000 / lat.mean()),
        "median_latency_ms": float(np.median(lat)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
    }


def run(model_key, bs, img_size, iters, warmup, repeats):
    cfg = MODEL_REGISTRY[model_key]
    is_cnx = cfg["arch"] == "convnext"
    rec = {"model": model_key, "batch_size": bs, "arch": cfg["arch"]}

    print(f"\n{'=' * 66}\n{model_key}  BS={bs}\n{'=' * 66}")

    # ---- baseline ----
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    n_gelu = count_gelu(cpu)
    rec["n_gelu_modules"] = n_gelu
    print(f"  nn.GELU modules found: {n_gelu}")
    if n_gelu == 0:
        rec["status"] = "not_applicable"
        print("  SKIP -- no nn.GELU (SwiGLU model)")
        return rec

    base = trace_dinov3(cpu, is_convnext=is_cnx, img_size=img_size,
                        batch_size=bs, inline_weights=True)
    base_runs = [bench(base, bs, img_size, iters, warmup) for _ in range(repeats)]
    rec["baseline_runs"] = [round(r["throughput_img_s"], 1) for r in base_runs]
    base_t = max(r["throughput_img_s"] for r in base_runs)   # best-case baseline
    rec["baseline_best_img_s"] = round(base_t, 1)
    rec["baseline_p50_ms"] = round(
        min(r["median_latency_ms"] for r in base_runs), 2)
    print(f"  baseline: {rec['baseline_runs']} -> best {base_t:.1f} img/s  "
          f"p50={rec['baseline_p50_ms']}ms")
    del base, cpu

    # ---- NKI ----
    from nki_gelu_trace import NkiGELU
    cpu2 = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    n_sw = swap_all_gelu(cpu2, NkiGELU)
    rec["n_swapped"] = n_sw
    print(f"  swapped {n_sw} nn.GELU -> NkiGELU")

    try:
        nkim = trace_dinov3(cpu2, is_convnext=is_cnx, img_size=img_size,
                            batch_size=bs, inline_weights=True)
    except Exception as e:
        rec["status"] = "nki_compile_failed"
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"  NKI COMPILE FAILED: {rec['error'][:200]}")
        return rec

    # accuracy vs the same CPU model (NkiGELU falls back to F.gelu on CPU)
    xv = torch.randn(bs, 3, img_size, img_size)
    with torch.no_grad():
        ref = cpu2(xv)
    got = nkim(xv)
    rec["cos_sim"] = float(F.cosine_similarity(
        ref.flatten().unsqueeze(0).float(), got.flatten().unsqueeze(0).float()).item())
    rec["max_abs_diff"] = float((ref.float() - got.float()).abs().max().item())
    print(f"  cos_sim={rec['cos_sim']:.6f}  max_abs_diff={rec['max_abs_diff']:.2e}")

    nki_runs = [bench(nkim, bs, img_size, iters, warmup) for _ in range(repeats)]
    rec["nki_runs"] = [round(r["throughput_img_s"], 1) for r in nki_runs]
    nki_t = max(r["throughput_img_s"] for r in nki_runs)
    rec["nki_best_img_s"] = round(nki_t, 1)
    rec["nki_p50_ms"] = round(min(r["median_latency_ms"] for r in nki_runs), 2)
    print(f"  NKI:      {rec['nki_runs']} -> best {nki_t:.1f} img/s  "
          f"p50={rec['nki_p50_ms']}ms")

    rec["delta_pct"] = round((nki_t / base_t - 1) * 100, 1)
    rec["verdict"] = "GO" if rec["delta_pct"] > 2 else (
        "NEUTRAL" if rec["delta_pct"] > -2 else "NO-GO")
    rec["status"] = "ok"
    print(f"  RESULT: {rec['delta_pct']:+.1f}%  -> {rec['verdict']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_s,vit_b,convnext_tiny,convnext_base")
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="results_gelu_ab.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bss = [int(b) for b in args.batch_sizes.split(",")]

    out = []
    for mk in models:
        for bs in bss:
            try:
                out.append(run(mk, bs, args.img_size, args.iters,
                               args.warmup, args.repeats))
            except Exception as e:
                out.append({"model": mk, "batch_size": bs, "status": "error",
                            "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(f"  ERROR: {type(e).__name__}: {e}")
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)

    print(f"\n{'=' * 72}\nNKI GELU A/B SUMMARY (inf2.xlarge, 1 core)\n{'=' * 72}")
    print(f"{'model':16s} {'BS':>3s} {'baseline':>9s} {'NKI':>9s} {'delta':>8s} "
          f"{'cos':>9s} {'verdict':>8s}")
    print("-" * 70)
    for r in out:
        if r.get("status") != "ok":
            print(f"{r['model']:16s} {r.get('batch_size', '?'):>3} "
                  f"{r.get('status', 'err'):>9s}")
            continue
        print(f"{r['model']:16s} {r['batch_size']:>3d} "
              f"{r['baseline_best_img_s']:>9.1f} {r['nki_best_img_s']:>9.1f} "
              f"{r['delta_pct']:>+7.1f}% {r['cos_sim']:>9.6f} {r['verdict']:>8s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
