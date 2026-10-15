"""Fused-kernel port feasibility gate on inf2 (torch_neuronx.trace()).

A whole-block fused NKI path measured 1.5-3.4x on Trainium2 (a different compilation path).
Question: can any of it be ported to inf2 under torch_neuronx.trace()?

Three gates, cheapest first. Stop at the first hard failure.

  G1  COMPILE  -- does the ported res_ln kernel compile and run on gen2 at all?
  G2  CORRECT  -- matches F.layer_norm(a+b) to fp32 tolerance, standalone, for every ViT H.
  G3  FAST     -- inside a real DINOv3 block, replacing `x_attn = x + ls1(...)` +
                  `norm2(x_attn)` with one fused kernel, does the block get FASTER?
                  Converged baselines on both arms (best-of-N, last-2 within 3%).

G3 is the one that matters. The single-op NKI GELU kernel passed G1+G2 and then lost
13 of 14 configs at G3, because a standalone kernel adds an HBM round-trip the compiler
had already fused away. The fused-block design is supposed to avoid that by fusing
across the boundary -- but on trace, only ONE boundary (residual+LN2) is in scope here.

Usage (inf2, 1 core):
    python gate_fused_port.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import torch_neuronx


def cos(a, b):
    return F.cosine_similarity(a.flatten().float().unsqueeze(0),
                               b.flatten().float().unsqueeze(0)).item()


def timeit(m, x, iters=60, warmup=60, repeats=6):
    for _ in range(warmup):
        m(*x)
    runs = []
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(iters):
            m(*x)
        runs.append(iters / (time.time() - t0))
    conv = abs(runs[-1] - runs[-2]) / runs[-2] < 0.03
    return runs, max(runs), conv


class ResLN(torch.nn.Module):
    """Reference: x_new = a + b ; z = layer_norm(x_new) (affine-free)."""

    def __init__(self, fused):
        super().__init__()
        self.fused = fused

    def forward(self, a, b):
        if self.fused:
            from nki_res_ln_trace import nki_res_ln
            return nki_res_ln(a, b)
        s = a + b
        return s, F.layer_norm(s, (s.shape[-1],), eps=1e-6)


def g1_g2(out):
    print("=" * 66 + "\nG1 compile + G2 correctness (standalone)\n" + "=" * 66, flush=True)
    for name, H in (("ViT-S", 384), ("ViT-B", 768), ("ViT-L", 1024), ("ViT-H+", 1280)):
        N = 201                      # DINOv3 224px: 196 patches + 1 CLS + 4 storage
        a = torch.randn(1, N, H)
        b = torch.randn(1, N, H)
        ref = ResLN(False)(a, b)
        rec = {"model": name, "H": H}
        try:
            t0 = time.time()
            m = torch_neuronx.trace(ResLN(True), (a, b),
                                    compiler_args=["--auto-cast", "none"])
            rec["compile_s"] = round(time.time() - t0, 1)
            got = m(a, b)
            rec["cos_xnew"] = cos(ref[0], got[0])
            rec["cos_z"] = cos(ref[1], got[1])
            rec["maxabs_z"] = float((ref[1] - got[1]).abs().max())
            rec["G1"] = "PASS"
            rec["G2"] = "PASS" if rec["cos_z"] > 0.9999 else "FAIL"
        except Exception as e:
            rec["G1"] = "FAIL"
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out.append(rec)
        print(f"  {name:7s} H={H:5d}  G1={rec['G1']}  G2={rec.get('G2')}  "
              f"cos_z={rec.get('cos_z')}  maxabs={rec.get('maxabs_z')}  "
              f"{rec.get('error', '')[:120]}", flush=True)


class Block(torch.nn.Module):
    """One DINOv3 transformer block, with the residual+LN2 boundary optionally fused.

    Unfused:  x_attn = x + ls1(attn(norm1(x), rope)); y = x_attn + ls2(mlp(norm2(x_attn)))
    Fused:    x_attn, z = nki_res_ln(x, ls1(attn(...)))   -- one kernel
              y = x_attn + ls2(mlp(norm2_affine(z)))
    norm2's affine (gamma, beta) is applied after the affine-free kernel output so the
    math is identical.
    """

    def __init__(self, blk, rope, fused):
        super().__init__()
        self.blk, self.rope, self.fused = blk, rope, fused

    def forward(self, x):
        b = self.blk
        r = b.ls1(b.attn(b.norm1(x), rope=self.rope))
        if self.fused:
            from nki_res_ln_trace import nki_res_ln
            x_attn, z = nki_res_ln(x, r, eps=b.norm2.eps)
            n2 = z * b.norm2.weight + b.norm2.bias
        else:
            x_attn = x + r
            n2 = b.norm2(x_attn)
        return x_attn + b.ls2(b.mlp(n2))


def g3(out, models=("vit_s", "vit_b", "vit_l")):
    print("=" * 66 + "\nG3 speed -- one real DINOv3 block, residual+LN2 fused vs not\n"
          + "=" * 66, flush=True)
    import dinov3.hub.backbones as B
    fac = {"vit_s": B.dinov3_vits16, "vit_b": B.dinov3_vitb16, "vit_l": B.dinov3_vitl16}
    for key in models:
        torch.manual_seed(0)
        m = fac[key](pretrained=False).eval()
        with torch.no_grad():
            rope = m.rope_embed(H=14, W=14)
        blk = m.blocks[0]
        H = m.cls_token.shape[-1]
        x = torch.randn(1, 201, H)
        rec = {"model": key}
        res = {}
        for arm, fused in (("compiler", False), ("fused_nki", True)):
            mod = Block(blk, rope, fused).eval()
            with torch.no_grad():
                ref = Block(blk, rope, False)(x)
            try:
                tm = torch_neuronx.trace(
                    mod, x, compiler_args=["--auto-cast", "matmult", "--auto-cast-type",
                                           "bf16", "--model-type", "transformer"])
                c = cos(ref, tm(x))
                runs, best, conv = timeit(tm, (x,))
                res[arm] = {"cos": c, "runs": [round(r, 1) for r in runs],
                            "best_it_s": round(best, 1), "converged": conv}
            except Exception as e:
                res[arm] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  {key} {arm:10s} {res[arm]}", flush=True)
        if "best_it_s" in res.get("compiler", {}) and "best_it_s" in res.get("fused_nki", {}):
            rec["speedup"] = round(res["fused_nki"]["best_it_s"] / res["compiler"]["best_it_s"], 3)
        rec.update(res)
        out.append(rec)
        print(f"  -> {key} fused/compiler = {rec.get('speedup')}x", flush=True)


def main():
    out = {"g12": [], "g3": []}
    g1_g2(out["g12"])
    if all(r.get("G2") == "PASS" for r in out["g12"]):
        g3(out["g3"])
    else:
        print("G1/G2 did not pass for all shapes -- stopping before G3", flush=True)
    with open("results_fused_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results_fused_gate.json")


if __name__ == "__main__":
    main()
