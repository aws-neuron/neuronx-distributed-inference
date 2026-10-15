"""Accuracy validation with REAL pretrained DINOv3 weights.

Every accuracy number in this project so far used `pretrained=False`, i.e. random
init. That establishes ARCHITECTURAL equivalence -- the compiled graph computes the
same function as CPU -- but it does NOT validate accuracy on real checkpoints,
because BF16 error behaviour depends on the actual weight and activation
distributions, and pretrained weights are not distributed like randn().

This loads the real LVD-1689M checkpoints from HuggingFace and measures error on
real image statistics, reporting metrics that actually matter for a downstream
embedding consumer rather than just cos_sim on one random tensor:

  cos_sim             per-sample, min/mean/p1 over N inputs (not a single value)
  rel_l2              ||got-ref|| / ||ref||, per sample
  knn_agreement       does the compiled model rank a gallery the same way?
  retrieval_recall@1  nearest-neighbour identity preserved under compilation?

The kNN/retrieval metrics are the point: an embedding model is used for similarity
search, so the question is not "are the floats close" but "does it retrieve the
same neighbours". A model can have cos_sim 0.9999 and still reorder neighbours.

Weights: `facebook/dinov3-*-pretrain-lvd1689m` on HF (gated; needs an approved
token in ~/.cache/huggingface/token). These are transformers-format, so we load
via AutoModel rather than the dinov3 repo's hub factory.

Usage:
    python validate_real_weights.py --models vit_s,vit_b,vit_l --dtypes fp32,bf16
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HF_REPOS = {
    "vit_s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vit_b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vit_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vit_h_plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
}

IMG = 224


REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
USE_REPO_ARCH = os.environ.get("USE_REPO_ARCH", "1") == "1"


def load_real(key):
    """Load the real pretrained checkpoint.

    Default (USE_REPO_ARCH=1): the dinov3-repo architecture with the real HF
    weights remapped into it. This is the architecture the contrib traces, and it
    compiles correctly.

    USE_REPO_ARCH=0 gives the raw transformers implementation, which is what the
    checkpoints ship as but which torch_neuronx.trace() MISCOMPILES on SDK 2.31
    (cos ~0.47-0.62 even at --auto-cast=none, while torch.jit.trace of the same
    module on CPU is exactly 1.0).
    """
    if USE_REPO_ARCH:
        from hf_to_repo_weights import load_real_into_repo
        tok = None
        tp = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tp):
            tok = open(tp).read().strip()
        m, rep = load_real_into_repo(key, REPO_DIR, HF_REPOS[key], token=tok)
        # Gate: if the remap is wrong, everything downstream is meaningless.
        if rep.get("remap_cos", 0) < 0.999:
            raise RuntimeError(f"weight remap failed: {rep}")
        return m
    from transformers import AutoModel
    m = AutoModel.from_pretrained(HF_REPOS[key], dtype=torch.float32)
    m.eval()
    return m


def pooled(out):
    """Reduce a model output to a single embedding vector per sample.

    DINOv3 backbones expose pooler_output (CLS) plus the patch tokens. Use CLS if
    present, else mean-pool the sequence -- matching how an embedding consumer
    would use the model.
    """
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if h.dim() == 3:
        return h[:, 0]           # CLS token
    if h.dim() == 4:
        return h.mean(dim=(2, 3))  # ConvNeXt feature map
    return h


class Wrap(torch.nn.Module):
    """Return a plain tensor -- traced/compiled graphs cannot emit dataclasses."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        if USE_REPO_ARCH:
            return self.m(x)          # repo backbone returns the CLS embedding
        return pooled(self.m(pixel_values=x))


def real_image_batch(n, seed=0):
    """Inputs with realistic image statistics, not white noise.

    White noise is the easy case for a quantized model: it has no spatial
    structure and activations stay small. Real images have low-frequency spatial
    correlation and produce larger activation dynamic range, which is where BF16
    error actually shows up. Build smooth, correlated inputs and apply the
    standard ImageNet normalization the checkpoints expect.
    """
    g = torch.Generator().manual_seed(seed)
    # low-frequency base upsampled to full res + mild high-frequency detail
    lo = torch.rand(n, 3, 14, 14, generator=g)
    x = F.interpolate(lo, size=(IMG, IMG), mode="bicubic", align_corners=False)
    x = x + 0.08 * torch.rand(n, 3, IMG, IMG, generator=g)
    x = x.clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def metrics(ref, got):
    """ref/got: [N, D] float32 CPU embeddings from the SAME inputs."""
    r, g = ref.float(), got.float()
    # Sanity gate. A cos_sim far below ~0.99 on a correctly-wired comparison of the
    # same pretrained weights is not "quantization error" -- it means the harness is
    # broken (wrong reference, mutated model, mismatched inputs). An earlier version
    # reported 0.4756 this way. Flag it loudly rather than publishing it as accuracy.
    _cs = F.cosine_similarity(r, g, dim=1).mean().item()
    _suspect = _cs < 0.99
    cs = F.cosine_similarity(r, g, dim=1)
    rel = (g - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-12)

    # Retrieval: use sample 0..k as queries against the rest as a gallery, and
    # check whether the compiled embeddings pick the same nearest neighbour.
    rn = F.normalize(r, dim=1)
    gn = F.normalize(g, dim=1)
    sim_r = rn @ rn.T
    sim_g = gn @ gn.T
    eye = torch.eye(len(r), dtype=torch.bool)
    sim_r.masked_fill_(eye, -2.0)
    sim_g.masked_fill_(eye, -2.0)
    nn_r = sim_r.argmax(dim=1)
    nn_g = sim_g.argmax(dim=1)
    recall1 = (nn_r == nn_g).float().mean().item()

    # top-5 neighbour set overlap -- a softer, more informative version
    k = min(5, len(r) - 1)
    top_r = sim_r.topk(k, dim=1).indices
    top_g = sim_g.topk(k, dim=1).indices
    overlap = np.mean([len(set(a.tolist()) & set(b.tolist())) / k
                       for a, b in zip(top_r, top_g)])

    return {
        "HARNESS_SUSPECT": bool(_suspect),
        "n_samples": int(len(r)),
        "cos_mean": float(cs.mean()),
        "cos_min": float(cs.min()),
        "cos_p1": float(torch.quantile(cs, 0.01)),
        "rel_l2_mean": float(rel.mean()),
        "rel_l2_max": float(rel.max()),
        "retrieval_recall_at1": float(recall1),
        "knn_top5_overlap": float(overlap),
    }


def run_trace(key, dtype_s, n, bs):
    import torch_neuronx
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_s]
    cargs = (["--auto-cast", "matmult", "--auto-cast-type", "bf16"]
             if dtype_s == "fp32" else ["--auto-cast", "none"])
    if "convnext" not in key:
        cargs += ["--model-type", "transformer"]

    x = real_image_batch(n)
    # Reference FIRST, on its own model instance. Do NOT reuse the same object:
    # `cpu.to(bfloat16)` mutates in place, so tracing a downcast copy of the very
    # model that produced the FP32 reference silently corrupts the comparison.
    ref_model = Wrap(load_real(key))
    with torch.no_grad():
        ref = torch.cat([ref_model(x[i:i + bs]) for i in range(0, n, bs)]).float()
    del ref_model

    # Separate instance for the device path.
    dev_model = Wrap(load_real(key)).to(dtype)
    ex = torch.randn(bs, 3, IMG, IMG).to(dtype)
    mn = torch_neuronx.trace(dev_model, ex, compiler_args=cargs,
                             inline_weights_to_neff=True)
    del dev_model
    got = torch.cat([mn(x[i:i + bs].to(dtype)).float()
                     for i in range(0, n, bs)])
    del mn
    return ref, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_s,vit_b,vit_l")
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--n", type=int, default=64, help="number of eval samples")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--_one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dtype", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._one:
        rec = {"model": args.model, "dtype": args.dtype,
               "hf_repo": HF_REPOS[args.model], "weights": "pretrained_lvd1689m"}
        try:
            ref, got = run_trace(args.model, args.dtype, args.n, args.batch_size)
            rec.update(metrics(ref, got))
            rec["status"] = "ok"
        except Exception as e:
            import traceback
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rec["traceback"] = traceback.format_exc()[-1500:]
        with open(args._result, "w") as f:
            json.dump(rec, f)
        return

    dts = [d.strip() for d in args.dtypes.split(",")]
    out = args.out or "results_real_trace.json"
    recs = []
    for key in [m.strip() for m in args.models.split(",")]:
        for dt in dts:
            rec = {"model": key, "dtype": dt,
                   "hf_repo": HF_REPOS[key], "weights": "pretrained_lvd1689m"}
            print(f"\n=== {key} trace {dt} (real weights, n={args.n}) ===",
                  flush=True)
            try:
                t0 = time.time()
                # Subprocess per config: torch_xla leaves async state behind that
                # poisons subsequent runs in the same process with
                # "Check failed: handle->HasValue() ... async operation in flight".
                import subprocess
                rp = f"/tmp/real_{key}_{dt}.json"
                if os.path.exists(rp):
                    os.unlink(rp)
                subprocess.run(
                    [sys.executable, __file__, "--_one", "--model", key,
                     "--dtype", dt, "--n", str(args.n),
                     "--batch-size", str(args.batch_size), "--_result", rp],
                    env=os.environ.copy())
                if not os.path.exists(rp):
                    raise RuntimeError("subprocess produced no result")
                rec.update(json.load(open(rp)))
                rec["elapsed_s"] = round(time.time() - t0, 1)
                if rec.get("status") != "ok":
                    raise RuntimeError(rec.get("error", "unknown"))
                print(f"  cos mean={rec['cos_mean']:.6f} min={rec['cos_min']:.6f} "
                      f"p1={rec['cos_p1']:.6f}")
                print(f"  rel_l2 mean={rec['rel_l2_mean']:.2e} "
                      f"max={rec['rel_l2_max']:.2e}")
                print(f"  retrieval recall@1={rec['retrieval_recall_at1']:.4f}  "
                      f"knn top5 overlap={rec['knn_top5_overlap']:.4f}")
            except Exception as e:
                import traceback
                rec["status"] = "failed"
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                rec["traceback"] = traceback.format_exc()[-1500:]
                print(f"  FAILED: {rec['error'][:250]}")
            recs.append(rec)
            with open(out, "w") as f:
                json.dump(recs, f, indent=2)

    print(f"\n{'=' * 84}\nREAL PRETRAINED WEIGHTS -- torch_neuronx.trace()\n{'=' * 84}")
    print(f"{'model':10s} {'dt':5s} {'cos_mean':>10s} {'cos_min':>10s} "
          f"{'rel_l2':>9s} {'recall@1':>9s} {'top5':>7s}")
    print("-" * 66)
    for r in recs:
        if r.get("status") != "ok":
            print(f"{r['model']:10s} {r['dtype']:5s} {'FAILED':>10s}")
            continue
        print(f"{r['model']:10s} {r['dtype']:5s} {r['cos_mean']:>10.6f} "
              f"{r['cos_min']:>10.6f} {r['rel_l2_mean']:>9.2e} "
              f"{r['retrieval_recall_at1']:>9.4f} {r['knn_top5_overlap']:>7.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
