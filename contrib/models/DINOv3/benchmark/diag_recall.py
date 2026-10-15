"""Is retrieval recall@1 of 0.90-0.95 a real accuracy problem?

Both paths measured cos_sim ~0.9998 on real weights but retrieval recall@1 of only
0.875-0.953. Two very different explanations:

  (a) REAL: compilation reorders nearest neighbours, which would matter a lot for
      an embedding model used in similarity search.
  (b) ARTIFACT: my synthetic "real-ish" images are mutually near-identical, so the
      gallery has many near-ties. When the 1st and 2nd neighbour are separated by
      less than the numerical error, recall@1 flips on noise and measures nothing.

The discriminator is the MARGIN between the top-1 and top-2 similarity in the
reference embeddings. If that margin is tiny relative to the perturbation, (b).

Also re-tests recall with genuinely distinct inputs to see whether recall recovers,
and reports how far down the ranking the "wrong" neighbour actually is -- a swap
between rank 1 and 2 is meaningless, a swap to rank 30 is not.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_real_weights import IMG, HF_REPOS, Wrap, load_real, real_image_batch


def distinct_batch(n, seed=0):
    """Inputs that are genuinely different from each other.

    real_image_batch() builds every sample from the same low-frequency recipe, so
    samples end up highly correlated. Here each sample gets its own structure:
    different spatial frequency, orientation, phase and colour balance, so the
    gallery has real separation and recall@1 becomes a meaningful test.
    """
    g = torch.Generator().manual_seed(seed)
    xs = []
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, IMG),
                            torch.linspace(-1, 1, IMG), indexing="ij")
    for i in range(n):
        f = 1.0 + 6.0 * torch.rand(1, generator=g).item()
        th = 3.14159 * torch.rand(1, generator=g).item()
        ph = 6.28318 * torch.rand(1, generator=g).item()
        base = torch.sin(f * (xx * np.cos(th) + yy * np.sin(th)) * 3.14159 + ph)
        blob = torch.exp(-((xx - (torch.rand(1, generator=g).item() * 2 - 1))**2 +
                           (yy - (torch.rand(1, generator=g).item() * 2 - 1))**2) * 3)
        col = torch.rand(3, 1, 1, generator=g) * 0.8 + 0.2
        img = (0.5 + 0.35 * base + 0.4 * blob).clamp(0, 1) * col
        img = img + 0.03 * torch.rand(3, IMG, IMG, generator=g)
        xs.append(img.clamp(0, 1))
    x = torch.stack(xs)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def analyze(ref, got, label):
    r = F.normalize(ref.float(), dim=1)
    g = F.normalize(got.float(), dim=1)
    n = len(r)
    sr, sg = r @ r.T, g @ g.T
    eye = torch.eye(n, dtype=torch.bool)
    sr.masked_fill_(eye, -2.0)
    sg.masked_fill_(eye, -2.0)

    top2 = sr.topk(2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1])
    nn_r, nn_g = sr.argmax(1), sg.argmax(1)
    agree = (nn_r == nn_g)

    # Where does the reference's true top-1 land in the compiled ranking?
    rank_of_true = []
    order_g = sg.argsort(dim=1, descending=True)
    for i in range(n):
        rank_of_true.append((order_g[i] == nn_r[i]).nonzero()[0, 0].item() + 1)
    rank_of_true = np.array(rank_of_true)

    # Pairwise-similarity fidelity: does compilation preserve the geometry?
    off = ~eye
    sim_corr = np.corrcoef(sr[off].numpy(), sg[off].numpy())[0, 1]

    out = {
        "label": label,
        "n": n,
        "cos_mean": float(F.cosine_similarity(ref.float(), got.float(), dim=1).mean()),
        "recall_at1": float(agree.float().mean()),
        "top1_top2_margin_mean": float(margin.mean()),
        "top1_top2_margin_median": float(margin.median()),
        "margin_when_disagree_mean": float(margin[~agree].mean()) if (~agree).any() else None,
        "margin_when_agree_mean": float(margin[agree].mean()) if agree.any() else None,
        "true_top1_rank_max": int(rank_of_true.max()),
        "true_top1_rank_mean": float(rank_of_true.mean()),
        "pct_true_top1_within_top3": float((rank_of_true <= 3).mean()),
        "pairwise_sim_correlation": float(sim_corr),
    }
    print(f"\n--- {label} ---")
    print(f"  cos_mean                  {out['cos_mean']:.6f}")
    print(f"  recall@1                  {out['recall_at1']:.4f}")
    print(f"  top1-top2 margin (median) {out['top1_top2_margin_median']:.6f}")
    print(f"  margin when DISAGREE      {out['margin_when_disagree_mean']}")
    print(f"  margin when AGREE         {out['margin_when_agree_mean']}")
    print(f"  true top1 rank: mean {out['true_top1_rank_mean']:.2f}, "
          f"worst {out['true_top1_rank_max']}, within top3 "
          f"{out['pct_true_top1_within_top3'] * 100:.1f}%")
    print(f"  pairwise sim correlation  {out['pairwise_sim_correlation']:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_l")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results_recall_diag.json")
    args = ap.parse_args()

    import torch_neuronx
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cargs = (["--auto-cast", "matmult", "--auto-cast-type", "bf16"]
             if args.dtype == "fp32" else ["--auto-cast", "none"])
    cargs += ["--model-type", "transformer"]

    recs = []
    for label, maker in (("correlated (original harness)", real_image_batch),
                         ("distinct images", distinct_batch)):
        x = maker(args.n)
        rm = Wrap(load_real(args.model))
        with torch.no_grad():
            ref = torch.cat([rm(x[i:i + args.batch_size])
                             for i in range(0, args.n, args.batch_size)]).float()
        del rm
        dm = Wrap(load_real(args.model)).to(dtype)
        mn = torch_neuronx.trace(dm, torch.randn(args.batch_size, 3, IMG, IMG).to(dtype),
                                 compiler_args=cargs, inline_weights_to_neff=True)
        del dm
        got = torch.cat([mn(x[i:i + args.batch_size].to(dtype)).float()
                         for i in range(0, args.n, args.batch_size)])
        del mn
        recs.append(analyze(ref, got, f"{args.model} {args.dtype} / {label}"))

    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
