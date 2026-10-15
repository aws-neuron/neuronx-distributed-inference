"""Test the `torch_neuronx.DataParallel` num_workers defect on inf2.

Background: `DataParallel` sets `num_workers = 2 * len(loaded_modules)`, and when
device IDs are consecutive (the default) `_load_modules()` batch-loads into a
SINGLE module object -- so num_workers becomes 2 regardless of core count, and
`parallel_apply()`'s ThreadPoolExecutor serializes all but 2 core dispatches.

This was documented in OpencodeDocs/steering/neuron-sdk.md (discovered on trn2,
BERT reranker: 2 workers = 167 qps / 2.0x, 8 workers = 634 qps / 7.6x) but was
NOT applied to the DINOv3 benchmarks. The DINOv3 inf2.24xlarge run measured
DataParallel at exactly 2.01x on 12 cores, which matches the defect signature
rather than host-dispatch saturation.

This script measures throughput vs num_workers to (a) confirm the defect on inf2
and (b) determine whether fixed-DataParallel closes the gap to multi-process.

Usage:
    python bench_dp_numworkers.py --model vit_l --cores 12 --workers 2,4,8,12,16
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
import torch_neuronx
from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, trace_dinov3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_l")
    ap.add_argument("--cores", type=int, default=12)
    ap.add_argument("--workers", default="2,4,8,12,16")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="total batch fed to DataParallel; default = cores")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bs = args.batch_size or args.cores
    out = args.out or f"results_dpworkers_{args.model}.json"
    cfg = MODEL_REGISTRY[args.model]

    print(f"Loading + tracing {args.model} (BS=1 per core) ...")
    model_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=os.environ.get(
        "DINOV3_REPO_DIR", "/mnt/models/dinov3"))
    m = trace_dinov3(model_cpu, is_convnext=(cfg["arch"] == "convnext"),
                     img_size=args.img_size, batch_size=1, inline_weights=True)
    del model_cpu

    # Single-core reference for scaling math.
    x1 = torch.randn(1, 3, args.img_size, args.img_size)
    for _ in range(10):
        m(x1)
    lat = []
    for _ in range(50):
        t = time.time(); m(x1); lat.append(time.time() - t)
    base = 1.0 / np.mean(lat)
    print(f"single-core reference: {base:.1f} img/s")

    recs = []
    for nw in [int(w) for w in args.workers.split(",")]:
        model_dp = torch_neuronx.DataParallel(
            m, device_ids=list(range(args.cores)), dim=0)
        default_nw = getattr(model_dp, "num_workers", None)
        model_dp.num_workers = nw

        x = torch.randn(bs, 3, args.img_size, args.img_size)
        for _ in range(args.warmup):
            model_dp(x)

        lat = []
        t0 = time.time()
        for _ in range(args.iters):
            t = time.time(); model_dp(x); lat.append((time.time() - t) * 1000)
        wall = time.time() - t0

        lat = np.array(lat)
        tput = args.iters * bs / wall
        rec = {
            "num_workers": nw,
            "default_num_workers": default_nw,
            "cores": args.cores,
            "batch_total": bs,
            "throughput_img_s": round(tput, 1),
            "scaling_vs_1core": round(tput / base, 2),
            "median_latency_ms": round(float(np.median(lat)), 2),
            "p99_latency_ms": round(float(np.percentile(lat, 99)), 2),
        }
        recs.append(rec)
        print(f"  num_workers={nw:>3d} (default was {default_nw}): "
              f"{tput:>7.1f} img/s  {tput / base:>5.2f}x  "
              f"p50={rec['median_latency_ms']}ms p99={rec['p99_latency_ms']}ms")

        del model_dp
        time.sleep(2)

    best = max(recs, key=lambda r: r["throughput_img_s"])
    payload = {
        "model": args.model,
        "cores": args.cores,
        "single_core_img_s": round(base, 1),
        "results": recs,
        "best": best,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nBEST: num_workers={best['num_workers']} -> "
          f"{best['throughput_img_s']} img/s ({best['scaling_vs_1core']}x)")
    d2 = next((r for r in recs if r["num_workers"] == 2), None)
    if d2:
        print(f"vs num_workers=2 (library default): {d2['throughput_img_s']} img/s "
              f"({d2['scaling_vs_1core']}x)  -> fix is worth "
              f"{best['throughput_img_s'] / d2['throughput_img_s']:.2f}x")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
