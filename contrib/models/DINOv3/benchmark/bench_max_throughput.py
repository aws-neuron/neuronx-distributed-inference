"""Find MAXIMUM throughput on a small inf2 instance (inf2.xlarge = 2 cores, 4 vCPU).

Prior work on inf2.24xlarge established that `torch_neuronx.DataParallel` is
host-dispatch-bound and that independent worker processes are the right way to
saturate the cores. But inf2.24xlarge has 8 vCPU per NeuronCore, while
inf2.xlarge has only 2 vCPU per core. If dispatch is CPU-bound, per-core
throughput on inf2.xlarge will be LOWER than on inf2.24xlarge, and batching
becomes the lever that recovers it (one dispatch amortized over N images).

So this script sweeps the two axes that matter jointly:
  - batch size per inference call (amortizes host dispatch)
  - number of worker processes (1 per core, plus oversubscription check)

and reports the single best (workers x batch) configuration per model.

Usage:
    python bench_max_throughput.py --models vit_l --batch-sizes 1,2,4,8,16
    python bench_max_throughput.py --models vit_s,vit_b --workers 1,2 --batch-sizes 1,4,16
"""

import argparse
import json
import os
import subprocess
import sys
import time

CONTRIB_SRC = os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src")
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def worker_main(model_key, batch_size, img_size, bench_iters, warmup_iters, result_path):
    sys.path.insert(0, CONTRIB_SRC)
    sys.path.insert(0, REPO_DIR)
    import numpy as np
    import torch
    from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, trace_dinov3

    cfg = MODEL_REGISTRY[model_key]
    model_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO_DIR)
    # inline_weights_to_neff matters enormously on small inf2 hosts: with weights
    # NOT inlined, ViT-L BS=1 drops from 56.9 -> 8.4 img/s (6.8x) on inf2.xlarge,
    # because the weights are re-fed over the host bus on every call and there are
    # only 4 vCPU to do it. Always inline unless the caller forces otherwise.
    inline = os.environ.get("BENCH_INLINE_WEIGHTS", "1") == "1"
    m = trace_dinov3(
        model_cpu,
        is_convnext=(cfg["arch"] == "convnext"),
        img_size=img_size,
        batch_size=batch_size,
        inline_weights=inline,
    )
    del model_cpu

    x = torch.randn(batch_size, 3, img_size, img_size)
    for _ in range(warmup_iters):
        m(x)

    start_at = float(os.environ["BENCH_START_AT"])
    while time.time() < start_at:
        time.sleep(0.005)

    lat = []
    t_begin = time.time()
    for _ in range(bench_iters):
        t0 = time.time()
        m(x)
        lat.append((time.time() - t0) * 1000)
    wall = time.time() - t_begin

    lat = np.array(lat)
    with open(result_path, "w") as f:
        json.dump({
            "throughput_img_s": bench_iters * batch_size / wall,
            "mean_latency_ms": float(lat.mean()),
            "median_latency_ms": float(np.median(lat)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
            "wall_s": wall,
        }, f)


def run_config(model_key, n_workers, batch_size, img_size, bench_iters,
               warmup_iters, n_cores, tmpdir):
    """Run n_workers processes at the given batch size; aggregate throughput."""
    os.makedirs(tmpdir, exist_ok=True)
    # Compile+load is slow on a 4-vCPU host; give a generous shared start barrier.
    start_at = time.time() + 90 + 45 * max(0, n_workers - 2)

    procs, paths = [], []
    for w in range(n_workers):
        rp = os.path.join(tmpdir, f"w{w}_bs{batch_size}.json")
        if os.path.exists(rp):
            os.unlink(rp)
        paths.append(rp)
        env = dict(os.environ)
        # Round-robin workers onto the available cores (allows oversubscription).
        env["NEURON_RT_VISIBLE_CORES"] = str(w % n_cores)
        env["NEURON_RT_NUM_CORES"] = "1"
        env["BENCH_START_AT"] = str(start_at)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--_worker", "--model", model_key,
             "--batch-size", str(batch_size), "--img-size", str(img_size),
             "--bench-iters", str(bench_iters), "--warmup-iters", str(warmup_iters),
             "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        ))

    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-700:])

    per = []
    for rp in paths:
        if os.path.exists(rp):
            with open(rp) as f:
                per.append(json.load(f))

    if not per:
        return {"workers": n_workers, "batch_size": batch_size,
                "status": "failed", "errors": errs[:1]}

    agg = sum(w["throughput_img_s"] for w in per)
    return {
        "workers": n_workers,
        "batch_size": batch_size,
        "workers_ok": len(per),
        "status": "ok" if len(per) == n_workers else "partial",
        "aggregate_img_s": round(agg, 1),
        "per_worker_img_s": [round(w["throughput_img_s"], 1) for w in per],
        "median_latency_ms": round(sum(w["median_latency_ms"] for w in per) / len(per), 2),
        "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2),
        "errors": errs[:1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_l")
    ap.add_argument("--workers", default="1,2")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--n-cores", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bench-iters", type=int, default=60)
    ap.add_argument("--warmup-iters", type=int, default=10)
    ap.add_argument("--out", default="results_max_tput.json")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        worker_main(args.model, args.batch_size, args.img_size,
                    args.bench_iters, args.warmup_iters, args._result)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    workers = [int(w) for w in args.workers.split(",")]
    bss = [int(b) for b in args.batch_sizes.split(",")]

    all_out = {}
    for mk in models:
        recs = []
        print(f"\n{'#' * 70}\n# {mk}  (cores={args.n_cores})\n{'#' * 70}")
        for nw in workers:
            for bs in bss:
                print(f"  workers={nw} bs={bs} ...", end=" ", flush=True)
                r = run_config(mk, nw, bs, args.img_size, args.bench_iters,
                               args.warmup_iters, args.n_cores,
                               f"/home/ubuntu/mt_{mk}_{nw}_{bs}")
                recs.append(r)
                if r["status"] == "failed":
                    print(f"FAILED {r.get('errors', [''])[0][:160]}")
                else:
                    print(f"{r['aggregate_img_s']} img/s  "
                          f"p50={r['median_latency_ms']}ms p99={r['p99_latency_ms']}ms"
                          + ("" if r["status"] == "ok" else f"  [{r['status']}]"))
                all_out[mk] = recs
                with open(args.out, "w") as f:
                    json.dump(all_out, f, indent=2)

        ok = [r for r in recs if r.get("status") == "ok"]
        if ok:
            best = max(ok, key=lambda r: r["aggregate_img_s"])
            print(f"\n  BEST {mk}: {best['aggregate_img_s']} img/s "
                  f"@ workers={best['workers']} bs={best['batch_size']} "
                  f"(p50 {best['median_latency_ms']}ms)")

    print(f"\n{'=' * 70}\nMAX THROUGHPUT SUMMARY ({args.n_cores} cores)\n{'=' * 70}")
    print(f"{'model':16s} {'best img/s':>11s} {'workers':>8s} {'bs':>4s} {'p50 ms':>8s}")
    print("-" * 54)
    for mk, recs in all_out.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        if not ok:
            print(f"{mk:16s} {'ALL FAILED':>11s}")
            continue
        b = max(ok, key=lambda r: r["aggregate_img_s"])
        print(f"{mk:16s} {b['aggregate_img_s']:>11.1f} {b['workers']:>8d} "
              f"{b['batch_size']:>4d} {b['median_latency_ms']:>8.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
