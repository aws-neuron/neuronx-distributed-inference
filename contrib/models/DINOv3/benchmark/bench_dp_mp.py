"""True per-instance inf2 throughput via independent worker processes.

`torch_neuronx.DataParallel` dispatches from one Python process, so it saturates
on host-side dispatch well before it saturates the NeuronCores -- on
inf2.24xlarge (12 cores) it plateaus at only ~2.0-2.6x single-core. That is a
harness artifact, not a hardware limit.

This script instead runs N independent single-core worker processes, each pinned
to its own NeuronCore via NEURON_RT_VISIBLE_CORES, and sums their throughput.
This is how a real serving deployment would be built (N model server replicas)
and gives the honest per-instance number.

Usage:
    python bench_dp_mp.py --model vit_l --cores 12 --batch-size 1
    python bench_dp_mp.py --model vit_b --cores 1,2,4,8,12 --batch-size 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

CONTRIB_SRC = os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src")
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def worker(model_key, batch_size, img_size, bench_iters, warmup_iters, result_path):
    """One worker: compile (cache-hit) + benchmark on the single visible core."""
    sys.path.insert(0, CONTRIB_SRC)
    sys.path.insert(0, REPO_DIR)
    import numpy as np
    import torch
    from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, trace_dinov3

    cfg = MODEL_REGISTRY[model_key]
    model_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO_DIR)
    m = trace_dinov3(
        model_cpu,
        is_convnext=(cfg["arch"] == "convnext"),
        img_size=img_size,
        batch_size=batch_size,
        inline_weights=(batch_size == 1),
    )
    del model_cpu

    x = torch.randn(batch_size, 3, img_size, img_size)
    for _ in range(warmup_iters):
        m(x)

    # Barrier: wait for a shared start time so all workers measure concurrently.
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
            "core": os.environ.get("NEURON_RT_VISIBLE_CORES"),
            "iters": bench_iters,
            "wall_s": wall,
            "throughput_img_s": bench_iters * batch_size / wall,
            "mean_latency_ms": float(lat.mean()),
            "median_latency_ms": float(np.median(lat)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
        }, f)


def run_fanout(model_key, n_cores, batch_size, img_size, bench_iters, warmup_iters, tmpdir):
    """Launch n_cores workers, one per core, and aggregate."""
    os.makedirs(tmpdir, exist_ok=True)
    # Generous warmup budget: every worker compiles/loads before the timed window.
    start_at = time.time() + 60 + 25 * max(0, n_cores - 4)

    procs, paths = [], []
    for c in range(n_cores):
        rp = os.path.join(tmpdir, f"w{c}.json")
        if os.path.exists(rp):
            os.unlink(rp)
        paths.append(rp)
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = str(c)
        env["BENCH_START_AT"] = str(start_at)
        env["NEURON_RT_NUM_CORES"] = "1"
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--_worker",
             "--model", model_key, "--batch-size", str(batch_size),
             "--img-size", str(img_size), "--bench-iters", str(bench_iters),
             "--warmup-iters", str(warmup_iters), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        ))

    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-600:])

    per = []
    for rp in paths:
        if os.path.exists(rp):
            with open(rp) as f:
                per.append(json.load(f))

    agg = sum(w["throughput_img_s"] for w in per) if per else 0.0
    return {
        "cores_requested": n_cores,
        "cores_succeeded": len(per),
        "batch_size": batch_size,
        "aggregate_img_s": round(agg, 1),
        "per_worker_img_s": [round(w["throughput_img_s"], 1) for w in per],
        "median_latency_ms": (
            round(sum(w["median_latency_ms"] for w in per) / len(per), 3) if per else None
        ),
        "p99_latency_ms": (
            round(max(w["p99_latency_ms"] for w in per), 3) if per else None
        ),
        "errors": errs[:2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_l")
    ap.add_argument("--cores", default="12")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bench-iters", type=int, default=200)
    ap.add_argument("--warmup-iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        worker(args.model, args.batch_size, args.img_size,
               args.bench_iters, args.warmup_iters, args._result)
        return

    out = args.out or f"results_mp_{args.model}_bs{args.batch_size}.json"
    core_list = [int(c) for c in args.cores.split(",")]
    recs = []
    for n in core_list:
        print(f"\n### {args.model} BS={args.batch_size}: fanning out to {n} core(s)")
        r = run_fanout(args.model, n, args.batch_size, args.img_size,
                       args.bench_iters, args.warmup_iters,
                       f"/home/ubuntu/mp_tmp_{args.model}_{n}")
        recs.append(r)
        print(f"  cores ok: {r['cores_succeeded']}/{n}  "
              f"aggregate: {r['aggregate_img_s']} img/s  "
              f"p50={r['median_latency_ms']}ms")
        if r["errors"]:
            print(f"  errors: {r['errors'][0][:300]}")
        with open(out, "w") as f:
            json.dump({"model": args.model, "results": recs}, f, indent=2)

    print(f"\n{'=' * 64}\n{args.model} multi-process scaling (BS={args.batch_size})\n{'=' * 64}")
    print(f"{'cores':>6s} {'aggregate img/s':>16s} {'scaling':>9s} {'p50 ms':>9s}")
    base = recs[0]["aggregate_img_s"] / recs[0]["cores_requested"] if recs else 1
    for r in recs:
        sc = r["aggregate_img_s"] / base / r["cores_requested"] if base else 0
        print(f"{r['cores_requested']:>6d} {r['aggregate_img_s']:>16.1f} "
              f"{sc:>8.2f}x {str(r['median_latency_ms']):>9s}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
