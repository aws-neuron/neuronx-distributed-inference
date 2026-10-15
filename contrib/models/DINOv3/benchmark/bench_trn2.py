"""Task 020 (trn2 half): re-benchmark DINOv3 on trn2 with SDK 2.31.

Every existing trn2 number in this project is suspect for one of two reasons:

1. **The DataParallel num_workers defect.** Task 021 showed
   `torch_neuronx.DataParallel` pins num_workers=2 regardless of core count,
   costing 5.18x on 12 inf2 cores. All prior trn2 DP=4 numbers were measured with
   the broken default, so the DP columns understate the hardware.
2. **BS=1 only.** The prior trn2 sweeps did not vary batch size. On inf2, ViT-H+
   gained 6.8x from BS=1 -> BS=4.

This re-measures with:
  - independent worker processes (the production pattern), AND
  - fixed-num_workers DataParallel (to quantify the defect on trn2), AND
  - a batch-size sweep.

trn2.3xlarge is 1 Trainium2 device. LNC=2 (default) exposes 4 logical cores at
24 GB each; LNC=1 exposes 8 at 12 GB. Task 019 found LNC=1 wins 1.81x for ViT-L,
but that conclusion rested on DP numbers measured with the broken num_workers --
so re-test both modes.

LNC is a per-process runtime setting: NEURON_LOGICAL_NC_CONFIG (runtime) must be
paired with NEURON_CC_FLAGS=--lnc=N (compile). If they disagree the runtime
rejects the NEFF at load.

Usage (LNC is inherited from the environment, set by the caller):
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--lnc=2" \
        python bench_trn2.py --models vit_l --cores 4 --batch-sizes 1,4,8
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch

import torch_neuronx
from modeling_dinov3 import (COMPILER_ARGS_CONVNEXT, COMPILER_ARGS_VIT,
                             MODEL_REGISTRY, load_dinov3_model, trace_dinov3)

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")

def _lnc_args():
    """Extra compiler args to force the LNC target.

    `trace_dinov3()` passes an explicit `compiler_args` list to
    torch_neuronx.trace(), which OVERRIDES the NEURON_CC_FLAGS env var. So
    exporting NEURON_CC_FLAGS="--lnc=1" silently has no effect: the NEFF is
    compiled at the default --lnc=2 and the runtime then rejects it with
      NRT:nrt_load_util Mismatch detected between Runtime configuration and NEFF.
      Runtime currently configured with `NEURON_LOGICAL_NC_CONFIG=1` but NEFF ...
      was compiled with `--lnc=2`.
    Pass --logical-nc-config through to the compiler explicitly instead.
    """
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    return ["--logical-nc-config", str(lnc)] if lnc else []


def _trace(cpu, cfg, bs, img_size):
    """trace_dinov3 with the LNC target appended to compiler_args."""
    base = (COMPILER_ARGS_CONVNEXT if cfg["arch"] == "convnext"
            else COMPILER_ARGS_VIT)
    args = list(base) + _lnc_args()
    x = torch.randn(bs, 3, img_size, img_size)
    return torch_neuronx.trace(cpu, x, compiler_args=args,
                               inline_weights_to_neff=True)



# ----------------------------------------------------------------- worker mode
def worker_main(model_key, bs, img_size, iters, warmup, result_path):
    cfg = MODEL_REGISTRY[model_key]
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    m = _trace(cpu, cfg, bs, img_size)
    del cpu
    x = torch.randn(bs, 3, img_size, img_size)
    for _ in range(warmup):
        m(x)

    start_at = float(os.environ["BENCH_START_AT"])
    while time.time() < start_at:
        time.sleep(0.005)

    lat = []
    t0 = time.time()
    for _ in range(iters):
        t = time.time(); m(x); lat.append((time.time() - t) * 1000)
    wall = time.time() - t0
    lat = np.array(lat)
    with open(result_path, "w") as f:
        json.dump({"throughput_img_s": iters * bs / wall,
                   "median_latency_ms": float(np.median(lat)),
                   "p99_latency_ms": float(np.percentile(lat, 99))}, f)


def precompile(model_key, bs, img_size):
    """Compile+cache the NEFF in ONE process before fanning out.

    Without this, N workers each invoke neuronx-cc concurrently and contend for
    the runtime, producing:
      RuntimeError: The PyTorch Neuron Runtime could not be initialized.
    Once the NEFF is in the on-disk compile cache, every worker gets a cache hit
    and only needs to load it.
    """
    cfg = MODEL_REGISTRY[model_key]
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    m = _trace(cpu, cfg, bs, img_size)
    del cpu, m


def run_mp(model_key, n_cores, bs, img_size, iters, warmup, tmpdir):
    """N independent single-core processes -- the production serving pattern."""
    os.makedirs(tmpdir, exist_ok=True)
    print("    precompiling (single process) ...", flush=True)
    precompile(model_key, bs, img_size)
    time.sleep(3)   # let the compiling process fully release the cores
    start_at = time.time() + 60 + 20 * max(0, n_cores - 4)
    procs, paths = [], []
    for c in range(n_cores):
        rp = os.path.join(tmpdir, f"w{c}_bs{bs}.json")
        if os.path.exists(rp):
            os.unlink(rp)
        paths.append(rp)
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = str(c)
        env["NEURON_RT_NUM_CORES"] = "1"
        env["BENCH_START_AT"] = str(start_at)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--_worker", "--model", model_key,
             "--batch-size", str(bs), "--img-size", str(img_size),
             "--iters", str(iters), "--warmup", str(warmup), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-500:])
    per = []
    for rp in paths:
        if os.path.exists(rp):
            per.append(json.load(open(rp)))
    if not per:
        return {"status": "failed", "errors": errs[:1]}
    return {
        "status": "ok" if len(per) == n_cores else "partial",
        "cores_ok": len(per),
        "aggregate_img_s": round(sum(w["throughput_img_s"] for w in per), 1),
        "median_latency_ms": round(
            sum(w["median_latency_ms"] for w in per) / len(per), 2),
        "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2),
    }


def run_dp(model_key, n_cores, bs, img_size, iters, warmup, fix_workers):
    """DataParallel, with and without the num_workers fix."""
    import torch_neuronx
    cfg = MODEL_REGISTRY[model_key]
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    m = _trace(cpu, cfg, 1, img_size)
    del cpu
    dp = torch_neuronx.DataParallel(m, device_ids=list(range(n_cores)), dim=0)
    default_nw = getattr(dp, "num_workers", None)
    if fix_workers:
        dp.num_workers = n_cores
    total = bs * n_cores
    x = torch.randn(total, 3, img_size, img_size)
    for _ in range(warmup):
        dp(x)
    lat = []
    t0 = time.time()
    for _ in range(iters):
        t = time.time(); dp(x); lat.append((time.time() - t) * 1000)
    wall = time.time() - t0
    lat = np.array(lat)
    return {"status": "ok", "num_workers": dp.num_workers,
            "default_num_workers": default_nw, "batch_total": total,
            "aggregate_img_s": round(iters * total / wall, 1),
            "median_latency_ms": round(float(np.median(lat)), 2),
            "p99_latency_ms": round(float(np.percentile(lat, 99)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_l")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", default="mp,dp", help="mp | dp | mp,dp")
    ap.add_argument("--out", default="results_trn2.json")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        worker_main(args.model, args.batch_size, args.img_size, args.iters,
                    args.warmup, args._result)
        return

    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    env = {"lnc": lnc, "cc_flags": os.environ.get("NEURON_CC_FLAGS", ""),
           "cores": args.cores, "torch": torch.__version__}
    print(json.dumps(env, indent=2))

    modes = [m.strip() for m in args.mode.split(",")]
    out = []
    for mk in [m.strip() for m in args.models.split(",")]:
        for bs in [int(b) for b in args.batch_sizes.split(",")]:
            if "mp" in modes:
                print(f"\n### {mk} BS={bs} LNC={lnc}: {args.cores} independent procs")
                r = run_mp(mk, args.cores, bs, args.img_size, args.iters,
                           args.warmup, f"/home/ubuntu/t20_{mk}_{lnc}_{bs}")
                r.update({"model": mk, "batch_size": bs, "mode": "multiprocess",
                          "lnc": lnc, "cores": args.cores})
                out.append(r)
                print(f"    {r.get('aggregate_img_s', 'FAILED')} img/s  "
                      f"p50={r.get('median_latency_ms')}ms")
            if "dp" in modes and bs == 1:
                # Only at BS=1: this exists to quantify the num_workers defect.
                for fix in (False, True):
                    tag = "fixed" if fix else "default"
                    print(f"### {mk} LNC={lnc}: DataParallel num_workers {tag}")
                    try:
                        r = run_dp(mk, args.cores, bs, args.img_size,
                                   args.iters, args.warmup, fix)
                    except Exception as e:
                        r = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                    r.update({"model": mk, "batch_size": bs,
                              "mode": f"dataparallel_{tag}", "lnc": lnc,
                              "cores": args.cores})
                    out.append(r)
                    print(f"    {r.get('aggregate_img_s', 'FAILED')} img/s "
                          f"(num_workers={r.get('num_workers')})")
            with open(args.out, "w") as f:
                json.dump({"env": env, "results": out}, f, indent=2)

    print(f"\n{'=' * 78}\ntrn2 SDK 2.31, LNC={lnc}, {args.cores} cores\n{'=' * 78}")
    print(f"{'model':14s} {'BS':>3s} {'mode':22s} {'img/s':>9s} {'p50 ms':>8s}")
    print("-" * 62)
    for r in out:
        if r.get("status") != "ok":
            print(f"{r['model']:14s} {r['batch_size']:>3d} {r['mode']:22s} {'FAILED':>9s}")
            continue
        print(f"{r['model']:14s} {r['batch_size']:>3d} {r['mode']:22s} "
              f"{r['aggregate_img_s']:>9.1f} {r['median_latency_ms']:>8.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
