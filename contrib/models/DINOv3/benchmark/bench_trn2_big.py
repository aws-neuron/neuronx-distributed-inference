"""Close the last trn2 gap: ViT-H+ (840M) and ViT-7B (6.7B, TP) on SDK 2.31.

These two were never re-measured. The README still carries their SDK 2.28
figures -- ViT-H+ 10.5 img/s (DataParallel DP=4, BS=1) and ViT-7B 38.8 img/s
(TP=4) -- and ViT-H+ is certainly understated for the two known reasons:
  - measured with the DataParallel num_workers=2 defect
  - measured at BS=1 only, and ViT-H+ gained 6.8x from BS=1 -> BS=4 on inf2

ViT-H+ uses trace + independent worker processes (it fits in one core).
ViT-7B needs neuronx-distributed ModelBuilder TP; on trn2.3xlarge LNC=2 that is
TP=4 (4 logical cores), LNC=1 would allow TP=8.

Usage:
    NEURON_LOGICAL_NC_CONFIG=2 python bench_trn2_big.py --mode vith --cores 4
    NEURON_RT_NUM_CORES=4 NEURON_LOGICAL_NC_CONFIG=2 python bench_trn2_big.py --mode vit7b --tp 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch
import torch_neuronx

from modeling_dinov3 import (COMPILER_ARGS_CONVNEXT, COMPILER_ARGS_VIT,
                             MODEL_REGISTRY, benchmark_model, compile_vit7b_tp,
                             load_dinov3_model)

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def _lnc_args():
    """LNC must go through compiler_args, not NEURON_CC_FLAGS.

    trace_dinov3()/torch_neuronx.trace() receive an explicit compiler_args list
    which OVERRIDES NEURON_CC_FLAGS, so exporting NEURON_CC_FLAGS="--lnc=1" is
    silently ignored and the runtime then rejects the NEFF at load with a
    Runtime/NEFF LNC mismatch.
    """
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    return ["--logical-nc-config", str(lnc)] if lnc else []


def _trace(cpu, cfg, bs, img_size):
    base = (COMPILER_ARGS_CONVNEXT if cfg["arch"] == "convnext"
            else COMPILER_ARGS_VIT)
    x = torch.randn(bs, 3, img_size, img_size)
    return torch_neuronx.trace(cpu, x, compiler_args=list(base) + _lnc_args(),
                               inline_weights_to_neff=True)


# --------------------------------------------------------------- ViT-H+ (trace)
def worker_main(key, bs, img_size, iters, warmup, result_path):
    cfg = MODEL_REGISTRY[key]
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


def run_vith(n_cores, bs, img_size, iters, warmup, tmpdir):
    """ViT-H+ via N independent single-core processes."""
    os.makedirs(tmpdir, exist_ok=True)
    # Precompile once: ViT-H+ takes ~330-690s and N concurrent neuronx-cc
    # invocations contend for the runtime and fail to init.
    print("    precompiling ViT-H+ (single process, slow) ...", flush=True)
    cfg = MODEL_REGISTRY["vit_h_plus"]
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    t0 = time.time()
    m = _trace(cpu, cfg, bs, img_size)
    ct = time.time() - t0
    del cpu, m
    time.sleep(5)
    print(f"    compile {ct:.1f}s", flush=True)

    start_at = time.time() + 180 + 30 * max(0, n_cores - 4)
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
            [sys.executable, __file__, "--_worker", "--model", "vit_h_plus",
             "--batch-size", str(bs), "--img-size", str(img_size),
             "--iters", str(iters), "--warmup", str(warmup), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-500:])
    per = [json.load(open(rp)) for rp in paths if os.path.exists(rp)]
    if not per:
        return {"status": "failed", "errors": errs[:1], "compile_time_s": round(ct, 1)}
    return {"status": "ok" if len(per) == n_cores else "partial",
            "cores_ok": len(per), "compile_time_s": round(ct, 1),
            "aggregate_img_s": round(sum(w["throughput_img_s"] for w in per), 1),
            "median_latency_ms": round(
                sum(w["median_latency_ms"] for w in per) / len(per), 2),
            "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2)}


# ------------------------------------------------------------------ ViT-7B (TP)
def run_vit7b(tp, bs, img_size, iters, warmup):
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    rec = {"model": "vit_7b", "tp_degree": tp, "batch_size": bs, "lnc": lnc}
    try:
        t0 = time.time()
        m = compile_vit7b_tp(tp_degree=tp, img_size=img_size, batch_size=bs,
                             save_dir=f"/mnt/models/compiled/v7b_tp{tp}_bs{bs}_lnc{lnc}")
        rec["compile_time_s"] = round(time.time() - t0, 1)
        perf = benchmark_model(m, img_size=img_size, batch_size=bs,
                               warmup_iters=warmup, bench_iters=iters)
        rec["aggregate_img_s"] = round(float(perf["throughput_img_s"]), 1)
        rec["median_latency_ms"] = round(float(perf["median_latency_ms"]), 2)
        rec["p99_latency_ms"] = round(float(perf["p99_latency_ms"]), 2)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vith", choices=["vith", "vit7b"])
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default=None)
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
    out = args.out or f"results_trn2_{args.mode}_lnc{lnc}.json"
    recs = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        if args.mode == "vith":
            print(f"\n### ViT-H+ LNC={lnc} BS={bs}: {args.cores} independent procs",
                  flush=True)
            r = run_vith(args.cores, bs, args.img_size, args.iters, args.warmup,
                         f"/home/ubuntu/vith_{lnc}_{bs}")
            r.update({"model": "vit_h_plus", "batch_size": bs, "lnc": lnc,
                      "cores": args.cores})
        else:
            print(f"\n### ViT-7B TP={args.tp} LNC={lnc} BS={bs}", flush=True)
            r = run_vit7b(args.tp, bs, args.img_size, args.iters, args.warmup)
        recs.append(r)
        print(f"    {r.get('aggregate_img_s', 'FAILED')} img/s  "
              f"p50={r.get('median_latency_ms')}ms", flush=True)
        if r.get("status") == "failed":
            print(f"    {str(r.get('error'))[:250]}", flush=True)
        with open(out, "w") as f:
            json.dump({"lnc": lnc, "results": recs}, f, indent=2)

    print(f"\n{'=' * 62}\ntrn2 SDK 2.31 {args.mode} LNC={lnc}\n{'=' * 62}")
    print(f"{'BS':>3s} {'img/s':>9s} {'p50 ms':>9s} {'compile':>9s}")
    print("-" * 36)
    for r in recs:
        if r.get("status") != "ok":
            print(f"{r['batch_size']:>3d} {'FAILED':>9s}")
            continue
        print(f"{r['batch_size']:>3d} {r['aggregate_img_s']:>9.1f} "
              f"{r['median_latency_ms']:>9.2f} {r.get('compile_time_s', '-'):>8}s")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
