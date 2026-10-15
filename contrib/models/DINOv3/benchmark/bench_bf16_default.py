"""Should the traced path default to BF16 weights instead of FP32+autocast?

The contrib ships FP32 weights + `--auto-cast=matmult`. Evidence that this is
leaving performance on the table:
  ViT-H+ single-core BS=1: FP32 10.2 -> BF16 55.2 img/s = 5.41x
  ViT-S/B/L 2-core best:   FP32 -> BF16 = 1.03-1.10x
All at cos_sim 0.999997 vs 1.000000, i.e. accuracy-neutral.

Two gaps this closes:
  1. ConvNeXt has NEVER been measured in BF16 on any path.
  2. ViT-H+'s 5.41x is a single-core BS=1 point; needs the full batch sweep.

Configs per model:
  fp32_matmult -- what the contrib ships today
  bf16_matmult -- BF16 weights, autocast still on
  bf16_none    -- BF16 weights, autocast off (should be equivalent: once the
                  weights are BF16 there is nothing left for matmult autocast
                  to cast, so a delta here would mean autocast is doing
                  something unexpected)

ConvNeXt gets no --model-type=transformer (it is not a transformer); ViT does.

Usage:
    python bench_bf16_default.py --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
        --cores 12 --batch-sizes 1,4,8
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
import torch.nn.functional as F
import torch_neuronx

from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")

# dtype + the autocast flags. --model-type is appended per-architecture.
CONFIGS = {
    "fp32_matmult": (torch.float32, ["--auto-cast", "matmult",
                                     "--auto-cast-type", "bf16"]),
    "bf16_matmult": (torch.bfloat16, ["--auto-cast", "matmult",
                                      "--auto-cast-type", "bf16"]),
    "bf16_none": (torch.bfloat16, ["--auto-cast", "none"]),
}


def compiler_args(key, cfg_name):
    dtype, base = CONFIGS[cfg_name]
    args = list(base)
    # ConvNeXt is not a transformer -- the contrib deliberately omits this flag
    # for it, so keep that difference intact.
    if MODEL_REGISTRY[key]["arch"] != "convnext":
        args += ["--model-type", "transformer"]
    return dtype, args


def build(key, cfg_name, bs, img_size):
    dtype, cargs = compiler_args(key, cfg_name)
    cfg = MODEL_REGISTRY[key]
    torch.manual_seed(0)
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    m = cpu.to(dtype)
    x = torch.randn(bs, 3, img_size, img_size).to(dtype)
    return torch_neuronx.trace(m, x, compiler_args=cargs,
                               inline_weights_to_neff=True), x


def accuracy(key, cfg_name, img_size):
    """cos_sim vs a CPU FP32 reference built from identically-seeded weights."""
    cfg = MODEL_REGISTRY[key]
    torch.manual_seed(0)
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    xc = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        ref = cpu(xc).float()
    del cpu
    dtype, _ = compiler_args(key, cfg_name)
    mn, _ = build(key, cfg_name, 1, img_size)
    got = mn(xc.to(dtype)).float()
    del mn
    return float(F.cosine_similarity(ref.flatten().unsqueeze(0),
                                     got.flatten().unsqueeze(0)).item())


def worker_main(key, cfg_name, bs, img_size, iters, warmup, result_path):
    mn, x = build(key, cfg_name, bs, img_size)
    for _ in range(warmup):
        mn(x)
    start_at = float(os.environ["BENCH_START_AT"])
    while time.time() < start_at:
        time.sleep(0.005)
    lat = []
    t0 = time.time()
    for _ in range(iters):
        t = time.time(); mn(x); lat.append((time.time() - t) * 1000)
    wall = time.time() - t0
    lat = np.array(lat)
    with open(result_path, "w") as f:
        json.dump({"throughput_img_s": iters * bs / wall,
                   "median_latency_ms": float(np.median(lat)),
                   "p99_latency_ms": float(np.percentile(lat, 99))}, f)


def _sub(argv, extra_env=None):
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": "0", "NEURON_RT_NUM_CORES": "1"}
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, __file__] + argv, env=env,
                          capture_output=True)


def run_cfg(key, cfg_name, n_cores, bs, img_size, iters, warmup, tmpdir):
    """Fan out to n_cores independent processes. Assumes the NEFF is cached."""
    os.makedirs(tmpdir, exist_ok=True)
    start_at = time.time() + 60 + 15 * max(0, n_cores - 4)
    procs, paths = [], []
    for c in range(n_cores):
        rp = os.path.join(tmpdir, f"w{c}.json")
        if os.path.exists(rp):
            os.unlink(rp)
        paths.append(rp)
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = str(c)
        env["NEURON_RT_NUM_CORES"] = "1"
        env["BENCH_START_AT"] = str(start_at)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--_worker", "--model", key,
             "--config", cfg_name, "--batch-size", str(bs),
             "--img-size", str(img_size), "--iters", str(iters),
             "--warmup", str(warmup), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-500:])
    per = [json.load(open(rp)) for rp in paths if os.path.exists(rp)]
    if not per:
        return {"status": "failed", "errors": errs[:1]}
    return {"status": "ok" if len(per) == n_cores else "partial",
            "cores_ok": len(per),
            "aggregate_img_s": round(sum(w["throughput_img_s"] for w in per), 1),
            "median_latency_ms": round(
                sum(w["median_latency_ms"] for w in per) / len(per), 2),
            "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_s,vit_b,vit_l,convnext_tiny,convnext_base")
    ap.add_argument("--configs", default="fp32_matmult,bf16_matmult,bf16_none")
    ap.add_argument("--cores", type=int, default=12)
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results_bf16_default.json")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_precompile", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_accuracy", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._precompile:
        build(args.model, args.config, args.batch_size, args.img_size)
        return
    if args._accuracy:
        cos = accuracy(args.model, args.config, args.img_size)
        with open(args._result, "w") as f:
            json.dump({"cos_sim": cos}, f)
        return
    if args._worker:
        worker_main(args.model, args.config, args.batch_size, args.img_size,
                    args.iters, args.warmup, args._result)
        return

    models = [m.strip() for m in args.models.split(",")]
    cfgs = [c.strip() for c in args.configs.split(",")]
    bss = [int(b) for b in args.batch_sizes.split(",")]

    # ---- Phase 1: compile everything in subprocesses ----
    # Keeps neuronx-cc off the box during timing, and process exit is the only
    # reliable way to release a NeuronCore held by a loaded traced model.
    print("=" * 66 + "\nPhase 1: compile\n" + "=" * 66, flush=True)
    for key in models:
        for cname in cfgs:
            for bs in bss:
                t0 = time.time()
                _sub(["--_precompile", "--model", key, "--config", cname,
                      "--batch-size", str(bs), "--img-size", str(args.img_size)])
                print(f"  {key} {cname} BS={bs}: {time.time() - t0:.0f}s", flush=True)
                time.sleep(3)

    # ---- Phase 2: accuracy ----
    print("=" * 66 + "\nPhase 2: accuracy\n" + "=" * 66, flush=True)
    cos_map = {}
    for key in models:
        for cname in cfgs:
            rp = f"/tmp/acc_{key}_{cname}.json"
            _sub(["--_accuracy", "--model", key, "--config", cname,
                  "--img-size", str(args.img_size), "--_result", rp])
            time.sleep(4)
            cos = None
            if os.path.exists(rp):
                cos = json.load(open(rp)).get("cos_sim")
            cos_map[(key, cname)] = cos
            print(f"  {key:16s} {cname:14s} cos={cos}", flush=True)

    # ---- Phase 3: throughput ----
    print("=" * 66 + f"\nPhase 3: throughput ({args.cores} cores)\n" + "=" * 66,
          flush=True)
    out = []
    for key in models:
        for cname in cfgs:
            for bs in bss:
                r = run_cfg(key, cname, args.cores, bs, args.img_size,
                            args.iters, args.warmup,
                            f"/home/ubuntu/bd_{key}_{cname}_{bs}")
                r.update({"model": key, "config": cname, "batch_size": bs,
                          "cores": args.cores, "cos_sim": cos_map.get((key, cname))})
                out.append(r)
                print(f"  {key:16s} {cname:14s} BS={bs:<2d} "
                      f"{r.get('aggregate_img_s', 'FAILED')}", flush=True)
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)

    # ---- Summary ----
    print(f"\n{'=' * 78}\nBF16-as-default on the traced path ({args.cores} cores, "
          f"batch-swept best)\n{'=' * 78}")
    print(f"{'model':16s} {'FP32+mm':>9s} {'BF16+mm':>9s} {'BF16 none':>10s} "
          f"{'best gain':>10s} {'cos(bf16)':>10s}")
    print("-" * 70)
    for key in models:
        def best(c):
            rs = [r for r in out if r["model"] == key and r["config"] == c
                  and r.get("status") == "ok"]
            return max((r["aggregate_img_s"] for r in rs), default=None)
        f32, bmm, bnone = best("fp32_matmult"), best("bf16_matmult"), best("bf16_none")
        bb = max([v for v in (bmm, bnone) if v], default=None)
        g = f"{bb / f32:.2f}x" if (bb and f32) else "-"
        cs = cos_map.get((key, "bf16_matmult"))
        fmt = lambda v: f"{v:.1f}" if v else "-"
        print(f"{key:16s} {fmt(f32):>9s} {fmt(bmm):>9s} {fmt(bnone):>10s} "
              f"{g:>10s} {(f'{cs:.6f}' if cs else '-'):>10s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
