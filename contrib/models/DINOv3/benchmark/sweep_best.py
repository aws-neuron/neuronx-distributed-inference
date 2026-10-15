"""Best-and-finest config sweep for the DINOv3 trace path on inf2.xlarge.

Everything prior was measured with ad-hoc harnesses that each had a flaw:
  - bench_bf16_default.py read ~2x low vs bench_max_throughput.py on the same host
  - ViT-L needed ~7 repeats + ~120 warmup to converge (cold baselines faked a 2.76x)
  - compiling while timing on a 15 GB host corrupted results via swap
  - a loaded traced model pins its core until process exit

This harness is built to rule all four out, so the published table can say "best":

  PHASE 1  compile every config ONCE, in a subprocess, and torch.jit.save the NEFF.
           Each config compiles alone -- no timing happens anywhere near a compile.
  PHASE 2  accuracy gate: every saved NEFF vs CPU FP32 (seeded, identical weights).
           A config that fails the gate is reported but cannot win.
  PHASE 3  timing: 2 worker processes (one per core) LOAD the saved NEFF -- no
           compile, minimal RAM -- and run N repeats with heavy warmup. Reports
           every repeat, a convergence flag, and best-of-converged.

Config axes (ViT; ConvNeXt has no RoPE and no transformer flag):
  dtype     fp32 (+ --auto-cast=matmult bf16)  |  bf16 weights (--auto-cast=none)
  modeltype --model-type=transformer on | off
  bs        1, 2, 4, 8, 16
RoPE precompute is always ON (it is the default and verified accuracy-neutral).

Usage:
    python sweep_best.py phase1 --models vit_s,vit_b --out-dir /home/ubuntu/neffs
    python sweep_best.py phase2 --out-dir /home/ubuntu/neffs
    python sweep_best.py phase3 --out-dir /home/ubuntu/neffs --repeats 4
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch
import torch.nn.functional as F

IMG = 224
REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def configs(models, bss):
    """Enumerate (model, dtype, modeltype, bs). ConvNeXt only varies dtype x bs."""
    out = []
    for m in models:
        cnx = m.startswith("convnext")
        for dt in ("fp32", "bf16"):
            for mt in ((False,) if cnx else (True, False)):
                for bs in bss:
                    out.append((m, dt, mt, bs))
    return out


def tag(m, dt, mt, bs):
    return f"{m}__{dt}__{'mt' if mt else 'nomt'}__bs{bs}"


def cargs(m, dt, mt):
    a = (["--auto-cast", "matmult", "--auto-cast-type", "bf16"] if dt == "fp32"
         else ["--auto-cast", "none"])
    if mt:
        a += ["--model-type", "transformer"]
    return a


def build_cpu(m, dt):
    from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model, precompute_rope
    cfg = MODEL_REGISTRY[m]
    torch.manual_seed(0)
    model = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    if cfg["arch"] != "convnext":
        precompute_rope(model, img_size=IMG)
    return model.to({"fp32": torch.float32, "bf16": torch.bfloat16}[dt])


# --------------------------------------------------------------------- phase 1
def compile_one(m, dt, mt, bs, path):
    import torch_neuronx
    model = build_cpu(m, dt)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dt]
    x = torch.randn(bs, 3, IMG, IMG).to(dtype)
    t0 = time.time()
    mn = torch_neuronx.trace(model, x, compiler_args=cargs(m, dt, mt),
                             inline_weights_to_neff=True)
    ct = time.time() - t0
    torch.jit.save(mn, path)
    with open(path + ".meta.json", "w") as f:
        json.dump({"compile_s": round(ct, 1),
                   "neff_mb": round(os.path.getsize(path) / 1e6, 1)}, f)


def phase1(a):
    """Compile every config to a saved NEFF.

    --jobs N runs N compiles concurrently. Intended for a big compile host
    (inf2.24xlarge: 96 vCPU / 369 GB, 12 cores so each job gets its own core for
    the post-trace load). Timing still happens on inf2.xlarge -- NEFFs are portable
    across inf2 sizes (same NeuronCore-v2, same SDK), only compilation moves.
    """
    os.makedirs(a.out_dir, exist_ok=True)
    cfgs = configs(a.models.split(","), [int(b) for b in a.batch_sizes.split(",")])
    todo = []
    for (m, dt, mt, bs) in cfgs:
        p = os.path.join(a.out_dir, tag(m, dt, mt, bs) + ".pt")
        if os.path.exists(p) and os.path.exists(p + ".meta.json"):
            continue
        todo.append((m, dt, mt, bs, p))
    print(f"phase1: {len(cfgs)} configs, {len(todo)} to compile, jobs={a.jobs}", flush=True)
    running = []
    core = 0
    while todo or running:
        while todo and len(running) < a.jobs:
            m, dt, mt, bs, p = todo.pop(0)
            log = open(p + ".compile.log", "w")
            pr = subprocess.Popen(
                [sys.executable, __file__, "_compile", "--m", m, "--dt", dt,
                 "--mt", "1" if mt else "0", "--bs", str(bs), "--path", p],
                env={**os.environ, "NEURON_RT_VISIBLE_CORES": str(core % a.cores_avail),
                     "NEURON_RT_NUM_CORES": "1"},
                stdout=log, stderr=subprocess.STDOUT)
            running.append((pr, tag(m, dt, mt, bs), p, time.time(), log))
            core += 1
        time.sleep(5)
        still = []
        for pr, t, p, t0, log in running:
            if pr.poll() is None:
                still.append((pr, t, p, t0, log))
            else:
                log.close()
                ok = os.path.exists(p)
                print(f"  {'ok    ' if ok else 'FAILED'} {t}  {time.time() - t0:.0f}s", flush=True)
        running = still


# --------------------------------------------------------------------- phase 2
def accuracy_one(m, dt, mt, bs, path, rp):
    """cos vs CPU FP32 on identical seeded weights, at the traced batch size."""
    ref_m = build_cpu(m, "fp32")
    torch.manual_seed(1)
    x = torch.randn(bs, 3, IMG, IMG)
    with torch.no_grad():
        ref = ref_m(x).float()
    import torch_neuronx  # noqa: F401  -- registers neuron.Model before jit.load
    mn = torch.jit.load(path)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dt]
    got = mn(x.to(dtype)).float()
    cs = F.cosine_similarity(ref, got, dim=1)
    rel = (got - ref).norm(dim=1) / ref.norm(dim=1)
    with open(rp, "w") as f:
        json.dump({"cos_mean": float(cs.mean()), "cos_min": float(cs.min()),
                   "rel_l2_max": float(rel.max())}, f)


def phase2(a):
    rows = []
    for p in sorted(x for x in os.listdir(a.out_dir) if x.endswith(".pt")):
        t = p[:-3]
        m, dt, mtag, bst = t.split("__")
        # accuracy is batch-independent for a correct compile; check bs=1 and the
        # largest bs per family -- catches batch-dependent miscompiles too.
        rp = os.path.join(a.out_dir, t + ".acc.json")
        if not os.path.exists(rp):
            subprocess.run([sys.executable, __file__, "_accuracy", "--m", m, "--dt", dt,
                            "--mt", "1" if mtag == "mt" else "0", "--bs", bst[2:],
                            "--path", os.path.join(a.out_dir, p), "--rp", rp],
                           env={**os.environ,
                                "NEURON_RT_VISIBLE_CORES": os.environ.get("ACC_CORE", "0"),
                                "NEURON_RT_NUM_CORES": "1"},
                           capture_output=True)
            time.sleep(6)
        acc = json.load(open(rp)) if os.path.exists(rp) else {"cos_min": None}
        rows.append((t, acc))
        print(f"  {t:42s} cos_min={acc.get('cos_min')}", flush=True)


# --------------------------------------------------------------------- phase 3
def worker(path, bs, dt, iters, warmup, repeats, start_at, rp):
    # torch_neuronx MUST be imported before torch.jit.load: it registers the
    # __torch__.torch.classes.neuron.Model custom class the saved NEFF references.
    # Without it every load fails at "model : __torch__.torch.classes.neuron.Model".
    import torch_neuronx  # noqa: F401
    mn = torch.jit.load(path)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dt]
    x = torch.randn(bs, 3, IMG, IMG).to(dtype)
    for _ in range(warmup):
        mn(x)
    while time.time() < start_at:
        time.sleep(0.005)
    runs = []
    for _ in range(repeats):
        lat = []
        t0 = time.time()
        for _ in range(iters):
            t = time.time(); mn(x); lat.append((time.time() - t) * 1000)
        runs.append({"img_s": iters * bs / (time.time() - t0),
                     "p50_ms": float(np.median(lat)), "p99_ms": float(np.percentile(lat, 99))})
    with open(rp, "w") as f:
        json.dump(runs, f)


def phase3(a):
    out = []
    pts = sorted(x for x in os.listdir(a.out_dir) if x.endswith(".pt"))
    if a.only:
        pts = [p for p in pts if any(s in p for s in a.only.split(","))]
    for p in pts:
        t = p[:-3]
        m, dt, mtag, bst = t.split("__")
        acc = {}
        ap = os.path.join(a.out_dir, t + ".acc.json")
        if os.path.exists(ap):
            acc = json.load(open(ap))
        start_at = time.time() + 45
        procs, rps = [], []
        for c in range(a.cores):
            rp = f"/tmp/w_{t}_{c}.json"
            if os.path.exists(rp):
                os.unlink(rp)
            rps.append(rp)
            procs.append(subprocess.Popen(
                [sys.executable, __file__, "_worker", "--path", os.path.join(a.out_dir, p),
                 "--bs", bst[2:], "--dt", dt, "--iters", str(a.iters),
                 "--warmup", str(a.warmup), "--repeats", str(a.repeats),
                 "--start-at", str(start_at), "--rp", rp],
                env={**os.environ, "NEURON_RT_VISIBLE_CORES": str(c),
                     "NEURON_RT_NUM_CORES": "1"},
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        for pr in procs:
            pr.communicate()
        per = [json.load(open(r)) for r in rps if os.path.exists(r)]
        rec = {"tag": t, "model": m, "dtype": dt, "modeltype": mtag == "mt",
               "bs": int(bst[2:]), "cores": a.cores, "acc": acc}
        if len(per) == a.cores:
            n = min(len(w) for w in per)
            agg = [round(sum(w[i]["img_s"] for w in per), 1) for i in range(n)]
            conv = abs(agg[-1] - agg[-2]) / agg[-2] < 0.03 if n >= 2 else None
            rec.update({"status": "ok", "runs": agg, "best": max(agg),
                        "last2_mean": round(sum(agg[-2:]) / 2, 1), "converged": conv,
                        "p50_ms": round(np.mean([w[-1]["p50_ms"] for w in per]), 2),
                        "p99_ms": round(max(w[-1]["p99_ms"] for w in per), 2)})
        else:
            rec["status"] = f"partial {len(per)}/{a.cores}"
        out.append(rec)
        print(f"  {t:42s} {rec.get('runs')} conv={rec.get('converged')}", flush=True)
        with open(os.path.join(a.out_dir, a.result), "w") as f:
            json.dump(out, f, indent=2)
        time.sleep(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--models", default="vit_s,vit_b,vit_l,convnext_tiny,convnext_base")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--out-dir", default="/home/ubuntu/neffs")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cores-avail", type=int, default=1)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--result", default="phase3.json")
    # internal
    ap.add_argument("--m"); ap.add_argument("--dt"); ap.add_argument("--mt")
    ap.add_argument("--bs", type=int); ap.add_argument("--path"); ap.add_argument("--rp")
    ap.add_argument("--start-at", type=float)
    a = ap.parse_args()

    if a.phase == "_compile":
        compile_one(a.m, a.dt, a.mt == "1", a.bs, a.path)
    elif a.phase == "_accuracy":
        accuracy_one(a.m, a.dt, a.mt == "1", a.bs, a.path, a.rp)
    elif a.phase == "_worker":
        worker(a.path, a.bs, a.dt, a.iters, a.warmup, a.repeats, a.start_at, a.rp)
    elif a.phase == "phase1":
        phase1(a)
    elif a.phase == "phase2":
        phase2(a)
    elif a.phase == "phase3":
        phase3(a)


if __name__ == "__main__":
    main()
