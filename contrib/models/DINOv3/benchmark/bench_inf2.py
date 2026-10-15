"""DINOv3 on Inferentia2: max model size + performance refresh.

Two jobs:
  1. MAX MODEL SIZE -- how large a DINOv3 backbone fits and runs on inf2?
     inf2 = 32 GB HBM per Inferentia2 device, 16 GB per NeuronCore.
     Sweeps single-core trace for every registry model, then reports NEFF size
     and HBM headroom. ViT-7B is attempted separately via TP.
  2. PERF REFRESH -- single-core latency/throughput + DataParallel scaling for
     every model that fits, plus CPU-vs-Neuron accuracy.

Usage:
    python bench_inf2.py --models vit_s,vit_b,vit_l,vit_h_plus,convnext_tiny,convnext_base
    python bench_inf2.py --models vit_l --dp-cores 12
    python bench_inf2.py --list

Results are written as JSON to --out (default results_inf2.json).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

CONTRIB_SRC = os.environ.get(
    "DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"
)
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
sys.path.insert(0, CONTRIB_SRC)
sys.path.insert(0, REPO_DIR)

import numpy as np
import torch

import torch_neuronx
from modeling_dinov3 import (
    MODEL_REGISTRY,
    benchmark_dataparallel,
    benchmark_model,
    load_dinov3_model,
    trace_dinov3,
    validate_accuracy,
)


def neff_size_bytes(model_neuron):
    """Best-effort NEFF size for a traced model, via its saved artifact."""
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            p = f.name
        torch.jit.save(model_neuron, p)
        n = os.path.getsize(p)
        os.unlink(p)
        return n
    except Exception:
        return None


def nrt_hbm_used_bytes():
    """Query neuron-monitor for current device memory usage (bytes)."""
    try:
        out = subprocess.run(
            ["neuron-monitor", "-c", "1"], capture_output=True, text=True, timeout=30
        )
        data = json.loads(out.stdout.strip().split("\n")[0])
        total = 0
        for rg in data.get("neuron_runtime_data", []):
            mem = rg.get("report", {}).get("memory_used", {})
            bd = mem.get("neuron_runtime_used_bytes", {})
            total += bd.get("device", 0)
        return total or None
    except Exception:
        return None


def run_one(key, batch_size, dp_cores, img_size, do_dp, do_acc):
    """Trace + validate + benchmark a single registry model."""
    cfg = MODEL_REGISTRY[key]
    is_convnext = cfg["arch"] == "convnext"
    rec = {
        "model": key,
        "hub_name": cfg["hub_name"],
        "params_M": cfg["params_M"],
        "arch": cfg["arch"],
        "img_size": img_size,
        "batch_size": batch_size,
        "status": "unknown",
    }

    print(f"\n{'=' * 72}\n{key}  ({cfg['params_M']}M, {cfg['arch']})  BS={batch_size}\n{'=' * 72}")

    try:
        t0 = time.time()
        model_cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO_DIR)
        rec["load_time_s"] = round(time.time() - t0, 1)

        n_params = sum(p.numel() for p in model_cpu.parameters())
        rec["params_actual"] = n_params
        rec["params_actual_M"] = round(n_params / 1e6, 1)
        print(f"  params (actual): {n_params:,} ({n_params / 1e6:.1f}M)")

        t0 = time.time()
        # inline_weights=False: inline_weights_to_neff=True causes an async XLA
        # error at BS>=4 (DINOv3 project Task 019).
        model_neuron = trace_dinov3(
            model_cpu,
            is_convnext=is_convnext,
            img_size=img_size,
            batch_size=batch_size,
            inline_weights=(batch_size == 1),
        )
        rec["compile_time_s"] = round(time.time() - t0, 1)
        print(f"  compile: {rec['compile_time_s']}s")

        nb = neff_size_bytes(model_neuron)
        if nb:
            rec["artifact_bytes"] = nb
            rec["artifact_MB"] = round(nb / 1e6, 1)
            print(f"  artifact: {rec['artifact_MB']} MB")

        hbm = nrt_hbm_used_bytes()
        if hbm:
            rec["hbm_used_bytes"] = hbm
            rec["hbm_used_MB"] = round(hbm / 1e6, 1)
            rec["hbm_pct_of_core_16GB"] = round(100.0 * hbm / (16 * 1024**3), 2)
            print(f"  HBM in use: {rec['hbm_used_MB']} MB "
                  f"({rec['hbm_pct_of_core_16GB']}% of 16 GB core)")

        if do_acc:
            acc = validate_accuracy(
                model_cpu, model_neuron, img_size=img_size, batch_size=batch_size
            )
            rec["accuracy"] = {k: float(v) for k, v in acc.items()}
            print(f"  cosine_sim: {acc['cosine_sim']:.6f}   "
                  f"max_diff: {acc['max_diff']:.2e}")

        perf = benchmark_model(
            model_neuron, img_size=img_size, batch_size=batch_size,
            warmup_iters=10, bench_iters=100,
        )
        rec["single_core"] = {k: round(float(v), 3) for k, v in perf.items()}
        print(f"  1-core: {perf['throughput_img_s']:.1f} img/s  "
              f"p50={perf['median_latency_ms']:.2f}ms  "
              f"p99={perf['p99_latency_ms']:.2f}ms")

        if do_dp:
            del model_cpu
            bss = [dp_cores, dp_cores * 2, dp_cores * 4]
            print(f"  DataParallel DP={dp_cores}, batch sizes {bss} ...")
            try:
                dp = benchmark_dataparallel(
                    model_neuron, num_cores=dp_cores, img_size=img_size,
                    batch_sizes=bss, warmup_iters=10, bench_iters=50,
                )
                rec["dataparallel"] = {
                    str(bs): {k: round(float(v), 3) for k, v in m.items()}
                    for bs, m in dp.items()
                }
                best = max(dp.items(), key=lambda kv: kv[1]["throughput_img_s"])
                rec["dp_peak_img_s"] = round(best[1]["throughput_img_s"], 1)
                rec["dp_peak_bs"] = best[0]
                rec["dp_cores"] = dp_cores
                for bs, m in sorted(dp.items()):
                    print(f"    DP BS={bs}: {m['throughput_img_s']:.1f} img/s")
                print(f"  DP={dp_cores} peak: {rec['dp_peak_img_s']} img/s "
                      f"(BS={best[0]}, {rec['dp_peak_img_s'] / perf['throughput_img_s']:.2f}x 1-core)")
            except Exception as e:
                rec["dataparallel_error"] = f"{type(e).__name__}: {e}"
                print(f"  DP FAILED: {type(e).__name__}: {e}")

        rec["status"] = "ok"

    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2500:]
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg or "insufficient" in msg:
            rec["failure_kind"] = "OOM"
        print(f"  FAILED: {type(e).__name__}: {e}")

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_s,vit_b,vit_l,vit_h_plus,convnext_tiny,convnext_base")
    ap.add_argument("--batch-sizes", default="1")
    ap.add_argument("--dp-cores", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-dp", action="store_true")
    ap.add_argument("--no-acc", action="store_true")
    ap.add_argument("--out", default="results_inf2.json")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in MODEL_REGISTRY.items():
            print(f"{k:16s} {v['hub_name']:24s} {v['params_M']:>7}M  {v['arch']}")
        return

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    bss = [int(b) for b in args.batch_sizes.split(",")]
    bad = [k for k in keys if k not in MODEL_REGISTRY]
    if bad:
        sys.exit(f"unknown models: {bad}. available: {list(MODEL_REGISTRY)}")

    env = {
        "instance_type": os.environ.get("BENCH_INSTANCE_TYPE", "inf2.24xlarge"),
        "hbm_per_device_GB": 32,
        "hbm_per_core_GB": 16,
        "dp_cores": args.dp_cores,
        "img_size": args.img_size,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        env["neuronx_cc"] = subprocess.run(
            ["neuronx-cc", "--version"], capture_output=True, text=True, timeout=60
        ).stdout.strip().split("\n")[0]
    except Exception:
        pass
    print(json.dumps(env, indent=2))

    results = []
    for bs in bss:
        for k in keys:
            results.append(
                run_one(k, bs, args.dp_cores, args.img_size,
                        not args.no_dp, not args.no_acc)
            )
            with open(args.out, "w") as f:
                json.dump({"env": env, "results": results}, f, indent=2)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    hdr = f"{'model':16s} {'params':>9s} {'compile':>8s} {'1-core':>10s} {'DP peak':>10s} {'cos':>9s} {'status':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sc = r.get("single_core", {}).get("throughput_img_s", None)
        cos = r.get("accuracy", {}).get("cosine_sim", None)
        print(f"{r['model']:16s} {r.get('params_actual_M', '?'):>8}M "
              f"{r.get('compile_time_s', '-'):>7}s "
              f"{(f'{sc:.1f}' if sc else '-'):>10s} "
              f"{(str(r.get('dp_peak_img_s')) if r.get('dp_peak_img_s') else '-'):>10s} "
              f"{(f'{cos:.6f}' if cos else '-'):>9s} "
              f"{r['status']:>7s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
