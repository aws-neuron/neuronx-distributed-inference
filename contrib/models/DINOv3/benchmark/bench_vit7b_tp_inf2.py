"""DINOv3 ViT-7B (6.7B params) on Inferentia2 via tensor parallelism.

This is the max-model-size probe. inf2 has 16 GB HBM per NeuronCore
(32 GB per Inferentia2 device, 2 cores per device). The ViT-7B BF16 weights
alone are ~13.4 GB, so TP is required to spread them across cores.

inf2.24xlarge = 6 devices / 12 cores -> TP up to 12 (powers of 2: 2, 4, 8).
inf2.48xlarge = 12 devices / 24 cores -> TP up to 24.

Sweeps TP degrees and records which ones fit, compile, and run.

Usage:
    NEURON_RT_NUM_CORES=8 python bench_vit7b_tp_inf2.py --tp 8
    python bench_vit7b_tp_inf2.py --tp 2,4,8        # sequential subprocess sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

CONTRIB_SRC = os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src")
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")


def run_single_tp(tp, batch_size, img_size, out_path):
    """Run one TP degree in this process. Must be launched with NEURON_RT_NUM_CORES=tp."""
    sys.path.insert(0, CONTRIB_SRC)
    sys.path.insert(0, REPO_DIR)

    import torch
    from modeling_dinov3 import benchmark_model, compile_vit7b_tp

    rec = {
        "model": "vit_7b",
        "params_M": 6716,
        "tp_degree": tp,
        "batch_size": batch_size,
        "img_size": img_size,
        "instance_type": os.environ.get("BENCH_INSTANCE_TYPE", "inf2.24xlarge"),
        "hbm_per_core_GB": 16,
        "status": "unknown",
    }

    print(f"\n{'=' * 72}\nViT-7B  TP={tp}  BS={batch_size}\n{'=' * 72}")
    try:
        save_dir = f"/mnt/models/compiled/dinov3_vit7b_tp{tp}_bs{batch_size}"
        t0 = time.time()
        nxd_model = compile_vit7b_tp(
            tp_degree=tp, img_size=img_size, batch_size=batch_size, save_dir=save_dir
        )
        rec["compile_total_s"] = round(time.time() - t0, 1)
        print(f"  total compile+load: {rec['compile_total_s']}s")

        # NEFF size on disk
        try:
            sz = subprocess.run(
                ["du", "-sb", save_dir], capture_output=True, text=True, timeout=120
            ).stdout.split()[0]
            rec["artifact_bytes"] = int(sz)
            rec["artifact_GB"] = round(int(sz) / 1e9, 2)
            print(f"  artifact dir: {rec['artifact_GB']} GB")
        except Exception:
            pass

        perf = benchmark_model(
            nxd_model, img_size=img_size, batch_size=batch_size,
            warmup_iters=5, bench_iters=50,
        )
        rec["perf"] = {k: round(float(v), 3) for k, v in perf.items()}
        rec["throughput_img_s"] = round(float(perf["throughput_img_s"]), 1)
        print(f"  TP={tp}: {perf['throughput_img_s']:.1f} img/s  "
              f"p50={perf['median_latency_ms']:.2f}ms  "
              f"p99={perf['p99_latency_ms']:.2f}ms")
        rec["status"] = "ok"

    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
        m = str(e).lower()
        if any(s in m for s in ("out of memory", "oom", "insufficient", "not enough")):
            rec["failure_kind"] = "OOM"
        elif "nc count" in m or "num cores" in m or "cores" in m:
            rec["failure_kind"] = "CORE_COUNT"
        print(f"  FAILED: {type(e).__name__}: {e}")

    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {out_path}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", default="8", help="comma list of TP degrees")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="results_vit7b_inf2.json")
    ap.add_argument("--_single", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._single is not None:
        run_single_tp(args._single, args.batch_size, args.img_size, args.out)
        return

    tps = [int(t) for t in args.tp.split(",")]
    if len(tps) == 1:
        # Caller is responsible for NEURON_RT_NUM_CORES
        run_single_tp(tps[0], args.batch_size, args.img_size, args.out)
        return

    # Multi-TP: each degree needs its own process with its own NEURON_RT_NUM_CORES
    all_recs = []
    for tp in tps:
        part = f"{args.out}.tp{tp}"
        env = dict(os.environ)
        env["NEURON_RT_NUM_CORES"] = str(tp)
        print(f"\n### subprocess: TP={tp} (NEURON_RT_NUM_CORES={tp})")
        subprocess.run(
            [sys.executable, __file__, "--_single", str(tp),
             "--batch-size", str(args.batch_size),
             "--img-size", str(args.img_size), "--out", part],
            env=env,
        )
        try:
            with open(part) as f:
                all_recs.append(json.load(f))
        except Exception:
            all_recs.append({"tp_degree": tp, "status": "no_result_file"})

    with open(args.out, "w") as f:
        json.dump({"results": all_recs}, f, indent=2)

    print(f"\n{'=' * 72}\nViT-7B TP SWEEP SUMMARY (inf2)\n{'=' * 72}")
    print(f"{'TP':>4s} {'status':>8s} {'img/s':>9s} {'p50 ms':>9s} {'artifact':>10s} {'note':<20s}")
    print("-" * 68)
    for r in all_recs:
        print(f"{r.get('tp_degree', '?'):>4} {r.get('status', '?'):>8s} "
              f"{str(r.get('throughput_img_s', '-')):>9s} "
              f"{str(r.get('perf', {}).get('median_latency_ms', '-')):>9s} "
              f"{str(r.get('artifact_GB', '-')):>10s} "
              f"{r.get('failure_kind', ''):<20s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
