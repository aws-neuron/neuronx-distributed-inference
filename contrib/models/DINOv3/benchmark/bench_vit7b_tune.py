"""ViT-H+ and ViT-7B best-config on inf2 -- the two models outside the main sweep.

ViT-H+ (840M, SwiGLU): its trace numbers predate RoPE precompute. Prior finding: FP32 at
BS=1 is pathological (SwiGLU lowered badly) and BF16 fixes it 4.87x. Sweep dtype x BS with
RoPE precompute on, via the same saved-NEFF method as sweep_best.py.

ViT-7B (6.7B, TP via neuronx-distributed ModelBuilder): compile_vit7b_tp() computes RoPE
IN-GRAPH every call (TPDinoViT7B.forward calls self.rope_embed(H=H, W=W)) and uses -O1.
RoPE precompute gave 2.0x on the trace ViTs; this checks whether hoisting it helps the TP
model too. The fix is a constant table, same as precompute_rope().

ViT-7B needs TP=2 = both cores of an inf2.xlarge (TP=1 OOMs at 15.95/16 GB).

Usage:
    python big_models.py vith_compile --out-dir /home/ubuntu/neffs_big
    python big_models.py vit7b --tp 2 --bs 1 --rope precompute
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch


def vit7b(tp, bs, rope_mode, iters, warmup, repeats, opt):
    """Compile + time ViT-7B TP, optionally with RoPE hoisted to a constant."""
    import modeling_dinov3 as M
    from neuronx_distributed import ModelBuilder
    from neuronx_distributed.trace.parallel_context import NxDParallelState

    os.environ["NEURON_RT_NUM_CORES"] = str(tp)
    x = torch.randn(bs, 3, 224, 224).bfloat16()
    cargs = (f"--auto-cast=matmult --auto-cast-type=bf16 --model-type=transformer "
             f"--enable-saturate-infinity {opt}")
    rec = {"model": "vit_7b", "tp": tp, "bs": bs, "rope": rope_mode, "opt": opt}
    with NxDParallelState(world_size=tp, tensor_model_parallel_size=tp):
        model = M.create_vit7b_tp(dtype=torch.bfloat16)
        if rope_mode == "precompute":
            with torch.no_grad():
                sin, cos = model.rope_embed(H=14, W=14)
            sin_c, cos_c = sin.detach().clone(), cos.detach().clone()
            model.rope_embed.forward = lambda *a, **k: (sin_c, cos_c)
        sd = model.state_dict()
        b = ModelBuilder(model)
        t0 = time.time()
        b.trace(args=x, tag="vit7b")
        nxd = b.compile(compiler_workdir=f"/tmp/v7b_{rope_mode}_{bs}_{opt.strip('-')}",
                        compiler_args=cargs)
        rec["compile_s"] = round(time.time() - t0, 1)
    nxd.set_weights([sd for _ in range(tp)])
    nxd.to_neuron()
    for _ in range(warmup):
        nxd(x)
    runs = []
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(iters):
            nxd(x)
        runs.append(round(iters * bs / (time.time() - t0), 2))
    rec.update({"runs": runs, "best": max(runs),
                "converged": abs(runs[-1] - runs[-2]) / runs[-2] < 0.03})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--rope", default="ingraph", choices=["ingraph", "precompute"])
    ap.add_argument("--opt", default="-O1")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", default="results_vit7b.json")
    a = ap.parse_args()
    if a.mode == "vit7b":
        r = vit7b(a.tp, a.bs, a.rope, a.iters, a.warmup, a.repeats, a.opt)
        print(json.dumps(r), flush=True)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
