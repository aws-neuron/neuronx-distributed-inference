#!/usr/bin/env python3
"""Probe which BigVGAN mel-len buckets compile on the current Neuron SDK.

History: README/code notes that ``T >= 256`` crashed with
``[NCC_ITIN902] TensorInitialization`` on an older neuronxcc. This forced
the default BIGVGAN_BUCKETS=[60, 128], which in turn made full-utterance
synthesis (T_mel ~= 600-2000) silently fall back to CPU BigVGAN — the
last remaining 2.2x slowdown vs H100 in the per-module bench.

This script attempts to compile BigVGAN at a sequence of bucket sizes
and reports which ones the current SDK accepts. It does NOT touch the
main pipeline; pass-results inform what BIGVGAN_BUCKETS should be set to.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0 \
      python examples/probe_bigvgan_buckets.py [--buckets 256,512,1024,2048]
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401


def _load_bigvgan(model_path):
    """Load just the BigVGAN sub-module from the HF Token2Wav weights."""
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniToken2WavBigVGANModel,
    )
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    bv_cfg = cfg.token2wav_config.bigvgan_config
    bigvgan = Qwen2_5OmniToken2WavBigVGANModel(bv_cfg).eval().float()
    # Try to load real weights if available, else random is fine for trace.
    sd_path = Path(model_path) / "model.safetensors.index.json"
    if sd_path.exists():
        try:
            from safetensors.torch import load_file
            import json
            with open(sd_path) as f:
                index = json.load(f)
            wanted = {
                k.replace("token2wav.code2wav_bigvgan_model.", ""): v
                for k, v in index["weight_map"].items()
                if "code2wav_bigvgan_model" in k
            }
            shards = {}
            base = Path(model_path)
            files = set(wanted.values())
            for fn in files:
                shards[fn] = load_file(str(base / fn))
            sd = {}
            for k, fn in wanted.items():
                full = "token2wav.code2wav_bigvgan_model." + k
                if full in shards[fn]:
                    sd[k] = shards[fn][full]
            missing, unexpected = bigvgan.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"  (sd load: {len(sd)} loaded, "
                    f"{len(missing)} missing, {len(unexpected)} unexpected)"
                )
        except Exception as e:
            print(f"  (skipping real-weight load: {e})")
    return bigvgan, int(bv_cfg.mel_dim)


class _BigVGANTraceWrapper(torch.nn.Module):
    """Same wrapper used in modeling_qwen25_omni_token2wav.py."""

    def __init__(self, bigvgan):
        super().__init__()
        self.bigvgan = bigvgan
        for module in self.bigvgan.modules():
            if module.__class__.__name__ == "TorchActivation1d":
                module.forward = module.act.forward

    def forward(self, mel_spectrogram):
        bv = self.bigvgan
        processed = bv.process_mel_spectrogram(mel_spectrogram)
        hidden = bv.conv_pre(processed)
        for layer_index in range(bv.num_upsample_layers):
            hidden = bv.ups[layer_index][0](hidden)
            residual = sum(
                bv.resblocks[layer_index * bv.num_residual_blocks + b](hidden)
                for b in range(bv.num_residual_blocks)
            )
            hidden = residual / bv.num_residual_blocks
        hidden = bv.activation_post(hidden)
        wav = bv.conv_post(hidden)
        return torch.clamp(wav, min=-1.0, max=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "QWEN25_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"
        ),
    )
    p.add_argument(
        "--buckets",
        default="256,512,1024,2048",
        help="Comma-separated mel_len buckets to probe.",
    )
    p.add_argument(
        "--workdir",
        default="/tmp/bigvgan_probe",
        help="Per-bucket compiler workdir parent.",
    )
    p.add_argument(
        "--bucket",
        type=int,
        default=256,
        help=(
            "Single bucket size to use when sweeping --compiler-args variants. "
            "Default 256, the smallest bucket that crashes on stock --auto-cast=none."
        ),
    )
    p.add_argument(
        "--sweep-args",
        action="store_true",
        help=(
            "Instead of trying every bucket with the default compiler-args, "
            "fix --bucket and sweep a list of compiler-args variants. Use this "
            "to probe whether a different compiler flag bypasses the "
            "TensorInitialization internal pass."
        ),
    )
    p.add_argument(
        "--compiler-args",
        default="--auto-cast=none",
        help=(
            "Compiler args used in non-sweep mode. Default mirrors the main "
            "code path. Use --auto-cast=all to bypass NCC_ITIN902 (with "
            "numerics drift)."
        ),
    )
    p.add_argument(
        "--verify-numerics",
        action="store_true",
        help=(
            "Run each PASS-ing NEFF on a real-ish mel input and compare to "
            "a CPU fp32 reference. Required when probing --auto-cast=all "
            "since that flag silently casts internal matmuls."
        ),
    )
    args = p.parse_args()

    buckets = sorted({int(x) for x in args.buckets.split(",") if x.strip()})
    print("=" * 60)
    print("BigVGAN bucket compile probe")
    print("=" * 60)
    print(f"  Model:   {args.model_path}")
    if args.sweep_args:
        print(f"  Mode:    --sweep-args at T_mel={args.bucket}")
    else:
        print(f"  Buckets: {buckets}")
    print(f"  Workdir: {args.workdir}")

    try:
        import torch_neuronx
    except ImportError:
        sys.exit("torch_neuronx not available; run inside the Neuron venv.")

    print("\n--- Loading BigVGAN ---")
    bigvgan, mel_dim = _load_bigvgan(args.model_path)
    wrapper = _BigVGANTraceWrapper(bigvgan).eval()
    print(f"  mel_dim={mel_dim}")

    os.makedirs(args.workdir, exist_ok=True)

    # CPU fp32 reference forward, used by --verify-numerics.
    ref_cache = {}

    def _ref_forward(example):
        key = tuple(example.shape)
        if key not in ref_cache:
            with torch.no_grad():
                ref_cache[key] = wrapper(example).clone()
        return ref_cache[key]

    def _verify(traced, example, label):
        with torch.no_grad():
            out = traced(example)
        ref = _ref_forward(example)
        diff = (ref - out.to(torch.float32)).abs()
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.to(torch.float32).flatten().unsqueeze(0),
        ).item()
        print(
            f"    [verify {label}] max_abs={diff.max().item():.3e} "
            f"mean_abs={diff.mean().item():.3e} cosine={cos:.6f}"
        )
        return cos, diff.max().item()

    def _try(label, example, compiler_args, wd):
        os.makedirs(wd, exist_ok=True)
        t0 = time.time()
        try:
            traced = torch_neuronx.trace(
                wrapper, example,
                compiler_workdir=wd, compiler_args=compiler_args,
            )
            dt = time.time() - t0
            out = traced(example)
            msg = (
                f"  OK in {dt:.1f}s, "
                f"out shape={tuple(out.shape)}, dtype={out.dtype}"
            )
            print(msg)
            extra = ""
            if args.verify_numerics:
                # Use a non-zero realistic-magnitude input rather than zeros.
                torch.manual_seed(0)
                real = torch.randn_like(example) * 1.5
                cos, max_abs = _verify(traced, real, label)
                extra = f", cos={cos:.4f}, max_abs={max_abs:.2e}"
            return ("PASS", f"{dt:.1f}s{extra}")
        except Exception as e:
            dt = time.time() - t0
            err = str(e).splitlines()[0][:120] if str(e) else type(e).__name__
            print(f"  FAIL after {dt:.1f}s: {type(e).__name__}: {err}")
            return ("FAIL", f"{type(e).__name__}: {err}")

    results = []
    if args.sweep_args:
        # Try to bypass NCC_ITIN902 TensorInitialization at a single bucket
        # by varying compiler args. These are the levers most likely to swap
        # out the offending pass without semantically changing the kernel.
        bucket = args.bucket
        example = torch.zeros((1, mel_dim, bucket), dtype=torch.float32)
        variants = [
            ("auto-cast=none (baseline)", "--auto-cast=none"),
            ("auto-cast=all", "--auto-cast=all"),
            ("auto-cast=matmul", "--auto-cast=matmult"),
            ("optlevel=1", "--auto-cast=none --optlevel=1"),
            ("optlevel=2", "--auto-cast=none --optlevel=2"),
            ("optlevel=3", "--auto-cast=none --optlevel=3"),
            ("target=trn2", "--auto-cast=none --target=trn2"),
            ("logical_nc_config=2", "--auto-cast=none --logical-nc-config=2"),
            ("disable_internal_io_dge",
             "--auto-cast=none --disable-internal-io-dge"),
            ("model-type=transformer",
             "--auto-cast=none --model-type=transformer"),
        ]
        print(f"\n--- Sweep compiler args at T_mel={bucket} ---")
        for label, cargs in variants:
            print(f"\n[{label}] cargs={cargs!r}")
            wd = os.path.join(args.workdir, f"sweep_T{bucket}_" +
                              label.replace(" ", "_").replace("=", ""))
            status, info = _try(label, example, cargs, wd)
            results.append((label, status, info))
    else:
        for bucket in buckets:
            print(f"\n--- Probe T_mel={bucket} ({args.compiler_args}) ---")
            example = torch.zeros((1, mel_dim, bucket), dtype=torch.float32)
            wd = os.path.join(args.workdir, f"T{bucket}")
            status, info = _try(f"T={bucket}", example, args.compiler_args, wd)
            results.append((f"T={bucket}", status, info))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, status, info in results:
        print(f"  {key:<40s}  {status}  {info}")

    passed = [k for k, s, _ in results if s == "PASS"]
    if passed:
        if args.sweep_args:
            print(
                f"\nCompiler-args variants that PASS at T_mel={args.bucket}: "
                f"{passed}\n"
                f"Re-run probe_bigvgan_buckets.py without --sweep-args using "
                f"the chosen --compiler-args to find the new max bucket.\n"
            )
        else:
            print(
                f"\nUsable BigVGAN buckets on this SDK: {passed}\n"
                f"To use these in the speech pipeline, set:\n"
                f"  export QWEN25_OMNI_BIGVGAN_BUCKETS="
                f"{','.join(k.split('=')[-1] for k in passed)}\n"
            )
    else:
        print("\nNothing passed. BigVGAN stays on CPU on this SDK.\n")


if __name__ == "__main__":
    main()
