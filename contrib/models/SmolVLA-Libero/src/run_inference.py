"""
Entry point for SmolVLA inference on AWS Trainium.

Usage:
    cd contrib/models/SmolVLA-Libero/src

    # First time only — compile the 3 NEFFs (one-shot, takes ~90s total)
    python run_inference.py --action compile \\
        --hf-checkpoint /path/to/HuggingFaceVLA/smolvla_libero \\
        --neff-dir      /path/to/output_neffs

    # Run inference (load NEFFs + benchmark with synthetic inputs)
    python run_inference.py --action run \\
        --hf-checkpoint /path/to/HuggingFaceVLA/smolvla_libero \\
        --neff-dir      /path/to/output_neffs

What runs where:
    On Neuron (3 compiled NEFFs):
        1. SigLIP vision encoder (12 layers) + connector + scaling
        2. VLM prefix decoder (16 SmolLM layers, returns full KV cache)
        3. Per-step action expert denoiser (16 expert layers, interleaved
           self/cross attn over the cached prefix KV) — including the
           sinusoidal timestep embedding

    On CPU (necessarily; flagged as deviations from "everything on Neuron"):
        - The 10-step Euler loop (Python control flow, can't fuse into
          a single static-shape NEFF without unrolling 10x larger graphs)
        - Tokenization and image preprocessing (data loading, not model
          compute — moving them to Neuron has no perf benefit)

Hardware constraints (also flagged):
    - tp_degree=1 because num_attention_heads=15, num_kv_heads=5 — neither
      divides cleanly into the 4 Neuron cores on trn3pd98.3xlarge.
      Production NxDI parallel primitives (ColumnParallelLinear /
      RowParallelLinear / ParallelEmbedding) are still used so the code is
      portable to instances with more divisor-friendly head counts. On this
      instance, 3 of 4 cores idle. The model (450M params, ~900 MB BF16)
      fits in one core's HBM with vast headroom.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch

# Make sibling modules importable when running as a script:
#     ``python run_inference.py ...``  from inside ``src/``
# or  ``python contrib/models/SmolVLA-Libero/src/run_inference.py ...``
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_constants as C
from modeling_smolvla import SmolVLAPolicy

logger = logging.getLogger("smolvla.run_inference")


def _make_dummy_inputs(batch_size: int = 1):
    """Synthetic inputs — same shapes the real policy expects."""
    images = [
        torch.randn(batch_size, 3, C.VISION_IMAGE_SIZE, C.VISION_IMAGE_SIZE,
                    dtype=torch.bfloat16)
        for _ in range(C.NUM_CAMERAS)
    ]
    lang = torch.randint(
        0, C.VLM_VOCAB_SIZE,
        (batch_size, C.NUM_TEXT_TOKENS),
        dtype=torch.int32,
    )
    state = torch.zeros(batch_size, C.MAX_STATE_DIM, dtype=torch.float32)
    return images, lang, state


def cmd_compile(args):
    policy = SmolVLAPolicy(
        hf_checkpoint_dir=args.hf_checkpoint,
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
    )
    t0 = time.monotonic()
    policy.compile(args.neff_dir)
    print(f"All 3 NEFFs compiled in {time.monotonic()-t0:.1f}s -> {args.neff_dir}")


def cmd_run(args):
    policy = SmolVLAPolicy(
        hf_checkpoint_dir=args.hf_checkpoint,
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
    )
    print("Loading 3 NEFFs to Neuron...")
    t0 = time.monotonic()
    policy.load(args.neff_dir)
    print(f"Loaded in {time.monotonic()-t0:.1f}s")

    images, lang, state = _make_dummy_inputs(args.batch_size)

    print("Cold inference (first call includes lazy device init)...")
    t0 = time.monotonic()
    chunk = policy.generate(images, lang, state, num_steps=args.num_steps)
    cold_ms = (time.monotonic() - t0) * 1000
    print(
        f"  cold: {cold_ms:.1f} ms  shape={tuple(chunk.shape)}  "
        f"hasNaN={torch.isnan(chunk).any().item()}  "
        f"mean={chunk.mean().item():.4f}  std={chunk.std().item():.4f}"
    )

    print(f"Warm benchmark — {args.bench_iters} iterations:")
    timings = []
    for i in range(args.bench_iters):
        t0 = time.monotonic()
        chunk = policy.generate(images, lang, state, num_steps=args.num_steps)
        timings.append((time.monotonic() - t0) * 1000)
    timings.sort()
    p50 = timings[len(timings) // 2]
    p99 = timings[int(len(timings) * 0.99)]
    print(f"  p50={p50:.1f} ms  p99={p99:.1f} ms  min={timings[0]:.1f} ms  max={timings[-1]:.1f} ms")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SmolVLA Trainium inference")
    parser.add_argument("--action", choices=["compile", "run"], required=True)
    parser.add_argument("--hf-checkpoint", required=True,
                        help="Path to HuggingFaceVLA/smolvla_libero HF snapshot dir")
    parser.add_argument("--neff-dir", required=True,
                        help="Where to save (or load) the 3 compiled NEFFs")
    parser.add_argument("--tp-degree", type=int, default=C.DEFAULT_TP_DEGREE)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--num-steps", type=int, default=C.NUM_DENOISE_STEPS,
                        help="Number of Euler steps in the denoising loop")
    parser.add_argument("--bench-iters", type=int, default=20)
    args = parser.parse_args()

    if args.action == "compile":
        cmd_compile(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
