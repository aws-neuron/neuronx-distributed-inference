#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compile the Qwen3.5 vision encoder (CPUVisionModel) to Neuron via
torch_neuronx.trace, one .pt per sequence-length bucket. The compiled
artifacts land at:

    <out-dir>/vision_encoder_<bucket>.pt

They can then be loaded by NeuronQwen35VisionModelWrapper.load_compiled().

Buckets: sequence length AFTER patch_embed + pos_embed. For Qwen3.5:
  spatial_merge_size = 2, temporal_patch_size = 2, patch_size = 16
  H x W image with `image_grid_thw = [1, H//16, W//16]` produces
  H//16 * W//16 patch tokens BEFORE merger. After merger there are
  (H//16 // 2) * (W//16 // 2) merged tokens.

Common image sizes (square):
    512x512  → 1024 patch tokens → 256 merged tokens
    1024x1024→ 4096 patch tokens → 1024 merged tokens
    2048x2048→16384 patch tokens → 4096 merged tokens

We compile at the *patch-token* seq_len (i.e., the input to the ViT blocks).

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-35B-A3B/test/integration/compile_vision_encoder.py \\
        --model-path /mnt/nvme/models/Qwen3.5-35B-A3B \\
        --out-dir    /tmp/qwen35_35b_a3b_vl_bench/vision \\
        --buckets    1024 4096 16384
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def build_cpu_vision(model_path):
    """Load CPUVisionModel with real weights from HF safetensors."""
    from src.modeling_qwen35_vision import CPUVisionModel
    from types import SimpleNamespace

    with open(os.path.join(model_path, "config.json")) as f:
        full = json.load(f)
    vc = full["vision_config"]
    vconf = SimpleNamespace(
        depth=vc["depth"],
        hidden_size=vc["hidden_size"],
        num_heads=vc["num_heads"],
        intermediate_size=vc["intermediate_size"],
        patch_size=vc["patch_size"],
        temporal_patch_size=vc.get("temporal_patch_size", 2),
        spatial_merge_size=vc.get("spatial_merge_size", 2),
        out_hidden_size=vc["out_hidden_size"],
        num_position_embeddings=vc["num_position_embeddings"],
        in_channels=vc.get("in_channels", 3),
    )

    # Reuse the wrapper's weight loader path
    from src.modeling_qwen35_vision import NeuronQwen35VisionModelWrapper
    w = NeuronQwen35VisionModelWrapper(config=vconf, model_cls=None, vision_seq_len_buckets=[16384])
    w.load_cpu_model(model_path)  # populates w._cpu_model with a CPUVisionModel bf16
    return w._cpu_model, vconf


def compile_bucket(cpu_model, vconf, bucket_len, out_path):
    """torch_neuronx.trace the CPUVisionModel at a specific patch-token seq_len."""
    import torch_neuronx

    dtype = torch.bfloat16
    hidden = vconf.hidden_size
    num_heads = vconf.num_heads
    head_dim = hidden // num_heads

    # Example inputs matching the CPUVisionModel.forward signature:
    #   (hidden_states, attention_mask, cos, sin)
    hidden_states = torch.zeros((bucket_len, hidden), dtype=dtype)
    attention_mask = torch.zeros((1, 1, bucket_len, bucket_len), dtype=dtype)
    # cos/sin shape: (seq_len, head_dim//2 doubled to head_dim) — see
    # CPUVisionModel._forward_attention. Actually we pass a shape matching
    # what wrapper produces: (seq_len, head_dim) each, where head_dim entries
    # are rope-emb-cos/sin values (padded via `torch.cat((rot,rot),-1)` in wrapper).
    cos = torch.zeros((bucket_len, head_dim), dtype=dtype)
    sin = torch.zeros((bucket_len, head_dim), dtype=dtype)

    print(f"[compile] bucket={bucket_len}, tracing...")
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        cpu_model,
        (hidden_states, attention_mask, cos, sin),
        compiler_workdir=f"/tmp/nxd_vision_ws_{bucket_len}",
        compiler_args=[
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--enable-mixed-precision-accumulation",
        ],
    )
    dt = time.perf_counter() - t0
    print(f"[compile] bucket={bucket_len}: done in {dt:.1f}s → {out_path}")
    torch.jit.save(traced, out_path)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-35B-A3B")
    ap.add_argument("--out-dir", default="/tmp/qwen35_35b_a3b_vl_bench/vision")
    ap.add_argument("--buckets", nargs="+", type=int, default=[1024, 4096, 16384],
                    help="Patch-token seq lengths (before merger). "
                         "1024→512x512, 4096→1024x1024, 16384→2048x2048.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cpu_model, vconf = build_cpu_vision(args.model_path)
    cpu_model.eval()
    print(f"[build] loaded vision encoder: depth={vconf.depth} "
          f"hidden={vconf.hidden_size} num_heads={vconf.num_heads}")

    for bucket in args.buckets:
        out_path = os.path.join(args.out_dir, f"vision_encoder_{bucket}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to force)")
            continue
        compile_bucket(cpu_model, vconf, bucket, out_path)
        gc.collect()

    print("[done] all buckets compiled")


if __name__ == "__main__":
    main()
