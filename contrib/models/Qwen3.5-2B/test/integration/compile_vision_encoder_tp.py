#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TP-sharded vision-encoder compile via `parallel_model_trace`.

Unlike compile_vision_encoder.py (single-core torch_neuronx.trace of the
plain-nn.Linear CPUVisionModel), this script traces NeuronQwen35VisionModel
across `--tp` cores. The traced graph shards QKV / MLP linear ops through
NxD's ColumnParallelLinear / RowParallelLinear, gathers or reduces at the
edges to preserve the plain (seq_len, hidden) API.

Output layout (per bucket, per TP rank):
    <out-dir>/tp{TP}_{bucket}/tp_0.pt
    <out-dir>/tp{TP}_{bucket}/tp_1.pt
    ...

Load with `NeuronQwen35VisionModelWrapper.load_compiled_tp(<out-dir>, tp)`.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-2B/test/integration/compile_vision_encoder_tp.py \\
        --model-path /mnt/nvme/models/Qwen3.5-2B \\
        --out-dir    /tmp/qwen35_2b_vl_bench/vision_tp2 \\
        --tp 2 --buckets 4096
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def _load_vconf(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
        full = json.load(f)
    vc = full["vision_config"]
    return SimpleNamespace(
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


def _build_hf_key_map(depth):
    """Map HF safetensor keys -> NeuronQwen35VisionModel state_dict keys."""
    km = {}
    for i in range(depth):
        hf = f"model.visual.blocks.{i}"
        loc = f"blocks.{i}"
        for suf in [
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
            "norm1.weight", "norm1.bias",
            "norm2.weight", "norm2.bias",
        ]:
            km[f"{hf}.{suf}"] = f"{loc}.{suf}"
    km["model.visual.merger.norm.weight"] = "merger_norm.weight"
    km["model.visual.merger.norm.bias"] = "merger_norm.bias"
    km["model.visual.merger.linear_fc1.weight"] = "merger_fc1.weight"
    km["model.visual.merger.linear_fc1.bias"] = "merger_fc1.bias"
    km["model.visual.merger.linear_fc2.weight"] = "merger_fc2.weight"
    km["model.visual.merger.linear_fc2.bias"] = "merger_fc2.bias"
    return km


def load_hf_state_dict(model_path, depth):
    """Load raw HF weights into a flat dict keyed as NeuronQwen35VisionModel."""
    from safetensors import safe_open

    km = _build_hf_key_map(depth)
    state = {}
    for sf_path in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt") as f:
            for hf_key in f.keys():
                if hf_key in km:
                    state[km[hf_key]] = f.get_tensor(hf_key).to(torch.bfloat16)
    return state


def _build_neuron_vision_model(vconf):
    """Return NeuronQwen35VisionModel (uses ColumnParallelLinear / RowParallelLinear)."""
    from src.modeling_qwen35_vision import NeuronQwen35VisionModel
    return NeuronQwen35VisionModel(vconf).to(torch.bfloat16).eval()


# The trace subprocess is spawned fresh and reimports this module, so
# `parallel_model_trace` requires the factory to be picklable. It's called
# once per rank inside its own subprocess; each rank shards HF weights
# in-place via `get_sharded_checkpoint` before load_state_dict. Args flow
# in through env vars (subprocess inherits the parent's environ).
def _picklable_model_factory():
    """Build the model with HF weights sharded for the current TP rank."""
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _root = _os.path.abspath(_os.path.join(_here, "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    model_path = _os.environ["QWEN35_VIT_MODEL_PATH"]
    vconf_json = _os.environ["QWEN35_VIT_VCONF_JSON"]
    vconf = SimpleNamespace(**json.loads(vconf_json))

    m = _build_neuron_vision_model(vconf)
    hf_state = load_hf_state_dict(model_path, vconf.depth)

    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_size,
    )
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_size()
    print(f"[rank {tp_rank}/{tp_size}] loading vision weights", flush=True)

    # NxD-aware in-place sharding of the HF checkpoint dict, using each
    # parameter's partition_dim metadata.
    from neuronx_distributed.trace.trace import get_sharded_checkpoint
    get_sharded_checkpoint(hf_state, m, tp_rank, tp_size)
    missing, unexpected = m.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"[rank {tp_rank}] missing: {len(missing)} keys "
              f"(e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"[rank {tp_rank}] unexpected: {len(unexpected)} keys "
              f"(e.g. {unexpected[:3]})", flush=True)
    return m, {}  # (model, input_output_alias={}) required by parallel_model_trace


def compile_tp(model_path, vconf, tp_degree, bucket_len, out_dir):
    """Compile one bucket with parallel_model_trace at tp_degree."""
    from neuronx_distributed.trace import (
        parallel_model_save,
        parallel_model_trace,
    )

    dtype = torch.bfloat16
    head_dim = vconf.hidden_size // vconf.num_heads

    hidden_states = torch.zeros((bucket_len, vconf.hidden_size), dtype=dtype)
    attention_mask = torch.zeros((1, 1, bucket_len, bucket_len), dtype=dtype)
    cos = torch.zeros((bucket_len, head_dim), dtype=dtype)
    sin = torch.zeros((bucket_len, head_dim), dtype=dtype)

    # Publish factory config through env so the spawned subprocess can rebuild
    # (SimpleNamespace goes through JSON since it's not directly picklable).
    os.environ["QWEN35_VIT_MODEL_PATH"] = model_path
    os.environ["QWEN35_VIT_VCONF_JSON"] = json.dumps(vconf.__dict__)

    print(f"[compile-tp{tp_degree}] bucket={bucket_len}, tracing (per-rank)...")
    t0 = time.perf_counter()
    parallel_model = parallel_model_trace(
        _picklable_model_factory,
        (hidden_states, attention_mask, (cos, sin)),
        tp_degree=tp_degree,
        compiler_workdir=f"/tmp/nxd_vision_tp{tp_degree}_ws_{bucket_len}",
        compiler_args=[
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--enable-mixed-precision-accumulation",
        ],
    )
    dt = time.perf_counter() - t0
    print(f"[compile-tp{tp_degree}] bucket={bucket_len}: done in {dt:.1f}s")

    parallel_model_save(parallel_model, out_dir)
    print(f"[compile-tp{tp_degree}] bucket={bucket_len}: saved → {out_dir}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-2B")
    ap.add_argument("--out-dir", default="/tmp/qwen35_2b_vl_bench/vision_tp2")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--buckets", nargs="+", type=int, default=[4096],
                    help="Patch-token seq lengths to compile.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    vconf = _load_vconf(args.model_path)
    print(f"[build] vconf: depth={vconf.depth} hidden={vconf.hidden_size} "
          f"num_heads={vconf.num_heads} intermediate={vconf.intermediate_size}")

    os.makedirs(args.out_dir, exist_ok=True)
    for bucket in args.buckets:
        bucket_dir = os.path.join(args.out_dir, f"bucket_{bucket}")
        if os.path.exists(bucket_dir) and os.listdir(bucket_dir) and not args.overwrite:
            print(f"[skip] {bucket_dir} not empty (use --overwrite)")
            continue
        os.makedirs(bucket_dir, exist_ok=True)
        compile_tp(args.model_path, vconf, args.tp, bucket, bucket_dir)
        gc.collect()


if __name__ == "__main__":
    main()
