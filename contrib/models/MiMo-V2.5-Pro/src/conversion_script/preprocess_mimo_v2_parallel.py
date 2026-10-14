#!/usr/bin/env python3
"""Parallel wrapper around preprocess_mimo_v2_fp8.process_layer.

Each layer is independent: `process_layer(L, lazy, config, ...)` reads only
keys under `model.layers.{L}.*` from the HF shards and returns the Neuron
layer-shard dict. With 70 layers and per-MoE-layer cost ~60s serial, 4-8
workers cuts wallclock from ~70 min to ~15-20 min (I/O + CPU FP8 math).

Each worker opens its own LazyWeightMap so there's no shared safetensors
handle. Output dir is a CLI arg so it can write to a clean path without
touching the serial run's output.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

# Resolve the sibling single-layer preprocess module. This file lives at
# .../MiMo-V2.5-Pro/src/conversion_script/preprocess_mimo_v2_parallel.py,
# so the importable parent is two levels up.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from conversion_script.preprocess_mimo_v2_fp8 import (  # noqa: E402
    LazyWeightMap,
    process_layer,
    save_shard,
)
from safetensors.torch import save_file  # noqa: E402


def _worker(task):
    layer_idx, hf_model_path, save_path, config = task
    hybrid = config.get(
        "hybrid_layer_pattern", [0] * config["num_hidden_layers"]
    )
    moe_freq = config.get("moe_layer_freq", [1] * config["num_hidden_layers"])
    is_dense = moe_freq[layer_idx] == 0
    is_swa = hybrid[layer_idx] == 1

    with open(
        os.path.join(hf_model_path, "model.safetensors.index.json")
    ) as fh:
        weight_map_in = json.load(fh)["weight_map"]
    lazy = LazyWeightMap(hf_model_path, weight_map_in)
    try:
        t0 = time.time()
        layer_sd = process_layer(
            layer_idx, lazy, config, is_dense=is_dense, is_swa=is_swa
        )
        filename = f"model_layer{layer_idx}.safetensors"
        path = os.path.join(save_path, filename)
        materialized = {}
        total_bytes = 0
        for k, v in layer_sd.items():
            if not v.is_contiguous():
                v = v.contiguous()
            v = v.detach().clone()
            materialized[k] = v
            total_bytes += v.numel() * v.element_size()
        save_file(materialized, path)
        keys = list(materialized.keys())
        del materialized, layer_sd
        gc.collect()
        elapsed = time.time() - t0
    finally:
        lazy.close()
    return layer_idx, is_dense, is_swa, keys, filename, total_bytes, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_path", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--tp_degree", type=int, default=64)
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("N_WORKERS", "12")),
    )
    args = p.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.hf_model_path, "config.json")) as fh:
        config = json.load(fh)
    num_layers = config["num_hidden_layers"]

    print(
        f"[par] {num_layers} layers x {args.workers} workers -> {args.save_path}",
        flush=True,
    )
    tasks = [
        (L, args.hf_model_path, args.save_path, config) for L in range(num_layers)
    ]
    weight_map_out = {}
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        done = 0
        for li, is_dense, is_swa, keys, filename, total_bytes, elapsed in pool.imap_unordered(
            _worker, tasks
        ):
            done += 1
            for k in keys:
                weight_map_out[k] = filename
            tag = "dense" if is_dense else "moe"
            attn = "swa" if is_swa else "full"
            print(
                f"  [{done:2d}/{num_layers}] layer {li:2d} [{tag:5s} {attn:4s}] "
                f"{total_bytes/1e9:6.2f} GB in {elapsed:5.1f}s "
                f"(wall {time.time()-t_start:5.1f}s)",
                flush=True,
            )

    print(
        f"[par] all {num_layers} layers done in {time.time()-t_start:.1f}s",
        flush=True,
    )

    print("[par] processing embed_tokens / norm / lm_head ...", flush=True)
    with open(
        os.path.join(args.hf_model_path, "model.safetensors.index.json")
    ) as fh:
        weight_map_in = json.load(fh)["weight_map"]
    lazy = LazyWeightMap(args.hf_model_path, weight_map_in)
    extras = {}
    try:
        for src, dst in (
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.norm.weight", "norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ):
            t = lazy.get(src)
            if t is not None:
                extras[dst] = t.detach().clone()
            else:
                print(f"  WARN: missing {src}", flush=True)
        if "lm_head.weight" not in extras and "embed_tokens.weight" in extras:
            extras["lm_head.weight"] = extras["embed_tokens.weight"].detach().clone()
    finally:
        lazy.close()
    save_shard(extras, args.save_path, "model_extras.safetensors", weight_map_out)
    del extras

    total_size = 0
    for f in set(weight_map_out.values()):
        total_size += os.path.getsize(os.path.join(args.save_path, f))
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(
        os.path.join(args.save_path, "model.safetensors.index.json"), "w"
    ) as fh:
        json.dump(index, fh, indent=2)

    for name in sorted(os.listdir(args.hf_model_path)):
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        src = os.path.join(args.hf_model_path, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.save_path, name))

    print(
        f"\n[par] DONE. total_size={total_size/1e9:.2f} GB "
        f"tensors={len(weight_map_out)} -> {args.save_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
