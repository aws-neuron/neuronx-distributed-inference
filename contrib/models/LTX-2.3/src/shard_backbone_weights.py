#!/usr/bin/env python3
"""
Pre-shard LTX-2.3 DiT backbone weights for Neuron TP=4.

Reads the full 41GB safetensors file once, extracts backbone keys,
shards them per TP rank, and saves compact per-rank .pt files (~5GB each).

This avoids loading the full 41GB file during generation, which would cause
memory pressure and swap thrashing on trn2.3xlarge (124GB RAM).

Output structure:
  backbone_sharded/
    rank_0.pt   (~5 GB)
    rank_1.pt
    rank_2.pt
    rank_3.pt

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  python3 shard_backbone_weights.py \
    --model-path /home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
    --output-dir /home/ubuntu/backbone_sharded
"""

import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TP_DEGREE = 4
MODEL_PATH = "/home/ubuntu/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"


def main():
    parser = argparse.ArgumentParser(description="Pre-shard LTX-2.3 backbone weights")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to LTX-2.3 safetensors file",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/backbone_sharded",
        help="Output directory for sharded weights",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=TP_DEGREE,
        help="Tensor parallelism degree",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pre-shard LTX-2.3 DiT backbone weights (TP=%d)" % args.tp_degree)
    print("=" * 60)

    # 1. Load backbone weights from safetensors (memory-mapped for efficiency)
    print("\n[1/3] Loading backbone weights from %s..." % args.model_path)
    t0 = time.time()

    from safetensors import safe_open
    from load_with_weights import shard_weight

    prefix = "model.diffusion_model."
    backbone_prefixes = (
        "transformer_blocks.",
        "norm_out.",
        "proj_out.",
        "scale_shift_table",
        "audio_norm_out.",
        "audio_proj_out.",
        "audio_scale_shift_table",
    )

    backbone_sd = {}
    with safe_open(args.model_path, framework="pt") as f:
        all_keys = list(f.keys())
        for k in all_keys:
            stripped = k[len(prefix) :] if k.startswith(prefix) else k
            if stripped.startswith(backbone_prefixes):
                backbone_sd[stripped] = f.get_tensor(k).to(torch.bfloat16).contiguous()

    # Add SPMDRank tensor
    backbone_sd["spmd_rank.rank"] = torch.arange(0, args.tp_degree, dtype=torch.int32)

    total_bytes = sum(v.numel() * v.element_size() for v in backbone_sd.values())
    print("  Backbone keys: %d (%.2f GB)" % (len(backbone_sd), total_bytes / 1e9))
    print("  Done in %.1fs" % (time.time() - t0))

    # 2. Convert safetensors key format to JIT param format and shard
    print("\n[2/3] Sharding and saving per-rank checkpoints...")
    os.makedirs(args.output_dir, exist_ok=True)

    def sf_key_to_jit_key(sf_key):
        return "weights." + sf_key.replace(".", "->")

    for rank in range(args.tp_degree):
        t0 = time.time()
        rank_sd = {}
        for sf_key, full_weight in backbone_sd.items():
            jit_key = sf_key_to_jit_key(sf_key)
            sharded = shard_weight(full_weight, jit_key, rank, args.tp_degree)
            # CRITICAL: .contiguous().clone() so torch.save doesn't serialize
            # the full unsharded storage backing sliced/narrowed tensors.
            rank_sd[jit_key] = sharded.contiguous().clone()

        rank_path = os.path.join(args.output_dir, "rank_%d.pt" % rank)
        torch.save(rank_sd, rank_path)
        size_gb = os.path.getsize(rank_path) / 1e9
        elapsed = time.time() - t0
        print(
            "  rank_%d.pt: %d keys, %.2f GB, %.1fs"
            % (rank, len(rank_sd), size_gb, elapsed)
        )
        del rank_sd
        gc.collect()

    del backbone_sd
    gc.collect()

    # 3. Verify
    print("\n[3/3] Verification...")
    total_size = 0
    for rank in range(args.tp_degree):
        rank_path = os.path.join(args.output_dir, "rank_%d.pt" % rank)
        size = os.path.getsize(rank_path)
        total_size += size
        ckpt = torch.load(rank_path, weights_only=True)
        print("  rank_%d.pt: %d keys, %.2f GB" % (rank, len(ckpt), size / 1e9))
        if rank == 0:
            for k in sorted(ckpt.keys())[:3]:
                print("    %s: %s %s" % (k, tuple(ckpt[k].shape), ckpt[k].dtype))
        del ckpt

    print("\n  Total sharded size: %.2f GB" % (total_size / 1e9))
    print("  Output dir: %s" % args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
