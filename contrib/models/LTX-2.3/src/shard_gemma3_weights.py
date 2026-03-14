#!/usr/bin/env python3
"""
Pre-shard Gemma 3-12B encoder weights for Neuron TP=4.

Produces per-rank checkpoint files that can be loaded directly with
replace_weights() -- no cloning or sharding at load time.

Output structure:
  gemma3_encoder_sharded/
    rank_0.pt   (~5.9 GB)
    rank_1.pt
    rank_2.pt
    rank_3.pt

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  python3 shard_gemma3_weights.py \
    --gemma-path /home/ubuntu/models/gemma-3-12b \
    --output-dir /home/ubuntu/gemma3_encoder_sharded
"""

import argparse
import gc
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TP_DEGREE = 4


def get_model_fn():
    """Create Gemma3 encoder model with TP layers (for sharding metadata)."""
    from modeling_gemma3_encoder import Gemma3TextEncoderModel

    model = Gemma3TextEncoderModel(
        vocab_size=262208,
        hidden_size=3840,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        intermediate_size=15360,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=131072,
        query_pre_attn_scalar=256,
        pad_token_id=0,
        dtype=torch.bfloat16,
    )
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, None


def main():
    parser = argparse.ArgumentParser(description="Pre-shard Gemma3 encoder weights")
    parser.add_argument(
        "--gemma-path",
        default="/home/ubuntu/models/gemma-3-12b",
        help="Path to HuggingFace Gemma 3 model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/gemma3_encoder_sharded",
        help="Output directory for sharded weights",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pre-shard Gemma 3-12B encoder weights (TP=%d)" % TP_DEGREE)
    print("=" * 60)

    # 1. Load HF weights from safetensors
    print("\n[1/3] Loading HF weights from %s..." % args.gemma_path)
    t0 = time.time()

    from modeling_gemma3_encoder import convert_hf_gemma3_to_encoder_state_dict
    from safetensors.torch import load_file

    # Find safetensors files in the model directory
    st_files = sorted(glob.glob(os.path.join(args.gemma_path, "model-*.safetensors")))
    if not st_files:
        st_files = sorted(glob.glob(os.path.join(args.gemma_path, "*.safetensors")))
    if not st_files:
        print("ERROR: No safetensors files found in %s" % args.gemma_path)
        sys.exit(1)

    print("  Loading %d safetensors files..." % len(st_files))
    hf_state_dict = {}
    for f in st_files:
        shard = load_file(f)
        hf_state_dict.update(shard)
    print("  Total HF keys: %d" % len(hf_state_dict))

    encoder_state_dict = convert_hf_gemma3_to_encoder_state_dict(hf_state_dict)
    del hf_state_dict
    gc.collect()

    total_bytes = sum(v.numel() * v.element_size() for v in encoder_state_dict.values())
    print("  Encoder keys: %d (%.2f GB)" % (len(encoder_state_dict), total_bytes / 1e9))
    print("  Done in %.1fs" % (time.time() - t0))

    # 2. Shard per rank and save
    print("\n[2/3] Sharding and saving per-rank checkpoints...")
    os.makedirs(args.output_dir, exist_ok=True)

    from neuronx_distributed.trace.trace import (
        _mock_parallel_state,
        init_on_device,
        get_sharded_checkpoint,
    )

    for rank in range(TP_DEGREE):
        t0 = time.time()
        ckpt = {k: v.clone() for k, v in encoder_state_dict.items()}

        _mock_parallel_state(TP_DEGREE, rank)
        with init_on_device(torch.device("meta")):
            model, _ = get_model_fn()
        get_sharded_checkpoint(ckpt, model, rank, TP_DEGREE)

        # CRITICAL: Force contiguous clones so torch.save doesn't serialize
        # the full unsharded storage backing sliced/narrowed tensors.
        # Without this, each rank file would be ~24 GB instead of ~6 GB.
        ckpt = {k: v.contiguous().clone() for k, v in ckpt.items()}

        rank_path = os.path.join(args.output_dir, "rank_%d.pt" % rank)
        torch.save(ckpt, rank_path)
        size_gb = os.path.getsize(rank_path) / 1e9
        num_keys = len(ckpt)
        elapsed = time.time() - t0
        print(
            "  rank_%d.pt: %d keys, %.2f GB, %.1fs" % (rank, num_keys, size_gb, elapsed)
        )
        del ckpt, model
        gc.collect()

    del encoder_state_dict
    gc.collect()

    # 3. Verify
    print("\n[3/3] Verification...")
    total_size = 0
    for rank in range(TP_DEGREE):
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
