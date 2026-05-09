#!/usr/bin/env python3
"""
LTX-2.3 Unified Weight Sharding Script
========================================
Pre-shards model weights for fast per-rank loading on Neuron:
  - backbone: DiT transformer weights (~9.3 GB/rank)
  - encoder: Gemma3 12B text encoder weights (~5.9 GB/rank)

Pre-sharding avoids loading the full unsharded checkpoint at generation
time, which would cause memory pressure on trn2.3xlarge (124 GB RAM).

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

  # Shard DiT backbone weights
  python3 shard_weights.py backbone \\
    --model-path ./models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \\
    --output-dir ./backbone_sharded

  # Shard Gemma3 encoder weights
  python3 shard_weights.py encoder \\
    --gemma-path ./models/gemma-3-12b \\
    --output-dir ./gemma3_encoder_sharded
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


def _check_file_exists(path, description="file"):
    """Exit with error if a required file/directory does not exist."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: {description} not found: {path}")


def shard_weight(full_weight, jit_param_name, tp_rank, tp_size):
    """Shard a full weight tensor for the given TP rank.

    Determines the sharding pattern from the JIT parameter name:
    - ColumnParallelLinear (to_q, to_k, to_v, to_gate_logits, GEGLU proj):
        weight sharded on dim 0, bias sharded on dim 0
    - RowParallelLinear (to_out->0, ff->net->2):
        weight sharded on dim 1, bias NOT sharded
    - DistributedRMSNorm (q_norm, k_norm):
        weight sharded on dim 0
    - SPMDRank: rank tensor, select element for this rank
    - Unsharded (scale_shift_table, norm_out, proj_out, etc.):
        return full weight unchanged
    """
    name = jit_param_name

    # SPMDRank: select this rank's value
    if "spmd_rank" in name:
        return torch.tensor([tp_rank], dtype=torch.int32)

    # Check if this is a sharded parameter
    is_column_weight = False
    is_column_bias = False
    is_row_weight = False
    is_row_bias = False
    is_norm_weight = False

    # Column-parallel: to_q, to_k, to_v, to_gate_logits
    # Use delimited patterns (->X->) to avoid false matches like
    # "to_v" matching "audio_to_video_attn"
    for col_name in ["->to_q->", "->to_k->", "->to_v->", "->to_gate_logits->"]:
        if col_name in name:
            if name.endswith("weight"):
                is_column_weight = True
            elif name.endswith("bias"):
                is_column_bias = True
            break

    # Column-parallel: GEGLU gate proj (ff->net->0->proj)
    if "ff->net->0->proj" in name:
        if name.endswith("weight"):
            is_column_weight = True
        elif name.endswith("bias"):
            is_column_bias = True

    # Row-parallel: output projection (to_out->0)
    if "to_out->0" in name:
        if name.endswith("weight"):
            is_row_weight = True
        elif name.endswith("bias"):
            is_row_bias = True  # Bias not sharded for RowParallel

    # Row-parallel: FFN down projection (ff->net->2)
    if "ff->net->2" in name:
        if name.endswith("weight"):
            is_row_weight = True
        elif name.endswith("bias"):
            is_row_bias = True  # Bias not sharded

    # DistributedRMSNorm: q_norm, k_norm
    if ("q_norm" in name or "k_norm" in name) and name.endswith("weight"):
        is_norm_weight = True

    # Apply sharding
    if is_column_weight or is_column_bias or is_norm_weight:
        # Shard on dim 0
        shard_size = full_weight.shape[0] // tp_size
        return full_weight[shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()

    elif is_row_weight:
        # Shard on dim 1
        shard_size = full_weight.shape[1] // tp_size
        return full_weight[:, shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()

    elif is_row_bias:
        # Not sharded — full copy
        return full_weight.clone()

    else:
        # Unsharded (scale_shift_table, norm_out, proj_out, audio variants)
        return full_weight.clone()


# ============================================================================
# Backbone weight sharding
# ============================================================================


def shard_backbone(args):
    """Pre-shard LTX-2.3 DiT backbone weights for Neuron TP."""
    from safetensors import safe_open

    _check_file_exists(args.model_path, "model safetensors")
    tp_degree = args.tp_degree

    print("=" * 60)
    print(f"Pre-shard LTX-2.3 DiT backbone weights (TP={tp_degree})")
    print("=" * 60)

    # 1. Load backbone weights from safetensors
    print(f"\n[1/3] Loading backbone weights from {args.model_path}...")
    t0 = time.time()

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
    backbone_sd["spmd_rank.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    total_bytes = sum(v.numel() * v.element_size() for v in backbone_sd.values())
    print(f"  Backbone keys: {len(backbone_sd)} ({total_bytes / 1e9:.2f} GB)")
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. Convert keys and shard per rank
    print("\n[2/3] Sharding and saving per-rank checkpoints...")
    os.makedirs(args.output_dir, exist_ok=True)

    def sf_key_to_jit_key(sf_key):
        return "weights." + sf_key.replace(".", "->")

    for rank in range(tp_degree):
        t0 = time.time()
        rank_sd = {}
        for sf_key, full_weight in backbone_sd.items():
            jit_key = sf_key_to_jit_key(sf_key)
            sharded = shard_weight(full_weight, jit_key, rank, tp_degree)
            rank_sd[jit_key] = sharded.contiguous().clone()

        rank_path = os.path.join(args.output_dir, f"rank_{rank}.pt")
        torch.save(rank_sd, rank_path)
        size_gb = os.path.getsize(rank_path) / 1e9
        elapsed = time.time() - t0
        print(
            f"  rank_{rank}.pt: {len(rank_sd)} keys, {size_gb:.2f} GB, {elapsed:.1f}s"
        )
        del rank_sd
        gc.collect()

    del backbone_sd
    gc.collect()

    # 3. Verify
    print("\n[3/3] Verification...")
    total_size = 0
    for rank in range(tp_degree):
        rank_path = os.path.join(args.output_dir, f"rank_{rank}.pt")
        size = os.path.getsize(rank_path)
        total_size += size
        ckpt = torch.load(rank_path, weights_only=True)
        print(f"  rank_{rank}.pt: {len(ckpt)} keys, {size / 1e9:.2f} GB")
        if rank == 0:
            for k in sorted(ckpt.keys())[:3]:
                print(f"    {k}: {tuple(ckpt[k].shape)} {ckpt[k].dtype}")
        del ckpt

    print(f"\n  Total sharded size: {total_size / 1e9:.2f} GB")
    print(f"  Output dir: {args.output_dir}")
    print("\nDone!")


# ============================================================================
# Gemma3 encoder weight sharding
# ============================================================================


def shard_encoder(args):
    """Pre-shard Gemma3 12B encoder weights for Neuron TP."""
    from modeling_gemma3_encoder import (
        Gemma3TextEncoderModel,
        convert_hf_gemma3_to_encoder_state_dict,
    )
    from safetensors.torch import load_file
    from neuronx_distributed.trace.trace import (
        _mock_parallel_state,
        init_on_device,
        get_sharded_checkpoint,
    )

    _check_file_exists(args.gemma_path, "Gemma 3 model directory")
    tp_degree = args.tp_degree

    print("=" * 60)
    print(f"Pre-shard Gemma 3-12B encoder weights (TP={tp_degree})")
    print("=" * 60)

    def get_model_fn():
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

    # 1. Load HF weights
    print(f"\n[1/3] Loading HF weights from {args.gemma_path}...")
    t0 = time.time()

    st_files = sorted(glob.glob(os.path.join(args.gemma_path, "model-*.safetensors")))
    if not st_files:
        st_files = sorted(glob.glob(os.path.join(args.gemma_path, "*.safetensors")))
    if not st_files:
        print(f"ERROR: No safetensors files found in {args.gemma_path}")
        sys.exit(1)

    print(f"  Loading {len(st_files)} safetensors files...")
    hf_state_dict = {}
    for f in st_files:
        shard = load_file(f)
        hf_state_dict.update(shard)
    print(f"  Total HF keys: {len(hf_state_dict)}")

    encoder_state_dict = convert_hf_gemma3_to_encoder_state_dict(hf_state_dict)
    del hf_state_dict
    gc.collect()

    total_bytes = sum(v.numel() * v.element_size() for v in encoder_state_dict.values())
    print(f"  Encoder keys: {len(encoder_state_dict)} ({total_bytes / 1e9:.2f} GB)")
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. Shard per rank and save
    print("\n[2/3] Sharding and saving per-rank checkpoints...")
    os.makedirs(args.output_dir, exist_ok=True)

    for rank in range(tp_degree):
        t0 = time.time()
        ckpt = {k: v.clone() for k, v in encoder_state_dict.items()}

        _mock_parallel_state(tp_degree, rank)
        with init_on_device(torch.device("meta")):
            model, _ = get_model_fn()
        get_sharded_checkpoint(ckpt, model, rank, tp_degree)

        # Force contiguous clones to avoid serializing full unsharded storage
        ckpt = {k: v.contiguous().clone() for k, v in ckpt.items()}

        rank_path = os.path.join(args.output_dir, f"rank_{rank}.pt")
        torch.save(ckpt, rank_path)
        size_gb = os.path.getsize(rank_path) / 1e9
        elapsed = time.time() - t0
        print(f"  rank_{rank}.pt: {len(ckpt)} keys, {size_gb:.2f} GB, {elapsed:.1f}s")
        del ckpt, model
        gc.collect()

    del encoder_state_dict
    gc.collect()

    # 3. Verify
    print("\n[3/3] Verification...")
    total_size = 0
    for rank in range(tp_degree):
        rank_path = os.path.join(args.output_dir, f"rank_{rank}.pt")
        size = os.path.getsize(rank_path)
        total_size += size
        ckpt = torch.load(rank_path, weights_only=True)
        print(f"  rank_{rank}.pt: {len(ckpt)} keys, {size / 1e9:.2f} GB")
        if rank == 0:
            for k in sorted(ckpt.keys())[:3]:
                print(f"    {k}: {tuple(ckpt[k].shape)} {ckpt[k].dtype}")
        del ckpt

    print(f"\n  Total sharded size: {total_size / 1e9:.2f} GB")
    print(f"  Output dir: {args.output_dir}")
    print("\nDone!")


# ============================================================================
# Main entry point with subcommands
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2.3 unified weight sharding script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shard DiT backbone weights
  python3 shard_weights.py backbone \\
    --model-path ./models/LTX-2.3/ltx-2.3-22b-distilled.safetensors

  # Shard Gemma3 encoder weights
  python3 shard_weights.py encoder \\
    --gemma-path ./models/gemma-3-12b
""",
    )
    subparsers = parser.add_subparsers(dest="component", help="Component to shard")
    subparsers.required = True

    # --- backbone subcommand ---
    p_bb = subparsers.add_parser("backbone", help="Shard DiT backbone weights")
    p_bb.add_argument(
        "--model-path",
        required=True,
        help="Path to LTX-2.3 safetensors file",
    )
    p_bb.add_argument(
        "--output-dir",
        default="./backbone_sharded",
        help="Output directory for sharded weights (default: ./backbone_sharded)",
    )
    p_bb.add_argument("--tp-degree", type=int, default=4, help="TP degree")
    p_bb.set_defaults(func=shard_backbone)

    # --- encoder subcommand ---
    p_enc = subparsers.add_parser("encoder", help="Shard Gemma3 encoder weights")
    p_enc.add_argument(
        "--gemma-path",
        required=True,
        help="Path to HuggingFace Gemma 3 model directory",
    )
    p_enc.add_argument(
        "--output-dir",
        default="./gemma3_encoder_sharded",
        help="Output directory for sharded weights (default: ./gemma3_encoder_sharded)",
    )
    p_enc.add_argument("--tp-degree", type=int, default=4, help="TP degree")
    p_enc.set_defaults(func=shard_encoder)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
