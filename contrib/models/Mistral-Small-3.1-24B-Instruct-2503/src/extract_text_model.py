#!/usr/bin/env python3
"""
Ministral 3B: Extract text-only backbone weights from multimodal checkpoint.

The HuggingFace checkpoint `mistralai/Ministral-3-3B-Instruct-2512-BF16` is a
`Mistral3ForConditionalGeneration` model (multimodal: Pixtral vision encoder +
text decoder). NxDI/vLLM can only serve the text-only decoder.

This script:
  1. Reads safetensors weights directly (no model class needed)
  2. Remaps keys: strips `model.language_model.` prefix
  3. Moves lm_head.weight from top-level to the text model
  4. Creates a MistralForCausalLM-compatible config.json
  5. Copies tokenizer files

Output: A directory loadable by vLLM as a standard MistralForCausalLM.

Usage:
    python extract_text_model.py [--src /path/to/full/model] [--dst /path/to/output]

Requires only: safetensors, torch (no transformers version constraint)
"""

import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path


def extract(src_dir, dst_dir):
    print("=" * 60)
    print("Ministral 3B: Extract text-only backbone")
    print("=" * 60)
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    # Load source config
    with open(os.path.join(src_dir, "config.json")) as f:
        full_config = json.load(f)

    # Extract text_config from the multimodal config
    text_config = full_config.get("text_config", {})
    if not text_config:
        print("  ERROR: no text_config found in source config.json")
        return

    print(f"\n[1/4] Reading weights from safetensors...")
    t0 = time.time()

    # Load safetensors index
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    single_path = os.path.join(src_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
    elif os.path.exists(single_path):
        # Single file model
        weight_map = None
    else:
        print("  ERROR: no model.safetensors or index found")
        return

    # Collect text-only weights
    # Key patterns in the multimodal checkpoint:
    #   language_model.model.layers.N.* -> model.layers.N.*
    #   language_model.model.embed_tokens.weight -> model.embed_tokens.weight
    #   language_model.model.norm.weight -> model.norm.weight
    #   vision_tower.* -> SKIP
    #   multi_modal_projector.* -> SKIP
    # Note: tie_word_embeddings=True, so no lm_head.weight in the checkpoint

    text_weights = {}
    skipped_count = 0
    text_prefix = "language_model."

    if weight_map:
        # Multi-shard model
        file_keys = {}
        for key, fname in weight_map.items():
            if fname not in file_keys:
                file_keys[fname] = []
            file_keys[fname].append(key)

        for fname in sorted(file_keys.keys()):
            keys = file_keys[fname]
            fpath = os.path.join(src_dir, fname)
            f = safe_open(fpath, framework="pt")

            for key in keys:
                if key.startswith(text_prefix):
                    # Strip "language_model." prefix, keeping "model." prefix
                    new_key = key[len(text_prefix) :]
                    tensor = f.get_tensor(key)
                    text_weights[new_key] = tensor
                else:
                    skipped_count += 1
    else:
        # Single file
        f = safe_open(single_path, framework="pt")
        for key in f.keys():
            if key.startswith(text_prefix):
                new_key = key[len(text_prefix) :]
                tensor = f.get_tensor(key)
                text_weights[new_key] = tensor
            else:
                skipped_count += 1

    print(
        f"  Extracted {len(text_weights)} text weights, skipped {skipped_count} vision/projector weights"
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # Save weights
    print(f"\n[2/4] Saving text-only weights...")
    t0 = time.time()

    # Save as a single safetensors file (3.3B model is small enough)
    total_bytes = sum(t.numel() * t.element_size() for t in text_weights.values())
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")

    if total_bytes > 10e9:
        # Multi-shard (unlikely for 3.3B)
        shard_idx = 0
        current_shard = {}
        current_size = 0
        MAX_SHARD = 5e9
        new_weight_map = {}

        def flush():
            nonlocal shard_idx, current_shard, current_size
            if not current_shard:
                return
            shard_idx += 1
            sname = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file(current_shard, os.path.join(dst_dir, sname))
            for k in current_shard:
                new_weight_map[k] = sname
            current_shard = {}
            current_size = 0

        for k in sorted(text_weights.keys()):
            t = text_weights[k]
            sz = t.numel() * t.element_size()
            if current_size + sz > MAX_SHARD and current_shard:
                flush()
            current_shard[k] = t
            current_size += sz
        flush()

        # Rename shards
        total_shards = shard_idx
        final_map = {}
        for k, sname in new_weight_map.items():
            final = sname.replace("PLACEHOLDER", f"{total_shards:05d}")
            if sname != final:
                old = os.path.join(dst_dir, sname)
                new = os.path.join(dst_dir, final)
                if os.path.exists(old):
                    os.rename(old, new)
            final_map[k] = final

        with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": final_map}, f, indent=2)
    else:
        # Single file
        save_file(text_weights, os.path.join(dst_dir, "model.safetensors"))

    print(f"  Saved in {time.time() - t0:.1f}s")

    # Create LlamaForCausalLM-compatible config
    # IMPORTANT: Must use LlamaForCausalLM, NOT MistralForCausalLM.
    # The Mistral NxDI code path has a hardcoded head_dim computation that
    # breaks for models where head_dim != hidden_size/num_attention_heads.
    # The Llama code path handles this correctly via getattr(config, "head_dim", ...).
    print(f"\n[3/4] Creating config.json...")

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "hidden_size": text_config["hidden_size"],
        "intermediate_size": text_config["intermediate_size"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config.get(
            "num_key_value_heads", text_config["num_attention_heads"]
        ),
        "head_dim": text_config.get("head_dim", 128),
        "vocab_size": text_config["vocab_size"],
        "max_position_embeddings": text_config.get("max_position_embeddings", 131072),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1e-5),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "bos_token_id": text_config.get("bos_token_id", 1),
        "eos_token_id": text_config.get("eos_token_id", 2),
        # rope_theta: extract from rope_parameters if not in text_config directly
        "rope_theta": text_config.get("rope_theta")
        or text_config.get("rope_parameters", {}).get("rope_theta", 1000000.0),
        # Do NOT include sliding_window -- causes NxDI tensor shape issues
    }

    # Print config summary
    computed_head_dim = config["hidden_size"] // config["num_attention_heads"]
    print(f"  model_type: {config['model_type']}")
    print(f"  architectures: {config['architectures']}")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  num_hidden_layers: {config['num_hidden_layers']}")
    print(f"  num_attention_heads: {config['num_attention_heads']}")
    print(f"  num_key_value_heads: {config['num_key_value_heads']}")
    print(
        f"  head_dim: {config['head_dim']} (config) vs {computed_head_dim} (computed)"
    )
    print(f"  intermediate_size: {config['intermediate_size']}")
    print(f"  vocab_size: {config['vocab_size']}")
    print(f"  rms_norm_eps: {config['rms_norm_eps']}")

    if config["head_dim"] != computed_head_dim:
        print(
            f"  NOTE: head_dim ({config['head_dim']}) != hidden_size/num_heads ({computed_head_dim})"
        )
        print(
            f"  This is intentional -- the model uses head_dim=128 with hidden_size=3072"
        )

    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer
    print(f"\n[4/4] Copying tokenizer files...")
    for fname in os.listdir(src_dir):
        if (
            fname.startswith("tokenizer")
            or fname == "special_tokens_map.json"
            or fname == "tekken.json"
            or fname == "generation_config.json"
            or fname == "chat_template.jinja"
        ):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, fname))
                print(f"  Copied {fname}")

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(dst_dir, f))
        for f in os.listdir(dst_dir)
        if os.path.isfile(os.path.join(dst_dir, f))
    )
    print(f"\n  Total output size: {total_size / 1e9:.2f} GB")
    print(f"  Output: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text-only model from Ministral 3B multimodal checkpoint"
    )
    parser.add_argument(
        "--src",
        default="/mnt/models/Ministral-3-3B-Instruct-2512-BF16",
        help="Path to full multimodal checkpoint",
    )
    parser.add_argument(
        "--dst",
        default="/mnt/models/Ministral-3B-text-only",
        help="Output directory for text-only model",
    )
    args = parser.parse_args()
    extract(args.src, args.dst)
