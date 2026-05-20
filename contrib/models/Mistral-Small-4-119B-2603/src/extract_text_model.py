#!/usr/bin/env python3
"""
Extract text-only weights from Mistral-Small-4-119B (multimodal) and dequantize FP8 -> BF16.
Creates a standalone DeepSeek-V3-compatible text model directory.

Input:  /home/ubuntu/models/Mistral-Small-4-119B (HF format with model-*.safetensors)
Output: /home/ubuntu/models/Mistral-Small-4-119B-text-only/

Weight key mapping (HF Mistral-Small-4 -> standalone text model):
  language_model.model.layers.N.* -> model.layers.N.*
  language_model.model.embed_tokens.weight -> model.embed_tokens.weight
  language_model.model.norm.weight -> model.norm.weight
  language_model.lm_head.weight -> lm_head.weight
  (vision_tower.* and multi_modal_projector.* are dropped)

FP8 dequantization:
  weight_bf16 = weight_fp8.float() * weight_scale_inv.float()
  then .to(bfloat16)
  Scale keys (*_scale_inv, *activation_scale) are dropped after dequant.
"""

import json
import os
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

SRC_DIR = "/mnt/nvme/models/Mistral-Small-4-119B-2603"
DST_DIR = "/mnt/nvme/models/Mistral-Small-4-119B-text-only"


def dequant_fp8(weight_fp8, scale_inv):
    """Dequantize FP8 weight to BF16 using per-tensor scale."""
    return (weight_fp8.float() * scale_inv.float()).to(torch.bfloat16)


def process_weights():
    os.makedirs(DST_DIR, exist_ok=True)

    # Load HF-format index
    with open(f"{SRC_DIR}/model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Group keys by source file
    file_keys = {}
    for key, fname in weight_map.items():
        if fname not in file_keys:
            file_keys[fname] = []
        file_keys[fname].append(key)

    new_weight_map = {}
    shard_idx = 0
    current_shard = {}
    current_shard_size = 0
    MAX_SHARD_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB per shard

    def flush_shard():
        nonlocal shard_idx, current_shard, current_shard_size
        if not current_shard:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        print(
            f"  Saving shard {shard_idx}: {len(current_shard)} tensors, {current_shard_size / 1e9:.2f} GB"
        )
        save_file(current_shard, f"{DST_DIR}/{shard_name}")
        for k in current_shard:
            new_weight_map[k] = shard_name
        current_shard = {}
        current_shard_size = 0

    def add_tensor(new_key, tensor):
        nonlocal current_shard, current_shard_size
        tensor_size = tensor.numel() * tensor.element_size()
        if current_shard_size + tensor_size > MAX_SHARD_SIZE and current_shard:
            flush_shard()
        current_shard[new_key] = tensor
        current_shard_size += tensor_size

    # Process each source file
    for fname in sorted(file_keys.keys()):
        keys = file_keys[fname]
        print(f"\nProcessing {fname} ({len(keys)} keys)...")

        f = safe_open(f"{SRC_DIR}/{fname}", framework="pt")

        # Build lookup of all tensors in this file for dequant
        # We need to handle several patterns:
        # 1. Regular FP8 weight: key.weight + key.weight_scale_inv -> dequant
        # 2. Expert grouped FP8: key (no .weight suffix) + key_scale_inv -> dequant
        # 3. BF16 weight: key.weight -> keep as-is
        # 4. Scale-only keys: *_scale_inv, *activation_scale -> skip (consumed by dequant)

        # First pass: identify all keys and their roles
        processed = set()

        for key in sorted(keys):
            if key in processed:
                continue

            # Skip vision and multimodal keys
            if key.startswith("vision_tower.") or key.startswith(
                "multi_modal_projector."
            ):
                processed.add(key)
                continue

            # Skip scale keys (will be consumed during dequant)
            if key.endswith("_scale_inv") or key.endswith("activation_scale"):
                processed.add(key)
                continue

            # Map key: strip language_model prefix
            if key.startswith("language_model.model."):
                new_key = key[len("language_model.") :]  # keep "model.layers.N..."
            elif key.startswith("language_model."):
                new_key = key[len("language_model.") :]  # "lm_head.weight"
            else:
                print(f"  WARNING: unexpected key prefix: {key}")
                processed.add(key)
                continue

            tensor = f.get_tensor(key)
            processed.add(key)

            if tensor.dtype == torch.float8_e4m3fn:
                # Find the scale
                # Pattern 1: key ends with .weight -> scale is key_base.weight_scale_inv
                # Pattern 2: expert grouped (no .weight) -> scale is key + "_scale_inv"
                if key.endswith(".weight"):
                    scale_key = key.replace(".weight", ".weight_scale_inv")
                else:
                    # Grouped expert tensors: e.g., mlp.experts.gate_up_proj
                    scale_key = key + "_scale_inv"

                if scale_key in weight_map:
                    scale_fname = weight_map[scale_key]
                    if scale_fname == fname:
                        scale = f.get_tensor(scale_key)
                    else:
                        sf = safe_open(f"{SRC_DIR}/{scale_fname}", framework="pt")
                        scale = sf.get_tensor(scale_key)
                    processed.add(scale_key)

                    # Also mark activation_scale as processed
                    if key.endswith(".weight"):
                        act_scale_key = key.replace(".weight", ".activation_scale")
                    else:
                        act_scale_key = key + "_activation_scale"
                    if act_scale_key in weight_map:
                        processed.add(act_scale_key)

                    tensor = dequant_fp8(tensor, scale)
                    print(f"  Dequant: {key} {list(tensor.shape)} -> {new_key}")
                else:
                    print(
                        f"  WARNING: no scale found for FP8 weight {key}, casting directly"
                    )
                    tensor = tensor.to(torch.bfloat16)
            else:
                print(
                    f"  Copy: {key} {list(tensor.shape)} dtype={tensor.dtype} -> {new_key}"
                )

            # Ensure BF16
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)

            add_tensor(new_key, tensor)

    # Flush remaining
    flush_shard()

    # Fix shard names (replace PLACEHOLDER with actual count)
    total_shards = shard_idx
    final_weight_map = {}
    for k, shard_name in new_weight_map.items():
        final_name = shard_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        if shard_name != final_name:
            old_path = f"{DST_DIR}/{shard_name}"
            new_path = f"{DST_DIR}/{final_name}"
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        final_weight_map[k] = final_name

    # Write new index
    new_index = {
        "metadata": {
            "total_size": sum(
                t.numel() * t.element_size()
                for shard_name in set(final_weight_map.values())
                for k, t in [
                    (
                        k2,
                        safe_open(f"{DST_DIR}/{shard_name}", framework="pt").get_tensor(
                            k2
                        ),
                    )
                    for k2 in [
                        k3 for k3, v in final_weight_map.items() if v == shard_name
                    ][:1]
                ]
            )
        },
        "weight_map": final_weight_map,
    }
    # Simpler: just compute total from what we know
    # We'll compute it properly after saving
    new_index = {
        "metadata": {},
        "weight_map": final_weight_map,
    }
    with open(f"{DST_DIR}/model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"\n=== Done! {len(final_weight_map)} tensors in {total_shards} shards ===")
    return total_shards


def create_config():
    """Create a DeepSeek-V3 compatible config for the text model."""
    with open(f"{SRC_DIR}/config.json") as f:
        orig_config = json.load(f)

    text_config = orig_config["text_config"]

    # Build DeepSeek-V3 compatible config
    # The NxDI DeepseekV3InferenceConfig needs these fields
    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        # From text_config
        "hidden_size": text_config["hidden_size"],  # 4096
        "intermediate_size": text_config["intermediate_size"],  # 12288 (shared expert)
        "num_hidden_layers": text_config["num_hidden_layers"],  # 36
        "num_attention_heads": text_config["num_attention_heads"],  # 32
        "num_key_value_heads": text_config.get(
            "num_key_value_heads", text_config["num_attention_heads"]
        ),  # 32
        "head_dim": text_config.get("head_dim", 128),  # 128
        "vocab_size": text_config["vocab_size"],  # 131072
        "max_position_embeddings": text_config["max_position_embeddings"],  # 1048576
        "rms_norm_eps": text_config["rms_norm_eps"],  # 1e-6
        "hidden_act": text_config.get("hidden_act", "silu"),
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "bos_token_id": text_config.get("bos_token_id", 1),
        "eos_token_id": text_config.get("eos_token_id", 2),
        "pad_token_id": text_config.get("pad_token_id", 11),
        # MLA (Multi-head Latent Attention) config
        "q_lora_rank": text_config["q_lora_rank"],  # 1024
        "qk_rope_head_dim": text_config["qk_rope_head_dim"],  # 64
        "qk_nope_head_dim": text_config["qk_nope_head_dim"],  # 64
        "kv_lora_rank": text_config["kv_lora_rank"],  # 256
        "v_head_dim": text_config["v_head_dim"],  # 128
        # MoE config
        "n_routed_experts": text_config["n_routed_experts"],  # 128
        "num_experts_per_tok": text_config["num_experts_per_tok"],  # 4
        "n_shared_experts": text_config["n_shared_experts"],  # 1
        "moe_intermediate_size": text_config["moe_intermediate_size"],  # 2048
        "first_k_dense_replace": text_config.get("first_k_dense_replace", 0),  # 0
        "norm_topk_prob": text_config.get("norm_topk_prob", True),
        "routed_scaling_factor": text_config.get("routed_scaling_factor", 1.0),
        "n_group": text_config.get("n_group", 1),
        "topk_group": text_config.get("topk_group", 1),
        # RoPE config
        "rope_theta": text_config.get("rope_parameters", {}).get("rope_theta", 10000.0),
        "rope_scaling": {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": text_config["rope_parameters"]["factor"],  # 128.0
            "original_max_position_embeddings": text_config["rope_parameters"][
                "original_max_position_embeddings"
            ],  # 8192
            "beta_fast": text_config["rope_parameters"]["beta_fast"],  # 32.0
            "beta_slow": text_config["rope_parameters"]["beta_slow"],  # 1.0
            "mscale": text_config["rope_parameters"].get("mscale", 1.0),
            "mscale_all_dim": text_config["rope_parameters"].get("mscale_all_dim", 1.0),
        },
    }

    with open(f"{DST_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {DST_DIR}/config.json")


def copy_tokenizer():
    """Copy tokenizer files from source."""
    tokenizer_files = [
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "SYSTEM_PROMPT.txt",
    ]
    # Also copy any tokenizer model files
    for fname in os.listdir(SRC_DIR):
        if fname.startswith("tokenizer") or fname == "special_tokens_map.json":
            tokenizer_files.append(fname)

    for fname in set(tokenizer_files):
        src = f"{SRC_DIR}/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{DST_DIR}/{fname}")
            print(f"Copied {fname}")


if __name__ == "__main__":
    print("=== Mistral-Small-4-119B Text Extraction + FP8 Dequantization ===")
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    os.makedirs(DST_DIR, exist_ok=True)
    create_config()
    copy_tokenizer()
    total_shards = process_weights()

    # Calculate total size
    total_bytes = 0
    for shard_file in Path(DST_DIR).glob("model-*.safetensors"):
        total_bytes += shard_file.stat().st_size
    print(f"\nTotal model size: {total_bytes / 1e9:.2f} GB")
    print(f"Output directory: {DST_DIR}")
