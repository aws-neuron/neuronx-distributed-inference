#!/usr/bin/env python3
"""
Ministral 14B: Extract text-only BF16 weights from FP8 multimodal checkpoint.

The HuggingFace checkpoint `mistralai/Ministral-3-14B-Instruct-2512` is a
`Mistral3ForConditionalGeneration` model (multimodal: Pixtral vision encoder +
text decoder) with FP8 E4M3 quantized linear weights.

This script:
  1. Reads safetensors weights directly (no model class needed)
  2. Strips `language_model.` prefix from text keys
  3. Dequantizes FP8 E4M3 weights to BF16 using per-tensor weight_scale_inv
  4. Drops vision_tower, multi_modal_projector, activation_scale, weight_scale_inv keys
  5. Creates a LlamaForCausalLM-compatible config.json (avoids Pixtral auto-promotion)
  6. Cleans tokenizer_config.json (removes processor_class references)
  7. Writes sharded BF16 safetensors

Output: A directory loadable by vLLM as a standard LlamaForCausalLM with --hf-overrides.

Model details:
  - 40 layers, hidden=5120, heads=32/8kv, head_dim=128, intermediate=16384
  - vocab=131072, tie_word_embeddings=false (separate lm_head)
  - YaRN RoPE scaling (factor=16, theta=1e9)
  - FP8 E4M3 linear weights with per-tensor BF16 scalar scales
  - embed_tokens, lm_head, layernorm weights are already BF16

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


def dequantize_fp8(weight, scale_inv):
    """Dequantize FP8 E4M3 weight to BF16 using per-tensor scale.

    Formula: bf16_weight = fp8_weight.to(bf16) * weight_scale_inv
    """
    return weight.to(torch.bfloat16) * scale_inv.to(torch.bfloat16)


def extract(src_dir, dst_dir):
    print("=" * 60)
    print("Ministral 14B: Extract text-only BF16 backbone")
    print("=" * 60)
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    # Load source config
    with open(os.path.join(src_dir, "config.json")) as f:
        full_config = json.load(f)

    text_config = full_config.get("text_config", {})
    if not text_config:
        print("  ERROR: no text_config found in source config.json")
        return

    # ---------------------------------------------------------------
    # Step 1: Read and dequantize weights
    # ---------------------------------------------------------------
    print(f"\n[1/5] Reading and dequantizing weights from safetensors...")
    t0 = time.time()

    # Load safetensors index
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Group keys by shard file
    file_keys = {}
    for key, fname in weight_map.items():
        if fname not in file_keys:
            file_keys[fname] = []
        file_keys[fname].append(key)

    text_prefix = "language_model."
    text_weights = {}
    skipped_vision = 0
    skipped_scales = 0
    dequantized = 0

    # First pass: collect all scale keys for lookup
    scale_map = {}  # "language_model.model.layers.0.self_attn.q_proj" -> shard_file
    for key in weight_map:
        if key.endswith(".weight_scale_inv") and key.startswith(text_prefix):
            base = key[: -len(".weight_scale_inv")]
            scale_map[base] = weight_map[key]

    for fname in sorted(file_keys.keys()):
        keys = file_keys[fname]
        fpath = os.path.join(src_dir, fname)
        print(f"  Processing {fname} ({len(keys)} keys)...")
        f = safe_open(fpath, framework="pt")

        for key in keys:
            # Skip non-text keys
            if not key.startswith(text_prefix):
                skipped_vision += 1
                continue

            # Skip activation_scale and weight_scale_inv (not needed in output)
            if key.endswith(".activation_scale") or key.endswith(".weight_scale_inv"):
                skipped_scales += 1
                continue

            new_key = key[len(text_prefix) :]
            tensor = f.get_tensor(key)

            # Dequantize FP8 weights
            if tensor.dtype == torch.float8_e4m3fn:
                # Find the corresponding scale
                base_key = key[: -len(".weight")]  # strip ".weight"
                scale_key = base_key + ".weight_scale_inv"
                if scale_key in weight_map:
                    # Scale might be in a different shard
                    scale_shard = weight_map[scale_key]
                    if scale_shard == fname:
                        scale = f.get_tensor(scale_key)
                    else:
                        sf = safe_open(
                            os.path.join(src_dir, scale_shard), framework="pt"
                        )
                        scale = sf.get_tensor(scale_key)
                    tensor = dequantize_fp8(tensor, scale)
                    dequantized += 1
                else:
                    print(f"    WARNING: no scale for {key}, casting directly to bf16")
                    tensor = tensor.to(torch.bfloat16)
            elif tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)

            text_weights[new_key] = tensor

    elapsed = time.time() - t0
    print(f"  Extracted {len(text_weights)} text weights")
    print(f"  Dequantized {dequantized} FP8 tensors to BF16")
    print(f"  Skipped {skipped_vision} vision/projector keys")
    print(f"  Skipped {skipped_scales} scale keys")
    print(f"  Time: {elapsed:.1f}s")

    # ---------------------------------------------------------------
    # Step 2: Save sharded weights
    # ---------------------------------------------------------------
    print(f"\n[2/5] Saving text-only BF16 weights...")
    t0 = time.time()

    total_bytes = sum(t.numel() * t.element_size() for t in text_weights.values())
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")

    MAX_SHARD = 5e9  # 5 GB per shard
    shard_idx = 0
    current_shard = {}
    current_size = 0
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
        print(
            f"    Shard {shard_idx}: {len(current_shard)} tensors, "
            f"{current_size / 1e9:.2f} GB"
        )
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

    # Rename shards with correct total count
    total_shards = shard_idx
    final_map = {}
    for k, sname in new_weight_map.items():
        final = sname.replace("PLACEHOLDER", f"{total_shards:05d}")
        final_map[k] = final

    # Rename shard files
    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_path = os.path.join(dst_dir, old_name)
        new_path = os.path.join(dst_dir, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": final_map}, f, indent=2)

    print(f"  Saved {total_shards} shards in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # Step 3: Create LlamaForCausalLM-compatible config
    # ---------------------------------------------------------------
    print(f"\n[3/5] Creating config.json...")

    # IMPORTANT: Use LlamaForCausalLM architecture.
    # vLLM 0.16 auto-promotes MistralForCausalLM to PixtralForConditionalGeneration
    # based on tokenizer/processor hints. Using LlamaForCausalLM avoids this entirely.
    # The Llama code path also correctly handles head_dim != hidden_size/num_heads.
    rope_params = text_config.get("rope_parameters", {})

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
        "max_position_embeddings": text_config.get("max_position_embeddings", 262144),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1e-5),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "tie_word_embeddings": full_config.get("tie_word_embeddings", False),
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "bos_token_id": text_config.get("bos_token_id", 1),
        "eos_token_id": text_config.get("eos_token_id", 2),
        "rope_theta": rope_params.get("rope_theta", 1000000000.0),
        # YaRN rope scaling -- required for correct position embeddings
        "rope_scaling": {
            "rope_type": rope_params.get("rope_type", "yarn"),
            "type": rope_params.get("type", "yarn"),
            "factor": rope_params.get("factor", 16.0),
            "beta_fast": rope_params.get("beta_fast", 32.0),
            "beta_slow": rope_params.get("beta_slow", 1.0),
            "original_max_position_embeddings": rope_params.get(
                "original_max_position_embeddings", 16384
            ),
            "mscale": rope_params.get("mscale", 1.0),
            "mscale_all_dim": rope_params.get("mscale_all_dim", 1.0),
        },
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
    print(f"  rope_theta: {config['rope_theta']}")
    print(
        f"  rope_scaling: {config['rope_scaling']['rope_type']}, "
        f"factor={config['rope_scaling']['factor']}"
    )
    print(f"  tie_word_embeddings: {config['tie_word_embeddings']}")

    if config["head_dim"] != computed_head_dim:
        print(
            f"  NOTE: head_dim ({config['head_dim']}) != "
            f"hidden_size/num_heads ({computed_head_dim})"
        )
        print(
            f"  This is intentional -- the model uses head_dim=128 with hidden_size=5120"
        )

    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---------------------------------------------------------------
    # Step 4: Copy and clean tokenizer files
    # ---------------------------------------------------------------
    print(f"\n[4/5] Copying tokenizer files...")
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

    # Clean tokenizer_config.json:
    # 1. Remove processor_class to prevent vLLM 0.16 Pixtral auto-promotion
    # 2. Fix tokenizer_class: "TokenizersBackend" is Mistral-internal, not in
    #    HuggingFace transformers. Replace with "PreTrainedTokenizerFast" which
    #    works with the standard tokenizer.json file.
    tok_config_path = os.path.join(dst_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        changed = False
        for field in ["processor_class", "auto_map"]:
            if field in tok_config:
                del tok_config[field]
                changed = True
                print(f"  Removed '{field}' from tokenizer_config.json")
        if tok_config.get("tokenizer_class") == "TokenizersBackend":
            tok_config["tokenizer_class"] = "PreTrainedTokenizerFast"
            changed = True
            print(
                f"  Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast"
            )
        # Remove Mistral-specific fields that break HF transformers:
        # - extra_special_tokens: list format not compatible (expects dict)
        # - backend: Mistral-internal tokenizer backend identifier
        for field in ["extra_special_tokens", "backend"]:
            if field in tok_config:
                del tok_config[field]
                changed = True
                print(f"  Removed '{field}' from tokenizer_config.json")
        if changed:
            with open(tok_config_path, "w") as f:
                json.dump(tok_config, f, indent=2)

    # Also clean processor_config.json if it was copied
    proc_config_path = os.path.join(dst_dir, "processor_config.json")
    if os.path.exists(proc_config_path):
        os.remove(proc_config_path)
        print(f"  Removed processor_config.json")

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    print(f"\n[5/5] Summary")
    total_size = sum(
        os.path.getsize(os.path.join(dst_dir, f))
        for f in os.listdir(dst_dir)
        if os.path.isfile(os.path.join(dst_dir, f))
    )
    n_files = len(os.listdir(dst_dir))
    print(f"  Output directory: {dst_dir}")
    print(f"  Total files: {n_files}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"\n  To serve with vLLM + NxDI TKG:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {dst_dir} \\")
    print(f"    --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 4 \\")
    print(f"    --no-enable-prefix-caching --block-size 8 \\")
    print(
        f'    --hf-overrides \'{{"architectures": ["LlamaForCausalLM"], '
        f'"model_type": "llama"}}\' \\'
    )
    print(
        f'    --additional-config \'{{"override_neuron_config": {{'
        f'"fused_qkv": true, "qkv_nki_kernel_enabled": true, '
        f'"qkv_kernel_enabled": true, '
        f'"attn_block_tkg_nki_kernel_enabled": true, '
        f'"attn_block_tkg_nki_kernel_cache_update": true}}}}\''
    )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text-only BF16 model from Ministral 14B FP8 multimodal checkpoint"
    )
    parser.add_argument(
        "--src",
        default="/home/ubuntu/models/Ministral-3-14B-Instruct-2512",
        help="Path to full multimodal FP8 checkpoint",
    )
    parser.add_argument(
        "--dst",
        default="/home/ubuntu/models/Ministral-3-14B-text-bf16",
        help="Output directory for text-only BF16 model",
    )
    args = parser.parse_args()
    extract(args.src, args.dst)
