#!/usr/bin/env python3
"""
Generate FP32 golden logits for DeepSeek V3 671B on CPU.

Loads FP8 HF weights, dequantizes to FP32 (NOT BF16), creates the HF model
in FP32, and runs greedy generation to produce high-fidelity reference logits.

Memory requirements:
  - FP8 weights load:  ~641 GB
  - FP32 dequantized:  ~2.68 TB (replaces FP8 in-place where possible)
  - Peak:              ~3+ TB (needs 2 TB RAM + 1 TB NVMe swap)
  - After model load:  ~2.68 TB (FP32 model on CPU)

Usage:
  python examples/generate_golden_logits_fp32.py \
      --model-path /scratch0/DeepSeek-V3-0324-FP8 \
      --output-path /scratch0/golden_logits_fp32.pt \
      --num-tokens 32

  # Resume from saved FP32 state dict (skip dequantization):
  python examples/generate_golden_logits_fp32.py \
      --model-path /scratch0/DeepSeek-V3-0324-FP8 \
      --fp32-state-dict-path /scratch2/fp32_state_dict \
      --output-path /scratch0/golden_logits_fp32.pt \
      --num-tokens 32
"""

import argparse
import gc
import glob
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Compatibility shim: DeepSeek HF model code was written for transformers ~4.38.
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)


def get_rss_gb():
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / 1e9
    except Exception:
        return -1


def dequantize_fp8_to_fp32(state_dict, block_size=128):
    """Dequantize FP8 block-wise weights to FP32 in-place.

    Unlike the BF16 version, this preserves full FP32 precision after
    dequantization, providing the highest-fidelity reference weights.
    """
    scale_inv_keys = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    if not scale_inv_keys:
        logger.info("No FP8 weights found — state dict already in standard dtype.")
        return state_dict

    total = len(scale_inv_keys)
    logger.info("Dequantizing %d FP8 weights to FP32 (block_size=%d)...", total, block_size)

    for idx, scale_key in enumerate(scale_inv_keys):
        if idx % 500 == 0:
            logger.info("  %d/%d weights (RSS: %.1f GB)", idx, total, get_rss_gb())
            gc.collect()

        weight_key = scale_key.replace(".weight_scale_inv", ".weight")
        if weight_key not in state_dict:
            del state_dict[scale_key]
            continue

        weight = state_dict[weight_key]
        scale_inv = state_dict[scale_key]

        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            del state_dict[scale_key]
            continue

        M, N = weight.shape
        nbm = (M + block_size - 1) // block_size
        nbn = (N + block_size - 1) // block_size

        pad_m = nbm * block_size - M
        pad_n = nbn * block_size - N
        if pad_m or pad_n:
            w = torch.zeros(nbm * block_size, nbn * block_size, dtype=torch.float32)
            w[:M, :N] = weight.to(torch.float32)
        else:
            w = weight.to(torch.float32)

        w = w.view(nbm, block_size, nbn, block_size)
        w = w * scale_inv.to(torch.float32)[:nbm, :nbn].unsqueeze(1).unsqueeze(3)
        w = w.view(nbm * block_size, nbn * block_size)

        # KEY DIFFERENCE: keep as FP32, not BF16
        state_dict[weight_key] = w[:M, :N]
        del state_dict[scale_key]

    # Clean up any remaining scale keys
    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale_inv"):
            del state_dict[key]

    gc.collect()
    logger.info("FP32 dequantization complete (RSS: %.1f GB)", get_rss_gb())
    return state_dict


def load_fp32_state_dict(model_path, block_size=128):
    """Load FP8 safetensors and dequantize to FP32."""
    from safetensors.torch import load_file

    shard_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No model*.safetensors found in {model_path}")

    logger.info("Loading %d safetensors shards from %s...", len(shard_files), model_path)
    state_dict = {}
    for i, f in enumerate(shard_files):
        shard = load_file(f)
        state_dict.update(shard)
        del shard
        if (i + 1) % 20 == 0:
            logger.info("  Loaded %d/%d shards (RSS: %.1f GB)", i + 1, len(shard_files), get_rss_gb())
    logger.info("All shards loaded: %d keys (RSS: %.1f GB)", len(state_dict), get_rss_gb())

    state_dict = dequantize_fp8_to_fp32(state_dict, block_size=block_size)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Generate FP32 golden logits for DeepSeek V3 671B")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HF FP8 weights directory")
    parser.add_argument("--output-path", type=str, default="/scratch0/golden_logits_fp32.pt",
                        help="Output path for golden logits tensor")
    parser.add_argument("--num-tokens", type=int, default=32,
                        help="Number of output tokens to generate")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--fp32-state-dict-path", type=str, default=None,
                        help="Path to pre-saved FP32 state dict (skip dequantization)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FP32 Golden Logit Generation — DeepSeek V3 671B")
    logger.info("=" * 60)
    logger.info("  Model path:  %s", args.model_path)
    logger.info("  Output path: %s", args.output_path)
    logger.info("  Num tokens:  %d", args.num_tokens)
    logger.info("  Prompt:      '%s'", args.prompt)
    logger.info("  RAM:         %.1f GB total, %.1f GB free",
                os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9,
                os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 1e9)
    logger.info("  RSS:         %.1f GB", get_rss_gb())

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    total_start = time.time()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    logger.info("Tokenized prompt: %d tokens", input_ids.shape[1])

    # --- Load or dequantize state dict ---
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    block_size = 128
    if hasattr(config, "quantization_config") and config.quantization_config:
        qc = config.quantization_config
        if isinstance(qc, dict):
            wbs = qc.get("weight_block_size", [128, 128])
        else:
            wbs = getattr(qc, "weight_block_size", [128, 128])
        block_size = wbs[0] if isinstance(wbs, (list, tuple)) else wbs

    if args.fp32_state_dict_path and os.path.exists(args.fp32_state_dict_path):
        logger.info("Loading pre-saved FP32 state dict from %s...", args.fp32_state_dict_path)
        t0 = time.time()
        state_dict = torch.load(args.fp32_state_dict_path, weights_only=True)
        logger.info("Loaded in %.1f min (RSS: %.1f GB)", (time.time() - t0) / 60, get_rss_gb())
    else:
        t0 = time.time()
        state_dict = load_fp32_state_dict(args.model_path, block_size=block_size)
        load_time = time.time() - t0
        logger.info("State dict ready in %.1f min (RSS: %.1f GB)", load_time / 60, get_rss_gb())

    # --- Create HF model in FP32 ---
    logger.info("Creating HF model on meta device (FP32)...")
    # Remove quantization_config so HF doesn't try to handle FP8 itself.
    # Set to empty dict (not None) to avoid AttributeError in config.to_dict().
    if hasattr(config, "quantization_config"):
        config.quantization_config = {}
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.float32
        )

    logger.info("Loading FP32 state dict into model (assign=True)...")
    t0 = time.time()

    # Cast any non-FP32 tensors (e.g. embedding, layernorm) to FP32
    for key in state_dict:
        if state_dict[key].dtype != torch.float32:
            state_dict[key] = state_dict[key].to(torch.float32)

    hf_model.load_state_dict(state_dict, assign=True, strict=False)
    del state_dict
    gc.collect()
    hf_model.eval()
    logger.info("Model loaded in %.1f min (RSS: %.1f GB)", (time.time() - t0) / 60, get_rss_gb())

    # --- Generate golden logits ---
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.num_tokens,
        min_new_tokens=args.num_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,  # Don't stop early
    )

    logger.info("Generating %d tokens on CPU in FP32...", args.num_tokens)
    t1 = time.time()
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    gen_time = time.time() - t1
    logger.info("Generation complete in %.1f min (%.1f sec/token, RSS: %.1f GB)",
                gen_time / 60, gen_time / args.num_tokens, get_rss_gb())

    # Stack scores: [num_tokens, batch_size, vocab_size]
    expected_logits = torch.stack(outputs.scores)[:args.num_tokens, :, :]
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_token_ids = outputs.sequences[0, input_ids.shape[1]:].tolist()

    logger.info("Golden logits shape: %s, dtype: %s", expected_logits.shape, expected_logits.dtype)
    logger.info("Generated text: %s", generated_text[:500])

    # --- Save outputs ---
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    torch.save(expected_logits, args.output_path)
    logger.info("Saved golden logits to %s (%.1f MB)",
                args.output_path, os.path.getsize(args.output_path) / 1e6)

    inputs_path = args.output_path.replace(".pt", "_inputs.pt")
    torch.save({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt": args.prompt,
        "dtype": "float32",
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "num_tokens": args.num_tokens,
        "generation_time_seconds": gen_time,
        "block_size": block_size,
    }, inputs_path)
    logger.info("Saved inputs metadata to %s", inputs_path)

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info("COMPLETE in %.1f hours", total_time / 3600)
    logger.info("  Golden logits: %s", args.output_path)
    logger.info("  Shape: %s, dtype: %s", expected_logits.shape, expected_logits.dtype)
    logger.info("  Generated: %s", generated_text[:200])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
