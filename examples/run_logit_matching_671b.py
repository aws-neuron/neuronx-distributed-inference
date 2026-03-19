#!/usr/bin/env python3
"""
Full 671B logit matching: generates golden logits on CPU from HF FP8 model,
then validates NXDI Neuron model against them using check_accuracy_logits_v2.

Three phases (each saves intermediate results, re-runnable from any point):
  1. Compile NEFFs with output_logits=True (reuses existing sharded weights)
  2. Generate golden logits on CPU from HF model (with FP8 dequantization)
  3. Load Neuron model and run check_accuracy_logits_v2

Requirements:
  - trn2.48xlarge (64 NeuronCores, 2TB RAM)
  - DeepSeek-V3-0324 FP8 weights on disk
  - Pre-sharded weights from Phase 8 (at --weights-path)
  - NXDI installed via install_deepseek.sh

Usage:
  # Full run (compile + golden + validate):
  python examples/run_logit_matching_671b.py \\
      --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \\
      --traced-model-path /scratch/deepseek_v3_logit \\
      --weights-path /scratch/deepseek_v3_traced/weights \\
      --num-tokens 32

  # Skip compile (reuse existing NEFFs):
  python examples/run_logit_matching_671b.py \\
      --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \\
      --traced-model-path /scratch/deepseek_v3_logit \\
      --weights-path /scratch/deepseek_v3_traced/weights \\
      --skip-compile --num-tokens 32

  # Skip golden generation (reuse saved golden logits):
  python examples/run_logit_matching_671b.py \\
      --traced-model-path /scratch/deepseek_v3_logit \\
      --weights-path /scratch/deepseek_v3_traced/weights \\
      --skip-compile --skip-golden --num-tokens 32
"""

import argparse
import gc
import glob
import json
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
# Patch DynamicCache methods removed/renamed in transformers 4.45+.
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


# ---------------------------------------------------------------------------
# Phase 1: Compile NEFFs with output_logits=True
# ---------------------------------------------------------------------------

def phase1_compile(args):
    logger.info("=" * 60)
    logger.info("Phase 1: Compile NEFFs with output_logits=True")
    logger.info("=" * 60)

    traced_path = args.traced_model_path
    neff_path = os.path.join(traced_path, "model.pt")

    if args.skip_compile and os.path.exists(neff_path):
        logger.info("Skipping compilation (--skip-compile, %s exists)", neff_path)
        return

    from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
    from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
        DeepseekV3InferenceConfig,
        NeuronDeepseekV3ForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        torch_dtype=torch.bfloat16,
        logical_nc_config=2,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        save_sharded_checkpoint=True,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1, do_sample=False),
    )
    inf_config = DeepseekV3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(args.model_path),
    )

    start = time.time()
    logger.info("Compiling (tp=64, lnc=2, seq_len=%d, output_logits=True)...", args.seq_len)
    model = NeuronDeepseekV3ForCausalLM(args.model_path, inf_config)
    model.compile(traced_path)
    elapsed = time.time() - start
    logger.info("Compilation complete in %.1f minutes", elapsed / 60)

    del model
    gc.collect()

    # Symlink existing sharded weights (they're batch-size and config independent)
    weights_dir = os.path.join(traced_path, "weights")
    if args.weights_path and os.path.isdir(args.weights_path):
        if os.path.islink(weights_dir):
            os.unlink(weights_dir)
        elif os.path.isdir(weights_dir) and not any(
            f.endswith(".safetensors") for f in os.listdir(weights_dir)
        ):
            os.rmdir(weights_dir)
        if not os.path.exists(weights_dir):
            os.symlink(os.path.abspath(args.weights_path), weights_dir)
            logger.info("Symlinked weights: %s -> %s", weights_dir, args.weights_path)

    # Copy tokenizer files to traced path
    from shutil import copy2
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "chat_template.jinja"]:
        src = os.path.join(args.model_path, fname)
        if os.path.exists(src) and not os.path.exists(os.path.join(traced_path, fname)):
            copy2(src, traced_path)


# ---------------------------------------------------------------------------
# Phase 2: Generate golden logits on CPU from HF model
# ---------------------------------------------------------------------------

def dequantize_fp8_state_dict(state_dict, block_size=128):
    """Dequantize FP8 block-wise weights to BF16 in-place."""
    scale_inv_keys = [k for k in state_dict if k.endswith(".weight_scale_inv")]
    if not scale_inv_keys:
        return state_dict

    total = len(scale_inv_keys)
    logger.info("Dequantizing %d FP8 weights to BF16 (block_size=%d)...", total, block_size)

    for idx, scale_key in enumerate(scale_inv_keys):
        if idx % 2000 == 0:
            logger.info("  %d/%d weights (RSS: %.1f GB)", idx, total, get_rss_gb())
            gc.collect()

        weight_key = scale_key.replace(".weight_scale_inv", ".weight")
        if weight_key not in state_dict:
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
        w = w * scale_inv[:nbm, :nbn].unsqueeze(1).unsqueeze(3)
        w = w.view(nbm * block_size, nbn * block_size)

        state_dict[weight_key] = w[:M, :N].to(torch.bfloat16)
        del state_dict[scale_key]

    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale_inv"):
            del state_dict[key]

    gc.collect()
    logger.info("Dequantization complete (RSS: %.1f GB)", get_rss_gb())
    return state_dict


def phase2_golden(args):
    logger.info("=" * 60)
    logger.info("Phase 2: Generate golden logits on CPU")
    logger.info("=" * 60)

    golden_path = os.path.join(args.traced_model_path, "golden_logits.pt")
    inputs_path = os.path.join(args.traced_model_path, "golden_inputs.pt")

    if args.skip_golden and os.path.exists(golden_path):
        logger.info("Skipping golden generation (--skip-golden, file exists)")
        expected_logits = torch.load(golden_path, weights_only=True)
        logger.info("Loaded golden logits: shape %s", expected_logits.shape)
        return expected_logits

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    model_path = args.model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompt
    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    logger.info("Prompt: '%s' (%d tokens)", prompt, input_ids.shape[1])

    # --- Load HF model with FP8 dequantization ---
    t0 = time.time()

    # 1. Load safetensors shards
    shard_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    logger.info("Loading %d safetensors shards...", len(shard_files))
    state_dict = {}
    for i, f in enumerate(shard_files):
        shard = load_file(f)
        state_dict.update(shard)
        del shard
        if (i + 1) % 20 == 0:
            logger.info("  Loaded %d/%d shards (RSS: %.1f GB)", i + 1, len(shard_files), get_rss_gb())
    logger.info("All shards loaded: %d keys (RSS: %.1f GB)", len(state_dict), get_rss_gb())

    # 2. Dequantize FP8 -> BF16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    block_size = 128
    if hasattr(config, "quantization_config") and config.quantization_config:
        wbs = getattr(config.quantization_config, "weight_block_size", None)
        if wbs:
            block_size = wbs[0] if isinstance(wbs, (list, tuple)) else wbs
    state_dict = dequantize_fp8_state_dict(state_dict, block_size=block_size)

    # 3. Create model on meta device, then load weights with assign=True
    logger.info("Creating HF model on meta device...")
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

    logger.info("Loading dequantized state dict (assign=True)...")
    hf_model.load_state_dict(state_dict, assign=True, strict=False)
    del state_dict
    gc.collect()
    hf_model.eval()

    load_time = time.time() - t0
    logger.info("HF model ready in %.1f minutes (RSS: %.1f GB)", load_time / 60, get_rss_gb())

    # --- Generate golden logits ---
    num_tokens = args.num_tokens
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=num_tokens,
        min_new_tokens=num_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,  # Don't stop early
    )

    logger.info("Generating %d golden tokens on CPU...", num_tokens)
    t1 = time.time()
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    gen_time = time.time() - t1
    logger.info("Generation complete in %.1f seconds (%.2f sec/token)", gen_time, gen_time / num_tokens)

    # Stack scores: [num_tokens, batch_size, vocab_size]
    expected_logits = torch.stack(outputs.scores)[:num_tokens, :, :]
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_token_ids = outputs.sequences[0, input_ids.shape[1]:].tolist()

    logger.info("Golden logits shape: %s", expected_logits.shape)
    logger.info("HF generated text: %s", generated_text[:500])

    # Save golden logits and inputs
    os.makedirs(args.traced_model_path, exist_ok=True)
    torch.save(expected_logits, golden_path)
    torch.save({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt": prompt,
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "num_tokens": num_tokens,
        "generation_time_seconds": gen_time,
        "model_load_time_seconds": load_time,
    }, inputs_path)
    logger.info("Saved golden logits to %s (%.1f MB)", golden_path,
                os.path.getsize(golden_path) / 1e6)

    # Free HF model
    del hf_model, outputs
    gc.collect()
    logger.info("HF model freed (RSS: %.1f GB)", get_rss_gb())

    return expected_logits


# ---------------------------------------------------------------------------
# Phase 3: Logit matching on Neuron
# ---------------------------------------------------------------------------

def phase3_validate(args, expected_logits):
    logger.info("=" * 60)
    logger.info("Phase 3: Logit matching on Neuron")
    logger.info("=" * 60)

    from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
        NeuronDeepseekV3ForCausalLM,
    )
    from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits_v2
    from transformers import AutoTokenizer, GenerationConfig

    traced_path = args.traced_model_path

    # Load Neuron model
    logger.info("Loading Neuron model from %s...", traced_path)
    t0 = time.time()
    neuron_model = NeuronDeepseekV3ForCausalLM(traced_path)
    neuron_model.load(traced_path)
    load_time = time.time() - t0
    logger.info("Neuron model loaded in %.1f seconds (RSS: %.1f GB)", load_time, get_rss_gb())

    # Load tokenizer and inputs
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path or traced_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs_path = os.path.join(traced_path, "golden_inputs.pt")
    if os.path.exists(inputs_path):
        saved = torch.load(inputs_path, weights_only=False)
        input_ids = saved["input_ids"]
        attention_mask = saved["attention_mask"]
        prompt = saved.get("prompt", args.prompt)
        logger.info("Loaded saved inputs: prompt='%s' (%d tokens)", prompt, input_ids.shape[1])
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

    generation_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Run logit matching
    num_tokens = min(args.num_tokens, expected_logits.shape[0])
    logger.info("Running check_accuracy_logits_v2 (%d tokens, divergence_tol=%.4f)...",
                num_tokens, args.divergence_tol)

    t1 = time.time()
    status = "PASS"
    error_msg = None
    results = {}
    try:
        results = check_accuracy_logits_v2(
            neuron_model=neuron_model,
            expected_logits=expected_logits,
            inputs_input_ids=input_ids,
            inputs_attention_mask=attention_mask,
            generation_config=generation_config,
            divergence_difference_tol=args.divergence_tol,
            num_tokens_to_check=num_tokens,
            tokenizer=tokenizer,
        )
        logger.info("Logit matching PASSED!")
    except Exception as e:
        status = "FAIL"
        error_msg = str(e)
        logger.error("Logit matching FAILED: %s", error_msg)
    validate_time = time.time() - t1

    # Save results
    report = {
        "status": status,
        "model": "DeepSeek-V3-0324 (671B)",
        "prompt": args.prompt,
        "num_tokens_checked": num_tokens,
        "divergence_difference_tol": args.divergence_tol,
        "tp_degree": 64,
        "logical_nc_config": 2,
        "seq_len": args.seq_len,
        "neuron_model_load_seconds": load_time,
        "validation_seconds": validate_time,
        "error": error_msg,
        "results": results if isinstance(results, dict) else str(results),
    }

    results_path = os.path.join(traced_path, "logit_matching_results.json")
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("LOGIT MATCHING RESULT: %s", status)
    logger.info("  Model: DeepSeek-V3-0324 (671B, tp=64, lnc=2)")
    logger.info("  Tokens checked: %d", num_tokens)
    logger.info("  Divergence tolerance: %.4f", args.divergence_tol)
    if isinstance(results, dict) and results:
        logger.info("  Results: %s", json.dumps(results, indent=2, default=str)[:1000])
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 671B logit matching")
    parser.add_argument("--model-path", type=str,
                        default=os.path.expanduser("~/environment/models/DeepSeek-V3-0324-FP8"),
                        help="Path to HF FP8 weights")
    parser.add_argument("--traced-model-path", type=str,
                        default="/scratch/deepseek_v3_logit",
                        help="Directory for compiled model with output_logits=True")
    parser.add_argument("--weights-path", type=str,
                        default="/scratch/deepseek_v3_traced/weights",
                        help="Path to existing pre-sharded weights (symlinked, not copied)")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=32,
                        help="Number of tokens to generate and validate")
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is",
                        help="Input prompt for golden generation")
    parser.add_argument("--divergence-tol", type=float, default=0.001,
                        help="Divergence difference tolerance for logit matching")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip Phase 1 (reuse existing NEFFs)")
    parser.add_argument("--skip-golden", action="store_true",
                        help="Skip Phase 2 (reuse saved golden logits)")
    args = parser.parse_args()

    logger.info("DeepSeek V3 671B Logit Matching")
    logger.info("  Model path: %s", args.model_path)
    logger.info("  Traced path: %s", args.traced_model_path)
    logger.info("  Weights path: %s", args.weights_path)
    logger.info("  Num tokens: %d", args.num_tokens)
    logger.info("  Prompt: '%s'", args.prompt)
    logger.info("  RSS: %.1f GB", get_rss_gb())

    total_start = time.time()

    # Phase 1: Compile
    phase1_compile(args)

    # Phase 2: Golden logits
    expected_logits = phase2_golden(args)

    # Phase 3: Validate
    report = phase3_validate(args, expected_logits)

    total_time = time.time() - total_start
    logger.info("Total elapsed: %.1f minutes", total_time / 60)
    logger.info("Final status: %s", report["status"])

    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
