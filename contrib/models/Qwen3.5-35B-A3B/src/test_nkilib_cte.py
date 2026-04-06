#!/usr/bin/env python3
"""
Test nkilib flash attention kernel for head_dim=256 (CTE performance).

Tests both approaches:
- Option A: monkey-patch _pre_prod_kernels (import patch_attn_kernel before model loading)
- Option B: direct nkilib call from perform_prefill (USE_NKILIB_KERNEL=1)

Usage:
    # Option A test (compile + test):
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python3 test_nkilib_cte.py --option A \
        --compile-dir /mnt/models/compiled_qwen35_nkilib_optA

    # Option B test:
    USE_NKILIB_KERNEL=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python3 test_nkilib_cte.py \
        --option B --compile-dir /mnt/models/compiled_qwen35_nkilib_optB

    # Baseline (no nkilib):
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python3 test_nkilib_cte.py --option baseline \
        --compile-dir /mnt/models/compiled_qwen35_sl2048 --skip-compile
"""

import argparse
import json
import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--option", choices=["A", "B", "baseline"], required=True)
parser.add_argument("--compile-dir", required=True)
parser.add_argument("--model-dir", default="/mnt/models/Qwen3.5-35B-A3B")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--skip-compile", action="store_true")
args = parser.parse_args()

# Option A: Apply monkey-patch BEFORE any NxDI imports
if args.option == "A":
    logger.info("=== Option A: Applying monkey-patch for _pre_prod_kernels ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import patch_attn_kernel

    assert patch_attn_kernel._patched, "Patch failed!"

# Option B: Set environment variable for direct nkilib
if args.option == "B":
    logger.info("=== Option B: Direct nkilib kernel (USE_NKILIB_KERNEL=1) ===")
    os.environ["USE_NKILIB_KERNEL"] = "1"

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

# Now import
import torch

sys.path.insert(0, "/home/ubuntu/nxdi-qwen35/contrib/models/Qwen3.5-35B-A3B/src")
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig
from modeling_qwen35_moe import (
    USE_NKILIB_KERNEL,
    NKILIB_PATCH_ACTIVE,
    _nkilib_flash_attn,
)
from neuronx_distributed_inference.models.config import MoENeuronConfig

logger.info(f"Option: {args.option}")
logger.info(f"USE_NKILIB_KERNEL: {USE_NKILIB_KERNEL}")
logger.info(f"NKILIB_PATCH_ACTIVE: {NKILIB_PATCH_ACTIVE}")
logger.info(f"_nkilib_flash_attn loaded: {_nkilib_flash_attn is not None}")

if args.option == "A":
    import neuronx_distributed_inference.modules.attention.attention_base as ab
    import inspect

    logger.info(
        f"NxDI _flash_fwd_call_nki type: {type(ab._flash_fwd_call_nki).__name__}"
    )
    try:
        fn = ab._flash_fwd_call_nki
        src = inspect.getfile(fn.func if hasattr(fn, "func") else fn)
        logger.info(f"NxDI kernel source: {src}")
    except:
        pass

# Load HF config
with open(os.path.join(args.model_dir, "config.json")) as f:
    full_config = json.load(f)
text_config = full_config.get("text_config", full_config)

# Configure model
max_context = args.seq_len // 2
max_new = args.seq_len - max_context

logger.info(
    f"seq_len={args.seq_len}, max_context={max_context}, max_new={max_new}, TP=4"
)

neuron_config = MoENeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    seq_len=args.seq_len,
    max_context_length=max_context,
    max_new_tokens=max_new,
    on_device_sampling_config=None,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    moe_tp_degree=4,
    moe_ep_degree=1,
    blockwise_matmul_config={"block_size": 2048},
)

config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
if "rope_parameters" in text_config:
    config_dict["rope_theta"] = text_config["rope_parameters"].get(
        "rope_theta", 10000000
    )
if "tie_word_embeddings" not in config_dict:
    config_dict["tie_word_embeddings"] = False

if not args.skip_compile:
    logger.info(f"=== Compiling model (option={args.option}) ===")
    logger.info(f"Save directory: {args.compile_dir}")
    os.makedirs(args.compile_dir, exist_ok=True)

    config = Qwen35MoeInferenceConfig(
        neuron_config=neuron_config,
        save_path=args.compile_dir,
        **config_dict,
    )

    compile_start = time.time()
    model = NeuronQwen35MoeForCausalLM(model_path=args.model_dir, config=config)
    logger.info("Model created, starting compilation...")

    try:
        model.compile(args.compile_dir)
    except Exception as e:
        compile_time = time.time() - compile_start
        logger.error(f"Compilation FAILED after {compile_time:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    compile_time = time.time() - compile_start
    logger.info(
        f"Compilation completed in {compile_time:.1f}s ({compile_time / 60:.1f}m)"
    )

    if args.compile_only:
        logger.info("Compile-only mode - exiting")
        sys.exit(0)

    # Load compiled model
    logger.info("Loading compiled model for verification...")
    load_start = time.time()
    model.load(args.compile_dir)
    load_time = time.time() - load_start
    logger.info(f"Load + warmup completed in {load_time:.1f}s")

else:
    # Load existing compiled model
    logger.info(f"=== Loading compiled model from {args.compile_dir} ===")
    config = Qwen35MoeInferenceConfig(
        neuron_config=neuron_config,
        **config_dict,
    )
    model = NeuronQwen35MoeForCausalLM(model_path=args.model_dir, config=config)
    load_start = time.time()
    model.load(args.compile_dir)
    load_time = time.time() - load_start
    logger.info(f"Load completed in {load_time:.1f}s")

# Test generation
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

test_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    context_len = input_ids.shape[1]

    logger.info(f"\n--- Prompt: '{prompt[:60]}' (context_len={context_len}) ---")

    # TTFT
    model.reset()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=1)
    ttft = (time.time() - start) * 1000
    first_token = tokenizer.decode(outputs[0, context_len : context_len + 1])
    logger.info(f"TTFT: {ttft:.0f}ms, first token: '{first_token}'")

    # Full generation
    model.reset()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=32)
    total_time = time.time() - start
    new_tokens = outputs.shape[1] - context_len
    tkg_time = total_time - (ttft / 1000)
    tkg_rate = (new_tokens - 1) / tkg_time if tkg_time > 0 else 0
    response = tokenizer.decode(outputs[0, context_len:], skip_special_tokens=True)
    logger.info(f"Generated {new_tokens} tokens in {total_time:.2f}s")
    logger.info(f"TKG rate: {tkg_rate:.1f} tok/s")
    logger.info(f"Response: {response[:200]}")

logger.info("\n=== Test complete ===")
logger.info(f"Option: {args.option}")
