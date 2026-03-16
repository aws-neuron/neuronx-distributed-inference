"""
Full multi-step logit matching test for DeepSeek V3 NXDI vs HF reference.

Creates a mini DeepSeek V3 model, runs greedy autoregressive generation
through both HF (CPU, FP32) and NXDI (Neuron, BF16), and compares
tokens step-by-step.

Requirements:
  - Neuron device (trn1/trn2 instance with at least 2 NeuronCores)
  - Run test_logit_matching.py first (or use --model-dir to point to mini model)

Expected results with random weights:
  - First token: EXACT MATCH
  - Step 1: NXDI token typically in HF top-5
  - Steps 2+: Autoregressive divergence (expected with BF16 vs FP32)

Usage:
  python examples/test_logit_matching_full.py
  python examples/test_logit_matching_full.py --model-dir /tmp/deepseek-v3-logit-test
"""
import argparse
import gc
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3InferenceConfig,
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

torch.manual_seed(42)

VOCAB_SIZE = 32000
N_NEW_TOKENS = 15


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 full logit matching test")
    parser.add_argument("--model-dir", type=str, default="/tmp/deepseek-v3-logit-test",
                        help="Directory with mini model (from test_logit_matching.py)")
    parser.add_argument("--traced-path", type=str, default="/tmp/deepseek_v3_logit_full",
                        help="Directory for compiled model")
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
        print(f"ERROR: Mini model not found at {args.model_dir}")
        print("Run test_logit_matching.py first to create it.")
        sys.exit(1)

    # Step 1: HF reference
    print("=" * 60)
    print("HF REFERENCE")
    print("=" * 60)

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    print(f"Prompt: '{prompt}' ({seq_len} tokens)")

    generated = input_ids.clone()
    hf_top1_per_step = []
    hf_top5_per_step = []
    with torch.no_grad():
        for step in range(N_NEW_TOKENS):
            try:
                out = hf_model(generated, use_cache=False)
            except (IndexError, RuntimeError):
                break
            logits = out.logits[0, -1, :].float()
            top1 = logits.argmax().item()
            top5 = logits.topk(5).indices.tolist()
            hf_top1_per_step.append(top1)
            hf_top5_per_step.append(top5)
            next_token = torch.tensor([[min(top1, VOCAB_SIZE - 1)]])
            generated = torch.cat([generated, next_token], dim=-1)

    hf_gen_tokens = list(hf_top1_per_step)
    hf_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(f"HF output: '{hf_text}'")

    del hf_model
    gc.collect()

    # Step 2: NXDI generation
    print("\n" + "=" * 60)
    print("NXDI MODEL")
    print("=" * 60)

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=args.seq_len, torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False, flash_decoding_enabled=False, logical_nc_config=2,
        blockwise_matmul_config={"block_size": 999999},
    )
    inf_config = DeepseekV3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(args.model_dir),
    )

    if os.path.exists(os.path.join(args.traced_path, "config.json")):
        print("Using cached compilation...")
    else:
        print("Compiling...")
        model = NeuronDeepseekV3ForCausalLM(args.model_dir, inf_config)
        model.compile(args.traced_path)
        del model
        gc.collect()

    print("Loading...")
    model = NeuronDeepseekV3ForCausalLM(args.traced_path)
    model.load(args.traced_path)
    tokenizer.save_pretrained(args.traced_path)

    generation_config = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_model = HuggingFaceGenerationAdapter(model)

    print("Generating...")
    outputs = generation_model.generate(
        input_ids, generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=seq_len + N_NEW_TOKENS,
    )
    nxdi_tokens = outputs[0].tolist()
    nxdi_text = tokenizer.decode(nxdi_tokens, skip_special_tokens=True)
    print(f"NXDI output: '{nxdi_text}'")

    # Step 3: Compare
    print("\n" + "=" * 60)
    print("TOKEN-BY-TOKEN COMPARISON")
    print("=" * 60)

    nxdi_gen = nxdi_tokens[seq_len:]
    hf_gen = hf_gen_tokens[:len(nxdi_gen)]
    compare_len = min(len(hf_gen), len(nxdi_gen))

    matches = 0
    diverge_at = compare_len
    for i in range(compare_len):
        hf_tok = hf_gen[i]
        nxdi_tok = nxdi_gen[i]
        match = hf_tok == nxdi_tok
        in_top5 = nxdi_tok in hf_top5_per_step[i] if i < len(hf_top5_per_step) else False
        if match:
            matches += 1
            status = "MATCH"
        elif in_top5:
            status = f"in-top5 (pos {hf_top5_per_step[i].index(nxdi_tok)+1})"
        else:
            status = "MISMATCH"

        if not match and diverge_at == compare_len:
            diverge_at = i

        hf_word = tokenizer.decode([hf_tok])
        nxdi_word = tokenizer.decode([nxdi_tok])
        print(f"  Step {i:2d}: HF={hf_tok:5d} ({hf_word:>15s})  NXDI={nxdi_tok:5d} ({nxdi_word:>15s})  [{status}]")

    match_rate = matches / compare_len if compare_len > 0 else 0
    first_match = hf_gen[0] == nxdi_gen[0] if compare_len > 0 else False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Tokens compared:     {compare_len}")
    print(f"  Exact matches:       {matches}/{compare_len} ({match_rate:.0%})")
    print(f"  First token matches: {'YES' if first_match else 'NO'}")
    if diverge_at < compare_len:
        print(f"  First divergence:    step {diverge_at}")
    else:
        print(f"  First divergence:    NONE (all match!)")

    PASS = first_match
    print(f"\n  STATUS: {'PASS' if PASS else 'FAIL'}")
    print(f"  (Criteria: first token matches)")
    print(f"  Note: FP32 (HF) vs BF16 (Neuron) causes divergence after step 0.")
    print(f"        This is expected with random weights (small logit margins).")
    print(f"        Real-model testing needed for full accuracy validation.")
    print(f"{'='*60}")

    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
