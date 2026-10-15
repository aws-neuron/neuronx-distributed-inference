#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Apples-to-apples TKG accuracy comparison: CPU reference vs Neuron.

Generates identical tokenized inputs, runs both CPU and Neuron generation,
and compares tokens position-by-position.

Phase 1: Generate CPU references (slow, ~5-15 min per prompt for 7B in bf16)
Phase 2: Run Neuron generation with same inputs
Phase 3: Compare token-by-token

Usage:
    # Generate CPU references (run once, saves to disk)
    python accuracy_test.py --cpu-ref

    # Run Neuron comparison against saved references
    python accuracy_test.py --neuron --skip-compile

    # Run both (full test)
    python accuracy_test.py --full --skip-compile
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_PATH = "/mnt/models/neuron_models/InternVL3-8B-VLM/"
REF_DIR = "/mnt/models/accuracy_refs/"
TEST_IMAGE = "/mnt/models/test_image.png"

# Test prompts
TEST_PROMPTS = [
    {
        "name": "france_capital",
        "prompt": "What is the capital of France?",
        "type": "text",
        "max_new_tokens": 32,
    },
    {
        "name": "meaning_of_life",
        "prompt": "I believe the meaning of life is",
        "type": "text",
        "max_new_tokens": 32,
    },
    {
        "name": "simple_math",
        "prompt": "What is 2 + 2? Answer with just the number:",
        "type": "text",
        "max_new_tokens": 10,
    },
    {
        "name": "describe_image",
        "prompt": "Describe this image briefly.",
        "type": "multimodal",
        "max_new_tokens": 32,
    },
]

# Special tokens
IMG_CONTEXT_ID = 151667
IMG_START_ID = 151665
IMG_END_ID = 151666


def build_inputs(prompt_info, tokenizer, pixel_values=None):
    """Build identical inputs for both CPU and Neuron using chat template."""
    prompt = prompt_info["prompt"]

    # Use chat template for proper instruction formatting
    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(templated, return_tensors="pt")
    input_ids = inputs.input_ids

    if prompt_info["type"] == "multimodal" and pixel_values is not None:
        # Insert image tokens before the text: <img> <IMG_CONTEXT>*256 </img> <text>
        # InternVL3 convention: image tokens come in the user message
        text_ids = input_ids[0]
        img_tokens = torch.full((256,), IMG_CONTEXT_ID, dtype=torch.long)
        img_seq = torch.cat(
            [
                torch.tensor([IMG_START_ID]),
                img_tokens,
                torch.tensor([IMG_END_ID]),
            ]
        )
        # Insert image tokens right after the "user\n" tokens
        # Find position of user content in the templated string
        input_ids = torch.cat([text_ids, img_seq]).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def generate_cpu_references(tokenizer):
    """Generate reference outputs using HF model on CPU (slow but golden).

    For text-only prompts: uses the inner Qwen2ForCausalLM language model directly.
    For multimodal: uses InternVLChatModel.chat() which handles vision processing.
    """
    print("=" * 60)
    print("PHASE 1: Generating CPU reference outputs")
    print("=" * 60)
    print("WARNING: This is slow (~5-15 min per prompt for 7B bf16 on CPU)")

    from transformers import AutoModelForCausalLM

    print("\nLoading HF InternVLChatModel on CPU (bf16)...")
    start = time.time()
    vlm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    vlm_model.eval()
    # InternVLChatModel.generate() asserts img_context_token_id is set
    vlm_model.img_context_token_id = IMG_CONTEXT_ID
    # Get the inner language model for text-only generation
    lm = vlm_model.language_model
    lm.eval()
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Load test image
    pixel_values = None
    if os.path.exists(TEST_IMAGE):
        from PIL import Image
        from torchvision import transforms

        img = Image.open(TEST_IMAGE).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16)
        print(f"Test image loaded: {TEST_IMAGE}")

    os.makedirs(REF_DIR, exist_ok=True)
    results = {}

    # EOS for InternVL3 is <|im_end|> = 151645
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"Using eos_token_id={eos_token_id} (<|im_end|>)")

    for pinfo in TEST_PROMPTS:
        name = pinfo["name"]
        if pinfo["type"] == "multimodal" and pixel_values is None:
            print(f"\nSkipping {name} (no test image)")
            continue

        # Skip if reference already exists
        ref_path = os.path.join(REF_DIR, f"{name}.json")
        if os.path.exists(ref_path):
            print(f"\n--- {name}: already exists, skipping ---")
            with open(ref_path) as f:
                results[name] = json.load(f)
            continue

        print(f"\n--- {name} ---")
        pv = pixel_values if pinfo["type"] == "multimodal" else None
        input_ids, attention_mask = build_inputs(pinfo, tokenizer, pv)
        prompt_len = input_ids.shape[-1]
        print(f"Input length: {prompt_len} tokens")

        # Generate on CPU
        print(f"Generating {pinfo['max_new_tokens']} tokens on CPU...")
        start = time.time()
        with torch.no_grad():
            if pinfo["type"] == "multimodal" and pv is not None:
                # Use full VLM model for multimodal (handles vision encoder)
                output_ids = vlm_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pv,
                    max_new_tokens=pinfo["max_new_tokens"],
                    do_sample=False,
                    eos_token_id=eos_token_id,
                )
            else:
                # Use inner LM directly for text-only (cleaner, avoids custom generate)
                output_ids = lm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=pinfo["max_new_tokens"],
                    do_sample=False,
                    eos_token_id=eos_token_id,
                )
        elapsed = time.time() - start
        print(f"Generated in {elapsed:.1f}s")

        generated_ids = output_ids[0, prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Output: {generated_text!r}")

        ref = {
            "name": name,
            "prompt": pinfo["prompt"],
            "type": pinfo["type"],
            "input_ids": input_ids[0].tolist(),
            "prompt_len": prompt_len,
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "max_new_tokens": pinfo["max_new_tokens"],
            "cpu_time_s": round(elapsed, 1),
        }

        ref_path = os.path.join(REF_DIR, f"{name}.json")
        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"Saved: {ref_path}")
        results[name] = ref

    del vlm_model, lm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


def run_neuron_comparison(tokenizer, skip_compile=False):
    """Run Neuron generation and compare against CPU references."""
    print("\n" + "=" * 60)
    print("PHASE 2: Neuron generation + comparison")
    print("=" * 60)

    from modeling_internvl3 import InternVL3InferenceConfig, NeuronInternVL3ForCausalLM
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    # Create config and load model
    text_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        seq_len=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        save_sharded_checkpoint=True,
    )
    vision_neuron_config = NeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        buckets=[1],
        fused_qkv=True,
        save_sharded_checkpoint=True,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        MODEL_PATH,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    model = NeuronInternVL3ForCausalLM(MODEL_PATH, config=config)
    if not skip_compile:
        model.compile(COMPILED_PATH)
    model.load(COMPILED_PATH)

    adapter = HuggingFaceGenerationAdapter(model)

    # Load test image
    pixel_values = None
    if os.path.exists(TEST_IMAGE):
        from PIL import Image
        from torchvision import transforms

        img = Image.open(TEST_IMAGE).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        pixel_values = transform(img).unsqueeze(0)

    all_results = {}

    for pinfo in TEST_PROMPTS:
        name = pinfo["name"]
        ref_path = os.path.join(REF_DIR, f"{name}.json")
        if not os.path.exists(ref_path):
            print(f"\n--- {name}: SKIP (no CPU reference) ---")
            continue

        with open(ref_path) as f:
            ref = json.load(f)

        print(f"\n--- {name} ---")
        print(f"Prompt: {ref['prompt']!r}")

        # Build IDENTICAL inputs
        pv = pixel_values if pinfo["type"] == "multimodal" else None
        input_ids, attention_mask = build_inputs(pinfo, tokenizer, pv)
        prompt_len = input_ids.shape[-1]

        # Verify inputs match reference
        ref_input_ids = ref["input_ids"]
        actual_input_ids = input_ids[0].tolist()
        if actual_input_ids != ref_input_ids:
            print(f"WARNING: Input IDs differ!")
            print(f"  Neuron: {actual_input_ids[:10]}... (len={len(actual_input_ids)})")
            print(f"  CPU:    {ref_input_ids[:10]}... (len={len(ref_input_ids)})")

        # Generate on Neuron
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=pinfo["max_new_tokens"],
            do_sample=False,
        )
        if pv is not None:
            gen_kwargs["pixel_values"] = pv

        start = time.time()
        output_ids = adapter.generate(**gen_kwargs)
        neuron_time = time.time() - start

        neuron_generated_ids = output_ids[0, prompt_len:].tolist()
        neuron_text = tokenizer.decode(neuron_generated_ids, skip_special_tokens=True)

        # Compare token-by-token (trim at first EOS for both sides)
        cpu_ids = ref["generated_ids"]
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Trim at first EOS
        def trim_at_eos(ids, eos):
            for i, t in enumerate(ids):
                if t == eos:
                    return ids[:i]
            return ids

        neuron_trimmed = trim_at_eos(neuron_generated_ids, eos_id)
        cpu_trimmed = trim_at_eos(cpu_ids, eos_id)

        n_compare = min(
            max(len(neuron_trimmed), len(cpu_trimmed)),
            pinfo["max_new_tokens"],
        )
        # Use max length for comparison (missing tokens count as mismatches)
        if n_compare == 0:
            n_compare = max(len(neuron_generated_ids), len(cpu_ids))

        matches = 0
        first_mismatch = None
        details = []
        for i in range(n_compare):
            n_id = neuron_trimmed[i] if i < len(neuron_trimmed) else -1
            c_id = cpu_trimmed[i] if i < len(cpu_trimmed) else -1
            match = n_id == c_id
            if match:
                matches += 1
            elif first_mismatch is None:
                first_mismatch = i

            n_tok = tokenizer.decode([n_id]) if n_id >= 0 else "<EOS>"
            c_tok = tokenizer.decode([c_id]) if c_id >= 0 else "<EOS>"
            details.append(
                {
                    "pos": i,
                    "match": match,
                    "neuron_id": n_id,
                    "neuron_token": n_tok,
                    "cpu_id": c_id,
                    "cpu_token": c_tok,
                }
            )

        match_rate = matches / n_compare if n_compare > 0 else 0

        # Compute prefix match (up to shorter EOS-trimmed sequence)
        min_len = min(len(neuron_trimmed), len(cpu_trimmed))
        prefix_matches = 0
        for i in range(min_len):
            if neuron_trimmed[i] == cpu_trimmed[i]:
                prefix_matches += 1
            else:
                break  # Stop at first divergence for prefix match
        prefix_rate = prefix_matches / min_len if min_len > 0 else 1.0

        # Print results
        print(f"CPU output:    {ref['generated_text']!r}")
        print(f"Neuron output: {neuron_text!r}")
        print(
            f"Tokens (EOS-trimmed): neuron={len(neuron_trimmed)}, cpu={len(cpu_trimmed)}"
        )
        print(f"Prefix match:  {prefix_matches}/{min_len} ({prefix_rate:.1%})")
        print(f"Full match:    {matches}/{n_compare} ({match_rate:.1%})")
        if first_mismatch is not None:
            d = details[first_mismatch]
            print(
                f"First mismatch at pos {first_mismatch}: "
                f"neuron={d['neuron_token']!r} vs cpu={d['cpu_token']!r}"
            )

        # Detailed per-position output
        for d in details:
            flag = "MATCH" if d["match"] else "MISS "
            print(
                f"  pos {d['pos']:2d}: {flag} | neuron={d['neuron_token']!r:12s} cpu={d['cpu_token']!r:12s}"
            )

        # Status based on prefix match (more meaningful for EOS-trimmed comparison)
        status = (
            "PASS" if prefix_rate >= 0.9 else ("WARN" if prefix_rate >= 0.5 else "FAIL")
        )
        print(f"Status: {status}")

        all_results[name] = {
            "status": status,
            "prefix_match": f"{prefix_matches}/{min_len} ({prefix_rate:.1%})",
            "full_match": f"{matches}/{n_compare} ({match_rate:.1%})",
            "neuron_tokens": len(neuron_trimmed),
            "cpu_tokens": len(cpu_trimmed),
            "first_mismatch_pos": first_mismatch,
            "neuron_text": neuron_text[:200],
            "cpu_text": ref["generated_text"][:200],
            "neuron_time_s": round(neuron_time, 3),
            "cpu_time_s": ref.get("cpu_time_s"),
            "details": details,
        }

    # Summary
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON SUMMARY")
    print("=" * 60)
    overall_pass = True
    for name, r in all_results.items():
        print(
            f"  {name:25s} {r['status']:5s}  prefix={r['prefix_match']}  full={r['full_match']}"
        )
        if r["status"] == "FAIL":
            overall_pass = False

    result_path = os.path.join(REF_DIR, "accuracy_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")

    return overall_pass, all_results


def main():
    parser = argparse.ArgumentParser(description="Apples-to-apples accuracy test")
    parser.add_argument(
        "--cpu-ref", action="store_true", help="Generate CPU references only"
    )
    parser.add_argument(
        "--neuron", action="store_true", help="Run Neuron comparison only"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run both CPU ref + Neuron comparison"
    )
    parser.add_argument(
        "--skip-compile", action="store_true", help="Skip Neuron compilation"
    )
    args = parser.parse_args()

    if not any([args.cpu_ref, args.neuron, args.full]):
        parser.print_help()
        print("\nSpecify --cpu-ref, --neuron, or --full")
        sys.exit(1)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if args.cpu_ref or args.full:
        generate_cpu_references(tokenizer)

    if args.neuron or args.full:
        passed, results = run_neuron_comparison(tokenizer, args.skip_compile)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
