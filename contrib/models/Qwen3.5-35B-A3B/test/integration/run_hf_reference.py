#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run the HuggingFace Qwen3.5-35B-A3B reference on CPU and dump greedy outputs to JSON.

Because transformers 4.57.6 (shipped with NxDI SDK 2.29) predates Qwen3.5
support, this must be run in an isolated venv with transformers>=5.13:

    python3 -m venv /tmp/hf_ref_venv
    /tmp/hf_ref_venv/bin/pip install "transformers==5.13.0" "torch>=2.6" \\
        "safetensors" "sentencepiece" "accelerate"
    /tmp/hf_ref_venv/bin/python \\
        contrib/models/Qwen3.5-35B-A3B/test/integration/run_hf_reference.py \\
        --model-path /mnt/nvme/models/Qwen3.5-35B-A3B \\
        --max-new-tokens 16 \\
        --out-json /tmp/qwen35_35b_a3b_hf_reference.json

The output JSON can then be fed to compare_accuracy.py alongside a Neuron JSON.
"""

import argparse
import json

import torch


DEFAULT_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water boils at",
    "A haiku about autumn leaves:",
    "In one sentence, explain photosynthesis.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-35B-A3B")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[hf] loading {args.model_path} on CPU (bf16)")
    # For a VL model, AutoModelForCausalLM may or may not work. Try it first.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    results = []
    for i, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                enc.input_ids,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_ids = out[0].tolist()[enc.input_ids.shape[-1]:]
        text = tok.decode(out[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "hf_new_tokens": new_ids,
            "hf_text": text,
        })
        print(f"[hf] #{i}: {prompt!r} -> {new_ids}")

    with open(args.out_json, "w") as f:
        json.dump({"results": results, "max_new_tokens": args.max_new_tokens}, f, indent=2)
    print(f"[hf] wrote {args.out_json}")


if __name__ == "__main__":
    main()
