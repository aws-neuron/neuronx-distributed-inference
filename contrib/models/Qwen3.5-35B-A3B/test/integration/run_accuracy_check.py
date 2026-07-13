#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cross-check Qwen3.5-35B-A3B accuracy on Neuron against a HuggingFace CPU reference.

Runs a batch of prompts through both:
 (a) The compiled Neuron model (loaded from --compiled-path)
 (b) The HuggingFace text decoder on CPU (bf16, greedy)

For each prompt, both are run greedy (top_k=1 / do_sample=False) for
--max-new-tokens tokens. Reports per-prompt token match rate and overall
match rate.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-35B-A3B/test/integration/run_accuracy_check.py \\
        --compiled-path /tmp/qwen35_35b_a3b_traced \\
        --max-new-tokens 16
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water boils at",
    "A haiku about autumn leaves:",
    "In one sentence, explain photosynthesis.",
]


def load_neuron(compiled_path: str):
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM
    m = NeuronQwen35ForCausalLM(compiled_path)
    m.load(compiled_path)
    return m


def load_hf(model_path: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    # transformers 4.57.6 may not have Qwen3_5ForConditionalGeneration
    # registered by config's model_type. Try trust_remote_code.
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        # Load the text half only — vision would blow up CPU memory.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception:
        # Fall back to the text sub-config
        from transformers import AutoModel
        text_cfg = getattr(cfg, "text_config", cfg)
        model = AutoModelForCausalLM.from_config(text_cfg, torch_dtype=torch.bfloat16)
        raise RuntimeError(
            "HF Qwen3.5-35B-A3B did not load via AutoModelForCausalLM; "
            "cross-check on CPU is not supported for this model_type"
        )
    model.eval()
    return model


def generate_neuron(nxd_model, tok, prompt: str, max_new: int):
    from transformers import GenerationConfig
    import transformers
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    gen_cfg = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    gen_cfg.transformers_version = transformers.__version__
    gen_model = HuggingFaceGenerationAdapter(nxd_model)
    gen_model.generation_config.transformers_version = transformers.__version__

    enc = tok(prompt, return_tensors="pt")
    out = gen_model.generate(
        enc.input_ids, generation_config=gen_cfg, max_new_tokens=max_new,
    )
    new_ids = out[0].tolist()[enc.input_ids.shape[-1]:]
    return new_ids, tok.decode(out[0], skip_special_tokens=True)


def generate_hf(hf_model, tok, prompt: str, max_new: int):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = hf_model.generate(
            enc.input_ids,
            do_sample=False,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_ids = out[0].tolist()[enc.input_ids.shape[-1]:]
    return new_ids, tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-35B-A3B")
    ap.add_argument("--compiled-path", default="/tmp/qwen35_35b_a3b_traced")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--skip-hf", action="store_true", help="Only run Neuron")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[accuracy] loading Neuron model from {args.compiled_path}")
    nxd = load_neuron(args.compiled_path)

    hf_model = None
    if not args.skip_hf:
        print(f"[accuracy] loading HF reference (CPU, bf16) from {args.model_path}")
        try:
            hf_model = load_hf(args.model_path)
        except Exception as e:
            print(f"[accuracy] HF load failed ({e}); running Neuron-only")
            hf_model = None

    results = []
    total_match = 0
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        print(f"\n=== prompt #{i}: {prompt!r} ===")
        nxd_ids, nxd_txt = generate_neuron(nxd, tok, prompt, args.max_new_tokens)
        entry = {"prompt": prompt, "neuron_tokens": nxd_ids, "neuron_text": nxd_txt}
        print(f"[neuron]  ids={nxd_ids}")
        print(f"[neuron]  text={nxd_txt!r}")

        if hf_model is not None:
            hf_ids, hf_txt = generate_hf(hf_model, tok, prompt, args.max_new_tokens)
            entry["hf_tokens"] = hf_ids
            entry["hf_text"] = hf_txt
            L = min(len(nxd_ids), len(hf_ids))
            match = sum(1 for a, b in zip(nxd_ids[:L], hf_ids[:L]) if a == b)
            entry["match"] = match
            entry["match_denom"] = L
            total_match += match
            total_tokens += L
            print(f"[hf]      ids={hf_ids}")
            print(f"[hf]      text={hf_txt!r}")
            print(f"[match]   {match}/{L}")
        results.append(entry)

    if hf_model is not None:
        rate = total_match / max(1, total_tokens)
        print(f"\n=== TOTAL: {total_match}/{total_tokens} tokens match ({rate*100:.1f}%) ===")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"results": results,
                       "total_match": total_match,
                       "total_tokens": total_tokens}, f, indent=2)
        print(f"[accuracy] wrote {args.out_json}")


if __name__ == "__main__":
    main()
