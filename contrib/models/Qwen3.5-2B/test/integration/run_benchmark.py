#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TTFT / TPOT benchmark for Qwen3.5-2B on Neuron.

For each prompt length, we run:
  * a 1-token generation (measures TTFT = prefill + first-decode)
  * an N-token generation (extracts TPOT = (elapsed - TTFT) / (N-1))

Results are averaged across --repeats runs; the first run is treated as warmup
and discarded.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python contrib/models/Qwen3.5-2B/test/integration/run_benchmark.py \\
        --compiled-path /tmp/qwen35_2b_traced \\
        --seq-len 512 --max-new-tokens 64 --repeats 5
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)


def build_prompt_at_length(tok, target_len: int) -> str:
    """Build a prompt whose tokenization length equals target_len (approx)."""
    base = ("Once upon a time, there lived a curious explorer who traveled "
            "across mountains and seas. ")
    # Grow until at least target_len tokens
    prompt = base
    while True:
        ids = tok(prompt, return_tensors="pt").input_ids[0]
        if len(ids) >= target_len:
            break
        prompt += base
    ids = tok(prompt, return_tensors="pt").input_ids[0][:target_len]
    return tok.decode(ids, skip_special_tokens=True)


def bench_once(gen_model, tok, prompt: str, gen_cfg, max_new: int):
    enc = tok(prompt, return_tensors="pt")
    # TTFT
    t0 = time.perf_counter()
    _ = gen_model.generate(enc.input_ids, generation_config=gen_cfg, max_new_tokens=1)
    ttft = (time.perf_counter() - t0) * 1000.0

    # Full run
    t1 = time.perf_counter()
    out = gen_model.generate(
        enc.input_ids, generation_config=gen_cfg, max_new_tokens=max_new,
    )
    total = (time.perf_counter() - t1) * 1000.0
    n_new = out.shape[-1] - enc.input_ids.shape[-1]
    n_decode = max(1, n_new - 1)
    tpot = (total - ttft) / n_decode
    return ttft, tpot, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/nvme/models/Qwen3.5-2B")
    ap.add_argument("--compiled-path", default="/tmp/qwen35_2b_traced")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, GenerationConfig
    import transformers
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    print(f"[bench] loading from {args.compiled_path}")
    m = NeuronQwen35ForCausalLM(args.compiled_path)
    m.load(args.compiled_path)

    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_cfg = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    gen_cfg.transformers_version = transformers.__version__
    gen_model = HuggingFaceGenerationAdapter(m)
    gen_model.generation_config.transformers_version = transformers.__version__

    results = []
    for plen in args.prompt_lens:
        prompt = build_prompt_at_length(tok, plen)
        actual_plen = len(tok(prompt, return_tensors="pt").input_ids[0])

        ttfts = []
        tpots = []
        for r in range(args.repeats + 1):  # +1 warmup
            ttft, tpot, n_new = bench_once(
                gen_model, tok, prompt, gen_cfg, args.max_new_tokens
            )
            if r == 0:
                print(f"[bench] len={actual_plen} warmup: TTFT={ttft:.1f} ms TPOT={tpot:.2f} ms (discarded)")
                continue
            ttfts.append(ttft)
            tpots.append(tpot)
            print(f"[bench] len={actual_plen} r={r}: TTFT={ttft:.1f} ms TPOT={tpot:.2f} ms n_new={n_new}")

        r = {
            "prompt_len": actual_plen,
            "target_prompt_len": plen,
            "ttft_ms_mean": statistics.mean(ttfts),
            "ttft_ms_median": statistics.median(ttfts),
            "ttft_ms_stdev": statistics.pstdev(ttfts) if len(ttfts) > 1 else 0.0,
            "tpot_ms_mean": statistics.mean(tpots),
            "tpot_ms_median": statistics.median(tpots),
            "tpot_ms_stdev": statistics.pstdev(tpots) if len(tpots) > 1 else 0.0,
            "throughput_tok_per_s_mean": 1000.0 / statistics.mean(tpots),
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
        }
        results.append(r)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  prompt_len={r['prompt_len']:4d}"
              f"  TTFT={r['ttft_ms_median']:7.1f} ms"
              f"  TPOT={r['tpot_ms_median']:6.2f} ms"
              f"  ({r['throughput_tok_per_s_mean']:6.1f} tok/s)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
