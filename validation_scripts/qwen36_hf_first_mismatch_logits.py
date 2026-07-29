#!/usr/bin/env python3
"""Inspect HF logits at Neuron/HF first mismatch positions for Qwen3.6."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _insert_hf_ref(path: Path | None) -> None:
    if path is not None:
        sys.path.insert(0, str(path.expanduser().resolve()))


def _rank_of(logits, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def _token_entry(tokenizer: Any, token_id: int, logit: float, rank: int) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "rank": int(rank),
        "logit": float(logit),
        "text": tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hf-ref-pkgs", type=Path)
    parser.add_argument("--goldens-json", type=Path, required=True)
    parser.add_argument("--neuron-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _insert_hf_ref(args.hf_ref_pkgs)

    import torch  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    with args.goldens_json.expanduser().open(encoding="utf-8") as handle:
        goldens = json.load(handle)
    golden_by_index = {int(case["index"]): case for case in goldens["cases"]}

    with args.neuron_json.expanduser().open(encoding="utf-8") as handle:
        neuron = json.load(handle)
    mismatch_cases = [
        case
        for case in neuron["cases"]
        if case.get("first_mismatch") is not None
    ]
    if args.limit is not None:
        mismatch_cases = mismatch_cases[: args.limit]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_elapsed = time.perf_counter() - load_start

    rows = []
    with torch.no_grad():
        for neuron_case in mismatch_cases:
            case_index = int(neuron_case["index"])
            golden_case = golden_by_index[case_index]
            mismatch = neuron_case["first_mismatch"]
            position = int(mismatch["position"])
            expected_token = int(mismatch["expected"])
            actual_token = int(mismatch["actual"])

            prompt_ids = tokenizer.encode(
                golden_case["prompt"],
                add_special_tokens=False,
            )
            prefix_ids = prompt_ids + [
                int(token)
                for token in golden_case["hf_generated_tokens"][:position]
            ]
            input_ids = torch.tensor([prefix_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            elapsed = time.perf_counter() - start
            logits = outputs.logits[0, -1].float()
            top_values, top_indices = torch.topk(logits, k=args.top_k)
            top_tokens = [
                _token_entry(
                    tokenizer,
                    int(token_id),
                    float(logit),
                    rank + 1,
                )
                for rank, (logit, token_id) in enumerate(zip(top_values, top_indices))
            ]

            expected_rank = _rank_of(logits, expected_token)
            actual_rank = _rank_of(logits, actual_token)
            expected_logit = float(logits[expected_token].item())
            actual_logit = float(logits[actual_token].item())
            top1_logit = float(top_values[0].item())
            top2_logit = float(top_values[1].item()) if args.top_k > 1 else None
            row = {
                "index": case_index,
                "position": position,
                "prompt_tokens_reported": int(golden_case.get("prompt_tokens", 0)),
                "prompt_tokens_encoded": len(prompt_ids),
                "prefix_tokens_evaluated": len(prefix_ids),
                "hf_expected": _token_entry(
                    tokenizer,
                    expected_token,
                    expected_logit,
                    expected_rank,
                ),
                "neuron_actual": _token_entry(
                    tokenizer,
                    actual_token,
                    actual_logit,
                    actual_rank,
                ),
                "top1_minus_expected": top1_logit - expected_logit,
                "expected_minus_neuron": expected_logit - actual_logit,
                "top1_minus_top2": (
                    top1_logit - top2_logit if top2_logit is not None else None
                ),
                "forward_seconds": elapsed,
                "top_tokens": top_tokens,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "index": row["index"],
                        "position": row["position"],
                        "expected_rank": expected_rank,
                        "actual_rank": actual_rank,
                        "expected_minus_neuron": row["expected_minus_neuron"],
                        "top1_minus_expected": row["top1_minus_expected"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = {
        "stage": "hf_first_mismatch_logits",
        "model_path": str(args.model_path),
        "goldens_json": str(args.goldens_json),
        "neuron_json": str(args.neuron_json),
        "hf_ref_pkgs": str(args.hf_ref_pkgs) if args.hf_ref_pkgs else None,
        "dtype": args.dtype,
        "top_k": args.top_k,
        "hf_load_seconds": load_elapsed,
        "num_mismatches_checked": len(rows),
        "cases": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
