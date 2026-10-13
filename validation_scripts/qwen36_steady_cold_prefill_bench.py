#!/usr/bin/env python3
"""Steady-state cold-prefill benchmark for Qwen3.6 Hybrid APC artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import qwen36_hybrid_apc_context_sweep as context_sweep
import qwen36_hybrid_apc_validation as hybrid_validation


def _row_for_prompt(
    *,
    llm: Any,
    sampling: Any,
    tokenizer: Any,
    vocab_size: int,
    dummy_ids: set[int],
    target_tokens: int,
    suffix_tokens: int,
    role_index: int,
    phase: str,
    run: int,
) -> dict[str, Any]:
    prompt = context_sweep._prompt_for_length(
        tokenizer,
        target_tokens=target_tokens,
        suffix_tokens=suffix_tokens,
        role_index=role_index,
    )
    result = context_sweep._generate(llm, sampling, prompt)
    generated_tokens = [int(token_id) for token_id in result["generated_tokens"]]
    invalid_token_ids = [
        token_id for token_id in generated_tokens if token_id < 0 or token_id >= vocab_size
    ]
    non_dummy = [token_id for token_id in generated_tokens if token_id not in dummy_ids]
    elapsed = float(result["elapsed_seconds"])
    row = {
        "phase": phase,
        "run": run,
        "target_prompt_tokens": target_tokens,
        "actual_prompt_tokens": len(prompt["prompt_token_ids"]),
        "suffix_tokens": suffix_tokens,
        "elapsed_seconds": elapsed,
        "effective_prompt_tokens_per_second": (
            target_tokens / elapsed if elapsed > 0 else None
        ),
        "generated_text": result["generated_text"],
        "generated_token_count": result["generated_token_count"],
        "generated_tokens": generated_tokens,
        "unique_generated_tokens": sorted(set(generated_tokens)),
        "token_range_passed": not invalid_token_ids,
        "invalid_token_ids": sorted(set(invalid_token_ids)),
        "real_tokens_passed": bool(non_dummy),
        "non_dummy_generated_token_count": len(non_dummy),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [
        float(row["effective_prompt_tokens_per_second"])
        for row in rows
        if row.get("effective_prompt_tokens_per_second") is not None
    ]
    if not rates:
        return {
            "count": 0,
            "avg_effective_prompt_tokens_per_second": None,
            "min_effective_prompt_tokens_per_second": None,
            "max_effective_prompt_tokens_per_second": None,
        }
    return {
        "count": len(rates),
        "avg_effective_prompt_tokens_per_second": sum(rates) / len(rates),
        "min_effective_prompt_tokens_per_second": min(rates),
        "max_effective_prompt_tokens_per_second": max(rates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compiled-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--role-index-base", type=int, default=700000)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--cte-buckets")
    parser.add_argument("--context-encoding-bucket-pairs", nargs="+", default=None)
    parser.add_argument("--pa-num-blocks", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--ctx-batch-size", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", default=None)
    parser.add_argument("--token-generation-batches", nargs="+", default=None)
    parser.add_argument("--async-mode", action="store_true", default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=64)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-apc-prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument("--skip-fp8-env", action="store_true")
    parser.add_argument("--require-real-tokens", action="store_true")
    parser.add_argument("--dummy-token-ids", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_artifacts = args.compiled_artifacts.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()
    artifact_config = context_sweep._artifact_neuron_config(args.compiled_artifacts)
    runtime_args = context_sweep._build_args(args, artifact_config)

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    vocab_size = context_sweep._effective_vocab_size(args.model_path, tokenizer)
    configured_dummy_ids = {int(token_id) for token_id in args.dummy_token_ids}
    dummy_ids = configured_dummy_ids | hybrid_validation._effective_dummy_token_ids(
        runtime_args,
        tokenizer,
    )

    if args.prompt_tokens + args.max_tokens > runtime_args.seq_len:
        raise ValueError(
            "prompt_tokens + max_tokens exceeds seq_len: "
            f"{args.prompt_tokens} + {args.max_tokens} > {runtime_args.seq_len}"
        )

    rows: list[dict[str, Any]] = []
    llm = None
    try:
        llm, sampling = hybrid_validation._build_llm(
            runtime_args,
            enable_hybrid_apc=True,
        )
        total_runs = args.warmup_runs + args.measured_runs
        for run in range(total_runs):
            phase = "warmup" if run < args.warmup_runs else "measured"
            role_index = args.role_index_base + (run * 1009)
            rows.append(
                _row_for_prompt(
                    llm=llm,
                    sampling=sampling,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    dummy_ids=dummy_ids,
                    target_tokens=args.prompt_tokens,
                    suffix_tokens=args.suffix_tokens,
                    role_index=role_index,
                    phase=phase,
                    run=run + 1,
                )
            )
    finally:
        if llm is not None:
            hybrid_validation._shutdown_llm(llm)

    measured_rows = [row for row in rows if row["phase"] == "measured"]
    correctness_passed = all(
        row["token_range_passed"]
        and (row["real_tokens_passed"] or not args.require_real_tokens)
        for row in measured_rows
    )
    report = {
        "artifact": str(args.compiled_artifacts),
        "artifact_neuron_config": {
            key: artifact_config.get(key)
            for key in (
                "seq_len",
                "max_context_length",
                "context_encoding_buckets",
                "context_encoding_bucket_pairs",
                "prefix_buckets",
                "token_generation_buckets",
                "output_logits",
                "pa_num_blocks",
            )
        },
        "prompt_tokens": args.prompt_tokens,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "configured_dummy_token_ids": sorted(configured_dummy_ids),
        "effective_dummy_token_ids": sorted(dummy_ids),
        "vocab_size": vocab_size,
        "correctness_passed": correctness_passed,
        "summary": _summarize(measured_rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report["summary"], sort_keys=True), flush=True)
    return 0 if correctness_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
