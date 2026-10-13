#!/usr/bin/env python3
"""Compare Qwen3.6 Neuron greedy tokens against saved HF goldens."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_bench(repo_root: Path):
    sys.path.insert(0, str(repo_root / "validation_scripts"))
    import qwen36_offline_decode_bench as bench  # noqa: WPS433

    return bench


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compiled-artifacts", type=Path, required=True)
    parser.add_argument("--goldens-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--cte-buckets")
    parser.add_argument("--context-encoding-bucket-pairs")
    parser.add_argument("--token-generation-buckets")
    parser.add_argument("--token-generation-batches")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--pa-num-blocks", type=int)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=64)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)


def _prepare_args(args: argparse.Namespace, bench: Any) -> None:
    args.repo_root = args.repo_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_artifacts = args.compiled_artifacts.expanduser().resolve()
    resolved = bench._resolve_config_defaults(args)
    args.artifact_config = resolved["artifact_config"]
    args.seq_len = resolved["seq_len"]
    args.max_model_len = resolved["max_model_len"]
    args.resolved_cte_buckets = resolved["cte_buckets"]
    args.resolved_context_encoding_bucket_pairs = resolved[
        "context_encoding_bucket_pairs"
    ]
    args.resolved_token_generation_buckets = resolved["token_generation_buckets"]
    args.resolved_token_generation_batches = resolved["token_generation_batches"]
    args.pa_num_blocks = resolved["pa_num_blocks"]
    args.prompt = ""
    args.warmup_tokens = min(4, int(args.max_tokens or 4))


def _position_match(expected: list[int], actual: list[int]) -> dict[str, Any]:
    total = max(len(expected), len(actual))
    compared = min(len(expected), len(actual))
    matches = sum(1 for left, right in zip(expected, actual) if left == right)
    first_mismatch = None
    for index in range(total):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            first_mismatch = {
                "position": index,
                "expected": left,
                "actual": right,
            }
            break
    return {
        "expected_len": len(expected),
        "actual_len": len(actual),
        "compared_positions": compared,
        "total_positions": total,
        "matches": matches,
        "match_rate": (matches / total) if total else 1.0,
        "first_mismatch": first_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    _add_runtime_args(parser)
    args = parser.parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    bench = _load_bench(args.repo_root)

    with args.goldens_json.expanduser().open(encoding="utf-8") as handle:
        goldens = json.load(handle)
    cases = list(goldens["cases"])
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.max_tokens is None:
        args.max_tokens = int(goldens.get("max_new_tokens") or 16)

    _prepare_args(args, bench)
    bench._ensure_paths(args.repo_root)
    bench._ensure_runtime_env(args)

    llm = None
    try:
        llm, sampling, warmup_sampling = bench._build_llm(args)
        if args.warmup and cases:
            llm.generate([cases[0]["prompt"]], warmup_sampling)

        from transformers import AutoTokenizer  # noqa: WPS433

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        rows = []
        start_all = time.perf_counter()
        for case in cases:
            expected = [
                int(item)
                for item in case["hf_generated_tokens"][: args.max_tokens]
            ]
            start = time.perf_counter()
            outputs = llm.generate([case["prompt"]], sampling)
            elapsed = time.perf_counter() - start
            actual = [int(item) for item in outputs[0].outputs[0].token_ids]
            comparison = _position_match(expected, actual)
            row = {
                "index": int(case["index"]),
                "prompt_tokens": int(case.get("prompt_tokens", 0)),
                "elapsed_seconds": elapsed,
                "tok_s": (len(actual) / elapsed) if elapsed > 0 else None,
                "hf_generated_tokens": expected,
                "neuron_generated_tokens": actual,
                "hf_text": case.get("hf_text", ""),
                "neuron_text": tokenizer.decode(
                    actual,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **comparison,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "index": row["index"],
                        "match_rate": row["match_rate"],
                        "matches": row["matches"],
                        "total_positions": row["total_positions"],
                        "tok_s": row["tok_s"],
                        "first_mismatch": row["first_mismatch"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        elapsed_all = time.perf_counter() - start_all
        total_matches = sum(int(row["matches"]) for row in rows)
        total_positions = sum(int(row["total_positions"]) for row in rows)
        report = {
            "stage": "neuron_vs_hf_greedy_match",
            "goldens_json": str(args.goldens_json.expanduser()),
            "artifact": str(args.compiled_artifacts),
            "model_path": str(args.model_path),
            "max_tokens": args.max_tokens,
            "num_cases": len(rows),
            "overall_matches": total_matches,
            "overall_positions": total_positions,
            "overall_match_rate": (
                total_matches / total_positions if total_positions else 1.0
            ),
            "elapsed_seconds": elapsed_all,
            "avg_tok_s": (
                sum(float(row["tok_s"]) for row in rows if row["tok_s"])
                / max(1, sum(1 for row in rows if row["tok_s"]))
            ),
            "pa_num_blocks": args.pa_num_blocks,
            "cte_buckets": args.resolved_cte_buckets,
            "context_encoding_bucket_pairs": args.resolved_context_encoding_bucket_pairs,
            "token_generation_buckets": args.resolved_token_generation_buckets,
            "async_mode": args.async_mode,
            "cases": rows,
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.output_json:
            args.output_json.expanduser().write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if shutdown is not None:
                shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
