#!/usr/bin/env python3
"""Boundary-aligned APC probe for a Qwen3.6 OpenAI-compatible server.

This probe intentionally uses raw ``/v1/completions`` token-id prompts so the
prompt length is exactly the checkpoint boundary under test. If running behind
``qwen36_chat_proxy.py``, start that proxy with ``--allow-completions`` or send
this probe directly to the vLLM server.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split()]
    if not lengths:
        raise ValueError("at least one boundary length is required")
    return lengths


def _load_json_from_url(url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload


def _detect_model(base_url: str, fallback: str, timeout: float) -> str:
    status, payload = _load_json_from_url(base_url.rstrip("/") + "/v1/models", timeout)
    if status < 400:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list) and data:
            model_id = data[0].get("id")
            if isinstance(model_id, str) and model_id:
                return model_id
    return fallback


def _metric_snapshot(base_url: str, timeout: float) -> dict[str, float]:
    try:
        data = urllib.request.urlopen(
            base_url.rstrip("/") + "/metrics",
            timeout=timeout,
        ).read().decode("utf-8")
    except (OSError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError):
        return {}

    wanted: dict[str, float] = {}
    for line in data.splitlines():
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if line.startswith("vllm:prefix_cache_queries_total"):
            wanted["prefix_cache_queries_total"] = value
        elif line.startswith("vllm:prefix_cache_hits_total"):
            wanted["prefix_cache_hits_total"] = value
        elif line.startswith("vllm:prompt_tokens_cached_total"):
            wanted["prompt_tokens_cached_total"] = value
        elif (
            line.startswith("vllm:prompt_tokens_by_source_total")
            and 'source="local_compute"' in line
        ):
            wanted["local_compute"] = value
        elif (
            line.startswith("vllm:prompt_tokens_by_source_total")
            and 'source="local_cache_hit"' in line
        ):
            wanted["local_cache_hit"] = value
    return wanted


def _exact_token_ids(tokenizer: Any, length: int, salt: str) -> list[int]:
    filler = tokenizer.encode(
        " boundary aligned hybrid apc checkpoint validation",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced no filler tokens")
    token_ids = tokenizer.encode(f"Boundary APC probe {salt}. ", add_special_tokens=False)
    while len(token_ids) < length:
        token_ids.extend(filler)
    return token_ids[:length]


def _post_completion(
    *,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - start
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
            error = None
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        status = exc.code
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"error": {"message": str(exc)}}
        error = body

    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    usage = body.get("usage") if isinstance(body, dict) else None
    return {
        "status": status,
        "elapsed_seconds": elapsed,
        "text": choice.get("text") if isinstance(choice, dict) else None,
        "usage": usage,
        "valid_openai_body": isinstance(choices, list) and bool(choices) and usage is not None,
        "error": error,
    }


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(set(before) | set(after))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lengths", default="256,512,1024,2048,4096")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--require-prefix-cache-query",
        action="store_true",
        help="Return non-zero if repeated boundary prompts never query prefix cache.",
    )
    parser.add_argument(
        "--require-prefix-cache-hit",
        action="store_true",
        help="Return non-zero if repeated boundary prompts never hit prefix cache.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433

    base_url = args.base_url.rstrip("/")
    model = (
        _detect_model(base_url, "Qwen3.6-27B", args.timeout)
        if args.model == "auto"
        else args.model
    )
    endpoint = base_url + "/v1/completions"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    before_all = _metric_snapshot(base_url, args.timeout)
    with output_path.open("w", encoding="utf-8") as handle:
        header = {
            "phase": "before_metrics",
            "base_url": base_url,
            "model": model,
            "metrics": before_all,
        }
        print(json.dumps(header, sort_keys=True), flush=True)
        handle.write(json.dumps(header, sort_keys=True) + "\n")

        for length in _parse_lengths(args.lengths):
            prompt_ids = _exact_token_ids(tokenizer, length, f"len-{length}")
            for repeat in range(args.repeats):
                metrics_before = _metric_snapshot(base_url, args.timeout)
                result = _post_completion(
                    endpoint=endpoint,
                    model=model,
                    prompt_token_ids=prompt_ids,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                metrics_after = _metric_snapshot(base_url, args.timeout)
                elapsed = float(result["elapsed_seconds"])
                row = {
                    "phase": "case",
                    "label": f"boundary_{length}_repeat_{repeat}",
                    "length": length,
                    "repeat": repeat,
                    "prompt_tokens": length,
                    **result,
                    "effective_prompt_tokens_per_second": (
                        length / elapsed if elapsed > 0 and int(result["status"]) < 400 else None
                    ),
                    "metrics_before": metrics_before,
                    "metrics_after": metrics_after,
                    "metric_delta": _metric_delta(metrics_before, metrics_after),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()

        after_all = _metric_snapshot(base_url, args.timeout)
        total_delta = _metric_delta(before_all, after_all)
        repeated_rows = [row for row in rows if int(row["repeat"]) > 0]
        summary = {
            "phase": "summary",
            "all_status_ok": all(int(row["status"]) < 400 for row in rows),
            "all_valid_openai_body": all(bool(row["valid_openai_body"]) for row in rows),
            "total_rows": len(rows),
            "total_metric_delta": total_delta,
            "repeated_prefix_cache_query_delta": sum(
                row["metric_delta"].get("prefix_cache_queries_total", 0.0)
                for row in repeated_rows
            ),
            "repeated_prefix_cache_hit_delta": sum(
                row["metric_delta"].get("prefix_cache_hits_total", 0.0)
                for row in repeated_rows
            ),
            "output_jsonl": str(output_path),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        handle.write(json.dumps(summary, sort_keys=True) + "\n")

    failed = not summary["all_status_ok"] or not summary["all_valid_openai_body"]
    if args.require_prefix_cache_query and summary["repeated_prefix_cache_query_delta"] <= 0:
        failed = True
    if args.require_prefix_cache_hit and summary["repeated_prefix_cache_hit_delta"] <= 0:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
