#!/usr/bin/env python3
"""OpenAI-compatible chat validation for Qwen3.6 Hybrid APC serving."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split()]
    if not lengths:
        raise ValueError("at least one length is required")
    return lengths


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return len(token_ids)


def _build_messages(
    *,
    shared_key: str,
    salt: str,
    filler_repeats: int,
    suffix: str,
    turns: int,
) -> list[dict[str, str]]:
    filler = (
        " hybrid apc prefix checkpoint recurrent delta state validation "
        "attention blocks restore suffix deterministic "
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic latency validation assistant. "
                f"shared-key={shared_key}; salt={salt}. "
                "Return exactly one short answer token."
            ),
        }
    ]
    for idx in range(max(1, turns - 1)):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Shared setup turn {idx}. "
                    "Remember the validation marker and answer tersely."
                ),
            }
        )
        messages.append({"role": "assistant", "content": f"ack-{idx}"})
    messages.append(
        {
            "role": "user",
            "content": (
                "Shared benchmark document begins. "
                + (filler * max(0, filler_repeats))
                + f" Shared benchmark document ends. {suffix}"
            ),
        }
    )
    return messages


def _make_messages(
    tokenizer: Any,
    target_tokens: int,
    *,
    shared_key: str,
    salt: str,
    suffix: str,
    turns: int,
) -> tuple[list[dict[str, str]], int]:
    def build(repeats: int) -> list[dict[str, str]]:
        return _build_messages(
            shared_key=shared_key,
            salt=salt,
            filler_repeats=repeats,
            suffix=suffix,
            turns=turns,
        )

    low = 0
    high = 1
    while _chat_token_count(tokenizer, build(high)) <= target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        if _chat_token_count(tokenizer, build(mid)) <= target_tokens:
            low = mid
        else:
            high = mid

    messages = build(low)
    return messages, _chat_token_count(tokenizer, messages)


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


def _post_chat(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - start
            response_payload = json.loads(response.read().decode("utf-8"))
            choice = (response_payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            return {
                "status": response.status,
                "elapsed_seconds": elapsed,
                "content": "" if content is None else str(content),
                "finish_reason": choice.get("finish_reason"),
                "usage": response_payload.get("usage"),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error_payload = {"error": {"message": str(exc)}}
        return {
            "status": exc.code,
            "elapsed_seconds": elapsed,
            "content": "",
            "finish_reason": None,
            "usage": None,
            "error": error_payload,
        }


def _run_case(
    *,
    endpoint: str,
    model: str,
    label: str,
    messages: list[dict[str, str]],
    prompt_tokens: int,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    result = _post_chat(
        endpoint=endpoint,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )
    row = {
        "label": label,
        "prompt_tokens": prompt_tokens,
        **result,
    }
    elapsed = float(row["elapsed_seconds"])
    row["effective_prompt_tokens_per_second"] = (
        prompt_tokens / elapsed if elapsed > 0 and int(row["status"]) < 400 else None
    )
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def _semantic_smoke_cases() -> list[dict[str, Any]]:
    return [
        {
            "label": "semantic_arithmetic",
            "messages": [
                {
                    "role": "system",
                    "content": "You answer with only the requested value.",
                },
                {
                    "role": "user",
                    "content": "What is 17 * 23? Answer with digits only.",
                },
            ],
            "contains": "391",
        },
        {
            "label": "semantic_marker_copy",
            "messages": [
                {
                    "role": "system",
                    "content": "You answer with only the requested marker.",
                },
                {
                    "role": "user",
                    "content": "Return exactly this marker: BASELINE_OK_27B",
                },
            ],
            "contains": "BASELINE_OK_27B",
        },
        {
            "label": "semantic_multi_turn_recall",
            "messages": [
                {
                    "role": "system",
                    "content": "You answer with only the requested value.",
                },
                {
                    "role": "user",
                    "content": "Remember validation code ZX-417.",
                },
                {"role": "assistant", "content": "Remembered."},
                {
                    "role": "user",
                    "content": "What validation code did I ask you to remember?",
                },
            ],
            "contains": "ZX-417",
        },
    ]


def _avg(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _successful_elapsed(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["elapsed_seconds"])
        for row in rows
        if int(row.get("status", 500)) < 400
    ]


def _speedup_passes(value: float | None, minimum: float) -> bool:
    return minimum <= 0 or (value is not None and value >= minimum)


def _apc_gate_failures(summary: dict[str, Any]) -> list[str]:
    checks = {
        "all_status_ok": bool(summary["all_status_ok"]),
        "warm_full_exact_text": bool(summary["warm_full_exact_text"]),
        "partial_repeat_exact_text": bool(summary["partial_repeat_exact_text"]),
        "multi_turn_repeat_exact_text": bool(summary["multi_turn_repeat_exact_text"]),
        "semantic_smoke_passed": bool(summary["semantic_smoke_passed"]),
        "warm_full_speedup_passed": bool(summary["warm_full_speedup_passed"]),
        "partial_reference_speedup_passed": bool(
            summary["partial_reference_speedup_passed"]
        ),
    }
    return [name for name, passed in checks.items() if not passed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cold-lengths", default="256,512,1024,1536,1984")
    parser.add_argument("--target-tokens", type=int, default=1984)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--semantic-max-tokens", type=int, default=16)
    parser.add_argument("--cold-repeats", type=int, default=2)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--mixed-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--min-warm-full-speedup",
        type=float,
        default=1.5,
        help=(
            "Minimum warm-full repeat speedup over the initial request. "
            "Set to 0 to disable this speed gate."
        ),
    )
    parser.add_argument(
        "--min-partial-speedup",
        type=float,
        default=1.2,
        help=(
            "Minimum partial-prefix warm speedup over its cold reference. "
            "Set to 0 to disable this speed gate."
        ),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_url = args.base_url.rstrip("/")
    model = (
        _detect_model(base_url, "Qwen3.6-27B", args.timeout)
        if args.model == "auto"
        else args.model
    )
    endpoint = base_url + "/v1/chat/completions"
    stamp = int(time.time())

    rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []
    for length in _parse_lengths(args.cold_lengths):
        for repeat in range(args.cold_repeats):
            messages, prompt_tokens = _make_messages(
                tokenizer,
                length,
                shared_key=f"cold-{length}-{repeat}-{stamp}",
                salt=f"cold-early-{length}-{repeat}-{stamp}",
                suffix="Answer with the word cold.",
                turns=args.turns,
            )
            row = _run_case(
                endpoint=endpoint,
                model=model,
                label=f"cold_len_{length}_repeat_{repeat}",
                messages=messages,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            rows.append(row)
            cold_rows.append(row)

    warm_messages, warm_prompt_tokens = _make_messages(
        tokenizer,
        args.target_tokens,
        shared_key=f"warm-full-{stamp}",
        salt=f"warm-full-{stamp}",
        suffix="Answer with the word warm.",
        turns=args.turns,
    )
    warm_full_rows = [
        _run_case(
            endpoint=endpoint,
            model=model,
            label="warm_full_initial",
            messages=warm_messages,
            prompt_tokens=warm_prompt_tokens,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    ]
    for repeat in range(args.warm_repeats):
        warm_full_rows.append(
            _run_case(
                endpoint=endpoint,
                model=model,
                label=f"warm_full_repeat_{repeat}",
                messages=warm_messages,
                prompt_tokens=warm_prompt_tokens,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
    rows.extend(warm_full_rows)

    partial_key = f"partial-shared-{stamp}"
    partial_warmup_messages, partial_warmup_tokens = _make_messages(
        tokenizer,
        args.target_tokens,
        shared_key=partial_key,
        salt=partial_key,
        suffix="Suffix alpha. Answer with the word alpha.",
        turns=args.turns,
    )
    partial_target_messages, partial_target_tokens = _make_messages(
        tokenizer,
        args.target_tokens,
        shared_key=partial_key,
        salt=partial_key,
        suffix="Suffix beta. Answer with the word beta.",
        turns=args.turns,
    )
    partial_cold_messages, partial_cold_tokens = _make_messages(
        tokenizer,
        args.target_tokens,
        shared_key=f"partial-cold-reference-{stamp}",
        salt=f"partial-cold-reference-{stamp}",
        suffix="Suffix beta. Answer with the word beta.",
        turns=args.turns,
    )
    partial_rows = [
        _run_case(
            endpoint=endpoint,
            model=model,
            label="partial_cold_reference",
            messages=partial_cold_messages,
            prompt_tokens=partial_cold_tokens,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ),
        _run_case(
            endpoint=endpoint,
            model=model,
            label="partial_warmup_alpha",
            messages=partial_warmup_messages,
            prompt_tokens=partial_warmup_tokens,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ),
    ]
    for repeat in range(args.warm_repeats):
        partial_rows.append(
            _run_case(
                endpoint=endpoint,
                model=model,
                label=f"partial_warm_beta_repeat_{repeat}",
                messages=partial_target_messages,
                prompt_tokens=partial_target_tokens,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
    rows.extend(partial_rows)

    mixed_rows: list[dict[str, Any]] = []
    for repeat in range(args.mixed_repeats):
        mixed_warm_messages, mixed_warm_tokens = _make_messages(
            tokenizer,
            args.target_tokens,
            shared_key=partial_key,
            salt=partial_key,
            suffix=f"Suffix mixed warm {repeat}. Answer with the word beta.",
            turns=args.turns,
        )
        mixed_cold_messages, mixed_cold_tokens = _make_messages(
            tokenizer,
            args.target_tokens,
            shared_key=f"mixed-cold-{repeat}-{stamp}",
            salt=f"mixed-cold-{repeat}-{stamp}",
            suffix=f"Suffix mixed cold {repeat}. Answer with the word cold.",
            turns=args.turns,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _run_case,
                    endpoint=endpoint,
                    model=model,
                    label=f"mixed_warm_repeat_{repeat}",
                    messages=mixed_warm_messages,
                    prompt_tokens=mixed_warm_tokens,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
                executor.submit(
                    _run_case,
                    endpoint=endpoint,
                    model=model,
                    label=f"mixed_cold_repeat_{repeat}",
                    messages=mixed_cold_messages,
                    prompt_tokens=mixed_cold_tokens,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
            ]
            for future in futures:
                mixed_rows.append(future.result())
    rows.extend(mixed_rows)

    multi_messages, multi_prompt_tokens = _make_messages(
        tokenizer,
        min(args.target_tokens, 1536),
        shared_key=f"multi-turn-{stamp}",
        salt=f"multi-turn-{stamp}",
        suffix="Answer with the word multi.",
        turns=max(args.turns, 8),
    )
    multi_rows = []
    for repeat in range(args.warm_repeats):
        multi_rows.append(
            _run_case(
                endpoint=endpoint,
                model=model,
                label=f"multi_turn_repeat_{repeat}",
                messages=multi_messages,
                prompt_tokens=multi_prompt_tokens,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
    rows.extend(multi_rows)

    semantic_rows = []
    for case in _semantic_smoke_cases():
        prompt_tokens = _chat_token_count(tokenizer, case["messages"])
        row = _run_case(
            endpoint=endpoint,
            model=model,
            label=case["label"],
            messages=case["messages"],
            prompt_tokens=prompt_tokens,
            max_tokens=args.semantic_max_tokens,
            timeout=args.timeout,
        )
        row["expected_contains"] = case["contains"]
        row["semantic_passed"] = case["contains"] in row.get("content", "")
        semantic_rows.append(row)
    rows.extend(semantic_rows)

    warm_initial = warm_full_rows[0]
    warm_repeats = warm_full_rows[1:]
    partial_warm = [
        row for row in partial_rows if row["label"].startswith("partial_warm_beta")
    ]
    mixed_warm = [row for row in mixed_rows if row["label"].startswith("mixed_warm")]
    mixed_cold = [row for row in mixed_rows if row["label"].startswith("mixed_cold")]
    all_ok = all(int(row.get("status", 500)) < 400 for row in rows)
    warm_full_exact = len({row["content"] for row in warm_full_rows}) == 1
    partial_repeat_exact = bool(partial_warm) and len(
        {row["content"] for row in partial_warm}
    ) == 1
    multi_repeat_exact = bool(multi_rows) and len(
        {row["content"] for row in multi_rows}
    ) == 1
    semantic_passed = all(bool(row.get("semantic_passed")) for row in semantic_rows)

    warm_initial_elapsed = float(warm_initial["elapsed_seconds"])
    warm_repeat_avg = _avg(_successful_elapsed(warm_repeats))
    partial_cold_elapsed = float(partial_rows[0]["elapsed_seconds"])
    partial_warm_avg = _avg(_successful_elapsed(partial_warm))
    warm_full_speedup = (
        warm_initial_elapsed / warm_repeat_avg
        if warm_repeat_avg and warm_repeat_avg > 0
        else None
    )
    partial_reference_speedup = (
        partial_cold_elapsed / partial_warm_avg
        if partial_warm_avg and partial_warm_avg > 0
        else None
    )
    summary = {
        "all_status_ok": all_ok,
        "base_url": base_url,
        "model": model,
        "cold_request_count": len(cold_rows),
        "warm_full_exact_text": warm_full_exact,
        "partial_repeat_exact_text": partial_repeat_exact,
        "multi_turn_repeat_exact_text": multi_repeat_exact,
        "semantic_smoke_passed": semantic_passed,
        "warm_full_initial_seconds": warm_initial_elapsed,
        "warm_full_repeat_avg_seconds": warm_repeat_avg,
        "warm_full_speedup": warm_full_speedup,
        "min_warm_full_speedup": args.min_warm_full_speedup,
        "warm_full_speedup_passed": _speedup_passes(
            warm_full_speedup,
            args.min_warm_full_speedup,
        ),
        "partial_cold_reference_seconds": partial_cold_elapsed,
        "partial_warm_beta_avg_seconds": partial_warm_avg,
        "partial_reference_speedup": partial_reference_speedup,
        "min_partial_speedup": args.min_partial_speedup,
        "partial_reference_speedup_passed": _speedup_passes(
            partial_reference_speedup,
            args.min_partial_speedup,
        ),
        "mixed_warm_avg_seconds": _avg(_successful_elapsed(mixed_warm)),
        "mixed_cold_avg_seconds": _avg(_successful_elapsed(mixed_cold)),
        "multi_turn_avg_seconds": _avg(_successful_elapsed(multi_rows)),
    }
    failures = _apc_gate_failures(summary)
    summary["apc_validation_passed"] = not failures
    summary["apc_gate_failures"] = failures
    output = {
        "summary": summary,
        "results": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary, "output_json": str(output_path)}, sort_keys=True))
    return 0 if summary["apc_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
