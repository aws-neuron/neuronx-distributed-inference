#!/usr/bin/env python3
"""Benchmark OpenAI-compatible chat completions across context lengths.

The benchmark builds deterministic multi-turn chat histories, sends
``/v1/chat/completions`` requests with ``max_tokens=1``, and records wall/TTFT
latency. Streaming is used by default when the server supports it.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split()]
    if not lengths:
        raise ValueError("at least one context length is required")
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
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", token_ids)
    elif hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids

    if (
        isinstance(token_ids, list)
        and token_ids
        and isinstance(token_ids[0], list)
    ):
        return len(token_ids[0])
    return len(token_ids)


def _base_messages(turns: int, *, salt: str = "") -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic latency benchmark assistant. "
                "Reply with one concise token. "
                f"Benchmark salt: {salt}."
            ),
        }
    ]
    for idx in range(max(1, turns)):
        messages.append(
            {
                "role": "user",
                "content": f"Turn {idx}: remember benchmark key {idx}.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"ack {idx}",
            }
        )
    messages.append({"role": "user", "content": "Return the next benchmark token."})
    return messages


def _make_messages(
    tokenizer: Any,
    target_tokens: int,
    turns: int,
    *,
    salt: str = "",
) -> tuple[list[dict[str, str]], int]:
    messages = _base_messages(turns, salt=salt)
    filler_phrase = (
        " latency-prefix alpha beta gamma delta epsilon zeta eta theta iota kappa"
    )

    def set_repeats(repeats: int) -> None:
        messages[-1]["content"] = (
            "Return the next benchmark token."
            + (filler_phrase * max(0, repeats))
        )

    set_repeats(0)
    base_count = _chat_token_count(tokenizer, messages)
    if base_count >= target_tokens:
        return messages, base_count

    set_repeats(1)
    one_repeat_count = _chat_token_count(tokenizer, messages)
    filler_delta = max(1, one_repeat_count - base_count)
    repeats = max(0, (target_tokens - base_count) // filler_delta)

    set_repeats(repeats)
    prompt_tokens = _chat_token_count(tokenizer, messages)
    while repeats > 0 and prompt_tokens > target_tokens:
        overshoot = prompt_tokens - target_tokens
        repeats = max(0, repeats - max(1, (overshoot // filler_delta) + 1))
        set_repeats(repeats)
        prompt_tokens = _chat_token_count(tokenizer, messages)

    return messages, prompt_tokens


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload


def _completion_tokens_from_usage(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        return None
    try:
        return int(completion_tokens)
    except (TypeError, ValueError):
        return None


def _completion_tokens_from_text(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False)
        return len(encoded.get("input_ids", []))


def _token_latency_metrics(
    *,
    total_seconds: float,
    ttft_seconds: float | None,
    completion_tokens: int | None,
    content_chunk_count: int | None,
) -> dict[str, Any]:
    content_chunk_tpot_seconds = (
        (total_seconds - ttft_seconds) / (content_chunk_count - 1)
        if ttft_seconds is not None
        and content_chunk_count is not None
        and content_chunk_count > 1
        else None
    )
    completion_tokens_per_second = (
        completion_tokens / total_seconds
        if completion_tokens is not None
        and completion_tokens > 0
        and total_seconds > 0
        else None
    )
    decode_elapsed_seconds = (
        total_seconds - ttft_seconds
        if ttft_seconds is not None and total_seconds >= ttft_seconds
        else None
    )
    token_tpot_seconds = (
        decode_elapsed_seconds / (completion_tokens - 1)
        if decode_elapsed_seconds is not None
        and completion_tokens is not None
        and completion_tokens > 1
        else None
    )
    decode_tokens_per_second = (
        (completion_tokens - 1) / decode_elapsed_seconds
        if decode_elapsed_seconds is not None
        and decode_elapsed_seconds > 0
        and completion_tokens is not None
        and completion_tokens > 1
        else None
    )
    return {
        "tpot_seconds": token_tpot_seconds,
        "token_tpot_seconds": token_tpot_seconds,
        "content_chunk_tpot_seconds": content_chunk_tpot_seconds,
        "decode_elapsed_seconds": decode_elapsed_seconds,
        "decode_tokens_per_second": decode_tokens_per_second,
        "completion_tokens_per_second": completion_tokens_per_second,
    }


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return "" if content is None else str(content)
    text = first.get("text")
    return "" if text is None else str(text)


def _stream_chat(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[
    int,
    float | None,
    float,
    list[str],
    int,
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    chunks: list[str] = []
    content_parts: list[str] = []
    usage_payload = None
    first_content_seconds = None
    content_chunk_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    payload_chunk = json.loads(data)
                    usage = payload_chunk.get("usage")
                    if isinstance(usage, dict):
                        usage_payload = usage
                    choices = payload_chunk.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    content = delta.get("content")
                except Exception:
                    content = None
                if content:
                    content_parts.append(str(content))
                    content_chunk_count += 1
                    if first_content_seconds is None:
                        first_content_seconds = time.perf_counter() - start
                chunks.append(data)
            total_seconds = time.perf_counter() - start
            return (
                status,
                first_content_seconds,
                total_seconds,
                chunks,
                content_chunk_count,
                "".join(content_parts),
                usage_payload,
                None,
            )
    except urllib.error.HTTPError as exc:
        total_seconds = time.perf_counter() - start
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error_payload = {"error": {"message": str(exc)}}
        return (
            exc.code,
            first_content_seconds,
            total_seconds,
            chunks,
            content_chunk_count,
            "".join(content_parts),
            usage_payload,
            error_payload,
        )


def _run_one(
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    stream: bool,
    ignore_eos: bool,
    tokenizer: Any,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if stream:
        (
            status,
            first_chunk_seconds,
            total_seconds,
            chunks,
            content_chunk_count,
            content_text,
            usage_payload,
            error_payload,
        ) = _stream_chat(
            url,
            payload,
            timeout,
        )
        completion_tokens = _completion_tokens_from_usage(usage_payload)
        completion_token_source = "usage"
        if completion_tokens is None:
            completion_tokens = _completion_tokens_from_text(tokenizer, content_text)
            completion_token_source = "tokenizer"
        latency_metrics = _token_latency_metrics(
            total_seconds=total_seconds,
            ttft_seconds=first_chunk_seconds,
            completion_tokens=completion_tokens,
            content_chunk_count=content_chunk_count,
        )
        if status < 400:
            return {
                "status": status,
                "stream": True,
                "ttft_seconds": first_chunk_seconds,
                "total_seconds": total_seconds,
                "chunk_count": len(chunks),
                "content_chunk_count": content_chunk_count,
                "completion_tokens": completion_tokens,
                "completion_token_source": completion_token_source,
                "content_text": content_text,
                "usage": usage_payload,
                **latency_metrics,
                "error": None,
            }
        return {
            "status": status,
            "stream": True,
            "ttft_seconds": first_chunk_seconds,
            "total_seconds": total_seconds,
            "chunk_count": len(chunks),
            "content_chunk_count": content_chunk_count,
            "completion_tokens": completion_tokens,
            "completion_token_source": completion_token_source,
            "content_text": content_text,
            "usage": usage_payload,
            **latency_metrics,
            "error": error_payload,
        }

    start = time.perf_counter()
    status, response = _post_json(url, payload, timeout)
    total_seconds = time.perf_counter() - start
    usage_payload = response.get("usage") if isinstance(response, dict) else None
    content_text = _response_text(response) if isinstance(response, dict) else ""
    completion_tokens = _completion_tokens_from_usage(usage_payload)
    completion_token_source = "usage"
    if completion_tokens is None:
        completion_tokens = _completion_tokens_from_text(tokenizer, content_text)
        completion_token_source = "tokenizer"
    return {
        "status": status,
        "stream": False,
        "ttft_seconds": None,
        "total_seconds": total_seconds,
        "chunk_count": None,
        "content_text": content_text,
        "completion_tokens": completion_tokens,
        "completion_token_source": completion_token_source,
        "completion_tokens_per_second": (
            completion_tokens / total_seconds
            if completion_tokens > 0 and total_seconds > 0
            else None
        ),
        "error": None if status < 400 else response,
        "usage": usage_payload,
    }


def _row_passed(row: dict[str, Any], *, max_tokens: int) -> bool:
    if int(row["status"]) >= 400:
        return False
    if row.get("stream") and max_tokens > 0:
        return int(row.get("content_chunk_count") or 0) > 0
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--lengths",
        default="1024,2048,4096,8192,16384,32768",
        help="Comma or space separated target chat-template token lengths.",
    )
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests per length/repeat group.",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--unique-per-request",
        action="store_true",
        help="Add a unique system-message salt for each length/repeat.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    for target_tokens in _parse_lengths(args.lengths):
        for repeat_idx in range(args.repeats):
            requests = []
            for concurrency_idx in range(args.concurrency):
                salt = (
                    f"target={target_tokens};repeat={repeat_idx};"
                    f"concurrency={concurrency_idx};unique=1"
                    if args.unique_per_request
                    else ""
                )
                messages, prompt_tokens = _make_messages(
                    tokenizer,
                    target_tokens=target_tokens,
                    turns=args.turns,
                    salt=salt,
                )
                requests.append(
                    {
                        "messages": messages,
                        "prompt_tokens": prompt_tokens,
                        "concurrency_index": concurrency_idx,
                    }
                )

            group_start = time.perf_counter()
            if args.concurrency == 1:
                group_results = [
                    _run_one(
                        url=endpoint,
                        model=args.model,
                        messages=requests[0]["messages"],
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        stream=not args.no_stream,
                        ignore_eos=args.ignore_eos,
                        tokenizer=tokenizer,
                    )
                ]
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = [
                        executor.submit(
                            _run_one,
                            url=endpoint,
                            model=args.model,
                            messages=request["messages"],
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            stream=not args.no_stream,
                            ignore_eos=args.ignore_eos,
                            tokenizer=tokenizer,
                        )
                        for request in requests
                    ]
                    group_results = [future.result() for future in futures]
            group_wall_seconds = time.perf_counter() - group_start
            group_prompt_tokens = sum(int(request["prompt_tokens"]) for request in requests)
            group_effective_tps = (
                group_prompt_tokens / group_wall_seconds
                if group_wall_seconds > 0
                and all(int(result["status"]) < 400 for result in group_results)
                else None
            )

            for request, result in zip(requests, group_results):
                row = {
                    "target_tokens": target_tokens,
                    "prompt_tokens": request["prompt_tokens"],
                    "repeat": repeat_idx,
                    "concurrency": args.concurrency,
                    "concurrency_index": request["concurrency_index"],
                    "group_wall_seconds": group_wall_seconds,
                    "group_prompt_tokens": group_prompt_tokens,
                    "group_effective_prompt_tokens_per_second": group_effective_tps,
                    **result,
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                results.append(row)

    output = {
        "base_url": args.base_url,
        "model": args.model,
        "lengths": _parse_lengths(args.lengths),
        "turns": args.turns,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "passed": all(_row_passed(row, max_tokens=args.max_tokens) for row in results),
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
