#!/usr/bin/env python3
"""Live decode-throughput sweep against a running Qwen3.6-27B vLLM server.

Hits the OpenAI-compatible endpoint with streaming + include_usage + ignore_eos
so each case produces an EXACT number of decode tokens, and measures steady-state
decode rate as (completion_tokens - 1) / (t_last_token - t_first_token), i.e. 1/TPOT.
Batch=1 (server is max-num-seqs 1). Non-disruptive: read-only generation requests.
"""
import json
import sys
import time

import requests
from transformers import AutoTokenizer

BASE = "http://localhost:8001"
MODEL = "/home/ubuntu/models/Qwen3.6-27B"
DECODE_TOKENS = 96
PREFIX_TARGETS = [16, 2048, 8192, 16384, 32768, 65536]

tok = AutoTokenizer.from_pretrained(MODEL)
_filler = "The data pipeline ingests records, validates the schema, and writes partitioned output. "
_filler_ids = tok(_filler * 5000, add_special_tokens=False)["input_ids"]


def build_prompt(target_tokens):
    body = max(1, target_tokens - 40)  # reserve ~40 tokens for the chat template
    return tok.decode(_filler_ids[:body])


def run_case(target):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(target)}],
        "max_tokens": DECODE_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    chunks = 0
    with requests.post(BASE + "/v1/chat/completions", json=payload, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                chunks += 1
    end = time.perf_counter()
    comp = usage["completion_tokens"] if usage else chunks
    decode_tps = None
    if t_first and t_last and comp and comp > 1 and t_last > t_first:
        decode_tps = (comp - 1) / (t_last - t_first)
    return {
        "target_prefix": target,
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "completion_tokens": comp,
        "ttft_s": round(t_first - t0, 3) if t_first else None,
        "decode_tok_s": round(decode_tps, 2) if decode_tps else None,
        "tpot_ms": round(1000.0 / decode_tps, 2) if decode_tps else None,
        "total_s": round(end - t0, 3),
    }


def main():
    results = []
    for target in PREFIX_TARGETS:
        print(f"... prefix~{target}", file=sys.stderr, flush=True)
        try:
            res = run_case(target)
        except Exception as exc:  # noqa: BLE001
            res = {"target_prefix": target, "error": repr(exc)}
        results.append(res)
        print(json.dumps(res), flush=True)
    print("=== SUMMARY ===", flush=True)
    print(json.dumps({"model": MODEL, "decode_tokens": DECODE_TOKENS, "batch": 1, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
