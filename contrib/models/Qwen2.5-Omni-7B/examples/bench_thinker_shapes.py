#!/usr/bin/env python3
"""Long-shape micro-bench for the Qwen2.5-Omni Thinker on Neuron.

Reuses the existing ``thinker_tp4`` NEFF (compiled at seq_len=2048 by
``generate_qwen25_omni_speech.py --compile``). For each (input_len,
output_len) shape, runs ``num_runs`` end-to-end ``adapter.generate``
calls and reports:

- TTFT       wall time from generate() to first decoded token
- TPOT       per-token latency in the steady-state decode loop
- total_s    full generate() wall time

Median / min / max over all runs after one warmup. No talker, no
Token2Wav — this is pure thinker prefill+decode behaviour at the
two extreme shapes the user asked about (1K in / 10 out, 10 in / 1K
out).

Usage:
  NEURON_RT_VISIBLE_CORES=0-3 \
      python examples/bench_thinker_shapes.py \
      --compiled-path /opt/dlami/nvme/qwen25_omni_compiled \
      --num-runs 3
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# --- contrib bootstrap (mirror generate_qwen25_omni_speech.py) ---
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))
import _upstream_compat  # noqa: F401, E402

# Bring in the loader from the speech demo so we share the exact NEFF
# config used for the rest of the benchmarks. ``generate_qwen25_omni_speech``
# also pins ``NEURON_RT_VISIBLE_CORES`` at import time when unset.
import generate_qwen25_omni_speech as gen  # noqa: E402

DEFAULT_PROMPT_TOKENS = [
    "Recap the history of compilers in three short paragraphs. "
    "Walk through the major eras (assemblers, structured languages, "
    "object orientation, JIT engines, modern AI compilers) and what "
    "drove each transition. Keep it factual and avoid lists.",
]


def _build_chat(tokenizer, system_prompt, user_text):
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )


def _padded_input(tokenizer, target_input_len, system_prompt):
    """Build a chat message whose tokenized form is exactly
    ``target_input_len`` tokens long, by repeating filler text and
    truncating from the right.

    We need a deterministic input length so the prefill cost we measure
    is the cost of the requested shape, not whatever the tokenizer
    happens to emit for a fixed string.
    """
    filler = (
        "The compiler then proceeds through lexical analysis, syntactic "
        "parsing, semantic validation, intermediate representation "
        "lowering, optimization passes, and finally code generation. "
    )
    user_text = filler
    while True:
        text = _build_chat(tokenizer, system_prompt, user_text)
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] >= target_input_len:
            break
        user_text += filler

    text = _build_chat(tokenizer, system_prompt, user_text)
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] > target_input_len:
        # Trim from the user-content side. We keep the chat template
        # prefix (system + start tags) intact and just truncate the
        # tail; rebuild with the truncated content so HF still sees a
        # well-formed assistant prompt.
        # Simple approach: tokenize the filler alone, take exactly the
        # tokens we need, decode, rebuild.
        prompt_only = _build_chat(tokenizer, system_prompt, "")
        prompt_ids = tokenizer(prompt_only, return_tensors="pt").input_ids
        room = target_input_len - prompt_ids.shape[1]
        if room <= 0:
            # Pathological: the chat template alone is already longer
            # than target. Just truncate hard.
            return ids[:, :target_input_len]
        filler_ids = tokenizer(filler * 50, return_tensors="pt").input_ids
        body = filler_ids[:, :room]
        decoded_body = tokenizer.decode(body[0], skip_special_tokens=False)
        text = _build_chat(tokenizer, system_prompt, decoded_body)
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] != target_input_len:
            # Final hard trim / pad to be exact.
            ids = ids[:, :target_input_len]
    return ids


def _run_one(adapter, input_ids, max_new_tokens, ignore_eos, eos_ids):
    """One generate() call. ``ignore_eos=True`` strips eos_token_id and
    forces ``min_new_tokens == max_new_tokens`` so the run produces
    exactly the requested decode length (otherwise the model can stop
    early, which throws off the per-step latency derivation).
    """
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
    )
    if ignore_eos:
        kwargs["min_new_tokens"] = max_new_tokens
    else:
        kwargs["eos_token_id"] = eos_ids

    t0 = time.time()
    out = adapter.generate(**kwargs)
    t1 = time.time()
    return {
        "total_s": t1 - t0,
        "n_new_tokens": int(out.shape[1] - input_ids.shape[1]),
    }


def _bench_shape(adapter, input_ids, out_long, num_runs, ignore_eos, eos_ids):
    """For a fixed input length, run two output lengths (1 and out_long)
    so we can split TTFT (= total at out=1) from TPOT
    (= (T_long - T_short) / (out_long - 1)).
    """
    # Warmup at each shape (NEFF bucket may JIT-load on first call).
    _ = _run_one(adapter, input_ids, 1, ignore_eos, eos_ids)
    _ = _run_one(adapter, input_ids, out_long, ignore_eos, eos_ids)

    short_runs, long_runs = [], []
    for _ in range(num_runs):
        short_runs.append(
            _run_one(adapter, input_ids, 1, ignore_eos, eos_ids)
        )
        long_runs.append(
            _run_one(adapter, input_ids, out_long, ignore_eos, eos_ids)
        )

    short_totals = [r["total_s"] for r in short_runs]
    long_totals = [r["total_s"] for r in long_runs]
    ttft_med = statistics.median(short_totals)
    long_med = statistics.median(long_totals)
    long_n = long_runs[-1]["n_new_tokens"]
    tpot_ms = ((long_med - ttft_med) / max(long_n - 1, 1)) * 1000

    return {
        "ttft_s_median": round(ttft_med, 4),
        "ttft_s_min": round(min(short_totals), 4),
        "ttft_s_max": round(max(short_totals), 4),
        "long_total_s_median": round(long_med, 4),
        "long_total_s_min": round(min(long_totals), 4),
        "long_total_s_max": round(max(long_totals), 4),
        "long_n_new_tokens": long_n,
        "tpot_ms_derived": round(tpot_ms, 3),
        "short_runs_s": [round(t, 4) for t in short_totals],
        "long_runs_s": [round(t, 4) for t in long_totals],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trn2 thinker bench at extreme shapes (long input / short "
            "output and short input / long output). Reuses thinker_tp4 "
            "NEFF compiled by generate_qwen25_omni_speech.py --compile."
        )
    )
    parser.add_argument(
        "--model-path", default=os.environ.get("QWEN25_OMNI_MODEL_PATH"),
    )
    parser.add_argument(
        "--compiled-path",
        default=os.environ.get(
            "QWEN25_OMNI_COMPILED_PATH",
            "/opt/dlami/nvme/qwen25_omni_compiled",
        ),
    )
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--shapes", default="1024:10,10:1024",
        help=(
            "Comma-separated input:output token pairs. For each pair we "
            "additionally run input:1 to derive TTFT, then back out TPOT "
            "= (T_long - T_short) / (out_long - 1)."
        ),
    )
    parser.add_argument(
        "--greedy", action="store_true", default=True,
        help="Use thinker_tp4_greedy NEFF (deterministic).",
    )
    parser.add_argument(
        "--no-greedy", dest="greedy", action="store_false",
    )
    parser.add_argument(
        "--ignore-eos", action="store_true", default=True,
        help=(
            "Force min_new_tokens == max_new_tokens so the model emits "
            "the full requested decode length even when it would "
            "otherwise emit <|im_end|> early. Default true so the bench "
            "measures the requested shape, not whatever HF picks."
        ),
    )
    parser.add_argument(
        "--respect-eos", dest="ignore_eos", action="store_false",
    )
    args = parser.parse_args()

    if not args.model_path:
        from _model_path import resolve_model_path
        args.model_path = resolve_model_path()

    pairs = []
    for part in args.shapes.split(","):
        a, b = part.split(":")
        pairs.append((int(a), int(b)))

    print("=" * 72)
    print("Trn2 Thinker shape bench")
    print(f"  model:        {args.model_path}")
    print(f"  compiled:     {args.compiled_path}")
    print(f"  greedy:       {args.greedy}")
    print(f"  num_runs:     {args.num_runs}")
    print(f"  shapes:       {pairs}")
    print("=" * 72)

    # ---- Load thinker once ----
    thinker_adapter, tokenizer, _ = gen.load_thinker(
        args.model_path, args.compiled_path, greedy=args.greedy,
    )

    eos_ids = [tokenizer.eos_token_id]
    extra_eos = 151645  # qwen2 chat <|im_end|>
    if extra_eos not in eos_ids:
        eos_ids.append(extra_eos)

    rows = []
    for in_len, out_len in pairs:
        label = f"in={in_len}, out={out_len}"
        print(f"\n--- {label} ---")
        input_ids = _padded_input(tokenizer, in_len, gen.DEFAULT_SYSTEM)
        actual_in = int(input_ids.shape[1])
        if actual_in != in_len:
            print(
                f"  [warn] requested input_len={in_len}, "
                f"got {actual_in}; reporting against actual."
            )

        s = _bench_shape(
            thinker_adapter, input_ids, out_long=out_len,
            num_runs=args.num_runs, ignore_eos=args.ignore_eos,
            eos_ids=eos_ids,
        )
        s["label"] = label
        s["actual_input_len"] = actual_in
        s["requested_output_len"] = out_len
        rows.append(s)

        print(
            f"  short(out=1): runs={s['short_runs_s']} -> "
            f"ttft={s['ttft_s_median']:.3f}s"
        )
        print(
            f"  long (out={out_len}): runs={s['long_runs_s']} -> "
            f"total={s['long_total_s_median']:.3f}s "
            f"({s['long_n_new_tokens']} new tokens)"
        )
        print(f"  derived TPOT: {s['tpot_ms_derived']:.2f}ms")

    print("\n" + "=" * 78)
    print("Summary (median over runs)")
    print("=" * 78)
    print(
        f"{'shape':<22}{'in':>6}{'out':>6}"
        f"{'TTFT_s':>10}{'TPOT_ms':>10}{'total_s':>10}"
    )
    for s in rows:
        print(
            f"{s['label']:<22}"
            f"{s['actual_input_len']:>6}"
            f"{s['long_n_new_tokens']:>6}"
            f"{s['ttft_s_median']:>10.3f}"
            f"{s['tpot_ms_derived']:>10.2f}"
            f"{s['long_total_s_median']:>10.3f}"
        )


if __name__ == "__main__":
    main()
