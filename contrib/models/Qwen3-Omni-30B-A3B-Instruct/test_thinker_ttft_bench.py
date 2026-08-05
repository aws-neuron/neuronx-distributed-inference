#!/usr/bin/env python3
"""Thinker-only TTFT / throughput benchmark on /home/ubuntu/omni2 conversations 0-99.

Flow per conversation (talker/code2wav disabled — see test_ttfb_rtf_bench.py for
the full streaming TTFB version, which is currently blocked by the layers.23
tensor-capture only being wired at bucket 256).

Metrics:
  * ttft_ms         — from adapter.generate() start to the first token emission
                       (first hook fire after the prefill returns)
  * prefill_ms      — time to the first hook fire (== first forward complete)
  * decode_steps    — number of post-prefill forward calls (= new tokens - 1)
  * decode_mean_ms  — mean inter-token wall time during decode (ITL)
  * decode_p90_ms   — p90 of per-step decode time
  * tokens_per_s    — thinker_tokens / thinker_wall
  * rtf_vs_audio    — thinker_wall / input_audio_s (lower = faster than audio)

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0-7 python test_thinker_ttft_bench.py --num 100
"""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-7")
# The existing compiled artifacts include layers.23 capture; we still want it
# to flow (as a per-step timing signal), even though the shape is (1,) for
# buckets > 256. We only need the hook to *fire* each step.
os.environ.setdefault("QWEN3_OMNI_CAPTURE_LAYER_HIDDEN", "23")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if "/home/ubuntu" not in sys.path:
    sys.path.insert(0, "/home/ubuntu")

import _upstream_compat  # noqa: F401

import argparse
import json
import statistics
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from transformers import GenerationConfig

import test_asr_qwen3_omni as asr  # build_and_load_model applies pad_inputs patches

CONV_JSON = "/home/ubuntu/omni2/merged_conversations_with_audio_x10_with_system.json"
AUDIO_DIR = "/home/ubuntu/omni2/speech_wav_16k"
MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct"
COMPILED_PATH = "/tmp/qwen3_omni_compiled"


def build_messages(conv):
    msgs = conv["messages"]
    out = []
    for i, m in enumerate(msgs):
        if i == len(msgs) - 1:
            break  # drop the reference assistant reply
        role = m["role"]
        content = m["content"]
        if i == len(msgs) - 2 and role == "user":
            fname = os.path.basename(content)
            wav_path = os.path.join(AUDIO_DIR, fname)
            out.append({"role": role, "content": [{"type": "audio", "audio": wav_path}]})
        else:
            out.append({"role": role, "content": content})
    return out


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    i = int(round((len(s) - 1) * p / 100))
    return s[i]


def run_one(adapter, processor, conv, idx, max_new_tokens):
    messages = build_messages(conv)
    wav_path = messages[-1]["content"][0]["audio"]
    audio_np, sr = sf.read(wav_path)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    input_audio_s = float(len(audio_np) / sr)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    if sr != target_sr:
        import librosa
        audio_for_fe = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
    else:
        audio_for_fe = audio_np
    inputs = processor(text=[text], audio=[audio_for_fe], return_tensors="pt", padding=True)
    prompt_tokens = int(inputs.input_ids.shape[1])

    # The tensor_capture_hook fires once per forward pass (prefill + each decode step).
    # We use it purely as a timing tap.
    step_times = []  # absolute perf_counter timestamps
    def _hook(_m, _tensors):
        step_times.append(time.perf_counter())

    gc_cfg = GenerationConfig(do_sample=False, eos_token_id=[151645], pad_token_id=151645)
    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gc_cfg,
        max_new_tokens=max_new_tokens,
        tensor_capture_hook=_hook,
    )
    if getattr(inputs, "input_features", None) is not None:
        gen_kwargs["input_features"] = inputs.input_features.to(torch.bfloat16)
    if getattr(inputs, "feature_attention_mask", None) is not None:
        gen_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

    t_start = time.perf_counter()
    out_ids = adapter.generate(**gen_kwargs)
    t_end = time.perf_counter()

    thinker_wall = t_end - t_start
    new_tokens = int(out_ids.shape[1] - inputs.input_ids.shape[1])
    assistant_text = processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    if step_times:
        ttft_ms = (step_times[0] - t_start) * 1000
        prefill_ms = ttft_ms  # first forward covers the prefill
        if len(step_times) >= 2:
            decode_diffs_ms = [(step_times[i] - step_times[i - 1]) * 1000
                               for i in range(1, len(step_times))]
            decode_mean_ms = statistics.mean(decode_diffs_ms)
            decode_p50_ms = percentile(decode_diffs_ms, 50)
            decode_p90_ms = percentile(decode_diffs_ms, 90)
        else:
            decode_diffs_ms = []
            decode_mean_ms = decode_p50_ms = decode_p90_ms = None
    else:
        ttft_ms = prefill_ms = decode_mean_ms = decode_p50_ms = decode_p90_ms = None
        decode_diffs_ms = []

    tokens_per_s = new_tokens / thinker_wall if thinker_wall > 0 else None
    rtf_vs_audio = thinker_wall / input_audio_s if input_audio_s > 0 else None

    return {
        "idx": idx,
        "wav_path": wav_path,
        "input_audio_s": input_audio_s,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "thinker_wall_ms": thinker_wall * 1000,
        "ttft_ms": ttft_ms,
        "prefill_ms": prefill_ms,
        "decode_steps": max(0, len(step_times) - 1),
        "decode_mean_ms": decode_mean_ms,
        "decode_p50_ms": decode_p50_ms,
        "decode_p90_ms": decode_p90_ms,
        "tokens_per_s": tokens_per_s,
        "rtf_vs_audio": rtf_vs_audio,
        "text": assistant_text,
    }


def summary_row(name, xs, fmt="{:7.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {name:20s}  (no data)")
        return
    print(
        f"  {name:20s}  mean={fmt.format(statistics.mean(xs))}  "
        f"p50={fmt.format(percentile(xs, 50))}  "
        f"p90={fmt.format(percentile(xs, 90))}  "
        f"p95={fmt.format(percentile(xs, 95))}  "
        f"max={fmt.format(max(xs))}"
    )


def print_summary(results):
    ok = [r for r in results if "error" not in r]
    print("\n=== SUMMARY ===")
    print(f"  samples ok: {len(ok)}/{len(results)}")
    if not ok:
        return
    summary_row("TTFT ms", [r["ttft_ms"] for r in ok])
    summary_row("prefill ms", [r["prefill_ms"] for r in ok])
    summary_row("decode ITL mean ms", [r["decode_mean_ms"] for r in ok])
    summary_row("decode ITL p90 ms", [r["decode_p90_ms"] for r in ok])
    summary_row("thinker wall ms", [r["thinker_wall_ms"] for r in ok])
    summary_row("tokens/s (overall)", [r["tokens_per_s"] for r in ok], fmt="{:7.1f}")
    summary_row("RTF vs audio in", [r["rtf_vs_audio"] for r in ok], fmt="{:7.2f}")
    summary_row("prompt tokens", [r["prompt_tokens"] for r in ok], fmt="{:7.0f}")
    summary_row("new tokens", [r["new_tokens"] for r in ok], fmt="{:7.0f}")
    summary_row("input audio s", [r["input_audio_s"] for r in ok], fmt="{:7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--out", default="/tmp/qwen3_omni_thinker_ttft.json")
    args = parser.parse_args()

    with open(CONV_JSON) as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations; running [{args.start}, {args.start + args.num})")
    print("Building Neuron thinker (+ audio encoder)...")
    adapter, processor = asr.build_and_load_model(MODEL_PATH, COMPILED_PATH)
    print("Ready.\n")

    results = []
    bench_start = time.perf_counter()
    for k in range(args.num):
        idx = args.start + k
        if idx >= len(conversations):
            break
        try:
            r = run_one(adapter, processor, conversations[idx], idx, args.max_new_tokens)
            results.append(r)
            itl = r.get("decode_mean_ms")
            itl_str = f"{itl:5.1f}ms" if itl is not None else "  n/a"
            print(
                f"[{k+1:3d}/{args.num}] conv {idx:3d}  "
                f"in={r['input_audio_s']:4.1f}s  "
                f"prompt={r['prompt_tokens']:4d}tok  "
                f"new={r['new_tokens']:3d}tok  "
                f"TTFT={r['ttft_ms']:6.0f}ms  "
                f"ITL={itl_str}  "
                f"tok/s={r['tokens_per_s']:5.1f}  "
                f"wall={r['thinker_wall_ms']:5.0f}ms  "
                f"[{r['text'][:32]}]"
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[{k+1:3d}/{args.num}] conv {idx}: FAILED  {e}")
            results.append({"idx": idx, "error": str(e)})

        with open(args.out, "w") as f:
            json.dump({
                "cumulative_wall_s": time.perf_counter() - bench_start,
                "results": results,
            }, f, indent=2, ensure_ascii=False)

    print_summary(results)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
