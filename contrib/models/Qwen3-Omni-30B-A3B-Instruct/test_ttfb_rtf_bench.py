#!/usr/bin/env python3
"""TTFB / RTF benchmark on /home/ubuntu/omni2 chat conversations 0-99.

Per-conversation flow:
  * system prompt + prior user/assistant turns (all plain text)
  * FINAL user turn = audio (the JSON stores the wav path as its content)
  * assistant reply is produced by thinker → talker (streaming) → code2wav (CPU)

Metrics we compute:
  * input_audio_s  — duration of the audio user utterance
  * ttfb_ms        — request_start → first audio chunk delivered
  * thinker_ms     — thinker decode wall time
  * total_ms       — end-to-end (thinker → talker → finalize → stitch)
  * wav_s          — total emitted audio duration
  * RTF            — total_ms / wav_ms, lower = better (<1 = realtime)

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0-7 \
      python test_ttfb_rtf_bench.py --num 100
"""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-7")
os.environ.setdefault("QWEN3_OMNI_CAPTURE_LAYER_HIDDEN", "23")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Qwen3-Omni model src lives in two locations; prefer the current project copy
# and fall back to whn-ndi (which has the identical files). Some operations
# (git branch switch) can remove the local src/ directory.
for _candidate in (_HERE / "src", Path("/home/ubuntu/whn-ndi/contrib/models/Qwen3-Omni-30B-A3B-Instruct/src")):
    if (_candidate / "_upstream_compat.py").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
# The streaming/full-neuron helpers live at /home/ubuntu/*.py — make them importable.
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

import test_audio_streaming as STR  # build_all, install_*, _assemble_hidden, _build_talker_inputs, finalize_stream

# Local helpers (co-located with this bench script)
sys.path.insert(0, str(_HERE))
from code2wav_neuron import install_neuron_code2wav  # noqa: E402

CONV_JSON = "/home/ubuntu/omni2/merged_conversations_with_audio_x10_with_system.json"
AUDIO_DIR = "/home/ubuntu/omni2/speech_wav_16k"


def build_messages(conv):
    """Convert a JSON conversation into HF-style `messages` where the final user
    turn becomes an audio block pointing at AUDIO_DIR/<basename>.wav.

    The ground-truth assistant reply (last message) is dropped — we generate it.
    """
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
            # Plain string content is valid in HF chat templates.
            out.append({"role": role, "content": content})
    return out


def run_one(adapter, processor, hf_model, shim, ucp, conv, idx, out_wav_dir,
            max_thinker_tokens, max_talker_tokens, speaker="ethan"):
    messages = build_messages(conv)
    wav_path = messages[-1]["content"][0]["audio"]
    audio_np, sr = sf.read(wav_path)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    input_audio_s = float(len(audio_np) / sr)

    # --- Processor: chat template + feature extraction ---
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    if sr != target_sr:
        import librosa
        audio_for_fe = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
    else:
        audio_for_fe = audio_np
    inputs = processor(
        text=[text], audio=[audio_for_fe],
        return_tensors="pt", padding=True,
    )

    # --- Streaming state & callback ---
    stream_state = {
        "codes_list": [], "per_step_times": [],
        "emitted_up_to": 0, "chunk_index": 0, "c2w_per_chunk_s": [],
    }
    wav_chunks = []
    chunk_timing = []
    request_start = time.perf_counter()

    def on_audio(wav_np, chunk_index, c2w_ms, codec_tokens, final=False):
        rel_t_ms = (time.perf_counter() - request_start) * 1000
        wav_chunks.append(wav_np)
        chunk_timing.append({
            "idx": chunk_index, "t_ms": rel_t_ms, "c2w_ms": c2w_ms,
            "codec_tokens": codec_tokens,
            "wav_samples": int(len(wav_np)), "final": final,
        })

    STR.install_streaming_ucp(hf_model, ucp, stream_state)
    STR.install_streaming_talker_hook(hf_model, stream_state, hf_model.code2wav, on_audio)

    # --- Thinker ---
    gc_cfg = GenerationConfig(do_sample=False, eos_token_id=[151645], pad_token_id=151645)
    captured = []

    def _cap_hook(_m, tensors):
        if tensors:
            captured.append(tensors[0].clone().to("cpu"))

    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gc_cfg,
        max_new_tokens=max_thinker_tokens,
        tensor_capture_hook=_cap_hook,
    )
    if getattr(inputs, "input_features", None) is not None:
        gen_kwargs["input_features"] = inputs.input_features.to(torch.bfloat16)
    if getattr(inputs, "feature_attention_mask", None) is not None:
        gen_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

    t0 = time.perf_counter()
    out_ids = adapter.generate(**gen_kwargs)
    thinker_s = time.perf_counter() - t0
    thinker_end_ms = (time.perf_counter() - request_start) * 1000
    thinker_hidden = STR._assemble_hidden(captured, inputs, out_ids)
    thinker_tokens = int(out_ids.shape[1] - inputs.input_ids.shape[1])

    # --- Build talker inputs (on CPU; uses thinker hidden state captured above) ---
    t0 = time.perf_counter()
    talker_embed, talker_id, tts_pad, trailing = STR._build_talker_inputs(
        hf_model, out_ids, thinker_hidden, speaker=speaker,
    )
    build_talker_s = time.perf_counter() - t0

    # --- Talker (Neuron shim) + streaming code2wav fired from prepare_inputs hook ---
    # Suppress non-codec vocab tokens so only codec ids (0..2047) + codec_eos
    # (2150) can be picked. Matches HF's reference call in ``Qwen3OmniMoeForConditionalGeneration.generate``.
    talker_cfg = hf_model.config.talker_config
    talker_vocab = talker_cfg.text_config.vocab_size
    suppress_tokens = [
        i for i in range(talker_vocab - 1024, talker_vocab)
        if i != talker_cfg.codec_eos_token_id
    ]

    shim.reset_cache()
    t0 = time.perf_counter()
    hf_model.talker.generate(
        inputs_embeds=talker_embed,
        trailing_text_hidden=trailing,
        tts_pad_embed=tts_pad,
        talker_input_ids=talker_id,
        max_new_tokens=max_talker_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.8,
        temperature=0.9,
        repetition_penalty=1.1,
        suppress_tokens=suppress_tokens,
        eos_token_id=talker_cfg.codec_eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    talker_s = time.perf_counter() - t0

    # Emit any residual codec tokens that didn't fill a CHUNK_SIZE chunk.
    STR.finalize_stream(stream_state, hf_model.code2wav, on_audio)

    full_wav = np.concatenate(wav_chunks) if wav_chunks else np.zeros(0, dtype=np.float32)
    out_wav_path = os.path.join(out_wav_dir, f"conv_{idx:03d}.wav")
    sf.write(out_wav_path, full_wav, 24000)

    total_s = time.perf_counter() - request_start
    ttfb_ms = chunk_timing[0]["t_ms"] if chunk_timing else None
    wav_s = float(len(full_wav) / 24000)
    rtf = total_s / wav_s if wav_s > 0 else None
    asst_text = processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    return {
        "idx": idx,
        "wav_path": wav_path,
        "input_audio_s": input_audio_s,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "thinker_tokens": thinker_tokens,
        "thinker_s": thinker_s,
        "thinker_end_ms": thinker_end_ms,
        "build_talker_s": build_talker_s,
        "talker_s": talker_s,
        "codec_tokens": int(len(stream_state["codes_list"])),
        "num_chunks": len(chunk_timing),
        "ttfb_ms": ttfb_ms,
        "total_s": total_s,
        "wav_s": wav_s,
        "rtf": rtf,
        "out_wav": out_wav_path,
        "text": asst_text,
        "chunks": chunk_timing,
    }


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    i = int(round((len(s) - 1) * p / 100))
    return s[i]


def print_summary(results):
    ok = [r for r in results if "error" not in r]
    print("\n=== SUMMARY ===")
    print(f"  samples ok: {len(ok)}/{len(results)}")
    if not ok:
        return
    ttfbs = [r["ttfb_ms"] for r in ok if r.get("ttfb_ms") is not None]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
    thinker_ms = [r["thinker_s"] * 1000 for r in ok]
    total_ms = [r["total_s"] * 1000 for r in ok]
    in_audio = [r["input_audio_s"] for r in ok]
    out_wav = [r["wav_s"] for r in ok]
    th_toks = [r["thinker_tokens"] for r in ok]

    def row(name, xs, fmt="{:6.0f}"):
        s = statistics.mean(xs)
        p50 = percentile(xs, 50)
        p90 = percentile(xs, 90)
        p95 = percentile(xs, 95)
        print(f"  {name:18s}  mean={fmt.format(s)}  p50={fmt.format(p50)}  "
              f"p90={fmt.format(p90)}  p95={fmt.format(p95)}")

    row("TTFB ms", ttfbs)
    row("thinker ms", thinker_ms)
    row("total ms", total_ms)
    row("RTF", rtfs, fmt="{:6.2f}")
    row("input audio s", in_audio, fmt="{:6.2f}")
    row("output wav s", out_wav, fmt="{:6.2f}")
    row("thinker tokens", th_toks, fmt="{:6.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default="/tmp/qwen3_omni_ttfb_rtf.json")
    parser.add_argument("--wav-dir", default="/tmp/qwen3_omni_ttfb_rtf_wavs")
    parser.add_argument("--max-thinker", type=int, default=200)
    parser.add_argument("--max-talker", type=int, default=512)
    parser.add_argument("--speaker", default="ethan")
    parser.add_argument("--neuron-c2w", action="store_true",
                        help="Route code2wav through Neuron NEFFs (default: CPU)")
    args = parser.parse_args()

    os.makedirs(args.wav_dir, exist_ok=True)
    with open(CONV_JSON) as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations; "
          f"running [{args.start}, {args.start + args.num})")
    print("Building Neuron pipeline (thinker + audio + talker + UCP)...")
    adapter, processor, hf_model, shim, ucp = STR.build_all()
    if args.neuron_c2w:
        print("Installing Neuron code2wav shim...")
        install_neuron_code2wav(hf_model)
    print("Pipeline ready.\n")

    results = []
    bench_start = time.perf_counter()

    for k in range(args.num):
        idx = args.start + k
        if idx >= len(conversations):
            break
        conv = conversations[idx]
        try:
            r = run_one(
                adapter, processor, hf_model, shim, ucp, conv, idx, args.wav_dir,
                max_thinker_tokens=args.max_thinker,
                max_talker_tokens=args.max_talker,
                speaker=args.speaker,
            )
            results.append(r)
            print(f"[{k+1:3d}/{args.num}] conv {idx:3d} "
                  f"in={r['input_audio_s']:4.1f}s "
                  f"prompt_tok={r['prompt_tokens']:4d} "
                  f"thinker={r['thinker_s']*1000:4.0f}ms/{r['thinker_tokens']:3d}tok "
                  f"ttfb={r['ttfb_ms']:5.0f}ms "
                  f"total={r['total_s']*1000:5.0f}ms "
                  f"wav={r['wav_s']:4.1f}s "
                  f"RTF={r['rtf']:.2f} "
                  f"[{r['text'][:36]}]")
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
    print(f"\nFull JSON: {args.out}")
    print(f"WAVs:       {args.wav_dir}")


if __name__ == "__main__":
    main()
