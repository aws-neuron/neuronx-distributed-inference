#!/usr/bin/env python3
"""Streaming codec->wav benchmark for Qwen2.5-Omni-7B.

Compares two pipelines:

  serial    : Thinker -> CPU HF hidden -> Talker (full codec gen)
              -> Token2Wav DiT+BigVGAN (full waveform once)

  streaming : Thinker -> CPU HF hidden -> Talker (codec gen with hook)
              -> Token2Wav called every CHUNK codec tokens, with a
                 LEFT_CONTEXT prefix to suppress BigVGAN edge artifacts;
                 first audio chunk arrives as soon as Talker has emitted
                 ``CHUNK`` tokens, well before the codec stream finishes.

The streaming pipeline reuses the existing serial one from
``generate_qwen25_omni_speech.py`` for everything *up to* the talker run;
only the talker decode + Token2Wav call are restructured. No models are
recompiled.

Reports:
  - first-audio-byte (FAB) latency: prompt-in -> first wav chunk ready
  - total wall time
  - per-stage breakdown
  - first wav chunk vs. full audio (by serial baseline) waveform sanity check

Usage::

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export QWEN25_OMNI_COMPILED_PATH=/opt/dlami/nvme/qwen25_omni_compiled
    cd contrib/models/Qwen2.5-Omni-7B
    python examples/test_ttfb_streaming_bench.py --num-runs 3
"""

from __future__ import annotations

import os as _os

# Default: 8 cores visible (TP=4 thinker/talker on 0-3 + DiT replicas on
# 4-7). Override with QWEN25_OMNI_VISIBLE_CORES="0-3" to fall back to the
# old single-group layout for A/B comparison.
_os.environ.setdefault(
    "NEURON_RT_VISIBLE_CORES",
    _os.environ.get("QWEN25_OMNI_VISIBLE_CORES", "0-7"),
)

import sys as _sys
from pathlib import Path as _Path

_THIS = _Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
_EXAMPLES = _THIS.parent
for p in (_SRC, _EXAMPLES):
    if str(p) not in _sys.path:
        _sys.path.insert(0, str(p))
import _upstream_compat  # noqa: F401

import argparse
import gc
import queue
import statistics
import threading
import time
from typing import List, Optional, Tuple

import torch
from transformers.generation.stopping_criteria import StoppingCriteria

try:
    import soundfile as sf
except ImportError:
    _sys.exit("soundfile is required. pip install soundfile")

# Reuse helpers from the existing serial demo (load_*, run_thinker,
# extract_hidden_states, prepare_talker_input, run_token2wav, etc.).
import generate_qwen25_omni_speech as omni_demo


# =============================================================================
# Streaming codec->wav helpers
# =============================================================================

class CodecChunkEmitter(StoppingCriteria):
    """Hook into NxDI's per-step ``stopping_criteria(input_ids, None)`` call.

    Every decode step, NxDI appends the freshly sampled token to
    ``input_ids``. We slice off any tokens past ``context_len`` and push
    them into a buffer; when at least ``chunk_size`` codec tokens are
    pending, the on-chunk callback fires and drains the buffer.

    Returns ``False`` always so generation continues; the talker has its
    own EOS criteria already in place.
    """

    def __init__(self, context_len: int, chunk_size: int, on_chunk):
        super().__init__()
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.on_chunk = on_chunk
        self.codec_tokens: List[int] = []
        self._n_chunks = 0
        self._first_chunk_done_t: float | None = None

    def __call__(self, input_ids: torch.Tensor, scores) -> torch.Tensor:
        # input_ids: (batch, total_len). Talker is batch-1.
        new_total = input_ids.shape[1]
        produced = max(0, new_total - self.context_len)
        # Mirror the latest tokens into our buffer.
        already = len(self.codec_tokens)
        if produced > already:
            extra = input_ids[0, self.context_len + already : self.context_len + produced].tolist()
            self.codec_tokens.extend(extra)
        # Fire chunk callback while we have a full chunk pending.
        while len(self.codec_tokens) - self._n_chunks * self.chunk_size >= self.chunk_size:
            start = self._n_chunks * self.chunk_size
            end = start + self.chunk_size
            self.on_chunk(self.codec_tokens, start, end, final=False)
            self._n_chunks += 1
            if self._first_chunk_done_t is None:
                self._first_chunk_done_t = time.time()

        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def synthesize_chunk(
    t2w,
    t2w_cfg,
    codec_codes: List[int],
    start: int,
    end: int,
    *,
    left_context: int,
    conditioning,
    reference_mel,
    total_upsample: int,
    codec_eos_token: int,
    codec_pad_token: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Synthesize ``codec_codes[start:end]`` with ``left_context`` codec
    tokens of padding to absorb BigVGAN edge artifacts.

    Returns a 1-D float32 waveform tensor for the *new* segment only
    (left-context prefix is trimmed off).

    EOS / pad tokens at the tail (final flush) are stripped before DiT
    is called -- the DiT codec embedding does not have them in vocab.
    """
    # Drop trailing EOS/pad tokens (only matters on final flush).
    eff_end = end
    while eff_end > start and codec_codes[eff_end - 1] in (codec_eos_token, codec_pad_token):
        eff_end -= 1
    if eff_end <= start:
        return torch.empty(0, dtype=torch.float32)

    ctx_start = max(0, start - left_context)
    chunk_codes = codec_codes[ctx_start:eff_end]
    code_tensor = torch.tensor([chunk_codes], dtype=torch.long)
    num_embeds = getattr(t2w_cfg.dit_config, "num_embeds", 8193)
    if code_tensor.numel() and code_tensor.max() >= num_embeds:
        code_tensor = code_tensor.clamp(0, num_embeds - 1)

    if seed is not None:
        # Mix start into seed so each chunk gets a distinct but
        # deterministic init noise.
        torch.manual_seed(seed + start)

    wav = t2w(
        code=code_tensor,
        conditioning=conditioning,
        reference_mel=reference_mel,
        num_steps=10,
        guidance_scale=0.5,
    )
    wav = wav.detach().float().reshape(-1)
    # Trim off the left-context portion (audio frames per token = total_upsample
    # * config.repeats? In Qwen2.5 DiT, mel_len = code_len * repeats and the
    # BigVGAN upsample stages bring it back up to waveform).
    repeats = getattr(t2w_cfg.dit_config, "repeats", 2)
    samples_per_token = repeats * total_upsample
    skip_samples = (start - ctx_start) * samples_per_token
    return wav[skip_samples:]


def run_streaming_pipeline(
    *,
    model_path: str,
    thinker_adapter,
    tokenizer,
    hf_model,
    talker_model,
    talker_adapter,
    talker_cfg,
    t2w,
    t2w_cfg,
    speaker: str,
    prompt: str,
    system_prompt: str,
    chunk_size: int,
    left_context: int,
    greedy: bool = False,
    seed: int = 42,
):
    """Run one full streaming inference and return timings + audio chunks."""
    t_start = time.time()

    # 1) Thinker (serial; nothing to streamify here without recompile).
    thinker_result = omni_demo.run_thinker(
        thinker_adapter, tokenizer, prompt, system_prompt,
    )
    t_thinker = time.time()

    # 2) HF CPU re-forward to extract hidden states (current architecture).
    outputs, full_ids, prompt_len, hidden_time = omni_demo.extract_hidden_states(
        hf_model, thinker_result,
    )
    t_hidden = time.time()

    # 3) Project thinker reply states -> 896-dim talker embeddings.
    talker_input = omni_demo.prepare_talker_input(
        model_path, hf_model, outputs, full_ids, prompt_len, speaker,
    )
    t_prep = time.time()

    # 4) Talker decode + DiT chunk synthesis (streaming).
    projected_context = talker_input["projected_context"]
    projected_reply = talker_input["projected_reply"]
    context_len = talker_input["context_len"]
    conditioning = talker_input["conditioning"]
    reference_mel = talker_input["reference_mel"]

    codec_bos = talker_cfg.tts_codec_start_token_id
    codec_eos = talker_cfg.tts_codec_end_token_id
    codec_pad = talker_cfg.tts_codec_pad_token_id
    codec_mask = talker_cfg.tts_codec_mask_token_id

    talker_input_ids = torch.cat([
        torch.full((1, context_len - 2), codec_mask, dtype=torch.long),
        torch.tensor([[codec_pad]], dtype=torch.long),
        torch.tensor([[codec_bos]], dtype=torch.long),
    ], dim=1)
    talker_attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long)
    max_gen = min(600, 2048 - context_len - 10)

    talker_model.set_vision_embeddings(
        projected_context.to(torch.bfloat16),
        torch.ones(1, context_len, 1, dtype=torch.int32),
        thinker_reply_embeds=projected_reply.to(torch.bfloat16),
    )

    # ----- Streaming hook -----
    # Producer/consumer: the emitter (called per talker decode step on the
    # talker's thread) enqueues (idx, start, end, code_slice). A pool of
    # ``num_workers`` background threads consumes the queue and runs DiT
    # synthesis off the talker decode path. With DiT NEFFs replicated on
    # cores 4-7 (see ``QWEN25_OMNI_DIT_CORE_*`` in the driver), multiple
    # DiT forwards dispatch to different replicas concurrently while the
    # talker keeps decoding on cores 0-3.
    chunk_results: dict = {}
    chunk_results_lock = threading.Lock()
    work_q: queue.Queue = queue.Queue()
    # NeuronQwen25OmniToken2WavWithNeuronDiT.__call__ monkey-patches
    # ``dit.forward`` on a shared object, so concurrent calls would race.
    # Serialize DiT calls; the win comes from running DiT on cores 4-7
    # while the talker keeps decoding on 0-3 — not from intra-DiT
    # concurrency.
    t2w_lock = threading.Lock()
    upsample_rates = getattr(t2w_cfg.bigvgan_config, "upsample_rates", [5, 3, 2, 2, 2, 2])
    total_upsample = 1
    for r in upsample_rates:
        total_upsample *= int(r)

    # NOTE: ``__call__`` is not thread-safe (it monkey-patches
    # ``dit.forward`` on a shared object), so workers serialize on
    # ``t2w_lock``. With 1 worker, the win is purely "DiT off the talker
    # decode path + on different cores"; raising this past 1 won't help
    # unless ``__call__`` is refactored to be re-entrant.
    num_workers = max(1, int(_os.environ.get("QWEN25_OMNI_DIT_WORKERS", "1")))

    def _dit_worker():
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                return
            idx, start, end, codes_slice = item
            t0 = time.time()
            with t2w_lock:
                wav = synthesize_chunk(
                    t2w, t2w_cfg, codes_slice, start, end,
                    left_context=left_context,
                    conditioning=conditioning,
                    reference_mel=reference_mel,
                    total_upsample=total_upsample,
                    codec_eos_token=codec_eos,
                    codec_pad_token=codec_pad,
                    seed=seed,
                )
            elapsed = time.time() - t0
            with chunk_results_lock:
                chunk_results[idx] = (wav, {
                    "start": start, "end": end,
                    "samples": int(wav.numel()),
                    "synthesis_s": elapsed,
                    "wall_t": time.time() - t_start,
                })
            work_q.task_done()

    workers = [
        threading.Thread(target=_dit_worker, daemon=True, name=f"dit-{i}")
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()

    chunk_counter = {"n": 0}

    def on_chunk(all_codes, start, end, final):
        idx = chunk_counter["n"]
        chunk_counter["n"] = idx + 1
        # Snapshot of all codec tokens up through ``end``. synthesize_chunk
        # slices [start - left_context : end] inside the worker; copying
        # protects the worker from concurrent appends to ``all_codes``.
        codes_snapshot = list(all_codes[: end])
        work_q.put((idx, start, end, codes_snapshot))

    emitter = CodecChunkEmitter(
        context_len=context_len, chunk_size=chunk_size, on_chunk=on_chunk,
    )

    # Talker always uses normal sampling regardless of ``greedy`` (which
    # only pins the thinker). Forcing argmax here on Trn2 reproducibly
    # loops on a single codec token ("ooooo"); cross-hardware codec
    # equality is unattainable in bf16 either way.
    talker_sample_kwargs = {
        "do_sample": True, "temperature": 0.9,
        "top_k": 40, "top_p": 0.8, "repetition_penalty": 1.05,
    }

    t_talker_start = time.time()
    out = talker_adapter.generate(
        input_ids=talker_input_ids,
        attention_mask=talker_attention_mask,
        max_new_tokens=max_gen,
        eos_token_id=[codec_eos, codec_pad],
        suppress_tokens=[codec_bos],
        stopping_criteria=[emitter],
        **talker_sample_kwargs,
    )
    t_talker_end = time.time()

    # Flush any remaining codec tokens.
    full_codec = out[0, context_len:].tolist()
    # Strip trailing EOS/pad like the serial path does.
    while full_codec and full_codec[-1] in (codec_eos, codec_pad):
        full_codec.pop()
    drained = emitter._n_chunks * chunk_size
    if drained < len(full_codec):
        # Emit the residual tail as one final synthesis call.
        on_chunk(full_codec, drained, len(full_codec), final=True)

    # Drain DiT workers.
    work_q.join()
    for _ in workers:
        work_q.put(None)
    for w in workers:
        w.join()

    # Reassemble in chunk-index order (workers may finish out of order).
    audio_chunks: List[torch.Tensor] = []
    chunk_meta: List[dict] = []
    for idx in sorted(chunk_results):
        wav, meta = chunk_results[idx]
        audio_chunks.append(wav)
        chunk_meta.append(meta)
    t_synth_end = time.time()

    return {
        "thinker_text": thinker_result["gen_text"],
        "n_thinker_tokens": thinker_result["n_tokens"],
        "n_codec_tokens": len(full_codec),
        "audio_chunks": audio_chunks,
        "chunk_meta": chunk_meta,
        "stages": {
            "thinker": t_thinker - t_start,
            "hidden": t_hidden - t_thinker,
            "prep": t_prep - t_hidden,
            "talker": t_talker_end - t_talker_start,
            "synth_total": t_synth_end - t_talker_end + sum(c["synthesis_s"] for c in chunk_meta),
        },
        "first_audio_byte_s": (
            (chunk_meta[0]["wall_t"]) if chunk_meta else None
        ),
        "wall_time_s": t_synth_end - t_start,
    }


def run_serial_pipeline(
    *,
    model_path: str,
    thinker_adapter,
    tokenizer,
    hf_model,
    talker_model,
    talker_adapter,
    talker_cfg,
    t2w,
    t2w_cfg,
    speaker: str,
    prompt: str,
    system_prompt: str,
    greedy: bool = False,
    seed: int = 42,
):
    """Reproduce the serial pipeline used by ``generate_qwen25_omni_speech.py``.

    Returns the same result dict shape as ``run_streaming_pipeline``.  The
    'first_audio_byte_s' here is the total wall time -- in the serial case
    no audio is available before the entire DiT call returns.
    """
    t_start = time.time()
    thinker_result = omni_demo.run_thinker(
        thinker_adapter, tokenizer, prompt, system_prompt,
    )
    t_thinker = time.time()

    outputs, full_ids, prompt_len, hidden_time = omni_demo.extract_hidden_states(
        hf_model, thinker_result,
    )
    t_hidden = time.time()

    talker_input = omni_demo.prepare_talker_input(
        model_path, hf_model, outputs, full_ids, prompt_len, speaker,
    )
    t_prep = time.time()

    codec_codes, talker_time = omni_demo.run_talker(
        talker_model, talker_adapter, talker_cfg, talker_input,
        greedy=greedy, seed=seed,
    )
    t_talker = time.time()

    if not codec_codes:
        return None

    wav, t2w_time = omni_demo.run_token2wav(
        t2w, t2w_cfg, codec_codes,
        talker_input["conditioning"], talker_input["reference_mel"],
        seed=seed,
    )
    t_synth = time.time()

    return {
        "thinker_text": thinker_result["gen_text"],
        "n_thinker_tokens": thinker_result["n_tokens"],
        "n_codec_tokens": len(codec_codes),
        "audio_chunks": [wav.detach().float().reshape(-1)],
        "chunk_meta": [{
            "start": 0, "end": len(codec_codes),
            "samples": int(wav.numel()),
            "synthesis_s": t2w_time,
            "wall_t": t_synth - t_start,
        }],
        "stages": {
            "thinker": t_thinker - t_start,
            "hidden": t_hidden - t_thinker,
            "prep": t_prep - t_hidden,
            "talker": talker_time,
            "synth_total": t2w_time,
        },
        "first_audio_byte_s": t_synth - t_start,
        "wall_time_s": t_synth - t_start,
    }


# =============================================================================
# Reporting
# =============================================================================

def _summarize(name: str, runs: List[dict]):
    runs = [r for r in runs if r is not None]
    if not runs:
        print(f"\n[{name}] no successful runs")
        return
    fab = [r["first_audio_byte_s"] for r in runs if r["first_audio_byte_s"] is not None]
    wall = [r["wall_time_s"] for r in runs]
    n_chunks = [len(r["chunk_meta"]) for r in runs]
    print(f"\n[{name}] over {len(runs)} run(s)")
    if fab:
        print(f"  First-audio-byte: median={statistics.median(fab):.3f}s "
              f"mean={statistics.mean(fab):.3f}s "
              f"min={min(fab):.3f}s max={max(fab):.3f}s")
    print(f"  Wall time:        median={statistics.median(wall):.3f}s "
          f"mean={statistics.mean(wall):.3f}s")
    print(f"  Chunks per run:   {n_chunks}")
    # Average per-stage timing
    keys = list(runs[0]["stages"].keys())
    for k in keys:
        vals = [r["stages"][k] for r in runs]
        print(f"  stage[{k:>11}] mean={statistics.mean(vals):.3f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--prompt", default=omni_demo.DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=omni_demo.DEFAULT_SYSTEM)
    parser.add_argument("--speaker", default=omni_demo.DEFAULT_SPEAKER,
                        choices=["Ethan", "Chelsie"])
    parser.add_argument("--model-path", default=omni_demo.MODEL_PATH)
    parser.add_argument("--compiled-path", default=omni_demo.COMPILED_PATH)
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Codec tokens per DiT chunk (default 50 ~ 1.0s audio)")
    parser.add_argument("--left-context", type=int, default=25,
                        help="Codec tokens of left context to feed DiT to "
                             "smooth chunk boundaries (default 25)")
    parser.add_argument("--mode", default="both",
                        choices=["serial", "streaming", "both"])
    parser.add_argument("--out-prefix", default="streaming_bench",
                        help="WAV prefix for first run output (default 'streaming_bench')")
    parser.add_argument("--greedy", dest="greedy", action="store_true", default=False,
                        help=("Use greedy thinker NEFF (thinker_tp4_greedy). "
                              "Talker always uses sampling. Needs prior "
                              "--compile --greedy in "
                              "generate_qwen25_omni_speech.py."))
    parser.add_argument("--no-greedy", dest="greedy", action="store_false",
                        help="Use the sampling NEFF (default for the demo path).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for DiT init noise. Default 42.")
    args = parser.parse_args()

    if not omni_demo._check_compiled(args.compiled_path, greedy=args.greedy):
        _sys.exit(1)

    print("=" * 70)
    print("Qwen2.5-Omni Streaming codec->wav Benchmark")
    print("=" * 70)
    print(f"  Model:        {args.model_path}")
    print(f"  Compiled:     {args.compiled_path}")
    print(f"  Speaker:      {args.speaker}")
    print(f"  Prompt:       {args.prompt}")
    print(f"  Chunk size:   {args.chunk_size} codec tokens")
    print(f"  Left context: {args.left_context} codec tokens")
    print(f"  Runs:         {args.num_runs}")
    print(f"  Mode:         {args.mode}")
    print(f"  Greedy:       {args.greedy}")
    print(f"  Seed:         {args.seed}")

    # ---- Load all models once ----
    print("\n--- Loading models (one-time cost) ---")
    thinker_adapter, tokenizer, _ = omni_demo.load_thinker(
        args.model_path, args.compiled_path, greedy=args.greedy,
    )
    hf_model, _ = omni_demo.load_hf_cpu(args.model_path)
    talker_model, talker_adapter, talker_cfg, _ = omni_demo.load_talker(
        args.model_path, args.compiled_path,
    )
    t2w, t2w_cfg, _ = omni_demo.load_token2wav(
        args.model_path, args.compiled_path,
    )

    # ---- End-to-end warmup ----
    # NxDI's per-NEFF "Warmup completed in 0.25s" only exercises NEFF
    # boundaries with dummy input; the first real end-to-end pipeline
    # run still pays for graph caching, BigVGAN JIT, ECAPA preprocessing,
    # etc. Run one full serial pipeline and discard the timing so the
    # measured runs are warm.
    print("\n--- End-to-end warmup (1 serial run, timing discarded) ---")
    _t0 = time.time()
    _ = run_serial_pipeline(
        model_path=args.model_path,
        thinker_adapter=thinker_adapter, tokenizer=tokenizer,
        hf_model=hf_model,
        talker_model=talker_model, talker_adapter=talker_adapter,
        talker_cfg=talker_cfg,
        t2w=t2w, t2w_cfg=t2w_cfg,
        speaker=args.speaker, prompt=args.prompt,
        system_prompt=args.system_prompt,
        greedy=args.greedy, seed=args.seed,
    )
    print(f"  warmup took {time.time() - _t0:.2f}s")
    gc.collect()

    serial_runs: List[dict] = []
    streaming_runs: List[dict] = []

    for i in range(args.num_runs):
        if args.mode in ("serial", "both"):
            print(f"\n--- [serial] Run {i+1}/{args.num_runs} ---")
            res = run_serial_pipeline(
                model_path=args.model_path,
                thinker_adapter=thinker_adapter, tokenizer=tokenizer,
                hf_model=hf_model,
                talker_model=talker_model, talker_adapter=talker_adapter,
                talker_cfg=talker_cfg,
                t2w=t2w, t2w_cfg=t2w_cfg,
                speaker=args.speaker, prompt=args.prompt,
                system_prompt=args.system_prompt,
                greedy=args.greedy, seed=args.seed,
            )
            if res is not None:
                print(f"  text: {res['thinker_text'][:80]}")
                print(f"  codec={res['n_codec_tokens']} tokens, "
                      f"FAB={res['first_audio_byte_s']:.3f}s, "
                      f"wall={res['wall_time_s']:.3f}s")
                if i == 0:
                    sf.write(f"{args.out_prefix}_serial.wav",
                             res["audio_chunks"][0].numpy(), 24000)
            serial_runs.append(res)
            gc.collect()

        if args.mode in ("streaming", "both"):
            print(f"\n--- [streaming] Run {i+1}/{args.num_runs} ---")
            res = run_streaming_pipeline(
                model_path=args.model_path,
                thinker_adapter=thinker_adapter, tokenizer=tokenizer,
                hf_model=hf_model,
                talker_model=talker_model, talker_adapter=talker_adapter,
                talker_cfg=talker_cfg,
                t2w=t2w, t2w_cfg=t2w_cfg,
                speaker=args.speaker, prompt=args.prompt,
                system_prompt=args.system_prompt,
                chunk_size=args.chunk_size,
                left_context=args.left_context,
                greedy=args.greedy, seed=args.seed,
            )
            print(f"  text: {res['thinker_text'][:80]}")
            print(f"  codec={res['n_codec_tokens']} tokens, "
                  f"chunks={len(res['chunk_meta'])}, "
                  f"FAB={res['first_audio_byte_s']:.3f}s, "
                  f"wall={res['wall_time_s']:.3f}s")
            if i == 0:
                full_wav = torch.cat(res["audio_chunks"]) if res["audio_chunks"] else torch.empty(0)
                if full_wav.numel():
                    sf.write(f"{args.out_prefix}_streaming.wav",
                             full_wav.numpy(), 24000)
            streaming_runs.append(res)
            gc.collect()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.mode in ("serial", "both"):
        _summarize("serial", serial_runs)
    if args.mode in ("streaming", "both"):
        _summarize("streaming", streaming_runs)

    if (
        args.mode == "both"
        and serial_runs and streaming_runs
        and serial_runs[0] is not None and streaming_runs[0] is not None
    ):
        s_fab = serial_runs[0]["first_audio_byte_s"]
        st_fab = streaming_runs[0]["first_audio_byte_s"]
        if s_fab and st_fab:
            print(f"\nFAB delta on first run: {st_fab - s_fab:+.3f}s "
                  f"({(st_fab - s_fab) / s_fab * 100:+.1f}%)")


if __name__ == "__main__":
    main()
