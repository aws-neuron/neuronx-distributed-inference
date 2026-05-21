#!/usr/bin/env python3
"""Single-H100 baseline benchmark for Qwen2.5-Omni-7B (GPU reference).

Reports the same metrics as ``test_ttfb_streaming_bench.py`` so the GPU
baseline can be compared apples-to-apples with the Neuron Trn2 numbers:

  - Thinker time (decode of the text reply)
  - Talker time (codec-token decode)
  - Token2Wav time (DiT + BigVGAN)
  - First-audio-byte (FAB) latency: prompt-in -> first wav samples available
  - Total wall time

This script does NOT depend on neuronx_distributed_inference; it uses the
upstream HuggingFace Qwen2_5Omni implementation directly. Run it on a node
with one H100/A100/H200 80GB GPU.

Two modes:

  full       (default) -- single ``model.generate(..., return_audio=True)``.
                          Equivalent to the Neuron *serial* baseline:
                          first audio byte = total wall time.

  streaming  -- run the thinker, the talker, and the token2wav stages
                separately and chunk the codec stream into Token2Wav calls
                of ``--chunk-size`` codec tokens with ``--left-context``
                tokens of context. Mirrors the Neuron streaming bench so
                first-audio-byte is comparable across hardware.

Usage::

    # On the GPU host (P5EN-1 etc.):
    python -m venv ~/qwen-omni-gpu && source ~/qwen-omni-gpu/bin/activate
    pip install --upgrade transformers accelerate soundfile torchaudio \
        qwen-omni-utils flash-attn

    # Cache weights once (~16GB):
    huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /opt/dlami/nvme/qwen25_omni_hf

    # Benchmark
    python test_gpu_baseline_bench.py \
        --model-path /opt/dlami/nvme/qwen25_omni_hf \
        --num-runs 3 \
        --mode full

    # Streaming
    python test_gpu_baseline_bench.py --mode streaming --chunk-size 50
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from typing import List, Optional

import torch

try:
    import soundfile as sf
except ImportError:
    sys.exit("soundfile is required. pip install soundfile")


# Match the Neuron benchmark prompt so generated text length is similar.
DEFAULT_PROMPT = "Say hello and briefly introduce yourself in two sentences."
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
DEFAULT_SPEAKER = "Ethan"
SAMPLE_RATE = 24000


# =============================================================================
# Loading
# =============================================================================

def load_model(model_path: str, dtype: torch.dtype, attn_impl: str):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"  Loading processor from {model_path} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    print(f"  Loading model ({dtype}, attn={attn_impl}) ...")
    t0 = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda:0",
        attn_implementation=attn_impl,
        enable_audio_output=True,
    )
    model.eval()
    load_s = time.time() - t0
    print(f"  Model loaded in {load_s:.1f}s")
    return model, processor, load_s


def build_text_inputs(processor, prompt: str, system_prompt: str, device: str):
    """Build a text-only chat input identical to the Neuron benchmark."""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


# =============================================================================
# Mode 1: Full generate(), no streaming -- matches Neuron "serial" baseline
# =============================================================================

def run_full_pipeline(
    *,
    model,
    processor,
    prompt: str,
    system_prompt: str,
    speaker: str,
    max_new_tokens: int,
    use_audio_in_video: bool,
):
    """One end-to-end generate(); returns text + waveform + timing dict."""
    device = next(model.parameters()).device
    inputs = build_text_inputs(processor, prompt, system_prompt, str(device))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    with torch.inference_mode():
        text_ids, audio = model.generate(
            **inputs,
            speaker=speaker,
            use_audio_in_video=use_audio_in_video,
            return_audio=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.time()

    text = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    n_text = int(text_ids.shape[1] - inputs["input_ids"].shape[1])

    wav = audio.detach().float().reshape(-1).cpu()

    return {
        "text": text,
        "n_text_tokens": n_text,
        "wav": wav,
        "audio_seconds": float(wav.numel()) / SAMPLE_RATE,
        "first_audio_byte_s": t_end - t_start,
        "wall_time_s": t_end - t_start,
        "stages": {"end_to_end": t_end - t_start},
    }


# =============================================================================
# Mode 2: Streaming -- thinker -> talker -> chunked token2wav
# =============================================================================
#
# Mirrors the Neuron streaming bench:
#   - Run the thinker first (text generate on Qwen2_5OmniThinkerForConditionalGeneration)
#   - Pull thinker hidden states (always-on in HF)
#   - Project to talker, run talker.generate() with a StoppingCriteria that
#     emits codec chunks of `chunk_size` tokens
#   - For each chunk, call code2wav with `left_context` tokens of prefix and
#     trim the prefix samples back off
#
# Compared to Neuron, the HF Qwen2_5Omni already runs all 3 components in
# one process on the same device, so we don't need the CPU re-forward step
# (extract_hidden_states); we read thinker hidden states from generate().

class _ChunkEmitter:
    """StoppingCriteria-style hook for talker.generate().

    Records the wall-clock time of the first emitted chunk so we can
    report first-audio-byte latency relative to ``t_start``.
    """

    def __init__(self, context_len: int, chunk_size: int, on_chunk):
        from transformers.generation.stopping_criteria import StoppingCriteria
        self._base = StoppingCriteria
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.on_chunk = on_chunk
        self.codec_tokens: List[int] = []
        self._n_chunks = 0
        self.first_chunk_t: Optional[float] = None

    def __call__(self, input_ids: torch.Tensor, scores) -> torch.Tensor:
        new_total = input_ids.shape[1]
        produced = max(0, new_total - self.context_len)
        already = len(self.codec_tokens)
        if produced > already:
            extra = input_ids[
                0, self.context_len + already : self.context_len + produced
            ].tolist()
            self.codec_tokens.extend(extra)
        while len(self.codec_tokens) - self._n_chunks * self.chunk_size >= self.chunk_size:
            start = self._n_chunks * self.chunk_size
            end = start + self.chunk_size
            self.on_chunk(self.codec_tokens, start, end, final=False)
            self._n_chunks += 1
            if self.first_chunk_t is None:
                self.first_chunk_t = time.time()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _synthesize_chunk_gpu(
    code2wav,
    codec_codes: List[int],
    start: int,
    end: int,
    *,
    left_context: int,
    conditioning: torch.Tensor,
    reference_mel: torch.Tensor,
    samples_per_token: int,
    codec_eos: int,
    codec_pad: int,
    device,
    dtype,
):
    eff_end = end
    while eff_end > start and codec_codes[eff_end - 1] in (codec_eos, codec_pad):
        eff_end -= 1
    if eff_end <= start:
        return torch.empty(0, dtype=torch.float32)

    ctx_start = max(0, start - left_context)
    chunk_codes = codec_codes[ctx_start:eff_end]
    code_tensor = torch.tensor([chunk_codes], dtype=torch.long, device=device)

    with torch.inference_mode():
        wav = code2wav(
            code=code_tensor,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=10,
            guidance_scale=0.5,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wav = wav.detach().float().reshape(-1).cpu()

    skip_samples = (start - ctx_start) * samples_per_token
    return wav[skip_samples:]


def run_streaming_pipeline(
    *,
    model,
    processor,
    prompt: str,
    system_prompt: str,
    speaker: str,
    max_new_tokens: int,
    chunk_size: int,
    left_context: int,
):
    """Thinker -> talker(generate w/ chunk hook) -> token2wav per chunk.

    Uses the HF Qwen2_5Omni internals (thinker / talker / token2wav). On
    GPU these are sub-modules of ``model`` and share the same CUDA stream.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = build_text_inputs(processor, prompt, system_prompt, str(device))

    thinker = model.thinker
    talker = model.talker
    code2wav = model.token2wav

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    # ----- Thinker -----
    with torch.inference_mode():
        thinker_out = thinker.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_thinker = time.time()

    full_ids = thinker_out.sequences
    prompt_len = inputs["input_ids"].shape[1]
    new_text_ids = full_ids[:, prompt_len:]
    text = processor.batch_decode(new_text_ids, skip_special_tokens=True)[0]

    # ----- Build talker input from thinker hidden states -----
    # Mirror HF Qwen2_5OmniForConditionalGeneration.generate (talker section):
    # mask out audio/image/video token embeds, build talker_input_ids =
    # [mask*context, codec_pad, codec_bos], stitch thinker_reply_part with
    # eos/pad, hand all of it to talker.generate. This is the same logic
    # the Trn2 streaming bench uses (via prepare_talker_input + run_talker
    # paired with set_vision_embeddings/thinker_reply_embeds).
    speaker_params = model.speaker_map[speaker]

    embeds_to_talker = thinker_out.hidden_states[0][0].clone().to(device)
    input_ids_t = inputs["input_ids"]
    if "input_features" in inputs:
        audio_ids_mask = input_ids_t == model.config.thinker_config.audio_token_index
        m = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        z = torch.zeros(
            [int(audio_ids_mask.sum()), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype, device=device,
        )
        embeds_to_talker.masked_scatter_(m, z)

    processed_thinker_hidden = (
        (embeds_to_talker,) + thinker_out.hidden_states[0][1:],
    ) + thinker_out.hidden_states[1:]
    thinker_token_embeds = [
        h[0].to(device) for h in processed_thinker_hidden
    ]
    thinker_hidden_states = [
        h[-1].to(device) for h in processed_thinker_hidden
    ]

    talker_text_bos_token = int(speaker_params["bos_token"])
    talker_input_ids = torch.cat(
        [
            torch.full_like(input_ids_t, fill_value=int(talker.codec_mask_token)),
            torch.tensor([[int(talker.codec_pad_token)]], dtype=torch.long, device=device),
            torch.tensor([[int(talker.codec_bos_token)]], dtype=torch.long, device=device),
        ],
        dim=1,
    )

    thinker_embed_tokens = model.thinker.get_input_embeddings()
    thinker_reply_part = (
        torch.cat(thinker_hidden_states[1:], dim=1)
        + torch.cat(thinker_token_embeds[1:], dim=1)
    )
    talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
    bos_embed = thinker_embed_tokens(
        torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=device)
    )
    talker_inputs_embeds = torch.cat(
        [talker_inputs_embeds, bos_embed, thinker_reply_part[:, :1, :]], dim=1
    )

    eos_embed = thinker_embed_tokens(
        torch.tensor([[int(talker.text_eos_token)]], dtype=torch.long, device=device)
    )
    pad_embed = thinker_embed_tokens(
        torch.tensor([[int(talker.text_pad_token)]], dtype=torch.long, device=device)
    )
    thinker_reply_part = torch.cat(
        [thinker_reply_part[:, 1:, :], eos_embed, pad_embed], dim=1
    )

    talker_input_text_ids = torch.cat(
        [
            input_ids_t,
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=device),
            full_ids[:, prompt_len:prompt_len + 1],
        ],
        dim=-1,
    )

    talker_attention_mask = None
    if "attention_mask" in inputs:
        talker_attention_mask = torch.cat(
            [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 2))],
            dim=1,
        ).to(device)
    t_prep = time.time()

    context_len = int(talker_input_ids.shape[1])
    conditioning = speaker_params["cond"].to(device).float()
    reference_mel = speaker_params["ref_mel"].to(device).float()
    codec_eos = int(talker.config.tts_codec_end_token_id)
    codec_pad = int(talker.config.tts_codec_pad_token_id)
    codec_bos = int(talker.config.tts_codec_start_token_id)

    # token2wav must run in fp32 (HF does this in generate as well).
    if code2wav.dtype != torch.float:
        code2wav.float()

    # Compute samples-per-codec-token for left-context trimming.
    bigvgan_cfg = getattr(code2wav.config, "bigvgan_config", None) or code2wav.bigvgan.config
    upsample_rates = getattr(bigvgan_cfg, "upsample_rates", [5, 3, 2, 2, 2, 2])
    total_upsample = 1
    for r in upsample_rates:
        total_upsample *= int(r)
    repeats = getattr(code2wav.config.dit_config, "repeats", 2)
    samples_per_token = repeats * total_upsample  # 24kHz / 50 codec tok/s = 480

    audio_chunks: List[torch.Tensor] = []
    chunk_meta: List[dict] = []

    def on_chunk(all_codes, start, end, final):
        t0 = time.time()
        wav = _synthesize_chunk_gpu(
            code2wav, all_codes, start, end,
            left_context=left_context,
            conditioning=conditioning,
            reference_mel=reference_mel,
            samples_per_token=samples_per_token,
            codec_eos=codec_eos, codec_pad=codec_pad,
            device=device, dtype=dtype,
        )
        elapsed = time.time() - t0
        audio_chunks.append(wav)
        chunk_meta.append({
            "start": start, "end": end,
            "samples": int(wav.numel()),
            "synthesis_s": elapsed,
            "wall_t": time.time() - t_start,
        })

    emitter = _ChunkEmitter(
        context_len=context_len, chunk_size=chunk_size, on_chunk=on_chunk,
    )

    # ----- Talker decode with chunk hook -----
    t_talker_start = time.time()
    with torch.inference_mode():
        out = talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            max_new_tokens=min(600, max_new_tokens * 25),
            eos_token_id=[codec_eos, codec_pad],
            suppress_tokens=[codec_bos],
            do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
            repetition_penalty=1.05,
            stopping_criteria=[emitter],
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_talker_end = time.time()

    # Flush residual codec tokens.
    full_codec = out[0, context_len:].tolist()
    while full_codec and full_codec[-1] in (codec_eos, codec_pad):
        full_codec.pop()
    drained = emitter._n_chunks * chunk_size
    if drained < len(full_codec):
        on_chunk(full_codec, drained, len(full_codec), final=True)
    t_synth_end = time.time()

    return {
        "text": text,
        "n_text_tokens": int(new_text_ids.shape[1]),
        "n_codec_tokens": len(full_codec),
        "audio_chunks": audio_chunks,
        "chunk_meta": chunk_meta,
        "stages": {
            "thinker": t_thinker - t_start,
            "prep": t_prep - t_thinker,
            "talker": t_talker_end - t_talker_start,
            "synth_total": (
                t_synth_end - t_talker_end
                + sum(c["synthesis_s"] for c in chunk_meta)
            ),
        },
        "first_audio_byte_s": chunk_meta[0]["wall_t"] if chunk_meta else None,
        "wall_time_s": t_synth_end - t_start,
    }


# =============================================================================
# Reporting
# =============================================================================

def _summarize(name: str, runs: List[dict]):
    runs = [r for r in runs if r is not None]
    if not runs:
        print(f"\n[{name}] no successful runs")
        return
    fab = [r["first_audio_byte_s"] for r in runs if r["first_audio_byte_s"]]
    wall = [r["wall_time_s"] for r in runs]
    print(f"\n[{name}] over {len(runs)} run(s)")
    if fab:
        print(f"  First-audio-byte: median={statistics.median(fab):.3f}s "
              f"mean={statistics.mean(fab):.3f}s "
              f"min={min(fab):.3f}s max={max(fab):.3f}s")
    print(f"  Wall time:        median={statistics.median(wall):.3f}s "
          f"mean={statistics.mean(wall):.3f}s")
    keys = list(runs[0]["stages"].keys())
    for k in keys:
        vals = [r["stages"][k] for r in runs if k in r["stages"]]
        if vals:
            print(f"  stage[{k:>16}] mean={statistics.mean(vals):.3f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get(
        "QWEN25_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"))
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER,
                        choices=["Ethan", "Chelsie"])
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max thinker text tokens (default 128)")
    parser.add_argument("--mode", default="full",
                        choices=["full", "streaming", "both"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--left-context", type=int, default=25)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-impl", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--out-prefix", default="gpu_baseline")
    parser.add_argument("--use-audio-in-video", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- this script requires a GPU.")

    torch_dtype = {"bfloat16": torch.bfloat16,
                   "float16": torch.float16,
                   "float32": torch.float32}[args.dtype]

    print("=" * 70)
    print("Qwen2.5-Omni GPU baseline benchmark")
    print("=" * 70)
    print(f"  Device:       {torch.cuda.get_device_name(0)}")
    print(f"  Model path:   {args.model_path}")
    print(f"  Speaker:      {args.speaker}")
    print(f"  Prompt:       {args.prompt}")
    print(f"  Mode:         {args.mode}")
    print(f"  dtype:        {args.dtype}")
    print(f"  attn_impl:    {args.attn_impl}")
    print(f"  num_runs:     {args.num_runs}")
    if args.mode in ("streaming", "both"):
        print(f"  chunk_size:   {args.chunk_size}")
        print(f"  left_context: {args.left_context}")

    print("\n--- Loading model ---")
    model, processor, _ = load_model(args.model_path, torch_dtype, args.attn_impl)

    full_runs: List[dict] = []
    streaming_runs: List[dict] = []

    for i in range(args.num_runs):
        if args.mode in ("full", "both"):
            print(f"\n--- [full] Run {i+1}/{args.num_runs} ---")
            res = run_full_pipeline(
                model=model, processor=processor,
                prompt=args.prompt, system_prompt=args.system_prompt,
                speaker=args.speaker,
                max_new_tokens=args.max_new_tokens,
                use_audio_in_video=args.use_audio_in_video,
            )
            print(f"  text: {res['text'][:80]}")
            print(f"  audio={res['audio_seconds']:.2f}s, "
                  f"FAB={res['first_audio_byte_s']:.3f}s, "
                  f"wall={res['wall_time_s']:.3f}s")
            if i == 0:
                sf.write(f"{args.out_prefix}_full.wav",
                         res["wav"].numpy(), SAMPLE_RATE)
            full_runs.append(res)
            gc.collect()
            torch.cuda.empty_cache()

        if args.mode in ("streaming", "both"):
            print(f"\n--- [streaming] Run {i+1}/{args.num_runs} ---")
            res = run_streaming_pipeline(
                model=model, processor=processor,
                prompt=args.prompt, system_prompt=args.system_prompt,
                speaker=args.speaker,
                max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size,
                left_context=args.left_context,
            )
            print(f"  text: {res['text'][:80]}")
            print(f"  codec={res['n_codec_tokens']} tokens, "
                  f"chunks={len(res['chunk_meta'])}, "
                  f"FAB={res['first_audio_byte_s']:.3f}s, "
                  f"wall={res['wall_time_s']:.3f}s")
            if i == 0 and res["audio_chunks"]:
                full_wav = torch.cat(res["audio_chunks"])
                if full_wav.numel():
                    sf.write(f"{args.out_prefix}_streaming.wav",
                             full_wav.numpy(), SAMPLE_RATE)
            streaming_runs.append(res)
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.mode in ("full", "both"):
        _summarize("full", full_runs)
    if args.mode in ("streaming", "both"):
        _summarize("streaming", streaming_runs)


if __name__ == "__main__":
    main()
