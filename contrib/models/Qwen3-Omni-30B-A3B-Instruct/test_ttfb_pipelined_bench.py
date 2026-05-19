#!/usr/bin/env python3
"""Pipelined TTFB/RTF benchmark — thinker and talker run concurrently.

Baseline `test_ttfb_rtf_bench.py` is strictly serial: full thinker.generate →
build talker inputs → talker.generate. This bench overlaps them:

1. Thinker runs in a background thread. A custom streamer pushes every new
   thinker token into a queue. A forward hook captures the layer-23 hidden
   (one per decode step) and puts it in the same queue.
2. Main thread waits for the first 4 thinker tokens (needed to build the
   talker's prefill input), then kicks off talker.generate.
3. Talker's `prepare_inputs_for_generation` is monkey-patched: when the HF
   loop asks for `trailing_text_hidden[:, k]` at decode step k, the patched
   function pulls the (k+4)-th thinker embedding from the streaming buffer
   — blocking until that token is available. Usually it's already there
   because talker decode (~21 ms/step) is slower than thinker decode
   (~10 ms/step).

Streaming code2wav stays the same as the serial bench — chunk-sized inline
c2w calls.

Expected win: TTFB drops from ~2000 ms to ~1400 ms because we no longer
wait for the last ~60 thinker tokens before starting the talker.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0-7 CHUNK_SIZE=25 LEFT_CTX=5 \
      python test_ttfb_pipelined_bench.py --num 100 --neuron-c2w
"""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-7")
os.environ.setdefault("QWEN3_OMNI_CAPTURE_LAYER_HIDDEN", "23")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "src", Path("/home/ubuntu/whn-ndi/contrib/models/Qwen3-Omni-30B-A3B-Instruct/src")):
    if (_candidate / "_upstream_compat.py").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
if "/home/ubuntu" not in sys.path:
    sys.path.insert(0, "/home/ubuntu")

import _upstream_compat  # noqa: F401

import argparse
import functools
import json
import queue
import statistics
import threading
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer

import test_audio_streaming as STR
sys.path.insert(0, str(_HERE))
from code2wav_neuron import install_neuron_code2wav  # noqa: E402

CONV_JSON = "/home/ubuntu/omni2/merged_conversations_with_audio_x10_with_system.json"
AUDIO_DIR = "/home/ubuntu/omni2/speech_wav_16k"


def build_messages(conv):
    msgs = conv["messages"]
    out = []
    for i, m in enumerate(msgs):
        if i == len(msgs) - 1:
            break
        role = m["role"]
        content = m["content"]
        if i == len(msgs) - 2 and role == "user":
            fname = os.path.basename(content)
            wav_path = os.path.join(AUDIO_DIR, fname)
            out.append({"role": role, "content": [{"type": "audio", "audio": wav_path}]})
        else:
            out.append({"role": role, "content": content})
    return out


# ---------------------------------------------------------------------------
# Streaming thinker→talker plumbing
# ---------------------------------------------------------------------------

class PipelineState:
    """Shared between thinker thread and main thread.

    As thinker emits each assistant token, we accumulate the token id and the
    layer-23 hidden, and push-notify the waiting talker side via a condition
    variable. ``assistant_start_idx`` is the prompt length — we skip the
    prompt tokens the streamer sees at the beginning.
    """
    def __init__(self, assistant_start_idx: int):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.assistant_start_idx = assistant_start_idx
        self.token_ids: list[int] = []          # assistant tokens only
        self.layer23_hidden: list[torch.Tensor] = []  # one per thinker forward (prefill + decode)
        self.thinker_done = False
        self.thinker_error: Exception | None = None

        # Populated once the prefill's layer-23 output is captured (needed
        # for _build_talker_inputs' USER turn portions).
        self.prefill_hidden: torch.Tensor | None = None

    def push_token(self, token_id: int):
        with self.cond:
            self.token_ids.append(token_id)
            self.cond.notify_all()

    def push_layer23(self, hid: torch.Tensor):
        with self.cond:
            self.layer23_hidden.append(hid)
            if self.prefill_hidden is None:
                # First call = prefill, shape [1, bucket, 2048]. Later calls
                # are decode shape [1, 1, 2048].
                self.prefill_hidden = hid
            self.cond.notify_all()

    def mark_done(self, exc: Exception | None = None):
        with self.cond:
            self.thinker_done = True
            self.thinker_error = exc
            self.cond.notify_all()

    def wait_for_tokens(self, count: int, timeout: float = 30.0) -> bool:
        """Block until at least ``count`` assistant tokens are available."""
        deadline = time.perf_counter() + timeout
        with self.cond:
            while len(self.token_ids) < count:
                if self.thinker_done:
                    return len(self.token_ids) >= count
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=remaining)
            return True


class TokenStreamStoppingCriteria:
    """Abuses HF's StoppingCriteria plumbing as a per-step callback.

    NxDI's custom `_sample` ignores `streamer` kwarg, but it DOES call
    `stopping_criteria(input_ids, None)` after every decode step with the
    current ``input_ids`` buffer. We piggy-back on that to notify
    ``PipelineState`` whenever a new token is appended.
    """

    def __init__(self, state: PipelineState, prompt_len: int):
        self.state = state
        self.prompt_len = prompt_len
        self._last_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.Tensor:
        cur_len = int(input_ids.shape[1])
        if cur_len > self._last_len:
            # Push all tokens added since last call (usually just 1)
            for idx in range(self._last_len, cur_len):
                self.state.push_token(int(input_ids[0, idx].item()))
            self._last_len = cur_len
        # Never stop on our account
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def run_thinker(adapter, gen_kwargs, state: PipelineState, prompt_len: int):
    """Executed in a background thread. Runs the full thinker.generate while
    piping tokens + layer-23 hiddens into ``state``."""
    try:
        def _cap_hook(_m, tensors):
            if tensors:
                state.push_layer23(tensors[0].clone().to("cpu"))

        from transformers.generation.stopping_criteria import StoppingCriteriaList
        sc = StoppingCriteriaList([TokenStreamStoppingCriteria(state, prompt_len)])

        gen_kwargs = dict(gen_kwargs)
        gen_kwargs["tensor_capture_hook"] = _cap_hook
        gen_kwargs["stopping_criteria"] = sc
        out_ids = adapter.generate(**gen_kwargs)
        with state.cond:
            state.out_ids = out_ids
        state.mark_done()
    except Exception as e:
        state.mark_done(exc=e)


# ---------------------------------------------------------------------------
# Pipelined talker setup — incremental trailing_text_hidden
# ---------------------------------------------------------------------------

class StreamingTalkerInputs:
    """Replaces the single-shot ``_build_talker_inputs``.

    Holds:
      * ``talker_embed``: the user-parts + first-4-assistant-tokens buffer
        (built once both are ready)
      * ``trailing_text_hidden``: a tensor we grow by one row per newly-arrived
        thinker assistant token past the 4th
      * ``talker_input_ids``: mirrors talker_embed's sequence length

    Reads from ``PipelineState`` under its condition variable.
    """
    def __init__(self, hf_model, conv_inputs, state: PipelineState, speaker: str = "ethan"):
        self.hf_model = hf_model
        self.state = state
        self.cfg = hf_model.config
        self.speaker_id = self.cfg.talker_config.speaker_id[speaker.lower()]
        self.conv_inputs = conv_inputs  # original user-turn inputs
        self.dtype = torch.bfloat16

        # Pre-compute static parts
        embed_layer = hf_model.thinker.get_input_embeddings()
        talker_special = torch.tensor(
            [[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id]],
            dtype=torch.long,
        )
        with torch.no_grad():
            self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = (
                hf_model.talker.text_projection(embed_layer(talker_special)).chunk(3, dim=1)
            )
        self._embed_layer = embed_layer
        self._codec_special_tokens = torch.tensor(
            [[
                self.cfg.talker_config.codec_nothink_id,
                self.cfg.talker_config.codec_think_bos_id,
                self.cfg.talker_config.codec_think_eos_id,
                self.speaker_id,
                self.cfg.talker_config.codec_pad_id,
                self.cfg.talker_config.codec_bos_id,
            ]], dtype=torch.long,
        )
        self._codec_special_embeds = hf_model.talker.get_input_embeddings()(
            self._codec_special_tokens
        ).to(self.dtype)

    def build_prefill(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Block until 4 assistant tokens + all prefill USER hidden are ready,
        then assemble the talker_embed / talker_input_ids for talker prefill.

        Returns (talker_input_embed, talker_input_ids).
        """
        # Wait for prefill hidden (USER-turn hidden needed for _get_talker_user_parts)
        with self.state.cond:
            while self.state.prefill_hidden is None and not self.state.thinker_done:
                self.state.cond.wait()
            if self.state.thinker_done and self.state.thinker_error is not None:
                raise self.state.thinker_error
        # Wait for 4 assistant tokens (needed for the prefill slice)
        ok = self.state.wait_for_tokens(4)
        if not ok:
            raise RuntimeError("thinker did not produce 4 assistant tokens in time")

        # -- Build current thinker hidden for USER parts only --
        # At this point, prefill_hidden has the USER + system + assistant-role
        # header embedded. Decode-step hiddens (one per assistant token) are
        # appended to layer23_hidden[1:]. For the talker USER parts, we only
        # need positions up to assistant_start_idx — all in prefill_hidden.
        prompt_len = self.state.assistant_start_idx
        prefill_h = self.state.prefill_hidden[:, :prompt_len, :].to(self.dtype)

        # -- Build current thinker_embed up to assistant_start + 4 tokens --
        with self.state.cond:
            first_asst_ids = list(self.state.token_ids[:4])
        assistant_ids = torch.tensor([first_asst_ids], dtype=torch.long)
        prompt_ids = self.conv_inputs.input_ids
        all_ids = torch.cat([prompt_ids, assistant_ids], dim=1)
        with torch.no_grad():
            thinker_embed = self._embed_layer(all_ids).to(self.dtype)

        cfg = self.cfg
        im_start_indexes = torch.cat((
            torch.nonzero(all_ids[0] == cfg.im_start_token_id).squeeze(),
            torch.tensor([all_ids.shape[-1]], dtype=all_ids.dtype),
        ), dim=-1)
        multimodal_mask = (
            (all_ids == cfg.thinker_config.audio_token_id)
            | (all_ids == cfg.thinker_config.image_token_id)
            | (all_ids == cfg.thinker_config.video_token_id)
        )

        # assistant_hidden for the prefill = text_projection of first-4 assistant embeddings
        assistant_embed_first4 = thinker_embed[:, prompt_len:prompt_len + 4]
        assistant_hidden_first4 = self.hf_model.talker.text_projection(
            assistant_embed_first4
        ).to(self.dtype)

        # --- USER parts ---
        # Pad thinker_hidden out to all_ids length so _get_talker_user_parts
        # can index it; only the USER positions within prompt are really used.
        hidden_full = torch.zeros(
            (1, all_ids.shape[1], prefill_h.shape[-1]), dtype=self.dtype,
        )
        hidden_full[:, :prompt_len, :] = prefill_h
        talker_embeds = []
        talker_ids_list = []
        for i in range(len(im_start_indexes) - 1):
            ims = im_start_indexes[i]
            segend = im_start_indexes[i + 1]
            role = all_ids[0][ims + 1]
            if role == cfg.system_token_id:
                continue
            if role == cfg.user_token_id:
                part = self.hf_model._get_talker_user_parts(
                    ims, segend, multimodal_mask, hidden_full, thinker_embed,
                )
                talker_embeds.append(part)
                talker_ids_list.append(all_ids[:, ims:segend])
            # Assistant turn with im_start inside prompt (prior turns). For the
            # final assistant turn (our generated one), we manually build below.
            elif role == cfg.assistant_token_id:
                # Only the very last im_start_index is our freshly-started
                # assistant turn. Previous assistants are in prompt context
                # and skipped (HF does the same in _build_talker_inputs).
                if i == len(im_start_indexes) - 2:
                    # This is the new assistant turn — build from the first 4
                    # tokens now available.
                    assistant_text_hidden = torch.cat((
                        assistant_hidden_first4[:, :3],
                        self.tts_pad_embed.expand(-1, 4, -1),
                        self.tts_bos_embed,
                        assistant_hidden_first4[:, 3:4],
                    ), dim=1)
                    assistant_codec_hidden = torch.cat((
                        torch.zeros(
                            (1, 3, cfg.talker_config.text_config.hidden_size),
                            dtype=self.dtype,
                        ),
                        self._codec_special_embeds,
                    ), dim=1)
                    input_embeds_asst = assistant_text_hidden + assistant_codec_hidden
                    input_ids_asst = torch.full(
                        (1, assistant_text_hidden.shape[1]),
                        fill_value=cfg.tts_pad_token_id, dtype=torch.long,
                    )
                    talker_embeds.append(input_embeds_asst)
                    talker_ids_list.append(input_ids_asst)
                # else: prior assistant turns — skipped (HF also skips them)

        talker_embed = torch.cat(talker_embeds, dim=1)
        talker_input_ids = torch.cat(talker_ids_list, dim=1)
        return talker_embed, talker_input_ids

    def get_trailing_slice(self, k: int) -> torch.Tensor:
        """Return the k-th entry of trailing_text_hidden.

        trailing_text_hidden[k] corresponds to the (k+4)-th thinker assistant
        embedding (via text_projection), per HF's assembly. For the tail
        position past all thinker tokens, return tts_eos_embed.
        """
        needed = k + 5  # need tokens[0..k+4], which is k+5 tokens
        # Block until the (k+4)-th token is produced, or thinker finishes.
        ok = self.state.wait_for_tokens(needed, timeout=60.0)
        with self.state.cond:
            n = len(self.state.token_ids)
            done = self.state.thinker_done
        if n >= k + 5:
            tok_id = self.state.token_ids[k + 4]
            # text_projection of the embedding for this single token
            with torch.no_grad():
                e = self._embed_layer(torch.tensor([[tok_id]], dtype=torch.long)).to(self.dtype)
                h = self.hf_model.talker.text_projection(e).to(self.dtype)
            return h  # shape [1, 1, 1024]
        # Past end of thinker output → tts_eos_embed
        return self.tts_eos_embed


def install_pipelined_prepare_inputs(hf_model, sti: StreamingTalkerInputs):
    """Wrap the current ``Qwen3OmniMoeTalkerForConditionalGeneration.prepare_inputs_for_generation``
    (possibly already patched by ``install_streaming_talker_hook``) with a
    layer that rewrites ``kwargs["trailing_text_hidden"]`` before the call so
    that each decode step's indexing picks up a streaming-sourced row.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as Cls,
    )
    # Wrap WHATEVER is currently set (may be HF original, may be the streaming
    # hook's wrapper). We only install once per run — save and restore below.
    prev_prep = Cls.prepare_inputs_for_generation
    step_counter = {"n": 0}

    @functools.wraps(prev_prep)
    def patched(self_talker, input_ids, *args, **kwargs):
        # HF's talker prepare_inputs reads kwargs["trailing_text_hidden"] as a
        # dense pre-built tensor indexed by ``generation_step``. We replace
        # that slot on the fly with a tensor whose ``[:, gen_step]`` row is
        # fetched from the streaming thinker output (blocks if not yet
        # produced). Non-decode calls (prefill) have no such indexing and
        # pass-through unchanged.
        if "trailing_text_hidden" in kwargs:
            gen_step = kwargs.get("generation_step")
            if gen_step is None:
                gen_step = step_counter["n"]
            if gen_step is not None and gen_step >= 0:
                slice_h = sti.get_trailing_slice(gen_step)  # [1, 1, hidden]
                trailing = kwargs["trailing_text_hidden"]
                if trailing is None or trailing.shape[1] <= gen_step:
                    # Build a minimal tensor sized to the current step.
                    hidden_dim = slice_h.shape[-1]
                    fresh = torch.zeros((1, gen_step + 1, hidden_dim), dtype=slice_h.dtype)
                    fresh[:, gen_step:gen_step + 1, :] = slice_h
                    kwargs["trailing_text_hidden"] = fresh
                else:
                    trailing = trailing.clone()
                    trailing[:, gen_step:gen_step + 1, :] = slice_h
                    kwargs["trailing_text_hidden"] = trailing

        out = prev_prep(self_talker, input_ids, *args, **kwargs)
        step_counter["n"] += 1
        return out

    Cls.prepare_inputs_for_generation = patched

    def teardown():
        Cls.prepare_inputs_for_generation = prev_prep
        step_counter["n"] = 0
    return teardown


# ---------------------------------------------------------------------------
# Main run_one / main
# ---------------------------------------------------------------------------

def run_one(adapter, processor, hf_model, shim, ucp, conv, idx, out_wav_dir,
            max_thinker_tokens, max_talker_tokens, speaker="ethan"):
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

    # Streaming state & callback
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

    # Pipeline state — the thinker thread will fill it
    prompt_len = inputs.input_ids.shape[1]
    state = PipelineState(assistant_start_idx=prompt_len)
    sti = StreamingTalkerInputs(hf_model, inputs, state, speaker=speaker)

    # Thinker kwargs — run on bg thread
    gc_cfg = GenerationConfig(do_sample=False, eos_token_id=[151645], pad_token_id=151645)
    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gc_cfg,
        max_new_tokens=max_thinker_tokens,
    )
    if getattr(inputs, "input_features", None) is not None:
        gen_kwargs["input_features"] = inputs.input_features.to(torch.bfloat16)
    if getattr(inputs, "feature_attention_mask", None) is not None:
        gen_kwargs["feature_attention_mask"] = inputs.feature_attention_mask

    t_thinker_start = time.perf_counter()
    thinker_thread = threading.Thread(
        target=run_thinker, args=(adapter, gen_kwargs, state, prompt_len), daemon=True,
    )
    thinker_thread.start()

    # Build talker prefill inputs (blocks until 4 tokens + prefill hidden ready)
    t_blk = time.perf_counter()
    talker_embed, talker_input_ids = sti.build_prefill()
    build_blocked_ms = (time.perf_counter() - t_blk) * 1000
    build_talker_s = time.perf_counter() - t_thinker_start

    # Install the pipelined prepare_inputs patch
    teardown = install_pipelined_prepare_inputs(hf_model, sti)

    try:
        # Talker config — match HF reference
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
            trailing_text_hidden=None,  # patched prepare_inputs fills slot-wise
            tts_pad_embed=sti.tts_pad_embed,
            talker_input_ids=talker_input_ids,
            max_new_tokens=max_talker_tokens,
            do_sample=True, top_k=50, top_p=0.8, temperature=0.9,
            repetition_penalty=1.1, suppress_tokens=suppress_tokens,
            eos_token_id=talker_cfg.codec_eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        talker_s = time.perf_counter() - t0

        # Emit residual codec tokens
        STR.finalize_stream(stream_state, hf_model.code2wav, on_audio)
    finally:
        teardown()
        thinker_thread.join(timeout=60.0)

    if state.thinker_error is not None:
        raise state.thinker_error

    full_wav = np.concatenate(wav_chunks) if wav_chunks else np.zeros(0, dtype=np.float32)
    out_wav_path = os.path.join(out_wav_dir, f"conv_{idx:03d}.wav")
    sf.write(out_wav_path, full_wav, 24000)

    total_s = time.perf_counter() - request_start
    ttfb_ms = chunk_timing[0]["t_ms"] if chunk_timing else None
    wav_s = float(len(full_wav) / 24000)
    rtf = total_s / wav_s if wav_s > 0 else None

    out_ids = getattr(state, "out_ids", None)
    asst_text = ""
    thinker_tokens = len(state.token_ids)
    if out_ids is not None:
        asst_text = processor.batch_decode(
            out_ids[:, prompt_len:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        thinker_tokens = int(out_ids.shape[1] - prompt_len)

    return {
        "idx": idx,
        "wav_path": wav_path,
        "input_audio_s": input_audio_s,
        "prompt_tokens": int(prompt_len),
        "thinker_tokens": thinker_tokens,
        "build_talker_blocked_ms": build_blocked_ms,
        "build_talker_total_s": build_talker_s,
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
    total_ms = [r["total_s"] * 1000 for r in ok]
    blocked_ms = [r["build_talker_blocked_ms"] for r in ok]
    in_audio = [r["input_audio_s"] for r in ok]
    out_wav = [r["wav_s"] for r in ok]
    th_toks = [r["thinker_tokens"] for r in ok]

    def row(name, xs, fmt="{:6.0f}"):
        s = statistics.mean(xs)
        p50 = percentile(xs, 50)
        p90 = percentile(xs, 90)
        p95 = percentile(xs, 95)
        print(f"  {name:20s}  mean={fmt.format(s)}  p50={fmt.format(p50)}  "
              f"p90={fmt.format(p90)}  p95={fmt.format(p95)}")

    row("TTFB ms", ttfbs)
    row("blocked_build ms", blocked_ms)
    row("total ms", total_ms)
    row("RTF", rtfs, fmt="{:6.2f}")
    row("input audio s", in_audio, fmt="{:6.2f}")
    row("output wav s", out_wav, fmt="{:6.2f}")
    row("thinker tokens", th_toks, fmt="{:6.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default="/tmp/qwen3_omni_pipelined.json")
    parser.add_argument("--wav-dir", default="/tmp/qwen3_omni_pipelined_wavs")
    parser.add_argument("--max-thinker", type=int, default=200)
    parser.add_argument("--max-talker", type=int, default=500)
    parser.add_argument("--speaker", default="ethan")
    parser.add_argument("--neuron-c2w", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.wav_dir, exist_ok=True)
    with open(CONV_JSON) as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations; running [{args.start}, {args.start + args.num})")
    print("Building Neuron pipeline...")
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
        try:
            r = run_one(
                adapter, processor, hf_model, shim, ucp,
                conversations[idx], idx, args.wav_dir,
                max_thinker_tokens=args.max_thinker,
                max_talker_tokens=args.max_talker,
                speaker=args.speaker,
            )
            results.append(r)
            ttfb_str = f"{r['ttfb_ms']:5.0f}ms" if r['ttfb_ms'] is not None else "     n/a"
            print(
                f"[{k+1:3d}/{args.num}] conv {idx:3d}  "
                f"in={r['input_audio_s']:4.1f}s  "
                f"prompt={r['prompt_tokens']:4d}  "
                f"blocked={r['build_talker_blocked_ms']:5.0f}ms  "
                f"new={r['thinker_tokens']:3d}tok  "
                f"TTFB={ttfb_str}  "
                f"total={r['total_s']*1000:5.0f}ms  "
                f"wav={r['wav_s']:4.1f}s  "
                f"RTF={r['rtf']:.2f}  "
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
    print(f"WAVs: {args.wav_dir}")


if __name__ == "__main__":
    main()
