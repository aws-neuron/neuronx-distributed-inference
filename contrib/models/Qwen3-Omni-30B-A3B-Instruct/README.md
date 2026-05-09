# Qwen3-Omni-30B-A3B-Instruct on AWS Neuron

End-to-end inference of Qwen3-Omni-30B-A3B-Instruct on Trainium/Inferentia2
via `neuronx_distributed_inference` (NxDI). Covers both ASR (speech→text) and
speech output (text→speech) pipelines.

All five neural network modules run on Neuron with TP=8:

| Module | Parameters | Role |
|---|---|---|
| Thinker MoE text decoder | 48 layers, 128 experts | generates text tokens from multimodal input |
| Vision encoder (Qwen3-VL ViT) | 27 layers | image → token embeddings |
| Audio encoder | 32-layer transformer | mel → audio token embeddings |
| Talker MoE | 20 layers, 128 experts | text + hidden → codec tokens |
| Unified Code Predictor | 5-layer dense GQA, 15-step unroll | expands each codec token to 15 residual codes |

Code2Wav (codec→waveform) stays on CPU (~1s) — small enough that Neuron
offload overhead would negate the win.

---

## End-to-end results

### Audio output (text prompt → wav)

Prompt: *"Please say hello and tell me about Neuron chips briefly."*
Output: 7.9 s @ 24 kHz, correct spoken answer.

| Stage | Location | Time |
|---|---|---|
| Thinker text generation (80 tokens) | Neuron | 1.3 s |
| Layer-24 hidden assemble (from Neuron capture) | Neuron | 0.0 s |
| Talker decode loop (100 codec steps) | Neuron | 0.4 s |
| Unified Code Predictor (99 × 15 steps) | Neuron | 1.1 s |
| Code2Wav | CPU | 0.9 s |
| **Total** | | **~3.8 s** |

Real-time factor **0.48x** (generates 2× faster than playback).

**Progression** vs pure-CPU HF baseline:
| Version | Total | Note |
|---|---|---|
| CPU HF baseline | 91 s | reference |
| Thinker on Neuron only | 383 s | HF re-runs thinker on CPU inside `model.generate` |
| + skip HF thinker re-run | 107 s | replay thinker on CPU once for hidden states |
| + Neuron talker | 62 s | CPU talker (15 s) → Neuron (0.5 s) |
| + thinker hidden capture | 62 s | 45 s CPU re-forward → 0 s (directly from Neuron) |
| **+ Unified Code Predictor** | **3.8 s** | 59 s CPU CP (99 × 15 calls) → 1.1 s Neuron (99 calls) |

### ASR (LibriSpeech test-clean, 100 samples)

| Metric | Value |
|---|---|
| Samples | 100 |
| Total audio | 670 s |
| Total wall time | 66 s |
| Avg WER | 18.5 % (mostly casing/punctuation) |
| RTF | 0.12x (~8× real-time) |
| Per-clip latency (≤10 s clips) | 0.5–0.7 s |

### Audio benchmarks (`test_audio_bench.py`)

Four benchmarks run against the full Neuron pipeline. All generated wav files
are saved to `/tmp/qwen3_omni_bench/`.

**1. Multi-length TTS** — scales gracefully; RTF actually improves as output
gets longer because the fixed thinker cost amortizes over more talker tokens.

| Tag | Thinker tokens | Codec tokens | Wav | Total | RTF |
|---|---:|---:|---:|---:|---:|
| short  (`"Say hi."`) | 10 | 80 | 6.3 s | 2.76 s | 0.44x |
| medium | 115 | 150 | 11.9 s | 4.91 s | 0.41x |
| long   | 128 | 250 | 19.9 s | 6.98 s | 0.35x |
| xlong  | 128 | 400 | 31.9 s | 10.24 s | 0.32x |

**2. Multi-speaker TTS** — all three speakers (`chelsie`, `ethan`, `aiden`)
produce audio of identical length and near-identical latency, confirming the
speaker-ID plumbing works correctly.

| Speaker | Wav | Total | RTF |
|---|---:|---:|---:|
| chelsie | 11.9 s | 3.42 s | 0.29x |
| ethan   | 11.9 s | 3.31 s | 0.28x |
| aiden   | 11.9 s | 3.30 s | 0.28x |

**3. Audio-in → audio-out** — LibriSpeech clip as input, model repeats it
back as spoken audio. Full multimodal path (audio encoder → thinker → talker
→ code2wav).

| Stage | Time |
|---|---:|
| Thinker (incl. audio encoder) | 0.7 s |
| Build talker input | negligible |
| Talker generate | 2.9 s |
| Code2Wav | 1.2 s |
| **Total** | **4.8 s** |

- Input: 3.5 s speech, reference "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
- Model heard → repeated as "Concorde returned to its place amidst the tents."
- Output wav: 15.9 s (model adds a bit of extra speech / phrasing)

**4. Long TTS (up to 512 codec tokens)** — stress the code predictor /
code2wav chain in sustained mode. Latency scales linearly with codec length;
per-step cost stable ~11 ms on the unified code predictor throughout.

| Budget | Codec | Wav | Total | RTF | UCP time | UCP calls | UCP/call |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 20.4 s | 6.92 s | 0.34x | 2.3 s | 255 | 9.0 ms |
| 400 | 400 | 31.9 s | 9.44 s | 0.30x | 3.7 s | 399 | 9.3 ms |
| 512 | 512 | 40.8 s | 11.68 s | 0.29x | 4.7 s | 511 | 9.2 ms |

No drift in per-step UCP latency between 256 and 512 tokens.

### TTFT / ITL (`test_ttft.py`)

Per-stage time-to-first-token and inter-token latency, measured with a
`LogitsProcessor` that records `perf_counter()` on every token. Prompt
lengths 22, 28, and 57 tokens; talker budgets 80, 150, 250 codec tokens.

| Stage | Metric | Value |
|---|---|---:|
| **Thinker** (48-layer MoE, TP=8) | TTFT (prefill) | 344–354 ms |
|  | ITL mean | 12.1–13.1 ms |
|  | ITL p50 / p95 | 12 / 22 ms |
| **Talker** (20-layer MoE, TP=8) | TTFT (prefill) | 24–30 ms |
|  | ITL mean | 13.9–16.6 ms |
|  | ITL p50 / p95 | 14 / 14 ms |
| **Code2Wav** (CPU, batch) | latency | 0.9–1.5 s per utterance |

- Thinker TTFT ≈ 350 ms is dominated by the MoE prefill over a ~30-token
  prompt (~12 ms/layer × 48 / 2 ≈ 290 ms of compute, plus bucket pad and
  input move overhead).
- Thinker ITL ≈ 12 ms/token → **~80 tokens/s** text generation.
- Talker ITL ≈ 14 ms/token. Per talker step the pipeline also fires one
  unified-CP NEFF call (~9 ms), so ~14 ms/step is consistent with
  `talker (~5 ms) + UCP (~9 ms)`.
- Talker TTFT (~25 ms) is much lower than thinker TTFT because talker
  prefill runs over a very short sequence (just the speaker token + special
  prefix) — it's essentially a 14-token prefill through a 20-layer MoE.

**End-to-end TTFB** (Time-To-First-Byte, prompt arrival → first wav sample
available on host):

| Prompt length | Thinker tokens | Codec tokens | Wav length | Full TTFB |
|---:|---:|---:|---:|---:|
| 22  | 10  | 80  | 6.3 s | **2.73 s** |
| 28  | 94  | 150 | 11.9 s | **4.66 s** |
| 57  | 150 | 250 | 19.9 s | **7.13 s** |

### Streaming code2wav (`test_audio_streaming.py`)

Instead of waiting for the entire talker output and then running one large
`code2wav.chunked_decode`, we fire `code2wav` inline on 50-codec-token
chunks (~4 s audio each) as soon as they accumulate. Chunks are emitted
sequentially within the same thread — the talker pauses ~550 ms every 50
codec tokens while the CPU `code2wav` decodes that chunk. This sacrifices a
little total wall time (per-chunk overhead adds up) in exchange for a
dramatically lower time-to-first-audio.

Patch point: `Qwen3OmniMoeTalkerForConditionalGeneration.prepare_inputs_for_generation`
is wrapped at the class level; it intercepts `residual_codes` right after
HF builds it, appends to a shared list, and fires a chunk whenever the
list grows by 50.

**First-audio latency comparison (batch vs streaming, same prompts)**:

| Scenario | Wav length | Batch TTFB | Streaming TTFB | Improvement |
|---|---:|---:|---:|---:|
| short  | 6.3 s | 2.73 s | **2.02 s** | −26 % |
| medium | 11.9 s | 4.66 s | **2.99 s** | −36 % |
| long   | 23.8 s | 7.13 s | **3.41 s** | **−52 %** |
| xlong  | 40.6 s | ~11.7 s (extrap.) | **4.00 s** | **−66 %** |

Per-chunk code2wav cost stays steady (~550 ms per 50-token chunk). Total
wall time is slightly higher than batch mode (e.g. long: 9.66 s vs 7.13 s)
because small-chunk overhead (left-context re-compute, per-call Python
dispatch) dominates; but because user-perceived latency is TTFB, not total,
streaming is still a clear win for interactive use.

Config knobs: `CHUNK_SIZE=50`, `LEFT_CTX=10` in `test_audio_streaming.py`.

### Conversational audio-in benchmark (omni2, 100 convs)

See [`BENCHMARK_OMNI2_TTFB.md`](BENCHMARK_OMNI2_TTFB.md) for a detailed TTFB
/ RTF benchmark on 100 real multi-turn audio-in conversations (prompts
1164–1494 tokens). Covers the progressive optimizations that took TTFB from
**2727 ms → 1759 ms** (−35 %, p95 from 3564 → 1822 ms / −49 %) and the
talker max-token truncation rate from 100 % → 12 %:

1. Patched `TensorRegistry.clear()` so `layers.23` capture survives across
   all bucket traces (prompts ≥ 512 tokens previously hit a zero fallback).
2. Recompiled the talker with `TensorCaptureConfig(["norm"])` and wired the
   shim to use the real post-RMSNorm hidden — greedy decoding now reaches
   `codec_eos` instead of looping on `[318, 318, …]`.
3. Switched talker `generate()` to HF's reference settings
   (`do_sample=True, top_k=50, top_p=0.8, temperature=0.9,
   repetition_penalty=1.1, suppress_tokens=<non-codec range>`).
4. `CHUNK_SIZE=25` / `LEFT_CTX=5` (was 50 / 10): TTFB −487 ms.
5. Ported `code2wav` to Neuron (bit-exact vs CPU): first-chunk c2w 387 ms →
   122 ms.
6. Pipelined thinker ↔ talker — talker starts as soon as 4 thinker tokens
   are buffered and reads `trailing_text_hidden[k]` on demand. Mean TTFB
   2000 → 1759 ms; p95 cut nearly in half (3316 → 1822 ms).

Best configuration: `NEURON_RT_VISIBLE_CORES=0-7 CHUNK_SIZE=25 LEFT_CTX=5
python test_ttfb_pipelined_bench.py --num 100 --neuron-c2w`.

---

## Repository layout

```
contrib/models/Qwen3-Omni-30B-A3B-Instruct/
├── README.md                                   (this file)
└── src/
    ├── modeling_qwen3_omni.py                  top-level thinker + vision config / weight conversion
    ├── modeling_qwen3_omni_text.py             thinker MoE text decoder (48 layers, reuses Qwen3-VL attention)
    ├── modeling_qwen3_omni_audio.py            audio encoder (32-layer transformer on Neuron, Conv2d frontend / post-proc on CPU)
    ├── modeling_qwen3_omni_talker.py           talker MoE body (20 layers) + TalkerInferenceConfig + HF→Neuron weight conversion
    ├── modeling_qwen3_omni_code_predictor.py   per-call (debug) and unified (production) code predictor
    ├── _upstream_compat.py                     runtime patches to NxDI's HuggingFaceGenerationAdapter and qwen3_vl vision loader
    └── _model_path.py                          model-path helper
```

### Test scripts (in `/home/ubuntu/`)

| File | Purpose |
|---|---|
| `test_asr_qwen3_omni.py` | ASR benchmark (LibriSpeech); builds and loads thinker + vision + audio encoder. Also exposes `build_and_load_model` reused by other tests. |
| `test_audio_out_cpu.py` | Pure-CPU HF reference for audio output. |
| `test_audio_out_neuron.py` | Phase 1 mixed: thinker on Neuron, talker+code2wav on CPU. |
| `test_audio_out_full_neuron.py` | Full Neuron pipeline (thinker + talker + unified CP + code2wav-CPU). |
| `test_audio_bench.py` | Four-benchmark audio suite (multi-length, multi-speaker, audio-in→audio-out, long TTS). Dumps JSON + wavs. |
| `test_ttft.py` | TTFT / ITL micro-benchmark — records per-token timestamps for thinker + talker and computes end-to-end TTFB. |
| `test_audio_streaming.py` | Streaming code2wav — emits 50-codec-token audio chunks inline for low TTFB. `CHUNK_SIZE` / `LEFT_CTX` overridable via env. |

### Benchmark scripts (in the contrib dir)

| File | Purpose |
|---|---|
| `test_thinker_ttft_bench.py` | Thinker-only TTFT / ITL / throughput on the omni2 100-conv dataset. |
| `test_ttfb_rtf_bench.py` | Full streaming TTFB / RTF on the omni2 100-conv dataset (serial thinker→talker). `--neuron-c2w` routes code2wav through Neuron. |
| `test_ttfb_pipelined_bench.py` | Same dataset, but thinker and talker overlap: thinker streams tokens to a bg thread, talker reads `trailing_text_hidden[k]` on demand. Lowest TTFB and tightest tail latency. |
| `compile_talker.py` | Recompile the talker with `TensorCaptureConfig(["norm"])` so the NEFF exposes the real post-RMSNorm hidden (needed by the `code_predictor` path). Output: `talker_tp8_capnorm/`. |
| `compile_code2wav.py` | Compile the vocoder at fixed input buckets (default `{30, 50, 128}`). Output: `code2wav_buckets/`. |
| `code2wav_neuron.py` | Runtime shim that replaces `hf_model.code2wav` with a bucket-dispatching Neuron wrapper. |

---

## Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

Place HF model at `/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct`.

Expected compiled artifacts in `/tmp/qwen3_omni_compiled/`:

| Directory | Compiles | Size |
|---|---|---|
| `multimodal_tp8_cap23/` | Thinker (48L MoE) + vision, with layer-24 hidden capture | ~3 GB |
| `audio_encoder_tp8/` | 32-layer audio transformer | ~200 MB |
| `talker_tp8/` | Talker (20L MoE), no capture | ~1.2 GB |
| `talker_tp8_capnorm/` | Talker (20L MoE) with `norm` capture (needed by the audio-output pipeline; see `BENCHMARK_OMNI2_TTFB.md`) | ~1.2 GB |
| `code_predictor_unified_tp8/` | Unified CP (5L dense, 15-step unrolled) | ~150 MB |
| `code2wav_buckets/` | Optional Neuron code2wav, one NEFF per bucket size | ~500 MB total |

---

## How to run

### Audio output (text → speech)

```bash
cd /home/ubuntu
NEURON_RT_VISIBLE_CORES=0-7 python test_audio_out_full_neuron.py \
    --prompt "Please say hello and tell me about Neuron chips briefly." \
    --out /tmp/out.wav
```

First run compiles all components (~45 min total). Subsequent runs reuse
the compiled artifacts and take ~60 s to load + 4 s to infer.

### ASR

```bash
NEURON_RT_VISIBLE_CORES=0-7 python test_asr_qwen3_omni.py --num-samples 100
```

Opt-in flag `QWEN3_OMNI_CAPTURE_LAYER_HIDDEN=23` enables hidden-state capture
during thinker inference (required for the audio-output pipeline; compiled
separately from the ASR-only artifact to avoid the extra trace output).

---

## Key design decisions

### 1. Thinker weight sharing

`NeuronQwen3OmniForCausalLM.checkpoint_loader_fn` loads HF safetensors once,
partitions by owning model (text vs vision), and returns a fresh shallow-copy
dict per builder call. Tensors are shared between text and vision
partitions. Without this, sharding text followed by vision peaked CPU RAM at
200 GB (and the second builder encountered missing vision keys because NxDI's
`preprocess_checkpoint` destructively deletes keys not owned by the current
model). With the fix, peak RAM stays around 20 GB.

### 2. MRoPE + 3D position_ids

The thinker uses interleaved MRoPE with mrope_section `[24, 20, 20]`.
`rotary_position_ids` with shape `[3, B, S]` is passed as input-generator
arg 21 (see `NeuronQwen3OmniTextModelWrapper._ROTARY_POSITION_IDS_INDEX`).
The upstream `pad_inputs` is patched (`test_asr_qwen3_omni.py:_patched_pad_inputs`)
to preserve this and the trailing `deepstack_vision_embeds` slot.

### 3. Audio encoder — 20 heads don't divide TP=8

Padded num_heads 20 → 24 (next multiple of 8) and zero-fill the Q/K/V/out_proj
weight rows for the added heads (`NeuronAudioAttention.__init__` and
`convert_hf_to_neuron_state_dict`). Zero-padded heads produce zero output;
the zero column in `out_proj` ensures they don't contaminate the residual
stream.

### 4. Audio encoder attention window

`n_window=50` gives tiny 13-token attention blocks in the basic `cu_seqlens`
path, which corrupts long audio (≥15 s). Fixed by using
`_compute_inference_cu_seqlens` with `n_window_infer=800`, matching HF.

### 5. Scatter bug at bucket boundaries

HF's scatter path uses `fill_value = pad_limit - 1` which means when
`input_ids.shape[1] == bucket_size` exactly, the fill positions land on the
last *real* prompt token. Audio embeddings' padding zeros then clobber that
token (observed as "55." garbage on 16.8 s audio). Fixed by appending a pad
token when prompt length is a bucket boundary
(`modeling_qwen3_omni.py:forward`).

### 6. Talker: shared_expert differs from routed experts

Qwen3-Omni talker has `moe_intermediate_size=384` (routed experts) but
`shared_expert_intermediate_size=768` and a sigmoid-gated shared path. NxDI's
`initialize_moe_module` ties shared-expert size to `config.intermediate_size`,
so we build the shared expert as a separate `SharedExpertSwiGLU` module with
its own intermediate size and apply it alongside the routed MoE inside
`NeuronTalkerDecoderLayer`.

### 7. Talker: 2 KV heads don't divide TP=8

Replicated the 2 KV heads into 8 (one per rank) during weight conversion
(`convert_talker_hf_to_neuron`, the `kv_pad` logic). Each replicated head
computes the same attention, so this is bit-exact up to bf16 noise.

### 8. Talker ↔ HF generate integration

The HF talker pipeline computes, for every decode step, a sum of
`last_id_hidden + code_predictor mid hiddens + trailing text hidden` on CPU,
then feeds the resulting `inputs_embeds` to `talker.model.forward`. We
install a shim (`NeuronTalkerShim`) that replaces `talker.model` and
routes the already-summed `inputs_embeds` through the Neuron NEFF via the
`vision_embeddings` input slot — the same pattern Qwen2.5-Omni uses.
`codec_head` is swapped to `nn.Identity` since the Neuron NEFF already
applies its internal `lm_head`.

### 9. Thinker layer-24 hidden capture

HF's talker inputs need per-token hidden states at `accept_hidden_layer=24`
(the 24th post-layer hidden, i.e., output of the 0-indexed layer 23). These
are extracted from the Neuron thinker for free via
`TensorCaptureConfig(modules_to_capture=["layers.23"])` plus
`output_logits=True`, instead of replaying the 30 B model on CPU (~45 s).
The capture hook is passed to `adapter.generate` as `tensor_capture_hook`.

### 10. Unified Code Predictor

HF's code predictor generates 15 residual codes per talker decode step with
15 sequential forward passes. Per-call Neuron overhead (~10 ms) ×
(15 calls × 99 talker steps) would be ≈15 s — **slower** than HF's CPU
baseline (~60 s over same workload). Instead, the entire 15-step
argmax-loop is unrolled into a single NEFF
(`UnifiedNeuronCodePredictor.forward`) that completes in ~11 ms per talker
step, for **54× speedup** over HF CPU.

The unrolled trace uses a fixed 16-position buffer (2 prefill + 14 decode)
and re-runs full attention each round (no KV cache). The 15 codec embedding
tables and 15 LM heads are stacked into single tensors and indexed inside
the trace.

---

## Known limitations

- **bf16 numerical drift** — occasionally one out of 15 residual codes
  diverges by one unit (e.g., step 13 may pick code 293 vs golden 1025 when
  the top-2 logits are separated by <0.002). Audio quality is unaffected.
- **Code2Wav on Neuron (opt-in)** — `compile_code2wav.py` traces the
  vocoder at a handful of fixed input lengths; `code2wav_neuron.py`
  dispatches by chunk size at runtime. Bit-exact vs CPU and ~3× faster on
  the per-chunk streaming call. Enable via `--neuron-c2w` in
  `test_ttfb_rtf_bench.py`. Compile `/tmp/qwen3_omni_compiled/code2wav_buckets/`
  once with the expected chunk sizes (defaults cover `CHUNK_SIZE=25`).
- **Talker compilation time** — ~10 min for the 20-layer MoE.
- **CPU HF model required at inference time** — for the `_get_talker_user_parts`
  / `_get_talker_assistant_parts` helpers and `text_projection`/`hidden_projection`
  projections. ~60 GB CPU RAM. A future optimization would lift those
  projections onto Neuron too.
