# Qwen3-Omni TTFB / RTF benchmark on omni2 audio-in conversations

End-to-end benchmark on 100 real multi-turn conversations with audio user
inputs (source: `/home/ubuntu/omni2`). Each conversation has a system prompt,
2–4 prior text turns, and a final user turn that is a `.wav` audio file. The
pipeline must produce a spoken assistant reply.

This doc tracks the progressive optimizations that moved **TTFB from 2727 ms
→ 2000 ms** (−27 %) and the talker success rate from 0 % → 88 % (no max‐token
truncation).

## Setup

- Trn2.48xlarge, 8 Neuron cores pinned via `NEURON_RT_VISIBLE_CORES=0-7`
- TP=8 for every submodel
- Dataset: `/home/ubuntu/omni2/merged_conversations_with_audio_x10_with_system.json`
  (system prompt ~800 tokens, prompt lengths 1164–1494 tokens; all land in
  the 2048 bucket)
- 100 conversations; audio files in `/home/ubuntu/omni2/speech_wav_16k/`

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Thinker-only benchmark
NEURON_RT_VISIBLE_CORES=0-7 python test_thinker_ttft_bench.py --num 100

# Full streaming TTFB / RTF benchmark (best configuration)
NEURON_RT_VISIBLE_CORES=0-7 CHUNK_SIZE=25 LEFT_CTX=5 \
    python test_ttfb_rtf_bench.py --num 100 \
        --max-thinker 200 --max-talker 500 \
        --neuron-c2w
```

## Thinker-only TTFT & throughput (`test_thinker_ttft_bench.py`)

The `tensor_capture_hook` fires once per thinker forward (prefill + each
decode step), so we use it as a per-token timing tap. TTFT = time from
`adapter.generate()` start to the first hook fire (= end of prefill).

| metric | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| TTFT (prefill) | **668 ms** | 667 | 672 | 678 |
| decode ITL | **10.2 ms** | 10.1 | 10.3 | 10.3 |
| tokens/s (overall) | **48.3** | 47.1 | 54.9 | 69.1 |
| RTF vs audio input | **0.37** | 0.33 | 0.65 | 0.79 |
| prompt tokens | 1294 | 1278 | 1400 | 1419 |

All 100 samples succeeded. 100 conversations ran in 136 s wall time.

## Full streaming TTFB / RTF (`test_ttfb_rtf_bench.py`)

Streaming pipeline: thinker (Neuron) → talker (Neuron) → UCP (Neuron) →
code2wav. `code2wav` fires inline every `CHUNK_SIZE` codec tokens. TTFB =
request arrival → first audio chunk delivered to the host.

### TTFB progression across five configurations

| configuration | TTFB mean | TTFB p50 | TTFB p90 | TTFB p95 | hit-max / 100 |
|---|---:|---:|---:|---:|---:|
| 1. baseline streaming (broken talker) | 2727 | 2666 | 3113 | 3564 | **100** |
| 2. + TensorRegistry fix + norm capture + HF sampling | 2763 | 2698 | 3140 | 4128 | 15 |
| 3. + CHUNK_SIZE=25 / LEFT_CTX=5 | 2276 | 2193 | 2670 | 3581 | 14 |
| 4. + Neuron `code2wav` | **2000** | **1915** | **2389** | **3316** | **12** |

All milliseconds. "hit-max" counts samples where the talker reached
`max_new_tokens=500` instead of naturally emitting `codec_eos_token_id` —
smaller is better.

### TTFB breakdown (best configuration — step 4)

| stage | mean | p50 | p90 | note |
|---|---:|---:|---:|---|
| thinker (Neuron) | 1346 | 1263 | 1485 | bucket-2048 MoE prefill |
| build talker inputs + 25 talker steps + UCP | 532 | 543 | 553 | 25 × ~21 ms decode |
| first `code2wav` chunk | **122** | **122** | **122** | Neuron NEFF, T=30 bucket |
| **TTFB total** | **2000** | **1915** | **2389** | |

### Full-run stats at best configuration

| metric | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| TTFB | 2000 ms | 1915 | 2389 | 3316 |
| thinker | 1345 ms | 1262 | 1485 | 2656 |
| total (end-to-end) | 5648 ms | 4804 | 11428 | 11701 |
| RTF (total / wav) | 0.61 | 0.39 | 0.89 | 1.19 |
| input audio | 5.18 s | 4.27 | 9.26 | 9.86 |
| output wav | 16.06 s | 13.08 | 39.50 | 39.50 |
| thinker tokens | 68 | 59 | 81 | 151 |

100/100 succeeded. 88/100 talker runs ended at `codec_eos`; 12 hit
`max_new_tokens=500`.

---

## Fixes made

### 1. `TensorRegistry.clear()` wiped `modules_to_capture` across buckets

**Symptom.** With `tensor_capture_config={"layers.23"}` configured, capture
worked perfectly at the first bucket (256) and returned the empty fallback
`torch.zeros(1, dtype=bfloat16)` at every larger bucket (512 / 1024 / 2048 /
4096). All our omni2 prompts land in the 2048 bucket, so capture produced
`(1,)` and `_assemble_hidden` crashed on `captured[0][:, :prompt_len, :]`.

**Root cause.** `NeuronBaseModel._get_captured_tensors` is called once per HLO
trace (once per bucket) and ends with `registry.clear()`. Upstream
`TensorRegistry.clear` in
`neuronx_distributed/utils/tensor_capture/registry.py` replaces `model_info`
with a fresh `CapturedModelInfo([], 10, False)`, **erasing the configured
`modules_to_capture`**. Forward hooks installed by `enable_tensor_capture` keep
firing, but `register_tensor` no longer finds the module name in
`modules_to_capture` and falls through to the "manual" branch. Only the first
bucket gets a real capture; every subsequent bucket's NEFF bakes in a zero
fallback.

**Fix** (in `src/_upstream_compat.py::_patch_tensor_registry_clear`).
Monkey-patch `configure()` to stash the last non-empty module list and
`clear()` to restore it instead of wiping. Five lines of glue, applied at
import time.

After the fix, verified all five buckets emit real captures with the expected
shape `(1, bucket_size, 2048)`.

### 2. Talker shim fabricated hidden, blocking `codec_eos`

**Symptom.** In the baseline streaming run (v1 above), **100/100 samples hit
`max_new_tokens`**. The talker never emitted `codec_eos_token_id = 2150`. We
confirmed via argmax logging that the decoder locked into repetitive loops
like `[318, 318, 318, ...]`.

**Root cause.** HF's talker generate loop reads the per-step hidden from
`output.hidden_states[-1]` and feeds it to `code_predictor` as `past_hidden`.
Our `NeuronTalkerShim` (`test_audio_out_full_neuron.py`) returned a
**fabricated** hidden built by re-embedding the argmax'd codec token:

```python
tok = logits_last.argmax(dim=-1)
fake_hidden = hf_model.talker.get_input_embeddings()(tok).to(torch.bfloat16)
return BaseModelOutputWithPast(hidden_states=(fake_hidden,), ...)
```

That stand-in drifted far enough from the talker's real pre-lm_head hidden
that greedy decoding couldn't reach `codec_eos` at all.

**Fix.** Recompile the talker with `TensorCaptureConfig(modules_to_capture=["norm"])`
so the NEFF emits the real post-RMSNorm hidden `[B, S, 1024]` as an extra
output. New compile script: `compile_talker.py`. New artifact:
`/tmp/qwen3_omni_compiled/talker_tp8_capnorm/`. Compile time: ~11 min.

Then update `make_neuron_talker_shim` in `test_audio_out_full_neuron.py` to
parse `out[2]` from the NEFF output (logits, gathered_logits, **captured
norm**) and pass it as `hidden_states=(real_hidden,)` instead of `fake_hidden`.

### 3. Talker `generate()` call missed HF's reference settings

After fix 2, talker could reach `codec_eos` in principle, but 85 % of runs
still hit max with greedy decoding because the argmax trajectory occasionally
locks into loops (we saw `[318, 318, 318, ...]` looping at the end).

HF's reference `Qwen3OmniMoeForConditionalGeneration.generate` uses:

```python
suppress_tokens = [i for i in range(vocab - 1024, vocab) if i != codec_eos]
talker.generate(do_sample=True, top_k=50, top_p=0.8, temperature=0.9,
                repetition_penalty=1.1, suppress_tokens=suppress_tokens, ...)
```

`suppress_tokens` masks out the 1 024 non-codec ids (text-token range left
over in the talker's shared vocab). Sampling + repetition penalty breaks the
`[318, 318, ...]` loops.

**Fix.** `test_ttfb_rtf_bench.py` now passes the full HF-matching talker
config. Hit-max dropped from 100/100 → 15/100.

### 4. `CHUNK_SIZE=25` (from 50)

Streaming fires code2wav after every `CHUNK_SIZE` codec tokens. The baseline
CHUNK_SIZE=50 meant the user waited for 50 talker steps (~950 ms) plus one
big c2w call (~540 ms on CPU) before hearing the first audio. Halving to 25
cuts both.

**Fix.** `CHUNK_SIZE` and `LEFT_CTX` are now env-var-controlled in
`test_audio_streaming.py`. TTFB dropped 487 ms (from 2763 → 2276 ms).

```bash
CHUNK_SIZE=25 LEFT_CTX=5 python test_ttfb_rtf_bench.py ...
```

Trade-off: the small `LEFT_CTX` re-compute at each chunk boundary adds a few
hundred ms to the total wall time. Net win for TTFB, net neutral for total.

### 5. Code2Wav on Neuron

With the previous fixes, TTFB breakdown was thinker 1345 ms + talker 540 ms
+ **first_c2w 387 ms on CPU**. That last CPU step was the largest
remaining non-Neuron cost in the critical path.

**Compile.** `compile_code2wav.py` traces `Qwen3OmniMoeCode2Wav.forward`
(8-layer sliding-window transformer → upsample conv chain → BigVGAN decoder)
with `torch_neuronx.trace` at fixed input lengths. Bucket set `{30, 50, 128}`
covers the streaming chunk (CHUNK_SIZE=25 + LEFT_CTX=5 = 30) and the residual
tail at finalize. Single-core, fp32 (`--auto-cast=none`), no TP. Compile time:
~2.5 min per bucket.

**Runtime shim.** `code2wav_neuron.py::NeuronCode2WavShim` replaces
`hf_model.code2wav`. At call time it picks the smallest bucket ≥ T, zero-pads
the codec-token tensor up to the bucket, runs the Neuron NEFF, and trims the
output back to `T * total_upsample` samples. `chunked_decode` is forwarded
through the same shim for symmetry.

**Verified** bit-exact against CPU: `max_abs_diff = 0.00000`,
`cosine_similarity = 1.0000`.

**Result.** First-chunk c2w: **387 ms → 122 ms** (3.2× faster). TTFB: 2276 →
**2000 ms**.

Enable with the new flag on the bench:

```bash
python test_ttfb_rtf_bench.py --num 100 --neuron-c2w
```

---

## New files

| Path | Purpose |
|---|---|
| `test_thinker_ttft_bench.py` | Thinker-only TTFT / ITL / throughput on 100 convs |
| `test_ttfb_rtf_bench.py` | Full streaming TTFB / RTF on 100 convs. `--neuron-c2w` for Neuron-backed code2wav. |
| `compile_talker.py` | Compile talker with `TensorCaptureConfig(["norm"])` |
| `compile_code2wav.py` | Compile code2wav at fixed T buckets |
| `code2wav_neuron.py` | Runtime shim that routes code2wav through the compiled NEFFs |
| `src/_upstream_compat.py` | Added `_patch_tensor_registry_clear` |

Compiled artifacts:

| Path | Contents |
|---|---|
| `/tmp/qwen3_omni_compiled/talker_tp8_capnorm/` | Talker with norm capture (replaces `talker_tp8/` for the audio pipeline) |
| `/tmp/qwen3_omni_compiled/code2wav_buckets/model_T{30,50,128}.pt` | Per-bucket code2wav NEFFs |

## Modified files

| Path | Change |
|---|---|
| `src/_upstream_compat.py` | Patch `TensorRegistry.configure` / `clear` to preserve `modules_to_capture` across bucket traces |
| `test_audio_out_full_neuron.py` | Point `TALKER_COMPILED` at `talker_tp8_capnorm`; shim now reads `out[2]` (captured norm) as the real hidden |
| `test_audio_streaming.py` | `CHUNK_SIZE` / `LEFT_CTX` read from env vars |

## Remaining TTFB cost

Breakdown at TTFB = 2000 ms:

- thinker prefill (Neuron) — 1346 ms (67 %)
- talker + UCP 25 steps (Neuron) — 532 ms (27 %)
- first code2wav chunk (Neuron) — 122 ms (6 %)

Everything in the critical path runs on Neuron. Further wins would require
compressing the bucket-2048 thinker prefill (unlikely without architectural
changes) or speculating / batching talker decode steps.
