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

### TTFB progression across configurations

| configuration | TTFB mean | TTFB p50 | TTFB p90 | TTFB p95 | hit-max / 100 |
|---|---:|---:|---:|---:|---:|
| 1. baseline streaming (broken talker) | 2727 | 2666 | 3113 | 3564 | **100** |
| 2. + TensorRegistry fix + norm capture + HF sampling | 2763 | 2698 | 3140 | 4128 | 15 |
| 3. + CHUNK_SIZE=25 / LEFT_CTX=5 | 2276 | 2193 | 2670 | 3581 | 14 |
| 4. + Neuron `code2wav` | 2000 | 1915 | 2389 | 3316 | 12 |
| 5. + thinker↔talker pipelining | **1759** | **1778** | **1811** | **1822** | **12** |

All milliseconds. "hit-max" counts samples where the talker reached
`max_new_tokens=500` instead of naturally emitting `codec_eos_token_id` —
smaller is better.

The **biggest tail-latency win** from step 5: p95 dropped from 3316 → 1822 ms
(−45 %). With pipelining, TTFB no longer scales with thinker output length —
the user gets first audio in a near-constant window regardless of whether the
thinker reply is 50 or 200 tokens.

### TTFB breakdown (step 4 — fully serial pipeline)

| stage | mean | p50 | p90 | note |
|---|---:|---:|---:|---|
| thinker full generate (Neuron) | 1346 | 1263 | 1485 | prefill 668 + ~68 × 10 ms decode |
| build talker inputs + 25 talker steps + UCP | 532 | 543 | 553 | 25 × ~21 ms decode |
| first `code2wav` chunk | **122** | **122** | **122** | Neuron NEFF, T=30 bucket |
| **TTFB total** | **2000** | **1915** | **2389** | |

### TTFB breakdown (step 5 — pipelined)

In the pipelined run thinker and talker overlap, so the breakdown is no
longer a sum of stages. The dominant component is "wait for first 4 thinker
tokens", measured as `build_talker_blocked_ms`:

| stage | mean | p50 | note |
|---|---:|---:|---|
| wait for first 4 thinker tokens | 765 | 762 | thinker prefill (~668 ms) + a few decode steps + bg-thread overhead |
| talker prefill + 25 decode steps + first `code2wav` chunk | ~990 | ~1015 | running concurrently with the rest of thinker decode; sometimes blocks on get_trailing_slice waiting for the next thinker token |
| **TTFB total** | **1759** | **1778** | |

#### Why "thinker" is 1346 ms, not the 668 ms TTFT number

The `test_thinker_ttft_bench.py` table reports **TTFT = 668 ms** (time to first
token = prefill) and **ITL = 10 ms** (per decode step). In the streaming TTFB
pipeline, though, the talker cannot start until the thinker has generated the
**entire** assistant reply — HF's `_build_talker_inputs` needs the full token
sequence and the full layer-23 hidden tensor to assemble the talker's prompt.

So the "thinker" row in the TTFB breakdown covers **prefill + all decode
steps** of the thinker, not just TTFT. With mean 68 new tokens:

```
thinker_ms ≈ prefill + new_tokens × ITL
           ≈ 668 + 68 × 10
           ≈ 1348 ms   (measured: 1346 ms)
```

Concretely, this is the serial pipeline the bench runs today:

```
t=0          request arrives
t=668        thinker prefill done   (first thinker token available — TTFT)
t=1346       thinker done           (68 decode steps @ 10 ms, all tokens + hiddens ready)
t=~1400      talker prefill done    (~50 ms build + 1 prefill forward on talker)
t=1878       25 talker decode steps done (25 × ~21 ms, each pairs with one UCP call)
t=2000       first code2wav chunk returned → first audio delivered
```

The 1346 ms thinker cost is what dominates TTFB now, and it's mostly **not**
prefill (bucket 2048) — it's the 68 serial decode steps after prefill.

#### What the ceiling looks like if we pipeline

If we were willing to change architecture and let the talker consume thinker
tokens as they stream out (instead of waiting for the full thinker sequence),
TTFB could in principle drop to roughly the thinker-prefill time + a short
warmup before the talker finds enough context to emit codec tokens:

```
t=0          request arrives
t=~668       thinker prefill done, thinker begins streaming tokens
t=~668+K*10  K additional thinker tokens buffered so talker has enough context
             to start (K is small — tens of tokens)
             (in parallel: talker prefill + first few decode steps)
t=TTFB       first 25 codec tokens produced, first c2w chunk returned
```

Ballpark with K ≈ 30: `668 + 30 × 10 + talker_prefill + 25 × 21 + 122 ≈
1400 ms` — a ~600 ms reduction from the current 2000 ms. This requires:

1. Making `_build_talker_inputs` incremental so it can extend the talker
   context one thinker token at a time (today it's a single batched
   assemble).
2. Running thinker decode and talker decode concurrently — either two Python
   threads with separate Neuron queues, or a host-side coroutine that
   alternates `thinker_step()` / `talker_step()` calls.
3. Deciding when K is large enough to start the talker (static threshold
   sufficient; adaptive would be nicer).

The thinker and talker run on disjoint NEFFs, so there's no device-level
conflict; the work is "just" in the orchestration.

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

### 6. Thinker ↔ talker pipelining

With everything on Neuron, the talker still waited for the **complete**
thinker output before starting. That's because HF's `_build_talker_inputs`
takes the full token sequence + full layer-23 hidden, then `talker.generate`
is called once with a pre-built `trailing_text_hidden` tensor.

But HF's talker `prepare_inputs_for_generation` only reads
`trailing_text_hidden[:, generation_step]` at decode step `k`, which
corresponds to the `(k+4)`-th thinker assistant token. So the talker can in
principle start as soon as **4 thinker tokens** are available, and consume
the rest one-by-one as they stream in.

**Implementation (`test_ttfb_pipelined_bench.py`).**

1. **Background thread runs the thinker.** A custom
   `StoppingCriteria` is installed (NxDI's `_sample` ignores HF's
   `streamer` arg, but it does call `stopping_criteria(input_ids, ...)` on
   every decode step — perfect tap point) that pushes each newly-appended
   token into a `PipelineState` condition-variable buffer. The
   `tensor_capture_hook` for layer-23 captures the hidden tensor in the
   same callback path.

2. **Main thread builds the talker prefill incrementally.**
   `StreamingTalkerInputs.build_prefill()` blocks until (a) the prefill's
   layer-23 hidden is captured (one Neuron forward), and (b) the first 4
   assistant tokens are in the buffer. Then it assembles only the prefill
   slice that HF's `_get_talker_assistant_parts` would build — using
   `assistant_hidden[:, :4]` plus the codec specials.

3. **Talker decode reads the trailing buffer on demand.**
   `Qwen3OmniMoeTalkerForConditionalGeneration.prepare_inputs_for_generation`
   is wrapped (layered on top of the existing streaming-c2w wrapper) so
   that, for each decode step `k`, it overwrites `kwargs["trailing_text_hidden"]`
   with a tensor whose row `k` is `text_projection(embed(thinker_tokens[k+4]))`,
   pulled from the streaming buffer. If the (k+4)-th token isn't out yet,
   the call blocks on the condition variable until it arrives. Past the end
   of the thinker output, the slice falls back to `tts_eos_embed`.

**Result.** TTFB: 2000 → **1759 ms** (mean), and crucially p95: 3316 →
**1822 ms** (−45 %). The big tail-latency win is because TTFB no longer
scales with thinker output length — the user gets first audio within
~1800 ms whether the assistant reply is 50 or 200 tokens long.

The mean improvement is more modest (~12 %) than the naive "subtract all
thinker decode" estimate (~600 ms) for two reasons:

- **Neuron device queue serializes.** Both thinker and talker NEFFs are
  compiled at TP=8 and run on the same 8 cores. They interleave on the
  Neuron driver instead of running truly in parallel. The talker can start
  earlier, but its forwards still queue behind in-flight thinker forwards.
- **Per-step CPU overhead.** Each talker decode step now does an extra
  `text_projection` on a single token (~3 ms) plus condition-variable
  signal/wait (~1 ms). Across 25 steps that's ~100 ms of extra serial work.

A real ~600 ms win would require running the thinker and talker on
**different** Neuron core groups (e.g. cores 0-7 and 8-15 on a trn2
instance) so that they can dispatch in parallel. That's a separate
compile-and-deploy change.

Enable with the new bench script:

```bash
python test_ttfb_pipelined_bench.py --num 100 --neuron-c2w
```

---

## New files

| Path | Purpose |
|---|---|
| `test_thinker_ttft_bench.py` | Thinker-only TTFT / ITL / throughput on 100 convs |
| `test_ttfb_rtf_bench.py` | Full streaming TTFB / RTF on 100 convs (serial). `--neuron-c2w` for Neuron-backed code2wav. |
| `test_ttfb_pipelined_bench.py` | Full streaming TTFB / RTF on 100 convs with thinker↔talker pipelining (background thread + on-demand `trailing_text_hidden`). |
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

## Remaining TTFB cost (after step 5 — pipelined)

At the pipelined config (TTFB mean 1759 ms / p50 1778 ms):

- **wait for first 4 thinker tokens** — ~765 ms
  - thinker prefill 668 + 4 × ITL (40) + bg-thread overhead (~60)
  - this is the floor the talker can't start before
- **talker prefill + 25 decode steps (overlapped with thinker decode)** — ~870 ms
  - 25 × ~21 ms talker, plus startup, plus a few cv-blocks waiting for the next thinker token
- **first code2wav chunk** — 122 ms
- **TTFB total** — 1759 ms

Everything is on Neuron and overlapped where possible.

### Options to go below 1759 ms

1. **Run thinker and talker on disjoint Neuron core groups** (~250-500 ms
   headroom). Today both NEFFs are TP=8 and share cores 0-7, so the Neuron
   driver serializes their forwards. On a trn2 instance with 16 cores, we
   could compile a second copy of the talker on cores 8-15 and dispatch in
   true parallel. The talker's 25 × 21 ms then overlaps the thinker decode
   end-to-end; TTFB would drop toward `max(thinker_full, prefill +
   4·ITL + 25·talker_ITL + c2w)` ≈ 1300-1400 ms.

2. **Shorter thinker replies (task-dependent).** The 68-token mean is set by
   the dataset's average assistant reply length. Prompting the thinker to be
   more concise (or truncating via a stop sequence) cuts decode time
   linearly. With pipelining, this matters less than before — the talker
   hides most of the decode — but on samples where the talker happens to
   catch up to the thinker's output, fewer thinker tokens still helps.

3. **Thinker speculative decoding.** NxDI supports EAGLE-style speculation.
   A draft model that proposes 2-3 thinker tokens per target step pushes the
   effective ITL toward 4-5 ms, narrowing both the "wait for first 4 tokens"
   window and the per-step talker stall window.

4. **Thinker prefill bucketing.** All 100 prompts land in bucket 2048
   because the system prompt is ~800 tokens. Splitting the system prompt
   into a separately-cached prefix and running bucket-512 on the delta could
   shave the 668 ms prefill to ~250 ms — directly reducing the
   "wait for first 4 tokens" floor.

5. **Talker + UCP fusion.** The 25 × ~21 ms today is one talker forward + one
   UCP forward, both via separate NEFFs. Merging them into a single traced
   op per step would save ~3 ms/step on cross-NEFF dispatch, ~75 ms total
   at CHUNK_SIZE=25.

6. **Smaller CHUNK_SIZE.** 25 is already aggressive. Going to 15 would save
   ~200 ms on the talker portion but increases left-context recompute
   overhead in code2wav. Worth measuring if a lower floor is needed.

7. **Pure-C++ orchestration / GIL elimination.** ~100 ms of the pipelined
   TTFB is Python overhead from the bg-thread / cv-block / per-step
   `text_projection` glue. A C++ inference server that drives both Neuron
   models via the runtime API directly (no Python in the hot loop) would
   shed that.
