# Voxtral-Mini-3B on NeuronX Distributed Inference (Trn2 / Inf2)

`mistralai/Voxtral-Mini-3B-2507` is Mistral AI's audio-language model:
a Whisper-derived audio encoder + a small linear projector + a Llama-based
LLM backbone (Ministral-3B). It handles both text-only chat and audio inputs
(transcription, audio understanding, summarization) up to 30 s of audio per
request.

This contrib implements Voxtral-Mini-3B on AWS Neuron as a **decomposed
pipeline**:

1. **Audio encoder** (`VoxtralEncoder`, Whisper-like, 637 M params) is
   traced separately via `torch_neuronx.trace()` and runs on a single
   NeuronCore.
2. **Projector** (25 M params, Linear→GELU→Linear) runs on CPU. It packs
   four adjacent encoder hidden states into a single text-space token.
3. **Text decoder** (Ministral-3B, 3.3 B params, 30 layers, GQA 32/8)
   reuses NxDI's `NeuronLlamaModel` with **scatter-based audio-token
   injection** via `scatter_by_index_put` (the same pattern used by
   Pixtral and Llama-4 vision).

The whole pipeline is orchestrated by `NeuronApplicationVoxtral` in
`src/modeling_voxtral.py` — `compile()` builds both the audio encoder NEFF
and the text-decoder NEFFs; `load()` brings them onto NeuronCores and
loads the CPU projector; `transcribe()` and `generate()` run the audio +
text pipeline end-to-end via `HuggingFaceGenerationAdapter`.

## Contents

```
voxtral-mini-3B/
├── README.md
├── voxtral_trn2_walkthrough.ipynb   — end-to-end walkthrough on trn2.3xlarge
├── benchmark_encoder.py             — component-level encoder benchmark
├── benchmark_harness/               — serial latency harness (user-supplied audio)
│   ├── run_voxtral_benchmark.py
│   ├── backends.py
│   ├── backend_nxdi.py
│   ├── common/
│   └── dataset/                     — populate with your own audio
├── src/
│   ├── __init__.py
│   ├── modeling_voxtral.py          — the Neuron implementation
│   └── utils/
└── test/
    ├── __init__.py
    └── integration/
        ├── __init__.py
        └── test_model.py            — logit-validation + audio smoke
```

## Compatibility Matrix

| Instance | SDK | TP | LNC | dtype | Status |
|----------|-----|----|-----|-------|--------|
| trn2.3xlarge | 2.30 | 4 | 2 | bfloat16 | **Validated (recommended)** |
| trn2.3xlarge | 2.30 | 1 | 2 | bfloat16 | Validated |
| trn2.3xlarge | 2.31 | 4 | 2 | bfloat16 | Validated |
| trn2.3xlarge | 2.28 | 1 | 2 | bfloat16 | Validated |
| inf2.xlarge  | 2.28 | 1 | -- | bfloat16 | Validated |
| inf2.8xlarge | -- | -- | -- | -- | Not tested |

Recommended: **SDK 2.30 (DLAMI 20260522), TP=4, LNC=2, bfloat16** on
`trn2.3xlarge`. All performance numbers below are measured in this
configuration unless stated otherwise.

## Performance

Benchmark: 18 audio clips (0-30 s, mono 16 kHz) transcribed serially,
prompt `"Transcribe this audio."`, `max_new_tokens=256`, greedy. All numbers
measured end-to-end on `trn2.3xlarge`.

### trn2.3xlarge, TP=4, LNC=2, bfloat16, SDK 2.30

Two serving paths are supported.  The **standalone** path
(`NeuronApplicationVoxtral.transcribe()`) is ~7% faster because it can
use a smaller CTE bucket (`seq_len=512`) than the KV cache
(`n_positions=768`).  The **vLLM** path requires
`seq_len == n_positions` (see Known Limitations) so it recompiles with
`seq_len=768`.

Configuration common to both paths: `move_trace_to_device=True` on the
encoder, on-device sampling (greedy), TP=4 on trn2.3xlarge at LNC=2.

**Standalone `NeuronApplicationVoxtral` path** (`seq_len=512`,
`n_positions=768`):

| Metric | Value |
|--------|-------:|
| Mean latency per file (5-clip mix, 5-25 s) | **0.477 s** |
| Median decode throughput (18-clip TED benchmark, 3 clips per 5 s bin) | **155 tok/s** |
| Encoder wall (median) | 78 ms |
| CPU projector | 7 ms |
| Decoder wall (median) | 389 ms |

**vLLM path** (`seq_len=768`, `n_positions=768`):

| Metric | Value |
|--------|-------:|
| Mean latency per file (5-clip mix, 5-25 s) | **0.502 s** |
| Median per-file latency (same mix) | 0.456 s |
| P90 | 0.729 s |

Per-duration bin (5-clip harness, means in ms) -- both paths:

| Duration | 5-10 s | 10-15 s | 15-20 s | 20-25 s | 25-30 s |
|----------|:------:|:-------:|:-------:|:-------:|:-------:|
| Standalone ms | 184 | 419 | 590 | 700 | -- |
| vLLM ms       | 291 | 408 | 454 | 626 | 733 |

### trn2.3xlarge, TP=1, LNC=2, bfloat16, SDK 2.30

Standalone path only. Mean latency per file: **0.967 s**, decode
throughput median **66 tok/s**. Functional at TP=1; 16/18 transcripts
byte-identical to TP=4 (2 differ in punctuation or paraphrasing).

## Optimizations shipped in this contrib

Four changes over the initial port (stock SDK 2.31 was 0.656 s/file at
TP=4) that together bring the harness mean to ~0.47-0.50 s/file:

1. **`torch_neuronx.move_trace_to_device(encoder, 0)`** at load time
   (default; the constructor argument `move_trace_to_device=True` gates
   this -- set to `False` to fall back to `async_load`).
   Removes ~95 ms of per-call host overhead on the audio encoder.
2. **On-device sampling** (`on_device_sampling=True`): greedy argmax
   runs inside the compiled NEFF (`OnDeviceSamplingConfig(top_k=1,
   do_sample=False, dynamic=False, deterministic=False)`), eliminating a
   host round-trip per generated token. Both compile and runtime must
   agree, so a single constructor flag controls both.
3. **`n_positions=768`** (down from 4096): shrinks the KV cache to fit the
   maximum audio-token count (375) plus a short prompt. Per-token DMA
   cost drops ~5×.
4. **`seq_len=512`** (CTE bucket, down from 2048): matches the
   ~375-token audio prefill. Small additional decoder-side win.
   **Standalone path only.**  For vLLM serving, set `seq_len=768`
   (see Known Limitations).

## Usage

Standalone (fastest):

```python
from modeling_voxtral import NeuronApplicationVoxtral
import torch

MODEL_PATH = "/mnt/models/Voxtral-Mini-3B-2507"       # HF snapshot dir
COMPILED_PATH = "/mnt/models/compiled/voxtral_mini_3b"

app = NeuronApplicationVoxtral(
    model_path=MODEL_PATH,
    tp_degree=4,               # TP=4 recommended on trn2.3xlarge
    seq_len=512,               # SDK 2.30 optimum; use 2048 on other SDKs
    n_positions=768,           # sized for a single 30 s audio clip
    dtype=torch.bfloat16,
    on_device_sampling=True,   # greedy argmax on-device
    move_trace_to_device=True, # encoder pre-staged onto NeuronCore 0
)

# One-time compile (subsequent runs reuse compiled/)
import os
if not os.path.exists(os.path.join(COMPILED_PATH, "text_decoder",
                                   "text_model", "model.pt")):
    app.compile(COMPILED_PATH)

app.load(COMPILED_PATH)

# Audio transcription
text = app.transcribe("/path/to/audio.wav", max_new_tokens=256)
print(text)

# Text-only chat
answer = app.generate("What is the capital of France?", max_new_tokens=50)
print(answer)
```

For vLLM serving, compile with `seq_len=768` (matching `n_positions`)
and use the vllm-neuron plugin from the jimburtoft fork -- see the
walkthrough notebook.

## Example Checkpoints

- [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Testing Instructions

Prerequisites (SDK 2.30 DLAMI 20260522):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install 'mistral_common[audio]>=1.8.1' 'transformers>=4.54.0'

# HF token required (gated repo)
huggingface-cli download mistralai/Voxtral-Mini-3B-2507 \
    --local-dir /mnt/models/Voxtral-Mini-3B-2507
```

Run the integration tests:

```bash
export VOXTRAL_MODEL_PATH=/mnt/models/Voxtral-Mini-3B-2507
export VOXTRAL_COMPILED_PATH=/mnt/models/compiled/voxtral_mini_3b
export VOXTRAL_TP_DEGREE=4
export VOXTRAL_SEQ_LEN=512
export VOXTRAL_N_POSITIONS=768

pytest test/integration/test_model.py -v
```

The tests:
- `test_model_loads` — smoke-check that every component (encoder,
  projector, decoder, adapter, tokenizer, processor) loaded.
- `test_text_logit_validation` — compares Neuron logits to a CPU BF16
  reference over `NUM_TOKENS_TO_CHECK=16` tokens using NxDI's
  `check_accuracy_logits_v2`.
- `test_text_generation_deterministic` — greedy generation is
  reproducible run-to-run.
- `test_audio_transcription` — audio smoke test on the TED-60 sample
  from `reach-vb/random-audios` (16 kHz mono). Compares Neuron
  transcription tokens to CPU BF16 reference over the first 16 tokens
  using `neuron_allclose`.

## Notebook: `voxtral_trn2_walkthrough.ipynb`

An end-to-end Jupyter walkthrough covering: launching the trn2.3xlarge
instance, installing dependencies, downloading the model, compiling
(one-time), running a single-file transcription, and running the benchmark
harness in `benchmark_harness/`. See the notebook itself for details.

## Benchmark harness

`benchmark_harness/` reproduces the mean-per-file latency table above.
Populate `benchmark_harness/dataset/` with your own audio clips (16 kHz
mono `.wav` or any format that `soundfile` can decode) and update
`manifest.csv` with `audio_path,duration_sec,transcript` rows. Then:

```bash
cd benchmark_harness
python run_voxtral_benchmark.py \
    --backend nxdi_neuron \
    --manifest dataset/manifest.csv \
    --model-dir /mnt/models/Voxtral-Mini-3B-2507 \
    --compiled-dir /mnt/models/compiled/voxtral_mini_3b \
    --tp-degree 4 --seq-len 512 --n-positions 768 \
    --output-dir results/ --runs 3

python summarize_results.py results/
```

## Known Limitations

- **Batch size 1 only.** The stock `ImageToTextModelWrapper` uses
  `scatter_by_index_put` in a way that is not shape-safe for BS>1 CTE.
  A batched compile-time argument is not supported by this contrib and
  has not been benchmarked. Set `max_num_seqs=1` for any serving path.
- **30 s audio maximum.** Voxtral supports up to 30 min transcribe / 40
  min understand modes upstream; those code paths have not been
  validated on Neuron. All benchmarks are on 0-30 s clips.
- **vLLM path requires `seq_len == n_positions`.** vLLM 0.16.0's V1
  scheduler feeds the TKG NEFF positions that can span the range
  `[0, n_positions)` during decode. If the traced NEFF has
  `seq_len < n_positions` (e.g. the standalone-optimized `seq_len=512
  / n_positions=768` combo), positions in the gap trigger NRT status
  1006. Standalone `NeuronApplicationVoxtral.transcribe()` does not
  hit this because its generation loop pads inputs to `seq_len` before
  decoding.  Use `seq_len=n_positions` for vLLM serving; the
  standalone path can use a shorter `seq_len` for ~5-7% additional
  speedup.
- **`--auto-cast=matmult`** is *not* passed to the text decoder (NxDI
  handles the LLM path). It *is* included in the audio encoder trace
  compiler args, which is required for the FP32 → BF16 path in Whisper's
  attention.
- **Encoder is TP=1** even at LLM TP=4. The encoder is traced separately
  and always runs on a single NeuronCore. This ~78 ms is fixed regardless
  of the decoder TP degree.
- **Function calling** (available in Voxtral-Small-24B only) is out of
  scope for this contrib.
- **Voxtral-Small-24B** is not covered by this contrib. It uses the
  same architecture and could be onboarded with the same
  `NeuronApplicationVoxtral` pattern at TP=4, but has not been
  validated in this branch.

## Dependencies

- `transformers >= 4.54.0` (for `VoxtralForConditionalGeneration`)
- `mistral_common[audio] >= 1.8.1` (audio pre-processing +
  `apply_chat_template` audio slot)
- `neuronx-distributed-inference` (base classes, tested against
  0.10.17970 on SDK 2.30 and 0.10.18399 on SDK 2.31)
- `torch-neuronx` (audio-encoder trace and `move_trace_to_device`)

## Maintainer

Jim Burtoft ([jimburtoft](https://github.com/jimburtoft))

## Last Updated

2026-07-30
