# Voxtral-Mini-3B benchmark harness

Serial per-file latency benchmark for `NeuronApplicationVoxtral` (from the
sibling `src/modeling_voxtral.py`).  Modeled on the "measure each file with
a device-sync bracket" pattern used by upstream WhisperS2T benchmarks.

## What it does

For each audio file listed in `dataset/manifest.csv`:

1. Load audio (mono 16 kHz float32, via `soundfile` / `librosa` fallback).
2. Call `NeuronApplicationVoxtral.transcribe(audio_path)`.
3. Record `latency_sec` end-to-end.
4. Write the resulting `latency.csv` (or `run_1.csv`, `run_2.csv`, ... in
   multi-run mode).

The measured field is `latency_sec` (wall-clock `perf_counter` around the
Neuron call, bracketed by device sync).

## Populate your own dataset

The harness ships with an **empty manifest** (`dataset/manifest.csv` — header
row only) and **no audio files**.  Add your own audio to `dataset/` and list
each file in the manifest.

Manifest schema:

```
audio_path,duration_sec,transcript
```

- `audio_path` — relative or absolute path to the audio file.
- `duration_sec` — clip length in seconds (only used for reporting bins).
- `transcript` — ground-truth reference (optional; leave blank if you don't
  have one — WER is not computed by this harness).

Supported audio formats: anything `soundfile` can decode (WAV, FLAC, OGG,
MP3 via `libsndfile`).  Voxtral processes audio in 30 s chunks internally,
so clips longer than 30 s will be truncated by the audio encoder.

The performance numbers in the parent README (0.468 s/file mean) were
measured on 18 clips distributed across 6 duration bins (0-5, 5-10, 10-15,
15-20, 20-25, 25-30 seconds, three files per bin, TED-style single-speaker
speech).  Your numbers will vary by:

- Clip length distribution (longer clips generate more tokens → longer)
- Content complexity (higher token count for dense speech)
- SDK / DLAMI version (see the parent README compatibility matrix)

## Running

Prerequisites: the model must already be compiled (see `voxtral_trn2_walkthrough.ipynb`
or the parent README testing section).

```bash
python run_voxtral_benchmark.py \
    --backend nxdi_neuron \
    --manifest dataset/manifest.csv \
    --model-dir /mnt/models/Voxtral-Mini-3B-2507 \
    --compiled-dir /mnt/models/compiled/voxtral_mini_3b \
    --tp-degree 4 --seq-len 512 --n-positions 768 --ods \
    --output-dir results/ \
    --runs 3

python summarize_results.py results/
```

Options:

- `--runs N` — repeat the whole manifest N times (writes `run_1.csv`, ...).
  Use 3+ runs to filter out cold-cache noise from run 1.
- `--limit N` — only process the first N rows.  Useful for smoke tests.
- `--tp-degree {1,4}` — must match the compiled artifact.  Recompile if changed.
- `--seq-len 512` — SDK 2.30 only.  Use 2048 on SDK 2.31.
- `--n-positions 768` — matches the max audio prefill length.
- `--ods` — enable on-device sampling.  Only valid for greedy inference.
- `--skip-warmup` — do not run a warm-up call before measuring.

## Output

`results/latency.csv` (or `results/run_N.csv`) has these columns:

- `audio_path,duration_sec,transcript` — copied from the manifest.
- `transcript_hyp` — Neuron's transcription hypothesis.
- `latency_sec` — end-to-end perf_counter time for the `transcribe()` call.
- `encoder_ms,projector_ms,decoder_ms,num_generated,throughput_tok_s` —
  optional per-phase timings if the backend exposes them.

`summarize_results.py` prints mean / median / p90 / p99 latency plus a
per-duration-bin breakdown.
