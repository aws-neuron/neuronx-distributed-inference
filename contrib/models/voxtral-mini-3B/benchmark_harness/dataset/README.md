# Populate your dataset here

This directory is empty by design.  The benchmark harness expects you to add
your own audio files and list them in `manifest.csv`.

## Format

Any format `soundfile` can decode: WAV, FLAC, OGG.  MP3 works via
`libsndfile` on modern installs.  Sample rate does not need to be 16 kHz —
the loader resamples via librosa if needed.

## Manifest

`manifest.csv` has three columns:

```
audio_path,duration_sec,transcript
my_clip.wav,12.3,The quick brown fox jumps over the lazy dog.
another/subdir/clip2.wav,7.8,
```

- `audio_path`: relative to `dataset/` or absolute.
- `duration_sec`: clip length in seconds.  Only used for the per-duration-bin
  report in `summarize_results.py`.  Rough integer or float is fine.
- `transcript`: ground-truth text (optional; leave blank if unknown).  This
  harness does NOT compute WER — the transcript column is passed through
  to the output CSV for downstream analysis if you want it.

## Expected clip length

Voxtral processes audio in 30-second chunks internally.  Clips longer
than 30 s will be truncated by the audio encoder to the first 30 s.

## Where to get sample audio

For quick end-to-end validation you can use:

- [reach-vb/random-audios/ted_60.wav](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/ted_60.wav)
  (60 s single-speaker clip; will be truncated to 30 s by the encoder)
- Any 16 kHz mono WAV file you have on hand
- The [LibriSpeech](https://www.openslr.org/12/) dev-clean split

To reproduce the ~0.468 s/file mean documented in the parent README, use
18 clips spread across the six 5-second duration bins (0-5, 5-10, 10-15,
15-20, 20-25, 25-30 seconds), three clips per bin.  Numbers will vary by
content complexity — dense speech generates more tokens and therefore takes
longer.
