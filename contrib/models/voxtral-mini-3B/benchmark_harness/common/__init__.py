"""Common helpers for the Voxtral benchmark harness.

Three tiny modules:

- `audio` — 16 kHz mono float32 loader with soundfile / librosa fallback.
- `manifest` — CSV read/write for the audio manifest and results.
- `timing` — Neuron device sync helper.
"""
