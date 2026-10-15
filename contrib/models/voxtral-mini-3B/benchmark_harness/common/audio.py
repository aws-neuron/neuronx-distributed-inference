"""Audio loading helpers.

Load an audio file as a mono 16 kHz float32 waveform.  Uses soundfile if
available (fastest for WAV / FLAC), falls back to librosa for resampling
and codec support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000


def load_audio_16k(path: Path) -> np.ndarray:
    """Load an audio file as mono 16 kHz float32."""
    try:
        import soundfile as sf
    except ImportError:  # pragma: no cover
        sf = None

    if sf is not None:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != SAMPLE_RATE:
            audio, sr = _librosa_load(path)
            assert sr == SAMPLE_RATE
        return audio.astype(np.float32)

    audio, sr = _librosa_load(path)
    return audio.astype(np.float32)


def _librosa_load(path: Path) -> tuple[np.ndarray, int]:
    import librosa

    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio, sr


def pad_or_trim(audio: np.ndarray, length: int = SAMPLE_RATE * 30) -> np.ndarray:
    """Whisper 30-second window: pad short clips with zeros, trim long ones."""
    if audio.shape[0] > length:
        return audio[:length].astype(np.float32)
    if audio.shape[0] < length:
        return np.pad(audio, (0, length - audio.shape[0])).astype(np.float32)
    return audio.astype(np.float32)
