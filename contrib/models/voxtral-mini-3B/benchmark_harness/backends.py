"""Backend interface for the Voxtral benchmark harness.

A `Backend` implements one method:

    transcribe(audio_path, audio_wave) -> str

where `audio_path` is a `pathlib.Path` and `audio_wave` is a mono 16 kHz
float32 numpy array.  Both are provided; a backend may use whichever it
prefers.  The mistral_common processor requires a path, so most backends
just use `audio_path` and ignore `audio_wave`.

Backends may set `self.last_stats` (a dict of extra scalar metrics) after
each `transcribe()` call, and the driver will append them as columns in
the results CSV.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, Dict

import numpy as np


TRANSCRIBE_PROMPT_TEXT = "Transcribe this audio."


class Backend(abc.ABC):
    """Abstract Voxtral transcription backend."""

    @abc.abstractmethod
    def transcribe(self, audio_path: Path, audio_wave: np.ndarray) -> str: ...

    def warmup(self, audio_path: Path, audio_wave: np.ndarray) -> str:
        return self.transcribe(audio_path, audio_wave)


_REGISTRY: Dict[str, Callable[..., Backend]] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    _REGISTRY[name] = factory


def load_backend(name: str, **kwargs) -> Backend:
    if name not in _REGISTRY:
        # Trigger lazy import.  Adding a new backend? register it here.
        if name == "nxdi_neuron":
            from backend_nxdi import register as _reg  # type: ignore

            _reg(_REGISTRY)
        else:
            raise ValueError(f"Unknown backend {name!r}; known: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
