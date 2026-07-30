"""NxDI Voxtral-Mini-3B backend.

Thin wrapper around `NeuronApplicationVoxtral` from the sibling
`src/modeling_voxtral.py`.  Loads the pre-compiled model once at
construction time and calls `.transcribe(audio_path)` per file.

Exposes optional per-phase timing (`encoder_ms`, `projector_ms`,
`decoder_ms`, `num_generated`, `throughput_tok_s`) via `self.last_stats`
so the harness driver can log them.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from backends import Backend, TRANSCRIBE_PROMPT_TEXT

# Locate the sibling `src/` directory so we can import modeling_voxtral.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling_voxtral import NeuronApplicationVoxtral  # noqa: E402


class NxdiNeuronBackend(Backend):
    """NxDI Voxtral-Mini-3B, decomposed pipeline (traced encoder + NxDI LLM)."""

    def __init__(
        self,
        model_dir: str,
        compiled_dir: str,
        tp_degree: int = 4,
        seq_len: int = 512,
        n_positions: int = 768,
        dtype: str = "bfloat16",
        on_device_sampling: bool = True,
        move_trace_to_device: bool = True,
        max_new_tokens: int = 256,
        transcribe_prompt: str = TRANSCRIBE_PROMPT_TEXT,
    ):
        torch_dtype = getattr(torch, dtype)
        self.app = NeuronApplicationVoxtral(
            model_path=model_dir,
            tp_degree=tp_degree,
            seq_len=seq_len,
            n_positions=n_positions,
            dtype=torch_dtype,
            on_device_sampling=on_device_sampling,
            move_trace_to_device=move_trace_to_device,
        )

        marker = os.path.join(compiled_dir, "text_decoder", "text_model", "model.pt")
        if not os.path.exists(marker):
            print(
                f"[NxdiNeuronBackend] Compiling to {compiled_dir} (one-time, "
                "several minutes)..."
            )
            self.app.compile(compiled_dir)

        print(f"[NxdiNeuronBackend] Loading compiled model from {compiled_dir}...")
        self.app.load(compiled_dir)

        self.max_new_tokens = max_new_tokens
        self.transcribe_prompt = transcribe_prompt
        self.last_stats: dict = {}

    def transcribe(self, audio_path: Path, audio_wave: np.ndarray) -> str:
        # The mistral_common processor requires a path/url — audio_wave is
        # unused for this backend (kept in the signature for compatibility
        # with GPU / other backends that reuse the numpy array).
        del audio_wave

        t0 = time.perf_counter()
        text = self.app.transcribe(
            str(audio_path),
            prompt=self.transcribe_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        # Basic latency accounting.  Per-phase timings (encoder/decoder) are
        # exposed by NeuronApplicationVoxtral in future revisions; for now
        # we record only end-to-end and a synthetic tok/s.
        elapsed = time.perf_counter() - t0

        # Estimate number of generated tokens from the returned text
        # (approximate; tokenizer-round-trip gives an exact count).
        try:
            n_tokens = len(self.app.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            n_tokens = 0
        self.last_stats = {
            "latency_e2e_ms": elapsed * 1000.0,
            "num_generated": n_tokens,
            "throughput_tok_s": n_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return text


# --- Registration ---------------------------------------------------------

def register(registry: dict) -> None:
    registry["nxdi_neuron"] = NxdiNeuronBackend
