"""vLLM-neuron Voxtral-Mini-3B backend.

Uses the OpenAI-compatible in-process vLLM engine backed by the
`vllm-neuron` Voxtral plugin.  Requires the vLLM-neuron fork with
Voxtral support installed and the NxDI contrib `src/` directory on
`PYTHONPATH` so vLLM can import `NeuronApplicationVoxtral`.

Latency is measured end-to-end around `llm.chat()`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from backends import Backend, TRANSCRIBE_PROMPT_TEXT

# Locate the sibling `src/` directory so vLLM can import modeling_voxtral.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.environ["PYTHONPATH"] = str(_SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")


class VllmNeuronBackend(Backend):
    """vLLM-neuron in-process serving for Voxtral-Mini-3B."""

    def __init__(
        self,
        model_dir: str,
        compiled_dir: str,
        tp_degree: int = 4,
        seq_len: int = 768,
        n_positions: int = 768,
        max_num_seqs: int = 1,
        max_model_len: int = 768,
        max_new_tokens: int = 256,
        transcribe_prompt: str = TRANSCRIBE_PROMPT_TEXT,
        allowed_local_media_path: str = "/",
    ):
        # vLLM's Voxtral loader consults NEURON_COMPILED_ARTIFACTS for the
        # pre-compiled NEFFs; if missing it will re-compile from scratch.
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(compiled_dir)

        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams

        self._llm = LLM(
            model=str(model_dir),
            tokenizer_mode="mistral",
            tensor_parallel_size=tp_degree,
            dtype="bfloat16",
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            allowed_local_media_path=allowed_local_media_path,
            additional_config={
                "override_neuron_config": {
                    "on_device_sampling": True,
                    "move_trace_to_device": True,
                    "n_positions": n_positions,
                    "seq_len": seq_len,
                },
            },
        )
        self.max_new_tokens = max_new_tokens
        self.transcribe_prompt = transcribe_prompt
        self.last_stats: dict = {}

    def transcribe(self, audio_path: Path, audio_wave: np.ndarray) -> str:
        # vLLM's Mistral chat template needs the audio via audio_url with a
        # file:// or https:// URL.  audio_wave is unused here.
        del audio_wave

        conv = [{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": f"file://{audio_path.absolute()}"}},
                {"type": "text", "text": self.transcribe_prompt},
            ],
        }]
        sp = self._SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)

        t0 = time.perf_counter()
        outputs = self._llm.chat(conv, sampling_params=sp)
        elapsed = time.perf_counter() - t0

        text = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)
        self.last_stats = {
            "latency_e2e_ms": elapsed * 1000.0,
            "num_generated": n_tokens,
            "throughput_tok_s": n_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return text


def register(registry: dict) -> None:
    registry["vllm_neuron"] = VllmNeuronBackend
