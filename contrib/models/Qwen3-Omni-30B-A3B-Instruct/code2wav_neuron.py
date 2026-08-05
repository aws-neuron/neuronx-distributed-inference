"""Runtime shim: replace ``hf_model.code2wav`` CPU calls with Neuron NEFFs.

Buckets are chosen at install time; at call time we pick the smallest bucket
>= T and pad the codec-tokens tensor up to it. The output is trimmed back to
``T * total_upsample`` samples to match CPU behavior.

Install once per process via ``install_neuron_code2wav(hf_model)``.
"""
import os
from pathlib import Path
from typing import List, Optional

import torch

DEFAULT_BUCKETS_DIR = Path("/tmp/qwen3_omni_compiled/code2wav_buckets")


class NeuronCode2WavShim(torch.nn.Module):
    """Holds one compiled NEFF per bucket size; dispatches on T at call time."""

    def __init__(self, hf_c2w, buckets_dir: Path, buckets: Optional[List[int]] = None):
        super().__init__()
        # We want to keep ``config`` and ``total_upsample`` from the original so
        # callers that read those still work (``chunked_decode`` uses
        # ``self.total_upsample``).
        self.hf_c2w = hf_c2w
        self.config = hf_c2w.config
        self.total_upsample = hf_c2w.total_upsample

        found = {}
        for f in sorted(buckets_dir.glob("model_T*.pt")):
            # Parse T from filename "model_T{int}.pt"
            T = int(f.stem.split("_T")[-1])
            if buckets is None or T in buckets:
                found[T] = f
        if not found:
            raise RuntimeError(f"No code2wav NEFFs found in {buckets_dir}")

        self._neffs = {}
        for T in sorted(found):
            print(f"  [code2wav shim] loading T={T} from {found[T]}")
            self._neffs[T] = torch.jit.load(str(found[T]))
        self._bucket_sizes = sorted(self._neffs.keys())
        self._max_bucket = self._bucket_sizes[-1]

    def _pick_bucket(self, T: int) -> int:
        for b in self._bucket_sizes:
            if b >= T:
                return b
        # T exceeds the largest bucket — fall back to CPU.
        return -1

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        B, Q, T = codes.shape
        bucket = self._pick_bucket(T)
        if bucket == -1:
            # No NEFF big enough: use CPU
            return self.hf_c2w(codes)

        if T == bucket:
            padded_codes = codes
        else:
            # Right-pad with zeros (valid codec ids live in [0, codebook_size=2048))
            pad_amount = bucket - T
            pad = torch.zeros((B, Q, pad_amount), dtype=codes.dtype, device=codes.device)
            padded_codes = torch.cat([codes, pad], dim=-1)

        neuron = self._neffs[bucket]
        wav = neuron(padded_codes)
        # Output shape is (B, 1, bucket * total_upsample). Trim to real length.
        real_samples = T * self.total_upsample
        wav = wav[..., :real_samples]
        return wav

    # chunked_decode is inherited behavior on hf_c2w but our forward shim gets
    # called with codes — we re-implement here for symmetry and to avoid HF
    # accidentally calling the CPU forward.
    def chunked_decode(self, codes: torch.Tensor, chunk_size: int = 300,
                       left_context_size: int = 25) -> torch.Tensor:
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size: end_index]
            wav_chunk = self.forward(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample:])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


def install_neuron_code2wav(
    hf_model,
    buckets_dir: Path = DEFAULT_BUCKETS_DIR,
    buckets: Optional[List[int]] = None,
) -> NeuronCode2WavShim:
    """Replace ``hf_model.code2wav`` with a Neuron-backed shim.

    Returns the shim (holding the original HF code2wav on ``.hf_c2w`` in case
    callers want to fall back).
    """
    shim = NeuronCode2WavShim(hf_model.code2wav, buckets_dir, buckets=buckets)
    hf_model.code2wav = shim
    return shim
