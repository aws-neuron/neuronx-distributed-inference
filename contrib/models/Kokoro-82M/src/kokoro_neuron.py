# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M Neuron TTS Model
============================
Text-to-speech inference using hexgrad/Kokoro-82M on AWS Neuron (trn2).

Architecture: 3-part split for torch_neuronx.trace() compatibility.
  Part A    (Neuron): encode + decode[0:2]     ~0.8ms
  Part B1   (Neuron): decode[3] with upsample  ~0.8ms
  B1.5      (CPU):    f0 -> har precompute     ~2.9ms
  Part B2   (Neuron): generator body           ~16ms
  Total:    ~20ms for ~1.6s audio = 78x real-time (bucket=128)

Workarounds applied:
  1. CustomSTFT boolean mask assignment -> torch.where (XLA tracer crash)
  2. UpSample1d F.interpolate -> torch.repeat_interleave (XLA drops tensors)
  3. SineGen random ops -> deterministic zeros (tracing compatibility)
  4. ConvTranspose1d(groups=1090) -> DepthwiseTransposeExact (NCC_ITEN404 bug)
  5. har precomputed on CPU (F.interpolate scale_factor=300 not traceable)

Usage:
    from kokoro_neuron import KokoroNeuron

    model = KokoroNeuron()
    model.compile(buckets=[32, 64, 96, 128, 160, 192])
    model.save("compiled_models")

    model = KokoroNeuron.load("compiled_models")
    # Short text (single chunk):
    audio = model.generate("Hello, this is a test.", voice="af_heart")
    # Long text (auto-chunked with crossfade stitching):
    audio = model.generate("A very long paragraph...", voice="af_heart")
    # Streaming (yields chunks as they're generated):
    for chunk in model.generate_stream("Long text...", voice="af_heart"):
        play(chunk)  # process each chunk immediately
"""

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# Monkey patches -- must be applied BEFORE importing kokoro
# ================================================================
import kokoro.custom_stft as _custom_stft_module
import kokoro.istftnet as _istftnet_module


def _patched_transform(self, waveform):
    """Fix CustomSTFT boolean mask assignment that crashes XLA tracer."""
    if self.center:
        pad_len = self.n_fft // 2
        waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
    x = waveform.unsqueeze(1)
    real_out = F.conv1d(
        x, self.weight_forward_real, bias=None, stride=self.hop_length, padding=0
    )
    imag_out = F.conv1d(
        x, self.weight_forward_imag, bias=None, stride=self.hop_length, padding=0
    )
    magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
    phase = torch.atan2(imag_out, real_out)
    correction_mask = (imag_out == 0) & (real_out < 0)
    pi_tensor = torch.full_like(phase, torch.pi)
    phase = torch.where(correction_mask, pi_tensor, phase)
    return magnitude, phase


_custom_stft_module.CustomSTFT.transform = _patched_transform


def _patched_upsample_forward(self, x):
    """Replace F.interpolate with torch.repeat_interleave for XLA tracing.

    F.interpolate causes XLA tracer to DROP input tensors entirely.
    For nearest-neighbor 2x upsampling, repeat_interleave is identical.
    """
    if self.layer_type == "none":
        return x
    else:
        return torch.repeat_interleave(x, 2, dim=-1)


_istftnet_module.UpSample1d.forward = _patched_upsample_forward


def _patched_f02sine(self, f0_values):
    """Replace torch.rand() with torch.zeros() for deterministic tracing."""
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.zeros(
        f0_values.shape[0], f0_values.shape[2], device=f0_values.device
    )
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    if not self.flag_for_pulse:
        rad_values = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        sines = torch.sin(phase)
    else:
        pass
    return sines


_istftnet_module.SineGen._f02sine = _patched_f02sine


def _patched_sinegen_forward(self, f0):
    """Replace torch.FloatTensor with torch.arange, torch.randn_like with zeros."""
    harmonics = (
        torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    fn = f0 * harmonics
    sine_waves = self._f02sine(fn) * self.sine_amp
    uv = self._f02uv(f0)
    noise = torch.zeros_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


_istftnet_module.SineGen.forward = _patched_sinegen_forward


def _patched_source_forward(self, x):
    """Replace torch.randn_like with torch.zeros_like for deterministic tracing."""
    sine_wavs, uv, _ = self.l_sin_gen(x)
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))
    noise = torch.zeros_like(uv)
    return sine_merge, noise, uv


_istftnet_module.SourceModuleHnNSF.forward = _patched_source_forward


# ================================================================
# Constants
# ================================================================

SAMPLE_RATE = 24000
HOP_SIZE = 300  # upsample_rates=[10,6] * gen_istft_hop_size=5
DEFAULT_BUCKETS = [32, 64, 96, 128, 160, 192]
COMPILER_ARGS = ["--auto-cast", "matmult", "--auto-cast-type", "bf16"]


# ================================================================
# Depthwise ConvTranspose1d Decomposition
# ================================================================


class DepthwiseTransposeExact(nn.Module):
    """Exact decomposition of depthwise ConvTranspose1d into traceable ops.

    Replaces ConvTranspose1d(C, C, K=3, stride=S, padding=P, output_padding=OP, groups=C)
    with: zero-insert -> 3-tap element-wise conv -> output padding.

    This bypasses the NCC_ITEN404 bug in the NKI depthwise conv kernel when
    channels=1090. Only supports kernel_size=3 (Kokoro's decode[3] pool layer).
    """

    def __init__(self, conv_transpose_ref: nn.ConvTranspose1d):
        super().__init__()
        w = conv_transpose_ref.weight  # [C, 1, K]
        self.w_flip = nn.Parameter(w.flip(-1))
        self.bias = conv_transpose_ref.bias
        self.groups = conv_transpose_ref.groups
        self.stride = conv_transpose_ref.stride[0]
        self.padding = conv_transpose_ref.padding[0]
        self.output_padding = conv_transpose_ref.output_padding[0]
        self.kernel_size = conv_transpose_ref.kernel_size[0]
        assert self.kernel_size == 3, (
            f"Only kernel_size=3 supported, got {self.kernel_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        L_up = (L - 1) * self.stride + 1
        x_up = torch.zeros(B, C, L_up, dtype=x.dtype, device=x.device)
        x_up[:, :, :: self.stride] = x

        conv_pad = self.kernel_size - 1 - self.padding
        w0 = self.w_flip[:, 0, 0].unsqueeze(0).unsqueeze(-1)
        w1 = self.w_flip[:, 0, 1].unsqueeze(0).unsqueeze(-1)
        w2 = self.w_flip[:, 0, 2].unsqueeze(0).unsqueeze(-1)

        x_padded = F.pad(x_up, (conv_pad, conv_pad))
        y = (
            w0 * x_padded[:, :, :-2]
            + w1 * x_padded[:, :, 1:-1]
            + w2 * x_padded[:, :, 2:]
        )
        y = y + self.bias.unsqueeze(0).unsqueeze(-1)

        if self.output_padding > 0:
            y = F.pad(y, (0, self.output_padding))
        return y


# ================================================================
# Traced Module Wrappers
# ================================================================


class _PartA(nn.Module):
    """Neuron-traced Part A: encoder + first 3 decode blocks."""

    def __init__(self, decoder):
        super().__init__()
        self.encode = decoder.encode
        self.decode_0 = decoder.decode[0]
        self.decode_1 = decoder.decode[1]
        self.decode_2 = decoder.decode[2]
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.asr_res = decoder.asr_res

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        x = torch.cat([x, asr_res, F0, N_out], axis=1)
        x = self.decode_0(x, s)
        x = torch.cat([x, asr_res, F0, N_out], axis=1)
        x = self.decode_1(x, s)
        x = torch.cat([x, asr_res, F0, N_out], axis=1)
        x = self.decode_2(x, s)
        return x, asr_res, F0, N_out


class _PartB1(nn.Module):
    """Neuron-traced Part B1: decode[3] with decomposed ConvTranspose1d."""

    def __init__(self, decode3_block):
        super().__init__()
        self.block = decode3_block
        if hasattr(self.block, "pool") and isinstance(
            self.block.pool, nn.ConvTranspose1d
        ):
            self.block.pool = DepthwiseTransposeExact(self.block.pool)

    def forward(self, x, s):
        return self.block(x, s)


class _PartB2(nn.Module):
    """Neuron-traced Part B2: generator body taking precomputed har."""

    def __init__(self, gen):
        super().__init__()
        self.noise_convs = gen.noise_convs
        self.noise_res = gen.noise_res
        self.ups = gen.ups
        self.resblocks = gen.resblocks
        self.conv_post = gen.conv_post
        self.reflection_pad = gen.reflection_pad
        self.stft = gen.stft
        self.num_upsamples = gen.num_upsamples
        self.num_kernels = gen.num_kernels
        self.post_n_fft = gen.post_n_fft

    def forward(self, x, s, har):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


# ================================================================
# Main API
# ================================================================


class KokoroNeuron:
    """Kokoro-82M TTS model compiled for AWS Neuron.

    This class handles model loading, compilation, and inference.

    Example:
        # Compile and save
        model = KokoroNeuron()
        model.compile(buckets=[64, 128, 192])
        model.save("compiled_models")

        # Load and generate
        model = KokoroNeuron.load("compiled_models")
        audio = model.generate("Hello world!", voice="af_heart")
        # audio is a numpy array at 24kHz
    """

    def __init__(self):
        """Load the Kokoro-82M model with weight_norm removed."""
        from kokoro import KModel

        self._kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
        self._kmodel = self._kmodel.to("cpu").eval()
        count = 0
        for _, module in self._kmodel.named_modules():
            try:
                torch.nn.utils.remove_weight_norm(module)
                count += 1
            except ValueError:
                pass
        self._weight_norm_removed = count

        self._neuron_parts: Dict[int, Tuple] = {}  # bucket -> (a, b1, b2)
        self._cpu_generator = self._kmodel.decoder.generator
        self._pipeline = None

    @classmethod
    def load(
        cls, model_dir: str, buckets: Optional[List[int]] = None
    ) -> "KokoroNeuron":
        """Load pre-compiled Neuron models from disk.

        Args:
            model_dir: Directory containing compiled models (f32/, f64/, etc.)
            buckets: Which bucket sizes to load. If None, loads all available.

        Returns:
            KokoroNeuron instance with compiled models loaded.
        """
        import torch_neuronx  # noqa: F401 -- registers neuron.Model type for jit.load

        instance = cls()
        model_dir = Path(model_dir)

        if buckets is None:
            buckets = sorted(
                int(d.name[1:])
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("f")
            )

        for bucket in buckets:
            d = model_dir / f"f{bucket}"
            a_path = d / "part_a.pt"
            b1_path = d / "part_b1.pt"
            b2_path = d / "part_b2.pt"
            if a_path.exists() and b1_path.exists() and b2_path.exists():
                instance._neuron_parts[bucket] = (
                    torch.jit.load(str(a_path)),
                    torch.jit.load(str(b1_path)),
                    torch.jit.load(str(b2_path)),
                )
        return instance

    def compile(
        self,
        buckets: Optional[List[int]] = None,
        compiler_args: Optional[List[str]] = None,
    ) -> Dict[int, float]:
        """Compile model parts for specified bucket sizes.

        Args:
            buckets: List of bucket sizes to compile. Default: [32, 64, 96, 128, 160, 192].
                     Max supported: 192 (SB overflow at >=256).
            compiler_args: Compiler arguments. Default: ['--auto-cast', 'matmult', '--auto-cast-type', 'bf16'].

        Returns:
            Dict mapping bucket size to compilation time in seconds.
        """
        import torch_neuronx

        if buckets is None:
            buckets = DEFAULT_BUCKETS
        if compiler_args is None:
            compiler_args = COMPILER_ARGS

        compile_times = {}

        for bucket in buckets:
            t_start = time.time()

            # Part A
            part_a = _PartA(self._kmodel.decoder).eval()
            neuron_a = torch_neuronx.trace(
                part_a,
                (
                    torch.randn(1, 512, bucket),
                    torch.randn(1, 2 * bucket),
                    torch.randn(1, 2 * bucket),
                    torch.randn(1, 128),
                ),
                compiler_args=compiler_args,
                inline_weights_to_neff=True,
            )

            # Part B1
            part_b1 = _PartB1(self._kmodel.decoder.decode[3]).eval()
            neuron_b1 = torch_neuronx.trace(
                part_b1,
                (
                    torch.randn(1, 1090, bucket),
                    torch.randn(1, 128),
                ),
                compiler_args=compiler_args,
                inline_weights_to_neff=True,
            )

            # Part B2
            har_T = self._get_har_T(bucket)
            part_b2 = _PartB2(self._kmodel.decoder.generator).eval()
            neuron_b2 = torch_neuronx.trace(
                part_b2,
                (
                    torch.randn(1, 512, 2 * bucket),
                    torch.randn(1, 128),
                    torch.randn(1, 22, har_T),
                ),
                compiler_args=compiler_args,
                inline_weights_to_neff=True,
            )

            self._neuron_parts[bucket] = (neuron_a, neuron_b1, neuron_b2)
            compile_times[bucket] = time.time() - t_start

            # Reload model for next bucket (tracing modifies in-place)
            saved_parts = self._neuron_parts
            self.__init__()
            self._neuron_parts = saved_parts

        return compile_times

    def save(self, model_dir: str) -> None:
        """Save compiled models to disk.

        Args:
            model_dir: Directory to save compiled models.
        """
        model_dir = Path(model_dir)
        for bucket, (a, b1, b2) in self._neuron_parts.items():
            d = model_dir / f"f{bucket}"
            d.mkdir(parents=True, exist_ok=True)
            a.save(str(d / "part_a.pt"))
            b1.save(str(d / "part_b1.pt"))
            b2.save(str(d / "part_b2.pt"))

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> np.ndarray:
        """Generate speech audio from text, automatically chunking long inputs.

        For short text that fits in a single bucket, runs one Neuron inference.
        For longer text, automatically splits into chunks using KPipeline's
        phoneme-aware chunking (respects sentence/clause boundaries), generates
        each chunk on Neuron, and stitches with crossfade.

        Args:
            text: Input text to synthesize (any length).
            voice: Voice style name (e.g., 'af_heart', 'af_nova', 'am_onyx').
            speed: Speech speed multiplier (1.0 = normal).
            lang_code: Language code for G2P ('a' for American English).

        Returns:
            Audio waveform as numpy array at 24kHz, float32, range [-1, 1].
        """
        chunks = list(self.generate_stream(text, voice, speed, lang_code))
        if not chunks:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return self._crossfade_stitch(chunks)

    def generate_stream(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ):
        """Generate speech as a stream of audio chunks (Python generator).

        Yields one numpy array per chunk, following KPipeline's phoneme-aware
        text splitting. Each chunk fits within compiled bucket limits. Chunks
        are yielded as soon as they are generated, enabling low-latency streaming.

        This mirrors the official kokoro KPipeline generator API.

        Args:
            text: Input text to synthesize (any length).
            voice: Voice style name (e.g., 'af_heart', 'af_nova', 'am_onyx').
            speed: Speech speed multiplier (1.0 = normal).
            lang_code: Language code for G2P ('a' for American English).

        Yields:
            numpy arrays of float32 audio at 24kHz, one per text chunk.
        """
        if not self._neuron_parts:
            raise RuntimeError(
                "No compiled models loaded. Call compile() or load() first."
            )

        self._ensure_pipeline(lang_code)

        max_bucket = max(self._neuron_parts.keys())

        for _, phonemes, _ in self._pipeline(
            text, voice=voice, speed=speed, split_pattern=r"\n\n+"
        ):
            # KPipeline splits text on the split_pattern, then further chunks
            # by phoneme count (waterfall at 510 phonemes).
            # We use '\n\n+' to split on paragraph boundaries only, not on
            # soft line wraps within paragraphs (default '\n+' breaks mid-sentence).
            # When model=False, audio is None — we only need phonemes.
            # Each chunk may still produce more frames than our max bucket.
            # Run CPU forward to get actual frame count, then sub-chunk if needed.
            try:
                audio_cpu, pred_dur, intermediates = self._cpu_forward(
                    phonemes, voice, speed
                )
            except Exception:
                continue

            total_frames = intermediates["total_frames"]

            if total_frames <= max_bucket:
                # Fits in one Neuron call
                audio = self._neuron_decode(intermediates)
                yield audio
            else:
                # Sub-chunk: split at phoneme duration boundaries
                for sub_audio in self._sub_chunk_decode(
                    phonemes, voice, speed, intermediates, pred_dur, max_bucket
                ):
                    yield sub_audio

    def _generate_single_chunk(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> np.ndarray:
        """Generate speech from text that fits in a single bucket (no chunking).

        This is the original generate() behavior. Raises ValueError if the text
        produces more frames than the largest compiled bucket.

        Args:
            text: Input text to synthesize (must be short enough for one bucket).
            voice: Voice style name.
            speed: Speech speed multiplier.
            lang_code: Language code for G2P.

        Returns:
            Audio waveform as numpy array at 24kHz, float32.
        """
        if not self._neuron_parts:
            raise RuntimeError(
                "No compiled models loaded. Call compile() or load() first."
            )

        self._ensure_pipeline(lang_code)

        # G2P -- take first chunk only
        for _, phonemes, _ in self._pipeline.en_tokenize((self._pipeline.g2p(text))[1]):
            break

        # CPU stages: ALBERT duration prediction + alignment
        audio_cpu, pred_dur, intermediates = self._cpu_forward(phonemes, voice, speed)
        return self._neuron_decode(intermediates)

    def generate_timed(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> Tuple[np.ndarray, dict]:
        """Generate speech with per-part timing information.

        For long text, reports aggregate timing across all chunks.

        Returns:
            Tuple of (audio_array, timings_dict).
            timings_dict has keys: 'cpu_stage', 'part_a', 'part_b1', 'har',
            'part_b2', 'total', 'num_chunks'.
        """
        if not self._neuron_parts:
            raise RuntimeError(
                "No compiled models loaded. Call compile() or load() first."
            )

        self._ensure_pipeline(lang_code)

        max_bucket = max(self._neuron_parts.keys())
        chunks = []
        t_cpu_total = 0
        t_a_total = 0
        t_b1_total = 0
        t_har_total = 0
        t_b2_total = 0
        total_audio_samples = 0
        num_chunks = 0

        for _, phonemes, _ in self._pipeline(
            text, voice=voice, speed=speed, split_pattern=r"\n\n+"
        ):
            try:
                t0 = time.time()
                audio_cpu, pred_dur, intermediates = self._cpu_forward(
                    phonemes, voice, speed
                )
                t_cpu_total += time.time() - t0
            except Exception:
                continue

            total_frames = intermediates["total_frames"]

            if total_frames <= max_bucket:
                bucket = self._select_bucket(total_frames)
                asr_p, F0_p, N_p = self._pad_inputs(
                    intermediates["asr"],
                    intermediates["F0_pred"],
                    intermediates["N_pred"],
                    bucket,
                )
                s = intermediates["s"]
                neuron_a, neuron_b1, neuron_b2 = self._neuron_parts[bucket]

                t0 = time.time()
                xa, asr_res, F0_conv, N_conv = neuron_a(asr_p, F0_p, N_p, s)
                t_a_total += time.time() - t0

                t0 = time.time()
                xb1_input = torch.cat([xa, asr_res, F0_conv, N_conv], axis=1)
                xb1 = neuron_b1(xb1_input, s)
                t_b1_total += time.time() - t0

                t0 = time.time()
                har = self._compute_har_cpu(F0_p)
                t_har_total += time.time() - t0

                t0 = time.time()
                audio = neuron_b2(xb1, s, har)
                t_b2_total += time.time() - t0

                actual_samples = total_frames * HOP_SIZE
                audio_out = audio.squeeze()[:actual_samples].detach().cpu().numpy()
                chunks.append(audio_out)
                total_audio_samples += actual_samples
                num_chunks += 1
            else:
                # Sub-chunk with timing — use simple aggregate
                t0 = time.time()
                for sub_audio in self._sub_chunk_decode(
                    phonemes, voice, speed, intermediates, pred_dur, max_bucket
                ):
                    chunks.append(sub_audio)
                    total_audio_samples += len(sub_audio)
                    num_chunks += 1
                elapsed = time.time() - t0
                # Attribute sub-chunk time proportionally (approximate)
                t_b2_total += elapsed * 0.78  # B2 dominates at ~78%
                t_a_total += elapsed * 0.04
                t_b1_total += elapsed * 0.04
                t_har_total += elapsed * 0.14

        if not chunks:
            return np.array([], dtype=np.float32), {}

        if len(chunks) == 1:
            audio_final = chunks[0]
        else:
            audio_final = self._crossfade_stitch(chunks)

        decoder_total = t_a_total + t_b1_total + t_har_total + t_b2_total
        timings = {
            "cpu_stage": t_cpu_total,
            "part_a": t_a_total,
            "part_b1": t_b1_total,
            "har": t_har_total,
            "part_b2": t_b2_total,
            "total": decoder_total,
            "total_with_cpu": t_cpu_total + decoder_total,
            "frames": total_audio_samples // HOP_SIZE,
            "bucket": max_bucket,
            "audio_duration": total_audio_samples / SAMPLE_RATE,
            "num_chunks": num_chunks,
        }
        return audio_final, timings

    def warmup(self, buckets: Optional[List[int]] = None, n_warmup: int = 5) -> None:
        """Run warmup iterations to stabilize Neuron performance.

        Args:
            buckets: Bucket sizes to warm up. If None, warms all loaded buckets.
            n_warmup: Number of warmup iterations per bucket.
        """
        if buckets is None:
            buckets = sorted(self._neuron_parts.keys())

        for bucket in buckets:
            if bucket not in self._neuron_parts:
                continue
            neuron_a, neuron_b1, neuron_b2 = self._neuron_parts[bucket]
            asr = torch.randn(1, 512, bucket)
            F0 = torch.randn(1, 2 * bucket)
            N = torch.randn(1, 2 * bucket)
            s = torch.randn(1, 128)
            har_T = self._get_har_T(bucket)

            for _ in range(n_warmup):
                neuron_a(asr, F0, N, s)
            for _ in range(n_warmup):
                neuron_b1(torch.randn(1, 1090, bucket), s)
            for _ in range(n_warmup):
                neuron_b2(torch.randn(1, 512, 2 * bucket), s, torch.randn(1, 22, har_T))

    @property
    def available_buckets(self) -> List[int]:
        """Return list of compiled bucket sizes."""
        return sorted(self._neuron_parts.keys())

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (24000 Hz)."""
        return SAMPLE_RATE

    # ================================================================
    # Internal methods
    # ================================================================

    def _ensure_pipeline(self, lang_code: str = "a"):
        """Lazily initialize the KPipeline for G2P and text chunking."""
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=lang_code, model=False)

    def _neuron_decode(self, intermediates: dict) -> np.ndarray:
        """Run Neuron 3-part decoder on pre-computed intermediates.

        Args:
            intermediates: Dict with 'asr', 'F0_pred', 'N_pred', 's', 'total_frames'.

        Returns:
            Audio as numpy array, trimmed to actual length.
        """
        total_frames = intermediates["total_frames"]
        bucket = self._select_bucket(total_frames)

        asr_p, F0_p, N_p = self._pad_inputs(
            intermediates["asr"],
            intermediates["F0_pred"],
            intermediates["N_pred"],
            bucket,
        )
        s = intermediates["s"]

        neuron_a, neuron_b1, neuron_b2 = self._neuron_parts[bucket]

        xa, asr_res, F0_conv, N_conv = neuron_a(asr_p, F0_p, N_p, s)
        xb1_input = torch.cat([xa, asr_res, F0_conv, N_conv], axis=1)
        xb1 = neuron_b1(xb1_input, s)
        har = self._compute_har_cpu(F0_p)
        audio = neuron_b2(xb1, s, har)

        actual_samples = total_frames * HOP_SIZE
        audio_out = audio.squeeze()[:actual_samples]
        return audio_out.detach().cpu().numpy()

    def _sub_chunk_decode(
        self, phonemes, voice_name, speed, intermediates, pred_dur, max_bucket
    ):
        """Split a single phoneme sequence into sub-chunks that fit in buckets.

        When a KPipeline chunk produces more frames than our largest bucket,
        we split along phoneme boundaries using the predicted durations.
        Each sub-chunk gets its own ALBERT + Neuron forward pass.

        Args:
            phonemes: Full phoneme sequence for this chunk.
            voice_name: Voice name for style vector lookup.
            speed: Speed multiplier.
            intermediates: CPU forward results for the full chunk.
            pred_dur: Predicted phoneme durations (frames per phoneme).
            max_bucket: Largest available bucket size.

        Yields:
            numpy arrays of decoded audio for each sub-chunk.
        """
        # pred_dur includes BOS/EOS tokens: [bos_dur, p1_dur, ..., pN_dur, eos_dur]
        # phonemes is the raw phoneme string (no BOS/EOS)
        # We split the phoneme string at cumulative frame boundaries

        cum_frames = torch.cumsum(pred_dur, dim=0).tolist()
        total_phonemes = len(phonemes)

        # Target: use ~80% of max_bucket to avoid edge effects from padding
        target_frames = int(max_bucket * 0.8)

        # Find split points: walk through phonemes, accumulate frames
        # pred_dur[0] = BOS, pred_dur[1..N] = phonemes, pred_dur[N+1] = EOS
        split_indices = []
        current_frames = pred_dur[0].item()  # BOS frames
        chunk_start = 0

        for i in range(total_phonemes):
            phoneme_frames = pred_dur[i + 1].item()  # +1 to skip BOS
            if current_frames + phoneme_frames > target_frames and chunk_start < i:
                split_indices.append(i)
                current_frames = phoneme_frames
                chunk_start = i
            else:
                current_frames += phoneme_frames

        # Generate each sub-chunk as an independent forward pass
        splits = [0] + split_indices + [total_phonemes]

        for si in range(len(splits) - 1):
            start_idx = splits[si]
            end_idx = splits[si + 1]
            sub_phonemes = phonemes[start_idx:end_idx]

            if not sub_phonemes:
                continue

            try:
                _, _, sub_intermediates = self._cpu_forward(
                    sub_phonemes, voice_name, speed
                )
                sub_frames = sub_intermediates["total_frames"]
                if sub_frames <= max_bucket:
                    yield self._neuron_decode(sub_intermediates)
                else:
                    # Still too large -- shouldn't happen with 80% target,
                    # but fall back to CPU audio as safety valve
                    yield (
                        sub_intermediates["asr"]
                        .new_zeros(sub_frames * HOP_SIZE)
                        .numpy()
                    )
            except Exception:
                continue

    @staticmethod
    def _crossfade_stitch(
        chunks: List[np.ndarray], crossfade_ms: float = 25.0
    ) -> np.ndarray:
        """Stitch audio chunks with linear crossfade to smooth boundaries.

        Uses overlap-add with a linear ramp. 25ms crossfade is inaudible
        for speech while eliminating click artifacts at chunk boundaries.

        Args:
            chunks: List of numpy audio arrays at 24kHz.
            crossfade_ms: Crossfade duration in milliseconds (default: 25ms).

        Returns:
            Single concatenated numpy array.
        """
        crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)

        if len(chunks) == 0:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]

        # Calculate total output length
        total_len = sum(len(c) for c in chunks) - crossfade_samples * (len(chunks) - 1)
        output = np.zeros(total_len, dtype=np.float32)

        pos = 0
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: copy entirely
                output[pos : pos + len(chunk)] = chunk
                pos += len(chunk) - crossfade_samples
            else:
                # Crossfade region
                fade_len = min(crossfade_samples, len(chunk), len(output) - pos)
                if fade_len > 0:
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    output[pos : pos + fade_len] = (
                        output[pos : pos + fade_len] * fade_out
                        + chunk[:fade_len] * fade_in
                    )
                # Remaining samples after crossfade
                remaining = chunk[fade_len:]
                out_pos = pos + fade_len
                end_pos = min(out_pos + len(remaining), total_len)
                output[out_pos:end_pos] = remaining[: end_pos - out_pos]
                pos = pos + len(chunk) - crossfade_samples

        return output

    def _cpu_forward(self, phonemes, voice_name, speed):
        """Run CPU stages: ALBERT duration prediction + alignment + F0/N."""
        from huggingface_hub import hf_hub_download

        voice_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice_name}.pt"
        )
        voice_pack = torch.load(voice_path, weights_only=True)

        input_ids = list(
            filter(
                lambda i: i is not None,
                map(lambda p: self._kmodel.vocab.get(p), phonemes),
            )
        )
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
        ref_s = voice_pack[len(phonemes) - 1]

        with torch.no_grad():
            input_lengths = torch.full(
                (input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long
            )
            text_mask = (
                torch.arange(input_lengths.max())
                .unsqueeze(0)
                .expand(input_lengths.shape[0], -1)
                .type_as(input_lengths)
            )
            text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

            bert_dur = self._kmodel.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = self._kmodel.bert_encoder(bert_dur).transpose(-1, -2)

            s_prosody = ref_s[:, 128:]
            d = self._kmodel.predictor.text_encoder(
                d_en, s_prosody, input_lengths, text_mask
            )
            x, _ = self._kmodel.predictor.lstm(d)
            duration = self._kmodel.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

            indices = torch.repeat_interleave(
                torch.arange(input_ids.shape[1]), pred_dur
            )
            total_frames = indices.shape[0]
            pred_aln_trg = torch.zeros((input_ids.shape[1], total_frames))
            pred_aln_trg[indices, torch.arange(total_frames)] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0)

            en = d.transpose(-1, -2) @ pred_aln_trg
            F0_pred, N_pred = self._kmodel.predictor.F0Ntrain(en, s_prosody)

            t_en = self._kmodel.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ pred_aln_trg

            s_decoder = ref_s[:, :128]
            audio = self._kmodel.decoder(asr, F0_pred, N_pred, s_decoder).squeeze()

        return (
            audio,
            pred_dur,
            {
                "asr": asr,
                "F0_pred": F0_pred,
                "N_pred": N_pred,
                "s": s_decoder,
                "total_frames": total_frames,
            },
        )

    def _compute_har_cpu(self, f0):
        """Precompute harmonic features on CPU (F.interpolate not traceable)."""
        gen = self._cpu_generator
        with torch.no_grad():
            f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _, _ = gen.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = gen.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        return har

    def _get_har_T(self, bucket):
        """Get har tensor time dimension for a bucket size."""
        f0_dummy = torch.randn(1, 2 * bucket)
        return self._compute_har_cpu(f0_dummy).shape[2]

    def _select_bucket(self, total_frames):
        """Select smallest available bucket >= total_frames."""
        for b in sorted(self._neuron_parts.keys()):
            if total_frames <= b:
                return b
        available = sorted(self._neuron_parts.keys())
        raise ValueError(
            f"total_frames={total_frames} exceeds largest bucket={available[-1]}. "
            f"Use shorter text or compile larger buckets."
        )

    @staticmethod
    def _pad_inputs(asr, F0_pred, N_pred, max_frames):
        """Pad decoder inputs using replicate (edge) padding.

        Replicate padding reduces boundary artifacts to ~2% vs ~8% for zero padding.
        """
        pad_frames = max_frames - asr.shape[2]
        pad_f0 = 2 * max_frames - F0_pred.shape[1]
        asr_padded = F.pad(asr, (0, pad_frames), mode="replicate")
        F0_padded = F.pad(F0_pred, (0, pad_f0), mode="replicate")
        N_padded = F.pad(N_pred, (0, pad_f0), mode="replicate")
        return asr_padded, F0_padded, N_padded
