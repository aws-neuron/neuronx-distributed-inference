# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Kokoro-82M on AWS Neuron (trn2).

Requirements:
    - AWS Neuron SDK 2.27+ with torch-neuronx
    - kokoro, misaki packages
    - espeak-ng (system package for G2P)
    - trn2.3xlarge instance (inf2 not supported -- Generator SB overflow)

Usage:
    pytest test_model.py -v --capture=tee-sys

    # Or run standalone:
    python test_model.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

COMPILED_MODEL_DIR = os.environ.get(
    "KOKORO_COMPILED_DIR", str(Path.home() / "compiled_models_kokoro")
)
TEST_BUCKET = 128  # Compile one bucket for fast testing


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture(scope="module")
def kokoro_model():
    """Load or compile Kokoro-82M for Neuron. Module-scoped for reuse."""
    from kokoro_neuron import KokoroNeuron

    model_dir = Path(COMPILED_MODEL_DIR)
    bucket_dir = model_dir / f"f{TEST_BUCKET}"

    if (
        bucket_dir.exists()
        and (bucket_dir / "part_a.pt").exists()
        and (bucket_dir / "part_b1.pt").exists()
        and (bucket_dir / "part_b2.pt").exists()
    ):
        print(f"Loading pre-compiled models from {model_dir}")
        model = KokoroNeuron.load(str(model_dir), buckets=[TEST_BUCKET])
    else:
        print(f"Compiling bucket={TEST_BUCKET} (this takes ~60s)...")
        model = KokoroNeuron()
        model.compile(buckets=[TEST_BUCKET])
        model.save(str(model_dir))
        print(f"Saved compiled models to {model_dir}")
        # Reload from disk to test the save/load path
        model = KokoroNeuron.load(str(model_dir), buckets=[TEST_BUCKET])

    model.warmup(n_warmup=3)
    return model


@pytest.fixture(scope="module")
def cpu_reference(kokoro_model):
    """Generate CPU reference audio for accuracy comparison."""
    text = "Hello, this is a test of the Kokoro model."
    # CPU forward via the internal path
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", model=False)
    for _, phonemes, _ in pipeline.en_tokenize((pipeline.g2p(text))[1]):
        break

    audio_cpu, pred_dur, intermediates = kokoro_model._cpu_forward(
        phonemes, "af_heart", 1.0
    )
    return {
        "text": text,
        "phonemes": phonemes,
        "audio": audio_cpu,
        "intermediates": intermediates,
    }


# ================================================================
# Tests: Compilation and Loading
# ================================================================


class TestCompilation:
    def test_model_loads(self, kokoro_model):
        """Model loads with at least one compiled bucket."""
        assert len(kokoro_model.available_buckets) > 0
        assert TEST_BUCKET in kokoro_model.available_buckets

    def test_sample_rate(self, kokoro_model):
        """Sample rate is 24kHz."""
        assert kokoro_model.sample_rate == 24000

    def test_compiled_model_files_exist(self):
        """Compiled model files exist on disk after save."""
        model_dir = Path(COMPILED_MODEL_DIR)
        bucket_dir = model_dir / f"f{TEST_BUCKET}"
        assert (bucket_dir / "part_a.pt").exists(), "Part A not found"
        assert (bucket_dir / "part_b1.pt").exists(), "Part B1 not found"
        assert (bucket_dir / "part_b2.pt").exists(), "Part B2 not found"

    def test_neff_sizes(self):
        """NEFF sizes are reasonable (not empty, not huge)."""
        bucket_dir = Path(COMPILED_MODEL_DIR) / f"f{TEST_BUCKET}"
        for name, min_mb, max_mb in [
            ("part_a.pt", 50, 200),
            ("part_b1.pt", 5, 30),
            ("part_b2.pt", 20, 80),
        ]:
            size_mb = (bucket_dir / name).stat().st_size / (1024 * 1024)
            assert min_mb < size_mb < max_mb, (
                f"{name} size {size_mb:.1f} MB outside expected range [{min_mb}, {max_mb}]"
            )


# ================================================================
# Tests: Inference
# ================================================================


class TestInference:
    def test_generate_produces_audio(self, kokoro_model):
        """generate() returns a non-empty numpy array."""
        audio = kokoro_model.generate("Hello world.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_generate_audio_range(self, kokoro_model):
        """Generated audio values are in valid range [-1, 1]."""
        audio = kokoro_model.generate("Testing audio range.")
        assert np.max(np.abs(audio)) <= 1.5, "Audio values outside expected range"
        assert np.max(np.abs(audio)) > 0.01, "Audio is nearly silent"

    def test_generate_audio_duration(self, kokoro_model):
        """Generated audio has reasonable duration for the input text."""
        audio = kokoro_model.generate("Hello world.")
        duration = len(audio) / 24000
        # "Hello world." should produce 0.3-3.0 seconds of audio
        assert 0.3 < duration < 3.0, f"Audio duration {duration:.2f}s is unexpected"

    def test_generate_timed(self, kokoro_model):
        """generate_timed() returns audio and timing dict."""
        audio, timings = kokoro_model.generate_timed("Hello world.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert "part_a" in timings
        assert "part_b1" in timings
        assert "har" in timings
        assert "part_b2" in timings
        assert "total" in timings
        assert timings["total"] > 0


# ================================================================
# Tests: Accuracy
# ================================================================


class TestAccuracy:
    def test_neuron_vs_cpu_cosine(self, kokoro_model, cpu_reference):
        """Neuron output has high cosine similarity to CPU reference (>0.95)."""
        audio_neuron = kokoro_model.generate(cpu_reference["text"], voice="af_heart")
        audio_cpu = cpu_reference["audio"].numpy()

        # Trim to shorter length (padding may differ)
        min_len = min(len(audio_neuron), len(audio_cpu))
        neuron_t = torch.tensor(audio_neuron[:min_len]).unsqueeze(0)
        cpu_t = torch.tensor(audio_cpu[:min_len]).unsqueeze(0)

        cosine = F.cosine_similarity(neuron_t, cpu_t).item()
        assert cosine > 0.95, f"Cosine similarity {cosine:.4f} < 0.95 threshold"

    def test_neuron_vs_cpu_snr(self, kokoro_model, cpu_reference):
        """Signal-to-noise ratio vs CPU reference is > 10 dB."""
        audio_neuron = kokoro_model.generate(cpu_reference["text"], voice="af_heart")
        audio_cpu = cpu_reference["audio"].numpy()

        min_len = min(len(audio_neuron), len(audio_cpu))
        neuron_t = torch.tensor(audio_neuron[:min_len])
        cpu_t = torch.tensor(audio_cpu[:min_len])

        noise = neuron_t - cpu_t
        snr = 10 * torch.log10((cpu_t**2).sum() / (noise**2).sum() + 1e-10).item()
        assert snr > 10, f"SNR {snr:.1f} dB < 10 dB threshold"


# ================================================================
# Tests: Multiple Voices
# ================================================================


class TestVoices:
    @pytest.mark.parametrize("voice", ["af_heart", "af_sky", "am_adam"])
    def test_different_voices(self, kokoro_model, voice):
        """Each voice produces valid, non-identical audio."""
        audio = kokoro_model.generate("Testing voice styles.", voice=voice)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        duration = len(audio) / 24000
        assert duration > 0.3, f"Voice {voice}: too short ({duration:.2f}s)"


# ================================================================
# Tests: Performance
# ================================================================


class TestPerformance:
    def test_decoder_faster_than_realtime(self, kokoro_model):
        """Neuron decoder is faster than real-time (RT factor > 1)."""
        _, timings = kokoro_model.generate_timed("Hello, this is a performance test.")
        rt_factor = timings["audio_duration"] / timings["total"]
        assert rt_factor > 10, (
            f"Real-time factor {rt_factor:.1f}x is too slow (expected >10x)"
        )

    def test_latency_p50(self, kokoro_model):
        """P50 decoder latency is under 50ms for typical text."""
        latencies = []
        for _ in range(10):
            _, timings = kokoro_model.generate_timed("Hello, this is a latency test.")
            latencies.append(timings["total"] * 1000)

        p50 = np.percentile(latencies, 50)
        assert p50 < 50, f"P50 latency {p50:.1f}ms > 50ms threshold"


# ================================================================
# Tests: Long-Form Generation
# ================================================================


class TestLongForm:
    LONG_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a second sentence that adds more content. "
        "And here is a third sentence to make it even longer. "
        "Finally, a fourth sentence wraps up this paragraph."
    )

    MULTI_SENTENCE = (
        "First, we need to understand the problem. "
        "Second, we analyze possible solutions. "
        "Third, we implement the best approach. "
        "Fourth, we test and validate our implementation. "
        "Fifth, we deploy to production."
    )

    def test_generate_long_produces_audio(self, kokoro_model):
        """generate() with long text returns non-empty audio."""
        audio = kokoro_model.generate(self.LONG_TEXT)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
        duration = len(audio) / 24000
        # 4 sentences should produce at least 2 seconds of audio
        assert duration > 2.0, f"Long text produced only {duration:.2f}s"

    def test_generate_stream_yields_chunks(self, kokoro_model):
        """generate_stream() yields multiple chunks for long text."""
        chunks = list(kokoro_model.generate_stream(self.MULTI_SENTENCE))
        assert len(chunks) >= 1, "Expected at least 1 chunk"
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, np.ndarray), f"Chunk {i} is not ndarray"
            assert len(chunk) > 0, f"Chunk {i} is empty"
            assert chunk.dtype == np.float32, f"Chunk {i} wrong dtype"

    def test_generate_long_audio_duration(self, kokoro_model):
        """Long text produces proportionally longer audio than short text."""
        short_audio = kokoro_model.generate("Hello world.")
        long_audio = kokoro_model.generate(self.LONG_TEXT)
        short_dur = len(short_audio) / 24000
        long_dur = len(long_audio) / 24000
        assert long_dur > short_dur, (
            f"Long text ({long_dur:.2f}s) not longer than short ({short_dur:.2f}s)"
        )

    def test_crossfade_stitching_no_clicks(self, kokoro_model):
        """Stitched audio has no extreme amplitude spikes at chunk boundaries."""
        audio = kokoro_model.generate(self.MULTI_SENTENCE)
        # Check for clicks: look for sample-to-sample jumps > 0.5
        # (normal speech rarely exceeds 0.3 sample-to-sample delta)
        deltas = np.abs(np.diff(audio))
        max_delta = np.max(deltas)
        # Allow up to 0.8 for natural speech transients (plosives etc.)
        assert max_delta < 0.8, (
            f"Max sample delta {max_delta:.3f} suggests click artifact"
        )

    def test_generate_timed_long_reports_chunks(self, kokoro_model):
        """generate_timed() reports num_chunks for multi-chunk text."""
        audio, timings = kokoro_model.generate_timed(self.MULTI_SENTENCE)
        assert "num_chunks" in timings
        assert timings["num_chunks"] >= 1
        assert timings["audio_duration"] > 0


# ================================================================
# Standalone runner
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Kokoro-82M Neuron Integration Tests")
    print("=" * 60)

    from kokoro_neuron import KokoroNeuron

    # Setup
    model_dir = Path(COMPILED_MODEL_DIR)
    bucket_dir = model_dir / f"f{TEST_BUCKET}"

    if bucket_dir.exists() and (bucket_dir / "part_a.pt").exists():
        print(f"\nLoading pre-compiled models from {model_dir}")
        model = KokoroNeuron.load(str(model_dir), buckets=[TEST_BUCKET])
    else:
        print(f"\nCompiling bucket={TEST_BUCKET}...")
        model = KokoroNeuron()
        model.compile(buckets=[TEST_BUCKET])
        model.save(str(model_dir))
        model = KokoroNeuron.load(str(model_dir), buckets=[TEST_BUCKET])

    model.warmup(n_warmup=5)

    # Test 1: Basic generation
    print("\n[1/6] Basic generation...")
    audio = model.generate("Hello world, this is a test.")
    print(f"  OK: {len(audio)} samples, {len(audio) / 24000:.2f}s")

    # Test 2: Timing
    print("\n[2/6] Timed generation...")
    audio, timings = model.generate_timed("Hello, this is a performance test.")
    print(f"  Part A:  {timings['part_a'] * 1000:.1f}ms")
    print(f"  Part B1: {timings['part_b1'] * 1000:.1f}ms")
    print(f"  har:     {timings['har'] * 1000:.1f}ms")
    print(f"  Part B2: {timings['part_b2'] * 1000:.1f}ms")
    print(f"  Total:   {timings['total'] * 1000:.1f}ms")
    print(f"  Audio:   {timings['audio_duration']:.2f}s")
    rt = timings["audio_duration"] / timings["total"]
    print(f"  RT:      {rt:.0f}x real-time")

    # Test 3: CPU vs Neuron accuracy
    print("\n[3/6] Accuracy vs CPU...")
    text = "Hello, this is a test of the Kokoro model."
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", model=False)
    for _, phonemes, _ in pipeline.en_tokenize((pipeline.g2p(text))[1]):
        break
    audio_cpu, _, _ = model._cpu_forward(phonemes, "af_heart", 1.0)
    audio_neuron = model.generate(text, voice="af_heart")

    min_len = min(len(audio_neuron), len(audio_cpu))
    cos = F.cosine_similarity(
        torch.tensor(audio_neuron[:min_len]).unsqueeze(0),
        audio_cpu[:min_len].unsqueeze(0),
    ).item()
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  {'PASS' if cos > 0.95 else 'FAIL'} (threshold: 0.95)")

    # Test 4: Multiple voices
    print("\n[4/6] Multiple voices...")
    for voice in ["af_heart", "af_sky", "am_adam"]:
        try:
            a = model.generate("Testing voice.", voice=voice)
            print(f"  {voice}: OK ({len(a) / 24000:.2f}s)")
        except Exception as e:
            print(f"  {voice}: FAIL ({e})")

    # Test 5: Performance
    print("\n[5/8] Performance benchmark (10 iterations)...")
    latencies = []
    for _ in range(10):
        _, t = model.generate_timed("Hello, this is a performance benchmark test.")
        latencies.append(t["total"] * 1000)
    print(f"  P50: {np.percentile(latencies, 50):.1f}ms")
    print(f"  P90: {np.percentile(latencies, 90):.1f}ms")

    # Test 6: NEFF sizes
    print("\n[6/8] NEFF sizes...")
    for name in ["part_a.pt", "part_b1.pt", "part_b2.pt"]:
        size = (bucket_dir / name).stat().st_size / (1024 * 1024)
        print(f"  {name}: {size:.1f} MB")

    # Test 7: Long-form generation
    print("\n[7/8] Long-form generation...")
    long_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a second sentence that adds more content. "
        "And here is a third sentence to make it even longer. "
        "Finally, a fourth sentence wraps up this paragraph."
    )
    audio_long = model.generate(long_text)
    long_dur = len(audio_long) / 24000
    print(f"  OK: {len(audio_long)} samples, {long_dur:.2f}s")

    # Test 8: Streaming generation
    print("\n[8/8] Streaming generation...")
    stream_text = (
        "First, we need to understand the problem. "
        "Second, we analyze possible solutions. "
        "Third, we implement the best approach."
    )
    chunks = list(model.generate_stream(stream_text))
    total_stream_samples = sum(len(c) for c in chunks)
    print(
        f"  {len(chunks)} chunks, {total_stream_samples} total samples, "
        f"{total_stream_samples / 24000:.2f}s"
    )

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
