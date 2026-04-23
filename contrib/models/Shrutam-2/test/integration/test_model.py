# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Shrutam-2 on Neuron.

Tests validate:
  1. Conformer encoder: numerical accuracy vs CPU reference (neuron_allclose)
  2. SMEAR-MoE projector: numerical accuracy vs CPU reference (neuron_allclose)
  3. LLM decoder: generation quality and throughput
  4. End-to-end pipeline: transcription produces valid multilingual output

Requirements:
  - trn2.3xlarge (LNC=2, 4 logical NeuronCores)
  - bharatgenai/Shrutam-2 weights downloaded and extracted:
    - encoder.pt: Conformer encoder weights
    - model.pt: Downsampler + SMEAR weights
    - llm/ directory: Llama decoder (config.json, model.safetensors, tokenizer)
  - Pre-traced NEFFs (or will be traced during test, ~10 min):
    - encoder_neuron.pt: Traced Conformer encoder
    - smear_neuron.pt: Traced SMEAR-MoE projector
  - Pre-compiled LLM (or will be compiled during test, ~15 min):
    - compiled/shrutam2_decoder_tp1/: NxDI compiled Llama decoder
  - Neuron SDK 2.29, NxDI 0.9.x
  - soundfile (for audio I/O)

Usage:
    # Set paths before running
    export SHRUTAM2_ENCODER_WEIGHTS=/mnt/models/encoder.pt
    export SHRUTAM2_MODEL_WEIGHTS=/mnt/models/model.pt
    export SHRUTAM2_LLM_PATH=/mnt/models/Shrutam-2-hf/llm
    export SHRUTAM2_ENCODER_NEFF=/mnt/models/encoder_neuron.pt
    export SHRUTAM2_SMEAR_NEFF=/mnt/models/smear_neuron.pt
    export SHRUTAM2_LLM_COMPILED=/mnt/models/compiled/shrutam2_decoder_tp1

    pytest test_model.py -v --timeout=900
"""

import math
import os
import sys
import time
import logging

import pytest
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Paths from environment (with defaults matching standard layout)
ENCODER_WEIGHTS = os.environ.get("SHRUTAM2_ENCODER_WEIGHTS", "/mnt/models/encoder.pt")
MODEL_WEIGHTS = os.environ.get("SHRUTAM2_MODEL_WEIGHTS", "/mnt/models/model.pt")
LLM_PATH = os.environ.get("SHRUTAM2_LLM_PATH", "/mnt/models/Shrutam-2-hf/llm")
ENCODER_NEFF = os.environ.get("SHRUTAM2_ENCODER_NEFF", "/mnt/models/encoder_neuron.pt")
SMEAR_NEFF = os.environ.get("SHRUTAM2_SMEAR_NEFF", "/mnt/models/smear_neuron.pt")
LLM_COMPILED = os.environ.get(
    "SHRUTAM2_LLM_COMPILED", "/mnt/models/compiled/shrutam2_decoder_tp1"
)
TEST_AUDIO_DIR = os.environ.get("SHRUTAM2_TEST_AUDIO", "/mnt/models/test_audio")

# Add contrib src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Audio constants
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
ENCODER_DIM = 1024
TRACED_AUDIO_SECONDS = 10.0


def _skip_if_no_weights():
    if not os.path.isfile(ENCODER_WEIGHTS):
        pytest.skip(f"Encoder weights not found at {ENCODER_WEIGHTS}")
    if not os.path.isfile(MODEL_WEIGHTS):
        pytest.skip(f"Model weights not found at {MODEL_WEIGHTS}")


def _skip_if_no_neffs():
    if not os.path.isfile(ENCODER_NEFF):
        pytest.skip(f"Encoder NEFF not found at {ENCODER_NEFF}")
    if not os.path.isfile(SMEAR_NEFF):
        pytest.skip(f"SMEAR NEFF not found at {SMEAR_NEFF}")


def _skip_if_no_llm():
    if not os.path.isdir(LLM_PATH):
        pytest.skip(f"LLM path not found at {LLM_PATH}")
    if not os.path.isdir(LLM_COMPILED):
        pytest.skip(f"Compiled LLM not found at {LLM_COMPILED}")


def _generate_synthetic_mel(batch_size=1, audio_seconds=10.0):
    """Generate synthetic mel spectrogram for testing."""
    n_samples = int(audio_seconds * SAMPLE_RATE)
    T_mel = n_samples // HOP_LENGTH + 1
    mel = torch.randn(batch_size, N_MELS, T_mel)
    mel_lengths = torch.full((batch_size,), T_mel, dtype=torch.long)
    return mel, mel_lengths


def _generate_synthetic_audio(duration_s=5.0, sample_rate=16000):
    """Generate synthetic sine-wave audio for testing."""
    t = torch.linspace(0, duration_s, int(duration_s * sample_rate))
    wav = (
        0.3 * torch.sin(2 * math.pi * 200 * t)
        + 0.2 * torch.sin(2 * math.pi * 440 * t)
        + 0.1 * torch.sin(2 * math.pi * 800 * t)
        + 0.05 * torch.randn_like(t)
    )
    return wav.unsqueeze(0)


# ============================================================
# Test: Conformer Encoder Accuracy
# ============================================================


class TestConformerEncoder:
    """Validate Conformer encoder numerical accuracy: Neuron vs CPU."""

    @pytest.fixture(scope="class")
    def cpu_encoder(self):
        _skip_if_no_weights()
        from modeling_shrutam2 import (
            load_encoder_weights,
            load_downsampler_weights,
            ConformerEncoderForTrace,
        )

        encoder = load_encoder_weights(ENCODER_WEIGHTS)
        downsampler = load_downsampler_weights(MODEL_WEIGHTS)
        model = ConformerEncoderForTrace(encoder, downsampler)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def neuron_encoder(self):
        _skip_if_no_neffs()
        import torch_neuronx

        model = torch.jit.load(ENCODER_NEFF)
        torch_neuronx.async_load(model)
        return model

    def test_encoder_accuracy_neuron_allclose(self, cpu_encoder, neuron_encoder):
        """Compare encoder output: Neuron vs CPU using neuron_allclose-style check.

        Validates that the traced Conformer encoder produces numerically close
        outputs to the CPU reference for a synthetic mel spectrogram input.
        Uses cosine similarity > 0.99 and relative tolerance checks.
        """
        mel, mel_lengths = _generate_synthetic_mel(batch_size=1, audio_seconds=10.0)

        with torch.no_grad():
            cpu_out = cpu_encoder(mel, mel_lengths)
            neuron_out = neuron_encoder(mel, mel_lengths)

        # Shape check
        assert cpu_out.shape == neuron_out.shape, (
            f"Shape mismatch: CPU {cpu_out.shape} vs Neuron {neuron_out.shape}"
        )

        # Cosine similarity (global)
        cpu_flat = cpu_out.flatten().float()
        neuron_flat = neuron_out.flatten().float()
        cos_sim = F.cosine_similarity(
            cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)
        ).item()

        log.info(f"Encoder cosine similarity: {cos_sim:.6f}")
        assert cos_sim > 0.99, (
            f"Encoder cosine similarity too low: {cos_sim:.6f} (expected > 0.99)"
        )

        # Element-wise relative error
        abs_diff = (cpu_flat - neuron_flat).abs()
        max_abs_err = abs_diff.max().item()
        mean_abs_err = abs_diff.mean().item()
        log.info(f"Encoder max abs error: {max_abs_err:.6f}, mean: {mean_abs_err:.6f}")

        # Relative error on non-zero elements
        nonzero_mask = cpu_flat.abs() > 1e-6
        if nonzero_mask.any():
            rel_err = abs_diff[nonzero_mask] / cpu_flat[nonzero_mask].abs()
            max_rel_err = rel_err.max().item()
            log.info(f"Encoder max relative error: {max_rel_err:.4f}")
            assert max_rel_err < 0.5, (
                f"Encoder max relative error too high: {max_rel_err:.4f}"
            )

    def test_encoder_latency(self, neuron_encoder):
        """Verify encoder inference latency is within expected range."""
        mel, mel_lengths = _generate_synthetic_mel(batch_size=1, audio_seconds=10.0)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                neuron_encoder(mel, mel_lengths)

        # Measure
        n_runs = 20
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                neuron_encoder(mel, mel_lengths)
        avg_ms = (time.time() - t0) / n_runs * 1000

        log.info(f"Encoder latency: {avg_ms:.2f} ms")
        # On trn2.3xlarge LNC=2, expect ~9ms. Use generous threshold.
        assert avg_ms < 50, f"Encoder latency too high: {avg_ms:.2f} ms (expected < 50)"


# ============================================================
# Test: SMEAR-MoE Projector Accuracy
# ============================================================


class TestSMEARProjector:
    """Validate SMEAR-MoE projector numerical accuracy: Neuron vs CPU."""

    @pytest.fixture(scope="class")
    def cpu_smear(self):
        _skip_if_no_weights()
        from modeling_shrutam2 import build_smear, load_smear_weights, SMEARForTrace

        smear = build_smear()
        smear = load_smear_weights(smear, MODEL_WEIGHTS)
        wrapper = SMEARForTrace(smear)
        wrapper.eval()
        return wrapper

    @pytest.fixture(scope="class")
    def neuron_smear(self):
        _skip_if_no_neffs()
        import torch_neuronx

        model = torch.jit.load(SMEAR_NEFF)
        torch_neuronx.async_load(model)
        return model

    def test_smear_accuracy_neuron_allclose(self, cpu_smear, neuron_smear):
        """Compare SMEAR output: Neuron vs CPU using cosine similarity.

        The SMEAR projector maps encoder features (1024-dim) to LLM embeddings
        (2048-dim) via utterance-level soft MoE routing.
        """
        example_input = torch.randn(1, 126, ENCODER_DIM)

        with torch.no_grad():
            cpu_out = cpu_smear(example_input)
            neuron_out = neuron_smear(example_input)

        assert cpu_out.shape == neuron_out.shape, (
            f"Shape mismatch: CPU {cpu_out.shape} vs Neuron {neuron_out.shape}"
        )

        cpu_flat = cpu_out.flatten().float()
        neuron_flat = neuron_out.flatten().float()
        cos_sim = F.cosine_similarity(
            cpu_flat.unsqueeze(0), neuron_flat.unsqueeze(0)
        ).item()

        log.info(f"SMEAR cosine similarity: {cos_sim:.6f}")
        assert cos_sim > 0.99, (
            f"SMEAR cosine similarity too low: {cos_sim:.6f} (expected > 0.99)"
        )

        abs_diff = (cpu_flat - neuron_flat).abs()
        max_abs_err = abs_diff.max().item()
        mean_abs_err = abs_diff.mean().item()
        log.info(f"SMEAR max abs error: {max_abs_err:.6f}, mean: {mean_abs_err:.6f}")

    def test_smear_latency(self, neuron_smear):
        """Verify SMEAR inference latency is within expected range."""
        example_input = torch.randn(1, 126, ENCODER_DIM)

        with torch.no_grad():
            for _ in range(3):
                neuron_smear(example_input)

        n_runs = 50
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                neuron_smear(example_input)
        avg_ms = (time.time() - t0) / n_runs * 1000

        log.info(f"SMEAR latency: {avg_ms:.2f} ms")
        # On trn2.3xlarge LNC=2, expect ~1.6ms. Use generous threshold.
        assert avg_ms < 20, f"SMEAR latency too high: {avg_ms:.2f} ms (expected < 20)"


# ============================================================
# Test: LLM Decoder
# ============================================================


class TestLLMDecoder:
    """Validate LLM decoder generation quality."""

    @pytest.fixture(scope="class")
    def llm_adapter(self):
        _skip_if_no_llm()
        from modeling_shrutam2 import build_llm_model
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )

        model, _ = build_llm_model(LLM_PATH, tp_degree=1, batch_size=1)
        model.load(LLM_COMPILED)
        adapter = HuggingFaceGenerationAdapter(model)
        return adapter

    @pytest.fixture(scope="class")
    def tokenizer(self):
        _skip_if_no_llm()
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(LLM_PATH)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        return tok

    def test_llm_text_generation(self, llm_adapter, tokenizer):
        """Verify LLM can generate coherent text (text-only, no audio)."""
        prompt = "The capital of India is"
        inputs = tokenizer(prompt, return_tensors="pt")

        output_ids = llm_adapter.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False,
        )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        num_tokens = (new_tokens != tokenizer.pad_token_id).sum().item()

        log.info(f"LLM output ({num_tokens} tokens): {text}")
        assert num_tokens > 0, "No tokens generated"
        assert len(text.strip()) > 0, "Empty output text"

    def test_llm_decode_throughput(self, llm_adapter, tokenizer):
        """Verify LLM decode throughput meets minimum threshold."""
        prompt = "Hello, how are you doing today?"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Warmup
        llm_adapter.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            do_sample=False,
        )

        # Measure
        t0 = time.time()
        output_ids = llm_adapter.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False,
        )
        gen_time = time.time() - t0

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        num_tokens = (new_tokens != tokenizer.pad_token_id).sum().item()
        tok_per_s = num_tokens / gen_time if gen_time > 0 else 0

        log.info(
            f"LLM throughput: {tok_per_s:.1f} tok/s ({num_tokens} tokens in {gen_time:.2f}s)"
        )
        # On trn2.3xlarge TP=1, expect ~113 tok/s. Use generous threshold.
        assert tok_per_s > 50, (
            f"LLM throughput too low: {tok_per_s:.1f} tok/s (expected > 50)"
        )


# ============================================================
# Test: End-to-End Pipeline
# ============================================================


class TestEndToEndPipeline:
    """Validate the full three-stage ASR pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        _skip_if_no_weights()
        _skip_if_no_neffs()
        _skip_if_no_llm()

        from modeling_shrutam2 import Shrutam2Pipeline

        p = Shrutam2Pipeline(
            encoder_neff_path=ENCODER_NEFF,
            smear_neff_path=SMEAR_NEFF,
            llm_compiled_path=LLM_COMPILED,
            llm_path=LLM_PATH,
            tp_degree=1,
            batch_size=1,
            seq_len=2048,
            n_positions=4096,
            lnc=2,
        )
        return p

    def test_pipeline_synthetic_audio(self, pipeline):
        """Verify pipeline produces non-empty output for synthetic audio."""
        wav = _generate_synthetic_audio(duration_s=5.0)

        result = pipeline.transcribe_tensor(
            wav,
            prompt="Transcribe speech to Hindi text.",
            max_new_tokens=50,
        )

        assert result["num_tokens"] > 0, "No tokens generated"
        assert result["encoder_ms"] < 50, (
            f"Encoder too slow: {result['encoder_ms']:.1f}ms"
        )
        assert result["smear_ms"] < 20, f"SMEAR too slow: {result['smear_ms']:.1f}ms"

        log.info(
            f"Pipeline result: {result['num_tokens']} tokens, "
            f"encoder={result['encoder_ms']:.1f}ms, "
            f"smear={result['smear_ms']:.1f}ms, "
            f"gen={result['gen_time_s']:.2f}s, "
            f"text='{result['text'][:100]}'"
        )

    def test_pipeline_real_audio(self, pipeline):
        """Test pipeline with real FLEURS audio if available."""
        # Look for a Hindi test audio file
        test_files = []
        if os.path.isdir(TEST_AUDIO_DIR):
            for f in os.listdir(TEST_AUDIO_DIR):
                if f.endswith(".wav") and f.startswith("hi_"):
                    test_files.append(os.path.join(TEST_AUDIO_DIR, f))

        if not test_files:
            pytest.skip("No test audio files found")

        audio_path = sorted(test_files)[0]
        log.info(f"Testing with: {audio_path}")

        result = pipeline.transcribe(
            audio_path,
            prompt="Transcribe speech to Hindi text.",
            max_new_tokens=200,
            repetition_penalty=1.3,
        )

        assert result["num_tokens"] > 0, "No tokens generated"
        assert len(result["text"].strip()) > 0, "Empty transcription"

        # Check for garbled output (CJK characters in Hindi transcription)
        import re

        cjk_ratio = len(re.findall(r"[\u4e00-\u9fff]", result["text"])) / max(
            len(result["text"]), 1
        )
        assert cjk_ratio < 0.05, (
            f"Output appears garbled (CJK ratio: {cjk_ratio:.2%}): "
            f"{result['text'][:100]}"
        )

        # Check for excessive repetition
        words = result["text"].split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            assert unique_ratio > 0.15, (
                f"Excessive repetition (unique ratio: {unique_ratio:.2%}): "
                f"{result['text'][:100]}"
            )

        log.info(
            f"Real audio result: '{result['text'][:200]}' "
            f"({result['num_tokens']} tokens, {result['tok_per_s']:.1f} tok/s)"
        )

    def test_pipeline_throughput(self, pipeline):
        """Verify end-to-end throughput meets minimum threshold."""
        wav = _generate_synthetic_audio(duration_s=5.0)

        # Warmup
        pipeline.transcribe_tensor(wav, max_new_tokens=20)

        # Measure
        result = pipeline.transcribe_tensor(
            wav,
            prompt="Transcribe speech to Hindi text.",
            max_new_tokens=100,
        )

        audio_duration = wav.shape[-1] / SAMPLE_RATE
        total_time = result["total_time_s"]
        real_time_factor = audio_duration / total_time if total_time > 0 else 0

        log.info(
            f"E2E throughput: {result['tok_per_s']:.1f} tok/s, "
            f"RTF: {real_time_factor:.1f}x, "
            f"total: {total_time:.2f}s for {audio_duration:.1f}s audio"
        )

        # On trn2.3xlarge, expect ~30x real-time. Use generous threshold.
        assert result["tok_per_s"] > 30, (
            f"Throughput too low: {result['tok_per_s']:.1f} tok/s (expected > 30)"
        )
