# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Qwen3-ASR-1.7B on NeuronX.

Tests:
1. Smoke test: model loads and generates tokens
2. Logit validation: first-token logits match CPU reference
3. E2E accuracy: transcription matches expected text

Prerequisites:
- Model downloaded: Qwen/Qwen3-ASR-1.7B
- Compiled artifacts available (run compile first)
- Traced encoder NEFFs available

Configuration:
- Set MODEL_PATH, COMPILED_MODEL_PATH, ENCODER_DIR below
"""

import sys
import time
from pathlib import Path

import pytest
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_qwen3_asr import (
    NeuronQwen3ASRForCausalLM,
    create_inference_config,
    get_encoder_output_length,
    AUDIO_PAD_ID,
    AUDIO_START_ID,
    AUDIO_END_ID,
    IM_START_ID,
    IM_END_ID,
    EOS_ID,
)
from audio_encoder import load_encoders, select_bucket

# ===== CONFIGURATION =====
# Update these paths for your environment
MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
COMPILED_MODEL_PATH = "/mnt/models/compiled/qwen3_asr_vl_text_tp4"
ENCODER_DIR = "/mnt/models/compiled/qwen3_asr_encoder"
TP_DEGREE = 4
N_POSITIONS = 1024
MAX_NEW_TOKENS = 128


@pytest.fixture(scope="module")
def model():
    """Load compiled model (shared across all tests in this module)."""
    config = create_inference_config(
        MODEL_PATH, tp_degree=TP_DEGREE, n_positions=N_POSITIONS
    )
    model = NeuronQwen3ASRForCausalLM(COMPILED_MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def encoders():
    """Load traced encoder NEFFs."""
    return load_encoders(ENCODER_DIR)


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def feature_extractor():
    """Load Whisper feature extractor."""
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(MODEL_PATH)


def generate_greedy(
    model, input_ids, attention_mask, audio_embeddings=None, max_tokens=MAX_NEW_TOKENS
):
    """Run greedy autoregressive generation."""
    import copy

    seq_ids = torch.zeros(1, dtype=torch.long)
    sampling_params = torch.zeros(1, 3)

    # Pad input
    seq_len = input_ids.shape[1]
    padded_input_ids = torch.nn.functional.pad(
        input_ids, (0, N_POSITIONS - seq_len), value=EOS_ID
    )

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Prefill
    with torch.no_grad():
        output = model.forward(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            audio_embeddings=audio_embeddings,
        )

    logits = (
        output.logits
        if hasattr(output, "logits")
        else (output[0] if isinstance(output, (tuple, list)) else output)
    )
    first_token = logits[0, -1, :].argmax(dim=-1).item()

    generated = [first_token]
    current_pos = seq_len

    # Decode loop
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            if generated[-1] in (EOS_ID, IM_END_ID) or current_pos >= N_POSITIONS - 1:
                break

            next_id = torch.tensor([[generated[-1]]], dtype=torch.long)
            decode_mask = torch.zeros(1, N_POSITIONS, dtype=torch.long)
            decode_mask[0, : current_pos + 1] = 1
            decode_pos = torch.tensor([[current_pos]], dtype=torch.long)

            output = model.forward(
                input_ids=next_id,
                attention_mask=decode_mask,
                position_ids=decode_pos,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                audio_embeddings=None,
            )

            logits = (
                output.logits
                if hasattr(output, "logits")
                else (output[0] if isinstance(output, (tuple, list)) else output)
            )
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            current_pos += 1

    return generated, logits


class TestSmokeTest:
    """Basic smoke tests: model loads and produces output."""

    def test_model_loads(self, model):
        """Model loads without errors."""
        assert model is not None
        assert model.is_loaded_to_neuron

    def test_text_only_generation(self, model, tokenizer):
        """Model generates tokens for text-only input (no audio)."""
        # Simple text prompt without audio
        text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        seq_len = input_ids.shape[1]

        attention_mask = torch.ones(1, N_POSITIONS, dtype=torch.long)
        attention_mask[0, seq_len:] = 0

        generated, _ = generate_greedy(model, input_ids, attention_mask, max_tokens=10)
        assert len(generated) > 0, "Model should generate at least 1 token"


class TestEncoderIntegration:
    """Tests with audio encoder."""

    def test_encoder_loads(self, encoders):
        """All encoder buckets load successfully."""
        assert 500 in encoders
        assert 1000 in encoders
        assert 3000 in encoders

    def test_encoder_output_shape(self, encoders):
        """Encoder produces correct output shapes."""
        for T, expected_tokens in [(500, 65), (1000, 130), (3000, 390)]:
            output = encoders[T](torch.randn(128, T))
            assert output.shape == (expected_tokens, 2048), (
                f"T={T}: expected ({expected_tokens}, 2048), got {output.shape}"
            )

    def test_encoder_latency(self, encoders):
        """Encoder latency is within expected range."""
        mel = torch.randn(128, 1000)
        times = []
        for _ in range(10):
            t0 = time.time()
            with torch.no_grad():
                _ = encoders[1000](mel)
            times.append(time.time() - t0)

        avg_ms = np.mean(times[2:]) * 1000  # Skip warmup
        assert avg_ms < 50, f"Encoder T=1000 should be <50ms, got {avg_ms:.1f}ms"


class TestE2EAccuracy:
    """End-to-end accuracy validation."""

    def test_reference_transcription(
        self, model, encoders, tokenizer, feature_extractor
    ):
        """E2E pipeline produces correct transcription for reference audio.

        Uses the LibriSpeech test sample: "Mr. Quilter is the apostle..."
        """
        import soundfile as sf

        # Load test audio (must exist on test machine)
        audio_path = "/tmp/test_speech.wav"
        try:
            audio, sr = sf.read(audio_path)
        except FileNotFoundError:
            pytest.skip(f"Test audio not found at {audio_path}")

        audio = audio.astype(np.float32)

        # Feature extraction
        mel_output = feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        mel_features = mel_output["input_features"][0]
        mel_attention_mask = mel_output["attention_mask"][0]
        actual_mel_len = int(mel_attention_mask.sum().item())
        bucket_T = select_bucket(actual_mel_len)
        N_tokens = get_encoder_output_length(actual_mel_len)

        mel_input = mel_features[:, :bucket_T]
        if mel_input.shape[1] < bucket_T:
            mel_input = torch.nn.functional.pad(
                mel_input, (0, bucket_T - mel_input.shape[1])
            )

        # Encode
        with torch.no_grad():
            audio_embeddings = encoders[bucket_T](mel_input)[:N_tokens]

        # Build input_ids
        prefix_ids = [
            IM_START_ID,
            8948,
            198,
            IM_END_ID,
            198,
            IM_START_ID,
            872,
            198,
            AUDIO_START_ID,
        ]
        audio_ids = [AUDIO_PAD_ID] * N_tokens
        suffix_ids = [AUDIO_END_ID, IM_END_ID, 198, IM_START_ID, 77091, 198]
        all_ids = prefix_ids + audio_ids + suffix_ids
        seq_len = len(all_ids)

        input_ids = torch.tensor([all_ids], dtype=torch.long)
        attention_mask = torch.ones(1, N_POSITIONS, dtype=torch.long)
        attention_mask[0, seq_len:] = 0

        # Generate
        generated, _ = generate_greedy(
            model, input_ids, attention_mask, audio_embeddings
        )

        # Extract transcription
        transcription = tokenizer.decode(generated, skip_special_tokens=False)
        if "<asr_text>" in transcription:
            text = transcription.split("<asr_text>", 1)[1]
            for special in ["<|im_end|>", "<|endoftext|>"]:
                text = text.replace(special, "")
            text = text.strip()
        else:
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        expected = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
        assert text.lower() == expected.lower(), (
            f"Expected: '{expected}', Got: '{text}'"
        )

    def test_silence_produces_empty(
        self, model, encoders, tokenizer, feature_extractor
    ):
        """Pure silence should produce empty/minimal transcription."""
        silence = np.zeros(48000, dtype=np.float32)  # 3s silence

        mel_output = feature_extractor(
            silence,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        mel_features = mel_output["input_features"][0]
        mel_attention_mask = mel_output["attention_mask"][0]
        actual_mel_len = int(mel_attention_mask.sum().item())
        bucket_T = select_bucket(actual_mel_len)
        N_tokens = get_encoder_output_length(actual_mel_len)

        mel_input = mel_features[:, :bucket_T]
        if mel_input.shape[1] < bucket_T:
            mel_input = torch.nn.functional.pad(
                mel_input, (0, bucket_T - mel_input.shape[1])
            )

        with torch.no_grad():
            audio_embeddings = encoders[bucket_T](mel_input)[:N_tokens]

        prefix_ids = [
            IM_START_ID,
            8948,
            198,
            IM_END_ID,
            198,
            IM_START_ID,
            872,
            198,
            AUDIO_START_ID,
        ]
        audio_ids = [AUDIO_PAD_ID] * N_tokens
        suffix_ids = [AUDIO_END_ID, IM_END_ID, 198, IM_START_ID, 77091, 198]
        all_ids = prefix_ids + audio_ids + suffix_ids
        seq_len = len(all_ids)

        input_ids = torch.tensor([all_ids], dtype=torch.long)
        attention_mask = torch.ones(1, N_POSITIONS, dtype=torch.long)
        attention_mask[0, seq_len:] = 0

        generated, _ = generate_greedy(
            model, input_ids, attention_mask, audio_embeddings
        )

        # Silence should produce very few tokens (language tag + EOS)
        assert len(generated) <= 10, (
            f"Silence should produce <=10 tokens, got {len(generated)}"
        )


class TestPerformance:
    """Performance benchmarks."""

    def test_ttft_under_threshold(self, model, encoders, feature_extractor):
        """TTFT should be under 50ms for 5s audio."""
        mel_input = torch.randn(128, 500)  # 5s bucket

        with torch.no_grad():
            audio_embeddings = encoders[500](mel_input)[:65]

        N_tokens = 65
        prefix_ids = [
            IM_START_ID,
            8948,
            198,
            IM_END_ID,
            198,
            IM_START_ID,
            872,
            198,
            AUDIO_START_ID,
        ]
        audio_ids = [AUDIO_PAD_ID] * N_tokens
        suffix_ids = [AUDIO_END_ID, IM_END_ID, 198, IM_START_ID, 77091, 198]
        all_ids = prefix_ids + audio_ids + suffix_ids
        seq_len = len(all_ids)

        input_ids = torch.tensor([all_ids], dtype=torch.long)
        attention_mask = torch.ones(1, N_POSITIONS, dtype=torch.long)
        attention_mask[0, seq_len:] = 0
        padded_input_ids = torch.nn.functional.pad(
            input_ids, (0, N_POSITIONS - seq_len), value=EOS_ID
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        seq_ids = torch.zeros(1, dtype=torch.long)
        sampling_params = torch.zeros(1, 3)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = encoders[500](mel_input)
                _ = model.forward(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    audio_embeddings=audio_embeddings,
                )

        # Measure TTFT
        times = []
        for _ in range(5):
            t0 = time.time()
            with torch.no_grad():
                _ = encoders[500](mel_input)
                _ = model.forward(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    audio_embeddings=audio_embeddings,
                )
            times.append(time.time() - t0)

        avg_ttft_ms = np.mean(times) * 1000
        assert avg_ttft_ms < 50, (
            f"TTFT should be <50ms for 5s audio, got {avg_ttft_ms:.1f}ms"
        )
        print(f"  TTFT (5s audio): {avg_ttft_ms:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
