# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for LeVo 2 (SongGeneration v2) on Neuron.

Tests validate each pipeline stage independently against CPU reference outputs
using cosine similarity (GPT2, VAE) and end-to-end audio quality metrics.

Supports both v2-medium and v2-large via the LEVO2_VARIANT environment variable.

Requirements:
    - trn2.3xlarge instance with Neuron SDK 2.28
    - Model weights at paths specified by env vars or defaults
    - codeclm source repository at CODECLM_PATH

Environment variables:
    LEVO2_VARIANT: "v2-medium" (default) or "v2-large"
    LEVO2_MODEL_PATH: Path to model.pt
    LEVO2_CONFIG_PATH: Path to config.yaml
    LEVO2_SAFETENSORS_PATH: Path to model_2.safetensors (shared)
    LEVO2_PROMPT_PATH: Path to encode-s12k.pt (prompt audio)
    LEVO2_COMPILED_DIR: Path to pre-compiled models (skip compilation if set)
    CODECLM_PATH: Path to codeclm source (default: /mnt/models/songgeneration)

Usage:
    # Full test v2-medium (compile + run):
    LEVO2_VARIANT=v2-medium pytest test/integration/test_model.py -v --timeout=3600

    # Full test v2-large:
    LEVO2_VARIANT=v2-large pytest test/integration/test_model.py -v --timeout=3600

    # With pre-compiled models:
    LEVO2_COMPILED_DIR=/mnt/models/levo/compiled pytest test/integration/test_model.py -v
"""

import os
import sys
import time

import pytest
import torch
import torch.nn.functional as F
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ============================================================================
# Configuration from environment
# ============================================================================

VARIANT = os.environ.get("LEVO2_VARIANT", "v2-medium")

# Default paths per variant
if VARIANT == "v2-large":
    _DEFAULT_MODEL = "/mnt/models/levo/weights/v2-large/model.pt"
    _DEFAULT_CONFIG = "/mnt/models/levo/weights/v2-large/config.yaml"
else:
    _DEFAULT_MODEL = "/mnt/models/levo/weights/v2-medium/model.pt"
    _DEFAULT_CONFIG = "/mnt/models/levo/weights/v2-medium/config.yaml"

MODEL_PATH = os.environ.get("LEVO2_MODEL_PATH", _DEFAULT_MODEL)
CONFIG_PATH = os.environ.get("LEVO2_CONFIG_PATH", _DEFAULT_CONFIG)
SAFETENSORS_PATH = os.environ.get(
    "LEVO2_SAFETENSORS_PATH",
    "/mnt/models/levo/runtime/ckpt/model_septoken/model_2.safetensors",
)
PROMPT_PATH = os.environ.get(
    "LEVO2_PROMPT_PATH",
    "/mnt/models/levo/runtime/ckpt/encode-s12k.pt",
)
COMPILED_DIR = os.environ.get("LEVO2_COMPILED_DIR", None)
CODECLM_PATH = os.environ.get("CODECLM_PATH", "/mnt/models/songgeneration")

DURATION_SEC = 5.0
T_FRAMES = int(DURATION_SEC * 25)  # 125 frames

# English lyrics for testing
TEST_LYRICS = (
    "[intro-short] ; "
    "[verse] Sunlight breaks through morning haze."
    "Golden fields stretch far away."
    "Rivers flow with gentle grace."
    "Finding peace in nature's embrace ; "
    "[chorus] Sing along.Let the music carry you home."
    "Sing along.You were never meant to walk alone ; "
    "[outro-short]"
)

TEST_DESCRIPTIONS = "female, pop, upbeat, piano and acoustic guitar, the bpm is 120"


# ============================================================================
# Helper: load English prompts
# ============================================================================


def load_english_prompts(prompt_path):
    """Load language-aware prompt file and extract English prompts."""
    data = torch.load(prompt_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict) and "en" in first_val:
            return {g: data[g]["en"] for g in data if "en" in data[g]}
        else:
            return data
    return data


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def pipeline():
    """Build or load the LeVo 2 Neuron pipeline."""
    from modeling_levo2 import LeVo2Neuron, LeVo2Config

    if VARIANT == "v2-large":
        config = LeVo2Config.v2_large(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            safetensors_path=SAFETENSORS_PATH,
            prompt_path=PROMPT_PATH,
            codeclm_path=CODECLM_PATH,
            default_duration_sec=DURATION_SEC,
        )
    else:
        config = LeVo2Config.v2_medium(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            safetensors_path=SAFETENSORS_PATH,
            prompt_path=PROMPT_PATH,
            codeclm_path=CODECLM_PATH,
            default_duration_sec=DURATION_SEC,
        )

    model = LeVo2Neuron(config)

    if COMPILED_DIR and os.path.isdir(COMPILED_DIR):
        model.load(COMPILED_DIR)
    else:
        model.compile()

    # Override with English prompts
    prompt_data = torch.load(
        os.path.join(CODECLM_PATH, "tools/new_auto_prompt.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model._prompt_data = {
        g: prompt_data[g]["en"]
        for g in prompt_data
        if isinstance(prompt_data[g], dict) and "en" in prompt_data[g]
    }

    model.warmup()
    return model


@pytest.fixture(scope="module")
def gpt2_cpu_reference():
    """Generate CPU reference output for GPT2 diffusion backbone."""
    sys.path.insert(0, CODECLM_PATH)
    sys.path.insert(0, os.path.join(CODECLM_PATH, "codeclm/tokenizer/Flow1dVAE"))
    sys.path.insert(
        0,
        os.path.join(CODECLM_PATH, "codeclm/tokenizer/Flow1dVAE/models_gpt/models"),
    )

    from safetensors.torch import load_file
    from gpt2_config import GPT2Config
    from gpt2_rope2_time_new_correct_mask_noncasual_reflow import (
        GPT2Model as OrigGPT2Model,
    )

    gpt2_config = GPT2Config(
        n_positions=1000,
        n_layer=16,
        n_head=20,
        n_embd=2200,
        n_inner=4400,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    gpt2_config._attn_implementation = "eager"

    full_sd = load_file(SAFETENSORS_PATH)
    gpt2_sd = {
        k[len("cfm_wrapper.estimator.") :]: v
        for k, v in full_sd.items()
        if k.startswith("cfm_wrapper.estimator.")
    }

    model = OrigGPT2Model(gpt2_config)
    model.load_state_dict(gpt2_sd, strict=False)
    model.eval()

    torch.manual_seed(42)
    B, T = 2, T_FRAMES
    inputs = torch.randn(B, T, 2200)
    mask = torch.ones(B, 1, T, T)
    timestep = torch.tensor([0.5, 0.5])

    with torch.no_grad():
        output = model(
            inputs_embeds=inputs, attention_mask=mask, time_step=timestep
        ).last_hidden_state

    return {
        "inputs": inputs,
        "mask": mask,
        "timestep": timestep,
        "output": output,
    }


@pytest.fixture(scope="module")
def vae_cpu_reference():
    """Generate CPU reference output for VAE decoder."""
    sys.path.insert(0, os.path.join(CODECLM_PATH, "codeclm/tokenizer/Flow1dVAE"))
    from tools.get_1dvae_large import get_model

    vae_config = os.path.join(
        os.path.dirname(SAFETENSORS_PATH), "../vae/stable_audio_1920_vae.json"
    )
    vae_weights = os.path.join(
        os.path.dirname(SAFETENSORS_PATH), "../vae/autoencoder_music_1320k.ckpt"
    )

    vae = get_model(vae_config, vae_weights)
    vae.eval()

    for name, module in vae.named_modules():
        if hasattr(module, "weight_g"):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

    torch.manual_seed(42)
    latents = torch.randn(1, 64, T_FRAMES)

    with torch.no_grad():
        output = vae.decode_audio(latents)

    return {"latents": latents, "output": output}


# ============================================================================
# Test Classes
# ============================================================================


class TestCompilation:
    """Verify model compiles and loads correctly."""

    def test_pipeline_compiled(self, pipeline):
        """Pipeline should report as compiled."""
        assert pipeline._compiled

    def test_primary_loaded(self, pipeline):
        """Primary transformer should be loaded on Neuron."""
        assert pipeline._primary_neuron is not None

    def test_secondary_loaded(self, pipeline):
        """Secondary transformer should be loaded on Neuron."""
        assert pipeline._secondary_neuron is not None

    def test_gpt2_loaded(self, pipeline):
        """GPT2 diffusion model should be loaded on Neuron."""
        assert pipeline._neuron_gpt2 is not None

    def test_vae_loaded(self, pipeline):
        """VAE decoder should be loaded on Neuron."""
        assert pipeline._neuron_vae is not None

    def test_prompt_data_loaded(self, pipeline):
        """Prompt data should be loaded with at least one genre."""
        assert pipeline._prompt_data is not None
        assert len(pipeline._prompt_data) > 0
        assert "Pop" in pipeline._prompt_data

    def test_correct_variant(self, pipeline):
        """Pipeline variant should match LEVO2_VARIANT env var."""
        assert pipeline.config.variant == VARIANT


class TestGPT2Accuracy:
    """Validate GPT2 diffusion backbone accuracy vs CPU reference."""

    def test_gpt2_cosine_similarity(self, pipeline, gpt2_cpu_reference):
        """GPT2 Neuron output should have cosine similarity > 0.98 vs CPU."""
        ref = gpt2_cpu_reference
        neuron_output = pipeline._neuron_gpt2(
            ref["inputs"], ref["mask"], ref["timestep"]
        )
        cos_sim = F.cosine_similarity(
            ref["output"].flatten().unsqueeze(0),
            neuron_output.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.98, f"GPT2 cosine similarity {cos_sim:.6f} < 0.98"

    def test_gpt2_max_relative_error(self, pipeline, gpt2_cpu_reference):
        """GPT2 p99 relative error should be < 5%."""
        ref = gpt2_cpu_reference
        neuron_output = pipeline._neuron_gpt2(
            ref["inputs"], ref["mask"], ref["timestep"]
        )
        rel_error = (ref["output"] - neuron_output).abs() / (ref["output"].abs() + 1e-8)
        p99_rel = torch.quantile(rel_error.float(), 0.99).item()
        assert p99_rel < 0.05, f"GPT2 p99 relative error {p99_rel:.4f} > 0.05"


class TestVAEAccuracy:
    """Validate VAE decoder accuracy vs CPU reference."""

    def test_vae_cosine_similarity(self, pipeline, vae_cpu_reference):
        """VAE Neuron output should have cosine similarity > 0.98 vs CPU."""
        ref = vae_cpu_reference
        neuron_output = pipeline._neuron_vae(ref["latents"])
        cos_sim = F.cosine_similarity(
            ref["output"].flatten().unsqueeze(0),
            neuron_output.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.98, f"VAE cosine similarity {cos_sim:.6f} < 0.98"

    def test_vae_output_shape(self, pipeline, vae_cpu_reference):
        """VAE output shape should be [1, 2, T_frames * 1920]."""
        ref = vae_cpu_reference
        neuron_output = pipeline._neuron_vae(ref["latents"])
        expected_samples = T_FRAMES * 1920
        assert neuron_output.shape == (1, 2, expected_samples), (
            f"VAE shape {neuron_output.shape} != (1, 2, {expected_samples})"
        )

    def test_vae_signal_to_noise(self, pipeline, vae_cpu_reference):
        """VAE SNR vs CPU should be > 20 dB."""
        ref = vae_cpu_reference
        neuron_output = pipeline._neuron_vae(ref["latents"])
        signal_power = (ref["output"] ** 2).mean()
        noise_power = ((ref["output"] - neuron_output) ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()
        assert snr_db > 20, f"VAE SNR {snr_db:.1f} dB < 20 dB"


class TestE2EGeneration:
    """End-to-end generation tests with English lyrics."""

    def test_generates_audio(self, pipeline):
        """Pipeline should generate non-zero audio tensor."""
        audio, sr = pipeline.generate(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=42,
        )
        assert audio is not None
        assert sr == 48000
        assert audio.shape[0] >= 1  # batch
        assert audio.shape[1] == 2  # stereo
        assert audio.shape[2] > 0  # non-empty

    def test_audio_valid_range(self, pipeline):
        """Audio values should be in reasonable range."""
        audio, _ = pipeline.generate(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=123,
        )
        assert audio.abs().max() < 10.0, "Audio values out of range"
        assert audio.std() > 1e-6, "Audio is silent (zero std)"

    def test_timed_generation(self, pipeline):
        """generate_timed should return timing breakdown."""
        result = pipeline.generate_timed(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=99,
        )
        assert "audio" in result
        assert "sample_rate" in result
        assert "timings" in result
        t = result["timings"]
        assert t["lelm_s"] > 0
        assert t["diffusion_s"] > 0
        assert t["vae_s"] > 0
        assert t["total_s"] > 0

    def test_audio_rms_healthy(self, pipeline):
        """Audio RMS should indicate non-trivial content (not silence or buzz)."""
        audio, sr = pipeline.generate(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=42,
        )
        audio_np = audio.float().cpu().numpy().squeeze(0)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        peak = max(abs(audio_np.max()), abs(audio_np.min()), 1e-10)
        audio_int16 = (audio_np / peak * 32767).astype(np.int16)
        rms = np.sqrt(np.mean(audio_int16.astype(float) ** 2))
        assert rms > 1000, f"Audio RMS {rms:.0f} too low (likely silent)"


class TestPerformance:
    """Performance benchmarks."""

    def test_lelm_step_latency(self, pipeline):
        """LeLM combined step latency should be within expected range."""
        result = pipeline.generate_timed(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=42,
        )
        t = result["timings"]
        ms_per_step = (t["lelm_s"] * 1000) / t["lelm_steps"]
        # v2-medium: ~56ms/step, v2-large: ~97ms/step
        if VARIANT == "v2-large":
            threshold = 120
        else:
            threshold = 70
        assert ms_per_step < threshold, (
            f"LeLM step latency {ms_per_step:.1f}ms > {threshold}ms target"
        )

    def test_total_time_reasonable(self, pipeline):
        """Total E2E time for 5s audio should be within expected range."""
        result = pipeline.generate_timed(
            lyrics=TEST_LYRICS,
            descriptions=TEST_DESCRIPTIONS,
            genre="Pop",
            duration_sec=DURATION_SEC,
            seed=42,
        )
        total = result["timings"]["total_s"]
        # v2-medium: ~22s for 5s, v2-large: ~37s for 5s
        if VARIANT == "v2-large":
            threshold = 60
        else:
            threshold = 40
        assert total < threshold, (
            f"Total time {total:.1f}s > {threshold}s for {DURATION_SEC}s audio"
        )


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"LeVo 2 ({VARIANT}) Neuron Integration Tests (standalone)")
    print("=" * 70)

    from modeling_levo2 import LeVo2Neuron, LeVo2Config

    if VARIANT == "v2-large":
        config = LeVo2Config.v2_large(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            safetensors_path=SAFETENSORS_PATH,
            prompt_path=PROMPT_PATH,
            codeclm_path=CODECLM_PATH,
            default_duration_sec=DURATION_SEC,
        )
    else:
        config = LeVo2Config.v2_medium(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            safetensors_path=SAFETENSORS_PATH,
            prompt_path=PROMPT_PATH,
            codeclm_path=CODECLM_PATH,
            default_duration_sec=DURATION_SEC,
        )

    print(f"\nVariant: {VARIANT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Duration: {DURATION_SEC}s")

    print("\n[1/5] Building pipeline...")
    model = LeVo2Neuron(config)
    if COMPILED_DIR and os.path.isdir(COMPILED_DIR):
        model.load(COMPILED_DIR)
    else:
        model.compile()

    # Load English prompts
    prompt_data = torch.load(
        os.path.join(CODECLM_PATH, "tools/new_auto_prompt.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model._prompt_data = {
        g: prompt_data[g]["en"]
        for g in prompt_data
        if isinstance(prompt_data[g], dict) and "en" in prompt_data[g]
    }
    model.warmup()
    print("  PASS: Pipeline compiled and loaded")

    print("\n[2/5] Testing GPT2 accuracy...")
    sys.path.insert(0, CODECLM_PATH)
    sys.path.insert(0, os.path.join(CODECLM_PATH, "codeclm/tokenizer/Flow1dVAE"))
    sys.path.insert(
        0,
        os.path.join(CODECLM_PATH, "codeclm/tokenizer/Flow1dVAE/models_gpt/models"),
    )
    from safetensors.torch import load_file
    from gpt2_config import GPT2Config
    from gpt2_rope2_time_new_correct_mask_noncasual_reflow import (
        GPT2Model as OrigGPT2,
    )

    gpt2_cfg = GPT2Config(
        n_positions=1000,
        n_layer=16,
        n_head=20,
        n_embd=2200,
        n_inner=4400,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    gpt2_cfg._attn_implementation = "eager"
    full_sd = load_file(SAFETENSORS_PATH)
    gpt2_sd = {
        k[len("cfm_wrapper.estimator.") :]: v
        for k, v in full_sd.items()
        if k.startswith("cfm_wrapper.estimator.")
    }
    orig_gpt2 = OrigGPT2(gpt2_cfg)
    orig_gpt2.load_state_dict(gpt2_sd, strict=False)
    orig_gpt2.eval()

    torch.manual_seed(42)
    test_input = torch.randn(2, T_FRAMES, 2200)
    test_mask = torch.ones(2, 1, T_FRAMES, T_FRAMES)
    test_ts = torch.tensor([0.5, 0.5])

    with torch.no_grad():
        cpu_out = orig_gpt2(
            inputs_embeds=test_input, attention_mask=test_mask, time_step=test_ts
        ).last_hidden_state
    neuron_out = model._neuron_gpt2(test_input, test_mask, test_ts)
    cos_sim = F.cosine_similarity(
        cpu_out.flatten().unsqueeze(0), neuron_out.flatten().unsqueeze(0)
    ).item()
    print(
        f"  GPT2 cosine similarity: {cos_sim:.6f} "
        f"{'PASS' if cos_sim > 0.98 else 'FAIL'}"
    )

    print("\n[3/5] Testing VAE accuracy...")
    from tools.get_1dvae_large import get_model as get_vae

    vae_config_p = os.path.join(
        os.path.dirname(SAFETENSORS_PATH), "../vae/stable_audio_1920_vae.json"
    )
    vae_weights_p = os.path.join(
        os.path.dirname(SAFETENSORS_PATH), "../vae/autoencoder_music_1320k.ckpt"
    )
    vae = get_vae(vae_config_p, vae_weights_p)
    vae.eval()
    for _, m in vae.named_modules():
        if hasattr(m, "weight_g"):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass

    torch.manual_seed(42)
    test_latents = torch.randn(1, 64, T_FRAMES)
    with torch.no_grad():
        cpu_vae = vae.decode_audio(test_latents)
    neuron_vae = model._neuron_vae(test_latents)
    cos_sim_vae = F.cosine_similarity(
        cpu_vae.flatten().unsqueeze(0), neuron_vae.flatten().unsqueeze(0)
    ).item()
    sig_pow = (cpu_vae**2).mean()
    noise_pow = ((cpu_vae - neuron_vae) ** 2).mean()
    snr = 10 * torch.log10(sig_pow / (noise_pow + 1e-10)).item()
    print(
        f"  VAE cosine similarity: {cos_sim_vae:.6f} "
        f"{'PASS' if cos_sim_vae > 0.98 else 'FAIL'}"
    )
    print(f"  VAE SNR: {snr:.1f} dB {'PASS' if snr > 20 else 'FAIL'}")

    print("\n[4/5] Testing E2E generation...")
    result = model.generate_timed(
        lyrics=TEST_LYRICS,
        descriptions=TEST_DESCRIPTIONS,
        genre="Pop",
        duration_sec=DURATION_SEC,
        seed=42,
    )
    audio = result["audio"]
    t = result["timings"]
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"  Audio std: {audio.std():.6f}")

    audio_np = audio.float().cpu().numpy().squeeze(0)
    if audio_np.ndim > 1:
        audio_np_mono = audio_np.mean(axis=0)
    else:
        audio_np_mono = audio_np
    peak = max(abs(audio_np_mono.max()), abs(audio_np_mono.min()), 1e-10)
    audio_int16 = (audio_np_mono / peak * 32767).astype(np.int16)
    rms = np.sqrt(np.mean(audio_int16.astype(float) ** 2))
    print(f"  Audio RMS: {rms:.0f}")
    print(
        f"  {'PASS' if audio.std() > 1e-6 and rms > 1000 else 'FAIL'}: Audio is valid"
    )

    print("\n[5/5] Performance results...")
    ms_per_step = (t["lelm_s"] * 1000) / t["lelm_steps"]
    print(
        f"  LeLM: {t['lelm_s']:.1f}s ({t['lelm_steps']} steps, "
        f"{ms_per_step:.1f} ms/step)"
    )
    print(f"  Diffusion: {t['diffusion_s']:.3f}s")
    print(f"  VAE: {t['vae_s']:.3f}s")
    print(f"  Total: {t['total_s']:.1f}s")
    print(f"  RTF: {t['total_s'] / DURATION_SEC:.2f}x")

    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)
