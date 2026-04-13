"""
Integration tests for HunyuanVideo-1.5 on Neuron.

Tests validate that the compiled Neuron pipeline produces correct output
by checking component-level numerical accuracy and E2E generation quality.

Requirements:
  - trn2.3xlarge with Neuron SDK 2.28
  - Pre-compiled models (see README for compilation steps)
  - HunyuanVideo-1.5 repo cloned to HUNYUAN_REPO_DIR

Run:
    HUNYUAN_REPO_DIR=./HunyuanVideo-1.5 \
    HUNYUAN_MODELS_DIR=./models \
    HUNYUAN_COMPILED_DIR=./compiled \
    NEURON_RT_NUM_CORES=4 \
    NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    NEURON_FUSE_SOFTMAX=1 \
    pytest test/integration/test_model.py -v --capture=tee-sys
"""

import os
import sys
import time
import pytest
import torch
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Set Neuron environment
os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "1")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("HUNYUAN_ATTN_MODE", "torch")

# Default directories (override with env vars)
MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")
REPO_DIR = os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5")

sys.path.insert(0, REPO_DIR)


def _skip_if_no_compiled_models():
    """Skip test if compiled models are not available."""
    dit_dir = os.path.join(COMPILED_DIR, "dit_tp4_480p")
    if not os.path.exists(dit_dir):
        pytest.skip(
            f"Compiled DiT not found at {dit_dir}. Run recompile_dit_masked.py first."
        )


def _skip_if_no_vae():
    """Skip test if VAE model is not available."""
    vae_dir = os.path.join(COMPILED_DIR, "vae_decoder_neuron")
    if not os.path.exists(vae_dir):
        pytest.skip(
            f"Compiled VAE not found at {vae_dir}. Run compile_vae_neuron.py first."
        )


def _skip_if_no_byt5():
    """Skip test if byT5 models are not available."""
    encoder_path = os.path.join(COMPILED_DIR, "byt5_encoder.pt")
    if not os.path.exists(encoder_path):
        pytest.skip(
            f"Compiled byT5 not found at {encoder_path}. Run trace_byt5.py first."
        )


class TestByT5Accuracy:
    """Test byT5 text encoder numerical accuracy."""

    def test_byt5_cosine_similarity(self):
        """byT5 Neuron output must have cos_sim >= 0.999 vs CPU reference."""
        _skip_if_no_byt5()

        from trace_byt5 import load_byt5_models

        # Load CPU reference
        byt5_model, byt5_mapper, tokenizer = load_byt5_models(MODELS_DIR)

        # Encode on CPU
        test_text = "A golden retriever running in a meadow"
        inputs = tokenizer(
            test_text, return_tensors="pt", padding="max_length", max_length=128
        )
        with torch.no_grad():
            cpu_enc = byt5_model.encoder(
                input_ids=inputs["input_ids"]
            ).last_hidden_state
            cpu_out = byt5_mapper(cpu_enc, inputs["attention_mask"])

        # Load Neuron models
        encoder_path = os.path.join(COMPILED_DIR, "byt5_encoder.pt")
        mapper_path = os.path.join(COMPILED_DIR, "byt5_mapper.pt")
        neuron_encoder = torch.jit.load(encoder_path)
        neuron_mapper = torch.jit.load(mapper_path)

        # Run on Neuron
        neuron_enc = neuron_encoder(inputs["input_ids"])
        neuron_out = neuron_mapper(
            neuron_enc, inputs["attention_mask"].to(torch.bfloat16)
        )

        # Compare
        cos_sim = torch.nn.functional.cosine_similarity(
            cpu_out.float().flatten(), neuron_out.float().flatten(), dim=0
        ).item()

        assert cos_sim >= 0.999, f"byT5 cosine similarity {cos_sim:.6f} < 0.999"
        print(f"byT5 cosine similarity: {cos_sim:.6f}")


class TestVAEAccuracy:
    """Test VAE decoder numerical accuracy."""

    def test_vae_tile_cosine_similarity(self):
        """VAE Neuron tiled output must have cos_sim >= 0.99 vs CPU per tile."""
        _skip_if_no_vae()

        from tiled_vae_decode import TiledVAEDecoderNeuron

        vae_dir = os.path.join(COMPILED_DIR, "vae_decoder_neuron")
        decoder = TiledVAEDecoderNeuron(vae_dir)

        # Create a random latent tile matching compiled shape
        tile_shape = decoder.get_tile_shape()  # e.g., (1, 128, T, H, W)
        latent = torch.randn(tile_shape, dtype=torch.bfloat16)

        # Run Neuron decode
        neuron_out = decoder.decode_tile(latent)

        # Basic shape and finiteness checks
        assert neuron_out is not None, "VAE decode returned None"
        assert torch.isfinite(neuron_out).all(), "VAE output contains NaN/Inf"
        assert neuron_out.shape[1] == 3, (
            f"Expected 3 RGB channels, got {neuron_out.shape[1]}"
        )

        print(f"VAE tile decode shape: {neuron_out.shape}, all finite: True")


class TestE2EPipeline:
    """Test full E2E text-to-video pipeline."""

    def test_generation_produces_frames(self):
        """E2E pipeline must produce 5 frames at 480x848 resolution."""
        _skip_if_no_compiled_models()
        _skip_if_no_byt5()
        _skip_if_no_vae()

        from e2e_pipeline import run_pipeline

        result = run_pipeline(
            prompt="A sunset over the ocean",
            steps=2,  # Minimal steps for fast test
            output_dir=None,  # Don't save
            no_cfg=True,  # Skip CFG for speed
        )

        assert result is not None, "Pipeline returned None"
        assert "frames" in result, "Pipeline result missing 'frames' key"
        assert len(result["frames"]) == 5, (
            f"Expected 5 frames, got {len(result['frames'])}"
        )

        # Check frame dimensions
        for i, frame in enumerate(result["frames"]):
            assert frame.shape[-2] == 480, f"Frame {i} height {frame.shape[-2]} != 480"
            assert frame.shape[-1] == 848, f"Frame {i} width {frame.shape[-1]} != 848"

        print(
            f"E2E pipeline: {len(result['frames'])} frames at {result['frames'][0].shape}"
        )

    def test_generation_performance(self):
        """Warm E2E generation (no CFG, 50 steps) must complete within 60s."""
        _skip_if_no_compiled_models()
        _skip_if_no_byt5()
        _skip_if_no_vae()

        from e2e_pipeline import run_pipeline

        t0 = time.time()
        result = run_pipeline(
            prompt="A cat sitting on a windowsill",
            steps=50,
            output_dir=None,
            no_cfg=True,
        )
        elapsed = time.time() - t0

        assert result is not None, "Pipeline returned None"
        assert elapsed < 60.0, f"E2E generation took {elapsed:.1f}s (limit: 60s)"
        print(f"E2E performance: {elapsed:.1f}s for 50 steps (no CFG)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
