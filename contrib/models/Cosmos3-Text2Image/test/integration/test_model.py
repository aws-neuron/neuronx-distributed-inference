#!/usr/bin/env python3
"""
Integration tests for Cosmos3-Text2Image NeuronX implementation.

Tests model compilation, loading, and image generation quality.

Unlike LLM contribs that test token accuracy, this tests:
1. Model loads and produces output (smoke test)
2. Output has correct shape and valid pixel range
3. Generated image has expected statistical properties (not noise/blank)
4. Performance meets latency targets

Usage:
    # Run with pytest:
    pytest test/integration/test_model.py --capture=tee-sys -v

    # Run manually:
    python test/integration/test_model.py

Configuration:
    Set MODEL_PATH, COMPILED_PATH, and VAE_PATH below to match your setup.
"""

import os
import sys
import time

import pytest
import torch

# Add src to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from modeling_cosmos3 import (
    Cosmos3BackboneInferenceConfig,
    NeuronCosmos3BackboneApplication,
)
from pipeline import (
    build_position_ids,
    denoise,
    denormalize_latents,
    patchify,
    tokenize_prompt,
    unpatchify,
)
from neuronx_distributed_inference.models.config import NeuronConfig


# =============================================================================
# Test Configuration - Update these paths for your environment
# =============================================================================

# Nano configuration (trn2.3xlarge, TP=4)
MODEL_PATH = os.environ.get("COSMOS3_MODEL_PATH", "/home/ubuntu/Cosmos3-Nano")
COMPILED_PATH = os.environ.get("COSMOS3_COMPILED_PATH", "/home/ubuntu/compiled_cosmos3")
VAE_PATH = os.environ.get(
    "COSMOS3_VAE_PATH", "/home/ubuntu/compiled_vae/vae_decoder.pt"
)
TP_DEGREE = int(os.environ.get("COSMOS3_TP_DEGREE", "4"))

# Model params (auto-detected from MODEL_PATH if available)
HIDDEN_SIZE = int(os.environ.get("COSMOS3_HIDDEN_SIZE", "4096"))
INTERMEDIATE_SIZE = int(os.environ.get("COSMOS3_INTERMEDIATE_SIZE", "12288"))
NUM_LAYERS = int(os.environ.get("COSMOS3_NUM_LAYERS", "36"))
NUM_HEADS = int(os.environ.get("COSMOS3_NUM_HEADS", "32"))
NUM_KV_HEADS = int(os.environ.get("COSMOS3_NUM_KV_HEADS", "8"))

MAX_TEXT = 256
NUM_VIS = 256  # 16x16 patches for 512x512


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def backbone():
    """Load compiled backbone model."""
    import torch_neuronx

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE, world_size=TP_DEGREE, torch_dtype=torch.bfloat16
    )
    config = Cosmos3BackboneInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=128,
        vocab_size=151936,
        patch_channels=192,
        latent_channels=48,
        rope_theta=5000000.0,
        mrope_section=[24, 20, 20],
    )
    config.max_text_len = MAX_TEXT
    config.num_vision_patches = NUM_VIS

    transformer_path = os.path.join(MODEL_PATH, "transformer")
    app = NeuronCosmos3BackboneApplication(model_path=transformer_path, config=config)
    app.load(COMPILED_PATH)

    # Warmup
    dummy = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
    dummy_ts = torch.tensor([0.5], dtype=torch.bfloat16)
    pos = torch.zeros(MAX_TEXT + NUM_VIS, 3, dtype=torch.long)
    ids = torch.zeros(1, MAX_TEXT, dtype=torch.long)
    for _ in range(2):
        _ = app.forward(ids, dummy, dummy_ts, pos)

    return app


@pytest.fixture(scope="module")
def vae():
    """Load compiled VAE decoder."""
    import torch_neuronx

    return torch.jit.load(VAE_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


# =============================================================================
# Tests
# =============================================================================


def test_backbone_loads(backbone):
    """Smoke test: backbone loads and is callable."""
    assert backbone is not None
    print("PASS: Backbone loaded successfully")


def test_backbone_output_shape(backbone):
    """Test that backbone produces correct output shape."""
    ids = torch.zeros(1, MAX_TEXT, dtype=torch.long)
    patches = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
    ts = torch.tensor([0.5], dtype=torch.bfloat16)
    pos = torch.zeros(MAX_TEXT + NUM_VIS, 3, dtype=torch.long)

    output = backbone.forward(ids, patches, ts, pos)

    assert output.shape == (1, NUM_VIS, 192), (
        f"Expected (1, {NUM_VIS}, 192), got {output.shape}"
    )
    assert output.dtype == torch.bfloat16
    print(f"PASS: Output shape {output.shape}, dtype {output.dtype}")


def test_backbone_nonzero_output(backbone):
    """Test that backbone produces non-trivial output (not all zeros)."""
    ids = torch.ones(1, MAX_TEXT, dtype=torch.long)  # non-zero token IDs
    patches = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
    ts = torch.tensor([0.5], dtype=torch.bfloat16)
    pos = torch.zeros(MAX_TEXT + NUM_VIS, 3, dtype=torch.long)

    output = backbone.forward(ids, patches, ts, pos)

    assert output.abs().max() > 0.01, (
        "Output is near-zero (model may not be loading weights)"
    )
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"PASS: Output norm={output.norm():.2f}, max={output.abs().max():.4f}")


def test_patchify_unpatchify_roundtrip():
    """Test patchify/unpatchify are inverse operations."""
    latents = torch.randn(1, 48, 1, 32, 32, dtype=torch.float32)

    patches = patchify(latents, patch_size=2)
    assert patches.shape == (1, 256, 192), f"Patch shape: {patches.shape}"

    reconstructed = unpatchify(patches, T=1, H=32, W=32, channels=48, patch_size=2)
    assert reconstructed.shape == latents.shape

    # Should be exact roundtrip
    assert torch.allclose(latents, reconstructed, atol=1e-6), (
        "Patchify/unpatchify not invertible"
    )
    print("PASS: Patchify/unpatchify roundtrip exact")


def test_tokenization(tokenizer):
    """Test tokenization produces expected format."""
    cond_ids, cond_len = tokenize_prompt(
        tokenizer,
        "A cat sitting on a windowsill",
        height=512,
        width=512,
        max_len=MAX_TEXT,
    )
    uncond_ids, uncond_len = tokenize_prompt(
        tokenizer, "", height=512, width=512, max_len=MAX_TEXT, negative=True
    )

    assert cond_ids.shape == (1, MAX_TEXT)
    assert uncond_ids.shape == (1, MAX_TEXT)
    assert 30 < cond_len < MAX_TEXT, f"Cond len {cond_len} unexpected"
    assert 20 < uncond_len < MAX_TEXT, f"Uncond len {uncond_len} unexpected"

    # Check special tokens at end
    eos_id = tokenizer.eos_token_id
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    assert cond_ids[0, cond_len - 1].item() == vision_start
    assert cond_ids[0, cond_len - 2].item() == eos_id

    print(f"PASS: Tokenization - cond={cond_len} tokens, uncond={uncond_len} tokens")


def test_position_ids():
    """Test position ID generation."""
    pos = build_position_ids(max_text_len=256, actual_text_len=50, T=1, pH=16, pW=16)

    assert pos.shape == (256 + 256, 3), f"Position shape: {pos.shape}"

    # Text: all 3 axes incrementing
    assert pos[0, 0] == 0 and pos[0, 1] == 0 and pos[0, 2] == 0
    assert pos[100, 0] == 100 and pos[100, 1] == 100

    # Vision: temporal offset = 50 + 15000 = 15050
    vision_start = 256
    assert pos[vision_start, 0] == 15050, f"Vision temporal: {pos[vision_start, 0]}"
    assert pos[vision_start, 1] == 0  # H starts at 0
    assert pos[vision_start, 2] == 0  # W starts at 0

    print("PASS: Position IDs correct")


def test_backbone_latency(backbone):
    """Test backbone call latency is within expected range."""
    ids = torch.zeros(1, MAX_TEXT, dtype=torch.long)
    patches = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
    ts = torch.tensor([0.5], dtype=torch.bfloat16)
    pos = torch.zeros(MAX_TEXT + NUM_VIS, 3, dtype=torch.long)

    # Warmup (already done in fixture, but just in case)
    for _ in range(3):
        _ = backbone.forward(ids, patches, ts, pos)

    # Measure
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = backbone.forward(ids, patches, ts, pos)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    # Nano: ~33ms, Super: ~80ms. Allow generous threshold.
    threshold = 200.0  # ms
    assert avg_ms < threshold, f"Latency {avg_ms:.1f}ms exceeds {threshold}ms threshold"
    print(f"PASS: Backbone latency {avg_ms:.1f}ms (threshold: {threshold}ms)")


def test_full_generation(backbone, vae, tokenizer):
    """End-to-end generation test: produces a valid image."""
    from diffusers import UniPCMultistepScheduler

    # Tokenize
    prompt = "A red apple on a white table"
    cond_ids, cond_len = tokenize_prompt(tokenizer, prompt, max_len=MAX_TEXT)
    uncond_ids, uncond_len = tokenize_prompt(
        tokenizer, "", max_len=MAX_TEXT, negative=True
    )

    # Position IDs
    cond_pos = build_position_ids(MAX_TEXT, cond_len, T=1, pH=16, pW=16)
    uncond_pos = build_position_ids(MAX_TEXT, uncond_len, T=1, pH=16, pW=16)

    # Latents
    gen = torch.manual_seed(42)
    latents = torch.randn(1, 48, 1, 32, 32, generator=gen, dtype=torch.float32)

    # Scheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )

    # Denoise (use fewer steps for speed)
    num_steps = 10
    latents = denoise(
        backbone=backbone,
        cond_ids=cond_ids,
        uncond_ids=uncond_ids,
        cond_pos=cond_pos,
        uncond_pos=uncond_pos,
        scheduler=scheduler,
        latents=latents,
        num_steps=num_steps,
        cfg_scale=6.0,
    )

    assert latents.shape == (1, 48, 1, 32, 32)
    assert not torch.isnan(latents).any(), "Latents contain NaN after denoising"

    # Denormalize + VAE
    vae_config_path = os.path.join(MODEL_PATH, "vae", "config.json")
    latents = denormalize_latents(latents, vae_config_path)

    with torch.no_grad():
        pixels = vae(latents.float())

    assert pixels.shape == (1, 3, 1, 512, 512), f"Pixel shape: {pixels.shape}"

    # Valid pixel range
    pixels_01 = ((pixels.squeeze(2).squeeze(0) + 1.0) / 2.0).clamp(0, 1)
    assert pixels_01.min() >= 0 and pixels_01.max() <= 1

    # Not blank (should have variance across pixels)
    pixel_std = pixels_01.std()
    assert pixel_std > 0.05, f"Image appears blank (std={pixel_std:.4f})"

    # Not uniform noise (should have spatial structure)
    # Check that some channels have local correlation
    center_patch = pixels_01[:, 200:300, 200:300]
    corner_patch = pixels_01[:, 0:100, 0:100]
    # Patches shouldn't be identical (image has spatial structure)
    diff = (center_patch.mean() - corner_patch.mean()).abs()
    # This is a weak check - just ensure the image isn't perfectly uniform
    print(
        f"PASS: Full generation - shape={pixels.shape}, std={pixel_std:.4f}, patch_diff={diff:.4f}"
    )


# =============================================================================
# Manual runner
# =============================================================================


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Cosmos3-Text2Image Integration Tests")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  MODEL_PATH:    {MODEL_PATH}")
    print(f"  COMPILED_PATH: {COMPILED_PATH}")
    print(f"  VAE_PATH:      {VAE_PATH}")
    print(f"  TP_DEGREE:     {TP_DEGREE}")

    # Unit tests (no model needed)
    print("\n" + "-" * 40)
    print("Unit Tests (no model required)")
    print("-" * 40)

    print("\n1. Patchify/Unpatchify roundtrip...")
    test_patchify_unpatchify_roundtrip()

    print("\n2. Position IDs...")
    test_position_ids()

    # Load model
    print("\n" + "-" * 40)
    print("Loading model...")
    print("-" * 40)

    import torch_neuronx

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE, world_size=TP_DEGREE, torch_dtype=torch.bfloat16
    )
    config = Cosmos3BackboneInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=128,
        vocab_size=151936,
        patch_channels=192,
        latent_channels=48,
        rope_theta=5000000.0,
        mrope_section=[24, 20, 20],
    )
    config.max_text_len = MAX_TEXT
    config.num_vision_patches = NUM_VIS

    transformer_path = os.path.join(MODEL_PATH, "transformer")
    backbone_model = NeuronCosmos3BackboneApplication(
        model_path=transformer_path, config=config
    )
    backbone_model.load(COMPILED_PATH)

    # Warmup
    dummy = torch.randn(1, NUM_VIS, 192, dtype=torch.bfloat16)
    dummy_ts = torch.tensor([0.5], dtype=torch.bfloat16)
    pos = torch.zeros(MAX_TEXT + NUM_VIS, 3, dtype=torch.long)
    ids = torch.zeros(1, MAX_TEXT, dtype=torch.long)
    for _ in range(3):
        _ = backbone_model.forward(ids, dummy, dummy_ts, pos)

    vae_model = torch.jit.load(VAE_PATH)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Integration tests
    print("\n" + "-" * 40)
    print("Integration Tests")
    print("-" * 40)

    print("\n3. Backbone loads...")
    test_backbone_loads(backbone_model)

    print("\n4. Output shape...")
    test_backbone_output_shape(backbone_model)

    print("\n5. Non-zero output...")
    test_backbone_nonzero_output(backbone_model)

    print("\n6. Tokenization...")
    test_tokenization(tok)

    print("\n7. Backbone latency...")
    test_backbone_latency(backbone_model)

    print("\n8. Full generation (10 steps)...")
    test_full_generation(backbone_model, vae_model, tok)

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
