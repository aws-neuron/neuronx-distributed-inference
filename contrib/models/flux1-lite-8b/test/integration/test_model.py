"""
Integration tests for FLUX.1-lite-8B-alpha on Neuron.

FLUX.1-lite-8B-alpha is architecturally identical to FLUX.1-dev with fewer
double-stream blocks (8 vs 19). It runs natively on NxDI's first-party
FLUX.1 implementation with no code modifications.

Tests:
1. test_smoke_pipeline_loads: Pipeline loads without errors
2. test_generation_produces_image: Generates a 1024x1024 image
3. test_warm_generation_time: Warm generation < 15s (25 steps, no CFG)

Requirements:
    - trn2.3xlarge with LNC=2 (4 logical cores)
    - Neuron SDK 2.29+
    - diffusers >= 0.37.1, transformers, sentencepiece, protobuf
    - Model downloaded to /shared/flux1-lite-8b/ or $FLUX_LITE_MODEL_PATH

Run:
    pytest test_model.py -v
    # Or standalone:
    python test_model.py
"""

import gc
import os
import sys
import time

import numpy as np
import pytest
import torch

MODEL_PATH = os.environ.get("FLUX_LITE_MODEL_PATH", "/shared/flux1-lite-8b/")
COMPILE_DIR = os.environ.get("FLUX_LITE_COMPILE_DIR", "/tmp/flux_lite_test/")
TP_DEGREE = int(os.environ.get("FLUX_LITE_TP_DEGREE", "4"))
HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 25
GUIDANCE_SCALE = 3.5
PROMPT = "A cat holding a sign that says hello world"


@pytest.fixture(scope="module")
def neuron_app():
    """Create, compile, and load FLUX.1-lite using NxDI's FLUX.1 application."""
    from neuronx_distributed_inference.models.diffusers.flux.application import (
        NeuronFluxApplication,
        create_flux_config,
        get_flux_parallelism_config,
    )
    from neuronx_distributed_inference.utils.random import set_random_seed

    set_random_seed(0)

    world_size = get_flux_parallelism_config(TP_DEGREE)
    dtype = torch.bfloat16

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        MODEL_PATH,
        world_size,
        TP_DEGREE,
        dtype,
        HEIGHT,
        WIDTH,
    )

    app = NeuronFluxApplication(
        model_path=MODEL_PATH,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=HEIGHT,
        width=WIDTH,
    )

    app.compile(COMPILE_DIR)
    app.load(COMPILE_DIR)

    # Warmup
    app(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
    )

    yield app

    del app
    gc.collect()


def test_smoke_pipeline_loads(neuron_app):
    """Pipeline loads without errors and has required components."""
    assert neuron_app is not None
    assert neuron_app.pipe is not None
    assert neuron_app.pipe.transformer is not None
    assert neuron_app.pipe.text_encoder is not None
    assert neuron_app.pipe.text_encoder_2 is not None
    assert neuron_app.pipe.vae is not None


def test_generation_produces_image(neuron_app):
    """Generates an image at the expected resolution."""
    result = neuron_app(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
    )

    assert result is not None
    assert hasattr(result, "images")
    assert len(result.images) == 1

    image = result.images[0]
    assert image.size == (WIDTH, HEIGHT), (
        f"Expected ({WIDTH}, {HEIGHT}), got {image.size}"
    )

    # Verify the image has reasonable pixel values (not blank/noise)
    img_array = np.array(image)
    assert img_array.shape == (HEIGHT, WIDTH, 3)
    assert img_array.std() > 10, "Image appears blank or uniform"

    # Save for inspection
    os.makedirs(os.path.join(COMPILE_DIR, "test_outputs"), exist_ok=True)
    image.save(os.path.join(COMPILE_DIR, "test_outputs", "test_generation.png"))
    print(f"Image saved, pixel std={img_array.std():.1f}")


def test_warm_generation_time(neuron_app):
    """Warm generation should complete in reasonable time."""
    t0 = time.time()
    neuron_app(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
    )
    elapsed = time.time() - t0
    print(f"Warm generation time: {elapsed:.2f}s")

    # FLUX.1-lite with 25 steps, no CFG, TP=4: expect ~6s, allow up to 15s
    assert elapsed < 15, f"Generation took {elapsed:.2f}s, expected < 15s"


# Standalone runner
if __name__ == "__main__":
    from neuronx_distributed_inference.models.diffusers.flux.application import (
        NeuronFluxApplication,
        create_flux_config,
        get_flux_parallelism_config,
    )
    from neuronx_distributed_inference.utils.random import set_random_seed

    set_random_seed(0)

    print("=" * 60)
    print("FLUX.1-lite-8B-alpha Integration Tests")
    print("=" * 60)

    world_size = get_flux_parallelism_config(TP_DEGREE)

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        MODEL_PATH,
        world_size,
        TP_DEGREE,
        torch.bfloat16,
        HEIGHT,
        WIDTH,
    )

    print(f"\nModel config:")
    print(f"  num_layers (double blocks): {backbone_config.num_layers}")
    print(f"  num_single_layers: {backbone_config.num_single_layers}")
    print(f"  TP degree: {TP_DEGREE}")

    app = NeuronFluxApplication(
        model_path=MODEL_PATH,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=HEIGHT,
        width=WIDTH,
    )

    print("\n[1/5] Compiling...")
    t0 = time.time()
    app.compile(COMPILE_DIR)
    print(f"  Compilation: {time.time() - t0:.1f}s")

    print("\n[2/5] Loading...")
    app.load(COMPILE_DIR)

    print("\n[3/5] Warmup...")
    app(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
    )

    print("\n[4/5] test_smoke_pipeline_loads")
    test_smoke_pipeline_loads(app)
    print("  PASSED")

    print("\n[5/5] test_generation_produces_image")
    test_generation_produces_image(app)
    print("  PASSED")

    print("\n[6/6] test_warm_generation_time")
    test_warm_generation_time(app)
    print("  PASSED")

    print("\nAll tests passed!")
