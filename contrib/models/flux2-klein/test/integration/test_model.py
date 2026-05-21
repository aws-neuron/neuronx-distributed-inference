"""
Integration tests for FLUX.2-klein-base-9B on Neuron.

Tests:
1. test_smoke_pipeline_loads: Pipeline loads without errors
2. test_generation_produces_image: Generates a 1024x1024 image
3. test_accuracy_vs_cpu: SSIM > 0.7 between Neuron and CPU-generated reference
4. test_warm_generation_time: Warm generation < 120s

Requirements:
    - trn2.3xlarge with LNC=2 (4 logical cores)
    - Neuron SDK 2.29+
    - diffusers >= 0.37.1
    - Model downloaded to /shared/flux2-klein/ or $FLUX2_MODEL_PATH

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

# Add contrib to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRIB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(CONTRIB_ROOT, ".."))

MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "/shared/flux2-klein/")
COMPILE_DIR = os.environ.get("FLUX2_COMPILE_DIR", "/tmp/flux2_klein_test/")
TP_DEGREE = int(os.environ.get("FLUX2_TP_DEGREE", "4"))
HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 50
GUIDANCE_SCALE = 4.0
PROMPT = "A cat holding a sign that says hello world"


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images (H, W, C) in [0, 255] uint8."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)


@pytest.fixture(scope="module")
def neuron_app():
    """Create, compile, and load the FLUX.2-klein NxDI application."""
    from flux2_klein.src.application import (
        NeuronFlux2KleinApplication,
        create_flux2_klein_config,
    )

    backbone_config = create_flux2_klein_config(
        model_path=MODEL_PATH,
        backbone_tp_degree=TP_DEGREE,
        dtype=torch.bfloat16,
        height=HEIGHT,
        width=WIDTH,
    )

    app = NeuronFlux2KleinApplication(
        model_path=MODEL_PATH,
        backbone_config=backbone_config,
        height=HEIGHT,
        width=WIDTH,
    )

    # Compile
    app.compile(COMPILE_DIR)
    # Load
    app.load(COMPILE_DIR)

    # Warmup
    app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )

    yield app

    # Cleanup
    del app
    gc.collect()


def test_smoke_pipeline_loads(neuron_app):
    """Pipeline loads without errors and has required attributes."""
    assert neuron_app is not None
    assert neuron_app.pipe is not None
    assert hasattr(neuron_app, "backbone_app")
    assert neuron_app.pipe.transformer is not None
    assert neuron_app.pipe.text_encoder is not None
    assert neuron_app.pipe.vae is not None


def test_generation_produces_image(neuron_app):
    """Generates an image at the expected resolution."""
    result = neuron_app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )

    assert result is not None
    assert hasattr(result, "images")
    assert len(result.images) == 1

    image = result.images[0]
    assert image.size == (WIDTH, HEIGHT), (
        f"Expected ({WIDTH}, {HEIGHT}), got {image.size}"
    )

    # Save for inspection
    os.makedirs(os.path.join(COMPILE_DIR, "test_outputs"), exist_ok=True)
    image.save(os.path.join(COMPILE_DIR, "test_outputs", "test_generation.png"))


def test_accuracy_vs_cpu(neuron_app):
    """
    Compare Neuron output against CPU reference using SSIM.

    Since FLUX.2-klein uses classic CFG (not guidance distillation),
    the Neuron output should closely match CPU output when using the
    same random seed and parameters.
    """
    from neuronx_distributed_inference.utils.random import set_random_seed

    # Generate on Neuron
    set_random_seed(42)
    neuron_result = neuron_app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    neuron_image = np.array(neuron_result.images[0])

    # Generate CPU reference
    from diffusers import Flux2KleinPipeline

    set_random_seed(42)
    cpu_pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    cpu_result = cpu_pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    cpu_image = np.array(cpu_result.images[0])

    del cpu_pipe
    gc.collect()

    # Compare
    ssim = compute_ssim(neuron_image, cpu_image)
    print(f"SSIM (Neuron vs CPU): {ssim:.4f}")

    # Save both for visual comparison
    os.makedirs(os.path.join(COMPILE_DIR, "test_outputs"), exist_ok=True)
    from PIL import Image

    Image.fromarray(neuron_image).save(
        os.path.join(COMPILE_DIR, "test_outputs", "accuracy_neuron.png")
    )
    Image.fromarray(cpu_image).save(
        os.path.join(COMPILE_DIR, "test_outputs", "accuracy_cpu.png")
    )

    assert ssim > 0.7, f"SSIM {ssim:.4f} below threshold 0.7"


def test_warm_generation_time(neuron_app):
    """Warm generation should complete in reasonable time."""
    t0 = time.time()
    neuron_app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    elapsed = time.time() - t0
    print(f"Warm generation time: {elapsed:.2f}s")

    # FLUX.2-klein with 50 steps + CFG (2x forward) on TP=4 should be < 120s
    assert elapsed < 120, f"Generation took {elapsed:.2f}s, expected < 120s"


# Standalone runner
if __name__ == "__main__":
    from flux2_klein.src.application import (
        NeuronFlux2KleinApplication,
        create_flux2_klein_config,
    )

    print("=" * 60)
    print("FLUX.2-klein-base-9B Integration Tests")
    print("=" * 60)

    backbone_config = create_flux2_klein_config(
        model_path=MODEL_PATH,
        backbone_tp_degree=TP_DEGREE,
        dtype=torch.bfloat16,
        height=HEIGHT,
        width=WIDTH,
    )

    app = NeuronFlux2KleinApplication(
        model_path=MODEL_PATH,
        backbone_config=backbone_config,
        height=HEIGHT,
        width=WIDTH,
    )

    print("\n[1/5] Compiling...")
    app.compile(COMPILE_DIR)

    print("\n[2/5] Loading...")
    app.load(COMPILE_DIR)

    print("\n[3/5] Warmup...")
    app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
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
