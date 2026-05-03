#!/usr/bin/env python3
"""
Integration tests for LongCat-Image-Edit NeuronX adaptation.

Tests model compilation, loading, and inference on Trainium2.

Requirements:
  - trn2.48xlarge instance
  - Compiled models at COMPILED_MODELS_DIR (run compile.sh first)
  - HuggingFace model cached at HUGGINGFACE_CACHE_DIR

Usage:
  # Run with pytest:
  PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py --capture=tee-sys -v

  # Run directly:
  PYTHONPATH=src:$PYTHONPATH python test/integration/test_model.py
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path

# Add src directory to path
SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Configuration - update these paths for your environment
COMPILED_MODELS_DIR = os.environ.get(
    "COMPILED_MODELS_DIR", "/opt/dlami/nvme/compiled_models")
HUGGINGFACE_CACHE_DIR = os.environ.get(
    "HUGGINGFACE_CACHE_DIR", "/opt/dlami/nvme/longcat_hf_cache")
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"
TEST_IMAGE = str(Path(__file__).parent.parent.parent / "assets" / "test.png")


def is_neuron_available():
    """Check if Neuron runtime is available."""
    try:
        import torch_neuronx
        return True
    except ImportError:
        return False


def compiled_models_exist():
    """Check if compiled models are available."""
    required = [
        f"{COMPILED_MODELS_DIR}/transformer/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/vision_encoder/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/language_model/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/vae_decoder/model.pt",
    ]
    return all(os.path.exists(p) for p in required)


skip_no_neuron = pytest.mark.skipif(
    not is_neuron_available(),
    reason="Neuron runtime not available (requires trn2 instance)")

skip_no_compiled = pytest.mark.skipif(
    not compiled_models_exist(),
    reason="Compiled models not found (run compile.sh first)")


@pytest.fixture(scope="module")
def pipeline():
    """Load the LongCat pipeline with compiled Neuron models."""
    import torch
    from PIL import Image

    # Set environment
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["NEURON_CUSTOM_SILU"] = "1"

    from diffusers import LongCatImageEditPipeline
    from neuron_commons import NeuronTextEncoderWrapper

    try:
        from neuronx_distributed.trace.nxd_model.nxd_model import NxDModel
    except ImportError:
        pytest.skip("NxDModel not available")

    # Load pipeline
    print("Loading pipeline...")
    pipe = LongCatImageEditPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=HUGGINGFACE_CACHE_DIR,
    )

    # Load compiled components using the same loading logic as run script
    from run_longcat_image_edit import (
        load_transformer, load_text_encoder, load_vae,
    )

    class Args:
        compiled_models_dir = COMPILED_MODELS_DIR
        transformer_dir = None
        image_size = 448
        use_cfg_parallel = False

    args = Args()

    print("Loading compiled transformer...")
    pipe.transformer = load_transformer(COMPILED_MODELS_DIR, pipe, args)

    print("Loading compiled text encoder...")
    pipe.text_encoder = load_text_encoder(COMPILED_MODELS_DIR, pipe, args)

    print("Loading compiled VAE...")
    pipe.vae = load_vae(COMPILED_MODELS_DIR, pipe)

    return pipe


@skip_no_neuron
@skip_no_compiled
def test_model_loads(pipeline):
    """Test that all compiled models load successfully (smoke test)."""
    assert pipeline is not None
    assert pipeline.transformer is not None
    assert pipeline.text_encoder is not None
    assert pipeline.vae is not None
    print("PASS: All compiled models loaded successfully")


@skip_no_neuron
@skip_no_compiled
def test_inference_produces_output(pipeline):
    """Test that inference produces a valid output image."""
    import torch
    from PIL import Image

    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"
    source_image = Image.open(TEST_IMAGE).convert("RGB")

    with torch.inference_mode():
        result = pipeline(
            image=source_image,
            prompt="change the cat to a dog",
            negative_prompt=" ",
            num_inference_steps=10,  # Fewer steps for faster testing
            guidance_scale=4.5,
            generator=torch.manual_seed(42),
        )

    output_image = result.images[0]

    # Verify output is a valid image
    assert output_image is not None
    assert output_image.size[0] > 0
    assert output_image.size[1] > 0
    print(f"PASS: Inference produced output image: {output_image.size}")


@skip_no_neuron
@skip_no_compiled
def test_output_is_different_from_input(pipeline):
    """Test that the output image is different from the input (model actually edited)."""
    import torch
    import numpy as np
    from PIL import Image

    source_image = Image.open(TEST_IMAGE).convert("RGB")

    with torch.inference_mode():
        result = pipeline(
            image=source_image,
            prompt="change the cat to a dog",
            negative_prompt=" ",
            num_inference_steps=10,
            guidance_scale=4.5,
            generator=torch.manual_seed(42),
        )

    output_image = result.images[0]

    # Resize input to output size for comparison
    source_resized = source_image.resize(output_image.size)
    source_array = np.array(source_resized).astype(float)
    output_array = np.array(output_image).astype(float)

    # Compute mean absolute difference
    mean_diff = np.abs(source_array - output_array).mean()

    # The output should be significantly different from input
    assert mean_diff > 5.0, (
        f"Output too similar to input (mean_diff={mean_diff:.2f}). "
        "Model may not be editing correctly."
    )
    print(f"PASS: Output differs from input (mean_diff={mean_diff:.2f})")


@skip_no_neuron
@skip_no_compiled
def test_inference_timing(pipeline):
    """Test inference timing (informational, no strict threshold)."""
    import torch
    from PIL import Image

    source_image = Image.open(TEST_IMAGE).convert("RGB")

    # Warmup
    with torch.inference_mode():
        _ = pipeline(
            image=source_image,
            prompt="change the cat to a dog",
            negative_prompt=" ",
            num_inference_steps=5,
            guidance_scale=4.5,
            generator=torch.manual_seed(42),
        )

    # Timed run
    start = time.perf_counter()
    with torch.inference_mode():
        _ = pipeline(
            image=source_image,
            prompt="change the cat to a dog",
            negative_prompt=" ",
            num_inference_steps=50,
            guidance_scale=4.5,
            generator=torch.manual_seed(42),
        )
    elapsed = time.perf_counter() - start

    steps_per_sec = 50 / elapsed
    print(f"PASS: 50 steps in {elapsed:.2f}s ({steps_per_sec:.2f} steps/sec)")


if __name__ == "__main__":
    print("=" * 70)
    print("LongCat-Image-Edit Integration Tests")
    print("=" * 70)

    if not is_neuron_available():
        print("ERROR: Neuron runtime not available. Run on a trn2 instance.")
        sys.exit(1)

    if not compiled_models_exist():
        print("ERROR: Compiled models not found. Run compile.sh first.")
        print(f"  Expected at: {COMPILED_MODELS_DIR}")
        sys.exit(1)

    # Load pipeline
    print("\n[Setup] Loading pipeline with compiled models...")
    pipe = pipeline.__wrapped__() if hasattr(pipeline, '__wrapped__') else None

    # For direct execution, manually load
    import torch
    from PIL import Image

    os.environ["LOCAL_WORLD_SIZE"] = "4"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["NEURON_CUSTOM_SILU"] = "1"

    from diffusers import LongCatImageEditPipeline
    from run_longcat_image_edit import load_transformer, load_text_encoder, load_vae

    pipe = LongCatImageEditPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        local_files_only=True, cache_dir=HUGGINGFACE_CACHE_DIR)

    class Args:
        compiled_models_dir = COMPILED_MODELS_DIR
        transformer_dir = None
        image_size = 448
        use_cfg_parallel = False

    args = Args()
    pipe.transformer = load_transformer(COMPILED_MODELS_DIR, pipe, args)
    pipe.text_encoder = load_text_encoder(COMPILED_MODELS_DIR, pipe, args)
    pipe.vae = load_vae(COMPILED_MODELS_DIR, pipe)

    print("\n[Test 1] Smoke test (model loading)...")
    test_model_loads(pipe)

    print("\n[Test 2] Inference produces output...")
    test_inference_produces_output(pipe)

    print("\n[Test 3] Output differs from input...")
    test_output_is_different_from_input(pipe)

    print("\n[Test 4] Inference timing...")
    test_inference_timing(pipe)

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
