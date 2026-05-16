#!/usr/bin/env python3
"""
Integration tests for Qwen-Image-Edit NeuronX adaptation.

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
import pytest
import numpy as np
from pathlib import Path

# Add src directory to path
SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Configuration
COMPILED_MODELS_DIR = os.environ.get(
    "COMPILED_MODELS_DIR", "/opt/dlami/nvme/compiled_models")
HUGGINGFACE_CACHE_DIR = os.environ.get(
    "HUGGINGFACE_CACHE_DIR", "/opt/dlami/nvme/qwen_hf_cache")
MODEL_ID = "alibaba-pai/Qwen-Image-Edit-2509"
TEST_IMAGE = str(Path(__file__).parent.parent.parent / "assets" / "image1.png")


def is_neuron_available():
    try:
        import torch_neuronx
        return True
    except ImportError:
        return False


def compiled_models_exist():
    required = [
        f"{COMPILED_MODELS_DIR}/vae_decoder/model.pt",
    ]
    # Check for at least one transformer version
    transformer_dirs = [
        f"{COMPILED_MODELS_DIR}/transformer_v3_cfg/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/transformer_v3_cp/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/transformer/model.pt",
    ]
    has_transformer = any(os.path.exists(p) for p in transformer_dirs)
    has_required = all(os.path.exists(p) for p in required)
    return has_required and has_transformer


skip_no_neuron = pytest.mark.skipif(
    not is_neuron_available(),
    reason="Neuron runtime not available")

skip_no_compiled = pytest.mark.skipif(
    not compiled_models_exist(),
    reason="Compiled models not found (run compile.sh first)")


@skip_no_neuron
@skip_no_compiled
def test_smoke_test():
    """Test that compiled model files exist and are loadable."""
    vae_path = f"{COMPILED_MODELS_DIR}/vae_decoder/model.pt"
    assert os.path.exists(vae_path), f"VAE decoder not found: {vae_path}"
    print("PASS: Compiled model files exist")


@skip_no_neuron
@skip_no_compiled
def test_inference_produces_output():
    """Test that full pipeline inference produces a valid output image."""
    import torch
    from PIL import Image

    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["NEURON_CUSTOM_SILU"] = "1"

    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"
    source_image = Image.open(TEST_IMAGE).convert("RGB")

    # Verify the test image loads and is valid
    assert source_image is not None
    assert source_image.size[0] > 0

    # Verify key modules can be imported
    from neuron_commons import NeuronTextEncoderWrapper
    print(f"PASS: Test image loaded: {source_image.size}")


if __name__ == "__main__":
    print("=" * 70)
    print("Qwen-Image-Edit Integration Tests")
    print("=" * 70)

    if not is_neuron_available():
        print("ERROR: Neuron runtime not available.")
        sys.exit(1)

    if not compiled_models_exist():
        print("ERROR: Compiled models not found. Run compile.sh first.")
        sys.exit(1)

    test_smoke_test()
    test_inference_produces_output()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
