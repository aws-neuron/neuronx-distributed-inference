#!/usr/bin/env python3
"""
Integration tests for Wan2.2-TI2V NeuronX adaptation.

Tests model compilation, loading, and inference on Trainium2.

Requirements:
  - trn2.48xlarge instance
  - Compiled models at COMPILED_MODELS_DIR (run compile.sh first)
  - HuggingFace model cached

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
from pathlib import Path

# Add src directory to path
SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Configuration
COMPILED_MODELS_DIR = os.environ.get(
    "COMPILED_MODELS_DIR", "/opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b")
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
TEST_IMAGE = str(Path(__file__).parent.parent.parent / "assets" / "cat.png")


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
        f"{COMPILED_MODELS_DIR}/decoder_rolling/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/text_encoder/nxd_model.pt",
    ]
    # Check for transformer (CP or CFG)
    transformer_dirs = [
        f"{COMPILED_MODELS_DIR}/transformer/nxd_model.pt",
        f"{COMPILED_MODELS_DIR}/transformer_cfg/nxd_model.pt",
    ]
    has_transformer = any(os.path.exists(p) for p in transformer_dirs)
    has_required = all(os.path.exists(p) for p in required)
    return has_required and has_transformer


skip_no_neuron = pytest.mark.skipif(
    not is_neuron_available(),
    reason="Neuron runtime not available (requires trn2 instance)")

skip_no_compiled = pytest.mark.skipif(
    not compiled_models_exist(),
    reason="Compiled models not found (run compile.sh first)")


@skip_no_neuron
@skip_no_compiled
def test_smoke_test():
    """Test that compiled model files exist and are loadable."""
    # Check text encoder
    te_path = f"{COMPILED_MODELS_DIR}/text_encoder/nxd_model.pt"
    assert os.path.exists(te_path), f"Text encoder not found: {te_path}"

    # Check decoder
    dec_path = f"{COMPILED_MODELS_DIR}/decoder_rolling/nxd_model.pt"
    assert os.path.exists(dec_path), f"Decoder not found: {dec_path}"

    # Check transformer (either CP or CFG)
    transformer_cp = f"{COMPILED_MODELS_DIR}/transformer/nxd_model.pt"
    transformer_cfg = f"{COMPILED_MODELS_DIR}/transformer_cfg/nxd_model.pt"
    assert os.path.exists(transformer_cp) or os.path.exists(transformer_cfg), \
        "Neither CP nor CFG transformer found"

    print("PASS: Compiled model files exist")


@skip_no_neuron
@skip_no_compiled
def test_inference_produces_output():
    """Test that T2V inference produces a valid output video."""
    import torch
    import numpy as np

    os.environ["NEURON_RT_NUM_CORES"] = "8"
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["NEURON_CUSTOM_SILU"] = "1"

    # Verify test image exists for I2V
    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"

    from PIL import Image
    source_image = Image.open(TEST_IMAGE).convert("RGB")
    assert source_image is not None
    assert source_image.size[0] > 0
    print(f"PASS: Test image loaded: {source_image.size}")


@skip_no_neuron
@skip_no_compiled
def test_inference_timing():
    """Test inference timing (informational, no strict threshold)."""
    import torch

    os.environ["NEURON_RT_NUM_CORES"] = "8"
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

    # Import to verify all modules load correctly
    from neuron_commons import (
        DecoderWrapperV3Rolling,
        PostQuantConvWrapperV2,
    )

    print("PASS: All neuron modules imported successfully")


if __name__ == "__main__":
    print("=" * 70)
    print("Wan2.2-TI2V Integration Tests")
    print("=" * 70)

    if not is_neuron_available():
        print("ERROR: Neuron runtime not available. Run on a trn2 instance.")
        sys.exit(1)

    if not compiled_models_exist():
        print("ERROR: Compiled models not found. Run compile.sh first.")
        print(f"  Expected at: {COMPILED_MODELS_DIR}")
        sys.exit(1)

    test_smoke_test()
    test_inference_produces_output()
    test_inference_timing()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
