#!/usr/bin/env python3
"""Integration tests for Wan 2.2 T2V-A14B NeuronX pipeline."""

import pytest
import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

MODEL_PATH = os.environ.get(
    "WAN_MODEL_PATH",
    "/mnt/work/.cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)
PORT_DIR = os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-lint-fix")


def test_modeling_imports():
    """Test that modeling code imports without errors."""
    from modeling_wan_cp import NeuronWanTransformer3DModel, CPWanFirstHalf, CPWanSecondHalf
    assert NeuronWanTransformer3DModel is not None
    assert CPWanFirstHalf is not None
    assert CPWanSecondHalf is not None


def test_application_imports():
    """Test that application wrapper imports."""
    from application_cp import NeuronWanCPApplication, WanCPInferenceConfig
    assert NeuronWanCPApplication is not None
    assert WanCPInferenceConfig is not None


def test_model_config(model_path=MODEL_PATH):
    """Test model config loads correctly from HF weights."""
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    from application_cp import WanCPInferenceConfig
    from neuronx_distributed_inference.models.config import NeuronConfig
    nc = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16, batch_size=1)
    config = WanCPInferenceConfig.from_pretrained(model_path, neuron_config=nc,
                                                   num_frames=13, height=480, width=832)
    assert config.num_attention_heads == 40
    assert config.num_layers == 40
    assert config.ffn_dim == 13824


def test_compiled_neffs_exist(port_dir=PORT_DIR):
    """Test that compiled NEFFs are present."""
    required = [
        "compiled_cp_transformer_first/model.pt",
        "compiled_cp_transformer_second/model.pt",
        "compiled_cp_transformer_2_first/model.pt",
        "compiled_cp_transformer_2_second/model.pt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(port_dir, f))]
    if missing:
        pytest.skip(f"Missing NEFFs (run compile first): {missing}")
    for f in required:
        assert os.path.exists(os.path.join(port_dir, f))


def test_pipeline_output_valid(port_dir=PORT_DIR):
    """Test that pipeline output is a valid video tensor."""
    output_path = os.path.join(port_dir, "neuron_output_allcp", "video_tensor.pt")
    if not os.path.exists(output_path):
        pytest.skip("No pipeline output — run run_inference.py first")
    video = torch.load(output_path, weights_only=True)
    # Shape: [1, 13, 3, 480, 832]
    assert video.dim() == 5
    assert video.shape[1] == 13 or video.shape[2] == 3
    assert not torch.isnan(video).any(), "Output contains NaN"
    assert not torch.isinf(video).any(), "Output contains Inf"
    assert video.float().std() > 0.1, "Output is degenerate (near-zero std)"


def test_output_equivalence(port_dir=PORT_DIR):
    """Test output is equivalent to CPU reference (cosine > 0.95)."""
    output_path = os.path.join(port_dir, "neuron_output_allcp", "video_tensor.pt")
    ref_path = os.path.join(port_dir, "reference", "reference_frames.pt")
    if not os.path.exists(output_path) or not os.path.exists(ref_path):
        pytest.skip("Missing output or reference tensors")
    video = torch.load(output_path, weights_only=True)
    ref = torch.load(ref_path, weights_only=True)
    cos = torch.nn.functional.cosine_similarity(
        video.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0),
    ).item()
    assert cos > 0.95, f"Cosine similarity {cos:.4f} below 0.95 threshold"


def test_frames_exist(port_dir=PORT_DIR):
    """Test that output frames were saved as PNGs."""
    frames_dir = os.path.join(port_dir, "neuron_output_allcp", "frames")
    if not os.path.exists(frames_dir):
        pytest.skip("No frames directory")
    pngs = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    assert len(pngs) >= 13, f"Expected 13+ frames, got {len(pngs)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
