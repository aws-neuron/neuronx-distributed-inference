"""Pretrained weight smoke tests for V-JEPA 2.1 on Neuron.

Validates that pretrained ViT-B produces correct features on Neuron
by comparing BF16 Neuron output against FP32 CPU reference.

Requires: trn2/inf2 instance with torch-neuronx.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    import torch_neuronx
    HAS_NEURON = True
except ImportError:
    HAS_NEURON = False

from modeling_jepa21 import build_vjepa21_encoder


@pytest.fixture
def pretrained_encoder():
    """Build pretrained ViT-B encoder (FP32)."""
    encoder = build_vjepa21_encoder(
        arch="vit_base", img_size=384, num_frames=16,
        pretrained=True, use_sdpa=False,
    )
    encoder.eval()
    return encoder


@pytest.fixture
def synthetic_video():
    """Deterministic synthetic video input (1, 3, 16, 384, 384)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 16, 384, 384)


class TestPretrainedCPU:
    """CPU-only sanity checks for pretrained weights."""

    def test_pretrained_loads(self, pretrained_encoder):
        """Pretrained weights load without error."""
        assert pretrained_encoder is not None

    def test_pretrained_forward_shape(self, pretrained_encoder, synthetic_video):
        """Pretrained encoder produces correct output shape."""
        with torch.no_grad():
            out = pretrained_encoder(synthetic_video)
        assert out.shape == (1, 4608, 768)

    def test_pretrained_no_nan(self, pretrained_encoder, synthetic_video):
        """Pretrained encoder output has no NaN/Inf."""
        with torch.no_grad():
            out = pretrained_encoder(synthetic_video)
        assert not out.isnan().any()
        assert not out.isinf().any()


@pytest.mark.skipif(not HAS_NEURON, reason="torch_neuronx not available")
class TestPretrainedNeuron:
    """Neuron accuracy tests with pretrained weights."""

    def test_pretrained_neuron_vs_cpu(self, pretrained_encoder, synthetic_video):
        """Pretrained ViT-B on Neuron matches CPU (cosine sim > 0.999)."""
        with torch.no_grad():
            cpu_out = pretrained_encoder(synthetic_video)

        pretrained_encoder.bfloat16()
        video_bf16 = synthetic_video.bfloat16()
        traced = torch_neuronx.trace(
            pretrained_encoder, video_bf16,
            compiler_args=["--auto-cast", "none"],
        )

        # Warmup
        for _ in range(3):
            traced(video_bf16)

        neuron_out = traced(video_bf16)

        cos_sim = torch.nn.functional.cosine_similarity(
            cpu_out.float().flatten().unsqueeze(0),
            neuron_out.float().flatten().unsqueeze(0),
        ).item()

        assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} below 0.999 threshold"

    def test_pretrained_neuron_no_nan(self, pretrained_encoder, synthetic_video):
        """Neuron output has no NaN/Inf values."""
        pretrained_encoder.bfloat16()
        video_bf16 = synthetic_video.bfloat16()
        traced = torch_neuronx.trace(
            pretrained_encoder, video_bf16,
            compiler_args=["--auto-cast", "none"],
        )
        neuron_out = traced(video_bf16)

        assert not neuron_out.isnan().any(), "Neuron output contains NaN"
        assert not neuron_out.isinf().any(), "Neuron output contains Inf"
