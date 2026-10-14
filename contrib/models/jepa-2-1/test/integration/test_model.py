"""Integration tests for V-JEPA 2.1 on Neuron.

These tests require Neuron hardware (trn2/inf2) and torch-neuronx.
They will be skipped on CPU-only machines.
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


@pytest.mark.skipif(not HAS_NEURON, reason="torch_neuronx not available")
class TestNeuronTrace:
    """Test tracing and running V-JEPA 2.1 encoder on Neuron."""

    def test_trace_vit_base_image(self):
        """Trace ViT-B encoder with single image input."""
        encoder = build_vjepa21_encoder(
            arch="vit_base", img_size=384, num_frames=16,
            pretrained=False, use_sdpa=False,
        )
        encoder.eval()

        example = torch.randn(1, 3, 1, 384, 384)
        traced = torch_neuronx.trace(encoder, example)

        # Run inference
        output = traced(example)
        assert output.shape == (1, 576, 768)

    def test_trace_vit_base_video(self):
        """Trace ViT-B encoder with 16-frame video input."""
        encoder = build_vjepa21_encoder(
            arch="vit_base", img_size=384, num_frames=16,
            pretrained=False, use_sdpa=False,
        )
        encoder.eval()

        example = torch.randn(1, 3, 16, 384, 384)
        traced = torch_neuronx.trace(encoder, example)

        output = traced(example)
        assert output.shape == (1, 4608, 768)

    def test_neuron_vs_cpu_accuracy(self):
        """Compare Neuron output against CPU reference using neuron_allclose."""
        from torch_neuronx.testing.validation import neuron_allclose

        encoder = build_vjepa21_encoder(
            arch="vit_base", img_size=384, num_frames=16,
            pretrained=False, use_sdpa=False,
        )
        encoder.eval()

        example = torch.randn(1, 3, 1, 384, 384)

        # CPU reference
        with torch.no_grad():
            cpu_output = encoder(example)

        # Neuron
        traced = torch_neuronx.trace(encoder, example)
        neuron_output = traced(example)

        result = neuron_allclose(neuron_output, cpu_output, rtol=0.01, atol=1e-5)
        assert result.allclose, f"Neuron vs CPU mismatch: max_rel_error={result.max_rel_error}"


@pytest.mark.skipif(not HAS_NEURON, reason="torch_neuronx not available")
class TestNeuronTraceVitLarge:
    """Test ViT-L on Neuron (larger model)."""

    def test_trace_vit_large_image(self):
        encoder = build_vjepa21_encoder(
            arch="vit_large", img_size=384, num_frames=16,
            pretrained=False, use_sdpa=False,
        )
        encoder.eval()

        example = torch.randn(1, 3, 1, 384, 384)
        traced = torch_neuronx.trace(encoder, example)

        output = traced(example)
        assert output.shape == (1, 576, 1024)
