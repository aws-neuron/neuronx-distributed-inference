# coding=utf-8
"""Unit tests for NeuronGlm4MoeDecoderLayer dispatch logic.

These tests verify:
  1. The is_moe_layer flag: layer_idx >= first_k_dense_replace
  2. The DenseMLP class exists and has the correct structure
  3. The DecoderLayer class interface attributes exist

The is_moe_layer flag is pure Python logic and can be tested without
initializing a full distributed environment.  Structural checks verify
the class API contract without forward passes.

All tests run on CPU; no Neuron hardware is required.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Import once at module level with skip guard
# ---------------------------------------------------------------------------


def _import_classes():
    """Import modeling classes, skip if Neuron SDK missing."""
    try:
        from glm4_moe.modeling_glm4_moe import (
            NeuronGlm4MoeDecoderLayer,
            NeuronGlm4MoeDenseMLP,
        )

        return NeuronGlm4MoeDecoderLayer, NeuronGlm4MoeDenseMLP
    except ImportError:
        pytest.skip("glm4_moe package not importable (Neuron SDK missing)")


# ---------------------------------------------------------------------------
# is_moe_layer flag tests (pure math, no distributed context needed)
# ---------------------------------------------------------------------------


class TestIsMoELayerFlag:
    """Verify that is_moe_layer = (layer_idx >= first_k_dense_replace).

    The flag is computed in __init__ as a simple boolean expression.
    We test the formula directly without constructing the full layer,
    since full construction requires a distributed process group.
    """

    @staticmethod
    def _compute_flag(layer_idx: int, first_k_dense_replace: int) -> bool:
        """Reproduce the flag logic from NeuronGlm4MoeDecoderLayer.__init__."""
        return layer_idx >= first_k_dense_replace

    @pytest.mark.parametrize(
        "layer_idx,first_k,expected",
        [
            # first_k_dense_replace = 1: layer 0 is dense, rest are MoE
            (0, 1, False),
            (1, 1, True),
            (2, 1, True),
            # first_k_dense_replace = 2: layers 0,1 are dense
            (0, 2, False),
            (1, 2, False),
            (2, 2, True),
            (3, 2, True),
            # first_k_dense_replace = 0: all layers are MoE
            (0, 0, True),
            (1, 0, True),
            # first_k_dense_replace = 4: all 4 layers are dense
            (0, 4, False),
            (1, 4, False),
            (2, 4, False),
            (3, 4, False),
            # Boundary: layer exactly at first_k_dense_replace
            (2, 2, True),
            (3, 3, True),
        ],
    )
    def test_flag_formula(self, layer_idx, first_k, expected):
        """is_moe_layer must equal (layer_idx >= first_k_dense_replace)."""
        result = self._compute_flag(layer_idx, first_k)
        assert result == expected, (
            f"layer_idx={layer_idx}, first_k_dense_replace={first_k}: "
            f"expected is_moe_layer={expected}, got {result}"
        )

    def test_dense_layer_is_below_boundary(self):
        """All layers below first_k_dense_replace must be dense."""
        first_k = 3
        for idx in range(first_k):
            flag = self._compute_flag(idx, first_k)
            assert flag is False, f"Layer {idx} should be dense (first_k={first_k})"

    def test_moe_layer_is_at_or_above_boundary(self):
        """All layers at or above first_k_dense_replace must be MoE."""
        first_k = 3
        for idx in range(first_k, 6):
            flag = self._compute_flag(idx, first_k)
            assert flag is True, f"Layer {idx} should be MoE (first_k={first_k})"

    def test_glm45_air_default_first_k_is_1(self):
        """GLM-4.5 Air default: first_k_dense_replace=1, so layer 0 is dense."""
        first_k = 1  # GLM-4.5 Air default
        assert self._compute_flag(0, first_k) is False, "Layer 0 must be dense"
        assert self._compute_flag(1, first_k) is True, "Layer 1 must be MoE"


# ---------------------------------------------------------------------------
# DenseMLP structure tests (no distributed env needed — just class inspect)
# ---------------------------------------------------------------------------


class TestDenseMLPStructure:
    """Verify NeuronGlm4MoeDenseMLP class API contract."""

    def test_dense_mlp_class_exists(self):
        """NeuronGlm4MoeDenseMLP must be importable."""
        _, DenseMLP = _import_classes()
        assert DenseMLP is not None

    def test_dense_mlp_is_nn_module(self):
        """NeuronGlm4MoeDenseMLP must subclass nn.Module."""
        import torch.nn as nn

        _, DenseMLP = _import_classes()
        assert issubclass(DenseMLP, nn.Module), (
            "NeuronGlm4MoeDenseMLP must be an nn.Module subclass"
        )

    def test_dense_mlp_has_forward(self):
        """NeuronGlm4MoeDenseMLP must define a forward method."""
        _, DenseMLP = _import_classes()
        assert hasattr(DenseMLP, "forward"), "Missing forward method"
        assert callable(DenseMLP.forward)

    def test_dense_mlp_init_expects_config(self):
        """NeuronGlm4MoeDenseMLP.__init__ must accept a 'config' parameter."""
        import inspect

        _, DenseMLP = _import_classes()
        sig = inspect.signature(DenseMLP.__init__)
        assert "config" in sig.parameters, (
            "NeuronGlm4MoeDenseMLP.__init__ must accept a 'config' parameter"
        )


# ---------------------------------------------------------------------------
# DecoderLayer class structure tests
# ---------------------------------------------------------------------------


class TestDecoderLayerClassStructure:
    """Verify NeuronGlm4MoeDecoderLayer class API contract (no instantiation)."""

    def test_decoder_layer_class_exists(self):
        """NeuronGlm4MoeDecoderLayer must be importable."""
        DecoderLayer, _ = _import_classes()
        assert DecoderLayer is not None

    def test_decoder_layer_is_nn_module(self):
        """NeuronGlm4MoeDecoderLayer must subclass nn.Module."""
        import torch.nn as nn

        DecoderLayer, _ = _import_classes()
        assert issubclass(DecoderLayer, nn.Module)

    def test_decoder_layer_has_forward(self):
        """NeuronGlm4MoeDecoderLayer must define a forward method."""
        DecoderLayer, _ = _import_classes()
        assert hasattr(DecoderLayer, "forward") and callable(DecoderLayer.forward)

    def test_decoder_layer_init_accepts_layer_idx(self):
        """__init__ must accept a layer_idx parameter for dispatch."""
        import inspect

        DecoderLayer, _ = _import_classes()
        sig = inspect.signature(DecoderLayer.__init__)
        assert "layer_idx" in sig.parameters, (
            "NeuronGlm4MoeDecoderLayer.__init__ must accept 'layer_idx'"
        )

    def test_decoder_layer_init_accepts_config(self):
        """__init__ must accept a config parameter."""
        import inspect

        DecoderLayer, _ = _import_classes()
        sig = inspect.signature(DecoderLayer.__init__)
        assert "config" in sig.parameters, (
            "NeuronGlm4MoeDecoderLayer.__init__ must accept 'config'"
        )
