"""Unit tests for NeuronSolarOpenDecoderLayer.

These tests run on CPU (no Neuron hardware required). They verify the
architectural properties of the decoder layer via source inspection —
no instantiation is attempted since the layer requires a distributed env.

Key Solar Open decoder properties:
  - ALL layers are MoE (first_k_dense_replace=0): no dense MLP branch
  - self_attn: NeuronSolarOpenAttention
  - mlp: MoE module (initialized via initialize_solar_open_moe_module)
  - input_layernorm + post_attention_layernorm: RMSNorm
  - forward returns (hidden_states, present_key_value, cos_cache, sin_cache, None)
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _import_classes():
    """Import decoder and related classes, skip if Neuron SDK is missing."""
    try:
        from solar_open.modeling_solar_open import (
            NeuronSolarOpenDecoderLayer,
            NeuronSolarOpenAttention,
            initialize_solar_open_moe_module,
        )

        return (
            NeuronSolarOpenDecoderLayer,
            NeuronSolarOpenAttention,
            initialize_solar_open_moe_module,
        )
    except ImportError as e:
        pytest.skip(f"solar_open package not importable (Neuron SDK missing): {e}")


# ---------------------------------------------------------------------------
# TestDecoderLayerClassStructure
# ---------------------------------------------------------------------------


class TestDecoderLayerClassStructure:
    """Verify NeuronSolarOpenDecoderLayer class API contract."""

    def test_decoder_class_exists(self):
        """NeuronSolarOpenDecoderLayer must be importable."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        assert NeuronSolarOpenDecoderLayer is not None

    def test_decoder_inherits_nn_module(self):
        """NeuronSolarOpenDecoderLayer must subclass nn.Module."""
        import torch.nn as nn

        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        assert issubclass(NeuronSolarOpenDecoderLayer, nn.Module), (
            "NeuronSolarOpenDecoderLayer must extend nn.Module"
        )

    def test_decoder_init_accepts_config_and_layer_idx(self):
        """__init__ must accept 'config' and 'layer_idx' parameters."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        sig = inspect.signature(NeuronSolarOpenDecoderLayer.__init__)
        params = sig.parameters
        assert "config" in params, "Missing 'config' parameter in __init__"
        assert "layer_idx" in params, "Missing 'layer_idx' parameter in __init__"

    def test_decoder_has_forward(self):
        """NeuronSolarOpenDecoderLayer must define a forward method."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        assert hasattr(NeuronSolarOpenDecoderLayer, "forward")
        assert callable(NeuronSolarOpenDecoderLayer.forward)

    def test_decoder_forward_signature(self):
        """forward must accept hidden_states, attention_mask, position_ids, past_key_value."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        sig = inspect.signature(NeuronSolarOpenDecoderLayer.forward)
        params = sig.parameters
        assert "hidden_states" in params, "Missing 'hidden_states' in forward"
        assert "attention_mask" in params, "Missing 'attention_mask' in forward"
        assert "position_ids" in params, "Missing 'position_ids' in forward"
        assert "past_key_value" in params, "Missing 'past_key_value' in forward"


# ---------------------------------------------------------------------------
# TestAllLayersMoE — Solar Open has first_k_dense_replace=0 (ALL MoE)
# ---------------------------------------------------------------------------


class TestAllLayersMoE:
    """Verify that Solar Open decoder always uses MoE (no dense MLP branch)."""

    def test_source_uses_moe_module_initializer(self):
        """__init__ must call initialize_solar_open_moe_module (not a dense MLP)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "initialize_solar_open_moe_module" in src, (
            "NeuronSolarOpenDecoderLayer must use initialize_solar_open_moe_module for mlp"
        )

    def test_source_has_no_dense_mlp_branch(self):
        """Source must NOT contain is_moe_layer conditional (all layers are MoE)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "is_moe_layer" not in src, (
            "NeuronSolarOpenDecoderLayer must not have is_moe_layer flag "
            "(first_k_dense_replace=0 means ALL layers are MoE)"
        )

    def test_source_has_no_first_k_dense_replace_check(self):
        """Source must NOT check first_k_dense_replace (always 0 for Solar Open)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "first_k_dense_replace" not in src, (
            "NeuronSolarOpenDecoderLayer should not check first_k_dense_replace "
            "(Solar Open always uses MoE for all layers)"
        )

    def test_source_assigns_mlp_attribute(self):
        """__init__ must assign self.mlp (the MoE module)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "self.mlp" in src, (
            "NeuronSolarOpenDecoderLayer must assign self.mlp in __init__"
        )

    def test_forward_calls_mlp(self):
        """forward must call self.mlp (the MoE module)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.forward)
        assert "self.mlp" in src, (
            "NeuronSolarOpenDecoderLayer.forward must call self.mlp"
        )


# ---------------------------------------------------------------------------
# TestDecoderLayerComponents
# ---------------------------------------------------------------------------


class TestDecoderLayerComponents:
    """Verify decoder layer has correct sub-modules."""

    def test_source_has_self_attn(self):
        """__init__ must assign self.self_attn = NeuronSolarOpenAttention."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "self.self_attn" in src, "Missing self.self_attn in __init__"
        assert "NeuronSolarOpenAttention" in src, (
            "self.self_attn must be NeuronSolarOpenAttention"
        )

    def test_source_has_input_layernorm(self):
        """__init__ must assign self.input_layernorm."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "self.input_layernorm" in src, "Missing self.input_layernorm in __init__"

    def test_source_has_post_attention_layernorm(self):
        """__init__ must assign self.post_attention_layernorm."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "self.post_attention_layernorm" in src, (
            "Missing self.post_attention_layernorm in __init__"
        )

    def test_source_has_layer_idx(self):
        """__init__ must store self.layer_idx."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.__init__)
        assert "self.layer_idx" in src, "Missing self.layer_idx in __init__"

    def test_forward_uses_residual_connections(self):
        """forward must use residual connections (hidden_states = residual + ...)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.forward)
        assert "residual" in src, (
            "NeuronSolarOpenDecoderLayer.forward must use residual connections"
        )

    def test_forward_returns_5_tuple(self):
        """forward must return a 5-tuple: (hidden_states, kv, cos, sin, None)."""
        NeuronSolarOpenDecoderLayer, _, _ = _import_classes()
        src = inspect.getsource(NeuronSolarOpenDecoderLayer.forward)
        # The return statement should have 5 elements
        assert "outputs = (" in src or "return (" in src, "forward must return a tuple"
        # Check for the None at the end (5th element)
        assert ", None)" in src, (
            "forward must return 5-tuple ending with None (no router logits)"
        )
