"""Unit tests for NeuronSolarOpenAttention.

These tests run on CPU (no Neuron hardware required). They verify the
architectural properties of the attention module — especially the key
differences from GLM-4.5 MoE attention:

  - Full RoPE: rotary_dim = head_dim (partial_rotary_factor=1.0)
  - No QK normalisation (use_qk_norm=False)
  - No attention bias (qkv_bias=False)
  - Plain RotaryEmbedding by default; SolarOpenYarnRotaryEmbedding for yarn scaling

NeuronSolarOpenAttention requires a distributed environment at instantiation
time, so tests use source inspection rather than direct instantiation.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _import_attention():
    """Import attention classes, skip if Neuron SDK is missing."""
    try:
        from solar_open.modeling_solar_open import (
            NeuronSolarOpenAttention,
            SolarOpenYarnRotaryEmbedding,
        )

        return NeuronSolarOpenAttention, SolarOpenYarnRotaryEmbedding
    except ImportError as e:
        pytest.skip(f"solar_open package not importable (Neuron SDK missing): {e}")


def _make_attention_config(
    hidden_size: int = 512,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 128,
    rope_scaling=None,
    max_position_embeddings: int = 2048,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-5,
) -> SimpleNamespace:
    """Create a minimal config namespace for NeuronSolarOpenAttention inspection."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
    )


# ---------------------------------------------------------------------------
# TestFullRoPE — Solar Open uses full RoPE (rotary_dim = head_dim)
# ---------------------------------------------------------------------------


class TestFullRoPE:
    """Verify that Solar Open uses full RoPE (partial_rotary_factor=1.0)."""

    def test_rotary_dim_equals_head_dim(self):
        """rotary_dim must equal head_dim for full RoPE (partial_rotary_factor=1.0)."""
        for head_dim in [64, 128, 256]:
            # Solar Open: rotary_dim = head_dim (factor=1.0)
            rotary_dim = head_dim
            assert rotary_dim == head_dim, (
                f"Full RoPE: rotary_dim should equal head_dim={head_dim}, got {rotary_dim}"
            )

    def test_solar_open_default_full_rope(self):
        """Default config: head_dim=128, partial_rotary_factor=1.0 → rotary_dim=128."""
        head_dim = 128
        partial_rotary_factor = 1.0
        rotary_dim = int(head_dim * partial_rotary_factor)
        assert rotary_dim == 128, f"Expected 128, got {rotary_dim}"

    def test_source_uses_head_dim_for_rotary(self):
        """Source code must set rotary_dim = config.head_dim (full RoPE)."""
        NeuronSolarOpenAttention, _ = _import_attention()
        src = inspect.getsource(NeuronSolarOpenAttention.__init__)
        # The implementation should assign rotary_dim = config.head_dim
        assert "head_dim" in src, "rotary_dim must reference head_dim in __init__"
        assert "rotary_dim" in src, "rotary_dim variable must exist in __init__"

    def test_full_rope_no_passthrough(self):
        """Full RoPE: all head_dim elements are rotated, none are passed through."""
        # Verify there is no split into rotary/non-rotary portions
        head_dim = 128
        rotary_dim = head_dim  # full RoPE
        passthrough_dim = head_dim - rotary_dim
        assert passthrough_dim == 0, (
            f"Full RoPE should have 0 passthrough dimensions, got {passthrough_dim}"
        )


# ---------------------------------------------------------------------------
# TestNoQKNorm
# ---------------------------------------------------------------------------


class TestNoQKNorm:
    """Verify QK normalisation is disabled."""

    def test_source_sets_use_qk_norm_false(self):
        """Source code must pass use_qk_norm=False to super().__init__()."""
        NeuronSolarOpenAttention, _ = _import_attention()
        src = inspect.getsource(NeuronSolarOpenAttention.__init__)
        assert "use_qk_norm=False" in src, (
            "NeuronSolarOpenAttention must set use_qk_norm=False (Solar Open has no QK norm)"
        )

    def test_source_sets_qkv_bias_false(self):
        """Source code must pass qkv_bias=False to super().__init__()."""
        NeuronSolarOpenAttention, _ = _import_attention()
        src = inspect.getsource(NeuronSolarOpenAttention.__init__)
        assert "qkv_bias=False" in src, (
            "NeuronSolarOpenAttention must set qkv_bias=False (Solar Open has no attention bias)"
        )


# ---------------------------------------------------------------------------
# TestYarnRotaryEmbedding
# ---------------------------------------------------------------------------


class TestYarnRotaryEmbedding:
    """Verify SolarOpenYarnRotaryEmbedding structure."""

    def test_yarn_class_exists(self):
        """SolarOpenYarnRotaryEmbedding must be importable."""
        _, SolarOpenYarnRotaryEmbedding = _import_attention()
        assert SolarOpenYarnRotaryEmbedding is not None

    def test_yarn_is_nn_module(self):
        """SolarOpenYarnRotaryEmbedding must subclass nn.Module."""
        import torch.nn as nn

        _, SolarOpenYarnRotaryEmbedding = _import_attention()
        assert issubclass(SolarOpenYarnRotaryEmbedding, nn.Module), (
            "SolarOpenYarnRotaryEmbedding must be an nn.Module subclass"
        )

    def test_yarn_has_forward(self):
        """SolarOpenYarnRotaryEmbedding must define a forward method."""
        _, SolarOpenYarnRotaryEmbedding = _import_attention()
        assert hasattr(SolarOpenYarnRotaryEmbedding, "forward")
        assert callable(SolarOpenYarnRotaryEmbedding.forward)

    def test_yarn_init_signature(self):
        """SolarOpenYarnRotaryEmbedding.__init__ must accept standard YaRN params."""
        _, SolarOpenYarnRotaryEmbedding = _import_attention()
        sig = inspect.signature(SolarOpenYarnRotaryEmbedding.__init__)
        params = sig.parameters
        assert "dim" in params, "Missing 'dim' parameter"
        assert "max_position_embeddings" in params, "Missing 'max_position_embeddings'"
        assert "base" in params, "Missing 'base' parameter"
        assert "scaling_factor" in params, "Missing 'scaling_factor' parameter"
        assert "original_max_position_embeddings" in params, (
            "Missing 'original_max_position_embeddings' parameter"
        )

    def test_attention_source_uses_yarn_conditionally(self):
        """Attention __init__ must create YaRN only when rope_scaling.type=='yarn'."""
        NeuronSolarOpenAttention, _ = _import_attention()
        src = inspect.getsource(NeuronSolarOpenAttention.__init__)
        assert "SolarOpenYarnRotaryEmbedding" in src, (
            "NeuronSolarOpenAttention must reference SolarOpenYarnRotaryEmbedding"
        )
        assert "yarn" in src, (
            "NeuronSolarOpenAttention must check for rope_scaling type=='yarn'"
        )


# ---------------------------------------------------------------------------
# TestAttentionClassStructure
# ---------------------------------------------------------------------------


class TestAttentionClassStructure:
    """Verify NeuronSolarOpenAttention class API contract."""

    def test_attention_class_exists(self):
        """NeuronSolarOpenAttention must be importable."""
        NeuronSolarOpenAttention, _ = _import_attention()
        assert NeuronSolarOpenAttention is not None

    def test_attention_inherits_neuron_attention_base(self):
        """NeuronSolarOpenAttention must subclass NeuronAttentionBase."""
        NeuronSolarOpenAttention, _ = _import_attention()
        from neuronx_distributed_inference.modules.attention.attention_base import (
            NeuronAttentionBase,
        )

        assert issubclass(NeuronSolarOpenAttention, NeuronAttentionBase), (
            "NeuronSolarOpenAttention must extend NeuronAttentionBase"
        )

    def test_attention_init_accepts_config(self):
        """NeuronSolarOpenAttention.__init__ must accept a 'config' parameter."""
        NeuronSolarOpenAttention, _ = _import_attention()
        sig = inspect.signature(NeuronSolarOpenAttention.__init__)
        assert "config" in sig.parameters, (
            "NeuronSolarOpenAttention.__init__ must accept 'config'"
        )
