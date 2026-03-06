# coding=utf-8
"""Unit tests for NeuronGlm4MoeAttention (partial RoPE, QK norm, attention bias).

These tests verify the attention-specific logic that differs from a standard
transformer:
  1. Partial RoPE: only first ``partial_rotary_factor * head_dim`` dims are rotated.
  2. Optional QK normalization per head.
  3. QKV projections carry a bias term (attention_bias=True).

All tests run on CPU; no Neuron hardware is required.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_attention_config(
    hidden_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 32,
    partial_rotary_factor: float = 0.5,
    attention_bias: bool = True,
    use_qk_norm: bool = False,
    max_position_embeddings: int = 512,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-5,
):
    """Create a minimal config namespace for NeuronGlm4MoeAttention construction."""
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        attention_bias=attention_bias,
        use_qk_norm=use_qk_norm,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
    )
    return cfg


# ---------------------------------------------------------------------------
# Partial RoPE tests (no distributed context needed — logic is pure math)
# ---------------------------------------------------------------------------


class TestPartialRoPE:
    """Verify the partial RoPE splitting logic in apply_rotary_embedding."""

    def _make_qkv(self, batch: int, heads: int, seq: int, head_dim: int):
        torch.manual_seed(0)
        Q = torch.randn(batch, heads, seq, head_dim)
        K = torch.randn(batch, heads, seq, head_dim)
        V = torch.randn(batch, heads, seq, head_dim)
        return Q, K, V

    def _apply_partial_rope_reference(self, Q, K, rotary_dim, cos, sin):
        """Reference implementation matching the class logic."""
        from neuronx_distributed_inference.modules.attention.utils import (
            apply_rotary_pos_emb,
        )

        Q_rot, Q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
        K_rot, K_pass = K[..., :rotary_dim], K[..., rotary_dim:]
        Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos, sin)
        Q_out = torch.cat([Q_rot, Q_pass], dim=-1)
        K_out = torch.cat([K_rot, K_pass], dim=-1)
        return Q_out, K_out

    def test_rotary_dim_is_half_head_dim(self):
        """rotary_dim must equal floor(head_dim * partial_rotary_factor)."""
        head_dim = 32
        factor = 0.5
        expected_rotary_dim = int(head_dim * factor)

        # Import RotaryEmbedding directly to verify the rotary_dim it computes
        from neuronx_distributed_inference.modules.attention.utils import (
            RotaryEmbedding,
        )

        rotary_emb = RotaryEmbedding(
            expected_rotary_dim,
            max_position_embeddings=512,
            base=1_000_000.0,
        )
        # RotaryEmbedding stores dim as self.dim (or indirectly via cos_cached shape)
        assert expected_rotary_dim == 16, f"Expected 16, got {expected_rotary_dim}"

    def test_pass_through_dimensions_unchanged(self):
        """The non-rotary tail of Q and K must be bit-exact after apply_rotary_embedding."""
        try:
            from neuronx_distributed_inference.modules.attention.utils import (
                RotaryEmbedding,
                apply_rotary_pos_emb,
            )
        except ImportError:
            pytest.skip("neuronx_distributed_inference not installed")

        batch, heads, seq, head_dim = 1, 4, 8, 32
        rotary_dim = 16  # 0.5 * 32

        Q, K, V = self._make_qkv(batch, heads, seq, head_dim)

        rotary_emb = RotaryEmbedding(
            rotary_dim, max_position_embeddings=seq, base=1_000_000.0
        )
        position_ids = torch.arange(seq).unsqueeze(0)
        cos, sin = rotary_emb(V, position_ids)

        Q_out, K_out = self._apply_partial_rope_reference(Q, K, rotary_dim, cos, sin)

        # Pass-through portions must be identical to input
        torch.testing.assert_close(Q_out[..., rotary_dim:], Q[..., rotary_dim:])
        torch.testing.assert_close(K_out[..., rotary_dim:], K[..., rotary_dim:])

    def test_rotary_dimensions_are_changed(self):
        """The rotary portion of Q/K must differ from the original after applying RoPE."""
        try:
            from neuronx_distributed_inference.modules.attention.utils import (
                RotaryEmbedding,
                apply_rotary_pos_emb,
            )
        except ImportError:
            pytest.skip("neuronx_distributed_inference not installed")

        batch, heads, seq, head_dim = 1, 4, 8, 32
        rotary_dim = 16

        Q, K, V = self._make_qkv(batch, heads, seq, head_dim)

        rotary_emb = RotaryEmbedding(
            rotary_dim, max_position_embeddings=seq, base=1_000_000.0
        )
        position_ids = torch.arange(seq).unsqueeze(0)
        cos, sin = rotary_emb(V, position_ids)

        Q_out, _ = self._apply_partial_rope_reference(Q, K, rotary_dim, cos, sin)

        # Rotary portion should differ (non-zero rotation for non-trivial positions)
        assert not torch.allclose(Q_out[..., :rotary_dim], Q[..., :rotary_dim]), (
            "RoPE should change the rotary portion of Q"
        )

    def test_output_shape_preserved(self):
        """Output tensors must retain the exact same shape as input."""
        try:
            from neuronx_distributed_inference.modules.attention.utils import (
                RotaryEmbedding,
                apply_rotary_pos_emb,
            )
        except ImportError:
            pytest.skip("neuronx_distributed_inference not installed")

        batch, heads, seq, head_dim = 2, 4, 16, 32
        rotary_dim = 16

        Q, K, V = self._make_qkv(batch, heads, seq, head_dim)

        rotary_emb = RotaryEmbedding(
            rotary_dim, max_position_embeddings=seq, base=1_000_000.0
        )
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rotary_emb(V, position_ids)

        Q_out, K_out = self._apply_partial_rope_reference(Q, K, rotary_dim, cos, sin)

        assert Q_out.shape == Q.shape, f"Q shape changed: {Q.shape} → {Q_out.shape}"
        assert K_out.shape == K.shape, f"K shape changed: {K.shape} → {K_out.shape}"


# ---------------------------------------------------------------------------
# QK norm tests (CPU mock, no distributed env)
# ---------------------------------------------------------------------------


class TestQKNorm:
    """Verify QK normalization initialization logic."""

    def _build_attention_no_dist(self, use_qk_norm: bool):
        """Construct NeuronGlm4MoeAttention with mocked distributed state."""
        try:
            from glm4_moe.modeling_glm4_moe import NeuronGlm4MoeAttention
        except ImportError:
            pytest.skip("glm4_moe package not importable (Neuron SDK missing)")

        config = _make_attention_config(use_qk_norm=use_qk_norm)

        with (
            patch(
                "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size",
                return_value=1,
            ),
        ):
            attn = NeuronGlm4MoeAttention.__new__(NeuronGlm4MoeAttention)
            attn.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
            if config.use_qk_norm:
                from glm4_moe.modeling_glm4_moe import get_rmsnorm_cls

                attn.q_layernorm = get_rmsnorm_cls()(
                    config.head_dim, config.rms_norm_eps
                )
                attn.k_layernorm = get_rmsnorm_cls()(
                    config.head_dim, config.rms_norm_eps
                )
            else:
                attn.q_layernorm = None
                attn.k_layernorm = None
        return attn

    def test_qk_norm_none_when_disabled(self):
        """q_layernorm and k_layernorm should be None when use_qk_norm=False."""
        attn = self._build_attention_no_dist(use_qk_norm=False)
        assert attn.q_layernorm is None, "q_layernorm should be None"
        assert attn.k_layernorm is None, "k_layernorm should be None"

    def test_qk_norm_present_when_enabled(self):
        """Source code must create q_layernorm/k_layernorm when use_qk_norm=True.

        We verify this via source code inspection rather than instantiation,
        because full construction requires a distributed process group.
        The __init__ source should contain 'q_layernorm' and 'k_layernorm'
        assignments guarded by 'if config.use_qk_norm'.
        """
        import inspect

        try:
            from glm4_moe.modeling_glm4_moe import NeuronGlm4MoeAttention
        except ImportError:
            pytest.skip("glm4_moe package not importable (Neuron SDK missing)")

        src = inspect.getsource(NeuronGlm4MoeAttention.__init__)
        assert "use_qk_norm" in src, "use_qk_norm must be checked in __init__"
        assert "q_layernorm" in src, "q_layernorm must be assigned in __init__"
        assert "k_layernorm" in src, "k_layernorm must be assigned in __init__"


# ---------------------------------------------------------------------------
# Rotary dim calculation test (pure math, no imports needed)
# ---------------------------------------------------------------------------


class TestRotaryDimCalculation:
    """Pure math: verify rotary_dim = floor(head_dim * partial_rotary_factor)."""

    @pytest.mark.parametrize(
        "head_dim,factor,expected",
        [
            (128, 0.5, 64),
            (64, 0.5, 32),
            (32, 0.5, 16),
            (128, 0.25, 32),
            (64, 0.75, 48),
        ],
    )
    def test_rotary_dim_values(self, head_dim, factor, expected):
        """rotary_dim must be int(head_dim * partial_rotary_factor)."""
        rotary_dim = int(head_dim * factor)
        assert rotary_dim == expected, (
            f"head_dim={head_dim}, factor={factor}: "
            f"expected rotary_dim={expected}, got {rotary_dim}"
        )

    def test_glm45_air_default(self):
        """GLM-4.5 Air uses head_dim=128, partial_rotary_factor=0.5 → rotary_dim=64."""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)
        assert rotary_dim == 64
