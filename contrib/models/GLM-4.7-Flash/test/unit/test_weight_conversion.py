# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for GLM-4.7-Flash state dict conversion.

Tests:
1. State dict key renaming (router, e_score_correction_bias)
2. Expert weight fusion (gate_proj + up_proj -> gate_up_proj, down_proj stacking)
3. Dense layers are skipped (first_k_dense_replace=1, only layer 0)
4. Rank utility tensors added
5. MTP layer weights removed
6. No FP8 dequantization (native BF16)
"""

import unittest
from unittest.mock import MagicMock

import torch

from src.modeling_glm4_moe_lite import (
    convert_glm4_moe_lite_hf_to_neuron_state_dict,
)


def _make_mock_config(
    num_hidden_layers=5,
    num_local_experts=8,
    tp_degree=4,
    first_k_dense_replace=1,
    hidden_size=64,
    intermediate_size=32,
):
    """Create a lightweight mock config for state dict conversion tests."""
    config = MagicMock()
    config.num_hidden_layers = num_hidden_layers
    config.num_local_experts = num_local_experts
    config.first_k_dense_replace = first_k_dense_replace
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.neuron_config = MagicMock()
    config.neuron_config.tp_degree = tp_degree
    return config


def _make_hf_moe_state_dict(
    layer_idx, num_experts=8, hidden_size=64, intermediate_size=32, dtype=torch.float32
):
    """Create a fake HF MoE layer state dict with router + experts."""
    sd = {}
    # Router
    sd[f"layers.{layer_idx}.mlp.gate.weight"] = torch.randn(
        num_experts, hidden_size, dtype=dtype
    )
    sd[f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(
        num_experts, dtype=dtype
    )
    # Experts
    for e in range(num_experts):
        sd[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype
        )
        sd[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype
        )
        sd[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size, dtype=dtype
        )
    # Shared experts (pass through unchanged)
    sd[f"layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, dtype=dtype
    )
    sd[f"layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, dtype=dtype
    )
    sd[f"layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
        hidden_size, intermediate_size, dtype=dtype
    )
    return sd


class TestStateDictConversion(unittest.TestCase):
    """Tests for convert_glm4_moe_lite_hf_to_neuron_state_dict."""

    def test_rank_util_tensors_added(self):
        """Rank utility tensors must be added for TP sharding."""
        config = _make_mock_config(num_hidden_layers=2, first_k_dense_replace=2)
        sd = {}
        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        assert "rank_util.rank" in result
        torch.testing.assert_close(
            result["rank_util.rank"], torch.arange(0, 4, dtype=torch.int32)
        )
        for i in range(2):
            assert f"layers.{i}.self_attn.rank_util.rank" in result

    def test_dense_layer_skipped(self):
        """Layer 0 (dense, first_k_dense_replace=1) should not have MoE conversion."""
        config = _make_mock_config(
            num_hidden_layers=3, first_k_dense_replace=1, num_local_experts=4
        )
        sd = {}
        # Add dense layer weights (layer 0)
        sd["layers.0.mlp.gate_proj.weight"] = torch.randn(32, 64)
        sd["layers.0.mlp.up_proj.weight"] = torch.randn(32, 64)
        sd["layers.0.mlp.down_proj.weight"] = torch.randn(64, 32)
        # Add MoE layers (layers 1-2)
        sd.update(_make_hf_moe_state_dict(1, num_experts=4))
        sd.update(_make_hf_moe_state_dict(2, num_experts=4))

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        # Dense layer weights should be unchanged
        assert "layers.0.mlp.gate_proj.weight" in result

        # MoE layers should have converted keys
        assert "layers.1.mlp.router.linear_router.weight" in result
        assert "layers.1.mlp.expert_mlps.mlp_op.gate_up_proj.weight" in result
        assert "layers.2.mlp.router.linear_router.weight" in result

    def test_router_rename(self):
        """gate.weight -> router.linear_router.weight"""
        config = _make_mock_config(
            num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4
        )
        sd = _make_hf_moe_state_dict(0, num_experts=4)

        original_router_w = sd["layers.0.mlp.gate.weight"].clone()
        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.mlp.gate.weight" not in result
        assert "layers.0.mlp.router.linear_router.weight" in result
        torch.testing.assert_close(
            result["layers.0.mlp.router.linear_router.weight"], original_router_w
        )

    def test_e_score_correction_bias_renamed(self):
        """gate.e_score_correction_bias should be renamed to router.e_score_correction_bias."""
        config = _make_mock_config(
            num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4
        )
        sd = _make_hf_moe_state_dict(0, num_experts=4)
        original_bias = sd["layers.0.mlp.gate.e_score_correction_bias"].clone()

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.mlp.gate.e_score_correction_bias" not in result
        assert "layers.0.mlp.router.e_score_correction_bias" in result
        torch.testing.assert_close(
            result["layers.0.mlp.router.e_score_correction_bias"], original_bias
        )

    def test_expert_gate_up_fusion(self):
        """Per-expert gate_proj + up_proj -> fused gate_up_proj [num_experts, hidden, 2*intermediate]."""
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32
        config = _make_mock_config(
            num_hidden_layers=1,
            first_k_dense_replace=0,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        sd = _make_hf_moe_state_dict(
            0,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        # Save originals for verification
        gate_projs = []
        up_projs = []
        for e in range(num_experts):
            gate_projs.append(sd[f"layers.0.mlp.experts.{e}.gate_proj.weight"].clone())
            up_projs.append(sd[f"layers.0.mlp.experts.{e}.up_proj.weight"].clone())

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        fused_key = "layers.0.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
        assert fused_key in result
        fused = result[fused_key]
        assert fused.shape == (num_experts, hidden_size, 2 * intermediate_size)

        # Verify fusion: gate_proj.T in first half, up_proj.T in second half
        for e in range(num_experts):
            torch.testing.assert_close(fused[e, :, :intermediate_size], gate_projs[e].T)
            torch.testing.assert_close(fused[e, :, intermediate_size:], up_projs[e].T)

        # Per-expert keys should be removed
        for e in range(num_experts):
            assert f"layers.0.mlp.experts.{e}.gate_proj.weight" not in result
            assert f"layers.0.mlp.experts.{e}.up_proj.weight" not in result

    def test_expert_down_proj_stacking(self):
        """Per-expert down_proj -> stacked [num_experts, intermediate, hidden]."""
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32
        config = _make_mock_config(
            num_hidden_layers=1,
            first_k_dense_replace=0,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        sd = _make_hf_moe_state_dict(
            0,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        down_projs = []
        for e in range(num_experts):
            down_projs.append(sd[f"layers.0.mlp.experts.{e}.down_proj.weight"].clone())

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        stacked_key = "layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"
        assert stacked_key in result
        stacked = result[stacked_key]
        assert stacked.shape == (num_experts, intermediate_size, hidden_size)

        for e in range(num_experts):
            torch.testing.assert_close(stacked[e], down_projs[e].T)
            assert f"layers.0.mlp.experts.{e}.down_proj.weight" not in result

    def test_shared_experts_unchanged(self):
        """Shared expert weights should pass through without renaming."""
        config = _make_mock_config(
            num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4
        )
        sd = _make_hf_moe_state_dict(0, num_experts=4)

        original_shared_gate = sd[
            "layers.0.mlp.shared_experts.gate_proj.weight"
        ].clone()
        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.mlp.shared_experts.gate_proj.weight" in result
        torch.testing.assert_close(
            result["layers.0.mlp.shared_experts.gate_proj.weight"], original_shared_gate
        )

    def test_mtp_layer_removed(self):
        """MTP layer weights (layer N where N=num_hidden_layers) should be removed."""
        num_hidden_layers = 3
        config = _make_mock_config(
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=0,
            num_local_experts=4,
        )
        sd = _make_hf_moe_state_dict(0, num_experts=4)
        sd.update(_make_hf_moe_state_dict(1, num_experts=4))
        sd.update(_make_hf_moe_state_dict(2, num_experts=4))
        # Add MTP layer (layer 3 = num_hidden_layers)
        sd[f"layers.{num_hidden_layers}.embed_tokens.weight"] = torch.randn(
            154880, 2048
        )
        sd[f"layers.{num_hidden_layers}.enorm.weight"] = torch.randn(2048)

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        # MTP keys should be gone
        assert f"layers.{num_hidden_layers}.embed_tokens.weight" not in result
        assert f"layers.{num_hidden_layers}.enorm.weight" not in result
        # Regular layer keys should still exist
        assert "layers.0.mlp.router.linear_router.weight" in result

    def test_no_fp8_handling(self):
        """GLM-4.7-Flash uses native BF16 — no FP8 scale_inv keys expected."""
        config = _make_mock_config(
            num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4
        )
        sd = _make_hf_moe_state_dict(0, num_experts=4, dtype=torch.bfloat16)

        result = convert_glm4_moe_lite_hf_to_neuron_state_dict(sd, config)

        # No scale_inv keys should exist
        scale_keys = [k for k in result if "scale_inv" in k]
        self.assertEqual(len(scale_keys), 0)


if __name__ == "__main__":
    unittest.main()
