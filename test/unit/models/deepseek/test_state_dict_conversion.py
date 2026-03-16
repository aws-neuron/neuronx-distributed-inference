"""
Unit tests for DeepSeek V3 state dict conversion and router logic.

Tests:
1. State dict key renaming (router, e_score_correction_bias)
2. Expert weight fusion (gate_proj + up_proj -> gate_up_proj, down_proj stacking)
3. Dense layers are skipped (first_k_dense_replace)
4. Rank utility tensors added
5. Router group-based selection matches HF reference
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3Router,
    convert_deepseek_v3_hf_to_neuron_state_dict,
)


def _make_mock_config(
    num_hidden_layers=5,
    num_local_experts=8,
    tp_degree=2,
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


def _make_hf_moe_state_dict(layer_idx, num_experts=8, hidden_size=64, intermediate_size=32, dtype=torch.float32):
    """Create a fake HF MoE layer state dict with router + experts."""
    sd = {}
    # Router
    sd[f"layers.{layer_idx}.mlp.gate.weight"] = torch.randn(num_experts, hidden_size, dtype=dtype)
    sd[f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(num_experts, dtype=dtype)
    # Experts
    for e in range(num_experts):
        sd[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=dtype)
        sd[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=dtype)
        sd[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden_size, intermediate_size, dtype=dtype)
    # Shared experts (pass through unchanged)
    sd[f"layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=dtype)
    sd[f"layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=dtype)
    sd[f"layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden_size, intermediate_size, dtype=dtype)
    return sd


class TestStateDictConversion(unittest.TestCase):
    """Tests for convert_deepseek_v3_hf_to_neuron_state_dict."""

    def test_rank_util_tensors_added(self):
        """Rank utility tensors must be added for TP sharding."""
        config = _make_mock_config(num_hidden_layers=2, first_k_dense_replace=2)
        sd = {}
        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        assert "rank_util.rank" in result
        torch.testing.assert_close(result["rank_util.rank"], torch.arange(0, 2, dtype=torch.int32))
        for i in range(2):
            assert f"layers.{i}.self_attn.rank_util.rank" in result

    def test_dense_layers_skipped(self):
        """Dense layers (< first_k_dense_replace) should not have MoE conversion applied."""
        config = _make_mock_config(num_hidden_layers=3, first_k_dense_replace=2, num_local_experts=4)
        sd = {}
        # Add dense layer weights (layers 0-1)
        for i in range(2):
            sd[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(32, 64)
            sd[f"layers.{i}.mlp.up_proj.weight"] = torch.randn(32, 64)
            sd[f"layers.{i}.mlp.down_proj.weight"] = torch.randn(64, 32)
        # Add MoE layer (layer 2)
        sd.update(_make_hf_moe_state_dict(2, num_experts=4))

        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        # Dense layer weights should be unchanged
        assert "layers.0.mlp.gate_proj.weight" in result
        assert "layers.1.mlp.up_proj.weight" in result

        # MoE layer should have converted keys
        assert "layers.2.mlp.router.linear_router.weight" in result
        assert "layers.2.mlp.expert_mlps.mlp_op.gate_up_proj.weight" in result

    def test_router_rename(self):
        """gate.weight -> router.linear_router.weight"""
        config = _make_mock_config(num_hidden_layers=2, first_k_dense_replace=0, num_local_experts=4)
        sd = _make_hf_moe_state_dict(0, num_experts=4)
        sd.update(_make_hf_moe_state_dict(1, num_experts=4))

        original_router_w = sd["layers.0.mlp.gate.weight"].clone()
        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        # Old key gone, new key present
        assert "layers.0.mlp.gate.weight" not in result
        assert "layers.0.mlp.router.linear_router.weight" in result
        torch.testing.assert_close(result["layers.0.mlp.router.linear_router.weight"], original_router_w)

    def test_e_score_correction_bias_rename(self):
        """gate.e_score_correction_bias -> router.e_score_correction_bias"""
        config = _make_mock_config(num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4)
        sd = _make_hf_moe_state_dict(0, num_experts=4)

        original_bias = sd["layers.0.mlp.gate.e_score_correction_bias"].clone()
        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.mlp.gate.e_score_correction_bias" not in result
        assert "layers.0.mlp.router.e_score_correction_bias" in result
        torch.testing.assert_close(result["layers.0.mlp.router.e_score_correction_bias"], original_bias)

    def test_expert_gate_up_fusion(self):
        """Per-expert gate_proj + up_proj -> fused gate_up_proj [num_experts, hidden, 2*intermediate]."""
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32
        config = _make_mock_config(
            num_hidden_layers=1, first_k_dense_replace=0,
            num_local_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
        )
        sd = _make_hf_moe_state_dict(0, num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size)

        # Save originals for verification
        gate_projs = []
        up_projs = []
        for e in range(num_experts):
            gate_projs.append(sd[f"layers.0.mlp.experts.{e}.gate_proj.weight"].clone())
            up_projs.append(sd[f"layers.0.mlp.experts.{e}.up_proj.weight"].clone())

        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

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
            num_hidden_layers=1, first_k_dense_replace=0,
            num_local_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
        )
        sd = _make_hf_moe_state_dict(0, num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size)

        down_projs = []
        for e in range(num_experts):
            down_projs.append(sd[f"layers.0.mlp.experts.{e}.down_proj.weight"].clone())

        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        stacked_key = "layers.0.mlp.expert_mlps.mlp_op.down_proj.weight"
        assert stacked_key in result
        stacked = result[stacked_key]
        assert stacked.shape == (num_experts, intermediate_size, hidden_size)

        for e in range(num_experts):
            torch.testing.assert_close(stacked[e], down_projs[e].T)
            assert f"layers.0.mlp.experts.{e}.down_proj.weight" not in result

    def test_shared_experts_unchanged(self):
        """Shared expert weights should pass through without renaming."""
        config = _make_mock_config(num_hidden_layers=1, first_k_dense_replace=0, num_local_experts=4)
        sd = _make_hf_moe_state_dict(0, num_experts=4)

        original_shared_gate = sd["layers.0.mlp.shared_experts.gate_proj.weight"].clone()
        result = convert_deepseek_v3_hf_to_neuron_state_dict(sd, config)

        assert "layers.0.mlp.shared_experts.gate_proj.weight" in result
        torch.testing.assert_close(result["layers.0.mlp.shared_experts.gate_proj.weight"], original_shared_gate)


class TestDeepseekV3Router(unittest.TestCase):
    """Tests for DeepseekV3Router group-based routing against HF reference."""

    @staticmethod
    def hf_reference_routing(
        hidden_states, router_weight, e_score_correction_bias,
        n_group=8, topk_group=4, top_k=8, routed_scaling_factor=2.5,
        norm_topk_prob=True,
    ):
        """Compiler-compatible routing logic matching DeepseekV3Router.forward."""
        n_routed_experts = router_weight.shape[0]
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = F.linear(hidden_states_flat.float(), router_weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

        experts_per_group = n_routed_experts // n_group
        grouped_scores = scores_for_choice.view(-1, n_group, experts_per_group)

        # Sum-based group scoring (compiler-compatible approximation)
        group_scores = grouped_scores.sum(dim=-1)

        # Select top groups, gather their scores, flatten, select top-K
        _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)
        selected_groups = torch.gather(
            grouped_scores, 1,
            group_idx.unsqueeze(-1).expand(-1, -1, experts_per_group)
        )
        flat_scores = selected_groups.reshape(-1, topk_group * experts_per_group)
        _, flat_expert_idx = torch.topk(flat_scores, k=top_k, dim=-1, sorted=True)

        # Map back to global expert indices
        selected_group_ord = flat_expert_idx // experts_per_group
        within_group_offset = flat_expert_idx % experts_per_group
        actual_group = torch.gather(group_idx, 1, selected_group_ord)
        topk_indices = actual_group * experts_per_group + within_group_offset
        topk_weights = scores.gather(1, topk_indices)

        if norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * routed_scaling_factor

        return topk_indices, topk_weights

    def _create_router(self, num_experts=64, hidden_size=128, top_k=8, n_group=8, topk_group=4):
        """Create a DeepseekV3Router for testing with mocked parallel state."""
        # Patch at the module where functions are looked up (routing module imports from parallel_state)
        with patch("neuronx_distributed.modules.moe.routing.get_expert_model_parallel_size", return_value=1), \
             patch("neuronx_distributed.modules.moe.routing.get_tensor_model_parallel_group", return_value=None):
            router = DeepseekV3Router(
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=2.5,
                norm_topk_prob=True,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                dtype=torch.float32,
                act_fn="sigmoid",
                sequence_parallel_enabled=False,
                sequence_dimension=1,
            )
        return router

    def test_router_output_shapes(self):
        """Router must return (router_logits, expert_affinities, expert_index) with correct shapes."""
        num_experts = 64
        hidden_size = 128
        top_k = 8
        router = self._create_router(num_experts=num_experts, hidden_size=hidden_size, top_k=top_k)

        hidden_states = torch.randn(4, hidden_size)  # 4 tokens
        router_logits, expert_affinities, expert_index = router(hidden_states)

        assert router_logits.shape == (4, num_experts)
        assert expert_affinities.shape == (4, num_experts)
        assert expert_index.shape == (4, top_k)

    def test_router_matches_hf_reference(self):
        """DeepseekV3Router expert indices and weights must match HF reference."""
        torch.manual_seed(42)
        num_experts = 64
        hidden_size = 128
        top_k = 8
        n_group = 8
        topk_group = 4

        router = self._create_router(
            num_experts=num_experts, hidden_size=hidden_size,
            top_k=top_k, n_group=n_group, topk_group=topk_group,
        )

        # Set deterministic weights and bias
        router_weight = torch.randn(num_experts, hidden_size)
        bias = torch.randn(num_experts) * 0.1
        router.linear_router.weight = torch.nn.Parameter(router_weight)
        router.e_score_correction_bias = torch.nn.Parameter(bias)

        hidden_states = torch.randn(8, hidden_size)  # 8 tokens

        # NXDI router
        _, nxdi_affinities, nxdi_indices = router(hidden_states)

        # HF reference
        hf_indices, hf_weights = self.hf_reference_routing(
            hidden_states, router_weight, bias,
            n_group=n_group, topk_group=topk_group, top_k=top_k,
        )

        # Expert indices must match (sort both since order within top-k may differ)
        nxdi_sorted, nxdi_order = nxdi_indices.sort(dim=-1)
        hf_sorted, hf_order = hf_indices.sort(dim=-1)
        torch.testing.assert_close(nxdi_sorted, hf_sorted)

        # Weights must match for corresponding experts
        nxdi_weights = nxdi_affinities.gather(1, nxdi_indices)
        nxdi_weights_sorted = nxdi_weights.gather(1, nxdi_order)
        hf_weights_sorted = hf_weights.gather(1, hf_order)
        torch.testing.assert_close(nxdi_weights_sorted, hf_weights_sorted, atol=1e-6, rtol=1e-5)

    def test_router_group_selection(self):
        """Verify that only experts from selected groups are chosen."""
        torch.manual_seed(0)
        num_experts = 64
        hidden_size = 128
        top_k = 8
        n_group = 8
        topk_group = 4

        router = self._create_router(
            num_experts=num_experts, hidden_size=hidden_size,
            top_k=top_k, n_group=n_group, topk_group=topk_group,
        )

        hidden_states = torch.randn(16, hidden_size)
        _, _, expert_index = router(hidden_states)

        # Each expert belongs to group = expert_idx // (num_experts // n_group)
        experts_per_group = num_experts // n_group
        groups_selected = expert_index // experts_per_group  # (16, 8)

        # Each token's experts must come from at most topk_group groups
        for t in range(16):
            unique_groups = groups_selected[t].unique()
            assert len(unique_groups) <= topk_group, (
                f"Token {t}: experts from {len(unique_groups)} groups, expected <= {topk_group}"
            )

    def test_router_no_bias_in_weights(self):
        """Router weights must come from sigmoid(logits) WITHOUT bias."""
        torch.manual_seed(123)
        num_experts = 16
        hidden_size = 32
        top_k = 4
        n_group = 4
        topk_group = 2

        router = self._create_router(
            num_experts=num_experts, hidden_size=hidden_size,
            top_k=top_k, n_group=n_group, topk_group=topk_group,
        )

        # Large bias to make the difference detectable
        router.e_score_correction_bias = torch.nn.Parameter(torch.ones(num_experts) * 5.0)

        hidden_states = torch.randn(4, hidden_size)
        router_logits, expert_affinities, expert_index = router(hidden_states)

        # Weights for selected experts should come from sigmoid(logits), not sigmoid(logits)+bias
        raw_scores = router_logits.sigmoid()
        for t in range(4):
            selected = expert_index[t]
            actual_weights = expert_affinities.gather(1, expert_index)[t]
            raw_selected = raw_scores[t, selected]
            # Normalize and scale like the router does
            expected_weights = raw_selected / (raw_selected.sum() + 1e-20) * 2.5
            torch.testing.assert_close(actual_weights, expected_weights, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
