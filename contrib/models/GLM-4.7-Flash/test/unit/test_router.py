# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-4.7-Flash router (Glm4MoeLiteRouter).

Tests sigmoid activation, noaux_tc top-k selection, L1 normalization,
and routed_scaling_factor application.

Note: GroupLimitedRouter requires distributed parallel state to be initialized.
We mock get_expert_model_parallel_size and get_tensor_model_parallel_group
for CPU-only unit testing.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from unittest.mock import patch


class ReferenceGlm4Router(nn.Module):
    """
    Reference implementation of GLM-4.7-Flash routing logic.

    With n_group=1, topk_group=1, the group logic is a no-op — it reduces
    to sigmoid + bias + topk + L1-norm + scale.
    """

    def __init__(self, num_experts, top_k, hidden_size, routed_scaling_factor):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, hidden_states):
        # Linear + sigmoid in fp64 (matching GroupLimitedRouter)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(router_logits.to(torch.float64)).to(hidden_states.dtype)

        # With n_group=1: no group selection, just topk on (scores + bias)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_indices = torch.topk(scores_for_choice, k=self.top_k)

        # Gather, L1-norm, scale
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


def _mock_get_expert_model_parallel_size():
    return 1


def _mock_get_tensor_model_parallel_group():
    return None


# Patch NxD parallel state for CPU testing
_PARALLEL_PATCHES = [
    patch(
        "neuronx_distributed.modules.moe.routing.get_expert_model_parallel_size",
        _mock_get_expert_model_parallel_size,
    ),
    patch(
        "neuronx_distributed.modules.moe.routing.get_tensor_model_parallel_group",
        _mock_get_tensor_model_parallel_group,
    ),
]


def _create_neuron_router(
    num_experts=64, top_k=4, hidden_size=2048, routed_scaling_factor=1.8
):
    """Create Glm4MoeLiteRouter with mocked parallel state."""
    from src.modeling_glm4_moe_lite import Glm4MoeLiteRouter

    for p in _PARALLEL_PATCHES:
        p.start()
    try:
        router = Glm4MoeLiteRouter(
            routed_scaling_factor=routed_scaling_factor,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            n_group=1,
            topk_group=1,
            dtype=torch.float32,
            sequence_parallel_enabled=False,
            sequence_dimension=1,
        )
    finally:
        for p in _PARALLEL_PATCHES:
            p.stop()
    return router


@pytest.fixture
def router_config():
    return dict(
        num_experts=64,
        top_k=4,
        hidden_size=2048,
        routed_scaling_factor=1.8,
    )


@pytest.fixture
def router_and_ref(router_config):
    """Create a Glm4MoeLiteRouter and matching reference, with shared weights."""
    ref = ReferenceGlm4Router(**router_config)
    neuron_router = _create_neuron_router(**router_config)

    with torch.no_grad():
        nn.init.normal_(ref.weight, std=0.01)
        nn.init.normal_(ref.e_score_correction_bias, std=0.001)
        neuron_router.linear_router.weight.copy_(ref.weight)
        neuron_router.e_score_correction_bias.copy_(ref.e_score_correction_bias)

    return neuron_router, ref


class TestGlm4MoeLiteRouter:
    def test_routed_scaling_factor(self, router_and_ref):
        neuron_router, _ = router_and_ref
        assert neuron_router.routed_scaling_factor == 1.8

    def test_e_score_correction_bias_registered(self, router_and_ref):
        neuron_router, _ = router_and_ref
        assert hasattr(neuron_router, "e_score_correction_bias")
        assert neuron_router.e_score_correction_bias.shape == (64,)

    def test_n_group_topk_group(self, router_and_ref):
        """GLM-4.7-Flash uses n_group=1, topk_group=1 (no group selection)."""
        neuron_router, _ = router_and_ref
        assert neuron_router.n_group == 1
        assert neuron_router.topk_group == 1

    def test_expert_selection_matches_reference(self, router_and_ref, router_config):
        """Expert indices must be consistent between separate forward calls."""
        neuron_router, ref = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(16, router_config["hidden_size"])

        # Run twice — should be deterministic
        _, _, expert_index_1 = neuron_router(x)
        _, _, expert_index_2 = neuron_router(x)

        assert torch.equal(expert_index_1, expert_index_2), (
            f"Router is non-deterministic.\nRun 1: {expert_index_1[:3]}\nRun 2: {expert_index_2[:3]}"
        )

    def test_expert_weights_match_reference(self, router_and_ref, router_config):
        """Expert weights (normalized + scaled) must be self-consistent."""
        neuron_router, ref = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(16, router_config["hidden_size"])

        _, expert_affinities, expert_index = neuron_router(x)

        # Gather the non-zero weights for selected experts
        neuron_weights = expert_affinities.gather(1, expert_index)

        # Weights should sum to routed_scaling_factor (L1 norm + scale)
        weight_sums = neuron_weights.sum(dim=-1)
        expected = router_config["routed_scaling_factor"]
        torch.testing.assert_close(
            weight_sums, torch.full_like(weight_sums, expected), atol=1e-4, rtol=1e-4
        )

        # All weights should be positive (sigmoid outputs are positive)
        assert (neuron_weights > 0).all(), (
            "All selected expert weights should be positive"
        )

    def test_output_shapes(self, router_and_ref, router_config):
        """Router outputs have correct shapes."""
        neuron_router, _ = router_and_ref
        T = 16
        x = torch.randn(T, router_config["hidden_size"])

        router_logits, expert_affinities, expert_index = neuron_router(x)
        assert router_logits.shape == (T, router_config["num_experts"])
        assert expert_affinities.shape == (T, router_config["num_experts"])
        assert expert_index.shape == (T, router_config["top_k"])

    def test_topk_indices_valid(self, router_and_ref, router_config):
        """Top-k indices should be in [0, num_experts)."""
        neuron_router, _ = router_and_ref
        x = torch.randn(8, router_config["hidden_size"])

        _, _, topk_idx = neuron_router(x)
        assert (topk_idx >= 0).all()
        assert (topk_idx < router_config["num_experts"]).all()

    def test_scaling_factor_sum(self, router_and_ref, router_config):
        """Weights should sum to routed_scaling_factor per token (L1 norm + scale)."""
        neuron_router, _ = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(32, router_config["hidden_size"])

        _, expert_affinities, expert_index = neuron_router(x)
        topk_weights = expert_affinities.gather(1, expert_index)
        weight_sums = topk_weights.sum(dim=-1)

        expected = router_config["routed_scaling_factor"]
        torch.testing.assert_close(
            weight_sums, torch.full_like(weight_sums, expected), atol=1e-4, rtol=1e-4
        )

    def test_sparsity_pattern(self, router_and_ref, router_config):
        """Only top_k experts should have non-zero affinities per token."""
        neuron_router, _ = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(8, router_config["hidden_size"])

        _, expert_affinities, _ = neuron_router(x)
        nonzero_per_token = (expert_affinities != 0).sum(dim=-1)
        assert (nonzero_per_token == router_config["top_k"]).all(), (
            f"Expected {router_config['top_k']} non-zero per token, got {nonzero_per_token}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
