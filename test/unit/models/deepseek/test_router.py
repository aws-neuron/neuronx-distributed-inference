"""Unit tests for DeepseekV3Router (group-based expert selection).

Tests the custom router against a pure-PyTorch reference implementation
matching the HuggingFace DeepseekV3TopkRouter logic.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from unittest.mock import patch


class ReferenceDeepseekV3Router(nn.Module):
    """Pure-PyTorch reference matching HF DeepseekV3TopkRouter.forward exactly."""

    def __init__(self, num_experts, top_k, hidden_size, n_group, topk_group,
                 routed_scaling_factor, norm_topk_prob):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        experts_per_group = self.num_experts // self.n_group
        group_scores = (
            scores_for_choice.view(-1, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


def _mock_get_expert_model_parallel_size():
    return 1


def _mock_get_tensor_model_parallel_group():
    return None


# Patch NxD parallel state for CPU testing
_PARALLEL_PATCHES = [
    patch("neuronx_distributed.modules.moe.routing.get_expert_model_parallel_size",
           _mock_get_expert_model_parallel_size),
    patch("neuronx_distributed.modules.moe.routing.get_tensor_model_parallel_group",
           _mock_get_tensor_model_parallel_group),
]


def _create_neuron_router(num_experts, top_k, hidden_size, n_group, topk_group,
                           routed_scaling_factor, norm_topk_prob):
    """Create DeepseekV3Router with mocked parallel state."""
    from neuronx_distributed_inference.models.deepseek.modeling_deepseek import DeepseekV3Router

    for p in _PARALLEL_PATCHES:
        p.start()
    try:
        router = DeepseekV3Router(
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=torch.float32,
            act_fn="sigmoid",
        )
    finally:
        for p in _PARALLEL_PATCHES:
            p.stop()
    return router


@pytest.fixture
def router_config():
    return dict(
        num_experts=64,
        top_k=8,
        hidden_size=128,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
    )


@pytest.fixture
def router_and_ref(router_config):
    """Create a DeepseekV3Router and matching reference, with shared weights."""
    ref = ReferenceDeepseekV3Router(**router_config)
    neuron_router = _create_neuron_router(**router_config)

    with torch.no_grad():
        nn.init.normal_(ref.weight, std=0.01)
        nn.init.normal_(ref.e_score_correction_bias, std=0.1)
        neuron_router.linear_router.weight.copy_(ref.weight)
        neuron_router.e_score_correction_bias.copy_(ref.e_score_correction_bias)

    return neuron_router, ref


class TestDeepseekV3Router:

    def test_expert_selection_matches_reference(self, router_and_ref, router_config):
        """Expert indices from neuron router must match HF reference exactly."""
        neuron_router, ref = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(4, 16, router_config["hidden_size"]).view(-1, router_config["hidden_size"])

        ref_indices, ref_weights = ref(x)
        router_logits, expert_affinities, expert_index = neuron_router(x)

        ref_sorted, _ = ref_indices.sort(dim=-1)
        neuron_sorted, _ = expert_index.sort(dim=-1)
        assert torch.equal(ref_sorted, neuron_sorted), (
            f"Expert selection mismatch.\nRef: {ref_sorted[:3]}\nNeuron: {neuron_sorted[:3]}"
        )

    def test_expert_weights_match_reference(self, router_and_ref, router_config):
        """Expert weights (normalized + scaled) must match HF reference."""
        neuron_router, ref = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(4, 16, router_config["hidden_size"]).view(-1, router_config["hidden_size"])

        ref_indices, ref_weights = ref(x)
        _, expert_affinities, expert_index = neuron_router(x)

        neuron_weights = expert_affinities.gather(1, expert_index)

        # Sort both by index for alignment
        ref_sort_order = ref_indices.sort(dim=-1)[1]
        neuron_sort_order = expert_index.sort(dim=-1)[1]

        ref_weights_sorted = ref_weights.gather(1, ref_sort_order)
        neuron_weights_sorted = neuron_weights.gather(1, neuron_sort_order)

        torch.testing.assert_close(neuron_weights_sorted, ref_weights_sorted, atol=1e-5, rtol=1e-5)

    def test_affinities_sparse(self, router_and_ref, router_config):
        """Expert affinities tensor should be sparse: only top_k non-zero per token."""
        neuron_router, _ = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(8, router_config["hidden_size"])

        _, expert_affinities, _ = neuron_router(x)
        nonzero_per_token = (expert_affinities != 0).sum(dim=-1)
        assert (nonzero_per_token == router_config["top_k"]).all(), (
            f"Expected {router_config['top_k']} non-zero affinities per token, got {nonzero_per_token}"
        )

    def test_scaling_factor_applied(self, router_and_ref, router_config):
        """Weights should sum to approximately routed_scaling_factor per token."""
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

    def test_group_masking_restricts_experts(self, router_and_ref, router_config):
        """Selected experts should come from at most topk_group groups."""
        neuron_router, _ = router_and_ref
        torch.manual_seed(42)
        x = torch.randn(32, router_config["hidden_size"])

        _, _, expert_index = neuron_router(x)
        experts_per_group = router_config["num_experts"] // router_config["n_group"]

        for token_idx in range(expert_index.shape[0]):
            groups_used = set()
            for eidx in expert_index[token_idx]:
                groups_used.add(eidx.item() // experts_per_group)
            assert len(groups_used) <= router_config["topk_group"], (
                f"Token {token_idx} used {len(groups_used)} groups, max is {router_config['topk_group']}"
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

    @pytest.mark.parametrize("n_group,topk_group,num_experts", [
        (8, 4, 256),   # DeepSeek V3 full
        (4, 2, 64),    # Smaller config
        (1, 1, 32),    # Single group (degenerates to simple top-k)
    ])
    def test_various_group_configs(self, n_group, topk_group, num_experts):
        """Router works correctly with different group configurations."""
        hidden_size = 64
        top_k = 8

        router = _create_neuron_router(
            num_experts=num_experts, top_k=top_k, hidden_size=hidden_size,
            n_group=n_group, topk_group=topk_group,
            routed_scaling_factor=2.5, norm_topk_prob=True,
        )
        ref = ReferenceDeepseekV3Router(
            num_experts=num_experts, top_k=top_k, hidden_size=hidden_size,
            n_group=n_group, topk_group=topk_group,
            routed_scaling_factor=2.5, norm_topk_prob=True,
        )

        with torch.no_grad():
            nn.init.normal_(ref.weight, std=0.01)
            nn.init.normal_(ref.e_score_correction_bias, std=0.1)
            router.linear_router.weight.copy_(ref.weight)
            router.e_score_correction_bias.copy_(ref.e_score_correction_bias)

        torch.manual_seed(123)
        x = torch.randn(8, hidden_size)

        ref_indices, _ = ref(x)
        _, _, expert_index = router(x)

        ref_sorted, _ = ref_indices.sort(dim=-1)
        neuron_sorted, _ = expert_index.sort(dim=-1)
        assert torch.equal(ref_sorted, neuron_sorted)

    def test_no_normalization(self):
        """When norm_topk_prob=False, weights are not normalized."""
        router = _create_neuron_router(
            num_experts=16, top_k=4, hidden_size=32,
            n_group=2, topk_group=1,
            routed_scaling_factor=2.5, norm_topk_prob=False,
        )

        torch.manual_seed(99)
        x = torch.randn(4, 32)
        _, expert_affinities, expert_index = router(x)
        topk_weights = expert_affinities.gather(1, expert_index)

        # All weights should be positive (sigmoid * scaling_factor)
        assert (topk_weights > 0).all()
        # Without normalization, sum should NOT equal routed_scaling_factor
        # (it would only if all selected sigmoid outputs happened to sum to 1)
        weight_sums = topk_weights.sum(dim=-1)
        assert not torch.allclose(weight_sums, torch.tensor(2.5)), (
            "Weight sums equal scaling factor even without normalization — unexpected"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
