# coding=utf-8
"""Unit tests for NeuronGlm4MoeRouter.

These tests run on CPU (no Neuron hardware required) and verify the
routing logic independently of the full model stack.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 64,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """Instantiate NeuronGlm4MoeRouter for CPU tests.

    Bypasses distributed environment by not calling super().__init__() in the
    normal path — instead we construct a minimal stub that exercises only the
    routing math we own.
    """
    from glm4_moe.modeling_glm4_moe import NeuronGlm4MoeRouter

    # We cannot call the full constructor because GroupLimitedRouter requires a
    # distributed process group.  Instead we build the key attributes directly.
    router = object.__new__(NeuronGlm4MoeRouter)
    # Patch required attributes from GroupLimitedRouter / NeuronGlm4MoeRouter
    router.num_experts = num_experts
    router.top_k = top_k
    router.hidden_size = hidden_size
    router.n_group = n_group
    router.topk_group = topk_group
    router.norm_topk_prob = norm_topk_prob
    router.routed_scaling_factor = routed_scaling_factor
    router.register_buffer = lambda name, tensor: setattr(router, name, tensor)
    router.register_buffer(
        "e_score_correction_bias", torch.zeros(num_experts, dtype=torch.float32)
    )
    return router


def _group_scores(scores, n_group, num_experts):
    """Compute per-group max-score without routing classes."""
    group_size = num_experts // n_group
    return scores.view(scores.shape[0], n_group, group_size).max(dim=-1).values


# ---------------------------------------------------------------------------
# noaux_tc_top_k unit tests
# ---------------------------------------------------------------------------


class TestRouterTopK:
    """Unit tests for NeuronGlm4MoeRouter.noaux_tc_top_k."""

    def test_topk_idx_shape(self):
        """topk_idx must have shape [batch, top_k]."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        topk_idx, _ = router.noaux_tc_top_k(scores)
        assert topk_idx.shape == (4, 2), f"Expected (4, 2), got {topk_idx.shape}"

    def test_full_affinities_shape(self):
        """full_affinities must have same shape as scores."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        assert full_affinities.shape == scores.shape

    def test_full_affinities_sparsity(self):
        """Only top_k entries per row should be non-zero."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        nonzero_per_row = (full_affinities != 0).sum(dim=-1)
        assert (nonzero_per_row == 2).all(), (
            f"Expected 2 non-zeros per row, got {nonzero_per_row}"
        )

    def test_normalized_weights_sum_to_one(self):
        """With norm_topk_prob=True, selected weights should sum to routed_scaling_factor."""
        factor = 2.0
        router = _make_router(
            num_experts=8, top_k=2, norm_topk_prob=True, routed_scaling_factor=factor
        )
        scores = torch.rand(4, 8).abs() + 0.1  # ensure positive
        _, full_affinities = router.noaux_tc_top_k(scores)
        row_sums = full_affinities.sum(dim=-1)
        # Each row should sum to ~routed_scaling_factor (within floating point tolerance)
        torch.testing.assert_close(
            row_sums,
            torch.full_like(row_sums, factor),
            atol=1e-5,
            rtol=1e-5,
            msg="Normalized + scaled weights should sum to routed_scaling_factor per row",
        )

    def test_no_normalization_scaling_only(self):
        """With norm_topk_prob=False, weights should be raw sigmoid scores * routed_scaling_factor."""
        factor = 0.5
        router = _make_router(
            num_experts=4, top_k=1, norm_topk_prob=False, routed_scaling_factor=factor
        )
        # Force known scores
        scores = torch.tensor([[0.2, 0.8, 0.1, 0.4]])
        topk_idx, full_affinities = router.noaux_tc_top_k(scores)
        selected_weight = full_affinities[0, topk_idx[0, 0]]
        # Raw top-1 score is 0.8; after scaling: 0.8 * 0.5 = 0.4
        expected = torch.tensor(0.8 * factor)
        torch.testing.assert_close(selected_weight, expected, atol=1e-5, rtol=1e-5)

    def test_e_score_correction_bias_shifts_routing_decision(self):
        """e_score_correction_bias should shift which expert gets selected."""
        router = _make_router(
            num_experts=4, top_k=1, n_group=1, topk_group=1, norm_topk_prob=False
        )
        # Scores: expert 0 wins normally
        scores = torch.tensor([[0.9, 0.5, 0.3, 0.1]])
        topk_before, _ = router.noaux_tc_top_k(scores)
        assert topk_before[0, 0].item() == 0, "Expert 0 should win without bias"

        # Add large bias to expert 1 → expert 1 should win now
        router.e_score_correction_bias = torch.tensor([0.0, 5.0, 0.0, 0.0])
        topk_after, _ = router.noaux_tc_top_k(scores)
        assert topk_after[0, 0].item() == 1, "Expert 1 should win with strong bias"

    def test_correction_bias_not_used_for_final_weights(self):
        """e_score_correction_bias must NOT pollute the selected weights."""
        router = _make_router(
            num_experts=4,
            top_k=1,
            n_group=1,
            topk_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        scores = torch.tensor([[0.9, 0.5, 0.3, 0.1]])

        # Add correction bias that changes routing decision
        router.e_score_correction_bias = torch.tensor([0.0, 5.0, 0.0, 0.0])
        topk_idx, full_affinities = router.noaux_tc_top_k(scores)

        selected_expert = topk_idx[0, 0].item()  # should be 1
        selected_weight = full_affinities[0, selected_expert]

        # Weight must come from original scores (0.5), NOT bias-corrected scores (5.5)
        torch.testing.assert_close(
            selected_weight,
            torch.tensor(0.5),
            atol=1e-5,
            rtol=1e-5,
            msg="Selected weight must be from original sigmoid scores, not bias-corrected",
        )

    def test_topk_idx_dtype(self):
        """topk_idx must be int64 (long) for MoE dispatch compatibility."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(2, 8)
        topk_idx, _ = router.noaux_tc_top_k(scores)
        # noaux_tc_top_k itself returns the idx from torch.topk (int64)
        assert topk_idx.dtype == torch.int64, f"Expected int64, got {topk_idx.dtype}"

    def test_full_affinities_non_negative(self):
        """Expert weights must be non-negative (softmax-like normalization)."""
        router = _make_router(num_experts=8, top_k=3)
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        assert (full_affinities >= 0).all(), "All expert weights must be >= 0"

    def test_batch_independence(self):
        """Each batch element is routed independently."""
        router = _make_router(
            num_experts=4, top_k=1, n_group=1, topk_group=1, norm_topk_prob=False
        )
        # Different scores per batch item
        scores = torch.tensor([[0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9]])
        topk_idx, _ = router.noaux_tc_top_k(scores)
        assert topk_idx[0, 0].item() == 0, "First item should select expert 0"
        assert topk_idx[1, 0].item() == 3, "Second item should select expert 3"
