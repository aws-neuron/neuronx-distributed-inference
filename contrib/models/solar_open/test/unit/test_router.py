"""Unit tests for NeuronSolarOpenRouter.

These tests run on CPU (no Neuron hardware required) and verify the
routing logic independently of the full model. The router is instantiated
by bypassing the distributed __init__ via object.__new__() + manual attribute
setup — identical to the GLM-4.5 MoE test pattern.

NeuronSolarOpenRouter is functionally identical to NeuronGlm4MoeRouter (same
sigmoid + group routing + e_score_correction_bias). Only the class name differs.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _import_router():
    """Import NeuronSolarOpenRouter, skip if Neuron SDK is missing."""
    try:
        from solar_open.modeling_solar_open import NeuronSolarOpenRouter

        return NeuronSolarOpenRouter
    except ImportError as e:
        pytest.skip(f"solar_open package not importable (Neuron SDK missing): {e}")


def _make_router(
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 16,
    n_group: int = 2,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """Instantiate NeuronSolarOpenRouter for CPU tests.

    Bypasses the distributed environment by using object.__new__() instead of
    the normal __init__ path — avoids parallel_state.get_expert_model_parallel_group()
    assertions. All attributes needed by noaux_tc_top_k are set manually.
    """
    NeuronSolarOpenRouter = _import_router()

    router = object.__new__(NeuronSolarOpenRouter)
    # Override register_buffer to use simple setattr (no nn.Module._buffers)
    router.register_buffer = lambda name, tensor: setattr(router, name, tensor)
    router.register_buffer(
        "e_score_correction_bias", torch.zeros(num_experts, dtype=torch.float32)
    )

    # GroupLimitedRouter attributes (needed by _calculate_group_scores etc.)
    router.n_group = n_group
    router.topk_group = topk_group
    router.top_k = top_k
    router.num_experts = num_experts

    # NeuronSolarOpenRouter-specific attributes
    router.norm_topk_prob = norm_topk_prob
    router.routed_scaling_factor = routed_scaling_factor

    return router


def _group_scores(scores: torch.Tensor, n_group: int) -> torch.Tensor:
    """Reference per-group max score (mirrors GroupLimitedRouter logic)."""
    T, E = scores.shape
    group_size = E // n_group
    return scores.view(T, n_group, group_size).max(dim=-1).values  # [T, n_group]


# ---------------------------------------------------------------------------
# TestRouterTopK
# ---------------------------------------------------------------------------


class TestRouterTopK:
    """Unit tests for NeuronSolarOpenRouter.noaux_tc_top_k."""

    def test_topk_idx_shape(self):
        """topk_idx must have shape [batch, top_k]."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        topk_idx, _ = router.noaux_tc_top_k(scores)
        assert topk_idx.shape == (4, 2), f"Expected (4, 2), got {topk_idx.shape}"

    def test_full_affinities_shape(self):
        """full_affinities must have same shape as input scores."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        assert full_affinities.shape == scores.shape, (
            f"full_affinities shape {full_affinities.shape} != scores shape {scores.shape}"
        )

    def test_full_affinities_sparsity(self):
        """Only top_k entries per row should be non-zero."""
        router = _make_router(num_experts=8, top_k=2)
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        nonzero_per_row = (full_affinities > 0).sum(dim=-1)
        assert nonzero_per_row.all(), (
            f"Expected 2 non-zeros per row, got {nonzero_per_row.tolist()}"
        )

    def test_normalized_weights_sum_to_routed_scaling_factor(self):
        """With norm_topk_prob=True, selected weights should sum to routed_scaling_factor."""
        factor = 1.5
        router = _make_router(norm_topk_prob=True, routed_scaling_factor=factor)
        scores = torch.rand(4, 8).abs()
        _, full_affinities = router.noaux_tc_top_k(scores)
        row_sums = full_affinities.sum(dim=-1)  # [4]
        torch.testing.assert_close(
            row_sums, torch.full_like(row_sums, factor), atol=1e-5, rtol=1e-5
        )

    def test_no_normalization_scaling_only(self):
        """With norm_topk_prob=False, weights are raw sigmoid * routed_scaling_factor."""
        factor = 2.0
        router = _make_router(
            num_experts=4,
            top_k=1,
            n_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=factor,
        )
        # Deterministic scores: expert 0 wins by a large margin
        scores = torch.tensor([[0.9, 0.1, 0.2, 0.3]])
        topk_idx, full_affinities = router.noaux_tc_top_k(scores)
        selected_weight = full_affinities[0, topk_idx[0, 0].item()]
        expected = torch.tensor(scores[0, 0] * factor)
        torch.testing.assert_close(selected_weight, expected, atol=1e-5, rtol=1e-5)

    def test_e_score_correction_bias_shifts_routing_decision(self):
        """e_score_correction_bias should shift which expert gets selected."""
        router = _make_router(num_experts=4, top_k=1, n_group=1)
        # Expert 0 has highest score
        scores = torch.tensor([[0.9, 0.5, 0.3, 0.1]])
        topk_before, _ = router.noaux_tc_top_k(scores)
        assert topk_before[0, 0].item() == 0, "Expert 0 should win without bias"

        # Apply large bias to expert 1 — it should now win
        router.e_score_correction_bias[1] = 5.0
        topk_after, _ = router.noaux_tc_top_k(scores)
        assert topk_after[0, 0].item() == 1, "Expert 1 should win with strong bias"

    def test_correction_bias_not_used_for_final_weights(self):
        """e_score_correction_bias must NOT pollute the selected expert weights."""
        router = _make_router(
            num_experts=4,
            top_k=1,
            n_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        scores = torch.tensor([[0.5, 0.9, 0.3, 0.1]])
        # Bias expert 0 so it wins routing decision, but weight must come from orig score
        router.e_score_correction_bias[0] = 10.0
        topk_idx, full_affinities = router.noaux_tc_top_k(scores)
        selected_expert = topk_idx[0, 0].item()
        selected_weight = full_affinities[0, selected_expert]
        # Weight must equal the original sigmoid score, not bias-corrected value
        expected = scores[0, selected_expert]
        torch.testing.assert_close(selected_weight, expected, atol=1e-5, rtol=1e-5)

    def test_topk_idx_dtype(self):
        """topk_idx must be int64 (torch.long) for MoE dispatch compatibility."""
        router = _make_router()
        scores = torch.rand(4, 8)
        topk_idx, _ = router.noaux_tc_top_k(scores)
        assert topk_idx.dtype == torch.int64, f"Expected int64, got {topk_idx.dtype}"

    def test_full_affinities_non_negative(self):
        """Expert weights must be non-negative (sigmoid scores are [0, 1])."""
        router = _make_router()
        scores = torch.rand(4, 8)
        _, full_affinities = router.noaux_tc_top_k(scores)
        assert (full_affinities >= 0).all(), "All expert weights must be >= 0"

    def test_batch_independence(self):
        """Each batch element must be routed independently."""
        router = _make_router(
            num_experts=4,
            top_k=1,
            n_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        # Item 0: expert 0 wins; item 1: expert 3 wins
        scores = torch.tensor(
            [
                [0.9, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.9],
            ]
        )
        topk_idx, _ = router.noaux_tc_top_k(scores)
        assert topk_idx[0, 0].item() == 0, "First item should select expert 0"
        assert topk_idx[1, 0].item() == 3, "Second item should select expert 3"
