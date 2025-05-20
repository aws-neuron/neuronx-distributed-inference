import unittest

import pytest
import torch
import torch_neuronx
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

from neuronx_distributed_inference.modules.flashdecode.utils import mask_util


def ckpt_loader():
    return {}


def get_traced_model(model_cls, num_cores_per_group, *args):
    builder = ModelBuilder(
        router=None,
        tp_degree=1,
        checkpoint_loader=ckpt_loader,
        num_cores_per_group=num_cores_per_group,
    )
    builder.add(
        key="main",
        model_instance=BaseModelInstance(model_cls, input_output_aliases={}),
        example_inputs=[(args)],
    )
    model = builder.trace(initialize_model_weights=True)
    return model


class MaskUtil(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cores_per_group = 2
        self.cache_size = 8

    def forward(self, pos_ids, rank_id):
        return mask_util(pos_ids, rank_id, self.num_cores_per_group, self.cache_size)


class TestFlashDecodingMasks(unittest.TestCase):
    def test_bsz_two(self):
        # validate rank 0 (core_id 0)
        expected_active_mask = torch.tensor([[0], [1], [1]], dtype=torch.int32)
        expected_prior_mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=torch.int32,
        )
        cache_ids = torch.tensor([[5], [4], [10]], dtype=torch.int32)
        core_id = torch.tensor([0], dtype=torch.int32)
        active, prior = mask_util(
            pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, cache_size=8
        )
        traced_model = get_traced_model(MaskUtil, 1, cache_ids, core_id)
        traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
        output = traced_model(cache_ids, core_id)

        torch_neuronx.testing.assert_close(expected_active_mask, output[0])
        torch_neuronx.testing.assert_close(expected_prior_mask, output[1])

        assert torch.equal(expected_active_mask, active)
        assert torch.equal(expected_prior_mask, prior)

        # validate rank 1 (core_id 1)
        expected_active_mask = torch.tensor([[1], [0], [0]], dtype=torch.int32)
        expected_prior_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=torch.int32,
        )

        core_id = torch.tensor([1], dtype=torch.int32)
        active, prior = mask_util(
            pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, cache_size=8
        )
        output = traced_model(cache_ids, core_id)
        torch_neuronx.testing.assert_close(expected_active_mask, output[0])
        torch_neuronx.testing.assert_close(expected_prior_mask, output[1])
        torch.testing.assert_close(expected_active_mask, active)
        torch.testing.assert_close(expected_prior_mask, prior)

    def test_multiple_rank(self):
        expected_prior_masks_0 = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        expected_active_masks_0 = torch.tensor([[0], [1], [1]], dtype=torch.int32)

        expected_prior_masks_1 = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        expected_active_masks_1 = torch.tensor([[1], [0], [0]], dtype=torch.int32)

        cache_ids = torch.tensor([[5], [4], [10]], dtype=torch.int32)
        rank_id = torch.tensor([0], dtype=torch.int32)
        active_core_0, prior_core_0 = mask_util(
            pos_ids=cache_ids, rank_id=rank_id, num_cores_per_group=2, cache_size=8
        )

        traced_model = get_traced_model(MaskUtil, 1, cache_ids, rank_id)
        traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
        output_0 = traced_model(cache_ids, rank_id)
        rank_id = torch.tensor([1], dtype=torch.int32)
        output_1 = traced_model(cache_ids, rank_id)

        active_core_1, prior_core_1 = mask_util(
            pos_ids=cache_ids, rank_id=rank_id, num_cores_per_group=2, cache_size=8
        )
        torch.testing.assert_close(expected_active_masks_0, active_core_0)
        torch.testing.assert_close(expected_prior_masks_0, prior_core_0)
        torch.testing.assert_close(expected_active_masks_1, active_core_1)
        torch.testing.assert_close(expected_prior_masks_1, prior_core_1)
        torch_neuronx.testing.assert_close(expected_active_masks_0, output_0[0])
        torch_neuronx.testing.assert_close(expected_prior_masks_0, output_0[1])
        torch_neuronx.testing.assert_close(expected_active_masks_1, output_1[0])
        torch_neuronx.testing.assert_close(expected_prior_masks_1, output_1[1])

    def test_speculation(self):
        cache_ids = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7], [10, 11, 12, 13]], dtype=torch.int32)
        k = 4
        expected_prior_masks_0 = torch.tensor(
            [
                [[1, 1, 1, 0, 0, 0, 0, 0]] * k,
                [[1, 1, 0, 0, 0, 0, 0, 0]] * k,
                [[1, 1, 1, 1, 1, 0, 0, 0]] * k,
            ],
            dtype=torch.int32,
        )
        expected_active_masks_0 = torch.tensor(
            [
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]],
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
            ],
            dtype=torch.int32,
        )

        expected_prior_masks_1 = torch.tensor(
            [
                [[1, 1, 0, 0, 0, 0, 0, 0]] * k,
                [[1, 1, 0, 0, 0, 0, 0, 0]] * k,
                [[1, 1, 1, 1, 1, 0, 0, 0]] * k,
            ],
            dtype=torch.int32,
        )
        expected_active_masks_1 = torch.tensor(
            [
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]],
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]],
            ],
            dtype=torch.int32,
        )

        rank_id = torch.tensor([0], dtype=torch.int32)
        active_core_0, prior_core_0 = mask_util(
            pos_ids=cache_ids, rank_id=rank_id, num_cores_per_group=2, cache_size=8
        )

        traced_model = get_traced_model(MaskUtil, 1, cache_ids, rank_id)
        traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
        output_0 = traced_model(cache_ids, rank_id)

        rank_id = torch.tensor([1], dtype=torch.int32)
        active_core_1, prior_core_1 = mask_util(
            pos_ids=cache_ids, rank_id=rank_id, num_cores_per_group=2, cache_size=8
        )
        output_1 = traced_model(cache_ids, rank_id)
        torch.testing.assert_close(expected_active_masks_0, active_core_0)
        torch.testing.assert_close(expected_prior_masks_0, prior_core_0)
        torch.testing.assert_close(expected_active_masks_1, active_core_1)
        torch.testing.assert_close(expected_prior_masks_1, prior_core_1)
        torch_neuronx.testing.assert_close(expected_active_masks_0, output_0[0])
        torch_neuronx.testing.assert_close(expected_prior_masks_0, output_0[1])
        torch_neuronx.testing.assert_close(expected_active_masks_1, output_1[0])
        torch_neuronx.testing.assert_close(expected_prior_masks_1, output_1[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
