import unittest

import torch
from neuronx_distributed_inference.modules.flashdecode.utils import mask_util


class TestFlashDecodingMasks(unittest.TestCase):

    def test_bsz_one(self):
        expected_active_mask = torch.tensor([[0]], dtype=torch.int32)
        expected_prior_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        cache_ids = torch.tensor([[6]], dtype=torch.int32)
        core_id = torch.tensor([19], dtype=torch.int32)
        active, prior = mask_util(pos_ids=cache_ids, rank_id=core_id,
                                  num_cores_per_group=4, cache_size=8)
        assert torch.equal(expected_active_mask, active)
        assert torch.equal(expected_prior_mask, prior)

    def test_bsz_two(self):
        # validate rank 0 (core_id 0)
        expected_active_mask = torch.tensor([[0], [1], [1]], dtype=torch.int32)
        expected_prior_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)
        cache_ids = torch.tensor([[5], [4], [10]], dtype=torch.int32)
        core_id = torch.tensor([0], dtype=torch.int32)
        active, prior = mask_util(pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, cache_size=8)
        assert torch.equal(expected_active_mask, active)
        assert torch.equal(expected_prior_mask, prior)

        # validate rank 1 (core_id 1)
        expected_active_mask = torch.tensor([[1], [0], [0]], dtype=torch.int32)
        expected_prior_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                            [1, 1, 0, 0, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)

        core_id = torch.tensor([1], dtype=torch.int32)
        active, prior = mask_util(pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, cache_size=8)
        assert torch.equal(expected_active_mask, active)
        assert torch.equal(expected_prior_mask, prior)


if __name__ == "__main__":
    unittest.main()
