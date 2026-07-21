import unittest

import torch

from neuronx_distributed_inference.models.diffusers.padder import (
    MaybePadder,
    pad,
    pad_interleaved,
    pad_sizes,
    round_up_to_divisor,
)


class TestPadder(unittest.TestCase):
    def test_round_up_to_divisor(self):
        self.assertEqual(round_up_to_divisor(5, 3), 6)
        self.assertEqual(round_up_to_divisor(10, 4), 12)
        self.assertEqual(round_up_to_divisor(8, 4), 8)

    def test_pad_sizes(self):
        # Test single dimension padding
        shape = (3, 4, 5)
        result = pad_sizes(shape, 0, 5)
        self.assertEqual(result, (0, 0, 0, 0, 0, 2))

        # Test multiple dimensions padding
        result = pad_sizes(shape, (0, 1), (5, 6))
        self.assertEqual(result, (0, 0, 0, 2, 0, 2))

        # Test left padding
        result = pad_sizes(shape, 0, 5, left=True)
        self.assertEqual(result, (0, 0, 0, 0, 2, 0))

        # Test no padding needed
        result = pad_sizes(shape, 0, 2)
        self.assertIsNone(result)

    def test_pad(self):
        # Test basic padding
        tensor = torch.ones(2, 3)
        padded = pad(tensor, 0, 4)
        self.assertEqual(padded.shape, (4, 3))
        self.assertTrue(torch.all(padded[2:] == 0))

        # Test left padding
        padded = pad(tensor, 0, 4, left=True)
        self.assertEqual(padded.shape, (4, 3))
        self.assertTrue(torch.all(padded[:2] == 0))

        # Test multiple dimensions
        tensor = torch.ones(2, 3)
        padded = pad(tensor, (0, 1), (4, 5))
        self.assertEqual(padded.shape, (4, 5))

        # Test with None input
        self.assertIsNone(pad(None, 0, 4))

    def test_pad_interleaved(self):
        # Test basic interleaved padding
        tensor = torch.tensor([1, 2, 3])
        result = pad_interleaved(tensor, dim=0, size=9, source_len_per_group=1, pad_len_per_group=2)
        expected = torch.tensor([1, 0, 0, 2, 0, 0, 3, 0, 0])
        self.assertTrue(torch.equal(result, expected))

        # Test 2D tensor with interleaved padding
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = pad_interleaved(tensor, dim=0, size=4, source_len_per_group=1, pad_len_per_group=1)
        expected = torch.tensor([[1, 2], [0, 0], [3, 4], [0, 0]])
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(torch.all(result[1] == 0))
        self.assertTrue(torch.all(result[3] == 0))
        self.assertTrue(torch.equal(result, expected))

    def test_maybe_padder(self):
        # Test end padding
        padder = MaybePadder(4096)
        tensor = torch.ones(96, 3072)
        result = padder(tensor, dim=1)
        self.assertEqual(result.shape, (96, 4096))
        self.assertTrue(torch.all(result[:, 3072:] == 0))

        # Test interleaved padding
        padder = MaybePadder(size=9, padding="interleaved", interleaved_factor=3)
        tensor = torch.tensor([1, 2, 3])
        result = padder(tensor, dim=0)
        expected = torch.tensor([1, 0, 0, 2, 0, 0, 3, 0, 0])
        self.assertTrue(torch.equal(result, expected))

        # Test with split_size
        padder = MaybePadder(size=8, padding="interleaved", split_size=2, interleaved_factor=2)
        tensor = torch.tensor([1, 2, 3, 4])
        result = padder(tensor, dim=0)
        expected = torch.tensor([1, 2, 0, 0, 3, 4, 0, 0])
        self.assertTrue(torch.equal(result, expected))

        # Test with None input
        self.assertIsNone(padder(None, dim=0))

        # Test invalid padding mode
        with self.assertRaises(AssertionError):
            padder = MaybePadder(size=5, padding="invalid")
            padder(tensor, dim=0)


if __name__ == "__main__":
    unittest.main()
