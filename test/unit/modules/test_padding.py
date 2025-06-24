import unittest

import torch

from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings


class TestPadding(unittest.TestCase):
    def test_padding(self):
        test_inputs = [
            [[1, 4, 256, 128], [1, 4, 256, 256]],  # pad index 3
            [[1, 4, 128, 128], [1, 4, 256, 256]],  # pad index 3, 2
            [[1, 1, 128, 128], [1, 4, 256, 256]],  # pad index 3, 2, 1
            [[1, 1, 128, 128], [4, 4, 256, 256]],  # pad index 3, 2, 1, 0
            [[1, 1024], [4, 1024]], # pad index 0
            [[1, 1024], [4, 2048]], # pad index 0, 1
            [[1, 1, 2], [4, 1, 2]], # pad index 0
        ]

        for original_shape, target_shape in test_inputs:
            original_unpadded_tensor = torch.randn(original_shape)

            padded_tensor, mask = pad_tensor(original_unpadded_tensor, target_shape)
            new_unpadded_tensor = unpad_tensor(padded_tensor, mask)

            # Compare output logits
            passed, max_err = check_accuracy_embeddings(
                new_unpadded_tensor,
                original_unpadded_tensor,
                plot_outputs=True,
                rtol=1.3e-6,
                atol=1e-5,
            )
            assert (
                passed
            ), f"Embeddings of original shape {original_shape}, target shape {target_shape} failed accuracy validation, max_err: {max_err}"


if __name__ == "__main__":
    unittest.main()
