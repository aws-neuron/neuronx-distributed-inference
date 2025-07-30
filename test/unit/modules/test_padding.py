import pytest

import torch

from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            id=f"dtype_{str(dtype).split('.')[-1]}",
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
def test_padding(dtype):
    test_inputs = [
        [[1, 4, 256, 128], [1, 4, 256, 256], torch.finfo(dtype).min],  # pad dim 3
        [[1, 4, 128, 128], [1, 4, 256, 256], 0],  # pad dim 3, 2
        [[1, 1, 128, 128], [1, 4, 256, 256], 1],  # pad dim 3, 2, 1
        [[1, 1, 128, 128], [4, 4, 256, 256], None],  # pad dim 3, 2, 1, 0
        [[1, 1024], [4, 1024], torch.finfo(dtype).min], # pad dim 0
        [[1, 1024], [4, 2048], None], # pad dim 0, 1
        [[1, 1, 2], [4, 1, 2], 0], # pad dim 0
    ]

    for original_shape, target_shape, pad_value in test_inputs:
        original_unpadded_tensor = torch.randn(original_shape, dtype=dtype)

        padded_tensor, original_idx_slices = pad_tensor(original_unpadded_tensor, target_shape, pad_value)
        new_unpadded_tensor = unpad_tensor(padded_tensor, original_idx_slices)

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
    test_padding(dtype=torch.float16)

