import os
import pytest
import torch

os.environ["BASE_COMPILE_WORK_DIR"] = "./compiler_workdir"

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLVisionModel,
    NeuronQwen3VLVisionModelWrapper,
    VISION_POSITION_ID_PAD_VALUE,
)
from neuronx_distributed_inference.utils.testing import build_function


# ============================================================================
# Tests for create_vision_position_ids (CPU side)
# ============================================================================


class TestCreateVisionPositionIds:
    """Unit tests for NeuronQwen3VLVisionModelWrapper.create_vision_position_ids."""

    # ==================== Shape Tests ====================

    def test_output_shape_single_image(self):
        image_grid_thw = torch.tensor([[2, 3, 4]])  # 24 tokens
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.shape == (24,)

    def test_output_shape_multiple_images(self):
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3]])  # 4 + 9 = 13
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.shape == (13,)

    def test_output_shape_many_images(self):
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [2, 2, 2], [1, 4, 4]])
        # 4 + 9 + 8 + 16 = 37
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.shape == (37,)

    # ==================== Dtype Tests ====================

    def test_output_dtype(self):
        image_grid_thw = torch.tensor([[1, 2, 2]])
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.dtype == torch.int32

    # ==================== Position ID Assignment Tests ====================

    def test_single_image_all_zeros(self):
        """Single image: all tokens should have position_id = 0."""
        image_grid_thw = torch.tensor([[1, 2, 3]])  # 6 tokens
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert torch.all(pos_ids == 0)

    def test_two_images_correct_ids(self):
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 1]])  # 4 + 3 = 7
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert torch.all(pos_ids[:4] == 0)
        assert torch.all(pos_ids[4:7] == 1)

    def test_three_images_correct_ids(self):
        image_grid_thw = torch.tensor([[1, 1, 2], [1, 2, 1], [1, 1, 3]])  # 2 + 2 + 3 = 7
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert torch.all(pos_ids[0:2] == 0)
        assert torch.all(pos_ids[2:4] == 1)
        assert torch.all(pos_ids[4:7] == 2)

    def test_token_count_per_image(self):
        """Count of each position ID matches tokens_per_image."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3]])
        # tokens: 4, 9, 6
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert (pos_ids == 0).sum().item() == 4
        assert (pos_ids == 1).sum().item() == 9
        assert (pos_ids == 2).sum().item() == 6

    # ==================== Contiguity Tests ====================

    def test_position_ids_are_contiguous(self):
        """All tokens of the same image appear consecutively."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3]])
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        diffs = pos_ids[1:] - pos_ids[:-1]
        assert torch.all((diffs == 0) | (diffs == 1))

    # ==================== Valid Range Tests ====================

    def test_no_negative_values(self):
        """Unpadded output should contain no negative values."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3]])
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert torch.all(pos_ids >= 0)

    def test_max_id_equals_num_images_minus_one(self):
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3]])
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.max().item() == 2  # 3 images, max id = 2

    # ==================== Edge Cases ====================

    def test_single_token_per_image(self):
        image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        expected = torch.tensor([0, 1, 2], dtype=torch.int32)
        assert torch.equal(pos_ids, expected)

    def test_asymmetric_image_dimensions(self):
        image_grid_thw = torch.tensor([[2, 3, 4], [1, 5, 2]])  # 24 + 10 = 34
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.shape == (34,)
        assert torch.all(pos_ids[:24] == 0)
        assert torch.all(pos_ids[24:34] == 1)

    def test_large_temporal_dimension(self):
        image_grid_thw = torch.tensor([[10, 2, 2]])  # 40 tokens
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        assert pos_ids.shape == (40,)
        assert torch.all(pos_ids == 0)

    # ==================== Known Output Tests ====================

    def test_known_output_simple(self):
        image_grid_thw = torch.tensor([[1, 2, 1], [1, 1, 2]])  # 2 + 2 = 4
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        expected = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        assert torch.equal(pos_ids, expected)

    def test_known_output_varying_sizes(self):
        image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 1]])  # 1 + 2 + 1 = 4
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        expected = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        assert torch.equal(pos_ids, expected)


# ============================================================================
# Tests for create_vision_attention_mask_from_pos_ids (Device side)
# ============================================================================


class TestCreateVisionAttentionMaskFromPosIds:
    """Unit tests for NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids."""

    # ==================== Shape Tests ====================

    def test_output_shape(self):
        pos_ids = torch.tensor([0, 0, 1, 1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.shape == (6, 6)

    def test_output_shape_no_padding(self):
        pos_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.shape == (5, 5)

    # ==================== Dtype Tests ====================

    def test_output_dtype_is_bool(self):
        pos_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.dtype == torch.bool

    # ==================== Block Diagonal Structure Tests ====================

    def test_single_image_full_attention(self):
        """Single image: all valid tokens attend to each other."""
        pos_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        expected = torch.ones(4, 4, dtype=torch.bool)
        assert torch.equal(mask, expected)

    def test_two_images_block_diagonal(self):
        """Two images produce a 2-block diagonal mask."""
        pos_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        expected = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ])
        assert torch.equal(mask, expected)

    def test_three_images_block_diagonal(self):
        """Three images produce a 3-block diagonal mask."""
        pos_ids = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        # Image 0: tokens 0-1, Image 1: tokens 2-4, Image 2: token 5
        expected = torch.zeros(6, 6, dtype=torch.bool)
        expected[0:2, 0:2] = True
        expected[2:5, 2:5] = True
        expected[5, 5] = True
        assert torch.equal(mask, expected)

    # ==================== Padding Exclusion Tests ====================

    def test_padding_rows_and_cols_all_false(self):
        """Padding tokens (-1) should have all-False rows and columns."""
        pos_ids = torch.tensor([0, 0, 1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        # Padding at indices 3, 4
        assert not torch.any(mask[3, :])
        assert not torch.any(mask[4, :])
        assert not torch.any(mask[:, 3])
        assert not torch.any(mask[:, 4])

    def test_all_padding_produces_zero_mask(self):
        """All-padding input produces an all-False mask."""
        pos_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert not torch.any(mask)

    def test_padding_does_not_match_padding(self):
        """Two padding tokens should NOT attend to each other (both are -1 but invalid)."""
        pos_ids = torch.tensor([0, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert not mask[1, 2].item()
        assert not mask[2, 1].item()

    # ==================== Symmetry Tests ====================

    def test_mask_is_symmetric(self):
        pos_ids = torch.tensor([0, 0, 1, 1, 2, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert torch.equal(mask, mask.T)

    # ==================== Diagonal Tests ====================

    def test_valid_tokens_self_attend(self):
        """Diagonal is True for all valid (non-padding) tokens."""
        pos_ids = torch.tensor([0, 0, 1, 1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        for i in range(4):  # valid tokens
            assert mask[i, i].item()
        for i in range(4, 6):  # padding tokens
            assert not mask[i, i].item()

    # ==================== True Count Tests ====================

    def test_true_count_equals_sum_of_squared_token_counts(self):
        """Number of True values = sum(n_i^2) for each image i."""
        # Image 0: 2 tokens, Image 1: 3 tokens -> 4 + 9 = 13
        pos_ids = torch.tensor([0, 0, 1, 1, 1, -1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.sum().item() == 2**2 + 3**2

    def test_true_count_no_padding(self):
        """Without padding, True count = sum(n_i^2)."""
        pos_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.sum().item() == 3**2 + 2**2

    # ==================== Binary Values Tests ====================

    def test_mask_contains_only_true_false(self):
        pos_ids = torch.tensor([0, 0, 1, 1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        assert mask.dtype == torch.bool

    # ==================== Known Output Tests ====================

    def test_known_output_with_padding(self):
        """Known input/output pair: [0, 0, 1, 1, -1, -1]."""
        pos_ids = torch.tensor([0, 0, 1, 1, -1, -1], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        expected = torch.tensor([
            [True,  True,  False, False, False, False],
            [True,  True,  False, False, False, False],
            [False, False, True,  True,  False, False],
            [False, False, True,  True,  False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ])
        assert torch.equal(mask, expected)

    def test_known_output_single_token_images(self):
        """Each image has exactly 1 token."""
        pos_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(pos_ids)
        expected = torch.eye(3, dtype=torch.bool)
        assert torch.equal(mask, expected)


# ============================================================================
# End-to-end: create_vision_position_ids -> pad -> create_vision_attention_mask_from_pos_ids
# ============================================================================


class TestEndToEndPositionIdsToMask:
    """Integration test: CPU-side position ID creation -> padding -> device-side mask computation."""

    def _pad_to_bucket(self, tensor, bucket_size, pad_value=VISION_POSITION_ID_PAD_VALUE):
        """Simple padding helper for tests."""
        if tensor.shape[0] >= bucket_size:
            return tensor[:bucket_size]
        padding = torch.full((bucket_size - tensor.shape[0],), pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding])

    def test_e2e_two_images_padded(self):
        """Two images padded to bucket_size=8 produce correct block-diagonal mask."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 1, 2]])  # 4 + 2 = 6 tokens
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        padded = self._pad_to_bucket(pos_ids, bucket_size=8)

        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(padded)

        assert mask.shape == (8, 8)
        assert mask.dtype == torch.bool
        # Image 0 block (4x4)
        assert torch.all(mask[0:4, 0:4])
        # Image 1 block (2x2)
        assert torch.all(mask[4:6, 4:6])
        # Cross-image attention is blocked
        assert not torch.any(mask[0:4, 4:6])
        assert not torch.any(mask[4:6, 0:4])
        # Padding rows/cols are all False
        assert not torch.any(mask[6:, :])
        assert not torch.any(mask[:, 6:])

    def test_e2e_three_images_padded(self):
        """Three images padded to bucket_size=16."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 1], [1, 1, 2]])  # 4 + 3 + 2 = 9
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        padded = self._pad_to_bucket(pos_ids, bucket_size=16)

        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(padded)

        assert mask.shape == (16, 16)
        # True count = 4^2 + 3^2 + 2^2 = 16 + 9 + 4 = 29
        assert mask.sum().item() == 29
        # Padding region
        assert not torch.any(mask[9:, :])
        assert not torch.any(mask[:, 9:])
        # Symmetry
        assert torch.equal(mask, mask.T)

    def test_e2e_single_image_fills_bucket(self):
        """Single image that exactly fills the bucket -- no padding."""
        image_grid_thw = torch.tensor([[1, 2, 4]])  # 8 tokens
        pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
        padded = self._pad_to_bucket(pos_ids, bucket_size=8)

        mask = NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids(padded)

        # Full attention -- single image, no padding
        assert torch.all(mask)


# ============================================================================
# Neuron Device Test: create_vision_attention_mask_from_pos_ids
# ============================================================================


def reference_create_vision_attention_mask(image_grid_thw, bucket_size):
    """
    CPU golden reference: build the block-diagonal attention mask directly
    from image_grid_thw (the original HF approach), padded to bucket_size.

    Returns:
        mask: Bool tensor of shape (bucket_size, bucket_size)
    """
    tokens_per_image = image_grid_thw.prod(dim=1).tolist()
    mask = torch.zeros(bucket_size, bucket_size, dtype=torch.bool)
    start = 0
    for n in tokens_per_image:
        end = start + n
        mask[start:end, start:end] = True
        start = end
    return mask


@pytest.mark.parametrize("image_grid_thw, bucket_size", [
    (torch.tensor([[1, 2, 2], [1, 1, 2]]), 8),          # 4+2=6, padded to 8
    (torch.tensor([[1, 2, 2], [1, 3, 1], [1, 1, 2]]), 16),  # 4+3+2=9, padded to 16
    (torch.tensor([[1, 4, 4]]), 16),                      # single image, exact fit
    (torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3], [1, 1, 1]]), 32),  # 4+9+6+1=20, padded to 32
])
def test_create_vision_attention_mask_on_device(image_grid_thw, bucket_size):
    """
    End-to-end test on Neuron device:
    1. create_vision_position_ids on CPU
    2. Pad to bucket_size
    3. Trace & run create_vision_attention_mask_from_pos_ids on device
    4. Compare with CPU golden reference built directly from image_grid_thw
    """
    # Step 1: CPU -- create position IDs and pad
    pos_ids = NeuronQwen3VLVisionModelWrapper.create_vision_position_ids(image_grid_thw)
    padded_pos_ids = torch.full((bucket_size,), VISION_POSITION_ID_PAD_VALUE, dtype=torch.int32)
    padded_pos_ids[:pos_ids.shape[0]] = pos_ids

    # Step 2: CPU golden reference -- build mask directly from image_grid_thw
    expected_mask = reference_create_vision_attention_mask(image_grid_thw, bucket_size)

    # Step 3: Trace and compile for Neuron device
    dummy_input = torch.zeros_like(padded_pos_ids)
    neuron_func = build_function(
        func=NeuronQwen3VLVisionModel.create_vision_attention_mask_from_pos_ids,
        example_inputs=[(dummy_input,)],
        tp_degree=2,
    )

    # Step 4: Run on device
    device_mask = neuron_func(padded_pos_ids)

    # Step 5: Compare
    torch.testing.assert_close(
        device_mask.to(torch.bool),
        expected_mask,
        rtol=0,
        atol=0,
    )
