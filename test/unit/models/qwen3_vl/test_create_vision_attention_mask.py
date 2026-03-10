import pytest
import torch

# Import the function from your module
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import NeuronQwen3VLVisionModelWrapper


class TestCreateVisionAttentionMask:
    """Unit tests for NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask function."""

    # ==================== Shape Tests ====================

    def test_output_shape_single_image(self):
        """Test output shape for a single image."""
        image_grid_thw = torch.tensor([[2, 3, 4]])  # 24 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        assert mask.shape == (24, 24)

    def test_output_shape_multiple_images(self):
        """Test output shape for multiple images."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3]])  # 4 + 9 = 13 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        assert mask.shape == (13, 13)

    def test_output_shape_many_images(self):
        """Test output shape for many images."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [2, 2, 2], [1, 4, 4]])
        # 4 + 9 + 8 + 16 = 37 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        assert mask.shape == (37, 37)

    # ==================== Dtype Tests ====================

    def test_output_dtype(self):
        """Test that output dtype is int32."""
        image_grid_thw = torch.tensor([[1, 2, 2]])
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        assert mask.dtype == torch.int32

    # ==================== Block Diagonal Structure Tests ====================

    def test_single_image_all_ones(self):
        """Test that single image produces all-ones mask."""
        image_grid_thw = torch.tensor([[1, 2, 3]])  # 6 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        expected = torch.ones(6, 6, dtype=torch.int32)
        assert torch.equal(mask, expected)

    def test_two_images_block_diagonal(self):
        """Test block diagonal structure with two images."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 1]])  # 4 + 3 = 7 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        # First block (0:4, 0:4) should be all ones
        assert torch.all(mask[0:4, 0:4] == 1)
        # Second block (4:7, 4:7) should be all ones
        assert torch.all(mask[4:7, 4:7] == 1)
        # Off-diagonal blocks should be zeros
        assert torch.all(mask[0:4, 4:7] == 0)
        assert torch.all(mask[4:7, 0:4] == 0)

    def test_three_images_block_diagonal(self):
        """Test block diagonal structure with three images."""
        image_grid_thw = torch.tensor([[1, 1, 2], [1, 2, 1], [1, 1, 3]])  # 2 + 2 + 3 = 7
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        # Diagonal blocks should be ones
        assert torch.all(mask[0:2, 0:2] == 1)
        assert torch.all(mask[2:4, 2:4] == 1)
        assert torch.all(mask[4:7, 4:7] == 1)

        # Off-diagonal blocks should be zeros
        assert torch.all(mask[0:2, 2:4] == 0)
        assert torch.all(mask[0:2, 4:7] == 0)
        assert torch.all(mask[2:4, 0:2] == 0)
        assert torch.all(mask[2:4, 4:7] == 0)
        assert torch.all(mask[4:7, 0:2] == 0)
        assert torch.all(mask[4:7, 2:4] == 0)

    def test_images_cannot_attend_to_each_other(self):
        """Test that tokens from different images cannot attend to each other."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])  # 4 + 4 = 8 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        # Cross-Image attention should be blocked
        for i in range(4):
            for j in range(4, 8):
                assert mask[i, j] == 0, f"Token {i} should not attend to token {j}"
                assert mask[j, i] == 0, f"Token {j} should not attend to token {i}"

    def test_self_attention_within_image(self):
        """Test that all tokens within same image can attend to each other."""
        image_grid_thw = torch.tensor([[1, 3, 3]])  # 9 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        for i in range(9):
            for j in range(9):
                assert mask[i, j] == 1, f"Token {i} should attend to token {j}"

    # ==================== Edge Cases ====================

    def test_single_token_per_image(self):
        """Test with images that have only 1 token each."""
        image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 3 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        expected = torch.eye(3, dtype=torch.int32)
        assert torch.equal(mask, expected)

    def test_asymmetric_image_dimensions(self):
        """Test with various asymmetric image dimensions."""
        image_grid_thw = torch.tensor([[2, 3, 4], [1, 5, 2]])  # 24 + 10 = 34 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        assert mask.shape == (34, 34)
        assert torch.all(mask[0:24, 0:24] == 1)
        assert torch.all(mask[24:34, 24:34] == 1)
        assert torch.all(mask[0:24, 24:34] == 0)
        assert torch.all(mask[24:34, 0:24] == 0)

    def test_large_temporal_dimension(self):
        """Test with large temporal dimension (video-like input)."""
        image_grid_thw = torch.tensor([[10, 2, 2]])  # 40 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        assert mask.shape == (40, 40)
        assert torch.all(mask == 1)

    # ==================== Symmetry Tests ====================

    def test_mask_is_symmetric(self):
        """Test that the attention mask is symmetric."""
        image_grid_thw = torch.tensor([[1, 2, 3], [2, 2, 2], [1, 4, 2]])
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        assert torch.equal(mask, mask.T)

    # ==================== Sum/Count Tests ====================

    def test_correct_number_of_ones(self):
        """Test that the total number of ones equals sum of squared token counts."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3]])
        # Tokens: 4, 9, 6
        # Expected ones: 4^2 + 9^2 + 6^2 = 16 + 81 + 36 = 133
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        assert mask.sum().item() == 16 + 81 + 36

    def test_diagonal_is_all_ones(self):
        """Test that the diagonal is always all ones (self-attention)."""
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 1], [2, 2, 2]])
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        diagonal = torch.diag(mask)
        assert torch.all(diagonal == 1)

    # ==================== Device Compatibility Tests ====================

    def test_input_on_cpu(self):
        """Test that function works with CPU tensor input."""
        image_grid_thw = torch.tensor([[1, 2, 2]], device='cpu')
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)
        assert mask.device.type == 'cpu'

    # ==================== Regression Tests ====================
    def test_known_output_2x2_images(self):
        """Test against known expected output for simple case."""
        image_grid_thw = torch.tensor([[1, 2, 1], [1, 1, 2]])  # 2 + 2 = 4 tokens
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        expected = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.int32)

        assert torch.equal(mask, expected)

    def test_known_output_varying_sizes(self):
        """Test against known expected output with varying image sizes."""
        image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 1]])  # 1 + 2 + 1 = 4
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        expected = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.int32)

        assert torch.equal(mask, expected)

    def test_realistic_batch_of_images(self):
        """Test with realistic batch of different sized images."""
        # Simulating a batch with:
        # - 224x224 image with patch size 14 -> 16x16 = 256 tokens (T=1)
        # - 336x336 image with patch size 14 -> 24x24 = 576 tokens (T=1)
        image_grid_thw = torch.tensor([[1, 16, 16], [1, 24, 24]])
        mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(image_grid_thw)

        total_tokens = 256 + 576
        assert mask.shape == (total_tokens, total_tokens)

        # Check block structure
        assert mask[0:256, 0:256].sum() == 256 * 256
        assert mask[256:, 256:].sum() == 576 * 576
        assert mask[0:256, 256:].sum() == 0
        assert mask[256:, 0:256].sum() == 0
