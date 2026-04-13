import pytest
import torch
from unittest.mock import MagicMock
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
    NeuronQwen3VLForCausalLM,
)


# Use well-known token IDs from Qwen3 VL config
VISION_START_TOKEN_ID = 151652
IMAGE_TOKEN_ID = 151655
PAD_TOKEN_ID = 0


def _make_model_stub():
    """Create a minimal stub of NeuronQwen3VLForCausalLM with just the config attrs needed."""
    stub = object.__new__(NeuronQwen3VLForCausalLM)
    stub.config = MagicMock()
    stub.config.image_token_id = IMAGE_TOKEN_ID
    stub.config.vision_start_token_id = VISION_START_TOKEN_ID
    return stub


class TestCountImagesPerBatchLine:
    """Tests for _count_images_per_batch_line."""

    def test_single_batch_one_image(self):
        """BS=1, 1 image → [1]."""
        model = _make_model_stub()
        # ... text ... <vision_start> <image_token> ... text ...
        input_ids = torch.tensor([[10, 20, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30, 40]])
        result = model._count_images_per_batch_line(input_ids, attention_mask=None)
        assert result == [1]

    def test_single_batch_two_images(self):
        """BS=1, 2 images → [2]."""
        model = _make_model_stub()
        input_ids = torch.tensor([[
            10, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 20,
            VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30,
        ]])
        result = model._count_images_per_batch_line(input_ids, attention_mask=None)
        assert result == [2]

    def test_two_batch_different_images(self):
        """BS=2, batch_line_0 has 2 images, batch_line_1 has 1 image → [2, 1]."""
        model = _make_model_stub()
        # Pad to same seq len
        input_ids = torch.tensor([
            [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30, PAD_TOKEN_ID],
            [10, 20, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30, PAD_TOKEN_ID],
        ])
        result = model._count_images_per_batch_line(input_ids, attention_mask=None)
        assert result == [2, 1]

    def test_no_images(self):
        """BS=1, text only → [0]."""
        model = _make_model_stub()
        input_ids = torch.tensor([[10, 20, 30, 40]])
        result = model._count_images_per_batch_line(input_ids, attention_mask=None)
        assert result == [0]

    def test_with_attention_mask(self):
        """Attention mask filters out padding tokens that might contain stale vision tokens."""
        model = _make_model_stub()
        # The padded region has a stale vision_start+image_token pair
        input_ids = torch.tensor([[
            VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30,
            VISION_START_TOKEN_ID, IMAGE_TOKEN_ID,  # in padding region
        ]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = model._count_images_per_batch_line(input_ids, attention_mask)
        assert result == [1], "Should only count images in non-padded region"

    def test_mixed_batch_with_text_only_line(self):
        """BS=3: batch_line_0 has 1 image, batch_line_1 text-only, batch_line_2 has 2 images."""
        model = _make_model_stub()
        input_ids = torch.tensor([
            [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 30, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
            [10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
            [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, 50, PAD_TOKEN_ID, PAD_TOKEN_ID],
        ])
        result = model._count_images_per_batch_line(input_ids, attention_mask=None)
        assert result == [1, 0, 2]


class TestSplitVisionInputsByBatchLine:
    """Tests for _split_vision_inputs_by_batch_line."""

    def test_two_batch_lines_different_images(self):
        """BS=2: batch_line_0 has 2 images, batch_line_1 has 1 image.
        
        image_grid_thw = [[1,2,3], [1,4,2], [1,3,3]]
        patches per image: 6, 8, 9 → total 23
        batch_line_0 gets images 0,1 (14 patches), batch_line_1 gets image 2 (9 patches)
        """
        model = _make_model_stub()
        
        image_grid_thw = torch.tensor([[1, 2, 3], [1, 4, 2], [1, 3, 3]])
        pixel_values = torch.randn(23, 128)  # 6+8+9=23 patches

        result = model._split_vision_inputs_by_batch_line(
            pixel_values, image_grid_thw, images_per_batch_line=[2, 1]
        )

        assert len(result) == 2

        # Batch line 0: 2 images, 14 patches
        pv_0, grid_0 = result[0]
        assert grid_0.shape == (2, 3)
        assert pv_0.shape == (14, 128)
        assert torch.equal(grid_0, image_grid_thw[:2])
        assert torch.equal(pv_0, pixel_values[:14])

        # Batch line 1: 1 image, 9 patches
        pv_1, grid_1 = result[1]
        assert grid_1.shape == (1, 3)
        assert pv_1.shape == (9, 128)
        assert torch.equal(grid_1, image_grid_thw[2:])
        assert torch.equal(pv_1, pixel_values[14:])

    def test_text_only_batch_line(self):
        """Batch line with 0 images returns (None, None)."""
        model = _make_model_stub()
        
        image_grid_thw = torch.tensor([[1, 2, 2]])
        pixel_values = torch.randn(4, 64)

        result = model._split_vision_inputs_by_batch_line(
            pixel_values, image_grid_thw, images_per_batch_line=[1, 0]
        )

        assert len(result) == 2
        assert result[0][0] is not None
        assert result[0][0].shape == (4, 64)
        assert result[1] == (None, None)

    def test_single_image_per_batch_line(self):
        """BS=2, 1 image each."""
        model = _make_model_stub()
        
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3]])
        pixel_values = torch.randn(13, 64)  # 4+9=13

        result = model._split_vision_inputs_by_batch_line(
            pixel_values, image_grid_thw, images_per_batch_line=[1, 1]
        )

        assert len(result) == 2
        assert result[0][0].shape == (4, 64)
        assert result[0][1].shape == (1, 3)
        assert result[1][0].shape == (9, 64)
        assert result[1][1].shape == (1, 3)

    def test_all_text_only(self):
        """All batch lines are text-only."""
        model = _make_model_stub()
        
        # These won't be used since all counts are 0, but need valid tensors
        image_grid_thw = torch.zeros(0, 3, dtype=torch.int32)
        pixel_values = torch.zeros(0, 64)

        result = model._split_vision_inputs_by_batch_line(
            pixel_values, image_grid_thw, images_per_batch_line=[0, 0]
        )

        assert len(result) == 2
        assert result[0] == (None, None)
        assert result[1] == (None, None)


class TestConcatCausalLmOutputs:
    """Tests for concat_causal_lm_outputs."""

    def test_concat_logits(self):
        """Logits from multiple outputs are concatenated along dim 0."""
        out1 = CausalLMOutputWithPast(logits=torch.tensor([[1.0, 2.0]]))
        out2 = CausalLMOutputWithPast(logits=torch.tensor([[3.0, 4.0]]))

        result = NeuronQwen3VLForCausalLM.concat_causal_lm_outputs([out1, out2])

        assert result.logits.shape == (2, 2)
        assert torch.equal(result.logits, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_concat_tokens(self):
        """If outputs have .tokens attribute, they are concatenated."""
        out1 = CausalLMOutputWithPast(logits=torch.tensor([[1.0]]))
        out1.tokens = torch.tensor([42])
        out2 = CausalLMOutputWithPast(logits=torch.tensor([[2.0]]))
        out2.tokens = torch.tensor([99])

        result = NeuronQwen3VLForCausalLM.concat_causal_lm_outputs([out1, out2])

        assert result.tokens is not None
        assert torch.equal(result.tokens, torch.tensor([42, 99]))

    def test_concat_single_output(self):
        """Single output is returned as-is (wrapped in CausalLMOutputWithPast)."""
        out1 = CausalLMOutputWithPast(logits=torch.tensor([[1.0, 2.0, 3.0]]))

        result = NeuronQwen3VLForCausalLM.concat_causal_lm_outputs([out1])

        assert torch.equal(result.logits, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_no_tokens_attribute(self):
        """If no output has .tokens, result.tokens should be None."""
        out1 = CausalLMOutputWithPast(logits=torch.tensor([[1.0]]))
        out2 = CausalLMOutputWithPast(logits=torch.tensor([[2.0]]))

        result = NeuronQwen3VLForCausalLM.concat_causal_lm_outputs([out1, out2])

        assert not hasattr(result, 'tokens') or result.tokens is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])