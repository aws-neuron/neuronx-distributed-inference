"""Unit tests for NeuronQwen3VLForImageEncoding and helper functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLForImageEncoding,
    _normalize_images,
    _build_content,
)


@pytest.fixture
def mock_processor():
    """Create mock processor."""
    processor = MagicMock()
    result = MagicMock()
    result.input_ids = MagicMock()
    result.attention_mask = MagicMock()
    result.pixel_values = MagicMock()
    result.image_grid_thw = MagicMock()
    processor.apply_chat_template.return_value = result
    return processor


@pytest.fixture
def mock_pil_image():
    """Create a mock PIL Image."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def temp_image_file(mock_pil_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        mock_pil_image.save(f, format="JPEG")
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestNormalizeImages:
    def test_none_input(self):
        result = _normalize_images(None, batch_size=3)
        assert result == [None, None, None]

    def test_empty_list(self):
        result = _normalize_images([], batch_size=2)
        assert result == [None, None]

    def test_single_image_batch_size_1(self, mock_pil_image):
        result = _normalize_images(mock_pil_image, batch_size=1)
        assert len(result) == 1
        assert result[0] == [mock_pil_image]

    def test_single_image_batch_size_greater_than_1(self, mock_pil_image):
        with pytest.raises(ValueError, match="Single image provided but batch_size=2"):
            _normalize_images(mock_pil_image, batch_size=2)

    def test_flat_list_batch_size_1(self, mock_pil_image):
        images = [mock_pil_image, mock_pil_image]
        result = _normalize_images(images, batch_size=1)
        assert len(result) == 1
        assert result[0] == images

    def test_flat_list_batch_size_greater_than_1(self, mock_pil_image):
        images = [mock_pil_image, mock_pil_image]
        with pytest.raises(ValueError, match="Flat list of images provided but batch_size=2"):
            _normalize_images(images, batch_size=2)

    def test_flat_list_strings_batch_size_1(self):
        images = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
        result = _normalize_images(images, batch_size=1)
        assert len(result) == 1
        assert result[0] == images

    def test_list_of_lists(self, mock_pil_image):
        images = [[mock_pil_image], [mock_pil_image, mock_pil_image]]
        result = _normalize_images(images, batch_size=2)
        assert result == images

    def test_list_of_lists_with_none(self, mock_pil_image):
        images = [[mock_pil_image], None, [mock_pil_image]]
        result = _normalize_images(images, batch_size=3)
        assert result == images

    def test_mismatched_lengths(self, mock_pil_image):
        images = [[mock_pil_image], [mock_pil_image]]
        with pytest.raises(ValueError, match="Number of image sets .* must match"):
            _normalize_images(images, batch_size=3)


class TestBuildContent:
    def test_no_images(self):
        content = _build_content("Hello", None)
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello"}

    def test_single_pil_image(self, mock_pil_image):
        content = _build_content("Describe this", [mock_pil_image])
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[0]["url"].startswith("data:image/jpeg;base64,")
        assert content[1] == {"type": "text", "text": "Describe this"}

    def test_single_image_not_list(self, mock_pil_image):
        content = _build_content("Describe", mock_pil_image)
        assert len(content) == 2
        assert content[0]["type"] == "image"

    def test_image_from_file_path(self, temp_image_file):
        content = _build_content("What's this?", [temp_image_file])
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[0]["url"].startswith("data:image/jpeg;base64,")

    def test_multiple_images(self, mock_pil_image):
        content = _build_content("Compare these", [mock_pil_image, mock_pil_image])
        assert len(content) == 3
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2] == {"type": "text", "text": "Compare these"}

    def test_invalid_image_type(self):
        with pytest.raises(TypeError, match="Invalid image_data type"):
            _build_content("Hello", [123])


class TestNeuronQwen3VLForImageEncoding:
    def test_prepare_input_args_single_prompt(self, mock_processor, mock_pil_image):
        input_ids, attention_mask, vision_inputs = NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Describe this image",
            images=[[mock_pil_image]],
            processor=mock_processor,
        )

        mock_processor.apply_chat_template.assert_called_once()
        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0][0]["role"] == "user"

    def test_prepare_input_args_batch_prompts(self, mock_processor, mock_pil_image):
        prompts = ["Describe image 1", "Describe image 2"]
        images = [[mock_pil_image], [mock_pil_image]]

        NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts=prompts,
            images=images,
            processor=mock_processor,
        )

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    def test_prepare_input_args_custom_role(self, mock_processor, mock_pil_image):
        NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Hello",
            images=[[mock_pil_image]],
            processor=mock_processor,
            role="assistant",
        )

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0][0]["role"] == "assistant"

    def test_prepare_input_args_no_images(self, mock_processor):
        NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts=["Text only prompt"],
            images=None,
            processor=mock_processor,
        )

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        content = messages[0][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_prepare_input_args_returns_vision_inputs(self, mock_processor, mock_pil_image):
        input_ids, attention_mask, vision_inputs = NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Test",
            images=[[mock_pil_image]],
            processor=mock_processor,
        )

        assert "pixel_values" in vision_inputs
        assert "image_grid_thw" in vision_inputs

    def test_prepare_input_args_no_vision_inputs(self, mock_processor):
        # Mock processor without vision outputs
        result = MagicMock()
        result.input_ids = MagicMock()
        result.attention_mask = MagicMock()
        result.pixel_values = None
        result.image_grid_thw = None
        mock_processor.apply_chat_template.return_value = result

        input_ids, attention_mask, vision_inputs = NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Text only",
            images=None,
            processor=mock_processor,
        )

        assert vision_inputs == {}

    def test_prepare_input_args_missing_vision_attrs(self, mock_processor):
        # Mock processor without vision attributes at all
        result = MagicMock(spec=["input_ids", "attention_mask"])
        result.input_ids = MagicMock()
        result.attention_mask = MagicMock()
        mock_processor.apply_chat_template.return_value = result

        input_ids, attention_mask, vision_inputs = NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Text only",
            images=None,
            processor=mock_processor,
        )

        assert vision_inputs == {}

    def test_prepare_input_args_processor_kwargs(self, mock_processor, mock_pil_image):
        NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts="Test",
            images=[[mock_pil_image]],
            processor=mock_processor,
        )

        call_kwargs = mock_processor.apply_chat_template.call_args[1]
        assert call_kwargs["tokenize"] is True
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["return_dict"] is True
        assert call_kwargs["return_tensors"] == "pt"
        assert call_kwargs["padding"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])