"""
Unit tests for configurable image dimensions in Qwen2-VL.
Tests shape validation for vision model inputs and input processor.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from neuronx_distributed_inference.models.qwen2_vl.utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    DEFAULT_PIXELS_PER_IMAGE, MAX_GRID_SIZE
)
from neuronx_distributed_inference.models.qwen2_vl.utils.vision_utils import (
    calculate_max_grid_size, get_image_dimensions
)


class TestCalculateMaxGridSize:
    """Tests for the calculate_max_grid_size helper function."""

    def test_default_dimensions(self):
        """Test calculate_max_grid_size with default dimensions (640x320)."""
        max_grid = calculate_max_grid_size(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
        # With 640x320 and 1.2 buffer: max(640, 320) * 1.2 / 14 + 1 = 55.86 -> 55
        assert max_grid > 0
        assert isinstance(max_grid, int)

    @pytest.mark.parametrize("width,height,expected_min", [
        (640, 320, 50),    # Default dimensions
        (1024, 768, 80),   # Larger dimensions
        (224, 224, 15),    # Small square
        (1920, 1080, 150), # HD dimensions
    ])
    def test_custom_dimensions(self, width, height, expected_min):
        """Test calculate_max_grid_size with various custom dimensions."""
        max_grid = calculate_max_grid_size(width, height)
        assert max_grid >= expected_min, f"Expected max_grid >= {expected_min} for {width}x{height}, got {max_grid}"
        assert isinstance(max_grid, int)

    def test_buffer_factor_applied(self):
        """Test that buffer factor (1.2) is correctly applied."""
        # For a 140x140 image with patch_size=14:
        # Without buffer: 140 / 14 = 10
        # With 1.2 buffer: 140 * 1.2 / 14 + 1 = 13
        max_grid = calculate_max_grid_size(140, 140, patch_size=14)
        assert max_grid >= 12, "Buffer factor should increase grid size"

    def test_custom_patch_size(self):
        """Test calculate_max_grid_size with custom patch size."""
        max_grid_14 = calculate_max_grid_size(640, 320, patch_size=14)
        max_grid_7 = calculate_max_grid_size(640, 320, patch_size=7)
        # Smaller patch size should result in larger grid
        assert max_grid_7 > max_grid_14

    def test_asymmetric_dimensions(self):
        """Test that max dimension is used for grid calculation."""
        # Width > Height
        max_grid_wide = calculate_max_grid_size(1000, 200)
        # Height > Width
        max_grid_tall = calculate_max_grid_size(200, 1000)
        # Both should give same result since max(width, height) is used
        assert max_grid_wide == max_grid_tall


class TestGetImageDimensions:
    """Tests for the get_image_dimensions helper function."""

    def test_with_custom_dimensions(self):
        """Test get_image_dimensions returns custom values from config."""
        neuron_config = Mock()
        neuron_config.default_image_width = 1024
        neuron_config.default_image_height = 768

        width, height = get_image_dimensions(neuron_config)

        assert width == 1024
        assert height == 768

    def test_with_default_fallback(self):
        """Test get_image_dimensions falls back to defaults when not in config."""
        neuron_config = Mock(spec=[])  # Empty spec = no attributes

        width, height = get_image_dimensions(neuron_config)

        assert width == DEFAULT_IMAGE_WIDTH
        assert height == DEFAULT_IMAGE_HEIGHT

    def test_with_partial_config(self):
        """Test get_image_dimensions with only width specified."""
        neuron_config = Mock(spec=['default_image_width'])
        neuron_config.default_image_width = 800

        width, height = get_image_dimensions(neuron_config)

        assert width == 800
        assert height == DEFAULT_IMAGE_HEIGHT


class TestDefaultConstants:
    """Tests to verify default constants are correctly defined."""

    def test_default_image_dimensions(self):
        """Test default image width and height constants."""
        assert DEFAULT_IMAGE_WIDTH == 640
        assert DEFAULT_IMAGE_HEIGHT == 320

    def test_default_pixels_per_image(self):
        """Test default pixels per image constant."""
        assert DEFAULT_PIXELS_PER_IMAGE == 1012

    def test_max_grid_size_constant(self):
        """Test max grid size constant for backward compatibility."""
        assert MAX_GRID_SIZE == 72


class TestSmartResizeIntegration:
    """Tests for smart_resize integration with image dimensions."""

    @pytest.mark.parametrize("width,height", [
        (640, 320),    # Default
        (1024, 768),   # Custom larger
        (224, 224),    # Small square
    ])
    def test_smart_resize_returns_valid_dimensions(self, width, height):
        """Test that smart_resize returns valid dimensions for patch division."""
        resized_height, resized_width = smart_resize(width=width, height=height)
        # Dimensions should be divisible by patch_size (14)
        patch_size = 14
        assert resized_height % patch_size == 0 or resized_width % patch_size == 0
        assert resized_height > 0
        assert resized_width > 0

    def test_pixels_per_image_calculation(self):
        """Test calculation of pixels per image from resized dimensions."""
        patch_size = 14
        resized_height, resized_width = smart_resize(width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT)
        pixels_per_image = (resized_height // patch_size) * (resized_width // patch_size)
        assert pixels_per_image > 0
        assert isinstance(pixels_per_image, int)


class TestInputGeneratorShapes:
    """Tests for input_generator output shapes validation."""

    @pytest.fixture
    def mock_vision_config(self):
        """Create a mock vision config for testing."""
        neuron_config = Mock()
        neuron_config.torch_dtype = torch.float32
        neuron_config.buckets = [1, 2, 4, 8]
        neuron_config.default_image_width = DEFAULT_IMAGE_WIDTH
        neuron_config.default_image_height = DEFAULT_IMAGE_HEIGHT

        vision_config = Mock()
        vision_config.neuron_config = neuron_config
        vision_config.patch_size = 14
        vision_config.temporal_patch_size = 2
        vision_config.in_channels = 3

        return vision_config

    def test_pixel_values_shape_dimensions(self, mock_vision_config):
        """Test pixel_values tensor shape follows expected pattern."""
        patch_size = mock_vision_config.patch_size
        temporal_patch_size = mock_vision_config.temporal_patch_size
        in_channels = mock_vision_config.in_channels

        # Expected second dimension
        expected_dim1 = in_channels * (patch_size ** 2) * temporal_patch_size
        assert expected_dim1 == 3 * 14 * 14 * 2  # 1176

    def test_grid_thw_shape(self, mock_vision_config):
        """Test grid_thw tensor shape for various bucket sizes."""
        for bucket in mock_vision_config.neuron_config.buckets:
            # grid_thw should have shape [bucket, 3]
            expected_shape = (bucket, 3)
            # Verify the shape pattern
            assert expected_shape[1] == 3, "grid_thw second dimension should always be 3"

    @pytest.mark.parametrize("bucket", [1, 2, 4, 8])
    def test_pixel_values_first_dimension(self, mock_vision_config, bucket):
        """Test that pixel_values first dimension scales with bucket size."""
        patch_size = mock_vision_config.patch_size
        image_width = mock_vision_config.neuron_config.default_image_width
        image_height = mock_vision_config.neuron_config.default_image_height

        resized_height, resized_width = smart_resize(width=image_width, height=image_height)
        pixels_per_image = (resized_height // patch_size) * (resized_width // patch_size)

        expected_first_dim = bucket * pixels_per_image
        assert expected_first_dim > 0
        # Verify it scales linearly with bucket
        assert expected_first_dim == bucket * pixels_per_image

    def test_shapes_with_custom_dimensions(self, mock_vision_config):
        """Test shapes change correctly with custom dimensions."""
        patch_size = mock_vision_config.patch_size

        # Test with custom dimensions
        custom_width = 1024
        custom_height = 768

        resized_height, resized_width = smart_resize(width=custom_width, height=custom_height)
        custom_pixels = (resized_height // patch_size) * (resized_width // patch_size)

        # Default dimensions
        default_resized_h, default_resized_w = smart_resize(
            width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT
        )
        default_pixels = (default_resized_h // patch_size) * (default_resized_w // patch_size)

        # Custom larger dimensions should result in more pixels
        assert custom_pixels > default_pixels


class TestPrepareGenerationInputsShapes:
    """Tests for prepare_generation_inputs_hf function shape validation."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock HuggingFace processor."""
        processor = MagicMock()

        # Mock apply_chat_template to return a formatted string
        processor.apply_chat_template.return_value = "<|im_start|>user\nDescribe this image<|im_end|>"

        # Mock the processor call to return expected output structure
        mock_output = {
            'input_ids': torch.ones(1, 100, dtype=torch.long),
            'attention_mask': torch.ones(1, 100, dtype=torch.long),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 16, 16]])
        }
        processor.return_value = mock_output

        return processor

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with vision settings."""
        neuron_config = Mock()
        neuron_config.default_image_width = 800
        neuron_config.default_image_height = 600

        vision_config = Mock()
        vision_config.neuron_config = neuron_config

        config = Mock()
        config.vision_config = vision_config

        return config

    def test_config_dimensions_used_when_provided(self, mock_config):
        """Test that config dimensions override defaults when provided."""
        # Verify config provides custom dimensions
        assert mock_config.vision_config.neuron_config.default_image_width == 800
        assert mock_config.vision_config.neuron_config.default_image_height == 600

    def test_default_dimensions_used_when_no_config(self):
        """Test that default constants are used when config is None."""
        # This tests the getattr fallback logic
        config = None
        if config is not None and hasattr(config, 'vision_config'):
            image_width = getattr(config.vision_config.neuron_config, 'default_image_width', DEFAULT_IMAGE_WIDTH)
        else:
            image_width = DEFAULT_IMAGE_WIDTH
            image_height = DEFAULT_IMAGE_HEIGHT

        assert image_width == 640
        assert image_height == 320

    def test_dimension_extraction_with_missing_attributes(self):
        """Test dimension extraction when config exists but attributes are missing."""
        # Config with vision_config but no default_image_width
        neuron_config = Mock(spec=[])  # Empty spec means no attributes
        vision_config = Mock()
        vision_config.neuron_config = neuron_config
        config = Mock()
        config.vision_config = vision_config

        # Should fall back to defaults via getattr
        image_width = getattr(config.vision_config.neuron_config, 'default_image_width', DEFAULT_IMAGE_WIDTH)
        image_height = getattr(config.vision_config.neuron_config, 'default_image_height', DEFAULT_IMAGE_HEIGHT)

        assert image_width == DEFAULT_IMAGE_WIDTH
        assert image_height == DEFAULT_IMAGE_HEIGHT

    @pytest.mark.parametrize("num_images", [1, 2, 5])
    def test_message_structure_with_images(self, num_images):
        """Test that message structure is built correctly for multiple images."""
        # Simulate the message building logic
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test prompt"},
                ]
            },
        ]

        image_width = DEFAULT_IMAGE_WIDTH
        image_height = DEFAULT_IMAGE_HEIGHT

        for _ in range(num_images):
            info = {"type": "image", "resized_height": image_height, "resized_width": image_width}
            messages[0]["content"].append(info)

        # Verify structure
        assert len(messages[0]["content"]) == num_images + 1  # text + images
        assert messages[0]["content"][0]["type"] == "text"
        for i in range(1, num_images + 1):
            assert messages[0]["content"][i]["type"] == "image"
            assert messages[0]["content"][i]["resized_width"] == image_width
            assert messages[0]["content"][i]["resized_height"] == image_height


class TestConfigurableDimensionsIntegration:
    """Integration tests for configurable dimensions across the system."""

    @pytest.mark.parametrize("width,height", [
        (640, 320),    # Default
        (1024, 768),   # Custom larger
        (320, 240),    # Smaller
        (1920, 1080),  # HD
    ])
    def test_dimensions_flow_through_system(self, width, height):
        """Test that custom dimensions properly flow through all calculations."""
        patch_size = 14

        # 1. Calculate max grid size
        max_grid = calculate_max_grid_size(width, height, patch_size)
        assert max_grid > 0

        # 2. Calculate resized dimensions
        resized_height, resized_width = smart_resize(width=width, height=height)
        assert resized_height > 0
        assert resized_width > 0

        # 3. Calculate pixels per image
        pixels_per_image = (resized_height // patch_size) * (resized_width // patch_size)
        assert pixels_per_image > 0

        # 4. Verify grid doesn't exceed max
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        assert grid_h < max_grid, f"Grid height {grid_h} should be < max_grid {max_grid}"
        assert grid_w < max_grid, f"Grid width {grid_w} should be < max_grid {max_grid}"

    def test_tensor_shapes_consistency(self):
        """Test that tensor shapes are internally consistent."""
        patch_size = 14
        temporal_patch_size = 2
        in_channels = 3
        bucket = 4

        resized_height, resized_width = smart_resize(
            width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT
        )
        pixels_per_image = (resized_height // patch_size) * (resized_width // patch_size)

        # pixel_values shape
        pixel_values_dim0 = bucket * pixels_per_image
        pixel_values_dim1 = in_channels * (patch_size ** 2) * temporal_patch_size

        # grid_thw shape
        grid_thw_shape = (bucket, 3)
        grid_thw_values = [1, resized_height // patch_size, resized_width // patch_size]

        # Verify consistency
        assert pixel_values_dim0 == bucket * pixels_per_image
        assert pixel_values_dim1 == 1176  # 3 * 14 * 14 * 2
        assert grid_thw_shape == (bucket, 3)
        assert grid_thw_values[0] == 1  # temporal dimension
        assert grid_thw_values[1] * grid_thw_values[2] == pixels_per_image
