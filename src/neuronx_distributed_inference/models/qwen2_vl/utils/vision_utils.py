"""
Utility functions for Qwen2-VL vision model.
"""

from neuronx_distributed_inference.models.qwen2_vl.utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
)


def get_image_dimensions(neuron_config):
    """
    Get image dimensions from neuron config with fallback to defaults.

    Args:
        neuron_config: The neuron config object that may contain
            default_image_width and default_image_height attributes

    Returns:
        Tuple of (image_width, image_height)
    """
    image_width = getattr(neuron_config, 'default_image_width', DEFAULT_IMAGE_WIDTH)
    image_height = getattr(neuron_config, 'default_image_height', DEFAULT_IMAGE_HEIGHT)
    return image_width, image_height


def calculate_max_grid_size(image_width, image_height, patch_size=14):
    """
    Calculate the maximum grid size based on image dimensions.

    Args:
        image_width: Width of the input image
        image_height: Height of the input image
        patch_size: Size of each patch (default: 14)

    Returns:
        Maximum grid size (max of height and width grids)
    """
    # Need to account for smart_resize which may increase dimensions
    # Add 20% buffer to handle smart_resize adjustments
    buffer_factor = 1.2
    max_dimension = max(image_width, image_height)
    max_grid = int((max_dimension * buffer_factor) / patch_size) + 1
    return max_grid
