# Default image dimensions for Qwen2-VL vision model
# These can be overridden via additional-config.json in vision_neuron_config:
#   "vision_neuron_config": {
#       "default_image_width": 640,
#       "default_image_height": 320,
#       ...
#   }
# TODO: add padding to handle dynamic input image resolution
DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT = 640, 320
DEFAULT_PIXELS_PER_IMAGE = 1012
# Max grid size: 72 * 14 (patch_size) = 1008 pixels per dimension
# Total pixels: 1008^2 ≈ 1M, matching Qwen2-VL's default max_pixels (28*28*1280)
MAX_GRID_SIZE = 72  # Equals 1008 pixels (for backward compatibility)
