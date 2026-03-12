from PIL import Image
from neuronx_distributed_inference.models.qwen2_vl.utils.constants import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
)


def prepare_generation_inputs_hf(text_prompt, image_paths, hf_qwen2_vl_processor, role="user", config=None):
    """
    Prepare inputs for Qwen2-VL generation.

    Args:
        text_prompt: The text prompt
        image_paths: List of paths to the image file
        hf_qwen2_vl_processor: HuggingFace processor
        role: Role for the message (default: "user")
        config: Optional InferenceConfig object. If provided, will use default_image_width and
                default_image_height from config.vision_config.neuron_config.
                Otherwise, falls back to DEFAULT_IMAGE_WIDTH and DEFAULT_IMAGE_HEIGHT constants.
    """
    # Get image dimensions from config if available, otherwise use constants
    if config is not None and hasattr(config, 'vision_config'):
        image_width = getattr(config.vision_config.neuron_config, 'default_image_width', DEFAULT_IMAGE_WIDTH)
        image_height = getattr(config.vision_config.neuron_config, 'default_image_height', DEFAULT_IMAGE_HEIGHT)
    else:
        image_width = DEFAULT_IMAGE_WIDTH
        image_height = DEFAULT_IMAGE_HEIGHT

    messages = [
        {
            "role": role,
            "content": [
                {"type": "text", "text": text_prompt},
            ]
        },
    ]
    image_data = None
    default_size = (image_width, image_height)
    if image_paths:
        image_data = []
        for image_path in image_paths:
            image = Image.open(image_path).resize(default_size)
            image_data.append(image)
            info = {"type": "image", "resized_height": image_height, "resized_width": image_width}
            messages[0]["content"].append(info)

    text_inputs = hf_qwen2_vl_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = hf_qwen2_vl_processor(
        text=[text_inputs],
        images=image_data,
        return_dict=True,
        return_tensors="pt",
        padding=True,)

    return inputs
