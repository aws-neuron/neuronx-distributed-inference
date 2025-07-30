# Meta Llama 3 is licensed under the Meta Llama 3 Community License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE in the
# # current directory, mllama/.

import collections
import logging
from typing import List

import torch
from PIL import Image

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.mllama.image_transform import custom_image_preprocessing

logger = logging.getLogger(__name__)

NUM_IMAGE_PER_PROMPT = 1


def get_input_shape(input_ids):
    if input_ids is None:
        raise ValueError("input_ids cannot be None")
    if len(input_ids) == 0:
        raise ValueError("input_ids cannot be empty")

    # Proceed if valid
    bs, _ = input_ids.shape
    return bs


def create_vision_mask(
    input_ids: torch.Tensor,
    vision_token: int,
) -> torch.Tensor:
    bs = get_input_shape(input_ids)
    vision_masks = []
    for batch_line in range(bs):
        tokens = input_ids[batch_line]
        vision_token_locations = [i for i, token in enumerate(tokens) if token == vision_token]
        if not vision_token_locations:
            vision_masks.append([[0, -1]])

        if len(vision_token_locations) == 1:
            # only one image present, unmask until end of sequence
            vision_masks.append([[vision_token_locations[0], -1]])
    vision_masks = torch.tensor(vision_masks)
    return vision_masks


def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def get_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img


def get_image_tensors(
    config: InferenceConfig, batch_images: List[List], is_for_context_encoding=True
):
    bsz = len(batch_images)

    if len(batch_images[0]) == 0 or config.neuron_config.skip_vision:
        logger.info("Setting empty vision inputs...")
        if config.neuron_config.skip_vision or (not is_for_context_encoding):
            # We use dummy pixel_values for:
            # context-encoding when skip_vision==True, as we don't execute xatten layers
            # token-generation, actual pixel_values are aliased
            empty_pixel_values = torch.tensor(
                [0] * config.neuron_config.batch_size, dtype=torch.int32
            )
        else:
            empty_pixel_values = torch.zeros(
                [
                    bsz,
                    NUM_IMAGE_PER_PROMPT,
                    config.vision_config.max_num_tiles,
                    config.vision_config.num_channels,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                ],
                dtype=config.neuron_config.torch_dtype,
            )
        empty_aspect_ratios = torch.ones(
            (bsz, NUM_IMAGE_PER_PROMPT, 2),
            dtype=torch.int32,
        )
        num_chunks = torch.zeros((bsz, 1), dtype=torch.int32)  # dummy num_chunks, will not be used
        has_image = torch.zeros([bsz], dtype=torch.int32)
        return empty_pixel_values, empty_aspect_ratios, num_chunks, has_image

    # preprocess PIL images to pixel values tensors
    pixel_values, aspect_ratios, num_chunks = custom_image_preprocessing(config, batch_images)
    assert (
        pixel_values.dtype == config.neuron_config.torch_dtype
    ), f"pixel_values dtype {pixel_values.dtype} does not match config {config.neuron_config.torch_dtype}"

    has_image = torch.ones([bsz], dtype=torch.int32)
    return pixel_values.clone().detach(), aspect_ratios, num_chunks, has_image


def add_instruct(prompt: str, has_image: torch.Tensor):
    if has_image[0]:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
