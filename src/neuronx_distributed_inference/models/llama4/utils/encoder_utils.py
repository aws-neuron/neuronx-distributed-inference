# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def pad_vision_embeddings(vision_embeddings, pad_limit):
    padding_size = pad_limit - vision_embeddings.shape[1]
    if padding_size > 0:
        padding = torch.full(
            (vision_embeddings.shape[0], padding_size, vision_embeddings.shape[2]), 0, dtype=vision_embeddings.dtype, device=vision_embeddings.device
        )
        vision_embeddings = torch.cat([vision_embeddings, padding], dim=1)
    else:
        vision_embeddings = vision_embeddings[:, :pad_limit, :]

    return vision_embeddings


def pad_image_tensor(tensor, target_chunks):
    """
    Pads an image tensor by adding zero-filled chunks at the end until it reaches
    the target number of chunks.
    Note: we are padding in the end, instead of insert padded chunk between
    of the <first N-1> and <Global chunk>, because we've changed the VE to be traced
    by the image_flatten only and no complex depad logic is required.

    Args:
        tensor (torch.Tensor): The original image tensor of shape [num_original_chunks, C, H, W].
        target_chunks (int): The desired total number of chunks after padding. Should be equal
                            to the target bucket size supporting number of chunks.

    Returns:
        torch.Tensor: The padded image tensor of shape [target_chunks, C, H, W].
    """

    original_chunks = tensor.shape[0]
    num_chunks_to_add = target_chunks - original_chunks

    # If the tensor already has the required number of chunks or more, return it as is
    if num_chunks_to_add <= 0:
        return tensor

    # Create zero-filled padding chunks
    pad_chunks = torch.zeros((num_chunks_to_add, *tensor.shape[1:]), dtype=tensor.dtype)

    # Concatenate original tensor with padding chunks
    padded_tensor = torch.cat([tensor, pad_chunks], dim=0)

    return padded_tensor


def pad_image_mask(image_mask, num_original_chunks, num_total_chunks, patch_size=144):
    """
    Pads an image mask to accommodate additional chunks while preserving token alignment.

    Args:
        image_mask (torch.Tensor): The original image mask of shape [batch_size, seq_len, 1].
        num_original_chunks (int): The number of original image chunks before padding.
        num_total_chunks (int): The total number of chunks after padding.
        patch_size (int, optional): Number of patches per chunk. Default is 144.

    Returns:
        torch.Tensor: The padded image mask of shape [batch_size, new_seq_len, 1].
    """

    batch_size, seq_len, _ = image_mask.shape
    num_chunks_to_add = num_total_chunks - num_original_chunks

    # If no padding is needed, return the original mask
    if num_chunks_to_add <= 0:
        return image_mask

    step_size = patch_size + 1  # Each chunk spans `patch_size` + 1 step
    new_seq_len = seq_len + (num_chunks_to_add * step_size)

    # Initialize a new mask with zeros
    padded_image_mask = torch.zeros((batch_size, new_seq_len, 1), dtype=image_mask.dtype)

    # Find the first `<|patch|>` token position (pos1)
    pos1 = torch.nonzero(image_mask.squeeze(-1) == 1, as_tuple=True)[1][
        0
    ].item()  # First `1` in mask

    # Compute positions for all original and new chunk patches
    patch_positions = pos1 + torch.arange(num_total_chunks) * step_size  # Vectorized computation

    # Assign `1`s for all valid patch positions (Vectorized)
    indices = patch_positions.unsqueeze(1) + torch.arange(
        patch_size
    )  # Shape: [num_total_chunks, patch_size]
    padded_image_mask[:, indices.flatten(), :] = 1  # Flatten to apply in one step

    # Add `<|image|>` and `<|image_end|>` tokens
    pre_image_token_pos = (
        patch_positions[-2] + patch_size
    )  # Position for `<|image|>` (global chunk)
    image_token_pos = patch_positions[-1] + patch_size  # Position for `<|image_end|>`

    padded_image_mask[:, pre_image_token_pos : pre_image_token_pos + 2, :] = 0  # `<|image|>` token
    padded_image_mask[:, image_token_pos : image_token_pos + 1, :] = 1  # `<|image_end|>` token

    # Copy original prefix
    padded_image_mask[:, :pos1, :] = image_mask[:, :pos1, :]

    # Copy End Text Tokens
    pos_last_chunk = pos1 + (num_original_chunks - 1) * step_size  # Last chunk position
    num_end_tokens = image_mask.shape[1] - (pos_last_chunk + patch_size + 2)

    if num_end_tokens < 0:
        raise ValueError("Number of end tokens cannot be negative.")

    end_token_start = image_token_pos + 1
    padded_image_mask[:, end_token_start : end_token_start + num_end_tokens, :] = image_mask[
        :, -num_end_tokens:, :
    ]

    return padded_image_mask


def depad_output(
    padded_output, padded_image_mask, num_original_chunks, num_total_chunks, patch_size=144
):
    """
    Removes padded image chunks from the model output, restoring the original sequence length.

    Args:
        padded_output (torch.Tensor): The padded model output of shape [1, total_length_true, hidden_size].
        padded_image_mask (torch.Tensor): The mask tensor of shape [1, total_length_true, 1].
        num_original_chunks (int): The number of original image chunks before padding.
        num_total_chunks (int): The total number of chunks after padding.
        patch_size (int, optional): Number of patches per chunk. Default is 144.

    Returns:
        torch.Tensor: The depadded output tensor of shape [1, original_length, hidden_size].
    """

    # Ensure the mask is a boolean tensor
    valid_indices = padded_image_mask.squeeze(-1).bool()  # Shape: [1, total_length_true]
    image_start_idx = torch.nonzero(valid_indices, as_tuple=True)[1][
        0
    ].item()  # First `1` position in mask

    # Find the position of `num_original_chunks - 1`
    step_size = patch_size + 1
    pos_last_valid_chunk = image_start_idx + (
        (num_original_chunks - 1) * step_size
    )  # Up to the last original chunk

    # Find the start position of the last valid region (global chunk & text)
    pos_last_padded_chunk = image_start_idx + (
        (num_total_chunks - 1) * step_size
    )  # Start of last valid region

    # Create the valid indices to keep
    valid_keep_indices = torch.cat(
        [
            torch.arange(0, pos_last_valid_chunk),
            torch.arange(pos_last_padded_chunk, padded_output.shape[1]),
        ]
    )

    depadded_output = padded_output[:, valid_keep_indices, :]

    # Ensure the output shape is correct
    expected_length = len(valid_keep_indices)
    assert (
        depadded_output.shape[1] == expected_length
    ), f"Depadded output has incorrect shape: {depadded_output.shape}, expected {expected_length}."

    return depadded_output


def generate_llama4_vision_encoder_buckets(dp_degree, max_chunks):
    """
    Generates vision encoder buckets

    Args:
        dp_degree (int): Vision encoder model's dp degree.
        max_chunks (int): The maximum number of chunks that the model should support
        (will be the size of the biggest bucket)

    Returns:
        list[int]: List of bucket sizes for the vision encoder model.
    """
    # need to multiply by 2 because attention kernel expects even batch dimension
    # bucket_step = dp_degree * 2
    # largest_bucket_index = math.ceil(max_chunks/bucket_step)
    # return [bucket_step * i for i in range(1, largest_bucket_index + 1)]

    # Fix me: use the logic above once all buckets work e2e.
    return [max_chunks]


def generate_positions_from_mask(mask):
    """
    Generate position indices from a boolean mask.

    Args:
    mask (torch.Tensor): A 1D or 2D boolean tensor

    Returns:
    torch.Tensor: A 1D tensor containing the indices where the mask is True
    """
    # if mask.dim() == 2:
    #     mask = mask.squeeze(0)  # Remove batch dimension if present
    return torch.nonzero(mask).squeeze()


def pad_positions(positions, target_size, fill_value):
    """
    Pad the positions tensor to a target size.

    Args:
    positions (torch.Tensor): A 1D tensor containing position indices
    target_size (int): The desired size of the padded tensor
    fill_value (int): The value used for padding

    Returns:
    torch.Tensor: A 3D tensor of shape (1, target_size, 1) containing padded position indices
    """
    padding_size = target_size - len(positions)
    if padding_size > 0:
        padding = torch.full(
            (padding_size,), fill_value, dtype=positions.dtype, device=positions.device
        )
        positions_padded = torch.cat([positions, padding])
    else:
        positions_padded = positions[:target_size]

    # Add batch dimension and an extra dimension at the end
    return positions_padded.unsqueeze(0).unsqueeze(-1)  # Shape: [1, x, 1]


def scatter_by_index_put(h_image, encoded_patches_proj, positions):
    """
    Scatter encoded patches into an image tensor using index_put_ operation.
    Assumes batch size is always 1.

    Args:
    h_image (torch.Tensor): The target image tensor of shape (1, max_positions, embedding_dim)
    encoded_patches_proj (torch.Tensor): The encoded patches to be scattered, of shape (num_patches, patch_size, embedding_dim)
    positions (torch.Tensor): The positions where patches should be scattered, of shape (1, num_positions, 1)

    Returns:
    torch.Tensor: The updated image tensor with scattered patches
    """
    _, max_positions, embedding_dim = h_image.shape

    # Create a new tensor instead of modifying h_image in-place
    h_image_new = h_image.clone()

    # Flatten encoded_patches_proj
    encoded_patches_flat = encoded_patches_proj.view(-1, embedding_dim)

    # Flatten positions
    positions = positions.view(-1)

    # Use index_put_ to scatter the embeddings
    h_image_new.view(-1, embedding_dim).index_put_(
        (positions,), encoded_patches_flat, accumulate=False
    )

    return h_image_new
