from typing import List, Tuple

import torch


def pad_tensor(
    unpadded_tensor: torch.Tensor, target_shape: List[int], pad_value: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad an input tensor to a target shape in multiple dimensions and generate a mask tensor.

    Args:
        unpadded_tensor (torch.Tensor): The input tensor to pad.
        target_shape (List[int]): The target shape to pad the tensor to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_tensor (torch.Tensor): The padded tensor with the target shape.
            - mask (torch.Tensor): A mask tensor with 1 for original values and 0 for padded positions.
    """

    original_shape = list(unpadded_tensor.shape)
    assert len(original_shape) == len(
        target_shape
    ), f"Target shape {target_shape} must have the same number of dimensions as the input tensor {original_shape}."

    # Create the padded tensor
    # we sometimes see numerical errors with different compiler version if we pad with 0, 1, or random value
    # So here we default to use the actual max value of unpadded_tensor unless explicitly specified
    if pad_value is None:
        pad_value = torch.max(unpadded_tensor).item()

    # Calculate padding for each dimension
    padding = []
    for current, target in zip(
        original_shape[::-1], target_shape[::-1]
    ):  # Reverse for torch.nn.functional.pad format
        pad_size = max(target - current, 0)
        padding.extend([0, pad_size])  # Pad only at the end of each dimension

    padded_tensor = torch.nn.functional.pad(unpadded_tensor, padding, mode="constant", value=pad_value)

    # Store the start and end index of the original tensors
    original_idx_slices = [[0, size] for size in original_shape]

    return padded_tensor, original_idx_slices


def unpad_tensor(padded_tensor: torch.Tensor, original_idx_slices: List[List]) -> torch.Tensor:
    """
    Given a mask, unpad an input tensor to original shape.

    Args:
        padded_tensor (torch.Tensor): The padded tensor.
        mask (torch.Tensor): A mask tensor with 1 for original values and 0 for padded positions.

    Returns:
        Unpadded_tensor (torch.Tensor): The unpadded tensor in the original shape.
    """

    # Extract the unpadded tensor
    unpadded_tensor = padded_tensor[tuple(slice(start, end) for start, end in original_idx_slices)]

    return unpadded_tensor
