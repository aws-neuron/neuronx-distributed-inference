from typing import List, Tuple

import torch


def pad_tensor(
    unpadded_tensor: torch.Tensor, target_shape: List[int]
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

    # Calculate padding for each dimension
    padding = []
    for current, target in zip(
        original_shape[::-1], target_shape[::-1]
    ):  # Reverse for torch.nn.functional.pad format
        pad_size = max(target - current, 0)
        padding.extend([0, pad_size])  # Pad only at the end of each dimension

    # Create the padded tensor
    padded_tensor = torch.nn.functional.pad(unpadded_tensor, padding, mode="constant", value=0)

    # Create the mask
    mask = torch.zeros(target_shape, dtype=torch.int, device=unpadded_tensor.device)
    slices = tuple(slice(0, size) for size in original_shape)
    mask[slices] = 1

    return padded_tensor, mask


def unpad_tensor(padded_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Given a mask, unpad an input tensor to original shape.

    Args:
        padded_tensor (torch.Tensor): The padded tensor.
        mask (torch.Tensor): A mask tensor with 1 for original values and 0 for padded positions.

    Returns:
        Unpadded_tensor (torch.Tensor): The unpadded tensor in the original shape.
    """

    # Identify the bounds of the original tensor based on the mask
    non_zero_indices = torch.nonzero(mask, as_tuple=True)
    slices = tuple(
        slice(torch.min(indices).item(), torch.max(indices).item() + 1)
        for indices in non_zero_indices
    )

    # Extract the unpadded tensor
    unpadded_tensor = padded_tensor[slices]

    return unpadded_tensor
