import torch
import torch.nn.functional as F

##################################################
# Padding utility
##################################################


def pad_to_shape(
    input: torch.Tensor,
    expected_shape: torch.Size,
    mode="constant",
    value=None,
):
    """
    Pad input_tensor to expected_shape. It always pads at the end of each dim.

    If input.shape == expected_shape, return input directly.
    """
    if input.shape == expected_shape:
        return input

    for dim, expected in enumerate(expected_shape):
        input = pad_at_end(
            input, dim=dim, padded_len=expected, mode=mode, value=value
        )
    return input


def pad_at_end(
    input: torch.Tensor,
    dim: int,
    padded_len: int,
    mode="constant",
    value=None,
):
    """
    Pad input_tensor on a specific dim to len_after_padding, and it always
    pad at the end of that dim.

    If input.shape[dim] == padded_len, return input directly.
    """
    assert isinstance(input, torch.Tensor)
    assert input.ndim > dim >= 0
    assert padded_len >= input.shape[dim] >= 0

    if padded_len == input.shape[dim]:
        return input

    pad_shape = [0, 0] * (input.ndim - 1 - dim)
    pad_shape.extend([0, padded_len - input.shape[dim]])
    padded_tensor = F.pad(
        input,
        pad_shape,
        mode=mode,
        value=value,
    )
    return padded_tensor
