from typing import List, Union

import torch


def slice_by_image_hw(
    original_tensor: torch.Tensor,
    hs: Union[torch.Tensor, List[int]],
    ws: Union[torch.Tensor, List[int]],
) -> List[torch.Tensor]:
    """
    Slice a tensor into multiple tensors based on image height and width pairs.

    This is equivalent to: original_tensor.split([h * w for h, w in zip(hs, ws)])

    Args:
        original_tensor: The tensor to slice along dim 0, shape (N, ...)
        hs: Heights for each image
        ws: Widths for each image

    Returns:
        List of tensors, each with shape (h*w, ...)
    """
    # Convert to lists if tensors
    if isinstance(hs, torch.Tensor):
        hs = hs.tolist()
    if isinstance(ws, torch.Tensor):
        ws = ws.tolist()

    # Calculate indices for each image
    slice_indices = []
    curr_idx = 0
    for h, w in zip(hs, ws):
        slice_size = int(h * w)
        slice_indices.append(list(range(curr_idx, curr_idx + slice_size)))
        curr_idx += slice_size

    assert curr_idx == original_tensor.shape[0], (
        f"Sum of grid sizes {list(zip(hs, ws))} does not match "
        f"tensor length {original_tensor.shape[0]}."
    )

    # Slice tensor into list
    result = []
    for indices in slice_indices:
        split = torch.index_select(
            original_tensor,
            dim=0,
            index=torch.tensor(indices, device=original_tensor.device, dtype=torch.long),
        )
        result.append(split)

    return result
