import torch
import pytest

from neuronx_distributed_inference.utils.testing import build_function
from neuronx_distributed_inference.models.qwen3_vl.utils.slicing import slice_by_image_hw


def hf_reference_impl(original_tensor, hs, ws):
    return original_tensor.split([h * w for h, w in zip(hs, ws)])


@pytest.mark.parametrize("grid_hw, hidden_size", [
    (torch.tensor([[20, 40], [20, 40]]), 1152),
    (torch.tensor([[20, 40], [30, 30]]), 1152) # test different resolution, but currently limited to work with only compiled resolution
    # TODO: support when inference resolution is different from compiled
])
def test_original_vs_neuron(grid_hw, hidden_size):
    grid_hs, grid_ws = grid_hw[:, 0], grid_hw[:, 1]

    total_num_patches = grid_hw.prod(dim=1).sum()

    original_patch_pos_embeds = torch.randn(total_num_patches, hidden_size)

    # Golden - Run HF impl on CPU
    hf_output = hf_reference_impl(original_patch_pos_embeds, grid_hs, grid_ws)

    # Run on Neuron device
    neuron_func = build_function(
        func=slice_by_image_hw,
        example_inputs=[(torch.ones_like(original_patch_pos_embeds), grid_hs, grid_ws)],
        tp_degree=2)
    neuron_output = neuron_func(original_patch_pos_embeds, grid_hs, grid_ws)

    torch.testing.assert_close(hf_output, neuron_output, rtol=0, atol=0)
    print("passed")
