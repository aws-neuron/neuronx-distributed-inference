from typing import Tuple
import pytest
import torch

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import convert_gate_up_proj

@pytest.mark.parametrize(
    "tensor, is_bias, expected",
    [
        (torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).view(2, 6, 1), False, torch.Tensor([1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).view(2, 1, 6)),
        (torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).view(2, 6), True, torch.Tensor([1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).view(2, 6))
    ],
)
def test_convert_gate_up_proj(tensor, is_bias, expected):
    actual = convert_gate_up_proj(tensor, is_bias=is_bias)
    assert torch.equal(expected, actual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])