# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for GLM-5.2 FP8 weight handling (no Neuron device required)."""

import os
import sys

import pytest
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from modeling_glm5 import (  # noqa: E402
    FP8_E4M3_NEURON_MAX,
    _dequantize_fp8_blockwise,
    _rescale_fp8_for_neuron,
)


def test_blockwise_dequant_shape_and_values():
    # 4x4 weight, 2x2 blocks -> 2x2 scale grid.
    block = [2, 2]
    w = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out = _dequantize_fp8_blockwise(w, scales, block, torch.bfloat16)
    assert out.shape == (4, 4)
    assert out.dtype == torch.bfloat16
    # Each 2x2 block scaled by its scale (weights are all 1.0).
    assert torch.allclose(out[:2, :2].float(), torch.ones(2, 2))       # scale 1
    assert torch.allclose(out[:2, 2:].float(), torch.full((2, 2), 2.0))  # scale 2
    assert torch.allclose(out[2:, :2].float(), torch.full((2, 2), 3.0))  # scale 3
    assert torch.allclose(out[2:, 2:].float(), torch.full((2, 2), 4.0))  # scale 4


def test_blockwise_dequant_truncates_partial_blocks():
    # 3x3 weight with 2x2 blocks -> 2x2 scales expand to 4x4, must truncate to 3x3.
    w = torch.ones(3, 3, dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 2, dtype=torch.float32)
    out = _dequantize_fp8_blockwise(w, scales, [2, 2], torch.bfloat16)
    assert out.shape == (3, 3)


def test_nan_clamp_bounds_large_fp8_values():
    # FP8 e4m3 can represent values up to 448, but Neuron treats exp-15 as NaN;
    # dequant must clamp the FP8 magnitude to FP8_E4M3_NEURON_MAX (240) first.
    assert FP8_E4M3_NEURON_MAX == 240.0
    w = torch.tensor([[448.0]], dtype=torch.float8_e4m3fn)  # near FP8 max
    scales = torch.tensor([[1.0]], dtype=torch.float32)
    out = _dequantize_fp8_blockwise(w, scales, [1, 1], torch.float32)
    assert out.abs().item() <= FP8_E4M3_NEURON_MAX + 1e-3


def test_rescale_fp8_preserves_effective_value():
    # Rescaling shrinks the FP8 weight but grows the scale by the same factor,
    # so weight*scale (the effective dequantized value) is preserved.
    w = torch.tensor([[120.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    before = w.to(torch.float32) * scale
    rescaled_w, rescaled_scale = _rescale_fp8_for_neuron(w, scale)
    after = rescaled_w.to(torch.float32) * rescaled_scale
    assert rescaled_w.dtype == torch.float8_e4m3fn
    # Effective value preserved within FP8 rounding.
    assert torch.allclose(before, after, rtol=0.05)
    # Rescaled FP8 magnitude is now within Neuron-safe range.
    assert rescaled_w.to(torch.float32).abs().item() <= FP8_E4M3_NEURON_MAX + 1e-3
