import math
from typing import Optional, Tuple
from unittest.mock import Mock
import pytest
import torch
import torch_neuronx

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import convert_moe_packed_tensors
from neuronx_distributed_inference.models.gpt_oss.mx_layout_transform import (
    convert_down_proj_fp4,
    convert_gate_up_proj,
    convert_gate_up_proj_fp4,
    get_h_tiling_shard_i,
    get_i_tiling_shard_i,
    tile_transpose_down_proj_for_mxfp4_kernel,
    tile_transpose_gate_up_for_mxfp4_kernel,
    shuffle_hidden_dim,
    unshuffle_hidden_dim,
    convert_hf_format_state_dict_mxfp4_compute
)
from neuronx_distributed.quantization.quantization_config import QuantizedDtype
from neuronx_distributed.quantization.microscaling.transform_weights import get_mxfp4_tensor_from_uint16, split_byte_4bit_tensor, pack_byte_4bit_tensor


@pytest.mark.parametrize(
    "tensor, is_bias, expected",
    [
        (torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).view(2, 6, 1), False, torch.Tensor([1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).view(2, 1, 6)),
        (torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).view(2, 6), True, torch.Tensor([1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).view(2, 6))
    ],
)
def test_convert_gate_up_proj(tensor, is_bias, expected):
    actual = convert_gate_up_proj(tensor, is_bias=is_bias, concat_gate_up_proj=True)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "tensor, expected_gate, expected_up",
    [
        (
            torch.tensor([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0], dtype=torch.uint8).view(1, 4, 2, 1), # [E=1, 2I=4, H=2 * 1[x2]]
            torch.tensor([0x3412, 0xBC9A], dtype=torch.uint16).view([1, 1, 2]),
            torch.tensor([0x7856, 0xF0DE], dtype=torch.uint16).view([1, 1, 2]),
        ),
    ]
)
def test_convert_gate_up_proj_fp4(tensor, expected_gate, expected_up):
    actual_gate, actual_up = convert_gate_up_proj_fp4(tensor)

    assert torch.equal(expected_gate, actual_gate)
    assert torch.equal(expected_up, actual_up)

    actual = torch.cat([actual_gate, actual_up], dim=2)
    actual = move_packed_dim_to_outer(actual)

    # the expected order was determined by validating that the two flows are consistent:
    # - convert_moe_packed_tensors (dequantize) -> convert_gate_up_proj
    # - convert_gate_up_proj -> get_mxfp4_tensor_from_uint16
    # The second flow simulates that a checkpoint is loaded to HBM in MXFP4 format
    # and is dequantized at the last possible moment

    # test that dequantize -> convert gate up == convert gate up mx -> dequantize
    # isolated only to blocks, not scales yet
    scales = torch.full(tensor.shape[:-1], 127, dtype=torch.uint8)
    expected_dequant = convert_moe_packed_tensors(tensor, scales)
    expected_dequant = convert_gate_up_proj(expected_dequant)

    scales = torch.full(actual.shape[:-1], 127, dtype=torch.uint8)
    actual_dequant = get_mxfp4_tensor_from_uint16(actual, scales, output_quad_row=True)

    assert torch.equal(expected_dequant, actual_dequant)


@pytest.mark.parametrize(
    "tensor, expected_down",
    [
        (
            torch.tensor([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0], dtype=torch.uint8).view(1, 4, 2, 1), # [E=1, I=4, H=2 * 1[x2]]
            torch.tensor([0x3412, 0x7856, 0xBC9A, 0xF0DE], dtype=torch.uint16).view([1, 1, 4]),
        ),
    ]
)
def test_convert_down_proj_fp4(tensor, expected_down):
    actual = convert_down_proj_fp4(tensor)

    assert torch.equal(expected_down, actual)
    actual = move_packed_dim_to_outer(actual)

    # the expected order was determined by validating that the two flows are consistent:
    # - convert_moe_packed_tensors (dequantize) -> convert_gate_up_proj
    # - convert_gate_up_proj -> get_mxfp4_tensor_from_uint16
    # The second flow simulates that a checkpoint is loaded to HBM in MXFP4 format
    # and is dequantized at the last possible moment

    # test that dequantize -> convert down == convert down mx -> dequantize
    # isolated only to blocks, not scales yet
    scales = torch.full(tensor.shape[:-1], 127, dtype=torch.uint8)
    expected_dequant = convert_moe_packed_tensors(tensor, scales)
    expected_dequant = expected_dequant.transpose(1, 2)

    scales = torch.full(actual.shape[:-1], 127, dtype=torch.uint8)
    actual_dequant = get_mxfp4_tensor_from_uint16(actual, scales, output_quad_row=True)

    assert torch.equal(expected_dequant, actual_dequant)


def move_packed_dim_to_outer(tensor: torch.Tensor):
    """
    [E, H // 4, I * 1H[x4]] => [E, H, I // 4[x4]]
    [E, I // 4, H * 1I[x4]] => [E, I, H // 4[x4]]
    """
    assert tensor.ndim >= 3, "Tensor must have at least 3 dimensions corresponding to E, I, H dims."
    original_dtype = tensor.dtype
    tensor = tensor.unsqueeze(-1).contiguous()  # [E, H // 4, I, 1H[x4]]
    tensor = tensor.view(torch.uint8)  # [E, H // 4, I, 2H[x2]]
    tensor = split_byte_4bit_tensor(tensor)  # [E, H // 4, I, 4]
    tensor = tensor.transpose(2, 3).contiguous()  # [E, H // 4, 4, I]
    tensor = tensor.view([tensor.shape[0], -1, tensor.shape[3]])  # [E, H, I]
    tensor = pack_byte_4bit_tensor(tensor)  # [E, H, I // 2[x2]]
    tensor = tensor.view(original_dtype)
    return tensor


pmax = 128
q_width = 4
q_height = 8

def undo_tile_transpose_gate_up_for_mxfp4_kernel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    tp_degree: int,
    *,
    quantized_dtype: QuantizedDtype,
    E_size: int = 128,
    H_size: int = 3072,
    I_size: int = 3072,
    I_size_actual: Optional[int] = None,
    hidden_act_bias: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Undo tiling and transpose operations for gate/up projection tensors from MXFP4 kernel format.
    
    Reverse transposes applied:
    - Weight: Original shape [E, ]
              View as [E, Q_BLOCK_H_TILE, Q_HEIGHT_H, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_PACKED_H]
              Permute to [E, N_H_TILES, Q_BLOCK_H_TILE, Q_HEIGHT_H, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I, Q_PACKED_H]
              View as [E, 2I, H // 32, 16]
    - Scale: View as [E, Q_BLOCK_H_TILE, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I]
             Permute to [E, N_H_TILES, Q_BLOCK_H_TILE, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I]
             View as [E, 2I, H // 32]
    - Bias: View as [E, shard, Q_BLOCK_I_TILE, Q_HEIGHT_I, NUM_I_TILES, Q_WIDTH_I]
            to [E, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I]
            View as [E, 2I]
    
    Args:
        weight (torch.Tensor): Tiled gate/up projection weights
        scale (torch.Tensor): Tiled quantization scales
        bias (torch.Tensor): Tiled bias tensors
        tp_degree (int): Tensor parallel degree for sharding
        quantized_dtype (QuantizedDtype): Quantization data type
        gate_up_I_concat_dim (int): Dimension for concatenating gate/up in I axis. Defaults to 0.
        gate_up_I_concat_dim_bias (int): Dimension for concatenating gate/up bias. Defaults to 0.
        E_size (int): Expert dimension size. Defaults to 128.
        H_size (int): Hidden dimension size. Defaults to 3072.
        I_size (int): Input dimension size. Defaults to 3072.
        I_size_actual: (int): Unpadded input dimension size, used to avoid adding hidden_act_bias to padded regions of up_proj_bias. Defaults to None.
        hidden_act_bias (float): Hidden activation bias, which will be added to up_proj_bias. Defaults to 0.0.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - weight: [E, 2I, H // 32, 16] - Original interleaved weight format
            - scale: [E, 2I, H // 32] - Original interleaved scale format
            - bias: [E, 2I] - Original interleaved bias format
    """
    num_H_tiles, q_blocks_per_H_tile = get_h_tiling_shard_i(H_size)
    num_I_tiles, q_blocks_per_I_tile = get_i_tiling_shard_i(tp_degree, I_size)
    q_packed = q_width // quantized_dtype.get_packed_count()
    
    # Expected weight shape after adding back tile dimensions
    weight_shape = [E_size, q_blocks_per_H_tile, q_height, 2, num_H_tiles, tp_degree, num_I_tiles, q_width, q_blocks_per_I_tile, q_height, q_packed]

    w_concat_dim = 3
    weight = weight.reshape(*weight_shape)
    w_gate, w_up = weight.split(1, dim=w_concat_dim)
    w_gate = w_gate.squeeze(w_concat_dim)
    w_up = w_up.squeeze(w_concat_dim)
    
    E, Q_BLOCK_H_TILE, Q_HEIGHT_H, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_PACKED_H = list(range(w_gate.ndim))
    w_gate = w_gate.permute(E, N_H_TILES, Q_BLOCK_H_TILE, Q_HEIGHT_H, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I, Q_PACKED_H)
    w_up = w_up.permute(E, N_H_TILES, Q_BLOCK_H_TILE, Q_HEIGHT_H, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I, Q_PACKED_H)
    w_gate = w_gate.reshape(E_size, H_size // 4, I_size, q_packed)
    w_up = w_up.reshape(E_size, H_size // 4, I_size, q_packed)
    w_gate = w_gate.permute(0, 2, 1, 3)
    w_up = w_up.permute(0, 2, 1, 3)
    w_gate = w_gate.reshape(E_size, I_size, H_size // 32, -1).contiguous().view(torch.uint8)
    w_up = w_up.reshape(E_size, I_size, H_size // 32, -1).contiguous().view(torch.uint8)
    weight = torch.empty(E_size, I_size * 2, H_size // 32, 16, dtype=torch.uint8)
    weight[:, ::2, ...] = w_gate
    weight[:, 1::2, ...] = w_up
    
    # Expected scale shape after adding back tile dimensions
    scale_shape = [E_size, q_blocks_per_H_tile, 2, num_H_tiles, tp_degree, num_I_tiles, q_width, q_blocks_per_I_tile, q_height]
    
    s_concat_dim = 2
    scale = scale.reshape(*scale_shape)
    s_gate, s_up = scale.split(1, dim=s_concat_dim)
    s_gate = s_gate.squeeze(s_concat_dim)
    s_up = s_up.squeeze(s_concat_dim)

    # Undo transpose
    E, Q_BLOCK_H_TILE, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I = list(range(s_gate.ndim))
    s_gate = s_gate.permute(E, N_H_TILES, Q_BLOCK_H_TILE, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I)
    s_up = s_up.permute(E, N_H_TILES, Q_BLOCK_H_TILE, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I)
    s_gate = s_gate.reshape(E_size, H_size // 32, I_size)
    s_up = s_up.reshape(E_size, H_size // 32, I_size)
    s_gate = s_gate.permute(0, 2, 1)
    s_up = s_up.permute(0, 2, 1)
    scale = torch.empty(E_size, I_size * 2, H_size // 32, dtype=torch.uint8)
    scale[:, ::2, ...] = s_gate
    scale[:, 1::2, ...] = s_up
    
    # Expected bias shape after adding back tile dimensions
    bias_shape = [E_size, tp_degree, q_blocks_per_I_tile, q_height, 2, num_I_tiles, q_width]

    # Add the concatenation dim and split to undo concat
    b_concat_dim = 4
    bias = bias.reshape(*bias_shape)
    b_gate, b_up = bias.split(1, dim=b_concat_dim)
    b_gate = b_gate.squeeze(b_concat_dim)
    b_up = b_up.squeeze(b_concat_dim)

    # Undo transpose
    # [E, 16_I, 8_I, I/512, 4_I] -> [E, I/512, 16_I, 8_I, 4_I]
    E, shard, Q_BLOCK_I_TILE, Q_HEIGHT_I, NUM_I_TILES, Q_WIDTH_I = list(range(b_gate.ndim))
    b_gate = b_gate.permute(E, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I)
    b_up = b_up.permute(E, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I)
    b_gate = b_gate.reshape(E_size, I_size)
    b_up = b_up.reshape(E_size, I_size)

    # Undo b_up increment by hidden_act_bias
    # NOTE: Addition is performed in fp32 to avoid accumulating additional logit error
    if hidden_act_bias != 0.0:
        # Do not decrement padded region of I dim if I dim is padded
        b_I_upper_limit = I_size_actual if I_size_actual else I_size
        b_up[:, :b_I_upper_limit].sub_(torch.tensor([hidden_act_bias], dtype=torch.float32))

    bias = torch.empty(E_size, I_size * 2, dtype=torch.bfloat16)
    bias[:, ::2] = b_gate
    bias[:, 1::2] = b_up

    return weight, scale, bias

@pytest.mark.parametrize(
    "E_size,H_size,I_size,I_size_actual", [
        pytest.param(2, 3072, 3072, None),
        pytest.param(2, 3072, 3072, 2880),
    ]
)
@pytest.mark.parametrize(
    "hidden_act_bias,bias_atol,bias_rtol", [
        pytest.param(0.0, 0.0, 0.0),
        pytest.param(1.0, 1.5625e-2, 4.69970703125e-3),
        pytest.param(2.0, 1.5625e-2, 4.730224609375e-3),
    ]
)
def test_tile_transpose_gate_up_for_mxfp4_kernel(E_size, H_size, I_size, I_size_actual, hidden_act_bias, bias_atol, bias_rtol):
    
    expected_weights = torch.randint(low=0, high=256, size=(E_size, 2 * I_size, H_size // 32, 16), dtype=torch.uint8)
    expected_scales = torch.randint(low=0, high=256, size=(E_size, 2 * I_size, H_size // 32), dtype=torch.uint8)
    expected_bias = torch.randn((E_size, 2 * I_size), dtype=torch.bfloat16)
    
    actual_weights_transformed, actual_scales_transformed, actual_bias_transformed = tile_transpose_gate_up_for_mxfp4_kernel(
        expected_weights.clone(),
        expected_scales.clone(),
        expected_bias.clone(),
        tp_degree=8,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size,
        I_size_actual=I_size_actual,
        hidden_act_bias=hidden_act_bias,
    )
    
    actual_weights, actual_scales, actual_bias = undo_tile_transpose_gate_up_for_mxfp4_kernel(
        actual_weights_transformed,
        actual_scales_transformed,
        actual_bias_transformed,
        tp_degree=8,
        quantized_dtype=QuantizedDtype.F4E2M1FN_X4,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size,
        I_size_actual=I_size_actual,
        hidden_act_bias=hidden_act_bias,
    )

    assert torch.equal(expected_weights, actual_weights)
    assert torch.equal(expected_scales, actual_scales)
    torch_neuronx.testing.assert_close(expected_bias, actual_bias, atol=bias_atol, rtol=bias_rtol)


def undo_tile_transpose_down_proj_for_mxfp4_kernel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    tp_degree: int,
    *,
    quantized_dtype: QuantizedDtype,
    E_size: int = 128,
    H_size: int = 3072,
    I_size: int = 3072,
):
    """
    Undo tiling and transpose operations for down projection tensors from MXFP4 kernel format.

    [x2] and [x4] refers to the number of individual elements packed into the fp4 dtype.
    [x2] corresponds to float4_e2m1fn_x2 (uint8).
    [x4] corresponds to float4_e2m1fn_x4 (uint16).

    Reverse transposes applied:
    - Weight: Original shape [E, I // 4, H[x4]]
        - View as [E, shard, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H, Q_PACKED_I]
        - Permute to [E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, H, Q_PACKED_I]
        - View as [E, H, I // 32, 16]
    - Scale: View as [E, shard, NUM_Q_BLOCKS_I_TILE, NUM_I_TILES, H]
             Permute to [E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, H]
             View as [E, H, I // 32]
    - Bias: Unchanged
    
    Args:
        weight (torch.Tensor): Tiled down projection weights
        scale (torch.Tensor): Tiled quantization scales
        bias (torch.Tensor): Bias tensors
        tp_degree (int): Tensor parallel degree for sharding
        quantized_dtype (QuantizedDtype): Quantization data type
        E_size (int): Expert dimension size. Defaults to 128.
        H_size (int): Hidden dimension size. Defaults to 3072.
        I_size (int): Input dimension size. Defaults to 3072.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - weight: [E, H, I // 32, 16] - Original weight format
            - scale: [E, H, I // 32] - Original scale format
            - bias: Unchanged bias tensor
    """
    num_I_tiles, q_blocks_per_I_tile = get_i_tiling_shard_i(tp_degree, I_size)
    q_packed = q_width // quantized_dtype.get_packed_count()

    weight = weight.view([E_size, tp_degree, q_blocks_per_I_tile, q_height, num_I_tiles, H_size, q_packed])
    E, shard, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H, Q_PACKED_I = list(range(weight.ndim))
    weight = weight.permute(E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, H, Q_PACKED_I)
    weight = weight.reshape(E_size, -1, H_size, q_packed).contiguous().view(torch.uint8)
    weight = weight.transpose(1, 2)
    weight = weight.reshape(E_size, H_size, I_size // 32, 16)

    scale = scale.view([E_size, tp_degree, q_blocks_per_I_tile, num_I_tiles, H_size])
    E, shard, NUM_Q_BLOCKS_I_TILE, NUM_I_TILES, H = list(range(scale.ndim))
    scale = scale.permute(E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, H)
    scale = scale.reshape(E_size, I_size // 32, H_size)
    scale = scale.transpose(1, 2)

    return weight, scale, bias


def test_tile_transpose_down_proj_for_mxfp4_kernel():
    E_size = 2
    H_size = 512
    I_size = 3072
    
    expected_weights = torch.randint(low=0, high=256, size=(E_size, H_size, I_size // 32, 16), dtype=torch.uint8)
    expected_scales = torch.randint(low=0, high=256, size=(E_size, H_size, I_size // 32), dtype=torch.uint8)
    expected_bias = torch.randn((E_size, I_size), dtype=torch.bfloat16)
    
    actual_weights, actual_scales, actual_bias = tile_transpose_down_proj_for_mxfp4_kernel(
        convert_down_proj_fp4(expected_weights.clone()),
        expected_scales.clone().transpose(1, 2),
        expected_bias.clone(),
        tp_degree=8,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size,
    )
    
    actual_weights, actual_scales, actual_bias = undo_tile_transpose_down_proj_for_mxfp4_kernel(
        actual_weights,
        actual_scales,
        actual_bias,
        tp_degree=8,
        quantized_dtype=QuantizedDtype.F4E2M1FN_X4,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size,
    )

    assert torch.equal(expected_weights, actual_weights)
    assert torch.equal(expected_scales, actual_scales)
    assert torch.equal(expected_bias, actual_bias)


@pytest.mark.parametrize(
    "B,S,H", [
        pytest.param(1, 10240, 3072),
        pytest.param(1, 1, 3072),
        pytest.param(1, 4, 3072),
        pytest.param(128, 1, 3072),
        pytest.param(128, 4, 3072),
    ]
)
@pytest.mark.parametrize("flatten_BxS", [True, False])
def test_shuffle_hidden_dim(B, S, H, flatten_BxS):

    # Use arange so that indices / values are readable
    hidden_states = torch.arange(B*S*H).to(torch.bfloat16)

    hidden_shape = (B, S, H) if not flatten_BxS else (B * S, H)
    hidden_states = hidden_states.reshape(hidden_shape)

    *dim, _ = hidden_states.shape
    expected = hidden_states.clone().reshape(*dim, H//4, 4).transpose(-2, -1).reshape(*dim, H)

    actual = shuffle_hidden_dim(hidden_states.clone(), dim=-1)

    assert hidden_states.shape == actual.shape, "Tensor shape was updated during shuffling!"
    assert not torch.equal(actual, hidden_states), "Tensor was not shuffled!"
    assert torch.equal(actual, expected), "Shuffled output does not match expected output!"

@pytest.mark.parametrize(
    "shape", [(64, 64, 64, 64)]
)
@pytest.mark.parametrize(
    "dim", [-3, -2, -1, 0, 1, 2, 3]
)
def test_shuffle_hidden_dim_specified(shape, dim):

    # Use arange so that indices / values are readable
    tensor = torch.arange(math.prod(shape)).to(torch.bfloat16).reshape(shape)
    if dim < 0:
        dim += tensor.ndim
    
    pre = shape[:dim]
    H = shape[dim]
    post = shape[dim+1:] if dim < tensor.ndim else []
    expected = tensor.clone().reshape(*pre, H//4, 4, *post).transpose(dim, dim+1).reshape(shape)

    actual = shuffle_hidden_dim(tensor.clone(), dim=dim)

    assert tensor.shape == actual.shape, "Tensor shape was updated during shuffling!"
    assert not torch.equal(actual, tensor), "Tensor was not shuffled!"
    assert torch.equal(actual, expected), "Shuffled output does not match expected output!"


@pytest.mark.parametrize(
    "B,S,H", [
        pytest.param(1, 10240, 3072),
        pytest.param(1, 1, 3072),
        pytest.param(1, 4, 3072),
        pytest.param(128, 1, 3072),
        pytest.param(128, 4, 3072),
    ]
)
@pytest.mark.parametrize("flatten_BxS", [True, False])
def test_unshuffle_hidden_states(B, S, H, flatten_BxS):

    # Use arange so that indices / values are readable
    expected = torch.arange(B*S*H).to(torch.bfloat16)

    hidden_shape = (B, S, H) if not flatten_BxS else (B * S, H)
    expected = expected.reshape(hidden_shape)

    *dim, _  = expected.shape
    shuffled_hidden_states = expected.clone().reshape(*dim, H//4, 4).transpose(-2, -1).reshape(*dim, H)

    actual = unshuffle_hidden_dim(shuffled_hidden_states.clone(), dim=-1)

    assert shuffled_hidden_states.shape == actual.shape, "Tensor shape was updated during unshuffling!"
    assert not torch.equal(actual, shuffled_hidden_states), "Tensor was not unshuffled!"
    assert torch.equal(actual, expected), "Unshuffled output does not match expected output!"


@pytest.mark.parametrize(
    "shape", [(64, 64, 64, 64)]
)
@pytest.mark.parametrize(
    "dim", [-3, -2, -1, 0, 1, 2, 3]
)
def test_unshuffle_hidden_dim_specified(shape, dim):

    # Use arange so that indices / values are readable
    tensor = torch.arange(math.prod(shape)).to(torch.bfloat16).reshape(shape)
    if dim < 0:
        dim += tensor.ndim
    
    pre = shape[:dim]
    H = shape[dim]
    post = shape[dim+1:] if dim < tensor.ndim else []
    expected = tensor.clone().reshape(*pre, 4, H//4, *post).transpose(dim, dim+1).reshape(shape)

    actual = unshuffle_hidden_dim(tensor.clone(), dim=dim)

    assert tensor.shape == actual.shape, "Tensor shape was updated during shuffling!"
    assert not torch.equal(actual, tensor), "Tensor was not shuffled!"
    assert torch.equal(actual, expected), "Shuffled output does not match expected output!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])