import math
from typing import Optional, Tuple

import torch

from neuronx_distributed_inference.models.config import InferenceConfig
import neuronxcc.nki.language as nl

from neuronx_distributed.quantization.microscaling.transform_weights import pack_fp4_x4_uint16
from neuronx_distributed.quantization.quantization_config import QuantizedDtype


PMAX = nl.tile_size.pmax
Q_WIDTH = 4
Q_HEIGHT = 8


def convert_gate_up_proj(
    tensor: torch.Tensor,
    *,
    is_bias: bool = False,
    concat_gate_up_proj: bool = True
) -> torch.Tensor:
    """
    Returns separated gate and up weight from reference format where it is interleaved.
    Reference format: E, 2xI, H with interleaved gate and up projection.

    Each of gate and up: [E, H, I] if not bias, [E, I] if bias

    Note that 2I is also interleaved in the original checkpoint. Even elements along 2I belong to gate, and
    odd elements belong to up_proj. This transform also handles chunking it, so it becomes all gate elements followed
    by all up_proj elements along I.

    Args:
        tensor (torch.Tensor): the parameter to convert
        is_bias (bool): flag indicating if parameter is bias
        concat_gate_up_proj (bool): flag to concatenate gate and up_proj along I dim, for backwards compatibility with existing behavior

    Returns:
        torch.Tensor: in format needed for NxDI MoE modules
    """
    gate, up_proj = tensor[:, ::2, ...], tensor[:, 1::2, ...]
    if not is_bias:
        gate = gate.transpose(1, 2)
        up_proj = up_proj.transpose(1, 2)
    if concat_gate_up_proj:
        # for backwards compatibility with existing checkpoint code
        return torch.cat((gate, up_proj), dim=-1)
    else:
        return gate, up_proj


def convert_gate_up_proj_fp4(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Returns separated gate and up weight from reference format where it is interleaved.

    [x2] and [x4] refers to the number of individual elements packed into the fp4 dtype
    [x2] corresponds to float4_e2m1fn_x2 (uint8)
    [x4] corresponds to float4_e2m1fn_x4 (uint16)

    Args:
        tensor (torch.Tensor): the parameter to convert [E, 2I, H // 32, 16[x2]]

    Note that 2I is also interleaved in the original checkpoint. Even elements along 2I belong to gate, and
    odd elements belong to up_proj. This transform also handles chunking it, so it becomes all gate elements followed
    by all up_proj elements along I.

    Returns:
        gate: [E, H // 4, I * 1H[x4]]
        up: [E, H // 4, I * 1H[x4]]
    """

    assert tensor.ndim >= 3, "Tensor must have at least 3 dimensions corresponding to E, I, H dims."
    E_size, _2I, *_ = tensor.shape
    assert _2I % 2 == 0, "Interleaved 2I dim must be even in size to split evenly."
    I_size = _2I // 2

    # the packed dim of size 2[x2] will not be transposed
    tensor = tensor.reshape([E_size, _2I, -1, 2])  # [E, 2I, H // 4, 2 [x2]]

    gate, up_proj = tensor[:, ::2, ...], tensor[:, 1::2, ...]  # separated into [E, I, H // 4, 2 [x2]] each; I dim deinterleaved

    gate = gate.transpose(1, 2)
    up_proj = up_proj.transpose(1, 2)

    gate = pack_fp4_x4_uint16(gate.contiguous())  # [E, H // 4, I, 1[x4]] Now it is uint16 or fp4x4
    gate = gate.reshape(E_size, -1, I_size)  # [E, H // 4, I * 1H[x4]]

    up_proj = pack_fp4_x4_uint16(up_proj.contiguous())  # [E, H // 4, I, 1[x4]] Now it is uint16 or fp4x4
    up_proj = up_proj.reshape(E_size, -1, I_size)  # [E, H // 4, I * 1H[x4]]

    return gate, up_proj


def get_h_tiling_shard_i(H_size: int):
    """
    Calculate the number of H tiles and blocks per H tile for hidden dimension tiling.

    Computes tiling parameters for the H (hidden) dimension based on hardware constraints.
    Uses fixed tile size of 512 elements and quantization block dimensions.

    Args:
        H_size (int): Total size of the H (hidden) dimension

    Returns:
        tuple: A tuple containing:
            - num_H_tiles (int): Number of tiles in the H dimension (H_size // (pmax * q_width))
            - q_blocks_per_H_tile (int): Number of quantized blocks per H tile (512 // (q_width * q_height))
    """
    num_H_tiles = H_size // (PMAX * Q_WIDTH)
    q_blocks_per_H_tile = 512 // (Q_WIDTH * Q_HEIGHT)
    return num_H_tiles, q_blocks_per_H_tile


def get_i_tiling_shard_i(tp_degree: int, I_size: int):
    """
    Calculate the number of I tiles and blocks per I tile for tensor parallel sharding.

    This function determines the tiling parameters for the I (input) dimension when sharding
    across tensor parallel ranks. It handles two cases:
    1. When per-rank I size is > 512: Tiles of size 512 are used
    2. When per-rank I size is <= 512: A single tile is used

    Args:
        tp_degree (int): Tensor parallel degree - number of ranks to shard across
        I_size (int): Total size of the I dimension before sharding

    Returns:
        tuple: A tuple containing:
            - num_I_tiles (int): Number of tiles in the I dimension
            - q_blocks_per_I_tile (int): Number of quantized blocks per I tile

    Raises:
        AssertionError: If per-rank I size is not divisible by required block size
                       or if padding would be needed (not implemented)

    Examples:
        >>> get_i_tiling_shard_i(tp_degree=4, I_size=1536)  # per_rank_I_size=384
        (1, 12)  # Single tile with 12 blocks
        >>> get_i_tiling_shard_i(tp_degree=2, I_size=1024)  # per_rank_I_size=512
        (1, 16)  # Single tile with 16 blocks
    """
    mx_block_size = (Q_WIDTH * Q_HEIGHT)
    per_rank_I_size = I_size // tp_degree
    if per_rank_I_size > 512:
        assert per_rank_I_size % 512 == 0, f"Unsupported I_size {I_size} for tp degree {tp_degree}. Per rank I_size {per_rank_I_size} must be divisible by 512"
        num_I_tiles = int(math.ceil(per_rank_I_size / 512.0))
        q_blocks_per_I_tile, q_blocks_per_I_tile_padding = divmod(512, Q_WIDTH * Q_HEIGHT)
        assert q_blocks_per_I_tile_padding == 0, "Padding is required for q_blocks_per_I_tile dim, not implemented yet"
    else:
        assert per_rank_I_size % mx_block_size == 0, f"Unsupported I_size {I_size} for tp degree {tp_degree}. Per rank I_size {per_rank_I_size} must be divisible by {mx_block_size}"
        num_I_tiles = 1
        q_blocks_per_I_tile, q_blocks_per_I_tile_padding = divmod(per_rank_I_size, Q_WIDTH * Q_HEIGHT)
        assert q_blocks_per_I_tile_padding == 0, "Padding is required for q_blocks_per_I_tile dim, not implemented yet"
    return num_I_tiles, q_blocks_per_I_tile


def get_tiling_values(*, tp_degree, E_size, H_size, I_size):
    num_H_tiles, q_blocks_per_H_tile = get_h_tiling_shard_i(H_size)
    num_I_tiles, q_blocks_per_I_tile = get_i_tiling_shard_i(tp_degree, I_size)
    q_packed_H = Q_WIDTH // QuantizedDtype.F4E2M1FN_X4.get_packed_count()
    return num_H_tiles, q_blocks_per_H_tile, num_I_tiles, q_blocks_per_I_tile, q_packed_H


def _tile_transpose_concat_gate_up_weight(
    w_gate, w_up, *, tp_degree, E_size, H_size, I_size
):
    num_H_tiles, q_blocks_per_H_tile, num_I_tiles, q_blocks_per_I_tile, q_packed_H = get_tiling_values(
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )
    assert q_blocks_per_H_tile * Q_HEIGHT == 128
    _128_H = q_blocks_per_H_tile * Q_HEIGHT
    w_gate = w_gate.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            Q_HEIGHT,
            1,
            tp_degree,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
            q_packed_H,
        ]
    )
    w_up = w_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            Q_HEIGHT,
            1,
            tp_degree,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
            q_packed_H,
        ]
    )

    # Permute weight
    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        shard,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
        Q_PACKED_H,
    ) = list(range(w_gate.ndim))
    w_gate = w_gate.permute(
        E,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        N_H_TILES,
        shard,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_PACKED_H,
    )
    w_up = w_up.permute(
        E,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        N_H_TILES,
        shard,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_PACKED_H,
    )

    # Concate gate+up weight
    # gate_up_concat is now on axis 3
    weight = torch.cat((w_gate, w_up), dim=3)
    weight = weight.reshape(E_size, _128_H, 2, num_H_tiles, -1)  # [E, 128_H, 2, H // 512, TP * I[x4H]]
    return weight


def _tile_transpose_concat_gate_up_scale(
    s_gate, s_up, *, tp_degree, E_size, H_size, I_size
):
    num_H_tiles, q_blocks_per_H_tile, num_I_tiles, q_blocks_per_I_tile, _ = get_tiling_values(
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )
    s_gate = s_gate.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            1,
            tp_degree,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )
    s_up = s_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            1,
            tp_degree,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )

    # Permute scale
    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        shard,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
    ) = list(range(s_gate.ndim))
    s_gate = s_gate.permute(
        E,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        N_H_TILES,
        shard,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
    )
    s_up = s_up.permute(
        E,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        N_H_TILES,
        shard,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
    )

    # Concatenate gate+up scale
    scale = torch.cat((s_gate, s_up), dim=2)
    scale = scale.reshape(E_size, q_blocks_per_H_tile, 2, num_H_tiles, -1)  # [E, 16_H, 2, H // 512, TP * I]
    return scale


def _tile_transpose_concat_gate_up_bias(
    b_gate, b_up, *, tp_degree, E_size, H_size, I_size
):
    _, _, num_I_tiles, q_blocks_per_I_tile, _ = get_tiling_values(
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )
    b_gate = b_gate.reshape(
        [E_size, 1, tp_degree, num_I_tiles, q_blocks_per_I_tile, Q_HEIGHT, Q_WIDTH]
    )
    b_up = b_up.reshape([E_size, 1, tp_degree, num_I_tiles, q_blocks_per_I_tile, Q_HEIGHT, Q_WIDTH])

    # Permute bias
    E, gate_up_concat, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I = list(
        range(b_gate.ndim)
    )
    b_gate = b_gate.permute(
        E, shard, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    )
    b_up = b_up.permute(
        E, shard, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    )

    # Concatenate gate+up bias
    bias = torch.cat((b_gate, b_up), dim=4)
    bias = bias.reshape(E_size, -1)  # [E, 2I]
    return bias


def tile_transpose_gate_up_for_mxfp4_kernel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    tp_degree: int,
    E_size: int = 128,
    H_size: int = 3072,
    I_size: int = 3072,
    I_size_actual: Optional[int] = None,
    hidden_act_bias: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tile and transpose gate/up projection tensors for MXFP4 quantized MoE kernel.

    [x2] and [x4] refers to the number of individual elements packed into the fp4 dtype
    [x2] corresponds to float4_e2m1fn_x2 (uint8)
    [x4] corresponds to float4_e2m1fn_x4 (uint16)

    Transposes applied:
    - Weight: Original shape [E, 2I, H // 32, 16[x2]]
        - Transpose, deinterleave gate and up, and pack each of gate and up into [E, H, I // 4 * 1H[x4]]
        - View as [E, N_H_TILES, Q_BLOCK_H_TILE, Q_HEIGHT_H, gate_up_concat, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I, Q_PACKED_H]
        - Permute to [E, Q_BLOCK_H_TILE, Q_HEIGHT_H, gate_up_concat, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_PACKED_H]
        - Concat gate and up along gate_up_concat dim
        - View as [E, 128_H, 2, H // 512, TP * I[x4H]] (2 because of gate and up concatenation)
    - Scale: Original shape [E, 2I, H // 32]
        - Transpose, deinterleave gate and up, and pack each of gate and up into [E, H, I // 4 * 1H[x4]]
        - View as [E, N_H_TILES, Q_BLOCK_H_TILE, gate_up_concat, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I]
        - Permute to [E, Q_BLOCK_H_TILE, gate_up_concat, N_H_TILES, shard, NUM_I_TILES, Q_WIDTH_I, Q_BLOCK_I_TILE, Q_HEIGHT_I]
        - Concat gate and up along gate_up_concat dim
        - View as [E, 128_H, 2, H // 512, TP * I] (2 because of gate and up concatenation)
    - Bias: Original shape [E, 2I]
        - View as [E, gate_up_concat, shard, NUM_I_TILES, Q_BLOCK_I_TILE, Q_HEIGHT_I, Q_WIDTH_I]
        - Permute to [E, shard, Q_BLOCK_I_TILE, gate_up_concat, Q_HEIGHT_I, NUM_I_TILES, Q_WIDTH_I]
        - Concat gate and up along gate_up_concat dim
        - View as [E, 2I]  (2I because of gate and up concatenation)

    More details in doc `YFIQAmI1p2nr`

    Note that 2I is also interleaved in the original checkpoint. Even elements along 2I belong to gate, and
    odd elements belong to up_proj. This transform also handles chunking it, so it becomes all gate elements followed
    by all up_proj elements along I.

    Concatenation for gate_up can also happen between any two dims, which is configurable by gate_up_I_concat_dim and gate_up_I_concat_dim_bias.

    Args:
        weight (torch.Tensor): Interleaved gate/up projection weights
        scale (torch.Tensor): Quantization scales for weights
        bias (torch.Tensor): Bias tensors
        tp_degree (int): Tensor parallel degree for sharding
        E_size (int): Expert dimension size. Defaults to 128.
        H_size (int): Hidden dimension size. Defaults to 3072.
        I_size (int): Input dimension size. Defaults to 3072.
        I_size_actual: (int): Unpadded input dimension size, used to avoid adding hidden_act_bias to padded regions of up_proj_bias. Defaults to None.
        hidden_act_bias (float): Hidden activation bias, which will be added to up_proj_bias. Defaults to 0.0.


    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - weight: [E, H // 4, 2I[x4]] - Tiled and transposed weight tensor
            - scale: [E, H // 32, 2I] - Tiled and transposed scale tensor
            - bias: [E, 2I] - Tiled and transposed bias tensor
    """

    # Reshape weight
    w_gate, w_up = convert_gate_up_proj_fp4(
        weight
    )  # [E, H // 4, I * 1H[x4]] for each of gate and up
    weight = _tile_transpose_concat_gate_up_weight(
        w_gate,
        w_up,
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )

    # Reshape scale
    s_gate, s_up = convert_gate_up_proj(
        scale, is_bias=False, concat_gate_up_proj=False
    )  # [E, H // 32, I] for each of gate and up
    scale = _tile_transpose_concat_gate_up_scale(
        s_gate,
        s_up,
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )

    # Reshape bias
    b_gate, b_up = convert_gate_up_proj(
        bias, is_bias=True, concat_gate_up_proj=False
    )  # [E, I] for each of gate and up
    # Increment b_up by hidden_act_bias
    # NOTE: Addition is performed in fp32 to avoid accumulating additional logit error
    if hidden_act_bias != 0.0:
        # Do not increment padded region of I dim if I dim is padded
        b_I_upper_limit = I_size_actual if I_size_actual else I_size
        b_up[:, :b_I_upper_limit].add_(torch.tensor([hidden_act_bias], dtype=torch.float32))

    bias = _tile_transpose_concat_gate_up_bias(
        b_gate,
        b_up,
        tp_degree=tp_degree,
        E_size=E_size,
        H_size=H_size,
        I_size=I_size
    )
    return weight, scale, bias


def convert_down_proj_fp4(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Convert the down_proj tensor from GptOss reference format to NxDI format.

    [x2] and [x4] refers to the number of individual elements packed into the fp4 dtype
    [x2] corresponds to float4_e2m1fn_x2 (uint8)
    [x4] corresponds to float4_e2m1fn_x4 (uint16)

    Args:
        tensor (torch.Tensor): down_proj tensor in shape [E, H, I // 32, 16[x2]]

    Returns:
        torch.Tensor: [E, I // 4, H[x4]]
    """
    E_size, H_size, *_ = tensor.shape
    tensor = tensor.reshape([E_size, H_size, -1, 2])  # [E, H, I // 4, 2[x2]]
    tensor = tensor.transpose(1, 2)  # [E, I // 4, H, 2[x2]]
    tensor = tensor.contiguous()
    tensor = pack_fp4_x4_uint16(tensor)  # [E, I // 4, H, 1[x4]]
    tensor = tensor.reshape(E_size, -1, H_size)  # [E, I // 4, H[x4]]
    return tensor


def tile_transpose_down_proj_for_mxfp4_kernel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    tp_degree: int,
    E_size: int = 128,
    H_size: int = 3072,
    I_size: int = 3072,
    shuffle_output_hidden=False,
):
    """
    Tile and transpose down projection tensors for MXFP4 quantized MoE kernel.

    [x2] and [x4] refers to the number of individual elements packed into the fp4 dtype.
    [x2] corresponds to float4_e2m1fn_x2 (uint8).
    [x4] corresponds to float4_e2m1fn_x4 (uint16).

    Transposes applied:
    - Weight: Input shape [E, I // 4, H * 1I[x4]] (already transposed from HF checkpoint)
        - View as [E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, H, Q_PACKED_I]
        - Permute to [E, shard, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H, Q_PACKED_I]
        - View as [E, 128_I*, I // 512, H[x4I]]
    - Scale: Input shape [E, I // 32, H] (already transposed from HF checkpoint)
        - View as [E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, H]
        - Permute to [E, shard, NUM_Q_BLOCKS_I_TILE, NUM_I_TILES, H]
        - View as [E, 16_I*, I // 512, TP*H]
    - Bias: Input shape [E, H] unchanged

    When `shuffle_output_hidden` is enabled, H dim for each tensor has a transpose applied:
    [..., H, ...] -> [..., H//4, 4, ...] -> [..., 4, H//4, ...] -> [..., H, ...]
    This allows output H dim to be consistent with full model H dim shuffling

    More details in doc `YFIQAmI1p2nr`

    Args:
        weight (torch.Tensor): Down projection weights
        scale (torch.Tensor): Quantization scales for weights
        bias (torch.Tensor): Bias tensors
        tp_degree (int): Tensor parallel degree for sharding
        quantized_dtype (QuantizedDtype): Quantization data type
        E_size (int): Expert dimension size. Defaults to 128.
        H_size (int): Hidden dimension size. Defaults to 3072.
        I_size (int): Input dimension size. Defaults to 3072.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - weight: [E, I // 4, H[x4]] - Tiled and transposed weight tensor
            - scale: [E, I // 32, H] - Tiled and transposed scale tensor
            - bias: Unchanged bias tensor
    """
    num_I_tiles, q_blocks_per_I_tile = get_i_tiling_shard_i(tp_degree, I_size)
    q_packed = Q_WIDTH // QuantizedDtype.F4E2M1FN_X4.get_packed_count()

    if shuffle_output_hidden:
        # full model h dims are being shuffled, so h dim is also tiled and shuffled for down proj output
        weight = weight.reshape(
            [E_size, tp_degree, num_I_tiles, q_blocks_per_I_tile, Q_HEIGHT, H_size // 4, 4, q_packed]
        )
        E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_H_BLOCKS, H_BLOCK, Q_PACKED_I = list(range(weight.ndim))
        # H dim is permuted here: (NUM_H_BLOCKS, H_BLOCK) -> (H_BLOCK, NUM_H_BLOCKS)
        weight = weight.permute(E, shard, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS, Q_PACKED_I)
        weight = weight.reshape(E_size, tp_degree * q_blocks_per_I_tile * Q_HEIGHT, num_I_tiles, H_size * q_packed)  # [E, 128_I*, I // 512, H*[x4I]]

        scale = scale.reshape([E_size, tp_degree, num_I_tiles, q_blocks_per_I_tile, H_size // 4, 4])
        E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, NUM_H_BLOCKS, H_BLOCK = list(range(scale.ndim))
        scale = scale.permute(E, shard, NUM_Q_BLOCKS_I_TILE, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS)
        scale = scale.reshape(E_size, tp_degree * q_blocks_per_I_tile, num_I_tiles, H_size)  # [E, 16_I*, I // 512, H]

        bias = shuffle_hidden_dim(bias, dim=-1)
    else:
        # only shuffle I dims for down proj
        # Bias unchanged from E, H
        weight = weight.reshape(
            [E_size, tp_degree, num_I_tiles, q_blocks_per_I_tile, Q_HEIGHT, H_size, q_packed]
        )
        E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, H, Q_PACKED_I = list(range(weight.ndim))
        weight = weight.permute(E, shard, NUM_Q_BLOCKS_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H, Q_PACKED_I)
        weight = weight.reshape(E_size, tp_degree * q_blocks_per_I_tile * Q_HEIGHT, num_I_tiles, H_size * q_packed)  # [E, 128_I*, I // 512, H[x4I]]

        scale = scale.reshape([E_size, tp_degree, num_I_tiles, q_blocks_per_I_tile, H_size])
        E, shard, NUM_I_TILES, NUM_Q_BLOCKS_I_TILE, H = list(range(scale.ndim))
        scale = scale.permute(E, shard, NUM_Q_BLOCKS_I_TILE, NUM_I_TILES, H)
        scale = scale.reshape(E_size, tp_degree * q_blocks_per_I_tile, num_I_tiles, H_size)  # [E, 16_I*, I // 512, H]

    return weight, scale, bias


def shuffle_hidden_dim(tensor: torch.Tensor, *, dim=-1):
    """
    Shuffles the H dim of hidden states, preparing for residual add with shuffled attention block output.

    Transposes applied:
    - Hidden states: Original shape [*pre_shape, H, *post_shape], dim=index of dim H
        - View as [*pre_shape, H/4, 4, *post_shape]
        - Permute to [*pre_shape, 4, H/4, *post_shape]
        - View as [*pre_shape, H, *post_shape]

    Args:
        tensor (torch.Tensor)
        dim (int): dimension of hidden axis. Default: -1 (last dim of tensor)
            Returns:
        torch.Tensor: - Shuffled hidden states tensor
    """
    if dim < 0:
        dim = dim + tensor.ndim
    pre_shape, H = tensor.shape[:dim], tensor.shape[dim]
    if dim < tensor.ndim - 1:
        post_shape = tensor.shape[dim + 1:]
    else:
        post_shape = []
    assert H % 4 == 0, f"H dim (dim={dim=}) in shape {tensor.shape=} must be divisible by 4 to shuffle H dim, got H={H=}"
    tensor = tensor\
        .view(*pre_shape, H // 4, 4, *post_shape)\
        .transpose(dim, dim + 1)\
        .reshape(*pre_shape, H, *post_shape)
    return tensor


def unshuffle_hidden_dim(tensor: torch.Tensor, *, dim=-1):
    """
    Unshuffles the H dim of a tensor, preparing for residual add with unshuffled MoE block output.

    Transposes applied:
    - Hidden states: Original shape [*pre_shape, H, *post_shape], dim=index of dim H
        - View as [*pre_shape, 4, H/4, *post_shape]
        - Permute to [*pre_shape, H/4, 4, *post_shape]
        - View as [*pre_shape, H, *post_shape]

    Args:
        tensor (torch.Tensor)
        dim (int): dimension of hidden axis. Default: -1 (last dim of tensor)
            Returns:
        torch.Tensor: - Shuffled hidden states tensor
    """
    if dim < 0:
        dim = dim + tensor.ndim
    pre_shape, H = tensor.shape[:dim], tensor.shape[dim]
    if dim < tensor.ndim - 1:
        post_shape = tensor.shape[dim + 1:]
    else:
        post_shape = []
    assert H % 4 == 0, f"H dim (dim={dim=}) in shape {tensor.shape=} must be divisible by 4 to shuffle H dim, got H={H=}"
    tensor = tensor\
        .view(*pre_shape, 4, H // 4, *post_shape)\
        .transpose(dim, dim + 1)\
        .reshape(*pre_shape, H, *post_shape)
    return tensor


def convert_hf_format_state_dict_mxfp4_compute(state_dict: dict, config: InferenceConfig) -> dict:
    """
    Convert HuggingFace format state dict to Neuron format.

    Weights that ALWAYS have H dim shuffled:
    - model.layers.*.self_attn.o_proj.weight [H, D]
    - model.layers.*.self_attn.o_proj.bias [H]
    - model.layers.*.post_attention_layernorm.weight [H]
    - model.layers.*.mlp.router.weight [E, H]

    Weights that ALWAYS have I dim shuffled:
    - model.layers.*.mlp.experts.gate_up_proj_bias [E, 2I] -> [E, 2I]
    - model.layers.*.mlp.experts.down_proj_blocks [E, H, I/32, 16] -> [E, 128_I*, I // 512, H[x4I]]
    - model.layers.*.mlp.experts.down_proj_scales [E, H, I/32] -> [E, 16_I*, I // 512, H]

    Weights that ALWAYS have H and I dims shuffled
    - model.layers.*.mlp.experts.gate_up_proj_blocks [E, 2I, H/32, 16] -> [E, 128_H, 2, H // 512, TP * I[x4H]]
    - model.layers.*.mlp.experts.gate_up_proj_scales [E, 2I, H/32]-> [E, 16_H, 2, H // 512, TP * I]

    Additional weights that have H dim shuffled when neuron_config.is_full_model_shuffled:
    - model.embed_tokens.weight [V, H]
    - model.layers.*.input_layernorm.weight [H] -> [H//4, 4]
    - model.layers.*.self_attn.q_proj.weight [D * n_heads, H]
    - model.layers.*.self_attn.k_proj.weight [D * n_kv_heads, H]
    - model.layers.*.self_attn.v_proj.weight [D * n_kv_heads, H]
    - model.norm.weight [H]
    - lm_head.weight [V, H]

    Additional weights that have H and I dims shuffled when neuron_config.is_full_model_shuffled:
    - model.layers.*.mlp.experts.down_proj_blocks [E, H, I/32, 16] -> [E, 128_I*, I // 512, H[x4I]]
    - model.layers.*.mlp.experts.down_proj_scales [E, H, I/32] -> [E, 16_I*, I // 512, H]
    - model.layers.*.mlp.experts.down_proj_bias [E, H]
    """
    neuron_config = config.neuron_config
    assert neuron_config.is_mxfp4_compute, "mxfp4 compute must be enabled for convert_hf_format_state_dict_mxfp4_compute to be applicable"
    num_layers = config.num_hidden_layers

    is_full_model_shuffled = neuron_config.is_full_model_shuffled
    if is_full_model_shuffled:
        state_dict["lm_head.weight"] = shuffle_hidden_dim(state_dict["lm_head.weight"], dim=-1)  # [V, H]
        state_dict["embed_tokens.weight"] = shuffle_hidden_dim(state_dict["embed_tokens.weight"], dim=-1)  # [V, H]
        state_dict["norm.weight"] = shuffle_hidden_dim(state_dict["norm.weight"], dim=-1)  # [H]

    for layer in range(num_layers):
        if is_full_model_shuffled:
            state_dict[f"layers.{layer}.self_attn.q_proj.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.self_attn.q_proj.weight"], dim=-1)
            state_dict[f"layers.{layer}.self_attn.k_proj.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.self_attn.k_proj.weight"], dim=-1)
            state_dict[f"layers.{layer}.self_attn.v_proj.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.self_attn.v_proj.weight"], dim=-1)

        # Output Projection shuffled in order to get hidden states shuffled prior to MoE layer
        state_dict[f"layers.{layer}.self_attn.o_proj.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.self_attn.o_proj.weight"], dim=0)
        state_dict[f"layers.{layer}.self_attn.o_proj.bias"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.self_attn.o_proj.bias"], dim=-1)

        # Layer Norms
        # Add unpadded weight for input layer norm
        if is_full_model_shuffled:
            state_dict[f"layers.{layer}.input_layernorm.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.input_layernorm.weight"], dim=-1)
        else:
            state_dict[f"layers.{layer}.input_layernorm.weight_unpadded"] = state_dict[f"layers.{layer}.input_layernorm.weight"][:config.original_hidden_size].clone().contiguous()

        # Shuffle post-attention layernorm
        state_dict[f"layers.{layer}.post_attention_layernorm.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.post_attention_layernorm.weight"], dim=-1)

        # Sinks
        state_dict[f"layers.{layer}.self_attn.learned_sinks.sink"] = state_dict[f"layers.{layer}.self_attn.sinks"]
        # If attention is run in two different parallelisms across CTE and TKG, we duplicate the weights
        if config.neuron_config.attention_dp_degree != config.neuron_config.cp_degree:
            state_dict[f"layers.{layer}.self_attn.tkg_learned_sinks.sink"] = state_dict[f"layers.{layer}.self_attn.sinks"]
        del state_dict[f"layers.{layer}.self_attn.sinks"]

        # Router shuffled due to o_proj being shuffled
        state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.weight"] = shuffle_hidden_dim(state_dict[f"layers.{layer}.mlp.router.weight"], dim=-1)
        del state_dict[f"layers.{layer}.mlp.router.weight"]

        state_dict[f"layers.{layer}.feed_forward.moe.router.linear_router.bias"] = state_dict[f"layers.{layer}.mlp.router.bias"]
        del state_dict[f"layers.{layer}.mlp.router.bias"]

        # MoE
        for proj in ["down_proj", "gate_up_proj"]:
            weight = state_dict[f"layers.{layer}.mlp.experts.{proj}_blocks"]
            scale = state_dict[f"layers.{layer}.mlp.experts.{proj}_scales"]
            bias = state_dict[f"layers.{layer}.mlp.experts.{proj}_bias"]

            if proj == "gate_up_proj":
                weight, scale, bias = tile_transpose_gate_up_for_mxfp4_kernel(
                    weight,
                    scale,
                    bias,
                    tp_degree=config.neuron_config.tp_degree,
                    E_size=config.num_local_experts,
                    H_size=config.hidden_size,
                    I_size=config.intermediate_size,
                    I_size_actual=getattr(config, "original_intermediate_size", config.intermediate_size),
                    hidden_act_bias=config.neuron_config.hidden_act_bias,
                )
            else:
                weight, scale, bias = tile_transpose_down_proj_for_mxfp4_kernel(
                    convert_down_proj_fp4(weight),
                    scale.transpose(1, 2),
                    bias,
                    tp_degree=config.neuron_config.tp_degree,
                    E_size=config.num_local_experts,
                    I_size=config.intermediate_size,
                    H_size=config.hidden_size,
                    shuffle_output_hidden=is_full_model_shuffled,
                )

            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.weight"] = weight
            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.scale"] = scale
            state_dict[f"layers.{layer}.feed_forward.moe.expert_mlps.mlp_op.{proj}.bias"] = bias
            del state_dict[f"layers.{layer}.mlp.experts.{proj}_blocks"]
            del state_dict[f"layers.{layer}.mlp.experts.{proj}_scales"]
            del state_dict[f"layers.{layer}.mlp.experts.{proj}_bias"]
    return state_dict
