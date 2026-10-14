# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone depthwise Conv1D NKI kernel for ZAYA CCA attention.

This is a self-contained copy of nkilib's depthwise_conv1d_implicit_gemm
kernel with helper functions inlined to avoid nkilib package import conflicts
(the system nkilib version may differ from our fork, and putting our fork
on sys.path shadows the system package used by NxDI).

Source: nkilib/experimental/conv/depthwise_conv1d.py (SDK 2.29 / NKI 0.3.0)
"""

import nki
import nki.isa as nisa
import nki.language as nl


# Inlined helpers (from nkilib.core.utils)
def _kernel_assert(condition: bool, error_text: str):
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text} - Please check the validation message and adjust kernel inputs accordingly"
    )  # noqa: S101, E501


def _div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit
def depthwise_conv1d_implicit_gemm(
    img_ref: nl.ndarray,
    filter_ref: nl.ndarray,
    padding: tuple = ((0, 0), (0, 0)),
    stride: tuple = (1, 1),
    rhs_dilation: tuple = (1, 1),
    lhs_dilation: tuple = (1, 1),
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    in_perm: tuple = None,
    kern_perm: tuple = None,
    out_perm: tuple = None,
) -> nl.ndarray:
    """
    Depthwise Conv1D using implicit GEMM without full im2col materialization.

    Args:
        img_ref (nl.ndarray): [N, C, 1, W], Input tensor on HBM
        filter_ref (nl.ndarray): [C, 1, 1, S], Depthwise kernel weights on HBM
        padding (tuple): ((H_pad_l, H_pad_r), (W_pad_l, W_pad_r))
        stride (tuple): (stride_h, stride_w)
        feature_group_count (int): Must equal C for depthwise conv

    Returns:
        output (nl.ndarray): [N, C, 1, Q] where Q = (W+pad-S)//stride+1
    """
    W_padding_l, W_padding_r = padding[1]
    stride_h, stride_w = stride
    _kernel_assert(
        stride_h == 1, f"Only stride_h=1 is supported, got stride_h={stride_h}"
    )
    _kernel_assert(
        lhs_dilation[0] == 1 and lhs_dilation[1] == 1,
        f"Only lhs_dilation=(1,1) is supported, got {lhs_dilation}",
    )
    _kernel_assert(
        rhs_dilation[0] == 1 and rhs_dilation[1] == 1,
        f"Only rhs_dilation=(1,1) is supported, got {rhs_dilation}",
    )
    _kernel_assert(
        batch_group_count == 1,
        f"Only batch_group_count=1 is supported, got {batch_group_count}",
    )

    N = img_ref.shape[0]
    C = img_ref.shape[1]
    W = img_ref.shape[3]
    S = filter_ref.shape[3]
    W_padded = W + W_padding_l + W_padding_r
    Q = (W_padded - S) // stride_w + 1

    _kernel_assert(
        feature_group_count == C,
        f"Only depthwise convolution is supported (feature_group_count must equal C={C}), got {feature_group_count}",
    )

    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    S_TILE = min(S, P_MAX)
    Q_TILE = min(Q, F_MAX)
    NUM_SHARDS = nl.num_programs()
    C_per_shard = C // NUM_SHARDS
    C_TILE = min(C_per_shard, P_MAX)
    num_s_tiles = _div_ceil(S, S_TILE)
    num_q_tiles = _div_ceil(Q, Q_TILE)
    num_c_tiles = _div_ceil(C_per_shard, C_TILE)

    shard_id = nl.program_id(axis=0)

    output = nl.ndarray((N, C, 1, Q), dtype=img_ref.dtype, buffer=nl.shared_hbm)

    for batch_idx in nl.affine_range(N):
        for c_tile_idx in nl.affine_range(num_c_tiles):
            c_tile_start = c_tile_idx * C_TILE
            c_tile_size = min(C_TILE, C_per_shard - c_tile_start)
            c_global_start = shard_id * C_per_shard + c_tile_start

            # Preload and transpose filter tiles
            filter_all_tiles = []
            for s_tile_idx in nl.affine_range(num_s_tiles):
                s_start = s_tile_idx * S_TILE
                s_tile_size = min(S_TILE, S - s_start)

                filter_tmp = nl.ndarray(
                    (C_TILE, S_TILE), dtype=filter_ref.dtype, buffer=nl.sbuf
                )
                if c_tile_size < C_TILE or s_tile_size < S_TILE:
                    nisa.memset(filter_tmp, 0)
                nisa.dma_copy(
                    dst=filter_tmp[0:c_tile_size, 0:s_tile_size],
                    src=filter_ref.ap(
                        pattern=[[S, c_tile_size], [1, s_tile_size]],
                        offset=s_start + c_global_start * S,
                    ),
                )

                filter_psum = nl.ndarray(
                    (S_TILE, C_TILE), dtype=filter_ref.dtype, buffer=nl.psum
                )
                nisa.nc_transpose(
                    dst=filter_psum[0:s_tile_size, 0:c_tile_size],
                    data=filter_tmp[0:c_tile_size, 0:s_tile_size],
                )

                filter_tile = nl.ndarray(
                    (S_TILE, C_TILE), dtype=filter_ref.dtype, buffer=nl.sbuf
                )
                if c_tile_size < C_TILE or s_tile_size < S_TILE:
                    nisa.memset(filter_tile, 0)
                nisa.tensor_copy(
                    dst=filter_tile[0:s_tile_size, 0:c_tile_size],
                    src=filter_psum[0:s_tile_size, 0:c_tile_size],
                )
                filter_all_tiles.append(filter_tile)

            for channel_idx in nl.affine_range(c_tile_size):
                c_global = c_global_start + channel_idx
                input_base_offset = batch_idx * C * W + c_global * W

                input_tiles = []

                first_unaffected_tile = (
                    _div_ceil(W_padding_l, S_TILE) if W_padding_l > 0 else 0
                )
                last_unaffected_tile = (
                    (W_padding_l + W - Q * stride_w) // S_TILE
                    if W_padding_r > 0
                    else num_s_tiles
                )

                for s_tile_idx in nl.affine_range(num_s_tiles):
                    s_start = s_tile_idx * S_TILE
                    s_tile_size = min(S_TILE, S - s_start)

                    input_tile = nl.ndarray(
                        (S_TILE, Q), dtype=img_ref.dtype, buffer=nl.sbuf
                    )

                    if (
                        s_tile_idx < first_unaffected_tile
                        or s_tile_idx >= last_unaffected_tile
                    ):
                        nisa.memset(input_tile, 0)

                        q_first_full = None
                        q_last_full = None

                        for q_idx in range(Q):
                            q_pos = q_idx * stride_w
                            s_start_valid = max(0, W_padding_l - s_start - q_pos)
                            s_end_valid = min(
                                s_tile_size, W + W_padding_l - s_start - q_pos
                            )

                            if s_start_valid == 0 and s_end_valid == s_tile_size:
                                if q_first_full is None:
                                    q_first_full = q_idx
                                q_last_full = q_idx

                        if q_first_full is not None and q_first_full > 0:
                            for q_idx in nl.affine_range(q_first_full):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(
                                    s_tile_size, W + W_padding_l - s_start - q_pos
                                )
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = (
                                        s_start + s_start_valid + q_pos - W_padding_l
                                    )
                                    nisa.dma_copy(
                                        dst=input_tile[
                                            s_start_valid:s_end_valid, q_idx
                                        ],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]],
                                            offset=input_base_offset + input_pos,
                                        ),
                                        dge_mode=nisa.dge_mode.none,
                                    )

                        if q_first_full is not None:
                            q_bulk_count = q_last_full - q_first_full + 1
                            input_offset = (
                                input_base_offset
                                + s_start
                                + q_first_full * stride_w
                                - W_padding_l
                            )
                            nisa.dma_copy(
                                dst=input_tile[
                                    0:s_tile_size, q_first_full : q_last_full + 1
                                ],
                                src=img_ref.ap(
                                    pattern=[
                                        [1, s_tile_size],
                                        [stride_w, q_bulk_count],
                                    ],
                                    offset=input_offset,
                                ),
                                dge_mode=nisa.dge_mode.none,
                            )

                        if q_last_full is not None and q_last_full < Q - 1:
                            for q_idx in nl.affine_range(q_last_full + 1, Q):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(
                                    s_tile_size, W + W_padding_l - s_start - q_pos
                                )
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = (
                                        s_start + s_start_valid + q_pos - W_padding_l
                                    )
                                    nisa.dma_copy(
                                        dst=input_tile[
                                            s_start_valid:s_end_valid, q_idx
                                        ],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]],
                                            offset=input_base_offset + input_pos,
                                        ),
                                        dge_mode=nisa.dge_mode.none,
                                    )

                        if q_first_full is None:
                            for q_idx in nl.affine_range(Q):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(
                                    s_tile_size, W + W_padding_l - s_start - q_pos
                                )
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = (
                                        s_start + s_start_valid + q_pos - W_padding_l
                                    )
                                    nisa.dma_copy(
                                        dst=input_tile[
                                            s_start_valid:s_end_valid, q_idx
                                        ],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]],
                                            offset=input_base_offset + input_pos,
                                        ),
                                        dge_mode=nisa.dge_mode.none,
                                    )
                    else:
                        if s_tile_size < S_TILE:
                            nisa.memset(input_tile, 0)

                        if stride_w == 1:
                            input_offset = input_base_offset + s_start - W_padding_l
                            nisa.dma_copy(
                                dst=input_tile[0:s_tile_size, 0:Q],
                                src=img_ref.ap(
                                    pattern=[[1, s_tile_size], [1, Q]],
                                    offset=input_offset,
                                ),
                                dge_mode=nisa.dge_mode.none,
                            )
                        else:
                            bulk_load_size = (Q - 1) * stride_w + 1
                            input_bulk = nl.ndarray(
                                (S_TILE, bulk_load_size),
                                dtype=img_ref.dtype,
                                buffer=nl.sbuf,
                            )

                            input_offset = input_base_offset + s_start - W_padding_l
                            nisa.dma_copy(
                                dst=input_bulk[0:s_tile_size, 0:bulk_load_size],
                                src=img_ref.ap(
                                    pattern=[[1, s_tile_size], [1, bulk_load_size]],
                                    offset=input_offset,
                                ),
                                dge_mode=nisa.dge_mode.none,
                            )

                            nisa.tensor_copy(
                                dst=input_tile[0:s_tile_size, 0:Q],
                                src=input_bulk.ap(
                                    pattern=[
                                        [bulk_load_size, s_tile_size],
                                        [stride_w, Q],
                                    ]
                                ),
                            )

                    input_tiles.append(input_tile)

                for q_tile_idx in nl.affine_range(num_q_tiles):
                    q_start = q_tile_idx * Q_TILE
                    q_tile_size = min(Q_TILE, Q - q_start)

                    result_psum = nl.ndarray(
                        (1, Q_TILE), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.memset(result_psum, 0)

                    for s_tile_idx in nl.sequential_range(num_s_tiles):
                        s_tile_size = min(S_TILE, S - s_tile_idx * S_TILE)

                        nisa.nc_matmul(
                            dst=result_psum[0:1, 0:q_tile_size],
                            stationary=filter_all_tiles[s_tile_idx][
                                0:s_tile_size, channel_idx : channel_idx + 1
                            ],
                            moving=input_tiles[s_tile_idx][
                                0:s_tile_size, q_start : q_start + q_tile_size
                            ],
                        )

                    result_sbuf = nl.ndarray(
                        (1, Q_TILE), dtype=output.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=result_sbuf[0:1, 0:q_tile_size],
                        src=result_psum[0:1, 0:q_tile_size],
                    )

                    out_offset = batch_idx * C * Q + c_global * Q + q_start
                    nisa.dma_copy(
                        dst=output.ap(
                            pattern=[[1, 1], [1, q_tile_size]], offset=out_offset
                        ),
                        src=result_sbuf[0:1, 0:q_tile_size],
                    )

    return output
