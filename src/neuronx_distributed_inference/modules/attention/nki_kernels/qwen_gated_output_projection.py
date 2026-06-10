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

"""Qwen attention gate fusion for ROW FP8 output projection CTE."""

from typing import List, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_cte.output_projection_cte_parameters import (
    P_MAX,
    QuantizationConfig,
    TilingConfig,
    build_quantization_config,
    build_tiling_config,
    validate_output_projection_inputs,
)
from nkilib.core.output_projection.output_projection_cte.output_projection_cte_quantization import (
    _compute_row_matmul_dequantize,
    _perform_input_row_quantization,
    _write_results_to_output,
)
from nkilib.core.output_projection.output_projection_cte.output_projection_cte_tensor_io import (
    load_bias,
    load_quantized_weights,
    load_row_weight_dequant_scales,
)
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView


def _process_gated_row_quantized_batch_tile(
    attention_view: TensorView,
    gate_view: TensorView,
    output_view: TensorView,
    w_sbuf_list: List[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    weight_row_scale_sbuf: nl.ndarray,
    s_block_idx: int,
    h_block_idx: int,
    n_orig: int,
    d_orig: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)
    s_start = s_block_idx * cfg.s_tile.tile_size

    h_dim = n_orig * d_orig
    quant_attention_sb = []
    global_dequant_scales = []

    attn_flat_view = attention_view.reshape((attention_view.shape[0], h_dim))
    gate_flat_view = gate_view.reshape((gate_view.shape[0], h_dim))

    xpose_dtype = quant_config.quant_data_type if quant_config.use_double_row else nl.bfloat16
    transposed_heads = []
    for head_idx in range(n_orig):
        transposed_heads.append(
            nl.ndarray((d_orig, cfg.s_tile.tile_size), dtype=xpose_dtype, buffer=nl.sbuf)
        )

    for s_sub_idx in range(cfg.s_tile.subtile_dim_info.tile_count):
        curr_s_sub = cfg.s_tile.get_local_subtile_bound(s_block_idx, s_sub_idx)
        if curr_s_sub <= 0:
            break
        s_sub_start = cfg.s_tile.get_local_subtile_start(s_sub_idx)

        attn_sub = nl.ndarray((P_MAX, h_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        attn_sub_view = attn_flat_view.slice(
            dim=0, start=s_sub_start, end=s_sub_start + curr_s_sub
        )
        nisa.dma_copy(dst=attn_sub[:curr_s_sub, :h_dim], src=attn_sub_view.get_view())

        gate_sub = nl.ndarray((P_MAX, h_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        gate_sub_view = gate_flat_view.slice(
            dim=0, start=s_sub_start, end=s_sub_start + curr_s_sub
        )
        nisa.dma_copy(dst=gate_sub[:curr_s_sub, :h_dim], src=gate_sub_view.get_view())
        nisa.activation(
            dst=gate_sub[:curr_s_sub, :h_dim],
            data=gate_sub[:curr_s_sub, :h_dim],
            op=nl.sigmoid,
        )
        nisa.tensor_tensor(
            dst=attn_sub[:curr_s_sub, :h_dim],
            data1=attn_sub[:curr_s_sub, :h_dim],
            data2=gate_sub[:curr_s_sub, :h_dim],
            op=nl.multiply,
        )

        quant_sub, dequant_scale = _perform_input_row_quantization(
            input_sbuf=attn_sub[:curr_s_sub, :h_dim],
            quant_dtype=quant_config.quant_data_type,
        )
        global_dequant_scales.append(dequant_scale)

        quant_nd = quant_sub.reshape((curr_s_sub, n_orig, d_orig))
        for head_idx in range(n_orig):
            if quant_config.use_double_row:
                fp8_head = nl.ndarray(
                    (curr_s_sub, d_orig),
                    dtype=quant_config.quant_data_type,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(
                    dst=fp8_head,
                    src=quant_nd[:curr_s_sub, head_idx, :d_orig],
                )

                fp8_psum_step = 2
                xpose_psum = nl.ndarray(
                    (d_orig, curr_s_sub, fp8_psum_step),
                    dtype=quant_config.quant_data_type,
                    buffer=nl.psum,
                )
                nisa.nc_transpose(
                    dst=xpose_psum.ap(
                        [[curr_s_sub * fp8_psum_step, d_orig], [fp8_psum_step, curr_s_sub]],
                        offset=0,
                    ),
                    data=fp8_head,
                )
                nisa.tensor_copy(
                    dst=transposed_heads[head_idx][
                        :d_orig, s_sub_start : s_sub_start + curr_s_sub
                    ],
                    src=xpose_psum[:d_orig, :curr_s_sub, 0],
                )
            else:
                nisa.dma_transpose(
                    dst=transposed_heads[head_idx][
                        :d_orig, s_sub_start : s_sub_start + curr_s_sub
                    ],
                    src=quant_nd[:curr_s_sub, head_idx, :d_orig],
                )

    if quant_config.use_double_row:
        if n_orig == cfg.n_size // 2:
            for head_idx in range(n_orig):
                packed_sb = nl.ndarray(
                    (cfg.d_size, 2, cfg.s_tile.tile_size),
                    dtype=xpose_dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 0:1, : cfg.s_tile.tile_size],
                    src=transposed_heads[head_idx][: cfg.d_size, : cfg.s_tile.tile_size],
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 1:2, : cfg.s_tile.tile_size],
                    src=transposed_heads[head_idx][
                        cfg.d_size : d_orig, : cfg.s_tile.tile_size
                    ],
                )
                quant_attention_sb.append(packed_sb)
        else:
            for pair_idx in range(n_orig // 2):
                packed_sb = nl.ndarray(
                    (cfg.d_size, 2, cfg.s_tile.tile_size),
                    dtype=xpose_dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 0:1, : cfg.s_tile.tile_size],
                    src=transposed_heads[pair_idx * 2][: cfg.d_size, : cfg.s_tile.tile_size],
                )
                nisa.dma_copy(
                    dst=packed_sb[: cfg.d_size, 1:2, : cfg.s_tile.tile_size],
                    src=transposed_heads[pair_idx * 2 + 1][
                        : cfg.d_size, : cfg.s_tile.tile_size
                    ],
                )
                quant_attention_sb.append(packed_sb)
    else:
        quant_attention_sb = transposed_heads

    result_sb = _compute_row_matmul_dequantize(
        quant_attention_sb=quant_attention_sb,
        w_sbuf_list=w_sbuf_list,
        bias_sbuf=bias_sbuf,
        weight_row_scale_sbuf=weight_row_scale_sbuf,
        input_dequant_scale_sb=[global_dequant_scales],
        s_block_idx=s_block_idx,
        h_block_idx=h_block_idx,
        curr_h_block_size=curr_h_block_size,
        attention_dtype=nl.bfloat16,
        cfg=cfg,
        quant_config=quant_config,
    )

    _write_results_to_output(
        result_sb=result_sb,
        output_view=output_view,
        s_start=s_start,
        s_block_idx=s_block_idx,
        curr_h_block_size=curr_h_block_size,
        cfg=cfg,
    )


def _perform_gated_row_quantized_projection(
    attention_hbm: nl.ndarray,
    gate_hbm: nl.ndarray,
    weight_hbm: nl.ndarray,
    output_hbm: nl.ndarray,
    bias_hbm: Optional[nl.ndarray],
    weight_scale_hbm: nl.ndarray,
    prg_id: int,
    cfg: TilingConfig,
    quant_config: QuantizationConfig,
) -> None:
    weight_hbm = weight_hbm.reshape((cfg.n_size, cfg.d_size, cfg.h_size))
    n_orig = attention_hbm.shape[2]
    d_orig = attention_hbm.shape[3]

    for h_block_idx in range(cfg.h_tile.tile_count):
        h_start = cfg.h_sharded_size * prg_id + h_block_idx * cfg.h_tile.tile_size
        curr_h_block_size = cfg.h_tile.get_tile_bound(h_block_idx)

        weight_view = TensorView(weight_hbm).slice(
            dim=2, start=h_start, end=h_start + curr_h_block_size
        )
        w_sbuf_list = load_quantized_weights(
            weight_view=weight_view, cfg=cfg, quant_config=quant_config
        )

        weight_row_scale_sbuf = load_row_weight_dequant_scales(
            weight_scale_hbm,
            h_start,
            curr_h_block_size,
            cfg.h_tile.tile_size,
        )

        bias_sbuf = None
        if bias_hbm != None:
            bias_view = TensorView(bias_hbm).slice(
                dim=1, start=h_start, end=h_start + curr_h_block_size
            )
            bias_sbuf = load_bias(bias_view=bias_view, cfg=cfg)

        for batch_idx in range(cfg.b_size):
            for s_block_idx in range(cfg.s_tile.tile_count):
                curr_s_tile_size = cfg.s_tile.get_tile_bound(s_block_idx)
                s_start = s_block_idx * cfg.s_tile.tile_size

                attention_view = (
                    TensorView(attention_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=0, start=s_start, end=s_start + curr_s_tile_size)
                )
                gate_view = (
                    TensorView(gate_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=0, start=s_start, end=s_start + curr_s_tile_size)
                )
                output_view = (
                    TensorView(output_hbm)
                    .select(dim=0, index=batch_idx)
                    .slice(dim=1, start=h_start, end=h_start + curr_h_block_size)
                )

                _process_gated_row_quantized_batch_tile(
                    attention_view=attention_view,
                    gate_view=gate_view,
                    output_view=output_view,
                    w_sbuf_list=w_sbuf_list,
                    bias_sbuf=bias_sbuf,
                    weight_row_scale_sbuf=weight_row_scale_sbuf,
                    s_block_idx=s_block_idx,
                    h_block_idx=h_block_idx,
                    n_orig=n_orig,
                    d_orig=d_orig,
                    cfg=cfg,
                    quant_config=quant_config,
                )


@nki.jit
def qwen_gated_output_projection_cte(
    attention: nl.ndarray,
    gate: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    weight_scales: Optional[nl.ndarray] = None,
    output_dtype: Optional[type] = None,
) -> nl.ndarray:
    """Compute output projection over ``attention * sigmoid(gate)`` for Qwen CTE."""
    kernel_assert(
        len(attention.shape) == 4,
        f"Qwen gated output projection expects attention [B, S, N, D], got {len(attention.shape)}D",
    )
    kernel_assert(
        len(gate.shape) == 4,
        f"Qwen gated output projection expects gate [B, S, N, D], got {len(gate.shape)}D",
    )
    b_size, s_size, n_size, d_size = attention.shape
    kernel_assert(
        gate.shape[0] == b_size
        and gate.shape[1] == s_size
        and gate.shape[2] == n_size
        and gate.shape[3] == d_size,
        "Qwen gated output projection requires gate shape to match attention shape",
    )
    _, h_size = weight.shape

    _, n_prgs, prg_id = get_program_sharding_info()
    if n_prgs == None:
        n_prgs = 1
        prg_id = 0

    validate_output_projection_inputs(
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        n_prgs=n_prgs,
        attention_dtype=attention.dtype,
        weight_dtype=weight.dtype,
        quantization_type=QuantizationType.ROW,
        input_scales=None,
        weight_scales=weight_scales,
    )
    quant_config = build_quantization_config(
        quantization_type=QuantizationType.ROW,
        input_scales=None,
        weight_scales=weight_scales,
        input_data_type=attention.dtype,
        weight_data_type=weight.dtype,
    )
    tiling_config = build_tiling_config(
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        n_prgs=n_prgs,
        quant_config=quant_config,
        weight_dtype=weight.dtype,
    )

    out_dtype = output_dtype if output_dtype != None else attention.dtype
    out = nl.ndarray((b_size, s_size, h_size), dtype=out_dtype, buffer=nl.shared_hbm)
    _perform_gated_row_quantized_projection(
        attention_hbm=attention,
        gate_hbm=gate,
        weight_hbm=weight,
        output_hbm=out,
        bias_hbm=bias,
        weight_scale_hbm=weight_scales,
        prg_id=prg_id,
        cfg=tiling_config,
        quant_config=quant_config,
    )
    return out
