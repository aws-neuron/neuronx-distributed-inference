import os

import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from utils import (
    BlockDiagonalCausalFromBottomRightMask,
    get_active_block_tables,
    ref_context_attention,
    sample_input_sizes,
    sample_inputs,
)

from neuronx_distributed_inference.modules.chunked_prefill.flash_pa_with_schedule import (
    flash_attn_varlen_blocksparse_nkifunc,
)
from neuronx_distributed_inference.modules.chunked_prefill.scheduler import GridTileScheduler
from neuronx_distributed_inference.modules.chunked_prefill.utils import ceil_div


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def pad_to_next_power_of_2(a):
    return 2 ** int(a - 1).bit_length() if a > 0 else 0


def _run_test(
    query_lens,
    ctx_lens,
    max_model_len,
    num_heads,
    num_queries_per_kv,
    head_size,
    block_size,
    large_q_tile_size,
    large_kv_tile_size,
    pad_schedule,
    mixed_precision,
    dma_skipping,
):
    dtype = torch.bfloat16 if mixed_precision else torch.float32
    device = xm.xla_device()

    compiler_flags = [
        "-O1",
        "--retry_failed_compilation",
    ]
    compiler_flags_str = " ".join(compiler_flags)
    os.environ["NEURON_CC_FLAGS"] = compiler_flags_str

    max_block_per_request = ceil_div(max_model_len, block_size)
    num_kv_heads = num_heads // num_queries_per_kv
    query, k_active, v_active, k_cache, v_cache, block_table, key, value, query_lens, seq_lens = (
        sample_inputs(
            query_lens=query_lens,
            ctx_lens=ctx_lens,
            max_block_per_request=max_block_per_request,
            block_size=block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
    )

    output_ref, *_ = ref_context_attention(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_queries_per_kv,
        return_max_reduce=True,
    )

    # build neuron program
    B_P_SIZE = 128
    LARGE_KV_TILE_SZ = large_kv_tile_size
    assert LARGE_KV_TILE_SZ >= B_P_SIZE

    # calculate input shapes
    max_num_queries = max(pad_to_next_power_of_2(sum(query_lens)), large_q_tile_size)
    context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
    num_active_blocks = ceil_div(context_lens, block_size).sum().item()
    num_active_blocks = pad_to_multiple(num_active_blocks, LARGE_KV_TILE_SZ // block_size)
    context_kv_len = num_active_blocks * block_size
    assert context_kv_len % LARGE_KV_TILE_SZ == 0, f"invalid context_kv_len={context_kv_len}"

    # pad QKV tensors
    pad_dims = (
        0,
        0,
        0,
        0,
        0,
        max_num_queries - query.shape[0],
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k = F.pad(k_active, pad_dims, "constant", 0)
    v = F.pad(v_active, pad_dims, "constant", 0)

    # permute QKV tensors
    # query: (s, h, d) -> (1, n_heads, d, seq_q)
    # key:   (s, h, d) -> (1, n_kv_heads, d, seq_k)
    # value: (s, h, d) -> (1, n_kv_heads, seq_v, d)
    query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    k = k.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    # k_cache: (n_blocks, block_size, h, d) -> (n_blocks, h, block_size, d)
    # v_cache: (n_blocks, block_size, h, d) -> (n_blocks, h, block_size, d)
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    # transform block table
    active_block_table = get_active_block_tables(
        block_table,
        torch.tensor(query_lens),
        torch.tensor(seq_lens),
        block_size,
        num_active_blocks,
    )

    # Build attention masks
    _, active_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens, block_size=block_size
    )
    active_mask = F.pad(
        active_mask,
        (
            0,
            max_num_queries - active_mask.shape[1],
            0,
            max_num_queries - active_mask.shape[0],
        ),
        "constant",
        0,
    ).bool()

    pa_scheduler = GridTileScheduler(
        torch.tensor(query_lens, dtype=torch.int32),
        context_lens,
        tile_size_q=large_q_tile_size,
        tile_size_kv=large_kv_tile_size,
        block_size=block_size,
        column_order=dma_skipping,
    )
    ctx_token_schedule = pa_scheduler.compute_schedule()
    if pad_schedule:
        num_tiles_padded = pad_to_next_power_of_2(ctx_token_schedule.num_tiles) + 1
        ctx_token_schedule = ctx_token_schedule.pad_schedule(num_tiles_padded)

    num_kv_heads = num_heads // num_queries_per_kv
    tile_block_tables = ctx_token_schedule.build_tile_block_tables(
        active_block_table,
        skip_value=k_cache.shape[0] * num_kv_heads * 1000,
    )
    tile_masks = ctx_token_schedule.build_tile_masks()

    tile_q_indices = torch.tensor(ctx_token_schedule.tile_q_indices)
    tile_block_tables = torch.tensor(tile_block_tables)
    tile_masks = torch.tensor(tile_masks)

    input_args = (
        query.to(device=device),
        k.to(device=device),
        v.to(device=device),
        k_cache.to(device=device),
        v_cache.to(device=device),
        tile_q_indices.to(device=device),
        tile_block_tables.to(device=device),
        tile_masks.to(device=device),
        active_mask.to(device=device),
    )
    input_kwargs = dict(
        n_kv_head=num_kv_heads,
        head_size=head_size,
        mixed_precision=mixed_precision,
    )

    output_nki = flash_attn_varlen_blocksparse_nkifunc(*input_args, **input_kwargs)

    num_actual_tokens = sum(query_lens)
    # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
    output_nki = output_nki.cpu().permute(0, 2, 1, 3)
    output_nki = output_nki[0, :num_actual_tokens, :, :]
    output_ref_padded = F.pad(
        output_ref,
        (0, 0, 0, 0, 0, 0, 0, max_num_queries - output_ref.shape[0]),
        "constant",
        0,
    )
    output_ref = output_ref_padded.transpose(0, 1)[0, :num_actual_tokens, :, :]

    torch.testing.assert_close(output_nki, output_ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "large_q_tile_size,large_kv_tile_size,block_size",
    [
        (128, 2048, 32),  # 64 blocks
        # (128, 2048, 16),  # 128 blocks
        # (128, 8192, 32),  # 256 blocks
        # (128, 512, 1),  # 512 blocks
        # (256, 2048, 256),  # 8 blocks
        # (256, 4096, 32),  # 128 blocks
        (256, 8192, 64),  # 128 blocks
        # (256, 1024, 4),  # 256 blocks
    ],
)
@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size",
    [
        (4, 2, 16),
        # (32, 8, 64),
        # (4, 4, 128),
        # (8, 1, 32),
        (2, 2, 128),
    ],
)
@pytest.mark.parametrize(
    "prefill_batch_size,decode_batch_size",
    [
        (8, 0),
        (1, 7),
        (0, 8),
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@pytest.mark.parametrize("pad_schedule", [True, False])
@pytest.mark.parametrize("dma_skipping", [True, False])
@torch.inference_mode()
def test_blocksparse_flash_paged_attention(
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    large_q_tile_size,
    large_kv_tile_size,
    mixed_precision: bool,
    pad_schedule: bool,
    dma_skipping: bool,
) -> None:
    assert large_kv_tile_size % block_size == 0

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 32
    max_ctx_len = 1024
    min_query_len = 16
    max_query_len = 512
    query_lens, ctx_lens = sample_input_sizes(
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        min_query_len=min_query_len,
        max_query_len=max_query_len,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
    )
    max_model_len = max(max_query_len, max_ctx_len) * 4
    _run_test(
        query_lens=query_lens,
        ctx_lens=ctx_lens,
        max_model_len=max_model_len,
        num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        block_size=block_size,
        large_q_tile_size=large_q_tile_size,
        large_kv_tile_size=large_kv_tile_size,
        pad_schedule=pad_schedule,
        mixed_precision=mixed_precision,
        dma_skipping=dma_skipping,
    )
