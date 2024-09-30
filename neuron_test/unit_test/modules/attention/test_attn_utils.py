# fmt: off
import torch

from neuronx_distributed_inference.modules.attention.utils import create_block_diagonal_attn_mask


# Tests on create_block_diagonal_attn_mask()
def test_attn_mask_for_chunked_prefill_mostly_decode():
    query_lens=torch.tensor([2,3,1,0])
    key_lens=torch.tensor([4,5,4,0])
    max_query_len=torch.tensor(8)
    max_key_len=torch.tensor(16)

    traced_func = prepare_traced_create_block_diagonal_attn_mask(
        query_lens.shape,
        key_lens.shape,
        max_query_len,
        max_key_len
    )

    actual = traced_func(query_lens, key_lens, max_query_len, max_key_len)

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 1st sequence
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 1st sequence
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 5 attend to 2nd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 3 attend to 3rd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ]
    ).to(torch.bool)

    assert torch.equal(actual, expected)


def test_attn_mask_for_chunked_prefill_mostly_prefill():
    query_lens=torch.tensor([2,3,1])
    key_lens=torch.tensor([2,3,4])
    max_query_len=torch.tensor(6)
    max_key_len=torch.tensor(12)

    traced_func = prepare_traced_create_block_diagonal_attn_mask(
        query_lens.shape,
        key_lens.shape,
        max_query_len,
        max_key_len
    )

    actual = traced_func(query_lens, key_lens, max_query_len, max_key_len)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 1 attend to 1st sequence
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 2 attend to 1st sequence
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 1 attend to 2nd sequence
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 2 attend to 2nd sequence
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 4 attend to 3rd sequence
        ]
    ).to(torch.bool)

    assert torch.equal(actual, expected)


def prepare_traced_create_block_diagonal_attn_mask(
    query_lens_shape,
    key_lens_shape,
    max_query_len,
    max_key_len,
):
    example_inputs = (
        torch.zeros(query_lens_shape, dtype=torch.int),
        torch.zeros(key_lens_shape, dtype=torch.int),
        torch.tensor(max_query_len),
        torch.tensor(max_key_len),
    )

    traced_func = torch.jit.trace(create_block_diagonal_attn_mask, example_inputs)
    return traced_func
