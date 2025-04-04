# fmt: off
import torch
import torch_neuronx

from neuronx_distributed_inference.modules.attention.utils import create_block_diagonal_attn_mask


# Tests on create_block_diagonal_attn_mask()
def test_attn_mask_for_chunked_prefill_mostly_decode():
    query_lens = torch.tensor([2, 3, 1, 0]).int()
    key_lens = torch.tensor([4, 5, 4, 0]).int()
    max_query_len = torch.tensor(8).int()
    max_key_len = torch.tensor(16).int()

    traced_func = prepare_traced_create_block_diagonal_attn_mask(
        query_lens.shape, key_lens.shape, max_query_len, max_key_len
    )

    actual = traced_func(query_lens, key_lens, max_query_len, max_key_len)

    # fmt: off
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
    # fmt: on

    assert torch.equal(actual, expected)


def test_attn_mask_for_chunked_prefill_mostly_prefill():
    query_lens = torch.tensor([2, 3, 1]).int()
    key_lens = torch.tensor([2, 3, 4]).int()
    max_query_len = torch.tensor(6).int()
    max_key_len = torch.tensor(12).int()

    traced_func = prepare_traced_create_block_diagonal_attn_mask(
        query_lens.shape, key_lens.shape, max_query_len, max_key_len
    )

    actual = traced_func(query_lens, key_lens, max_query_len, max_key_len)
    
    # fmt: off
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
    # fmt: on

    assert torch.equal(actual, expected)


def prepare_traced_create_block_diagonal_attn_mask(
    query_lens_shape,
    key_lens_shape,
    max_query_len,
    max_key_len,
):
    example_inputs = (
        torch.zeros(query_lens_shape).int(),
        torch.zeros(key_lens_shape).int(),
        torch.tensor(max_query_len).int(),
        torch.tensor(max_key_len).int(),
    )

    traced_func = torch_neuronx.trace(
        create_block_diagonal_attn_mask,
        example_inputs,
        compiler_workdir="/tmp",
    )
    return traced_func
