import pytest
import torch

from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)


@pytest.mark.parametrize(
    "attn_kernel_enabled, lnc, q_len, expected_flash_attn_strategy",
    # fmt: off
    [
        (False, 2, 128, FlashAttentionStrategy.NONE),  # LNC2, q_len < 256
        (False, 2, 3968, FlashAttentionStrategy.NONE),  # LNC2, q_len not divisible by 256
        (True, 2, 128, FlashAttentionStrategy.NONE),  # LNC2, q_len < 256
        (False, 2, 256, FlashAttentionStrategy.SHARDED_KERNEL), # LNC2, q_len divisible by 256
        (True, 2, 3968, FlashAttentionStrategy.NONE),  # LNC2, q_len not divisible by 256
        (False, 2, 4096, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len divisible by 256
        (True, 2, 4096, FlashAttentionStrategy.SHARDED_KERNEL),  # LNC2, q_len divisible by 256
        (False, 1, 4096, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (True, 1, 4096, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, q_len >= 4096
        (False, 1, 1024, FlashAttentionStrategy.NONE),  # LNC1, 512 <= q_len < 4096
        (True, 1, 1024, FlashAttentionStrategy.UNSHARDED_KERNEL),  # LNC1, enabled, 512 <= q_len < 4096
        (False, 1, 256, FlashAttentionStrategy.NONE),  # LNC1, q_len < 512
        (True, 1, 256, FlashAttentionStrategy.NONE),  # LNC1, enabled, q_len < 512
    ],
    # fmt: on
)
def test_get_flash_attention_strategy(
    attn_kernel_enabled, lnc, q_len, expected_flash_attn_strategy
):
    attn_module = create_attn_module(attn_kernel_enabled, logical_nc_config=lnc)
    flash_attn_strategy = attn_module.get_flash_attention_strategy(q_len)
    assert flash_attn_strategy == expected_flash_attn_strategy


def create_attn_module(attn_kernel_enabled, logical_nc_config):
    attn_module = NeuronAttentionBase()
    attn_module.attn_kernel_enabled = attn_kernel_enabled
    attn_module.logical_nc_config = logical_nc_config
    return attn_module
