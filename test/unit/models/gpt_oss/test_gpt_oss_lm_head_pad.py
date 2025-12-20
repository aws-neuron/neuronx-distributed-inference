from typing import Optional, Tuple

import pytest

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import get_lm_head_pad_config


@pytest.mark.parametrize(
    "vocab_size, tp_degree, lm_head_pad_alignment_size, skip_lm_head_pad, expected_result",
    [
        (201088, 64, 2, False, (False, 1)), # 201088 is divisible by 64 * 2, no need to pad
        (64000, 64, 2, False, (False, 1)),  # 64000 is divisible by 64 * 2, no need to pad
        (64064, 64, 2, False, (True, 2)),   # 64064 is not divisble by 64 * 2, need to pad
        (64064, 64, 2, True, (False, 1)),   # padding is disabled, need to pad
    ],
)
def test_get_lm_head_pad_config(vocab_size: int, tp_degree: int, lm_head_pad_alignment_size: int, skip_lm_head_pad: bool, expected_result: Tuple[bool, int]):
    actual_result = get_lm_head_pad_config(
        vocab_size=vocab_size,
        tp_degree=tp_degree,
        lm_head_pad_alignment_size=lm_head_pad_alignment_size,
        skip_lm_head_pad=skip_lm_head_pad
    )

    assert expected_result == actual_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
