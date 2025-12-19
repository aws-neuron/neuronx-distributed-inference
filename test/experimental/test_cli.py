import pytest

import neuronx_distributed_inference.experimental.cli as cli


@pytest.mark.experimental
def test_generate_cpu():
    # TODO move to downloading weights from HF
    result = cli.generate_cpu(
        batch_size=1,
        prompts=[
            "I am going to count from 1 to 15 only once. I will not say anything further. 1,2,3,4,"
        ],
    )
    assert result == [
        "<|begin_of_text|>I am going to count from 1 to 15 only once. I will not say anything further. 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15.<|eot_id|>"
    ]


@pytest.mark.experimental
def test_generate_nxd():
    # TODO move to downloading weights from HF
    cli.compile()
    result = cli.generate_nxd(
        prompts=[
            "I am going to count from 1 to 15 only once. I will not say anything further. 1,2,3,4,"
        ]
    )
    assert result == [
        "<|begin_of_text|>I am going to count from 1 to 15 only once. I will not say anything further. 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15.<|eot_id|>"
    ]
