import logging
import os

import pytest
import torch
from transformers.models.llama4.modeling_llama4 import Llama4VisionPixelShuffleMLP

from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import PixelShuffleMLP
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.testing import build_module

from .test_config import get_llama4_config
from .test_utils import (
    cleanup_tmp_workdir,
    get_compiler_args,
    get_rand_weights,
    get_rtol,
    get_tmp_workdir,
    setup_debug_env,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()


@pytest.mark.parametrize(
    "dtype, num_chunks",
    [
        pytest.param(
            dtype,
            num_chunks,
            id=f"dtype_{str(dtype).split('.')[-1]}_num_chunks_{num_chunks}",
        )
        for dtype in [torch.float32, torch.float16]
        for num_chunks in [8, 16, 88]
    ],
)
def test_PixelShuffleMLP_vs_original(dtype, num_chunks):
    # config
    config = get_llama4_config(dtype=dtype)

    # inputs
    seq_len = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    input = torch.randn([num_chunks, seq_len, config.vision_config.hidden_size], dtype=dtype)
    example_inputs = [
        (torch.ones_like(input),),
    ]

    # Get expected output by running HF code on CPU
    module_cpu = Llama4VisionPixelShuffleMLP(config.vision_config)
    tmp_workdir = get_tmp_workdir()
    module_cpu = get_rand_weights(
        module_cpu, os.path.join(tmp_workdir, "checkpoint.pt"), dtype=dtype
    )
    expected_output = module_cpu(input)
    logger.info(f"Got expected output {expected_output.shape}, {expected_output}")

    # Compile neuron module
    module_neuron = build_module(
        PixelShuffleMLP,
        example_inputs,
        tp_degree=config.vision_config.neuron_config.tp_degree,
        module_init_kwargs={"config": config},
        checkpoint_path=os.path.join(tmp_workdir, "checkpoint.pt"),
        compiler_args=get_compiler_args(),
    )
    neuron_output = module_neuron(input)
    logger.info(f"Got neuron output {neuron_output.shape}, {neuron_output}")

    logger.info(f"\nValidating accuracy for num_chunks {num_chunks}")
    passed, max_error = check_accuracy_embeddings(
        neuron_output, expected_output, plot_outputs=True, rtol=get_rtol(dtype), atol=1e-5
    )
    logger.info(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}")
    assert passed

    cleanup_tmp_workdir(tmp_workdir)
    return


if __name__ == "__main__":
    test_PixelShuffleMLP_vs_original(dtype=torch.float16, num_chunks=8)
