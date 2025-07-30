import logging
import os

import pytest
import torch
from transformers.models.llama4.modeling_llama4 import Llama4UnfoldConvolution

from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import NeuronConv2dPatch
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


class NeuronConv2dPatchTestWrapper(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, dtype):
        super().__init__()
        self.conv1 = NeuronConv2dPatch(
            in_channels=in_channels,  # 3
            out_channels=out_channels,  # 1408
            kernel_size=kernel_size,  # 14
            stride=stride,  # 14
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x):
        return self.conv1(x)

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        # Convert unfold+linear to conv2d
        kernel_size = [14, 14]
        print(f"yidetest: {model_state_dict.keys()}")
        conv1_linear_weight = model_state_dict.pop("linear.weight")
        in_channels = conv1_linear_weight.shape[1] // (kernel_size[0] * kernel_size[1])
        new_shape = (-1, in_channels, kernel_size[0], kernel_size[1])
        model_state_dict["conv1.conv.weight"] = conv1_linear_weight.reshape(new_shape)


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
def test_original_cpu_vs_nxdi_neuron(dtype, num_chunks):
    # config
    config = get_llama4_config(dtype)

    # inputs
    input = torch.randn(
        [
            num_chunks,
            config.vision_config.num_channels,
            config.vision_config.image_size,
            config.vision_config.image_size,
        ],
        dtype=dtype,
    )
    # use different example_inputs to trace, to rule out hardcoded tensor in HLO
    example_inputs = [
        (torch.ones_like(input),),
    ]

    # Get expected output by running HF code on CPU
    module_cpu = Llama4UnfoldConvolution(config.vision_config)
    tmp_workdir = get_tmp_workdir()
    module_cpu = get_rand_weights(
        module_cpu, os.path.join(tmp_workdir, "checkpoint.pt"), dtype=dtype
    )
    expected_output = module_cpu(input)
    logger.info(f"Got expected output {expected_output.shape}, {expected_output}")

    # Compile neuron module
    module_neuron = build_module(
        NeuronConv2dPatchTestWrapper,
        example_inputs,
        tp_degree=config.vision_config.neuron_config.tp_degree,
        module_init_kwargs={
            "in_channels": config.vision_config.num_channels,
            "out_channels": config.vision_config.hidden_size,
            "kernel_size": config.vision_config.patch_size,
            "stride": config.vision_config.patch_size,
            "bias": False,
            "dtype": dtype,
        },
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
    test_original_cpu_vs_nxdi_neuron(dtype=torch.float16, num_chunks=88)
