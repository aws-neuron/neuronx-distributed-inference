import logging
import tempfile
import os
import pytest

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import PixtralVisionModelWrapper
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.testing import build_module

from neuronx_distributed_inference.models.config import InferenceConfig

from test_config import get_pixtral_config
from test_utils import (
    get_rtol,
    rand_interval,
    setup_debug_env,
    get_rand_weights,
)

NUM_BENCHMARK_ITER = 10
NUM_CHUNKS_PER_IMAGE = 5
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()

class RefImpl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.patch_size = config.patch_size
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        patch_embeds = self.patch_conv(pixel_values)
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
        return patch_embeds


class TestImplConv2dLinear(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        self.vision_patch_conv_linear = ColumnParallelLinear(
            self.vision_config.num_channels * self.vision_config.patch_size * self.vision_config.patch_size,
            self.vision_config.hidden_size,
            bias=False,
            gather_output=True,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )

    def forward(self, patch_embeds):
        return self.vision_patch_conv_linear(patch_embeds)


def convert_to_neuron_state_dict(config, checkpoint_dir):
    state_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location="cpu")
    state_dict["vision_patch_conv_linear.weight"] = state_dict.pop("patch_conv.weight").reshape(
            -1, config.vision_config.num_channels * config.vision_config.patch_size * config.vision_config.patch_size
        )
    torch.save(state_dict, os.path.join(checkpoint_dir, "checkpoint.pt"))
    return state_dict


@pytest.mark.parametrize(
    "dtype, pixel_values_shape, image_sizes",
    [
        (torch.float16, [1, 3, 512, 1024], [[512, 1024]]),
        (torch.float16, [2, 3, 512, 1024], [[512, 512], [512, 1024]]),
        (torch.float16, [32, 3, 512, 512], [[512, 512]] * 32),
    ],
)
def test_conv2d(dtype, pixel_values_shape, image_sizes):
    # config
    config = get_pixtral_config(dtype=dtype)

    # inputs
    input = torch.randn(pixel_values_shape, dtype=dtype)
    # CPU patchification processing
    model_wrapper = PixtralVisionModelWrapper(config=config, model_cls=TestImplConv2dLinear)
    patch_embeds, _, _ = model_wrapper.patchify(input, image_sizes)
    # use different example_inputs to trace, to rule out hardcoded tensor in HLO
    example_inputs = [
        (torch.ones_like(patch_embeds),),
    ]

    # Get expected output by running ref impl on CPU
    module_cpu = RefImpl(config.vision_config)
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    module_cpu = get_rand_weights(
        module_cpu, os.path.join(model_path, "checkpoint.pt"), dtype=dtype
    )
    expected_output = module_cpu(input, image_sizes)
    logger.info(f"Got expected output {expected_output.shape}, {expected_output}")

    # Compile neuron module
    _ = convert_to_neuron_state_dict(config, model_path)
    module_neuron = build_module(
        TestImplConv2dLinear,
        example_inputs,
        tp_degree=config.vision_config.neuron_config.tp_degree,
        module_init_kwargs={
            "config": config
        },
        checkpoint_path=os.path.join(model_path, "checkpoint.pt"),
    )
    neuron_output = module_neuron(patch_embeds)
    logger.info(f"Got neuron output {neuron_output.shape}, {neuron_output}")

    logger.info(f"\nValidating accuracy for image_size {image_sizes}")
    passed, max_error = check_accuracy_embeddings(
        neuron_output, expected_output, plot_outputs=True, rtol=get_rtol(dtype), atol=1e-5
    )
    logger.info(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}")
    assert passed

    # clean up
    model_tempdir.cleanup()
    print(f"Finished cleaning up {model_path}. Returning.")
    return


if __name__ == "__main__":
    test_conv2d(torch.float16, [1, 3, 1024, 1024], [[512, 1024]])
    test_conv2d(torch.float16, [4, 3, 512, 1024], [[512, 512], [512, 1024]])
    test_conv2d(torch.float16, [32, 3, 512, 512], [[512, 512]] * 32)
