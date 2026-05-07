import logging
import os
import pytest
import tempfile

import torch

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, build_cpu_model
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLVisionPatchEmbed,
)

from .test_config import get_qwen3_vl_text_config

VISION_TP_DEGREE = 16
WORLD_SIZE = 64
DTYPE = torch.bfloat16
RTOL = 5e-3

set_random_seed(0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_results(test_name, actual_output, expected_output, rtol=RTOL):
    print("-" * 20)
    print(f"Test result of {test_name}:")
    print("actual_output shape:", actual_output.shape)
    print("expected_output shape:", expected_output.shape)
    passed, max_err = check_accuracy_embeddings(
        actual_output, expected_output, rtol=rtol, atol=1e-5
    )
    assert passed, f"{test_name} failed with max_err={max_err}"
    print("-" * 20)


@pytest.mark.parametrize("seq_len", [1024, 2048, 16384])
def test_vision_patch_embed(seq_len):
    logger.info(f"Running Qwen3VLVisionPatchEmbed test (seq_len={seq_len}) ...")

    # Load the full inference config to get vision_config
    qwen3_vl_config = get_qwen3_vl_text_config(
        dtype=DTYPE,
        text_tp_degree=WORLD_SIZE,
        world_size=WORLD_SIZE,
    )
    vision_inference_config = qwen3_vl_config.vision_config

    # Build HF vision config for the CPU reference model
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_vision_config = Qwen3VLVisionConfig(**vars(hf_config.vision_config))

    # Input shape: (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
    in_channels = hf_vision_config.in_channels
    temporal_patch_size = hf_vision_config.temporal_patch_size
    patch_size = hf_vision_config.patch_size
    input_dim = in_channels * temporal_patch_size * patch_size * patch_size

    test_input = (torch.randn(seq_len, input_dim) * 0.05).to(DTYPE)

    # Build CPU reference model with random weights
    model_tempdir = tempfile.TemporaryDirectory()
    cpu_model, ckpt_path = build_cpu_model(
        Qwen3VLVisionPatchEmbed,
        hf_vision_config,
        dtype=DTYPE,
        checkpoint_dir=model_tempdir.name,
    )

    # Run CPU inference
    logger.info("Running inference on CPU model")
    with torch.no_grad():
        cpu_output = cpu_model(test_input)

    # Create example inputs for Neuron model
    example_inputs = [(torch.ones_like(test_input),)]

    # Build and trace Neuron model
    neuron_model = build_module(
        module_cls=NeuronQwen3VLVisionPatchEmbed,
        example_inputs=example_inputs,
        module_init_kwargs={
            "config": vision_inference_config,
        },
        tp_degree=VISION_TP_DEGREE,
        checkpoint_path=ckpt_path,
        logical_nc_config=2,
    )

    # Run Neuron inference
    logger.info("Running inference on Neuron model")
    neuron_output = neuron_model(test_input)

    # Check results
    check_results(
        f"patch_embed_seq_len={seq_len}",
        neuron_output,
        cpu_output,
        rtol=RTOL,
    )


if __name__ == "__main__":
    for seq_len in [1024, 2048, 16384]:
        test_vision_patch_embed(seq_len=seq_len)
