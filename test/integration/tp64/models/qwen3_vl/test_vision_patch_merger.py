import logging
import os
import pytest
import tempfile

import torch

from functools import partial

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger

from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, build_cpu_model
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import NeuronQwen3VLVisionPatchMerger

from .test_config import get_qwen3_vl_text_config

VISION_TP_DEGREE = 16
WORLD_SIZE = 16
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
        actual_output.to(torch.float32), expected_output.to(torch.float32), rtol=rtol, atol=1e-5
    )
    assert passed, f"{test_name} failed with max_err={max_err}"
    print("-" * 20)


@pytest.mark.parametrize("use_postshuffle_norm", [False, True])
@pytest.mark.parametrize("seq_len", [1024, 2048, 16384])
def test_vision_patch_merger(use_postshuffle_norm, seq_len):
    logger.info(f"Running Qwen3VLVisionPatchMerger test (use_postshuffle_norm={use_postshuffle_norm}, seq_len={seq_len}) ...")

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

    hidden_size = hf_vision_config.hidden_size  # 1152
    spatial_merge_size = hf_vision_config.spatial_merge_size  # 2
    merged_hidden_size = hidden_size * (spatial_merge_size ** 2)  # 4608

    # Determine input shape based on use_postshuffle_norm
    # use_postshuffle_norm=False: input is (seq_len, hidden_size) — norm on hidden_size, then view to merged_hidden_size
    # use_postshuffle_norm=True: input is (seq_len, merged_hidden_size) — view to merged_hidden_size first, then norm
    if use_postshuffle_norm:
        input_shape = (seq_len, merged_hidden_size)
    else:
        # Input must be (seq_len * spatial_merge_size^2, hidden_size) so that view(-1, merged_hidden_size) works
        input_shape = (seq_len * spatial_merge_size ** 2, hidden_size)

    test_input = (torch.randn(input_shape) * 0.05).to(DTYPE)

    # Build CPU reference model with random weights
    HFMergerCls = partial(Qwen3VLVisionPatchMerger, use_postshuffle_norm=use_postshuffle_norm)
    model_tempdir = tempfile.TemporaryDirectory()
    cpu_model, ckpt_path = build_cpu_model(
        HFMergerCls,
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
        module_cls=NeuronQwen3VLVisionPatchMerger,
        example_inputs=example_inputs,
        module_init_kwargs={
            "config": vision_inference_config,
            "use_postshuffle_norm": use_postshuffle_norm,
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
        f"patch_merger_postshuffle_norm={use_postshuffle_norm}",
        neuron_output,
        cpu_output,
        rtol=RTOL,
    )


if __name__ == "__main__":
    for seq_len in [1024, 2048, 16384]:
        test_vision_patch_merger(use_postshuffle_norm=False, seq_len=seq_len)
        test_vision_patch_merger(use_postshuffle_norm=True, seq_len=seq_len)
