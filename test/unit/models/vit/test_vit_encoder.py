import logging
import os
import time
from functools import partial

import pytest
import torch
from neuronx_distributed.trace.model_builder import ModelBuilder
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance
from neuronx_distributed_inference.models.vit.modeling_vit import (
    NeuronViTEncoder,
    ViTInferenceConfig,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from .test_vit_utils import (
    CKPT_DIR,
    VIT_CONFIG_PATH,
    get_compiler_args,
    get_model_output,
    get_rtol,
    run_on_cpu,
    setup_debug_env,
)

# Set flags for debugging
setup_debug_env()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_checkpoint_loader_fn(config):
    state_dict = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu")
    for layer in range(config.num_hidden_layers):
        state_dict[f"layer.{layer}.attention.qkv_proj.q_proj.weight"] = state_dict.pop(
            f"layer.{layer}.attention.attention.query.weight"
        )
        state_dict[f"layer.{layer}.attention.qkv_proj.q_proj.bias"] = state_dict.pop(
            f"layer.{layer}.attention.attention.query.bias"
        )
        state_dict[f"layer.{layer}.attention.qkv_proj.k_proj.weight"] = state_dict.pop(
            f"layer.{layer}.attention.attention.key.weight"
        )
        state_dict[f"layer.{layer}.attention.qkv_proj.k_proj.bias"] = state_dict.pop(
            f"layer.{layer}.attention.attention.key.bias"
        )
        state_dict[f"layer.{layer}.attention.qkv_proj.v_proj.weight"] = state_dict.pop(
            f"layer.{layer}.attention.attention.value.weight"
        )
        state_dict[f"layer.{layer}.attention.qkv_proj.v_proj.bias"] = state_dict.pop(
            f"layer.{layer}.attention.attention.value.bias"
        )

        state_dict[f"layer.{layer}.attention.o_proj.weight"] = state_dict.pop(
            f"layer.{layer}.attention.output.dense.weight"
        )
        state_dict[f"layer.{layer}.attention.o_proj.bias"] = state_dict.pop(
            f"layer.{layer}.attention.output.dense.bias"
        )

        logger.info(f"get_checkpoint_loader_fn converted_state_dict {state_dict.keys()}")

    return state_dict


@pytest.mark.parametrize(
    "dtype,batch_size,model_path",
    [
        pytest.param(
            dtype,
            batch_size,
            model_path,
            id=f"dtype_{str(dtype).split('.')[-1]}_batch_size_{batch_size}_model_path_{model_path.split('/')[-1]}",
        )
        for dtype in [torch.float32, torch.float16]
        for batch_size in [1, 4]
        for model_path in VIT_CONFIG_PATH
    ],
)
def test_4layer(dtype, batch_size, model_path):
    logger.info(f"\n\n Testing dtype {dtype}, batch_size {batch_size}, model_path {model_path}")
    hf_config = ViTConfig.from_pretrained(model_path)
    hf_config.num_hidden_layers = 4  # only test 4 layers

    neuron_config = NeuronConfig(
        tp_degree=2,
        torch_dtype=dtype,
    )
    inference_config = ViTInferenceConfig(
        neuron_config=neuron_config, load_config=load_pretrained_config(model_path)
    )
    inference_config.num_hidden_layers = 4  # only test 4 layers

    test_inputs = (
        torch.randn(
            [
                batch_size,
                (inference_config.image_size // inference_config.patch_size) ** 2,
                inference_config.hidden_size,
            ],
            dtype=dtype,
        ),  # hidden_states
    )
    golden_output = run_on_cpu(test_inputs, ViTEncoder, hf_config, dtype=dtype)[
        0
    ]  # Output is tuple (hidden_states, all_hidden_states, all_self_attentions)
    logger.info(f"golden_output, {golden_output.shape}, {golden_output}")

    # trace model
    # not using test_vit_util functions because we need to define state dict conversion in get_checkpoint_loader_fn
    example_inputs = tuple(torch.ones_like(input) for input in test_inputs)
    model_builder = ModelBuilder(
        router=None,
        tp_degree=inference_config.neuron_config.tp_degree,
        checkpoint_loader=partial(get_checkpoint_loader_fn, inference_config),
    )
    logger.info("Initiated model builder!")

    model_builder.add(
        key=NeuronViTEncoder.__name__,
        model_instance=BaseModelInstance(
            module_cls=partial(NeuronViTEncoder, inference_config), input_output_aliases={}
        ),
        example_inputs=[example_inputs],
        priority_model_idx=0,
        compiler_args=get_compiler_args(),
    )
    logger.info("Added model builder! Starting to trace!")
    start_time = time.time()

    neuron_model = model_builder.trace()

    elapsed_time = time.time() - start_time
    logger.info(f"Traced time taken {elapsed_time} s")

    logger.info("Done tracing the model!")

    # inference and benchmark
    neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")
    logger.info(f"neuron_output, {neuron_output.shape}, {neuron_output}")

    passed, max_err = check_accuracy_embeddings(
        neuron_output,
        golden_output,
        plot_outputs=True,
        rtol=get_rtol(dtype, num_layers=inference_config.num_hidden_layers),
        atol=1e-5,
    )
    logger.info(f"\n\n results {model_path} {passed}, {max_err}")
    assert (
        passed
    ), f"NeuronViTForImageEncoding did not pass dtype {dtype}, batch_size {batch_size}, model_path {model_path}, max error is {max_err}"
