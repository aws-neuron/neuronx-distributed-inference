import logging
import os
import time
import unittest
from functools import partial

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
    get_compiler_args,
    get_model_output,
    run_on_cpu,
    setup_debug_env,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAMES = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch32-384",
    "google/vit-huge-patch14-224-in21k",
]


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


class TestViTEncoder(unittest.TestCase):
    def test_4layer(self):
        for model_name in MODEL_NAMES:
            hf_config = ViTConfig.from_pretrained(model_name)
            hf_config.num_hidden_layers = 4  # only test 4 layers

            neuron_config = NeuronConfig(
                tp_degree=2,
                torch_dtype=torch.float32,
            )
            inference_config = ViTInferenceConfig(
                neuron_config=neuron_config, load_config=load_pretrained_config(model_name)
            )
            inference_config.num_hidden_layers = 4  # only test 4 layers

            test_inputs = (
                torch.randn(
                    1,
                    (inference_config.image_size // inference_config.patch_size) ** 2,
                    inference_config.hidden_size,
                ),  # hidden_states
            )
            golden_output = run_on_cpu(test_inputs, ViTEncoder, hf_config)[
                0
            ]  # Output is tuple (hidden_states, all_hidden_states, all_self_attentions)
            logger.info(f"golden_output, {golden_output.shape}")

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
            neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")[
                0
            ]  # return (hidden_states,) + all_hidden_states

            passed, max_err = check_accuracy_embeddings(
                neuron_output, golden_output, plot_outputs=False, rtol=1.3e-6, atol=1e-5
            )
            logger.info(f"\n\n results {model_name} {passed}, {max_err}")
            assert (
                passed
            ), f"test_patch_embeddings did not pass model {model_name}, max error is {max_err}"


if __name__ == "__main__":
    # Set flags for debugging
    setup_debug_env()

    unittest.main()
