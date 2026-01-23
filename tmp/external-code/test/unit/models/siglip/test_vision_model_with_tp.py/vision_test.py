# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import copy
import logging
import os
import time
import uuid

import numpy as np
import pytest
import torch
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.benchmark import LatencyCollector

from models.siglip.modeling_siglip import NeuronSiglipConfig, SiglipInferenceConfig, NeuronSiglipVisionModel
from models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig

from scripts.vision_test.test_config import get_gemma3_config
from scripts.vision_test.test_utils import (
    cleanup_tmp_workdir,
    get_rand_weights,
    get_rtol,
    get_tmp_workdir,
    rand_interval,
    setup_debug_env,
)

NUM_BENCHMARK_ITER = 1
NUM_CHUNKS_PER_IMAGE = 1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()


class original_vision_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
        hf_config.text_config.num_hidden_layers = 4
        hf_config.vision_config.num_hidden_layers = 4

        self.model = Gemma3ForConditionalGeneration(hf_config)
        # self.vision_model = SiglipVisionModel(hf_config)
        # self.multi_modal_projector = Llama4MultiModalProjector(config)

    def forward(self, pixel_values):
        image_outputs = self.model.vision_tower(pixel_values)
        hidden_state = image_outputs.last_hidden_state
        print(f"in original_vision_model hidden_state {hidden_state.shape}")

        projected_vision_emb = self.model.multi_modal_projector(hidden_state)
        print(f"in original_vision_model projected_vision_emb {projected_vision_emb.shape}")

        return projected_vision_emb


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            id=f"dtype_{str(dtype).split('.')[-1]}",
        )
        for dtype in [torch.bfloat16]
    ],
)
def test_original_cpu_vs_nxdi_neuron(dtype):
    # Config
    # Note: the config modified the original HF config "num_hidden_layers": 4 for tiny model integration test.
    config = get_gemma3_config(dtype)
    # Make sure the vision model gets the correct neuron_config
    # config.neuron_config = copy.deepcopy(config.vision_config.neuron_config)

    # logger.info(f"\nCONFIG {vars(config)}")
    # logger.info(f"\nCONFIG.vision_config {vars(config.vision_config)}")
    # logger.info(f"\nCONFIG.neuron_config {vars(config.neuron_config)}")
    # logger.info(f"\nCONFIG.vision_config.neuron_config {vars(config.vision_config.neuron_config)}")

    # Get reference CPU model
    cpu_model = original_vision_model().to(dtype)
    # get random weights
    tmp_workdir = get_tmp_workdir()
    cpu_model = get_rand_weights(
        cpu_model, os.path.join(tmp_workdir, "model.safetensors"), dtype=dtype
    )
    print(f"Got ref CPU model and saved random checkpoint to {tmp_workdir}")

    # Compile model on Neuron
    
    config._name_or_path = tmp_workdir
    module_neuron = NeuronGemma3ForCausalLM(model_path=tmp_workdir, config=config)

    traced_path = os.path.join(
        tmp_workdir,
        f"vision_test_original_cpu_vs_nxdi_neuron_traced_model_dtype-{dtype}_{uuid.uuid4()}",
    )
    os.makedirs(traced_path, exist_ok=True)
    module_neuron.compile(traced_path)
    print(f"Compiled Neuron model to {traced_path}")

    # Load model on Neuron
    module_neuron.load(traced_path)
    print(f"Loaded Neuron model from {traced_path}")

    for num_images in [1]: #[1, 2, 5]:
        # Inputs
        # Assuming each image has NUM_CHUNKS_PER_IMAGE=5 chunks, 1 image should hit bucket size 8
        # 2 images should hit bucket size 16
        # 5 images should hit bucket size 88
        pixel_values = torch.nn.Parameter(
            rand_interval(
                -1,
                1,
                (
                    NUM_CHUNKS_PER_IMAGE * num_images,
                    config.vision_config.num_channels,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                ),
            )
        ).to(dtype)

        print("Generating golden...")
        loaded_golden = cpu_model(pixel_values).to(torch.float32)
        print(f"Generated golden {loaded_golden.shape}, {loaded_golden}")

        # Run NxDI implementation on Neuron
        # neuron_latency_collector = LatencyCollector()
        for i in range(NUM_BENCHMARK_ITER):
            # neuron_latency_collector.pre_hook()
            neuron_output = module_neuron.vision_encoder_model(pixel_values)
            # neuron_latency_collector.hook()
        # NeuronLlama4VisionEmbeddings pad the output to max bucket size before returning
        # depad here to match with ref impl output
        neuron_output = neuron_output[: NUM_CHUNKS_PER_IMAGE * num_images]  # .flatten(0, 1)
        logger.info(f"Got neuron output {neuron_output.shape} {neuron_output}")
        # Benchmark report
        # for p in [25, 50, 90, 99]:
        #     latency = np.percentile(neuron_latency_collector.latency_list, p) * 1000
        #     print(f"Neuron inference latency_ms_p{p}: {latency}")

        print(
            f"\ntest_original_cpu_vs_nxdi_neuron Validating accuracy pixel_values {pixel_values.shape}"
        )
        passed, max_error = check_accuracy_embeddings(
            neuron_output,
            loaded_golden,
            plot_outputs=False,
            rtol=get_rtol(data_type=dtype, num_layers=config.vision_config.num_hidden_layers),
            atol=1e-5,
        )
        print(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}\n")
        assert passed

    # clean up traced_path
    cleanup_tmp_workdir(tmp_workdir)
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(dtype=torch.bfloat16)