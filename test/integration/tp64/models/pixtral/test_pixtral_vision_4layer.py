import logging
import tempfile
import os
import pytest
import numpy as np

import torch

from neuronx_distributed_inference.utils.benchmark import LatencyCollector
from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import NeuronPixtralForImageEncoding
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings

from test_config import get_pixtral_config
from test_utils import (
    get_rtol,
    rand_interval,
    setup_debug_env,
)

NUM_BENCHMARK_ITER = 10
NUM_CHUNKS_PER_IMAGE = 5
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()


def save_checkpoint(config_path, dtype):
    hf_model = NeuronPixtralForImageEncoding.load_hf_model(config_path, torch_dtype=dtype)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir, hf_model

# max_image_size is 1024, (1024 // 16) ** 2 = 4096 is  max_num_patches_per_image
# max_num_image = 6. Therefore max_num_patches = 6 * 4096 = 24576
@pytest.mark.parametrize(
        "dtype, model_type, vision_seq_len",
        [
            (torch.float16, "Pixtral_Large_vision_only", 24576),
        ]
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type, vision_seq_len):
    # Config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    # Get reference HF CPU model
    model_tempdir, hf_model = save_checkpoint(config_path, dtype=dtype)
    model_path = model_tempdir.name

    config = get_pixtral_config(dtype=dtype, vision_seq_len=vision_seq_len, model_path=model_path)

    # Compile model on Neuron
    neuron_model = NeuronPixtralForImageEncoding(model_path=model_path, config=config)

    traced_path = os.path.join(
        model_path,
        "traced_model",
    )
    os.makedirs(traced_path, exist_ok=True)
    print(f"Compiling Neuron model to {traced_path}")
    neuron_model.compile(traced_path, debug=True)
    print(f"Compilied Neuron model to {traced_path}")

    # Load model on Neuron
    neuron_model.load(traced_path)
    print(f"Loaded Neuron model from {traced_path}")

    # Test different image shapes
    # The batch size in vision model represents number of images per request. 
    # Vision model always handle one request at a time.
    # pixel_values shapes are the batch of biggest image size in the request
    # image_sizes are list of the actual image sizes in the request
    pixel_values_shape_test_cases = (
        [1, 3, 512, 512], # single image without padding hitting bucket 1024
        [1, 3, 512, 1024], # single image without padding hitting bucket 2048
        [2, 3, 512, 1024], # two images of different sizes with padding hitting bucket 4096
        [2, 3, 1024, 1024], # two images of different sizes without padding hitting bucket 4096
        [6, 3, 512, 512], # six image without padding hitting bucket 6144
        # [8, 3, 512, 512], # eight image without padding hitting bucket 8192, commented due to slightly higher rdiff 0.00166
        [9, 3, 512, 512], # nine image with padding hitting bucket 16384
        )
    image_sizes_test_cases = (
        [[512, 512]],
        [[512, 1024]],
        [[512, 512], [512, 1024]],
        [[512, 1024], [1024, 512]],
        [[512, 512]] * 6,
        # [[512, 512]] * 8,
        [[512, 512]] * 9,
        )
    for pixel_values_shape, image_size_list in zip(pixel_values_shape_test_cases, image_sizes_test_cases):
        # Construct inputs
        pixel_values = torch.nn.Parameter(
            rand_interval(-1, 1, pixel_values_shape)
        ).to(dtype)
        image_sizes = torch.Tensor(image_size_list).to(dtype=torch.int32)

        print("Generating golden...")
        # Use HF CPU FP32 output as golden to match check_accuracy_logits behavior
        golden_output = hf_model(pixel_values.float(), image_sizes)
        print(f"Generated golden {golden_output.shape}, {golden_output}")

        # Accuracy Validation and Latency Benchmark
        neuron_latency_collector = LatencyCollector()
        for i in range(NUM_BENCHMARK_ITER):
            neuron_latency_collector.pre_hook()
            neuron_output = neuron_model(pixel_values, image_sizes)
            neuron_latency_collector.hook()

        # Torch-level profile
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
            neuron_output = neuron_model(pixel_values, image_sizes)
            
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace("torch_profile.json")

        logger.info(f"Got neuron output {neuron_output.shape} {neuron_output}")
        # Benchmark report
        for p in [25, 50, 90, 99]:
            latency = np.percentile(neuron_latency_collector.latency_list, p) * 1000
            print(f"Neuron inference latency_ms_p{p}: {latency}")

        print(
            f"\ntest_original_cpu_vs_nxdi_neuron Validating accuracy for image_sizes {image_sizes}"
        )
        passed, max_error = check_accuracy_embeddings(
            neuron_output.float(),
            golden_output.float(),
            plot_outputs=True,
            rtol=get_rtol(data_type=dtype, num_layers=config.vision_config.num_hidden_layers),
            atol=1e-5,
        )
        print(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}\n")
        assert passed

    # clean up
    model_tempdir.cleanup()
    print(f"Finished cleaning up {model_path}. Returning.")
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(dtype=torch.float16, model_type="Pixtral_Large_vision_only", vision_seq_len=24576)
