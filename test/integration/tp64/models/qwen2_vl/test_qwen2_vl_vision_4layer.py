import logging
import pytest
import tempfile
import os
import numpy as np

import torch

from neuronx_distributed_inference.utils.benchmark import LatencyCollector
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import (
    NeuronQwen2VLForImageEncoding,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from .test_config import get_qwen2_vl_config

from neuronx_distributed_inference.utils.testing import _rand_interval


NUM_BENCHMARK_ITER = 1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_checkpoint(config_path, dtype):
    hf_model = NeuronQwen2VLForImageEncoding.load_hf_model(config_path)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir, hf_model


@pytest.mark.parametrize(
    "dtype, model_type",
    [(torch.bfloat16, "Qwen2_vl_vision_only")]
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type):
    # Config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    # Get reference HF CPU model
    model_tempdir, hf_model = save_checkpoint(config_path, dtype=dtype)
    model_path = model_tempdir.name

    num_of_images = 50
    config = get_qwen2_vl_config(dtype=dtype, vision_tp_degree=4, world_size=4, text_seq_length=253*num_of_images, vision_seq_len=1012*num_of_images, vision_buckets=[num_of_images], model_path=model_path)

    # Compile model on Neuron
    neuron_model = NeuronQwen2VLForImageEncoding(model_path=model_path, config=config)

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

    pixel_values_shape_test_cases = (
        [num_of_images * 1012, 1176],
    )
    grid_thw_test_cases = (
        torch.tensor([[1, 22, 46]] * num_of_images),
    )

    for pixel_values_shape, grid_thw in zip(pixel_values_shape_test_cases, grid_thw_test_cases):
        # Construct inputs
        pixel_values = torch.nn.Parameter(
            _rand_interval(-1, 1, dtype, *pixel_values_shape)
        ).to(dtype)

        print("Generating golden...")
        # Use HF CPU FP32 output as golden to match check_accuracy_logits behavior
        golden_output = hf_model(pixel_values.float(), grid_thw)
        print(f"Generated golden {golden_output.shape}, {golden_output}")

        # Accuracy Validation and Latency Benchmark
        neuron_latency_collector = LatencyCollector()
        for i in range(NUM_BENCHMARK_ITER):
            neuron_latency_collector.pre_hook()
            neuron_output = neuron_model(pixel_values, grid_thw)
            neuron_latency_collector.hook()

        # Torch-level profile
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
            neuron_output = neuron_model(pixel_values, grid_thw)

        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace("torch_profile.json")

        if isinstance(neuron_output, list):
            neuron_output = neuron_output[0][0].squeeze(0).to("cpu")

        # Benchmark report
        for p in [25, 50, 90, 99]:
            latency = np.percentile(neuron_latency_collector.latency_list, p) * 1000
            print(f"Neuron inference latency_ms_p{p}: {latency}")

        print(
            "\ntest_original_cpu_vs_nxdi_neuron Validating accuracy"
        )
        passed, max_error = check_accuracy_embeddings(
            neuron_output.float(),
            golden_output.float(),
            plot_outputs=True,
            rtol=1e-2,
            atol=1e-5,
        )
        print(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}\n")
        assert passed

    # clean up
    model_tempdir.cleanup()
    print(f"Finished cleaning up {model_path}. Returning.")
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(dtype=torch.bfloat16, model_type="qwen2_vl_vision_only")
