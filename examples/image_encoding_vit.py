from transformers import ViTModel, AutoImageProcessor
from PIL import Image
import time
import torch
import os
import numpy as np
import logging

import torch_xla

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.benchmark import LatencyCollector
from neuronx_distributed_inference.models.vit.modeling_vit import NeuronViTForImageEncoding, ViTInferenceConfig


NUM_BENCHMARK_ITER = 10
MODEL_PATH = "/home/ubuntu/model_hf/google--vit-huge-patch14-224-in21k/"
TRACED_MODEL_PATH = "/home/ubuntu/model_hf/google--vit-huge-patch14-224-in21k/traced_model/"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def run_vit_encoding(validate_accuracy=True):
    # Define configs
    neuron_config = NeuronConfig(
        tp_degree=32,
        torch_dtype=torch.float32,
    )
    inference_config = ViTInferenceConfig(
        neuron_config=neuron_config, 
        load_config=load_pretrained_config(MODEL_PATH),
        use_mask_token=False,
        add_pooling_layer=False,
        interpolate_pos_encoding=False
    )

    # input image
    image_file = "dog.jpg" # [512, 512]
    with open(image_file, "rb") as f:
        image = Image.open(f).convert("RGB")
    print(f"Input image size {image.size}")
    # preprocess input image
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

    # Get neuron model
    neuron_model = NeuronViTForImageEncoding(model_path=MODEL_PATH, config=inference_config)

    # Compile model on Neuron
    compile_start_time = time.time()
    neuron_model.compile(TRACED_MODEL_PATH)
    compile_elapsed_time = time.time() - compile_start_time
    print(f"Compilation time taken {compile_elapsed_time} s")

    # Load model on Neuron
    neuron_model.load(TRACED_MODEL_PATH)
    print("Done loading neuron model")

    # Run NxDI implementation on Neuron
    neuron_latency_collector = LatencyCollector()
    for i in range(NUM_BENCHMARK_ITER):
        neuron_latency_collector.pre_hook()
        neuron_output = neuron_model(pixel_values)[0] # NeuronViTModel output (sequence_output,) or (sequence_output, pooled_output)
        neuron_latency_collector.hook()
    print(f"Got neuron output {neuron_output.shape} {neuron_output}")
    # Benchmark report
    for p in [25, 50, 90, 99]:
        latency = np.percentile(neuron_latency_collector.latency_list, p) * 1000
        print(f"Neuron inference latency_ms_p{p}: {latency}")

    # The below section is optional, use if you want to validate e2e accuracy against golden
    if validate_accuracy:
        # Get CPU model
        cpu_model = ViTModel.from_pretrained(MODEL_PATH)
        print(f"cpu model {cpu_model}")

        # Get golden output by running original implementation on CPU
        cpu_latency_collector = LatencyCollector()
        for i in range(NUM_BENCHMARK_ITER):
            cpu_latency_collector.pre_hook()
            golden_output = cpu_model(pixel_values).last_hidden_state
            cpu_latency_collector.hook()
        print(f"expected_output {golden_output.shape} {golden_output}")
        # Benchmark report
        for p in [25, 50, 90, 99]:
            latency = np.percentile(cpu_latency_collector.latency_list, p) * 1000
            print(f"CPU inference latency_ms_p{p}: {latency}")

        # Compare output logits
        passed, max_err = check_accuracy_embeddings(neuron_output, golden_output, plot_outputs=True, atol=1e-5, rtol=1e-5)
        print(f"Golden and Neuron outputs match: {passed}, max relative error: {max_err}")

        

if __name__ == "__main__":
    # Set flags for debugging
    setup_debug_env()

    run_vit_encoding(validate_accuracy=True)