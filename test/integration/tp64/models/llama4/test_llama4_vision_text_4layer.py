import logging
import os
import time
import uuid
import pytest
import tempfile
from argparse import Namespace

import torch
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration,
    Llama4Config,
)
from transformers import GenerationConfig

from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.models.llama4.modeling_llama4 import NeuronLlama4ForCausalLM
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.config import to_dict

from .test_config import get_llama4_config
from .test_utils import (
    rand_interval,
    setup_debug_env,
)

NUM_BENCHMARK_ITER = 10
NUM_CHUNKS = 5
NUM_TOKENS_TO_CHECK = 16
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()

def get_inputs(config, dtype):
    # inputs
    text_token_len = 16
    num_vision_token_per_chunk = (config.vision_config.image_size // config.vision_config.patch_size) ** 2 * ((config.vision_config.pixel_shuffle_ratio) ** 2)
    vision_token_len = NUM_CHUNKS * int(num_vision_token_per_chunk)
    total_input_len = text_token_len + vision_token_len
    # construct text input tokens
    text_input_ids = torch.rand((config.neuron_config.batch_size, text_token_len)) * config.text_config.vocab_size
    # construct vision input tokens
    vision_input_ids = torch.full([config.neuron_config.batch_size, vision_token_len], fill_value=config.image_token_index)
    # assume vision tokens are before text tokens
    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1)
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, total_input_len), dtype=torch.int32)
    # vision inputs
    pixel_values = torch.nn.Parameter(
            rand_interval(
                -1,
                1,
                (
                    NUM_CHUNKS,
                    config.vision_config.num_channels,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                ),
            )
        ).to(dtype)
    vision_mask = (input_ids == config.image_token_index).unsqueeze(-1)
    vision_mask = vision_mask.to(torch.bool)
    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, vision_mask=vision_mask)

def validate_perf(neuron_model, generation_config, latency_threshold, throughput_threshold):
    print("\nPerformance Benchmarking text+image!")
    benchmark_results = benchmark_sampling(
        model=neuron_model, 
        draft_model=None, 
        generation_config=generation_config, 
        target="all", 
        image=True, 
        benchmark_report_path="benchmark_report_text_and_image.json"
        )
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"

def save_checkpoint(config_path):
    hf_config = Llama4Config.from_pretrained(config_path, torch_dtype=torch.float16)
    logger.info(f"HF config {to_dict(hf_config)}")
    hf_model = Llama4ForConditionalGeneration._from_config(hf_config)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

@pytest.mark.xfail(reason="Expected and actual logit seq len are different. check_accuracy_logits needs fix.", strict=False)
@pytest.mark.parametrize(
    "dtype, model_type, latency_threshold, throughput_threshold",
    [
        pytest.param(
            dtype, model_type, latency_threshold, throughput_threshold,
            id=f"dtype_{str(dtype).split('.')[-1]}_config_{model_type}",
        )
        for (dtype, model_type, latency_threshold, throughput_threshold) in [
            (torch.float16, "16E", 18437*1.1, 445*0.9),
            (torch.float16, "128E", 18310*1.1, 448*0.9)
            ]
    ],
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type, latency_threshold, throughput_threshold):
    # Config
    # Avoid checkpoint name to pass IPScanner
    # Note: the config modified the original HF config "num_hidden_layers": 4 for tiny model integration test.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"config_{model_type}_4layer.json")
    '''Get reference HF CPU model'''
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    config = get_llama4_config(dtype=dtype, model_path=model_path)
    generation_config = GenerationConfig(do_sample=False, pad_token_id=config.text_config.pad_token_id, output_logits=True)

    logger.info(f"\nInferenceConfig {to_dict(config)}\n")

    # Inputs
    inputs = get_inputs(config, dtype)

    '''Get Neuron Model'''
    # Compile model on Neuron
    neuron_model = NeuronLlama4ForCausalLM(model_path=model_path, config=config)

    traced_path = os.path.join(
        model_path,
        f"vision_test_original_cpu_vs_nxdi_neuron_traced_model_dtype_{dtype}",
    )
    os.makedirs(traced_path, exist_ok=True)
    print(f"Compiling Neuron model to {traced_path}")
    neuron_model.compile(traced_path)
    print(f"Compiled Neuron model to {traced_path}")

    # Load model on Neuron
    neuron_model.load(traced_path)
    print(f"Loaded Neuron model from {traced_path}")

    # Validations
    check_accuracy_logits(
        neuron_model,
        generation_config=generation_config,
        num_tokens_to_check=NUM_TOKENS_TO_CHECK,
        inputs=inputs,
        pad_token_id=config.text_config.pad_token_id,
        divergence_difference_tol=0.01,
    )
    validate_perf(neuron_model, generation_config, latency_threshold, throughput_threshold)

    # Clean up temp dir if test pass
    model_tempdir.cleanup()
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(dtype=torch.float16, model_type="128E", latency_threshold=100000, throughput_threshold=0)
