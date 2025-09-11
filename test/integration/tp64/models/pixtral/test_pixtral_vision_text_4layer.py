import logging
import os
import pytest
import tempfile
from argparse import Namespace

import torch
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaConfig,
)
from transformers import GenerationConfig

from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import NeuronPixtralForCausalLM
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.config import to_dict

from test_config import get_pixtral_config
from test_utils import setup_debug_env

NUM_BENCHMARK_ITER = 10
NUM_TOKENS_TO_CHECK = 16
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()

def get_inputs(config, dtype):
    # inputs
    num_channels = 3
    text_token_len = 16
    image_size = 512
    vision_token_len = 1024
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
    pixel_values = torch.zeros([config.neuron_config.batch_size, num_channels, image_size, image_size])
    vision_mask = (input_ids == config.image_token_index).unsqueeze(-1)
    image_sizes = [[image_size, image_size]] * config.neuron_config.batch_size

    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, vision_mask=vision_mask, image_sizes=image_sizes)

def validate_perf(neuron_model, generation_config, latency_threshold, throughput_threshold):
    print("\nPerformance Benchmarking text+image!")
    benchmark_results = benchmark_sampling(
        model=neuron_model, 
        draft_model=None, 
        generation_config=generation_config, 
        target="all", 
        image=True, 
        num_runs=5,
        benchmark_report_path="benchmark_report_text_and_image.json"
        )
    latency = benchmark_results["e2e_model"]["latency_ms_p50"]
    assert latency < latency_threshold, f"latency ({latency}) is above threshold ({latency_threshold})"
    throughput = benchmark_results["e2e_model"]["throughput"]
    assert throughput > throughput_threshold, f"throughput ({throughput}) is below threshold ({throughput_threshold})"

def save_checkpoint(config_path):
    hf_config = LlavaConfig.from_pretrained(config_path, torch_dtype=torch.float16)
    logger.info(f"HF config {to_dict(hf_config)}")
    hf_model = LlavaForConditionalGeneration._from_config(hf_config)
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

@pytest.mark.parametrize(
    "dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, text_buckets, flash_decoding_enabled, sequence_parallel_enabled, use_text_kernels, latency_threshold, throughput_threshold",
    [
        pytest.param(torch.float16, "Pixtral_Large_vision_text", 1, 64, 16, 64, 10240, 10240, [2048, 4096, 10240], False, True, True, 12073*1.1, 842*0.9),
        pytest.param(torch.float16, "Pixtral_Large_vision_text", 1, 64, 16, 64, 10240, 10240, [2048, 4096, 10240], True, False, False, 12748*1.1, 804*0.9),
        # TODO: add bs>1 test cases after resolving shape mismatch with expected logits
    ],
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, text_buckets, flash_decoding_enabled, sequence_parallel_enabled, use_text_kernels, latency_threshold, throughput_threshold):
    # Config
    # Avoid checkpoint name to pass IPScanner
    # Note: the config modified the original HF config "num_hidden_layers": 4 for tiny model integration test.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    '''Get reference HF CPU model'''
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    config = get_pixtral_config(dtype=dtype,
                                model_path=model_path,
                                tkg_batch_size=tkg_batch_size,
                                text_tp_degree=text_tp_degree,
                                vision_tp_degree=vision_tp_degree,
                                world_size=world_size,
                                text_seq_length=text_seq_length,
                                vision_seq_len=vision_seq_len,
                                text_buckets=text_buckets,
                                flash_decoding_enabled=flash_decoding_enabled,
                                sequence_parallel_enabled=sequence_parallel_enabled,
                                use_text_kernels=use_text_kernels,
                                )
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 1,
                                         eos_token_id = [2],
                                         pad_token_id=200018, 
                                         output_logits=True)

    logger.info(f"\nInferenceConfig {to_dict(config)}\n")

    # Inputs
    inputs = get_inputs(config, dtype)

    '''Get Neuron Model'''
    # Compile model on Neuron
    neuron_model = NeuronPixtralForCausalLM(model_path=model_path, config=config)

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
    test_original_cpu_vs_nxdi_neuron(torch.float16, "Pixtral_Large_vision_text", 1, 64, 16, 64, 10240, 10240, [2048, 4096, 10240], False, True, True, 12073*1.1, 842*0.9)
    test_original_cpu_vs_nxdi_neuron(torch.float16, "Pixtral_Large_vision_text", 1, 64, 16, 64, 10240, 10240, [2048, 4096, 10240], True, False, False, 12748*1.1, 804*0.9)