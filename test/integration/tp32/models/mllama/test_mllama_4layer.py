import logging
import os
import pytest
import tempfile
import torch_xla
from argparse import Namespace

import torch
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration, MllamaConfig
from transformers import GenerationConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.models.mllama.modeling_mllama import NeuronMllamaForCausalLM
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.config import to_dict
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.models.mllama.utils import get_image_tensors, create_vision_mask, get_image
from test_config import get_mllama_config

NUM_BENCHMARK_ITER = 10
NUM_TOKENS_TO_CHECK = 16
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dog.jpg")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# setup debug env
os.environ["XLA_FALLBACK_CPU"] = "0"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
torch_xla._XLAC._set_ir_debug(True)
set_random_seed(0)

def get_inputs(config, img, dtype):
    # inputs
    batch_size = 1
    
    # construct text input tokens
    # token ids of "<|image|>If I had to write a haiku for this one"
    input_ids = torch.tensor([[128000, 128256, 128000,   2746,    358,   1047,    311,   3350,    264,
           6520,  39342,    369,    420,    832]])
    input_ids = input_ids.to(dtype=torch.int32)
    text_token_len = input_ids.shape[1]
    attention_mask = torch.ones((batch_size, text_token_len), dtype=torch.int32)

    # HF vision model inputs
    cross_attention_mask = torch.ones([1, text_token_len, 1, 4])
    cross_attention_mask[0][0] *= 0

    aspect_ratio_ids = torch.tensor([[1]])
    aspect_ratio_mask  = torch.ones([1, 1, 4], dtype = torch.int64)

    # Neuron vision inputs
    vision_mask = create_vision_mask(input_ids, config.image_token_index)
    pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(
        config, [[img]] * batch_size, (text_token_len > 1)
    )
   
    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, vision_mask=vision_mask, aspect_ratios=aspect_ratios, num_chunks=num_chunks, has_image=has_image, aspect_ratio_ids=aspect_ratio_ids, aspect_ratio_mask=aspect_ratio_mask, cross_attention_mask=cross_attention_mask)

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
    hf_config = MllamaConfig.from_pretrained(config_path, torch_dtype=torch.float16)
    logger.info(f"HF config {to_dict(hf_config)}")
    hf_model = MllamaForConditionalGeneration._from_config(hf_config)
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

@pytest.mark.parametrize(
    "dtype, model_type, latency_threshold, throughput_threshold",
    [
        pytest.param(
            dtype, model_type, latency_threshold, throughput_threshold,
            id=f"dtype_{str(dtype).split('.')[-1]}_config_{model_type}",
        )
        for (dtype, model_type, latency_threshold, throughput_threshold) in [
            (torch.float16, "mllama", 11550*1.1, 712*0.9),
            ]
    ],
)

def test_original_cpu_vs_nxdi_neuron(dtype, model_type, latency_threshold, throughput_threshold):
    # Config
    # Avoid checkpoint name to pass IPScanner
    # Note: the config modified the original HF config "num_hidden_layers": 4 for tiny model integration test.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    '''Get reference HF CPU model'''
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    config = get_mllama_config(dtype=dtype, model_path=model_path)
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 1,
                                         eos_token_id = [2],
                                         pad_token_id=200018, 
                                         output_logits=True)

    logger.info(f"\nInferenceConfig {to_dict(config)}\n")

    '''Get Neuron Model'''
    # Compile model on Neuron
    neuron_model = NeuronMllamaForCausalLM(model_path=model_path, config=config)

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

    img = get_image(IMAGE_PATH)
    # Inputs
    inputs = get_inputs(config, img, dtype)

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
    test_original_cpu_vs_nxdi_neuron(dtype=torch.float16, model_type="mllama", latency_threshold=12704, throughput_threshold=640)