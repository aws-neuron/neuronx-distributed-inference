import logging
import os
import pytest
import tempfile
from argparse import Namespace

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig, GenerationConfig

from neuronx_distributed_inference.utils.accuracy import (
    generate_expected_logits,
    check_accuracy_logits_v2,
)
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.models.config import to_dict
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import _rand_interval

from .test_config import get_qwen2_vl_config


NUM_OF_IMAGES = 50
NUM_TOKENS_TO_CHECK = 16
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
set_random_seed(0)


def get_inputs(config, dtype):
    # Inputs
    prompt_length = 128
    vision_token_len = 253 * NUM_OF_IMAGES
    text_input_ids = torch.randint(low=0, high=100000, size=(config.neuron_config.batch_size, prompt_length))
    vision_input_ids = torch.full([config.neuron_config.batch_size, vision_token_len], fill_value=config.image_token_id)

    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1)
    input_ids = input_ids.to(dtype=torch.int64)  # default long tensor

    total_input_len = prompt_length + vision_token_len
    attention_mask = torch.ones((config.neuron_config.batch_size, total_input_len),dtype=torch.int64)  # default long tensor

    # vision inputs
    pixel_values_shape = [NUM_OF_IMAGES * 1012, 1176]
    pixel_values = torch.nn.Parameter(
        _rand_interval(-1, 1, dtype, *pixel_values_shape)
    )
    image_grid_thw = torch.tensor([[1, 22, 46]] * NUM_OF_IMAGES)

    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

def save_checkpoint(config_path):
    hf_config = Qwen2VLConfig.from_pretrained(config_path, torch_dtype=torch.float16)
    logger.info(f"HF config {to_dict(hf_config)}")
    hf_model = Qwen2VLForConditionalGeneration._from_config(hf_config)
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

@pytest.mark.parametrize(
    "dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, max_new_tokens",
    [
        pytest.param(torch.float16, "Qwen2_VL_Vision_Text", 1, 4, 4, 4, 256*(NUM_OF_IMAGES+1), 1012*NUM_OF_IMAGES, 256),
    ],
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, max_new_tokens):
    # Config
    # Avoid checkpoint name to pass IPScanner
    # Note: the config modified the original HF config "num_hidden_layers": 4 for tiny model integration test.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")
    '''Get reference HF CPU model'''
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    config = get_qwen2_vl_config(dtype=dtype,
                                model_path=model_path,
                                tkg_batch_size=tkg_batch_size,
                                text_tp_degree=text_tp_degree,
                                vision_tp_degree=vision_tp_degree,
                                world_size=world_size,
                                text_seq_length=text_seq_length,
                                vision_seq_len=vision_seq_len,
                                max_new_tokens=max_new_tokens,
                                text_buckets = [text_seq_length],
                                vision_buckets= [NUM_OF_IMAGES],
                                )
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 151643,
                                         eos_token_id = [151645],
                                         pad_token_id=151645, 
                                         output_logits=True)

    logger.info(f"\nInferenceConfig {to_dict(config)}\n")

    # Inputs
    inputs = get_inputs(config, dtype)

    '''Get Neuron Model'''
    # Compile model on Neuron
    neuron_model = NeuronQwen2VLForCausalLM(model_path=model_path, config=config)

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

    # Text-only validations
    expected_logits = generate_expected_logits(
        neuron_model,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config,
        NUM_TOKENS_TO_CHECK,
    )

    check_accuracy_logits_v2(
        neuron_model,
        expected_logits,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=NUM_TOKENS_TO_CHECK,
        divergence_difference_tol=0.02,
    )

    # Text + Image validations
    additional_hf_input_args = {
        "pixel_values": inputs.pixel_values.to(torch.float64),
        "image_grid_thw": inputs.image_grid_thw.to(torch.int64)
    }
    expected_logits = generate_expected_logits(
        neuron_model,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config,
        NUM_TOKENS_TO_CHECK,
        additional_hf_input_args,
    )

    additional_neuron_input_args = {
        "pixel_values": inputs.pixel_values,
        "image_grid_thw": inputs.image_grid_thw
    }
    check_accuracy_logits_v2(
        neuron_model,
        expected_logits,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=NUM_TOKENS_TO_CHECK,
        divergence_difference_tol=0.02,
        additional_input_args=additional_neuron_input_args,
    )
    # Clean up temp dir if test pass
    model_tempdir.cleanup()
    return

if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(torch.float16, "Qwen2_VL_Vision_Text", 1, 4, 4, 4, 256*(NUM_OF_IMAGES+1), 1012*NUM_OF_IMAGES, 256)