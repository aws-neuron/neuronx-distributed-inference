import logging
import os
import pytest
import tempfile
from argparse import Namespace

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig, GenerationConfig
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import NeuronQwen2VLTextForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.models.config import to_dict

from .test_config import get_qwen2_vl_config

NUM_TOKENS_TO_CHECK = 16
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    "dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, max_new_tokens, text_buckets",
    [
        (torch.float16, "Qwen2_VL_Text", 1, 4, 1, 4, 2560, 2560, 16, [1280, 2560])
    ]
)
def test_original_cpu_vs_nxdi_neuron(dtype, model_type, tkg_batch_size, text_tp_degree, vision_tp_degree, world_size, text_seq_length, vision_seq_len, max_new_tokens, text_buckets):
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
                                text_buckets=text_buckets
                                )
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 151643,
                                         eos_token_id = [151645],
                                         pad_token_id=151645, 
                                         output_logits=True)

    logger.info(f"\nInferenceConfig {to_dict(config)}\n")

    # Inputs
    prompt_length = 128
    input_ids = torch.randint(low=0, high=100000, size=(tkg_batch_size, prompt_length))
    input_ids = input_ids.to(dtype=torch.int32)

    attention_mask = torch.ones(size=(tkg_batch_size, prompt_length), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    '''Get Neuron Model'''
    # Compile model on Neuron
    neuron_model = NeuronQwen2VLTextForCausalLM(model_path=model_path, config=config.text_config)

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
    if tkg_batch_size == 1:
        check_accuracy_logits(
            neuron_model,
            generation_config=generation_config,
            num_tokens_to_check=NUM_TOKENS_TO_CHECK,
            inputs=inputs,
            pad_token_id=151645,
            divergence_difference_tol=0.01,
        )
    # Clean up temp dir if test pass
    model_tempdir.cleanup()
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(torch.float16, "Qwen2_VL_Text", 1, 4, 1, 4, 2560, 2560, 16, [1280,2560])
