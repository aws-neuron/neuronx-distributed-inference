from argparse import Namespace
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


@pytest.mark.tp32
def test_1_layer_accuracy():
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    neuron_config = MoENeuronConfig(
        tp_degree=32,
        batch_size=1,
        max_context_length=128,
        seq_len=256,
    )
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    validate_accuracy(model_path, config, generation_config)

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config):
    input_len = 16
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronQwen3MoeForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        # Logits matching for longer sequence length will fail, most likely because
        # experts weights are too close and the router could select different
        # experts because of numeric error.
        num_tokens_to_check=16,
        inputs=inputs,
    )


if __name__ == "__main__":
    test_1_layer_accuracy()