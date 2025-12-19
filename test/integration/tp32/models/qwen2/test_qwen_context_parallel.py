from argparse import Namespace
import copy
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import Qwen2InferenceConfig, NeuronQwen2ForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

sp_qkv_neuron_config = NeuronConfig(
    tp_degree=32,
    cp_degree=4,
    batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=True,
    fused_qkv=True,
    torch_dtype=torch.float32,
)

sp_disabled_neuron_config = NeuronConfig(
    tp_degree=32,
    cp_degree=4,
    batch_size=1,
    max_context_length=256,
    seq_len=256,
    sequence_parallel_enabled=False,
    torch_dtype=torch.float32,
)

@pytest.mark.tp32
@pytest.mark.context_parallel
@pytest.mark.parametrize(
    "neuron_config",
    # fmt: off
    [
        (sp_qkv_neuron_config),
        (sp_disabled_neuron_config),
    ],
    # fmt: on
)
def test_qwen2_4layer_context_parallel(neuron_config):
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = Qwen2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    validate_accuracy(model_path, config, generation_config)

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)

    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

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

    model = NeuronQwen2ForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        prompt=TEST_PROMPT,
        num_tokens_to_check=config.neuron_config.max_context_length - input_len,
        inputs=inputs,
    )