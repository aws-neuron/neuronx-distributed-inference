from argparse import Namespace
import copy
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

DTYPE = torch.float16

@pytest.mark.tp32
@pytest.mark.mixed_precision
def test_llama3_2_1b_4layer_mixed_precision():
    # Load model from config, and save with random weights.
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.json"

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=2,
        max_context_length=128,
        seq_len=256,
        cast_type="as-declared",
        torch_dtype=DTYPE,
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    validate_accuracy(model_path, config, generation_config)
    validate_dtypes(model_path, config)

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

    model = NeuronLlamaForCausalLM(model_path, config)
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


def validate_dtypes(model_path, config):
    model = NeuronLlamaForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    # With enable_mixed_precision, weights should be casted based on the dtype the parameter was defined with
    # In the case of this model, attn + mlp should be in DTYPE and norms should be in fp32

    rank0_weights = model.context_encoding_model.model.nxd_model.weights[0]

    assert rank0_weights["layers.0.self_attn.qkv_proj.v_proj.weight"].dtype == DTYPE
    assert rank0_weights["layers.1.mlp.up_proj.weight"].dtype == DTYPE
    assert rank0_weights["layers.2.input_layernorm.weight"].dtype == torch.float32
    assert rank0_weights["lm_head.weight"].dtype == DTYPE
    assert rank0_weights["embed_tokens.weight"].dtype == DTYPE

if __name__ == "__main__":
    test_llama3_2_1b_4layer_mixed_precision()