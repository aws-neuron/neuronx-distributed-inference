from argparse import Namespace
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.mistral.modeling_mistral import MistralInferenceConfig, MistralNeuronConfig, NeuronMistralForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

CONFIG_FILE = 'mistral_7b_0.1_4L_config.json'


torch.manual_seed(42)
@pytest.mark.tp32
@pytest.mark.windowed_attention
@pytest.mark.parametrize(
    "batch_size, seq_len, sliding_window, input_len, num_tokens_to_generate, cp_degree",
    # fmt: off
    [
        # torch-flow (seq_len < 512)
        (1, 16, 4, 2, 6, 1),   # bs=1, input_len < sliding_window
        (1, 16, 4, 6, 8, 1),   # bs=1, input_len > sliding_window
        (2, 16, 4, 6, 10, 1),  # bs=2
        (1, 16, 4, 2, 6, 4),   # bs=1, input_len < sliding_window, cp enabled
        (1, 16, 4, 6, 8, 4),   # bs=1, input_len > sliding_window, cp enabled
        (2, 4096, 1024, 2048, 128, 4),  # bs=2, longer seq_len and generation, cp_enabled

        # nki-flow (seq_len >= 512)
        (1, 1024, 512, 512, 6, 1),   # bs=1, seq_len < 2048, use smaller tile size
        (1, 2048, 1024, 1022, 6, 1),   # bs=1, input_len < sliding_window
        (1, 2048, 1024, 1026, 8, 1),   # bs=1, input_len > sliding_window
        (2, 2048, 1024, 1022, 8, 1),  # bs=2
        (2, 4096, 1024, 2048, 128, 1),  # bs=2, longer seq_len and generation
    ],
    # fmt: on
)
def test_mistral_7b_4layer(
    batch_size, seq_len, sliding_window, input_len, num_tokens_to_generate, cp_degree
):
    # Load model from config, and save with random weights.
    neuron_config = MistralNeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        seq_len=seq_len,
        cp_degree=cp_degree,
        torch_dtype=torch.float32,
    )
    
    config_path = f"{os.path.dirname(os.path.abspath(__file__))}/{CONFIG_FILE}"
    
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_config.sliding_window=sliding_window
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = MistralInferenceConfig( neuron_config, load_config=load_pretrained_config(model_path))

    validate_accuracy(model_path, config, generation_config, input_len, num_tokens_to_generate)

    model_tempdir.cleanup()


def validate_accuracy(model_path, config, generation_config, input_len, num_tokens_to_generate):
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronMistralForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path, True)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        num_tokens_to_check=num_tokens_to_generate,
        inputs=inputs,
    )
