from argparse import Namespace
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig 
from neuronx_distributed_inference.models.mistral.modeling_mistral import MistralInferenceConfig, MistralNeuronConfig, NeuronMistralForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

CONFIG_FILE = 'mistral_7b_0.1_4L_config.json'


torch.manual_seed(42)
@pytest.mark.tp32
@pytest.mark.mistral_7b_4L_windowed_context_encoding
@pytest.mark.parametrize(
    "batch_size, max_context_len, seq_len, input_len, wce_size, sliding_window",
    # fmt: off
    [
        # with sliding_window. Only supports SWA window_size = WCTE window_size as of now.
        pytest.param(1, 8, 16, [8], 4, 4, marks=pytest.mark.xfail(reason="WCTE is broken")), # input ends in last window
        pytest.param(1, 8, 16, [3], 4, 4, marks=pytest.mark.xfail(reason="WCTE is broken"))  # input ends in first window
    ],
    # fmt: on
)
def test_mistral_windowed_context_encoding(
    batch_size, max_context_len, seq_len, input_len, wce_size, sliding_window
):
    # Load model from config, and save with random weights.
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        seq_len=seq_len,
        max_length=seq_len,
        n_positions=max_context_len,
        max_context_length=max_context_len,
        torch_dtype=torch.bfloat16,
        windowed_context_encoding_size=wce_size,
    )
    
    config_path = f"{os.path.dirname(os.path.abspath(__file__))}/{CONFIG_FILE}"
    
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_config.sliding_window = sliding_window
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = MistralInferenceConfig(neuron_config, load_config=load_pretrained_config(model_path))

    num_tokens_to_generate = 2

    input_len = input_len[0]
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

    model_tempdir.cleanup()
