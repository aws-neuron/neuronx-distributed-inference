from argparse import Namespace
import os
import pytest
import tempfile
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig 
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM, LlamaInferenceConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

CONFIG_FILE = 'config_windowed_context_encoding.json'


torch.manual_seed(42)
@pytest.mark.tp32
@pytest.mark.llama3_1_8b_windowed_context_encoding
@pytest.mark.parametrize(
    "batch_size, max_context_len, seq_len, input_len, wce_size, sliding_window, enable_sp",
    # fmt: off
    # TODO: xfailing WCTE tests as they are failing.
    [
        # bs = 1
        pytest.param(1, 8, 32, [5], 4, None, False, marks=pytest.mark.xfail(reason="WCTE is broken")),  # input_len is less than 1 window
        pytest.param(1, 16, 32, [4], 4, None, False, marks=pytest.mark.xfail(reason="WCTE is broken")),  # input_len is exactly 1 window
        pytest.param(1, 16, 32, [5], 4, None, False, marks=pytest.mark.xfail(reason="WCTE is broken")),  # input_len needs 2 windows to cover
        pytest.param(1, 16, 32, [9], 4, None, False, marks=pytest.mark.xfail(reason="WCTE is broken")),  # input_len needs 3 windows to cover

        # bs > 1, where each request requires different # of windows to cover
        pytest.param(2, 8, 32, [3,5], 4, None, False, marks=pytest.mark.xfail(reason="WCTE is broken")), # first request requires one window, second request requires two windows

        # longer context len - 512k, input ends in second window
        pytest.param(1, 512*1024, 513*1024, [64*1024], 32*1024, 32*1024, False, marks=pytest.mark.xfail(reason="WCTE is broken")),

        # with SP
        pytest.param(1, 128, 128, [126], 64, None, True, marks=pytest.mark.xfail(reason="WCTE is broken")),
    ],
    # fmt: on
)
def test_llama3_1_8b_windowed_context_encoding(
    batch_size, max_context_len, seq_len, input_len, wce_size, sliding_window, enable_sp
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
        attn_kernel_enabled=True,
        sequence_parallel_enabled=enable_sp,
    )
    
    config_path = f"{os.path.dirname(os.path.abspath(__file__))}/{CONFIG_FILE}"
    
    hf_config = AutoConfig.from_pretrained(config_path)
    if sliding_window:
        hf_config.sliding_window = sliding_window
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    config = LlamaInferenceConfig(neuron_config, load_config=load_pretrained_config(model_path))

    num_tokens_to_generate = 10
    if len(input_len) == 1 and max_context_len < 512*1024:
        input_len = input_len[0]
        input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
        input_ids = input_ids.to(dtype=torch.int32)
        attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
        inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

        model = NeuronLlamaForCausalLM(model_path, config)
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        model.compile(compiled_model_path, True)
        model.load(compiled_model_path)

        check_accuracy_logits(
            model,
            generation_config=generation_config,
            num_tokens_to_check=num_tokens_to_generate,
            inputs=inputs,
        )
    else: 
        # for bs >= 2, to get the correct the HF CPU logits we need to left-pad. 
        #   However, WCTE as of now doesn't support left-padded inputs.
        #   As a temporary soln, we compare the logits with WCTE turned on Vs.
        #   WCTE turned off for bs >= 2.

        # for long context SWA, we also take this flow because the only HF model 
        #   that produces correct gold SWA cpu logits is Mistral. However, 512k 
        #   seq_len for our Mistral causes an OOM. To fix this, we must update
        #   modeling_mistral to use all the features that modeling_llama uses.

        inputs = construct_inputs(max_context_len, input_len)

        # 1. Get expected logits with windowed context encoding turned on
        neuron_model = NeuronLlamaForCausalLM(model_path, config)
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        neuron_model.compile(compiled_model_path, True)
        neuron_model.load(compiled_model_path)
        model = HuggingFaceGenerationAdapter(neuron_model)
        model_outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=num_tokens_to_generate,
            min_new_tokens=num_tokens_to_generate,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )
        expected_logits = torch.stack(model_outputs.scores)

        # 2. Compare against model with windowed context encoding turned off
        config.windowed_context_encoding_size = None
        model = NeuronLlamaForCausalLM(model_path, config)
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        model.compile(compiled_model_path, True)
        model.load(compiled_model_path)

        check_accuracy_logits(
            model,
            generation_config=generation_config,
            num_tokens_to_check=num_tokens_to_generate,
            inputs=inputs,
            expected_logits=expected_logits,
        )

    model_tempdir.cleanup()


def construct_inputs(max_context_len, input_len):
    pad_token_id = 0
    input_ids_list = []
    attention_mask_list = []
    for length in input_len:
        tokens = torch.randint(5, 100, (length,), dtype=torch.int32)
        padded = torch.cat([
            tokens,
            torch.full((max_context_len - length,), pad_token_id, dtype=torch.int32)
        ])
        mask = torch.cat([
            torch.ones(length, dtype=torch.int32),
            torch.zeros(max_context_len - length, dtype=torch.int32)
        ])
        input_ids_list.append(padded)
        attention_mask_list.append(mask)

    inputs = Namespace(input_ids=torch.stack(input_ids_list),
                       attention_mask=torch.stack(attention_mask_list))
    return inputs
