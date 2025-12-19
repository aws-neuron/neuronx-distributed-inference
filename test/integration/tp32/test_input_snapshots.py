import os
import pickle
import tempfile
import pytest
import torch
import uuid

from argparse import Namespace

from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.utils.snapshot import SnapshotOutputFormat


@pytest.mark.key_config_test
@pytest.mark.parametrize(
    "async_mode, capture_config",
    [
        (False, 'request'),
        (True, 'request'),
        (False, 'token'),
        (True, 'token')
    ]
)
def test_input_snapshots(monkeypatch, async_mode, capture_config):
    # Set compiler workdir for the test.
    compiler_workdir = os.path.join(os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model"), str(uuid.uuid4()))
    monkeypatch.setenv("BASE_COMPILE_WORK_DIR", compiler_workdir)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/llama/llama3.2/1b/config.json")
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    model = setup_model(model_path, async_mode)
    input_length = 16

    capture_at_requests = []
    capture_for_tokens = []
    if capture_config == 'request':
        capture_at_requests = [0, 1]
    elif capture_config == 'token':
        capture_for_tokens = [[input_length + 1, input_length + 2] for _ in range(model.neuron_config.batch_size)]

    model.register_snapshot_hooks(
        output_path=compiler_workdir,
        output_format=SnapshotOutputFormat.NUMPY_PICKLE,
        capture_at_requests=capture_at_requests,
        capture_for_tokens=capture_for_tokens
    )

    inputs = create_inputs(model.config, input_len=input_length)
    run_generation(model, inputs)

    validate_input_snapshots(compiler_workdir, capture_config)


def validate_input_snapshots(output_path, capture_config):
    # Basic validation check for expected number of input tensors and total element counts.
    if capture_config == 'request':
        input_snapshot_paths = [
            (f"{output_path}/context_encoding_model/_tp0_bk0/request0/inp-000.p", 49, 2128384),
            (f"{output_path}/token_generation_model/_tp0_bk0/request0/inp-000.p", 50, 2128258),
            (f"{output_path}/token_generation_model/_tp0_bk0/request1/inp-000.p", 50, 2128258),
            (f"{output_path}/token_generation_model/_tp0_bk1/request0/inp-000.p", 50, 2128386),
            (f"{output_path}/token_generation_model/_tp0_bk1/request1/inp-000.p", 50, 2128386),
        ]
    else:
        input_snapshot_paths = [
            (f"{output_path}/token_generation_model/_tp0_bk0/batch0_token17/inp-000.p", 50, 2128258),
            (f"{output_path}/token_generation_model/_tp0_bk0/batch0_token18/inp-000.p", 50, 2128258),
        ]
    for path, expected_num_inputs, expected_total_size in input_snapshot_paths:
        input_snapshot = load_pickle(path)
        actual_num_inputs = len(input_snapshot)
        actual_total_size = sum(input.size for input in input_snapshot.values())
        assert actual_num_inputs == expected_num_inputs
        assert actual_total_size == expected_total_size


def load_pickle(path):
    assert os.path.exists(path), f"Pickle file does not exist: {path}"
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)
 
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def setup_model(model_path, async_mode):
    compiled_model_path = os.path.join(model_path, "compiled_model")
    
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=1,
        max_context_length=128,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(),
        enable_bucketing=True,
        async_mode=async_mode,
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    return model


def create_inputs(config, input_len):
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    return Namespace(input_ids=input_ids, attention_mask=attention_mask)


def run_generation(model, inputs):
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    generation_model = HuggingFaceGenerationAdapter(model)
    num_new_tokens = model.config.neuron_config.seq_len - inputs.input_ids.shape[1]
    outputs = generation_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        min_new_tokens=num_new_tokens,
        max_new_tokens=num_new_tokens,
        generation_config=generation_config,
    )
    assert outputs.shape == (model.config.neuron_config.batch_size, model.config.neuron_config.seq_len)
