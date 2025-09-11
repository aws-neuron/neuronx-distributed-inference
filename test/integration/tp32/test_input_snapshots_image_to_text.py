import os
import pickle
import pytest
import tempfile
import torch
import uuid

from argparse import Namespace

from transformers import GenerationConfig, LlavaConfig, LlavaForConditionalGeneration

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import NeuronPixtralForCausalLM, PixtralInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.utils.snapshot import SnapshotOutputFormat


@pytest.mark.key_config_test
def test_input_snapshots(monkeypatch):
    # Set compiler workdir for the test.
    compiler_workdir = os.path.join(os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model"), str(uuid.uuid4()))
    monkeypatch.setenv("BASE_COMPILE_WORK_DIR", compiler_workdir)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/pixtral/config_tiny_pixtral.json")
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    model = setup_model(model_path)

    model.register_snapshot_hooks(
        output_path=compiler_workdir,
        output_format=SnapshotOutputFormat.NUMPY_PICKLE,
        capture_at_requests=[0, 1],
    )

    inputs = create_inputs(model.config)
    run_generation(model, inputs)

    validate_input_snapshots(compiler_workdir)


def validate_input_snapshots(output_path):
    # Basic validation check for expected number of input tensors and total element counts.
    input_snapshot_paths = [
        (f"{output_path}/text_model/context_encoding_model/_tp0_bk1/request0/inp-000.p", 51, 3263488),
        (f"{output_path}/text_model/token_generation_model/_tp0_bk1/request0/inp-000.p", 50, 3197442),
        (f"{output_path}/text_model/token_generation_model/_tp0_bk1/request1/inp-000.p", 50, 3197442),
        (f"{output_path}/vision_model/vision_encoder_model/_tp0_bk0/request0/inp-000.p", 43, 399872),
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
    hf_config = LlavaConfig.from_pretrained(config_path, torch_dtype=torch.bfloat16)
    hf_model = LlavaForConditionalGeneration._from_config(hf_config)
 
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def setup_model(model_path):
    compiled_model_path = os.path.join(model_path, "compiled_model")
    
    text_neuron_config = NeuronConfig(
        tp_degree=16,
        batch_size=1,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        enable_bucketing=True,
        fused_shared_experts=True,
        early_expert_affinity_modulation=True,
        disable_normalize_top_k_affinities=True,
        cast_type="as-declared",
    )
    vision_neuron_config = NeuronConfig(
        tp_degree=16,
        batch_size=1,
        seq_len=256,
        torch_dtype=torch.bfloat16,
        enable_bucketing=True,
        cast_type="as-declared",
    )
    config = PixtralInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    model = NeuronPixtralForCausalLM(model_path, config)
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    return model


def create_inputs(config):
    # inputs
    batch_size = 1
    num_channels = 3
    text_token_len = 16
    image_size = 64
    vision_token_len = 128
    total_input_len = text_token_len + vision_token_len
    # construct text input tokens
    text_input_ids = torch.rand((config.neuron_config.batch_size, text_token_len)) * config.text_config.vocab_size
    # construct vision input tokens
    vision_input_ids = torch.full([config.neuron_config.batch_size, vision_token_len], fill_value=config.image_token_index)
    # assume vision tokens are before text tokens
    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1)
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, total_input_len), dtype=torch.int32)
    # vision inputs
    pixel_values = torch.ones([batch_size, num_channels, image_size, image_size])

    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)


def run_generation(model, inputs):
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    generation_model = HuggingFaceGenerationAdapter(model)
    num_new_tokens = model.config.neuron_config.seq_len - inputs.input_ids.shape[1]
    outputs = generation_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        min_new_tokens=num_new_tokens,
        max_new_tokens=num_new_tokens,
        generation_config=generation_config,
    )
    assert outputs.shape == (model.config.neuron_config.batch_size, model.config.neuron_config.seq_len)
