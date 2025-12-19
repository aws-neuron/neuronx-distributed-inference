import copy
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from torch_neuronx.testing import neuron_allclose
from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
    TensorCaptureConfig,
)
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook


def create_test_inputs(config, input_len=16, seed=42):
    """Create consistent test inputs for both baseline and tensor capture runs"""
    torch.manual_seed(seed)

    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    position_ids = (
        torch.arange(input_len, dtype=torch.int32)
        .unsqueeze(0)
        .expand(config.neuron_config.batch_size, -1)
    )

    return input_ids, attention_mask, position_ids


def save_checkpoint(config_path):
    """Save a model checkpoint with random weights for testing"""
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def get_model(
    model_path: str,
    neuron_config: NeuronConfig,
    tensor_capture_config: Optional[TensorCaptureConfig] = None,
    on_cpu: bool = False,
) -> Tuple[NeuronLlamaForCausalLM, LlamaInferenceConfig]:
    """
    Initialize and compile a model

    Args:
        model_path: Path to the model
        neuron_config: NeuronConfig object
        tensor_capture_config: Optional TensorCaptureConfig for enabling tensor capture
        on_cpu: Whether to run on CPU or device

    Returns:
        Tuple containing initialized model and config
    """
    if tensor_capture_config:
        neuron_config = copy.deepcopy(neuron_config)
        neuron_config.tensor_capture_config = tensor_capture_config

    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    model = NeuronLlamaForCausalLM(model_path, config)

    if on_cpu:
        model.to_cpu()
    else:
        compiled_model_path = os.path.join(model_path, "compiled_checkpoint")
        if tensor_capture_config:
            compiled_model_path += "_tensor_capture"
        model.compile(compiled_model_path)
        model.load(compiled_model_path)

    return model, config


def run_model(
    model: NeuronLlamaForCausalLM,
    config: LlamaInferenceConfig,
    tensor_capture_dir: Optional[str] = None,
    input_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run inference with a pre-initialized model

    Args:
        model: Pre-initialized model
        config: Model configuration
        tensor_capture_dir: Directory to save captured tensors
        input_seed: Seed for input generation

    Returns:
        Tuple containing model outputs and metadata
    """
    if input_seed is not None:
        torch.manual_seed(input_seed)
        input_ids, attention_mask, position_ids = create_test_inputs(config, seed=input_seed)
    else:
        input_ids, attention_mask, position_ids = create_test_inputs(config)

    tensor_capture_hook = None
    if tensor_capture_dir:
        tensor_capture_hook = get_tensor_capture_hook(
            capture_indices=[1, 5, 10],
            tensor_capture_save_dir=tensor_capture_dir,
        )

    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(do_sample=False, pad_token_id=0, max_new_tokens=50)

    start_time = time.time()
    outputs = generation_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        generation_config=generation_config,
        tensor_capture_hook=tensor_capture_hook,
        return_dict_in_generate=True,
        output_scores=True,
    )
    execution_time = time.time() - start_time

    metadata = {
        "execution_time": execution_time,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "config": config,
        "model": model,
    }

    return outputs, metadata


def assert_close_with_baseline_logits(baseline_logits, tensor_capture_logits):
    for i in range(len(baseline_logits)):
        # TODO: Investigate the higher deviations with compiler updates
        all_close_summary = neuron_allclose(
            baseline_logits[i], tensor_capture_logits[i], rtol=5e-3, atol=1e-4
        )
        assert all_close_summary.allclose


def validate_tensor_shapes_n_values(baseline_dir, tensor_capture_dir):
    """Validate shapes of captured tensors match baseline tensors"""
    baseline_files = [f for f in os.listdir(baseline_dir) if f.startswith("captured_tensors_")]
    tc_files = [f for f in os.listdir(tensor_capture_dir) if f.startswith("captured_tensors_")]

    assert len(baseline_files) == len(
        tc_files
    ), f"File count mismatch: baseline {len(baseline_files)} vs tensor_capture {len(tc_files)}"

    for baseline_file, tc_file in zip(sorted(baseline_files), sorted(tc_files)):
        baseline_tensors = torch.load(os.path.join(baseline_dir, baseline_file), map_location="cpu")
        tc_tensors = torch.load(os.path.join(tensor_capture_dir, tc_file), map_location="cpu")

        baseline_shape = baseline_tensors.shape
        tc_shape = tc_tensors.shape
        assert (
            baseline_shape == tc_shape
        ), f"Shape mismatch for ({baseline_file}, {tc_file}): baseline {baseline_shape} vs tc {tc_shape}"
        outputs_match = torch.allclose(baseline_tensors, tc_tensors)
        assert (
            outputs_match
        ), f"(baseline {baseline_files}, neuron {tc_files}) do not match baseline"

    print(
        f"✓ Validated tensor shapes match between baseline and tensor capture ({len(baseline_files)} files)"
    )


def validate_tensor_capture(
    baseline_outputs, tensor_capture_outputs, baseline_dir, tensor_capture_dir
):
    """
    Validate that tensor capture worked correctly and didn't affect model outputs
    """
    print("\nVerifying output consistency:")
    outputs_match = torch.allclose(tensor_capture_outputs.sequences, baseline_outputs.sequences)
    assert outputs_match, "Outputs do not match baseline"
    print("✓ Outputs match baseline")

    assert_close_with_baseline_logits(baseline_outputs.scores, tensor_capture_outputs.scores)
    print("✓ Outputs logits match baseline")

    baseline_files = [f for f in os.listdir(baseline_dir) if f.startswith("captured_tensors_")]
    tensor_files = [f for f in os.listdir(tensor_capture_dir) if f.startswith("captured_tensors_")]

    print(f"✓ Found {len(tensor_files)} tensor capture files")
    print(f"✓ Found {len(baseline_files)} baseline tensor capture files")

    assert len(tensor_files) > 0, "No tensor capture files were created"
    assert len(baseline_files) > 0, "No tensor capture files were created"

    validate_tensor_shapes_n_values(baseline_dir, tensor_capture_dir)


def run_tensor_capture_test(
    test_name: str,
    modules_to_capture: List[str],
    capture_inputs: bool = False,
    max_intermediate_tensors: Optional[int] = None,
    on_cpu: bool = False,
    batch_size: int = 1,
):
    """Generic test runner for tensor capture tests"""
    config_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
    )
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    neuron_config = NeuronConfig(
        tp_degree=1 if on_cpu else 32,
        batch_size=batch_size,
        max_context_length=256 if on_cpu else 128,
        seq_len=256,
        output_logits=True,
        on_cpu=on_cpu,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )

    tensor_capture_dir = tempfile.TemporaryDirectory().name
    baseline_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture,
        capture_inputs=capture_inputs,
        max_intermediate_tensors=max_intermediate_tensors,
    )

    baseline_model, baseline_config = get_model(
        model_path, neuron_config, tensor_capture_config, on_cpu=on_cpu
    )
    baseline_outputs, baseline_metadata = run_model(baseline_model, baseline_config, baseline_dir)

    tc_model, tc_config = get_model(
        model_path, neuron_config, tensor_capture_config=tensor_capture_config, on_cpu=on_cpu
    )
    tensor_capture_outputs, tc_metadata = run_model(
        tc_model, tc_config, tensor_capture_dir=tensor_capture_dir
    )

    validate_tensor_capture(
        baseline_outputs, tensor_capture_outputs, baseline_dir, tensor_capture_dir
    )


@pytest.mark.tp32
def test_llama_tensor_capture_modules_only():
    """Test tensor capture with only modules_to_capture specified"""
    modules_to_capture = ["layers.0", "layers.1", "layers.3", "layers.3.self_attn.qkv_proj.k_proj"]
    run_tensor_capture_test("Device - Modules Only", modules_to_capture)


@pytest.mark.tp32
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_llama_tensor_capture_with_inputs_batch_sizes(batch_size):
    """Test tensor capture with different batch sizes"""
    modules_to_capture = [
        "layers.0.self_attn.rotary_emb",
        "layers.0.self_attn.qkv_proj.v_proj",
        "layers.0.mlp",
        "layers.1.self_attn",
        "layers.3.self_attn.qkv_proj",
    ]
    run_tensor_capture_test(
        f"Device - With Inputs - Batch {batch_size}",
        modules_to_capture,
        capture_inputs=True,
        batch_size=batch_size,
    )


@pytest.mark.tp32
def test_llama_tensor_capture_all_options():
    """Test tensor capture with all options enabled"""
    modules_to_capture = [
        "layers.0.mlp",
        "layers.1.self_attn",
        "layers.2.self_attn.qkv_proj",
        "layers.3.self_attn",
        "layers.3.mlp",
    ]
    run_tensor_capture_test(
        "Device - All Options", modules_to_capture, capture_inputs=True, max_intermediate_tensors=10
    )


def run_multiple_prompts_test(on_cpu: bool = False):
    """Common test logic for multiple prompts tensor capture test"""
    config_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
    )
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    neuron_config = NeuronConfig(
        tp_degree=1 if on_cpu else 32,
        batch_size=1,
        max_context_length=256 if on_cpu else 128,
        seq_len=256,
        output_logits=True,
        on_cpu=on_cpu,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )

    modules_to_capture = ["layers.0.mlp", "layers.1.self_attn", "layers.2.self_attn.qkv_proj"]
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture, capture_inputs=True, max_intermediate_tensors=10
    )

    # Initialize and compile model once
    model, config = get_model(model_path, neuron_config, tensor_capture_config, on_cpu)
    print("Model compiled and loaded once. Running inference with 2 different prompts...")

    # Create two different tensor capture directories
    tensor_capture_dir_1 = tempfile.TemporaryDirectory().name
    tensor_capture_dir_2 = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir_1, exist_ok=True)
    os.makedirs(tensor_capture_dir_2, exist_ok=True)

    print(f"Tensor capture directory 1: {tensor_capture_dir_1}")
    print(f"Tensor capture directory 2: {tensor_capture_dir_2}")

    # Run inference with first prompt (seed=100)
    print("\n=== Running inference with first prompt ===")
    outputs_1, metadata_1 = run_model(
        model=model, config=config, tensor_capture_dir=tensor_capture_dir_1, input_seed=100
    )
    # reset index for next generation
    model.set_tensor_capture_step(step=0)

    # Run inference with second prompt (seed=200)
    print("\n=== Running inference with second prompt ===")
    outputs_2, metadata_2 = run_model(
        model=model, config=config, tensor_capture_dir=tensor_capture_dir_2, input_seed=100
    )

    # Validate that both runs produced different outputs (due to different inputs)
    print("\n=== Validating results ===")
    validate_tensor_capture(
        outputs_1, outputs_2, tensor_capture_dir_1, tensor_capture_dir_2
    )

    return outputs_1, outputs_2, metadata_1, metadata_2


@pytest.mark.tp32
def test_llama_tensor_capture_multiple_prompts():
    """Test tensor capture with 2 different prompts dumping to different folders without recompilation"""
    return run_multiple_prompts_test(on_cpu=False)


@pytest.mark.tp32
@pytest.mark.cpu
def test_llama_tensor_capture_multiple_prompts_cpu():
    """Test tensor capture with 2 different prompts dumping to different folders without recompilation on CPU"""
    return run_multiple_prompts_test(on_cpu=True)


@pytest.mark.tp32
@pytest.mark.cpu
def test_llama_tensor_capture_modules_only_cpu():
    """Test tensor capture with only modules_to_capture specified on CPU"""
    modules_to_capture = ["layers.0", "layers.1", "layers.3", "layers.3.self_attn.qkv_proj.k_proj"]
    run_tensor_capture_test("CPU - Modules Only", modules_to_capture, on_cpu=True)


@pytest.mark.tp32
@pytest.mark.cpu
def test_llama_tensor_capture_with_inputs_cpu():
    """Test tensor capture with modules_to_capture and capture_inputs enabled on CPU"""
    modules_to_capture = [
        "layers.0.self_attn.rotary_emb",
        "layers.0.self_attn.qkv_proj.v_proj",
        "layers.0.mlp",
        "layers.1.self_attn",
        "layers.3.self_attn.qkv_proj",
    ]
    run_tensor_capture_test(
        "CPU - With Inputs", modules_to_capture, capture_inputs=True, on_cpu=True
    )


@pytest.mark.tp32
@pytest.mark.cpu
def test_llama_tensor_capture_all_options_cpu():
    """Test tensor capture with all options enabled on CPU"""
    modules_to_capture = [
        "layers.0.mlp",
        "layers.1.self_attn",
        "layers.2.self_attn.qkv_proj",
        "layers.3.self_attn",
        "layers.3.mlp",
    ]
    run_tensor_capture_test(
        "CPU - All Options",
        modules_to_capture,
        capture_inputs=True,
        max_intermediate_tensors=10,
        on_cpu=True,
    )


if __name__ == "__main__":
    test_llama_tensor_capture_modules_only()
    test_llama_tensor_capture_with_inputs_batch_sizes(1)
    test_llama_tensor_capture_all_options()
    test_llama_tensor_capture_modules_only_cpu()
    test_llama_tensor_capture_with_inputs_cpu()
    test_llama_tensor_capture_all_options_cpu()
    test_llama_tensor_capture_multiple_prompts()
    test_llama_tensor_capture_multiple_prompts_cpu()
