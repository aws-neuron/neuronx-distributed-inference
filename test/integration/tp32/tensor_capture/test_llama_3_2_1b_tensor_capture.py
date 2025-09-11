import os
import time
import pytest
import tempfile
import torch
import copy
from typing import List, Optional, Dict, Tuple

from transformers import AutoConfig, AutoModel, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig, TensorCaptureConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from torch_neuronx.testing import neuron_allclose


def create_test_inputs(config, input_len=16):
    """Create consistent test inputs for both baseline and tensor capture runs"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    input_ids = torch.rand((config.neuron_config.batch_size, input_len)) * config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    position_ids = torch.arange(input_len, dtype=torch.int32).unsqueeze(0).expand(config.neuron_config.batch_size, -1)
    
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

def run_model(
    model_path: str, 
    neuron_config: NeuronConfig, 
    tensor_capture_config: Optional[TensorCaptureConfig] = None,
    tensor_capture_dir: Optional[str] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Run a model with or without tensor capture enabled
    
    Args:
        model_path: Path to the model
        neuron_config: NeuronConfig object
        tensor_capture_config: Optional TensorCaptureConfig for enabling tensor capture
        tensor_capture_dir: Directory to save captured tensors
        
    Returns:
        Tuple containing model outputs and metadata
    """
    # Apply tensor capture config if provided
    if tensor_capture_config:
        neuron_config = copy.deepcopy(neuron_config)
        neuron_config.tensor_capture_config = tensor_capture_config
    
    # Create model config
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    
    # Create test inputs
    input_ids, attention_mask, position_ids = create_test_inputs(config)
    
    # Initialize model
    model = NeuronLlamaForCausalLM(model_path, config)
    
    # Compile and load model
    compiled_model_path = os.path.join(model_path, "compiled_checkpoint")
    if tensor_capture_config:
        compiled_model_path += "_tensor_capture"
    
    model.compile(compiled_model_path)
    model.load(compiled_model_path)
    
    # Create tensor capture hook if needed
    tensor_capture_hook = None
    if tensor_capture_config and tensor_capture_dir:
        tensor_capture_hook = get_tensor_capture_hook(
            capture_indices=[1, 5, 10],
            tensor_capture_save_dir=tensor_capture_dir,
        )
    
    # Run inference
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
        output_scores=True
    )
    execution_time = time.time() - start_time
    
    metadata = {
        "execution_time": execution_time,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "config": config,
        "model": model
    }
    
    return outputs, metadata

def assert_close_with_baseline_logits(baseline_logits, tensor_capture_logits):
    for i in range(len(baseline_logits)):
        all_close_summary = neuron_allclose(baseline_logits[i], tensor_capture_logits[i])
        assert all_close_summary.allclose

def validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir):
    """
    Validate that tensor capture worked correctly and didn't affect model outputs
    
    Args:
        baseline_outputs: Outputs from model without tensor capture
        tensor_capture_outputs: Outputs from model with tensor capture
        tensor_capture_dir: Directory where captured tensors were saved
    """
    # Verify outputs are consistent across all configurations
    print("\nVerifying output consistency:")
    outputs_match = torch.allclose(tensor_capture_outputs.sequences, baseline_outputs.sequences)
    assert outputs_match, "Outputs do not match baseline"
    print("✓ Outputs match baseline")

    # Verify output logits are consistent across all configurations
    assert_close_with_baseline_logits(baseline_outputs.scores, tensor_capture_outputs.scores)
    print("✓ Outputs logits match baseline")

    # Verify that tensor files were created
    tensor_files = [f for f in os.listdir(tensor_capture_dir) 
                    if f.startswith("captured_tensors_")]

    print(f"✓ Found {len(tensor_files)} tensor capture files")
    
    # Assert that at least one tensor file was created
    assert len(tensor_files) > 0, "No tensor capture files were created"


@pytest.mark.tp32
def test_llama_tensor_capture_modules_only():
    """Test tensor capture with only modules_to_capture specified"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create neuron config
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=1,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        neuron_config
    )
    
    # Run with tensor capture - modules only
    modules_to_capture = ['layers.0', 'layers.1', 'layers.3', 'layers.3.self_attn.qkv_proj.k_proj']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        neuron_config,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )

    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())
    
    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)

@pytest.mark.tp32
def test_llama_tensor_capture_with_inputs():
    """Test tensor capture with modules_to_capture and capture_inputs enabled"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create neuron config
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=1,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        neuron_config
    )
    
    # Run with tensor capture - modules and inputs
    modules_to_capture = ['layers.0.self_attn.rotary_emb', 'layers.0.self_attn.qkv_proj.v_proj','layers.0.mlp', 'layers.1.self_attn', 'layers.3.self_attn.qkv_proj']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture,
        capture_inputs=True
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        neuron_config,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )

    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())
    
    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)


@pytest.mark.tp32
def test_llama_tensor_capture_all_options():
    """Test tensor capture with all options enabled"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create neuron config
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=1,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        on_device_sampling_config=OnDeviceSamplingConfig(),
    )
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        neuron_config
    )
    
    # Run with tensor capture - all options
    modules_to_capture = ['layers.0.mlp', 'layers.1.self_attn', 'layers.2.self_attn.qkv_proj', 'layers.3.self_attn', 'layers.3.mlp']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture,
        capture_inputs=True,
        max_intermediate_tensors=10
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        neuron_config,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )
    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())
    
    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)

if __name__ == "__main__":
    test_llama_tensor_capture_modules_only()
    test_llama_tensor_capture_with_inputs()
    test_llama_tensor_capture_all_options()
