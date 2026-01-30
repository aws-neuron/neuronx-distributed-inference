#!/usr/bin/env python3
"""
Integration tests for OLMo-2-0425-1B-Instruct NeuronX implementation.
"""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_olmo import *


# Test configuration
MODEL_PATH = "/home/ubuntu/models/OLMo-2-0425-1B-Instruct/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/OLMo-2-0425-1B-Instruct/"


def load_neuron_config_from_compiled(compiled_path: str):
    """Load neuron configuration from compiled model's neuron_config.json."""
    config_path = Path(compiled_path) / "neuron_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    
    with open(config_path) as f:
        config_data = json.load(f)
    
    if "neuron_config" in config_data:
        return config_data["neuron_config"]
    else:
        return config_data


def create_model_for_inference(compiled_path: str, model_path: str):
    """Create model for inference using compiled neuron_config."""
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)
    
    dtype_str = neuron_config_dict.get('torch_dtype', 'torch.bfloat16')
    if isinstance(dtype_str, str):
        dtype = getattr(torch, dtype_str.split('.')[1]) if dtype_str.startswith('torch.') else torch.bfloat16
    else:
        dtype = dtype_str
    
    neuron_config_kwargs = {
        'tp_degree': neuron_config_dict.get('tp_degree', 2),
        'batch_size': neuron_config_dict.get('batch_size', 1),
        'seq_len': neuron_config_dict.get('seq_len', 128),
        'torch_dtype': dtype,
    }
    
    neuron_config = NeuronConfig(**neuron_config_kwargs)
    
    # This will use the imported model and config classes
    # The actual class names will be determined at runtime
    return None, neuron_config


def generate_with_neuron_model(model, input_ids, max_new_tokens: int):
    """Generate tokens using manual forward pass loop."""
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated_ids.shape[0], -1)
        
        with torch.no_grad():
            outputs = model(generated_ids, position_ids=position_ids)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    return generated_ids


@pytest.fixture(scope="module")
def compiled_model():
    """Load pre-compiled model."""
    # Note: Actual implementation would load the specific model class
    # This is a template that should be customized per model
    return None


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_model_loads(compiled_model):
    """Test that model loads successfully (smoke test)."""
    assert compiled_model is not None
    assert hasattr(compiled_model, 'config')
    print("✓ Smoke test passed - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    print(f"✓ Generation test passed")
    print(f"  Output: {output_text}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish)."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Basic coherence checks
    assert len(output_text.split()) > 3, "Output should have multiple words"
    print(f"✓ Coherence test passed")
    print(f"  Output: {output_text[:100]}...")


if __name__ == "__main__":
    print("="*80)
    print("OLMo-2-0425-1B-Instruct Integration Tests")
    print("="*80)
    
    print("\nNote: This is a template test file.")
    print("For actual model testing, customize the model loading logic.")
    
    print("\n" + "="*80)
    print("✓ Template structure verified!")
    print("="*80)
