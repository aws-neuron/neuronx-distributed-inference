#!/usr/bin/env python3
"""
Integration tests for Janus-1.3B NeuronX implementation.
"""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_janus import NeuronJanusForCausalLM, JanusInferenceConfig

# Test configuration
MODEL_PATH = "/home/ubuntu/models/Janus-1.3B/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/Janus-1.3B/"

# Copy helper functions from validated models
def load_neuron_config_from_compiled(compiled_path: str):
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data.get("neuron_config", config_data)

def generate_with_neuron_model(model, input_ids, max_new_tokens: int):
    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated_ids.shape[0], -1)
        with torch.no_grad():
            outputs = model(generated_ids, position_ids=position_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else (outputs[0] if isinstance(outputs, tuple) else outputs)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    return generated_ids

@pytest.fixture(scope="module")
def compiled_model():
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        neuron_config = NeuronConfig(tp_degree=1, batch_size=1, seq_len=128, torch_dtype=torch.bfloat16)
        config = JanusInferenceConfig(neuron_config, load_config=load_pretrained_config(MODEL_PATH))
        model = NeuronJanusForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
    
    neuron_config_dict = load_neuron_config_from_compiled(COMPILED_MODEL_PATH)
    dtype = getattr(torch, neuron_config_dict['torch_dtype'].split('.')[1]) if isinstance(neuron_config_dict['torch_dtype'], str) else neuron_config_dict['torch_dtype']
    neuron_config = NeuronConfig(tp_degree=neuron_config_dict['tp_degree'], batch_size=neuron_config_dict['batch_size'], seq_len=neuron_config_dict['seq_len'], torch_dtype=dtype)
    
    try:
        model_config = JanusInferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
    except:
        model_config = JanusInferenceConfig(neuron_config, load_config=load_pretrained_config(MODEL_PATH))
    
    try:
        model = NeuronJanusForCausalLM.from_pretrained(COMPILED_MODEL_PATH, config=model_config)
    except:
        model = NeuronJanusForCausalLM(MODEL_PATH, model_config)
    
    model.load(COMPILED_MODEL_PATH)
    return model

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)

def test_model_loads(compiled_model):
    assert compiled_model is not None
    print("✓ Smoke test passed")

def test_model_generates(compiled_model, tokenizer):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    assert len(output_text) > len(prompt)
    print(f"✓ Generation test passed: {output_text}")

if __name__ == "__main__":
    print("Janus-1.3B Integration Tests")
    print("="*80)
    # Run tests...
