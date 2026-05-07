#!/usr/bin/env python3
"""
Integration tests for CTRL NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.
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
from modeling_ctrl import NeuronCTRLForCausalLM, CTRLInferenceConfig


# Test configuration
MODEL_PATH = "/shared/dhwanw2/models/ctrl"
COMPILED_MODEL_PATH = "/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/ctrl/compiled_model"


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
    """Create model for inference using the exact pattern from validate_model.py."""
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)

    dtype_str = neuron_config_dict.get('torch_dtype', 'torch.bfloat16')
    if isinstance(dtype_str, str):
        dtype = getattr(torch, dtype_str.split('.')[1]) if dtype_str.startswith('torch.') else torch.bfloat16
    else:
        dtype = dtype_str

    neuron_config_kwargs = {
        'tp_degree': neuron_config_dict.get('tp_degree', 1),
        'batch_size': neuron_config_dict.get('batch_size', 1),
        'seq_len': neuron_config_dict.get('seq_len', 128),
        'torch_dtype': dtype,
        'save_sharded_checkpoint': neuron_config_dict.get('save_sharded_checkpoint', True),
        'on_cpu': neuron_config_dict.get('on_cpu', False),
    }

    optional_params = ['world_size', 'max_context_length', 'enable_bucketing']
    for param in optional_params:
        if param in neuron_config_dict:
            neuron_config_kwargs[param] = neuron_config_dict[param]

    if 'max_context_length' not in neuron_config_kwargs:
        neuron_config_kwargs['max_context_length'] = neuron_config_kwargs['seq_len']

    neuron_config = NeuronConfig(**neuron_config_kwargs)

    model_config = CTRLInferenceConfig.from_pretrained(
        model_path, neuron_config=neuron_config,
    )

    model = NeuronCTRLForCausalLM(model_path, model_config)
    return model, neuron_config


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
    """Compile and load model."""
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")

        neuron_config = NeuronConfig(
            tp_degree=1,
            batch_size=1,
            seq_len=128,
            max_context_length=128,
            torch_dtype=torch.bfloat16,
        )

        config = CTRLInferenceConfig.from_pretrained(
            MODEL_PATH, neuron_config=neuron_config,
        )

        model = NeuronCTRLForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)

    return model


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
    assert hasattr(compiled_model.config, 'neuron_config')
    print("Smoke test passed - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "The theory of general relativity"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    print(f"Generation test passed")
    print(f"  Output: {output_text}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish)."""
    prompt = "In the year 2025, artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text.split()) > 5, "Output should have multiple words"
    assert not _is_repetitive(output_text), "Output should not be repetitive"

    print(f"Coherence test passed")
    print(f"  Output: {output_text[:100]}...")


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False

    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i+j] == word for j in range(max_repeat)):
            return True

    return False


if __name__ == "__main__":
    print("=" * 80)
    print("CTRL Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")

        neuron_config = NeuronConfig(
            tp_degree=1,
            batch_size=1,
            seq_len=128,
            max_context_length=128,
            torch_dtype=torch.bfloat16,
        )

        config = CTRLInferenceConfig.from_pretrained(
            MODEL_PATH, neuron_config=neuron_config,
        )

        model = NeuronCTRLForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Generation Test...")
    test_model_generates(model, tokenizer)

    print("\n3. Coherence Test...")
    test_output_coherence(model, tokenizer)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
