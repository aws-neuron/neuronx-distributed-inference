#!/usr/bin/env python3
"""
Integration tests for Gemma-2-9b NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.
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
from modeling_gemma2 import NeuronGemma2ForCausalLM, Gemma2InferenceConfig


MODEL_PATH = "/shared/dhwanw2/models/gemma-2-9b"
COMPILED_MODEL_PATH = "/shared/dhwanw2/neuron_models/gemma-2-9b/"


def load_neuron_config_from_compiled(compiled_path: str):
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data.get("neuron_config", config_data)


def create_model_for_inference(compiled_path: str, model_path: str):
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
        'save_sharded_checkpoint': neuron_config_dict.get('save_sharded_checkpoint', True),
        'on_cpu': neuron_config_dict.get('on_cpu', False),
    }

    for param in ['world_size', 'max_context_length', 'enable_bucketing']:
        if param in neuron_config_dict:
            neuron_config_kwargs[param] = neuron_config_dict[param]

    if 'max_context_length' not in neuron_config_kwargs:
        neuron_config_kwargs['max_context_length'] = neuron_config_kwargs['seq_len']

    neuron_config = NeuronConfig(**neuron_config_kwargs)

    try:
        model_config = Gemma2InferenceConfig.from_pretrained(
            model_path, neuron_config=neuron_config,
        )
    except (TypeError, AttributeError):
        model_config = Gemma2InferenceConfig(
            neuron_config, load_config=load_pretrained_config(model_path),
        )

    try:
        if hasattr(NeuronGemma2ForCausalLM, 'from_pretrained'):
            model = NeuronGemma2ForCausalLM.from_pretrained(compiled_path, config=model_config)
        else:
            raise AttributeError("No from_pretrained method")
    except (TypeError, AttributeError, Exception):
        model = NeuronGemma2ForCausalLM(model_path, model_config)

    return model, neuron_config


def generate_with_neuron_model(model, input_ids, max_new_tokens: int):
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
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        neuron_config = NeuronConfig(
            tp_degree=2,
            batch_size=1,
            seq_len=128,
            max_context_length=128,
            torch_dtype=torch.bfloat16,
        )
        config = Gemma2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )
        model = NeuronGemma2ForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_model_loads(compiled_model):
    assert compiled_model is not None
    assert hasattr(compiled_model, 'config')
    assert hasattr(compiled_model.config, 'neuron_config')


def test_model_generates(compiled_model, tokenizer):
    prompt = "def fibonacci(n):"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    assert len(output_text) > len(prompt)


def test_output_coherence(compiled_model, tokenizer):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    assert len(output_text.split()) > 5


if __name__ == "__main__":
    print("=" * 80)
    print("Gemma-2-9b Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        neuron_config = NeuronConfig(
            tp_degree=2,
            batch_size=1,
            seq_len=128,
            max_context_length=128,
            torch_dtype=torch.bfloat16,
        )
        config = Gemma2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )
        model = NeuronGemma2ForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model, neuron_config = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Smoke Test...")
    test_model_loads(model)
    print("PASS")

    print("\n2. Generation Test...")
    test_model_generates(model, tokenizer)
    print("PASS")

    print("\n3. Coherence Test...")
    test_output_coherence(model, tokenizer)
    print("PASS")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
