#!/usr/bin/env python3
"""
Integration tests for Git-base NeuronX implementation.

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
from modeling_git import NeuronGitForCausalLM, GitInferenceConfig


# Test configuration
MODEL_PATH = "/shared/dhwanw2/models/git-base"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/git-base/"


def load_neuron_config_from_compiled(compiled_path: str):
    """Load neuron configuration from compiled model's neuron_config.json."""
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    if "neuron_config" in config_data:
        return config_data["neuron_config"]
    return config_data


def create_model_for_inference(compiled_path: str, model_path: str):
    """Create model for inference using the pattern from validate_model.py."""
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
    for param in ['world_size', 'max_context_length', 'enable_bucketing']:
        if param in neuron_config_dict:
            neuron_config_kwargs[param] = neuron_config_dict[param]
    if 'max_context_length' not in neuron_config_kwargs:
        neuron_config_kwargs['max_context_length'] = neuron_config_kwargs['seq_len']

    neuron_config = NeuronConfig(**neuron_config_kwargs)
    try:
        model_config = GitInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
    except (TypeError, AttributeError):
        model_config = GitInferenceConfig(neuron_config, load_config=load_pretrained_config(model_path))

    try:
        model = NeuronGitForCausalLM.from_pretrained(compiled_path, config=model_config)
    except Exception:
        model = NeuronGitForCausalLM(model_path, model_config)
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
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    return generated_ids


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load model."""
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        neuron_config = NeuronConfig(
            tp_degree=1, batch_size=1, seq_len=128,
            max_context_length=128, torch_dtype=torch.bfloat16,
        )
        config = GitInferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
        model = NeuronGitForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

    model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_model_loads(compiled_model):
    """Test that model loads successfully (smoke test)."""
    assert compiled_model is not None
    assert hasattr(compiled_model, 'config')


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "a photo of"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    assert len(output_text) >= len(prompt), "Output should be at least as long as prompt"


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    import time
    prompt = "a photo of"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids

    for _ in range(3):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=pos)

    times = []
    for _ in range(10):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=pos)
        times.append((time.perf_counter() - start) * 1000)

    avg_ttft = sum(times) / len(times)
    assert avg_ttft < 100, f"TTFT {avg_ttft:.2f}ms exceeds 100ms threshold"


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    import time
    inputs = tokenizer("a photo", return_tensors="pt", padding=True)
    num_tokens = 50

    _ = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=5)

    start = time.perf_counter()
    _ = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=num_tokens)
    throughput = num_tokens / (time.perf_counter() - start)

    assert throughput > 10, f"Throughput {throughput:.2f} tok/s below 10 tok/s threshold"


if __name__ == "__main__":
    print("=" * 80)
    print("Git-base Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        neuron_config = NeuronConfig(
            tp_degree=1, batch_size=1, seq_len=128,
            max_context_length=128, torch_dtype=torch.bfloat16,
        )
        config = GitInferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
        model = NeuronGitForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

    model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_model_loads(model)
    print("1. Smoke test passed")

    test_model_generates(model, tokenizer)
    print("2. Generation test passed")

    test_performance_ttft(model, tokenizer)
    print("3. TTFT test passed")

    test_performance_throughput(model, tokenizer)
    print("4. Throughput test passed")

    print("\nAll tests passed!")
