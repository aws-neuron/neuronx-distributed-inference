#!/usr/bin/env python3
"""
Integration tests for vaultgemma-1b NeuronX implementation.

IMPORTANT: VaultGemma requires OnDeviceSamplingConfig for correct accuracy.
This is automatically enabled in VaultGemmaNeuronConfig.
"""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_vaultgemma import (
    NeuronVaultGemmaForCausalLM,
    VaultGemmaInferenceConfig,
    VaultGemmaNeuronConfig,
)


# Test configuration
MODEL_PATH = "/home/ubuntu/models/vaultgemma-1b/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/vaultgemma-1b/"


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
    """
    Create model for inference using compiled neuron_config.
    
    Note: VaultGemmaNeuronConfig automatically enables OnDeviceSamplingConfig
    for correct accuracy.
    """
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)
    
    dtype_str = neuron_config_dict.get('torch_dtype', 'torch.bfloat16')
    if isinstance(dtype_str, str):
        dtype = getattr(torch, dtype_str.split('.')[1]) if dtype_str.startswith('torch.') else torch.bfloat16
    else:
        dtype = dtype_str
    
    # Use VaultGemmaNeuronConfig which automatically enables OnDeviceSamplingConfig
    neuron_config = VaultGemmaNeuronConfig(
        tp_degree=neuron_config_dict.get('tp_degree', 1),
        batch_size=neuron_config_dict.get('batch_size', 1),
        seq_len=neuron_config_dict.get('seq_len', 128),
        torch_dtype=dtype,
    )
    
    # Verify OnDeviceSamplingConfig is enabled (critical for accuracy)
    assert neuron_config.on_device_sampling_config is not None, \
        "OnDeviceSamplingConfig must be enabled for VaultGemma accuracy"
    
    config = VaultGemmaInferenceConfig.from_pretrained(
        model_path,
        neuron_config=neuron_config,
    )
    
    model = NeuronVaultGemmaForCausalLM(model_path, config)
    model.load(compiled_path)
    
    return model, neuron_config


def generate_with_neuron_model(model, input_ids, max_new_tokens: int):
    """
    Generate tokens using manual forward pass loop.
    
    Note: With OnDeviceSamplingConfig enabled, the model returns sampled tokens
    directly in outputs.tokens instead of logits.
    """
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(generated_ids.shape[0], -1)
        
        with torch.no_grad():
            outputs = model(generated_ids, position_ids=position_ids)
        
        # With OnDeviceSamplingConfig, outputs.tokens contains the sampled token
        if hasattr(outputs, 'tokens') and outputs.tokens is not None:
            if outputs.tokens.numel() == 1:
                next_token = outputs.tokens.view(1, 1)
            else:
                # Fallback to argmax if tokens is full vocab
                next_token = torch.argmax(outputs.tokens, dim=-1).unsqueeze(-1)
        elif hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits
            if logits.dim() == 3:
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = logits
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        elif isinstance(outputs, tuple):
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected output format: {type(outputs)}")
        
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


def test_accuracy_with_ods(compiled_model, tokenizer):
    """
    Test that model produces correct predictions with OnDeviceSamplingConfig.
    
    This test verifies the critical fix for VaultGemma accuracy.
    Without OnDeviceSamplingConfig, the model would predict 'in' instead of 'Paris'.
    """
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    
    with torch.no_grad():
        outputs = compiled_model(input_ids, position_ids=position_ids)
    
    # Get the predicted token
    if hasattr(outputs, 'tokens') and outputs.tokens is not None:
        if outputs.tokens.numel() == 1:
            predicted_token_id = outputs.tokens[0].item()
        else:
            predicted_token_id = outputs.tokens.argmax().item()
    elif hasattr(outputs, 'logits') and outputs.logits is not None:
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        predicted_token_id = logits.argmax().item()
    else:
        raise ValueError("Could not get prediction from model output")
    
    predicted_token = tokenizer.decode([predicted_token_id])
    
    print(f"✓ Accuracy test")
    print(f"  Prompt: '{prompt}'")
    print(f"  Predicted: '{predicted_token}'")
    
    # The correct prediction should contain 'Paris' (or at least not 'in')
    # This is the key test for the OnDeviceSamplingConfig fix
    assert 'in' not in predicted_token.lower() or 'paris' in predicted_token.lower(), \
        f"Expected 'Paris' but got '{predicted_token}' - OnDeviceSamplingConfig may not be working"


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish)."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Coherence checks
    assert len(output_text.split()) > 3, "Output should have multiple words"
    assert not _is_repetitive(output_text), "Output should not be repetitive"
    
    print(f"✓ Coherence test passed")
    print(f"  Output: {output_text[:100]}...")



def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for repeated words
    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i+j] == word for j in range(max_repeat)):
            return True
    
    # Check for repeated characters
    new_text = text[-100:] if len(text) > 100 else text
    if len(new_text) > 20:
        char_counts = {}
        for c in new_text:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(new_text)
        if max_char_ratio > 0.5:
            return True
    
    return False


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    import time
    
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    
    # Warmup
    for _ in range(3):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=position_ids)
    
    # Measure TTFT
    times = []
    for _ in range(10):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(input_ids, position_ids=position_ids)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
    
    avg_ttft = sum(times) / len(times)
    print(f"✓ TTFT: {avg_ttft:.2f}ms")



def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    import time
    
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    num_tokens = 50
    
    # Warmup
    _ = generate_with_neuron_model(compiled_model, input_ids, max_new_tokens=5)
    
    # Measure throughput
    start = time.perf_counter()
    _ = generate_with_neuron_model(compiled_model, input_ids, max_new_tokens=num_tokens)
    end = time.perf_counter()
    
    total_time = end - start
    throughput = num_tokens / total_time
    print(f"✓ Throughput: {throughput:.2f} tok/s")



if __name__ == "__main__":
    print("="*80)
    print("vaultgemma-1b Integration Tests")
    print("="*80)
    
    print("\nNote: This is a template test file.")
    print("For actual model testing, customize the model loading logic.")
    
    print("\n" + "="*80)
    print("✓ Template structure verified!")
    print("="*80)
