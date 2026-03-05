#!/usr/bin/env python3
"""
Integration tests for Baichuan2-7B-Base NeuronX implementation.

This model uses the Llama-2 architecture with:
- Fused QKV projection (W_pack) split during weight conversion
- NormHead lm_head (pre-normalized weights)
- RoPE position encoding
- Multi-head attention (32 heads, head_dim=128)
- SiLU activation
- 125696 token vocabulary
- Direct config/weight loading (bypasses trust_remote_code)
"""

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import NeuronConfig

# Import from src directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_baichuan2 import NeuronBaichuan2ForCausalLM, Baichuan2InferenceConfig


# Test configuration - UPDATE THESE PATHS for your environment
MODEL_PATH = "/shared/dhwanw2/models/Baichuan2-7B-Base"
COMPILED_MODEL_PATH = "/tmp/neuron_models/baichuan2-7b-base/"


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
    neuron_config = NeuronConfig(
        tp_degree=2,
        batch_size=1,
        seq_len=128,
        max_context_length=128,
        torch_dtype=torch.bfloat16,
    )

    config = Baichuan2InferenceConfig.from_pretrained(
        MODEL_PATH, neuron_config=neuron_config,
    )

    model = NeuronBaichuan2ForCausalLM(MODEL_PATH, config)

    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

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
    print("Smoke test passed - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=20)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    print(f"Generation test passed")
    print(f"  Output: {output_text}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish)."""
    prompt = "The theory of general relativity explains"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text.split()) > 3, "Output should have multiple words"
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

    new_text = text[-100:] if len(text) > 100 else text
    if len(new_text) > 20:
        char_counts = {}
        for c in new_text:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(new_text)
        if max_char_ratio > 0.5:
            return True

    return False


if __name__ == "__main__":
    print("=" * 80)
    print("Baichuan2-7B-Base Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    neuron_config = NeuronConfig(
        tp_degree=2, batch_size=1, seq_len=128,
        max_context_length=128, torch_dtype=torch.bfloat16,
    )
    config = Baichuan2InferenceConfig.from_pretrained(
        MODEL_PATH, neuron_config=neuron_config,
    )
    model = NeuronBaichuan2ForCausalLM(MODEL_PATH, config)

    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
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
