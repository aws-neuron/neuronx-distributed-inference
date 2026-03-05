#!/usr/bin/env python3
"""
Integration tests for CTRL NeuronX implementation.

This model uses the CTRL architecture with:
- Sequential residual connections (LN -> Attn -> Add -> LN -> MLP -> Add)
- Sinusoidal position embeddings (fixed, concatenated [sines|cosines] layout)
- Multi-head attention (16 heads, head_dim=80)
- Token embeddings scaled by sqrt(d_model)
- ReLU activation (not GELU)
- Large vocab (246534 tokens with control codes)
- LM head with bias
"""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Monkey-patch hf_adapter to handle read-only PretrainedConfig attrs
import neuronx_distributed_inference.utils.hf_adapter as hf_adapter_module
from transformers import PretrainedConfig

_original_to_pretrained_config = hf_adapter_module.to_pretrained_config

def _patched_to_pretrained_config(config):
    config_dict = hf_adapter_module.to_dict(config)
    readonly_attrs = {
        'use_return_dict', 'output_hidden_states', 'output_attentions',
        'torchscript', 'use_bfloat16', 'pruned_heads', 'tie_word_embeddings',
        'is_encoder_decoder', 'is_decoder', 'cross_attention_hidden_size',
        'add_cross_attention', 'tie_encoder_decoder', 'use_cache',
        'pad_token_id', 'bos_token_id', 'eos_token_id',
    }
    filtered_dict = {k: v for k, v in config_dict.items() if k not in readonly_attrs}
    return PretrainedConfig(**filtered_dict)

hf_adapter_module.to_pretrained_config = _patched_to_pretrained_config

# Import from src directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_ctrl import NeuronCTRLForCausalLM, CTRLInferenceConfig


# Test configuration - UPDATE THESE PATHS for your environment
MODEL_PATH = "/shared/dhwanw2/models/ctrl/"
COMPILED_MODEL_PATH = "/tmp/neuron_models/ctrl/"


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

        neuron_config = CTRLInferenceConfig.get_neuron_config_cls()(
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
        print("Compilation complete.")
    else:
        neuron_config = CTRLInferenceConfig.get_neuron_config_cls()(
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
    prompt = "Wikipedia The theory of general relativity"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=30)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text.split()) > 3, "Output should have multiple words"
    assert not _is_repetitive(output_text), "Output should not be repetitive"

    print(f"Coherence test passed")
    print(f"  Output: {output_text[:100]}...")




def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    import time
    prompt = "Wikipedia The theory of general relativity"
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
    assert avg_ttft < 100, f"TTFT {avg_ttft:.2f}ms exceeds 100ms threshold"
    print(f"TTFT test passed: {avg_ttft:.2f}ms (threshold: 100ms)")


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    import time
    prompt = "Wikipedia The theory of general relativity"
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

    assert throughput > 10, f"Throughput {throughput:.2f} tok/s below 10 tok/s threshold"
    print(f"Throughput test passed: {throughput:.2f} tok/s (threshold: 10 tok/s)")


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
    print("CTRL Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        neuron_config = CTRLInferenceConfig.get_neuron_config_cls()(
            tp_degree=1, batch_size=1, seq_len=128,
            max_context_length=128, torch_dtype=torch.bfloat16,
        )
        config = CTRLInferenceConfig.from_pretrained(
            MODEL_PATH, neuron_config=neuron_config,
        )
        model = NeuronCTRLForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")
    else:
        neuron_config = CTRLInferenceConfig.get_neuron_config_cls()(
            tp_degree=1, batch_size=1, seq_len=128,
            max_context_length=128, torch_dtype=torch.bfloat16,
        )
        config = CTRLInferenceConfig.from_pretrained(
            MODEL_PATH, neuron_config=neuron_config,
        )
        model = NeuronCTRLForCausalLM(MODEL_PATH, config)

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
