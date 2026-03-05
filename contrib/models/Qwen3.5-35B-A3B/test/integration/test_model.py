#!/usr/bin/env python3
"""
Integration tests for Qwen3.5-35B-A3B NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.

Environment:
  - trn2.3xlarge with Neuron SDK 2.28
  - source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  - export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
"""

import json
import os
import sys
import time

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig


# Test configuration -- update these paths for your environment
MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "/home/ubuntu/models/Qwen3.5-35B-A3B")
COMPILED_MODEL_PATH = os.environ.get(
    "QWEN35_COMPILED_PATH", "/home/ubuntu/compiled_qwen35/"
)


def create_config(model_path: str):
    """Create inference config from HF model config."""
    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    # IMPORTANT: block_size=2048 works around a blockwise MoE bug in SDK 2.28.
    neuron_config = MoENeuronConfig(
        tp_degree=4,
        max_batch_size=1,
        max_context_length=128,
        max_new_tokens=32,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        moe_tp_degree=4,
        moe_ep_degree=1,
        blockwise_matmul_config={"block_size": 2048},
    )

    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    if config_dict.get("tie_word_embeddings") is None:
        config_dict["tie_word_embeddings"] = False

    return Qwen35MoeInferenceConfig(neuron_config=neuron_config, **config_dict)


def generate_with_neuron_model(model, tokenizer, input_ids, max_new_tokens: int):
    """Generate tokens using CTE + TKG loop."""
    generated_ids = input_ids.clone()

    # Context encoding (prefill)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
    logits = output[0] if isinstance(output, tuple) else output.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

    # Token generation
    for _ in range(max_new_tokens - 1):
        pos_ids = torch.tensor([[generated_ids.shape[1] - 1]])
        last_token = generated_ids[:, -1:]
        with torch.no_grad():
            output = model(
                input_ids=last_token,
                position_ids=pos_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
        logits = output[0] if isinstance(output, tuple) else output.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

    return generated_ids


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load model."""
    compiled_path = Path(COMPILED_MODEL_PATH)

    config = create_config(MODEL_PATH)
    model = NeuronQwen35MoeForCausalLM(model_path=MODEL_PATH, config=config)

    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        os.environ.pop("XLA_DISABLE_FUNCTIONALIZATION", None)
        os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_model_loads(compiled_model):
    """Test that model loads successfully (smoke test)."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: Smoke test - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates 'Paris' for capital of France prompt."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    generated_ids = generate_with_neuron_model(
        compiled_model, tokenizer, inputs.input_ids, max_new_tokens=10
    )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    assert "Paris" in output_text, f"Should mention Paris, got: {output_text}"
    print(f"PASS: Generation test")
    print(f"  Output: {output_text}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not gibberish)."""
    prompts = [
        "1 + 1 =",
        "The color of the sky is",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        generated_ids = generate_with_neuron_model(
            compiled_model, tokenizer, inputs.input_ids, max_new_tokens=15
        )
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Basic coherence checks
        assert len(output_text.split()) > 3, f"Output too short: {output_text}"
        assert not _is_repetitive(output_text), f"Output is repetitive: {output_text}"

        print(f"PASS: Coherence test for '{prompt}'")
        print(f"  Output: {output_text[:120]}...")


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = compiled_model(
                input_ids=input_ids,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )

    # Measure TTFT
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(
                input_ids=input_ids,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ttft = sum(times) / len(times)

    # Threshold is generous for this complex model
    assert avg_ttft < 500, f"TTFT {avg_ttft:.2f}ms exceeds 500ms threshold"
    print(f"PASS: TTFT test: {avg_ttft:.2f}ms (threshold: 500ms)")


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    num_tokens = 20

    # Warmup
    _ = generate_with_neuron_model(
        compiled_model, tokenizer, input_ids, max_new_tokens=3
    )

    # Measure throughput
    start = time.perf_counter()
    _ = generate_with_neuron_model(
        compiled_model, tokenizer, input_ids, max_new_tokens=num_tokens
    )
    end = time.perf_counter()

    total_time = end - start
    throughput = num_tokens / total_time

    # Conservative threshold for hybrid architecture
    assert throughput > 5, f"Throughput {throughput:.2f} tok/s below 5 tok/s threshold"
    print(f"PASS: Throughput test: {throughput:.2f} tok/s (threshold: 5 tok/s)")


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i + j] == word for j in range(max_repeat)):
            return True
    return False


if __name__ == "__main__":
    print("=" * 80)
    print("Qwen3.5-35B-A3B Integration Tests")
    print("=" * 80)

    # Setup
    config = create_config(MODEL_PATH)
    model = NeuronQwen35MoeForCausalLM(model_path=MODEL_PATH, config=config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        os.environ.pop("XLA_DISABLE_FUNCTIONALIZATION", None)
        os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Run tests
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test (Model Loading)...")
    test_model_loads(model)

    print("\n2. Generation Test...")
    test_model_generates(model, tok)

    print("\n3. Coherence Test...")
    test_output_coherence(model, tok)

    print("\n4. TTFT Performance Test...")
    test_performance_ttft(model, tok)

    print("\n5. Throughput Performance Test...")
    test_performance_throughput(model, tok)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
