#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for InternVL3-8B-Instruct NxDI contrib model.

Usage:
    pytest test_model.py -v --tb=short

Prerequisites:
    - Model downloaded to MODEL_PATH
    - Compiled model at COMPILED_MODEL_PATH (run compile step first)
    - Neuron runtime available (trn2.3xlarge)
"""

import json
import time

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_internvl3 import (
    NeuronInternVL3ForCausalLM,
    InternVL3InferenceConfig,
)


# Test configuration
MODEL_PATH = "/mnt/models/InternVL3-8B-Instruct/"
COMPILED_MODEL_PATH = "/mnt/models/neuron_models/InternVL3-8B-Instruct/"


def load_neuron_config_from_compiled(compiled_path: str):
    """Load neuron configuration from compiled model."""
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data.get("neuron_config", config_data)


def create_model_for_inference(compiled_path: str, model_path: str):
    """Create model for inference using compiled neuron_config."""
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)

    dtype_str = neuron_config_dict.get("torch_dtype", "torch.bfloat16")
    if isinstance(dtype_str, str):
        dtype = (
            getattr(torch, dtype_str.split(".")[-1])
            if "torch" in dtype_str
            else torch.bfloat16
        )
    else:
        dtype = dtype_str

    neuron_config = NeuronConfig(
        tp_degree=neuron_config_dict.get("tp_degree", 4),
        batch_size=neuron_config_dict.get("batch_size", 1),
        seq_len=neuron_config_dict.get("seq_len", 2048),
        torch_dtype=dtype,
    )

    config = InternVL3InferenceConfig.from_pretrained(
        model_path, neuron_config=neuron_config
    )
    model = NeuronInternVL3ForCausalLM(config)
    model.load(compiled_path)
    return model


def generate_with_neuron_model(model, input_ids, max_new_tokens=20):
    """Simple greedy generation loop."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        outputs = model(generated)
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
        # Check for EOS (151645)
        if next_token.item() == 151645:
            break
    return generated


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compiled_model():
    """Load compiled model for testing."""
    return create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)


@pytest.fixture
def tokenizer():
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInternVL3TextOnly:
    """Test suite for InternVL3 text-only backbone."""

    def test_config_loads(self):
        """Test that config loads correctly from model directory."""
        config = InternVL3InferenceConfig.from_pretrained(MODEL_PATH)
        assert config.hidden_size == 3584
        assert config.num_attention_heads == 28
        assert config.num_key_value_heads == 4
        assert config.num_hidden_layers == 28
        assert config.intermediate_size == 18944
        assert config.vocab_size == 151674
        assert config.rope_theta == 1000000.0

    def test_model_generates(self, compiled_model, tokenizer):
        """Test that model generates coherent text."""
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        generated = generate_with_neuron_model(
            compiled_model, inputs.input_ids, max_new_tokens=20
        )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        assert len(response) > len(prompt), "Model should generate additional tokens"

    def test_greedy_match(self, compiled_model, tokenizer):
        """Test greedy decoding matches CPU reference top-1 token."""
        prompt = "I believe the meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = compiled_model(inputs.input_ids)
            logits = outputs.logits
            top1_id = logits[0, -1, :].argmax().item()

        # CPU reference: top-1 is "to" (id=311)
        expected_id = 311
        top1_token = tokenizer.decode([top1_id])
        print(
            f"Top-1 predicted: {top1_token!r} (id={top1_id}), expected: 'to' (id={expected_id})"
        )
        assert top1_id == expected_id, (
            f"Greedy mismatch: got {top1_id}, expected {expected_id}"
        )

    def test_performance_ttft(self, compiled_model, tokenizer):
        """Measure time to first token (TTFT)."""
        prompt = "Explain quantum computing in simple terms:"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Warmup
        for _ in range(3):
            compiled_model(inputs.input_ids)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            compiled_model(inputs.input_ids)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        avg_ttft = sum(times) / len(times)
        print(f"TTFT: {avg_ttft:.2f} ms (avg of {len(times)} runs)")
        assert avg_ttft < 500, f"TTFT too high: {avg_ttft:.2f} ms"

    def test_performance_throughput(self, compiled_model, tokenizer):
        """Measure token generation throughput."""
        prompt = "Write a short story about a robot:"
        inputs = tokenizer(prompt, return_tensors="pt")
        max_tokens = 50

        # Warmup
        generate_with_neuron_model(compiled_model, inputs.input_ids, max_new_tokens=10)

        start = time.perf_counter()
        generated = generate_with_neuron_model(
            compiled_model, inputs.input_ids, max_new_tokens=max_tokens
        )
        elapsed = time.perf_counter() - start

        new_tokens = generated.shape[1] - inputs.input_ids.shape[1]
        tok_per_sec = new_tokens / elapsed
        print(
            f"Generated {new_tokens} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s"
        )
        assert tok_per_sec > 1, f"Throughput too low: {tok_per_sec:.1f} tok/s"
