#!/usr/bin/env python3
"""
Integration tests for Qwen3-Omni-30B-A3B-Instruct (thinker text model) on NeuronX.

Tests model compilation, loading, and inference on Neuron devices.
Requires: trn1.32xlarge or trn2.48xlarge with enough cores for tp_degree.
"""

import pytest
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_qwen3_omni_moe import (
    NeuronQwen3OmniMoeForCausalLM,
    Qwen3OmniMoeInferenceConfig,
    load_qwen3_omni_thinker_text_config,
)

MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct/"
COMPILED_MODEL_PATH = "/home/ubuntu/traced_model/Qwen3-Omni-30B-A3B-Instruct/"
TP_DEGREE = 32
BATCH_SIZE = 1
SEQ_LEN = 512
MAX_CONTEXT_LENGTH = 256


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load the Qwen3-Omni thinker text model."""
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_context_length=MAX_CONTEXT_LENGTH,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={"top_k": 1, "do_sample": False},
    )

    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(MODEL_PATH),
    )

    model = NeuronQwen3OmniMoeForCausalLM(MODEL_PATH, config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model can generate text."""
    prompt = "The capital of France is"

    _, output_tokens = get_generate_outputs(
        compiled_model,
        [prompt],
        tokenizer,
        is_hf=False,
        do_sample=False,
        max_length=compiled_model.neuron_config.max_length,
    )

    output_text = output_tokens[0]
    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    print(f"PASS: Generated: {output_text[:200]}")


def test_throughput(compiled_model, tokenizer):
    """Measure token generation throughput."""
    prompt = "Hello"

    # warmup
    get_generate_outputs(
        compiled_model, [prompt], tokenizer,
        is_hf=False, do_sample=False,
        max_length=compiled_model.neuron_config.max_length,
    )

    start = time.perf_counter()
    _, output_tokens = get_generate_outputs(
        compiled_model, [prompt], tokenizer,
        is_hf=False, do_sample=False,
        max_length=compiled_model.neuron_config.max_length,
    )
    elapsed = time.perf_counter() - start

    output_len = len(tokenizer.encode(output_tokens[0]))
    input_len = len(tokenizer.encode(prompt))
    num_new = output_len - input_len
    throughput = num_new / elapsed if elapsed > 0 else 0
    assert throughput > 1, f"Throughput {throughput:.2f} tok/s is too low"
    print(f"PASS: Throughput {throughput:.2f} tok/s ({num_new} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-Omni-30B-A3B-Instruct Integration Tests")
    print("=" * 60)

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_context_length=MAX_CONTEXT_LENGTH,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={"top_k": 1, "do_sample": False},
    )
    config = Qwen3OmniMoeInferenceConfig(
        neuron_config,
        load_config=load_qwen3_omni_thinker_text_config(MODEL_PATH),
    )
    model = NeuronQwen3OmniMoeForCausalLM(MODEL_PATH, config)
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists():
        print("Compiling...")
        model.compile(COMPILED_MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test_model_loads(model)
    test_model_generates(model, tok)
    test_throughput(model, tok)
    print("\nAll tests passed!")
