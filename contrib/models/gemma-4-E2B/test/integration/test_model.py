#!/usr/bin/env python3
"""
Integration tests for Gemma4-E2B NeuronX Distributed Inference implementation.

Tests model compilation, loading, and inference accuracy/performance.
Text-only tests only -- VLM compilation is currently blocked by NCC_ITEN404.
"""

import pytest
import time
import torch
from pathlib import Path

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ndxi_patch import apply_patch

apply_patch()

from modeling_gemma4_e2b import NeuronGemma4E2BForCausalLM, Gemma4E2BInferenceConfig

# Test configuration -- update these paths for your environment
MODEL_PATH = "/mnt/models/gemma-4-E2B/"
COMPILED_MODEL_PATH = "/mnt/models/gemma4-e2b-compiled/"


def create_config():
    """Create inference config for testing."""
    neuron_config = NeuronConfig(
        tp_degree=1,
        batch_size=1,
        max_batch_size=1,
        max_length=1024,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        attn_kernel_enabled=False,
    )

    config = Gemma4E2BInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )
    return config


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load model."""
    config = create_config()
    model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def test_model_loads(compiled_model):
    """Test that model loads successfully (smoke test)."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    print("PASS: Smoke test - Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates coherent text."""
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Pad to bucket size
    bucket_size = compiled_model.config.neuron_config.seq_len
    max_length = compiled_model.config.neuron_config.max_length
    seq_len = input_ids.shape[1]

    if seq_len < bucket_size:
        input_ids_padded = torch.nn.functional.pad(
            input_ids, (0, bucket_size - seq_len), value=0
        )
        attention_mask = torch.zeros(1, bucket_size, dtype=torch.int32)
        attention_mask[0, :seq_len] = 1
    else:
        input_ids_padded = input_ids[:, :bucket_size]
        attention_mask = torch.ones(1, bucket_size, dtype=torch.int32)

    position_ids = torch.zeros(1, bucket_size, dtype=torch.int32)
    position_ids[0, : min(seq_len, bucket_size)] = torch.arange(
        min(seq_len, bucket_size), dtype=torch.int32
    )
    seq_ids = torch.tensor([0], dtype=torch.int32)

    compiled_model.reset_kv_cache()

    # Prefill
    with torch.inference_mode():
        outputs = compiled_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )

    # Extract first token
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_logits, dim=-1).item()
    elif hasattr(outputs, "tokens") and outputs.tokens is not None:
        next_token = (
            outputs.tokens[0, -1].item()
            if outputs.tokens.dim() > 1
            else outputs.tokens[0].item()
        )
    else:
        pytest.fail("No logits or tokens in output")

    # Generate a few more tokens
    generated = [next_token]
    cur_pos = min(seq_len, bucket_size)

    for _ in range(19):
        tok_input = torch.tensor([[next_token]], dtype=torch.int64)
        tok_pos = torch.tensor([[cur_pos]], dtype=torch.int32)
        tok_attn = torch.ones(1, max_length, dtype=torch.int32)
        tok_attn[0, cur_pos + 1 :] = 0

        with torch.inference_mode():
            outputs = compiled_model(
                input_ids=tok_input,
                attention_mask=tok_attn,
                position_ids=tok_pos,
                seq_ids=seq_ids,
            )
        cur_pos += 1

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(next_logits, dim=-1).item()
        elif hasattr(outputs, "tokens") and outputs.tokens is not None:
            next_token = (
                outputs.tokens[0, -1].item()
                if outputs.tokens.dim() > 1
                else outputs.tokens[0].item()
            )
        else:
            break

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    assert len(output_text.strip()) > 0, "Should generate non-empty text"
    print(f"PASS: Generation test")
    print(f"  Prompt: {prompt}")
    print(f"  Output: {output_text}")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent for a math question."""
    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    bucket_size = compiled_model.config.neuron_config.seq_len
    max_length = compiled_model.config.neuron_config.max_length
    seq_len = input_ids.shape[1]

    input_ids_padded = torch.nn.functional.pad(
        input_ids, (0, bucket_size - seq_len), value=0
    )
    attention_mask = torch.zeros(1, bucket_size, dtype=torch.int32)
    attention_mask[0, :seq_len] = 1
    position_ids = torch.zeros(1, bucket_size, dtype=torch.int32)
    position_ids[0, :seq_len] = torch.arange(seq_len, dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)

    compiled_model.reset_kv_cache()

    with torch.inference_mode():
        outputs = compiled_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )

    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_logits, dim=-1).item()
    elif hasattr(outputs, "tokens") and outputs.tokens is not None:
        next_token = (
            outputs.tokens[0, -1].item()
            if outputs.tokens.dim() > 1
            else outputs.tokens[0].item()
        )
    else:
        pytest.fail("No output")

    generated = [next_token]
    cur_pos = seq_len

    for _ in range(29):
        tok_input = torch.tensor([[next_token]], dtype=torch.int64)
        tok_pos = torch.tensor([[cur_pos]], dtype=torch.int32)
        tok_attn = torch.ones(1, max_length, dtype=torch.int32)
        tok_attn[0, cur_pos + 1 :] = 0

        with torch.inference_mode():
            outputs = compiled_model(
                input_ids=tok_input,
                attention_mask=tok_attn,
                position_ids=tok_pos,
                seq_ids=seq_ids,
            )
        cur_pos += 1

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(next_logits, dim=-1).item()
        elif hasattr(outputs, "tokens") and outputs.tokens is not None:
            next_token = (
                outputs.tokens[0, -1].item()
                if outputs.tokens.dim() > 1
                else outputs.tokens[0].item()
            )
        else:
            break

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    assert "4" in output_text, f"Should contain '4' in response, got: {output_text}"
    print(f"PASS: Coherence test")
    print(f"  Output: {output_text}")


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    bucket_size = compiled_model.config.neuron_config.seq_len
    seq_len = input_ids.shape[1]

    input_ids_padded = torch.nn.functional.pad(
        input_ids, (0, bucket_size - seq_len), value=0
    )
    attention_mask = torch.zeros(1, bucket_size, dtype=torch.int32)
    attention_mask[0, :seq_len] = 1
    position_ids = torch.zeros(1, bucket_size, dtype=torch.int32)
    position_ids[0, :seq_len] = torch.arange(seq_len, dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)

    # Warmup
    for _ in range(3):
        compiled_model.reset_kv_cache()
        with torch.inference_mode():
            _ = compiled_model(
                input_ids=input_ids_padded,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
            )

    # Measure
    times = []
    for _ in range(10):
        compiled_model.reset_kv_cache()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = compiled_model(
                input_ids=input_ids_padded,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
            )
        times.append((time.perf_counter() - start) * 1000)

    avg_ttft = sum(times) / len(times)
    assert avg_ttft < 100, f"TTFT {avg_ttft:.1f}ms exceeds 100ms threshold"
    print(f"PASS: TTFT test: {avg_ttft:.1f}ms (threshold: 100ms)")


def test_performance_tpot(compiled_model, tokenizer):
    """Test Time Per Output Token (TPOT) performance."""
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    bucket_size = compiled_model.config.neuron_config.seq_len
    max_length = compiled_model.config.neuron_config.max_length
    seq_len = input_ids.shape[1]

    input_ids_padded = torch.nn.functional.pad(
        input_ids, (0, bucket_size - seq_len), value=0
    )
    attention_mask = torch.zeros(1, bucket_size, dtype=torch.int32)
    attention_mask[0, :seq_len] = 1
    position_ids = torch.zeros(1, bucket_size, dtype=torch.int32)
    position_ids[0, :seq_len] = torch.arange(seq_len, dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)

    compiled_model.reset_kv_cache()

    # Prefill
    with torch.inference_mode():
        outputs = compiled_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )

    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_logits, dim=-1).item()
    else:
        next_token = outputs.tokens[0, -1].item()

    cur_pos = seq_len
    num_tokens = 20

    # Warmup decode
    for i in range(3):
        tok_input = torch.tensor([[next_token]], dtype=torch.int64)
        tok_pos = torch.tensor([[cur_pos + i]], dtype=torch.int32)
        tok_attn = torch.ones(1, max_length, dtype=torch.int32)
        tok_attn[0, cur_pos + i + 1 :] = 0
        with torch.inference_mode():
            _ = compiled_model(
                input_ids=tok_input,
                attention_mask=tok_attn,
                position_ids=tok_pos,
                seq_ids=seq_ids,
            )

    # Reset and re-prefill for clean measurement
    compiled_model.reset_kv_cache()
    with torch.inference_mode():
        outputs = compiled_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
        )
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token = torch.argmax(next_logits, dim=-1).item()
    else:
        next_token = outputs.tokens[0, -1].item()

    cur_pos = seq_len

    # Measure decode
    start = time.perf_counter()
    for i in range(num_tokens):
        tok_input = torch.tensor([[next_token]], dtype=torch.int64)
        tok_pos = torch.tensor([[cur_pos]], dtype=torch.int32)
        tok_attn = torch.ones(1, max_length, dtype=torch.int32)
        tok_attn[0, cur_pos + 1 :] = 0

        with torch.inference_mode():
            outputs = compiled_model(
                input_ids=tok_input,
                attention_mask=tok_attn,
                position_ids=tok_pos,
                seq_ids=seq_ids,
            )
        cur_pos += 1

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            next_token = outputs.tokens[0, -1].item()

    elapsed = time.perf_counter() - start
    tpot = elapsed / num_tokens * 1000
    throughput = num_tokens / elapsed

    assert tpot < 50, f"TPOT {tpot:.1f}ms exceeds 50ms threshold"
    assert throughput > 20, (
        f"Throughput {throughput:.1f} tok/s below 20 tok/s threshold"
    )
    print(f"PASS: TPOT test: {tpot:.1f}ms, {throughput:.1f} tok/s")


def test_kv_cache_sharing(compiled_model):
    """Test that KV cache sharing is configured correctly."""
    config = compiled_model.config
    assert hasattr(config, "num_kv_shared_layers"), (
        "Config should have num_kv_shared_layers"
    )
    assert config.num_kv_shared_layers == 20, (
        f"Expected 20 shared layers, got {config.num_kv_shared_layers}"
    )
    print(f"PASS: KV cache sharing configured (20 shared layers)")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("=" * 80)
    print("Gemma4 E2B Integration Tests (Text-Only)")
    print("=" * 80)

    # Compile and load
    config = create_config()
    model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete")

    print(f"\nLoading from {COMPILED_MODEL_PATH}...")
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Generation Test...")
    test_model_generates(model, tok)

    print("\n3. Coherence Test...")
    test_output_coherence(model, tok)

    print("\n4. TTFT Performance...")
    test_performance_ttft(model, tok)

    print("\n5. TPOT Performance...")
    test_performance_tpot(model, tok)

    print("\n6. KV Cache Sharing...")
    test_kv_cache_sharing(model)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
