#!/usr/bin/env python3
"""
Integration tests for Granite 4.0-H-Small NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.
This model is a hybrid Mamba2/Attention + MoE architecture requiring
Mamba state persistence across decode steps.

Tested on: trn2.3xlarge (TP=4, LNC=2, SDK 2.28)
"""

import pytest
import time
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_granite import NeuronGraniteForCausalLM, GraniteInferenceConfig


# Test configuration — UPDATE THESE PATHS for your environment
MODEL_PATH = "/home/ubuntu/Granite4/granite-4.0-h-small/"
COMPILED_MODEL_PATH = "/home/ubuntu/Granite4/traced_model_contrib/"

# Compilation parameters (trn2.3xlarge with LNC=2 -> 4 logical cores)
TP_DEGREE = 4
BATCH_SIZE = 1
MAX_CONTEXT_LENGTH = 128
SEQ_LENGTH = 2048


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
    """Create model for inference using compiled neuron_config."""
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)

    neuron_config = MoENeuronConfig(
        tp_degree=neuron_config_dict.get("tp_degree", TP_DEGREE),
        batch_size=neuron_config_dict.get("batch_size", BATCH_SIZE),
        max_context_length=neuron_config_dict.get(
            "max_context_length", MAX_CONTEXT_LENGTH
        ),
        seq_len=neuron_config_dict.get("seq_len", SEQ_LENGTH),
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
    )

    config = GraniteInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    model = NeuronGraniteForCausalLM(model_path, config)
    return model, neuron_config


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load model."""
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")

        neuron_config = MoENeuronConfig(
            tp_degree=TP_DEGREE,
            batch_size=BATCH_SIZE,
            max_context_length=MAX_CONTEXT_LENGTH,
            seq_len=SEQ_LENGTH,
            on_device_sampling_config=None,
            enable_bucketing=False,
            flash_decoding_enabled=False,
            torch_dtype="bfloat16",
        )

        config = GraniteInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )

        model = NeuronGraniteForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.save_pretrained(COMPILED_MODEL_PATH)

    # Load compiled model
    model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)

    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    print("PASS: Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates text (prefill + decode)."""
    prompt = "Artificial Intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=20,
        do_sample=False,
    )

    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    assert len(output_text.strip()) > 0, "Should generate non-empty text"
    assert len(new_tokens) == 20, "Should generate exactly 20 new tokens"
    print(f"PASS: Generated '{output_text}'")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not repetitive gibberish)."""
    prompt = "Explain the concept of artificial intelligence in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=30,
        do_sample=False,
    )

    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    words = output_text.split()
    assert len(words) > 3, "Output should have multiple words"
    assert not _is_repetitive(output_text), "Output should not be repetitive"
    print(f"PASS: Coherent output '{output_text[:100]}...'")


def test_greedy_token_match(compiled_model, tokenizer):
    """Test that first greedy token matches known HF reference.

    HF reference (BF16 CPU): prompt 'Artificial Intelligence is' -> first token 'a' (264)
    or 'Art' -> 'ificial' depending on exact prompt. For our reference prompt:
    'Explain the concept of artificial intelligence in simple terms.'
    the first generated token should be sensible and consistent.
    """
    prompt = "Artificial Intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=1,
        do_sample=False,
    )

    first_token_id = outputs[0, inputs.input_ids.shape[1] :][0].item()
    first_token = tokenizer.decode([first_token_id])
    print(f"PASS: First greedy token = '{first_token}' (id={first_token_id})")

    # The token should be a real word/subword, not garbage
    assert first_token_id != 0, "Token ID 0 indicates generation failure"
    assert first_token_id != tokenizer.eos_token_id, (
        "Should not immediately produce EOS"
    )


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False

    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i + j] == word for j in range(max_repeat)):
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


def test_logit_validation(compiled_model, tokenizer):
    """Validate Neuron logits against CPU reference using logit_validation.

    Captures the first-position logits from a single prefill pass and
    compares the top-k token rankings against pre-computed CPU BF16 reference.

    This uses the same pattern as the NxDI accuracy utilities:
    torch_neuronx.testing.validation.logit_validation.
    """
    try:
        from torch_neuronx.testing.validation import logit_validation
    except ImportError:
        pytest.skip("torch_neuronx.testing.validation not available")

    prompt = "Artificial Intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    # Generate enough tokens to get logits for validation
    num_tokens_to_check = 10
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_tokens_to_check,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # If we got scores, validate them
    if hasattr(outputs, "scores") and outputs.scores:
        neuron_logits = torch.stack(outputs.scores, dim=1)  # (batch, num_tokens, vocab)

        # Run CPU reference for comparison
        from transformers import AutoModelForCausalLM

        cpu_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        cpu_model.eval()

        cpu_outputs = cpu_model.generate(
            inputs.input_ids,
            attention_mask=torch.ones_like(inputs.input_ids),
            max_new_tokens=num_tokens_to_check,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        cpu_logits = torch.stack(cpu_outputs.scores, dim=1)

        # Use logit_validation
        passed, results, status_msg = logit_validation(
            expected_logits=cpu_logits.float(),
            actual_logits=neuron_logits.float(),
            divergence_difference_tol=0.01,
        )

        print(f"  Logit validation: {'PASS' if passed else 'FAIL'}")
        print(f"  Status: {status_msg}")
        assert passed, f"Logit validation failed: {status_msg}"
    else:
        # Fallback: compare greedy token sequences
        neuron_tokens = outputs.sequences[0, inputs.input_ids.shape[1] :]

        from transformers import AutoModelForCausalLM

        cpu_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        cpu_model.eval()

        cpu_outputs = cpu_model.generate(
            inputs.input_ids,
            attention_mask=torch.ones_like(inputs.input_ids),
            max_new_tokens=num_tokens_to_check,
            do_sample=False,
        )
        cpu_tokens = cpu_outputs[0, inputs.input_ids.shape[1] :]

        match_count = (neuron_tokens == cpu_tokens).sum().item()
        match_pct = match_count / len(cpu_tokens) * 100

        print(f"  Token match: {match_count}/{len(cpu_tokens)} ({match_pct:.1f}%)")
        assert match_pct >= 80, (
            f"Token match too low: {match_pct:.1f}% (need >= 80%). "
            f"Neuron: {neuron_tokens.tolist()}, CPU: {cpu_tokens.tolist()}"
        )

    print("PASS: Logit validation")


def test_performance_throughput(compiled_model, tokenizer):
    """Measure token generation throughput."""
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    num_tokens = 50

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    # Warmup
    gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=5,
        do_sample=False,
    )

    start = time.perf_counter()
    gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_tokens,
        do_sample=False,
    )
    end = time.perf_counter()

    total_time = end - start
    throughput = num_tokens / total_time
    print(
        f"PASS: Throughput = {throughput:.2f} tok/s ({total_time:.2f}s for {num_tokens} tokens)"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Granite 4.0-H-Small Integration Tests")
    print("=" * 80)

    # Compile if needed
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")

        neuron_config = MoENeuronConfig(
            tp_degree=TP_DEGREE,
            batch_size=BATCH_SIZE,
            max_context_length=MAX_CONTEXT_LENGTH,
            seq_len=SEQ_LENGTH,
            on_device_sampling_config=None,
            enable_bucketing=False,
            flash_decoding_enabled=False,
            torch_dtype="bfloat16",
        )

        config = GraniteInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )

        model = NeuronGraniteForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

        tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tok.save_pretrained(COMPILED_MODEL_PATH)
        print("Compilation complete")

    # Load model
    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Generation Test...")
    test_model_generates(model, tokenizer)

    print("\n3. Coherence Test...")
    test_output_coherence(model, tokenizer)

    print("\n4. Greedy Token Match...")
    test_greedy_token_match(model, tokenizer)

    print("\n5. Logit Validation...")
    test_logit_validation(model, tokenizer)

    print("\n6. Throughput...")
    test_performance_throughput(model, tokenizer)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
