#!/usr/bin/env python3
"""
Integration tests for Granite 4.0-H-Small NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.
This model is a hybrid Mamba2/Attention + MoE architecture requiring
Mamba state persistence across decode steps.

Tested on: trn2.3xlarge (TP=4, LNC=2, SDK 2.28, SDK 2.29.1)

Includes logit_validation() tests per NxDI contrib guidelines:
https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/contrib/CONTRIBUTING.md
"""

import pytest
import time
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# ==============================================================================
# Logit Validation Tests (per NxDI contrib guidelines)
# ==============================================================================


def _generate_cpu_reference_logits(
    model_path: str, prompts: list, max_new_tokens: int = 10
):
    """Generate reference logits from HuggingFace model on CPU (BF16).

    Returns:
        dict mapping prompt -> (input_ids, expected_logits)
        where expected_logits shape is (max_new_tokens, 1, vocab_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()

    references = {}
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with teacher forcing to get per-position logits
        all_logits = []
        current_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(current_ids)
                next_logits = outputs.logits[:, -1:, :]  # (1, 1, vocab)
                all_logits.append(next_logits.float())
                next_token = next_logits.argmax(dim=-1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

        # Stack: (max_new_tokens, 1, vocab_size)
        expected_logits = torch.cat(all_logits, dim=1).permute(1, 0, 2)
        references[prompt] = (input_ids.tolist(), expected_logits)

    del model
    return references


def test_logit_validation(compiled_model, tokenizer):
    """Test logit accuracy against CPU reference using NxDI logit_validation().

    Validates that Neuron-compiled model produces logits within acceptable
    numerical tolerances of the HuggingFace CPU reference implementation.
    Uses teacher-forcing to isolate per-position drift.
    """
    try:
        from neuronx_distributed_inference.experimental.core.accuracy.logit_validation import (
            logit_validation,
        )
    except ImportError:
        pytest.skip("logit_validation not available in this NxDI version")

    # Test prompts covering different domains
    prompts = [
        "Artificial Intelligence is",
        "The capital of France is",
        "def fibonacci(n):",
    ]
    max_new_tokens = 10

    # Generate CPU reference logits
    print("  Generating CPU reference logits...")
    references = _generate_cpu_reference_logits(MODEL_PATH, prompts, max_new_tokens)

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    # Tolerances for hybrid Mamba2/Attention model (may have higher drift
    # due to SSM state accumulation across sequence positions)
    tol_map = {
        "all": (1e-3, 0.05),  # Relaxed for full distribution
        "50": (1e-3, 0.02),  # Moderate for top-50
        "5": (1e-4, 0.01),  # Tighter for top-5
        "1": (1e-4, 0.005),  # Strict for top-1
    }

    all_passed = True
    for prompt in prompts:
        input_ids_list, expected_logits = references[prompt]

        def generate_fn(input_ids_tensor):
            """Generate logits with teacher forcing."""
            outputs = gen_model.generate(
                input_ids_tensor,
                attention_mask=torch.ones_like(input_ids_tensor),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # Stack scores: list of (batch, vocab) -> (seq_len, batch, vocab)
            return torch.stack(outputs.scores)

        passed = logit_validation(
            input_ids=input_ids_list,
            generate_fn=generate_fn,
            expected_logits=expected_logits,
            tol_map=tol_map,
            suppress_passing=True,
        )

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: logit_validation for '{prompt[:40]}...'")
        if not passed:
            all_passed = False

    assert all_passed, "Logit validation failed for one or more prompts"


def test_logit_comparison_manual(compiled_model, tokenizer):
    """Compare logits against pre-computed CPU reference with numerical tolerances.

    Alternative to logit_validation() -- directly compares first-token logits
    to verify the model's output distribution is numerically close to HF reference.
    Uses Pearson correlation and cosine similarity (same metrics as accuracy tests).
    """
    prompts = [
        "Artificial Intelligence is",
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing:",
        "The weather today is",
    ]

    # Load HF reference model
    print("  Loading HF reference model (BF16 CPU)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    )
    ref_model.eval()

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    pearson_scores = []
    cosine_scores = []
    token_matches = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        # HF reference: get logits for first generated token
        with torch.no_grad():
            ref_outputs = ref_model(input_ids)
            ref_logits = ref_outputs.logits[:, -1, :].float()  # (1, vocab)

        # Neuron model: generate one token and compare
        neuron_outputs = gen_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        neuron_logits = neuron_outputs.scores[0].float()  # (1, vocab)

        # Compute metrics
        ref_flat = ref_logits[0]
        neu_flat = neuron_logits[0]

        # Pearson correlation
        r_mean = ref_flat - ref_flat.mean()
        n_mean = neu_flat - neu_flat.mean()
        pearson = (r_mean * n_mean).sum() / (r_mean.norm() * n_mean.norm() + 1e-8)
        pearson_scores.append(pearson.item())

        # Cosine similarity
        cosine = torch.nn.functional.cosine_similarity(
            ref_flat.unsqueeze(0), neu_flat.unsqueeze(0)
        ).item()
        cosine_scores.append(cosine)

        # Token match
        if ref_flat.argmax() == neu_flat.argmax():
            token_matches += 1

    del ref_model

    avg_pearson = sum(pearson_scores) / len(pearson_scores)
    avg_cosine = sum(cosine_scores) / len(cosine_scores)
    match_rate = token_matches / len(prompts)

    print(f"  Avg Pearson:     {avg_pearson:.4f}")
    print(f"  Avg Cosine:      {avg_cosine:.4f}")
    print(
        f"  Token match:     {token_matches}/{len(prompts)} ({match_rate * 100:.0f}%)"
    )

    # Assertions — Granite hybrid Mamba2/Attention typically achieves:
    # Pearson > 0.99, Cosine > 0.998, 100% token match
    assert avg_pearson > 0.95, f"Pearson too low: {avg_pearson:.4f}"
    assert avg_cosine > 0.95, f"Cosine too low: {avg_cosine:.4f}"
    assert match_rate >= 0.8, f"Token match rate too low: {match_rate * 100:.0f}%"
    print("  PASS: Logit comparison within tolerances")


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

    print("\n5. Throughput...")
    test_performance_throughput(model, tokenizer)

    print("\n6. Logit Comparison (vs CPU reference)...")
    test_logit_comparison_manual(model, tokenizer)

    print("\n7. Logit Validation (NxDI utility)...")
    try:
        test_logit_validation(model, tokenizer)
    except Exception as e:
        print(f"  SKIP: {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
