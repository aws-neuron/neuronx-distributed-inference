#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integration tests for Sarvam-30B (SarvamMoEForCausalLM) NeuronX implementation.

Tests accuracy via logit_validation (check_accuracy_logits_v2) and basic
generation coherence.

Usage:
    # Set required environment variables
    export SARVAM_MODEL_PATH="/path/to/sarvamai/sarvam-30b"
    export SARVAM_COMPILED_PATH="/path/to/compiled/artifacts"
    export SARVAM_GOLDEN_PATH="/path/to/golden_references.pt"

    # Run tests
    pytest test/integration/test_sarvam_moe.py -v --capture=tee-sys

Prerequisites:
    - Downloaded model weights at SARVAM_MODEL_PATH (requires trust_remote_code=True)
    - Pre-compiled model at SARVAM_COMPILED_PATH (compiled with on_device_sampling_config=None)
    - Golden CPU reference logits at SARVAM_GOLDEN_PATH (generated via generate_golden_logits.py)
    - trn2.3xlarge instance with LNC=2 (TP=4)

Generating golden references:
    # On CPU (any machine with enough RAM for BF16 weights, ~64 GB):
    python generate_golden_logits.py --model-path /path/to/model --output /path/to/golden.pt

Compiling for validation (on_device_sampling must be None):
    # On trn2.3xlarge:
    python -c "
    from modeling_sarvam_moe import *
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    nc = MoENeuronConfig(tp_degree=4, batch_size=1, max_context_length=256, seq_len=256,
        on_device_sampling_config=None, torch_dtype='bfloat16', fused_qkv=True, glu_mlp=True,
        blockwise_matmul_config={'use_shard_on_intermediate_dynamic_while': True, 'skip_dma_token': True})
    cfg = SarvamMoEInferenceConfig(neuron_config=nc, load_config=load_sarvam_moe_config('/path/to/model'))
    m = NeuronSarvamMoEForCausalLM('/path/to/model', cfg)
    m.compile(compiled_model_path='/path/to/compiled')
    "
"""

import logging
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits_v2
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_sarvam_moe import (
    NeuronSarvamMoEForCausalLM,
    SarvamMoEInferenceConfig,
    load_sarvam_moe_config,
)

logger = logging.getLogger(__name__)

# ── Configuration via environment variables ──────────────────────────────

MODEL_PATH = os.environ.get("SARVAM_MODEL_PATH")
COMPILED_MODEL_PATH = os.environ.get("SARVAM_COMPILED_PATH")
GOLDEN_PATH = os.environ.get("SARVAM_GOLDEN_PATH")

_MISSING_ENV = []
if not MODEL_PATH:
    _MISSING_ENV.append("SARVAM_MODEL_PATH")
if not COMPILED_MODEL_PATH:
    _MISSING_ENV.append("SARVAM_COMPILED_PATH")
if not GOLDEN_PATH:
    _MISSING_ENV.append("SARVAM_GOLDEN_PATH")

if _MISSING_ENV:
    pytestmark = pytest.mark.skip(
        reason=f"Required environment variables not set: {', '.join(_MISSING_ENV)}"
    )

# ── Accuracy tolerances ─────────────────────────────────────────────────
# BF16 sigmoid MoE with 128 experts + top-6 routing produces higher logit
# divergence than dense models due to BF16 accumulation across expert dispatch.
# These tolerances were empirically validated across 5 diverse prompts × 20 tokens
# with 5/5 PASS on trn2.3xlarge TP=4, SDK 2.29.

NUM_TOKENS_TO_CHECK = 20
TOLERANCE_MAP = {
    5: (1e-5, 1.2),
    50: (1e-5, 1.2),
    1000: (1e-5, 1.2),
    None: (1e-5, 1.2),
}
DIVERGENCE_DIFFERENCE_TOL = 0.30

# Performance thresholds (conservative upper bounds)
MAX_TTFT_MS = float(os.environ.get("SARVAM_MAX_TTFT_MS", "5000"))
MIN_THROUGHPUT_TOK_S = float(os.environ.get("SARVAM_MIN_THROUGHPUT_TOK_S", "10"))


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def compiled_model():
    """Load and compile the Neuron model (module-scoped for reuse)."""
    neuron_config = MoENeuronConfig(
        tp_degree=4,
        batch_size=1,
        max_context_length=256,
        seq_len=256,
        on_device_sampling_config=None,  # Required for logit access
        torch_dtype="bfloat16",
        fused_qkv=True,
        glu_mlp=True,
        blockwise_matmul_config={
            "use_shard_on_intermediate_dynamic_while": True,
            "skip_dma_token": True,
        },
    )

    config = SarvamMoEInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_sarvam_moe_config(MODEL_PATH),
    )

    model = NeuronSarvamMoEForCausalLM(MODEL_PATH, config)

    t0 = time.time()
    model.compile(compiled_model_path=COMPILED_MODEL_PATH)
    logger.info(f"Compilation: {time.time() - t0:.1f}s")

    t0 = time.time()
    model.load(COMPILED_MODEL_PATH)
    logger.info(f"Weight loading: {time.time() - t0:.1f}s")

    return model


@pytest.fixture(scope="module")
def golden_references():
    """Load golden CPU reference logits."""
    assert GOLDEN_PATH and os.path.exists(GOLDEN_PATH), (
        f"Golden references not found at {GOLDEN_PATH}. "
        "Generate them first with generate_golden_logits.py"
    )
    data = torch.load(GOLDEN_PATH, weights_only=False)
    logger.info(f"Golden refs: {len(data['references'])} prompts, {data['dtype']}")
    return data


# ── Tests ────────────────────────────────────────────────────────────────


def test_model_loads(compiled_model):
    """Verify the model compiles and loads onto Neuron successfully."""
    assert compiled_model is not None


def test_model_generates(compiled_model, tokenizer):
    """Verify the model can generate text."""
    generation_model = HuggingFaceGenerationAdapter(compiled_model)
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt")
    output = generation_model.generate(
        tokens.input_ids,
        generation_config=gen_config,
        attention_mask=tokens.attention_mask,
    )

    generated = output[0][tokens.input_ids.shape[1] :]
    assert len(generated) > 0, "Model generated zero tokens"
    text = tokenizer.decode(generated, skip_special_tokens=True)
    logger.info(f"Generated: {text}")
    assert len(text.strip()) > 0, "Generated text is empty"


def test_output_coherence(compiled_model, tokenizer):
    """Verify generated output is coherent (not garbage)."""
    generation_model = HuggingFaceGenerationAdapter(compiled_model)
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=32,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt = "Explain quantum computing in one sentence:"
    tokens = tokenizer(prompt, return_tensors="pt")
    output = generation_model.generate(
        tokens.input_ids,
        generation_config=gen_config,
        attention_mask=tokens.attention_mask,
    )

    generated = output[0][tokens.input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    logger.info(f"Coherence check output: {text}")

    # Basic sanity: output has real words, not just special tokens or garbage
    words = text.strip().split()
    assert len(words) >= 3, f"Output too short ({len(words)} words): {text}"


def test_logit_accuracy(compiled_model, tokenizer, golden_references):
    """Validate Neuron model accuracy using check_accuracy_logits_v2.

    Compares full logit distributions against pre-computed CPU reference logits
    across multiple diverse prompts with teacher forcing.
    """
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=NUM_TOKENS_TO_CHECK,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    references = golden_references["references"]
    prompts = golden_references["prompts"]

    passed_count = 0
    failed_prompts = []

    for prompt in prompts:
        ref = references[prompt]
        input_ids = ref["input_ids"]
        attention_mask = ref["attention_mask"]
        expected_logits = ref["expected_logits"]

        # Trim to NUM_TOKENS_TO_CHECK
        if expected_logits.shape[0] > NUM_TOKENS_TO_CHECK:
            expected_logits = expected_logits[:NUM_TOKENS_TO_CHECK]

        try:
            check_accuracy_logits_v2(
                neuron_model=compiled_model,
                expected_logits=expected_logits,
                inputs_input_ids=input_ids,
                inputs_attention_mask=attention_mask,
                generation_config=gen_config,
                num_tokens_to_check=NUM_TOKENS_TO_CHECK,
                divergence_difference_tol=DIVERGENCE_DIFFERENCE_TOL,
                tol_map=TOLERANCE_MAP,
                tokenizer=tokenizer,
            )
            passed_count += 1
            logger.info(f"PASS: {prompt[:50]}...")
        except Exception as e:
            failed_prompts.append((prompt[:50], str(e)[:200]))
            logger.warning(f"FAIL: {prompt[:50]}... -- {str(e)[:100]}")

    logger.info(f"Logit validation: {passed_count}/{len(prompts)} passed")

    assert passed_count == len(prompts), (
        f"Logit validation failed for {len(prompts) - passed_count}/{len(prompts)} prompts: "
        f"{failed_prompts}"
    )


def test_performance_ttft(compiled_model, tokenizer):
    """Verify TTFT is within acceptable bounds."""
    generation_model = HuggingFaceGenerationAdapter(compiled_model)
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=1,
        min_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt = "Hello, how are you?"
    tokens = tokenizer(prompt, return_tensors="pt")

    # Warmup
    _ = generation_model.generate(
        tokens.input_ids,
        generation_config=gen_config,
        attention_mask=tokens.attention_mask,
    )

    # Measure
    ttft_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = generation_model.generate(
            tokens.input_ids,
            generation_config=gen_config,
            attention_mask=tokens.attention_mask,
        )
        ttft_times.append((time.perf_counter() - t0) * 1000)

    avg_ttft = sum(ttft_times) / len(ttft_times)
    logger.info(f"TTFT: {avg_ttft:.1f}ms (threshold: {MAX_TTFT_MS}ms)")
    assert avg_ttft < MAX_TTFT_MS, (
        f"TTFT {avg_ttft:.1f}ms exceeds threshold {MAX_TTFT_MS}ms"
    )


def test_performance_throughput(compiled_model, tokenizer):
    """Verify generation throughput is within acceptable bounds."""
    generation_model = HuggingFaceGenerationAdapter(compiled_model)
    gen_tokens = 64
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=gen_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt = "Write a short essay about the future of artificial intelligence."
    tokens = tokenizer(prompt, return_tensors="pt")

    # Warmup
    _ = generation_model.generate(
        tokens.input_ids,
        generation_config=gen_config,
        attention_mask=tokens.attention_mask,
    )

    # Measure
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        output = generation_model.generate(
            tokens.input_ids,
            generation_config=gen_config,
            attention_mask=tokens.attention_mask,
        )
        times.append(time.perf_counter() - t0)

    actual_gen = len(output[0]) - tokens.input_ids.shape[1]
    avg_time = sum(times) / len(times)
    tok_s = actual_gen / avg_time

    logger.info(
        f"Throughput: {tok_s:.1f} tok/s ({actual_gen} tokens in {avg_time:.2f}s, "
        f"threshold: {MIN_THROUGHPUT_TOK_S} tok/s)"
    )
    assert tok_s > MIN_THROUGHPUT_TOK_S, (
        f"Throughput {tok_s:.1f} tok/s below threshold {MIN_THROUGHPUT_TOK_S} tok/s"
    )
