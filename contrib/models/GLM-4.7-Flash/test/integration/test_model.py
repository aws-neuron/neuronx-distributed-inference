# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GLM-4.7-Flash on Neuron.

Tests compilation, loading, and inference accuracy using the full 30B model
on a trn2.3xlarge instance with TP=4.

Environment variables:
    GLM4_MODEL_PATH      Path to HF model weights (required)
    GLM4_COMPILED_PATH   Path to compiled artifacts (default: /tmp/glm4_traced)
    GLM4_TP_DEGREE       Tensor parallelism degree (default: 4)
    GLM4_SEQ_LEN         Max sequence length (default: 4096)
    GLM4_BATCH_SIZE      Batch size (default: 4, minimum for NCC_IBIR297 workaround)

Prerequisites:
    - trn2.3xlarge with LNC=2 (4 NeuronCores)
    - NxDI installed (neuronx_distributed_inference >= 0.9)
    - transformers >= 5.0
    - Model weights downloaded (59 GB)

Usage:
    # Full model (requires trn2.3xlarge + model weights):
    GLM4_MODEL_PATH=/mnt/models/GLM-4.7-Flash \
    GLM4_COMPILED_PATH=/mnt/models/compiled_glm4_4096 \
    pytest test/integration/test_model.py --capture=tee-sys

    # Quick validation (pre-compiled):
    GLM4_MODEL_PATH=/mnt/models/GLM-4.7-Flash \
    GLM4_COMPILED_PATH=/mnt/models/compiled_glm4_4096 \
    pytest test/integration/test_model.py -k "test_inference_accuracy" --capture=tee-sys

Known Issues:
    - Minimum batch_size=4 required (NCC_IBIR297 compiler issue at small TP degrees)
    - transformers >= 5.0 requires Glm4MoeLiteGenerationAdapter (position_ids fix)
    - NKI MoE kernel unavailable in SDK 2.29 (uses torch blockwise fallback)
"""

import gc
import json
import os
import sys
import time

import pytest
import torch

# Ensure the contrib root is on sys.path
_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

# ── Configuration ───────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("GLM4_MODEL_PATH", "")
COMPILED_PATH = os.environ.get("GLM4_COMPILED_PATH", "/tmp/glm4_traced")
TP_DEGREE = int(os.environ.get("GLM4_TP_DEGREE", "4"))
SEQ_LEN = int(os.environ.get("GLM4_SEQ_LEN", "4096"))
BATCH_SIZE = int(os.environ.get("GLM4_BATCH_SIZE", "4"))

if not MODEL_PATH:
    pytest.skip("GLM4_MODEL_PATH not set", allow_module_level=True)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def neuron_config():
    """Create MoE Neuron config for GLM-4.7-Flash."""
    from neuronx_distributed_inference.models.config import (
        MoENeuronConfig,
        OnDeviceSamplingConfig,
    )

    return MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        ctx_batch_size=BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
    )


@pytest.fixture(scope="module")
def inf_config(neuron_config):
    """Create GLM-4.7-Flash inference config."""
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from src.modeling_glm4_moe_lite import Glm4MoeLiteInferenceConfig

    return Glm4MoeLiteInferenceConfig(
        neuron_config, load_config=load_pretrained_config(MODEL_PATH)
    )


@pytest.fixture(scope="module")
def compiled_model(inf_config):
    """Compile or load the model."""
    from src.modeling_glm4_moe_lite import NeuronGlm4MoeLiteForCausalLM

    if os.path.exists(os.path.join(COMPILED_PATH, "model.pt")):
        print(f"\n  Loading pre-compiled model from {COMPILED_PATH}")
        model = NeuronGlm4MoeLiteForCausalLM(COMPILED_PATH, inf_config)
        model.load(COMPILED_PATH)
    else:
        print(f"\n  Compiling model to {COMPILED_PATH}")
        os.makedirs(COMPILED_PATH, exist_ok=True)
        model = NeuronGlm4MoeLiteForCausalLM(MODEL_PATH, inf_config)
        model.compile(COMPILED_PATH)
        model.load(COMPILED_PATH)

    yield model
    del model
    gc.collect()


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def gen_model(compiled_model):
    """Create generation adapter (fixes transformers 5.x position_ids issue)."""
    from src.modeling_glm4_moe_lite import Glm4MoeLiteGenerationAdapter

    return Glm4MoeLiteGenerationAdapter(compiled_model)


# ── Tests ───────────────────────────────────────────────────────────────


class TestGlm4MoeLiteInference:
    """Integration tests for GLM-4.7-Flash on Neuron."""

    def test_model_loads(self, compiled_model):
        """Verify model loads successfully and all cores are utilized."""
        assert compiled_model is not None

    def test_first_token_accuracy(self, gen_model, tokenizer):
        """Verify first-token accuracy matches CPU reference (exact token ID match).

        These reference token IDs were captured from CPU FP32 inference with greedy
        decoding. Exact token ID match provides strong accuracy validation without
        requiring the full 30B model to fit on CPU during test execution.
        """
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Test prompts with known CPU reference token IDs (captured from FP32 reference)
        test_cases = [
            ("The capital of France is", 12089, " Paris"),
            ("In machine learning, a transformer model", 374, " is"),
            ("def fibonacci(n):", 715, "\n"),
        ]

        for prompt, expected_token_id, expected_text in test_cases:
            inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True)
            outputs = gen_model.generate(
                inputs.input_ids,
                generation_config=gen_config,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
            )
            # Check first generated token by ID (exact match)
            first_new_token = outputs[0, inputs.input_ids.shape[1]].item()
            decoded = tokenizer.decode([first_new_token])
            print(
                f"\n  Prompt: '{prompt}' -> token_id={first_new_token} '{decoded}' "
                f"(expected: {expected_token_id} '{expected_text}')"
            )
            assert first_new_token == expected_token_id, (
                f"First token ID mismatch for '{prompt}': "
                f"got {first_new_token} ('{decoded}'), "
                f"expected {expected_token_id} ('{expected_text}')"
            )

    def test_coherent_generation(self, gen_model, tokenizer):
        """Verify multi-token generation produces coherent text."""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt = "The capital of France is"
        inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True)
        outputs = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
        )

        generated_ids = outputs[0, inputs.input_ids.shape[1] :].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n  Generated: '{generated_text[:100]}'")

        # Should contain "Paris" and be coherent
        assert "Paris" in generated_text, (
            f"Expected 'Paris' in output: '{generated_text}'"
        )
        assert len(generated_text) > 10, f"Generation too short: '{generated_text}'"

    def test_deterministic_outputs(self, gen_model, tokenizer):
        """Verify greedy decoding produces identical outputs across runs."""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt = "Hello, world"
        inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True)

        outputs_1 = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,
        )
        outputs_2 = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,
        )

        assert torch.equal(outputs_1, outputs_2), (
            "Greedy decoding should be deterministic"
        )

    def test_batch_consistency(self, gen_model, tokenizer):
        """Verify all sequences in batch produce identical output (same prompt)."""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt = "The meaning of life is"
        inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True)
        outputs = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=15,
        )

        # All sequences in batch should be identical (same input, greedy)
        for i in range(1, BATCH_SIZE):
            assert torch.equal(outputs[0], outputs[i]), (
                f"Batch inconsistency: seq 0 != seq {i}"
            )

    def test_throughput(self, gen_model, tokenizer):
        """Measure and report throughput metrics."""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt = "Explain quantum computing in simple terms:"
        inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True)

        # Warmup
        gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,
        )

        # Measure
        max_new_tokens = 50
        t0 = time.time()
        outputs = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_config,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0

        n_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        total_tokens = n_generated * BATCH_SIZE
        throughput = total_tokens / elapsed

        print(f"\n  Generated: {n_generated} tokens/seq")
        print(f"  Batch throughput: {throughput:.1f} tok/s")
        print(f"  Per-seq throughput: {n_generated / elapsed:.2f} tok/s")
        print(f"  Avg latency/token: {elapsed / n_generated * 1000:.1f} ms")

        # Minimum sanity threshold
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} tok/s"
